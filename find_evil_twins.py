#!/usr/bin/env python3
"""
Batch prompt optimization script using optim_gcg.

This script takes a JSON file of original prompts and optimizes them using GCG
(Greedy Coordinate Gradient) to find prompts with similar output distributions.

Usage:
    python optimize_prompts.py \\
        --original_prompts original_prompts.json \\
        --output_path results.json \\
        --model_name teknium/OpenHermes-2.5-Mistral-7B \\
        [--starting_prompts starting_prompts.json] \\
        [--n_docs 10] \\
        [--doc_len 50] \\
        [--n_epochs 500] \\
        [--kl_every 10]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from datetime import datetime, timezone, timedelta

from func_from_evil_twins import load_model_tokenizer, DocDataset, optim_gcg, optim_gcg_pruned

import pdb


def load_prompts_json(filepath: str) -> dict[str, str]:
    """Load prompts from JSON file with integer string keys and string values."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Ensure keys are strings and values are strings
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Batch optimize prompts using GCG algorithm"
    )

    # Required arguments
    parser.add_argument(
        '--original_prompts',
        type=str,
        required=True,
        help='Path to JSON file with original prompts (keys: str of integers, values: strings)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to store optimization results (JSON format)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='HuggingFace model name or path (e.g., teknium/OpenHermes-2.5-Mistral-7B)'
    )

    # Optional arguments
    parser.add_argument(
        '--starting_prompts',
        type=str,
        default=None,
        help='Path to JSON file with starting prompts (same format as original_prompts). '
             'If not provided, defaults to "!" * 15 for all prompts.'
    )
    parser.add_argument(
        '--n_docs',
        type=int,
        default=100,
        help='Number of document continuations to generate (default: 100)'
    )
    parser.add_argument(
        '--doc_len',
        type=int,
        default=32,
        help='Length of each document continuation in tokens (default: 32)'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=50,
        help='Number of optimization epochs (token flips) (default: 50)'
    )
    parser.add_argument(
        '--kl_every',
        type=int,
        default=1,
        help='Compute KL divergence every N epochs (default: 1)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size for document forward passes (default: 50)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=256,
        help='Top-k tokens to keep from gradients (default: 256)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.0,
        help='Fluency penalty coefficient (default: 0.0)'
    )
    parser.add_argument(
        '--early_stop_kl',
        type=float,
        default=-10000000000000,
        help='KL divergence threshold for early stopping (default: 0.0, disabled)'
    )
    parser.add_argument(
        '--use_flash_attn_2',
        action='store_true',
        help='Use Flash Attention 2 for faster inference (requires flash-attn package)'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./optimization_logs',
        help='Directory to store individual optimization logs (default: ./optimization_logs)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information during optimization'
    )
    parser.add_argument(
        '--pruned_vocab',
        type=str,
        default=None,
        help='Path to JSON file containing pruned vocabulary token IDs (list or set of integers). '
             'If provided, uses optim_gcg_pruned instead of optim_gcg.'
    )

    args = parser.parse_args()

    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load original prompts
    print(f"Loading original prompts from: {args.original_prompts}")
    original_prompts = load_prompts_json(args.original_prompts)
    print(f"Loaded {len(original_prompts)} prompts")

    # Load or create starting prompts
    if args.starting_prompts:
        print(f"Loading starting prompts from: {args.starting_prompts}")
        starting_prompts = load_prompts_json(args.starting_prompts)
        print(f"Loaded {len(starting_prompts)} starting prompts")

        # Verify keys match
        if set(original_prompts.keys()) != set(starting_prompts.keys()):
            print("WARNING: Keys in original_prompts and starting_prompts do not match!")
            print(f"Original keys: {sorted(original_prompts.keys())}")
            print(f"Starting keys: {sorted(starting_prompts.keys())}")
    else:
        print("No starting prompts provided, using default: '!' * 15")
        starting_prompts = {k: "!" * 15 for k in original_prompts.keys()}

    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    print(f"Using Flash Attention 2: {args.use_flash_attn_2}")
    model, tokenizer = load_model_tokenizer(
        args.model_name,
        use_flash_attn_2=args.use_flash_attn_2
    )
    print(f"Model loaded successfully on device: {model.device}")

    # Load pruned vocabulary if provided
    pruned_vocab = None
    if args.pruned_vocab:
        print(f"\nLoading pruned vocabulary from: {args.pruned_vocab}")
        with open(args.pruned_vocab, 'r') as f:
            pruned_vocab = json.load(f)
        # Convert to set for efficient lookup
        if isinstance(pruned_vocab, list):
            pruned_vocab = set(pruned_vocab)
        print(f"Loaded pruned vocabulary with {len(pruned_vocab)} tokens")
        print("Will use optim_gcg_pruned instead of optim_gcg")

    # Results storage
    results = {}

    # Iterate through all prompts
    print(f"\nStarting optimization for {len(original_prompts)} prompts...")
    print(f"Configuration:")
    print(f"  - n_docs: {args.n_docs}")
    print(f"  - doc_len: {args.doc_len}")
    print(f"  - n_epochs: {args.n_epochs}")
    print(f"  - kl_every: {args.kl_every}")
    print(f"  - batch_size: {args.batch_size}")
    print(f"  - top_k: {args.top_k}")
    print(f"  - gamma: {args.gamma}")
    print(f"  - early_stop_kl: {args.early_stop_kl}")
    print()

    # Create progress bar for outer loop
    pbar = tqdm(sorted(original_prompts.keys(), key=int),
                desc="Optimizing prompts",
                position=0,
                leave=True,
                dynamic_ncols=True)

    for key in pbar:
        orig_prompt = original_prompts[key]
        optim_prompt = starting_prompts[key]

        # Update progress bar description with current prompt
        pbar.set_description(f"Optimizing prompt {key}")

        if args.verbose:
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"Processing prompt {key}")
            tqdm.write(f"Original prompt: {orig_prompt[:100]}...")
            tqdm.write(f"Starting prompt: {optim_prompt[:100]}...")
            tqdm.write(f"{'='*80}")

        # Get current west coast time (PST/PDT: UTC-8/UTC-7) in MMDD_HHMM format
        # Using UTC-8 for Pacific Standard Time
        west_coast_tz = timezone(timedelta(hours=-8))
        now = datetime.now(west_coast_tz)
        timestamp = now.strftime("%m%d_%H%M")
        # print(timestamp)

        # Create log file path for this prompt
        log_fpath = str(log_dir / f"prompt_{key}_{timestamp}_log.json")

        try:
            # Create DocDataset with nested progress bars
            if args.verbose:
                tqdm.write("Creating DocDataset...")
            dataset = DocDataset(
                model=model,
                tokenizer=tokenizer,
                orig_prompt=orig_prompt,
                optim_prompt=optim_prompt,
                n_docs=args.n_docs,
                doc_len=args.doc_len,
                gen_batch_size=args.batch_size,
                validate_prompt=True,
                gen_train_docs=True,
                gen_dev_docs=True,
                verbose=args.verbose,
                # position=1 if args.verbose else None
            )
            if args.verbose:
                tqdm.write("DocDataset created successfully\n")

            # Run optimization with nested progress bar
            if args.verbose:
                if pruned_vocab:
                    tqdm.write(f"Running optim_gcg_pruned for {args.n_epochs} epochs...")
                else:
                    tqdm.write(f"Running optim_gcg for {args.n_epochs} epochs...")

            if pruned_vocab:
                # Use pruned vocabulary version
                epoch_results, best_ids = optim_gcg_pruned(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    allowed_token_ids=pruned_vocab,
                    n_epochs=args.n_epochs,
                    kl_every=args.kl_every,
                    log_fpath=log_fpath,
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                    gamma=args.gamma,
                    early_stop_kl=args.early_stop_kl,
                    suffix_mode=False,
                    verbose=args.verbose,  # Controls print messages only
                    position=1,  # Nested progress bar below the outer one (always shown)
                )
            else:
                # Use standard version
                epoch_results, best_ids = optim_gcg(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    n_epochs=args.n_epochs,
                    kl_every=args.kl_every,
                    log_fpath=log_fpath,
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                    gamma=args.gamma,
                    early_stop_kl=args.early_stop_kl,
                    suffix_mode=False,
                    verbose=args.verbose,  # Controls print messages only
                    position=1,  # Nested progress bar below the outer one (always shown)
                )

            # Decode best prompt
            best_prompt_text = tokenizer.decode(best_ids[0], skip_special_tokens=False)

            # Store results
            final_epoch = epoch_results[-1]
            results[key] = {
                'original_prompt': orig_prompt,
                'starting_prompt': optim_prompt,
                'optimized_prompt': best_prompt_text,
                'optimized_token_ids': best_ids.tolist(),
                'final_metrics': {
                    'best_kl': final_epoch['best_kl'],
                    'best_std': final_epoch['best_std'],
                    'best_loss': final_epoch['best_loss'],
                    'final_epoch': final_epoch['epoch'],
                },
                'log_file': log_fpath,
            }

            if args.verbose:
                tqdm.write(f"\nOptimization complete for prompt {key}")
                tqdm.write(f"Best KL: {final_epoch['best_kl']:.6f}")
                tqdm.write(f"Best prompt: {best_prompt_text[:100]}...")

            # Save intermediate results after each prompt
            if args.verbose:
                tqdm.write(f"Saving intermediate results to: {args.output_path}")
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2)

            # Clean up GPU memory after each optimization
            del dataset
            del epoch_results
            del best_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.verbose:
                if torch.cuda.is_available():
                    tqdm.write(f"GPU memory cleaned. Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        except Exception as e:
            tqdm.write(f"\nERROR processing prompt {key}: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()

            # Store error in results
            results[key] = {
                'original_prompt': orig_prompt,
                'starting_prompt': optim_prompt,
                'error': str(e),
            }

            # Save results even on error
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2)

            # Clean up GPU memory even on error
            try:
                del dataset
            except (NameError, UnboundLocalError):
                pass
            try:
                del epoch_results
            except (NameError, UnboundLocalError):
                pass
            try:
                del best_ids
            except (NameError, UnboundLocalError):
                pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.verbose:
                if torch.cuda.is_available():
                    tqdm.write(f"GPU memory cleaned after error. Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Save final results
    print(f"\n{'='*80}")
    print(f"All optimizations complete!")
    print(f"Final results saved to: {args.output_path}")
    print(f"Individual logs saved to: {log_dir}")
    print(f"Successfully optimized: {sum(1 for r in results.values() if 'error' not in r)}/{len(results)}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
