#!/usr/bin/env python3
"""
Script to generate model outputs from optimization results.
Takes a JSON file from exp_results and generates outputs for both original and optimized prompts.
"""

import json
import argparse
import os
import pickle
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import threading

# Global variables for token tracking
cumulative_input_tokens = 0
cumulative_output_tokens = 0
cumulative_count = 0
lock = threading.Lock()


def update_global_counter(input_tokens, output_tokens):
    """Update global token counters in thread-safe manner."""
    global cumulative_input_tokens, cumulative_output_tokens, cumulative_count

    with lock:
        cumulative_input_tokens += input_tokens
        cumulative_output_tokens += output_tokens
        cumulative_count += 1


def check_and_create_directory(dir_path):
    """Check if a directory exists, and if not, create it."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")


def call_until_timeout(func, timeout_seconds=30, delay=5, model_name="", **kwargs):
    """
    Calls the specified function until it succeeds or the timeout is reached.

    Args:
    - func: The function to call.
    - timeout_seconds: Total time allowed for retries in seconds.
    - delay: Delay between retries in seconds.
    - model_name: Name of the model (for token tracking)
    """
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            print("Timeout reached, stopping attempts.")
            return "**********NO OUTPUT**********"
        try:
            result, tok_related = func(**kwargs)

            # Track tokens (skip for Gemini as they return (0,0))
            if tok_related and 'gemini' not in model_name.lower():
                input_tokens = tok_related[0] if len(tok_related) > 0 else 0
                output_tokens = tok_related[1] if len(tok_related) > 1 else 0
                update_global_counter(input_tokens, output_tokens)

            return result
        except Exception as e:
            print(f"Function call failed with error: {e}")
            remaining_time = timeout_seconds - elapsed_time
            if remaining_time <= 0:
                print(f"Timeout reached, stopping attempts.")
                return "**********NO OUTPUT**********"
            print(f"Retrying in {min(delay, remaining_time)} seconds...")
            time.sleep(min(delay, remaining_time))


def select_generation_function(model_name):
    """Select the appropriate generation function based on model name (lazy import)."""
    if 'gpt' in model_name.lower() or 'o1' in model_name.lower() or 'o3' in model_name.lower():
        from models.openai_related import gpt_generation
        return gpt_generation
    elif 'claude' in model_name.lower():
        from models.anthropic_related import generate_claude_response
        return generate_claude_response
    elif 'gemini' in model_name.lower():
        from models.google_related import generate_gemini_response
        return generate_gemini_response
    elif 'deepseek' in model_name.lower():
        from models.deepseek_related import deepseek_generation
        return deepseek_generation
    elif 'grok' in model_name.lower():
        from models.xai_related import generate_grok_response_multi_turn
        return generate_grok_response_multi_turn
    else:
        # Default to TogetherAI for other models
        from models.togetherai_related import togetherai_generation
        return togetherai_generation


def generate_for_prompt(prompt_text, model_name, temperature=0, max_tokens=1024, top_p=1.0, timeout_seconds=60, verbose=False):
    """Generate output for a single prompt."""
    generation_func = select_generation_function(model_name)

    if verbose:
        print(f"  Generating with model: {model_name}")
        print(f"  Prompt: {prompt_text[:100]}..." if len(prompt_text) > 100 else f"  Prompt: {prompt_text}")

    # Claude and Gemini don't accept top_p parameter
    if 'claude' in model_name.lower() or 'gemini' in model_name.lower():
        result = call_until_timeout(
            func=generation_func,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            prompt=prompt_text,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        result = call_until_timeout(
            func=generation_func,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
            prompt=prompt_text,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate model outputs from optimization results (exp_results JSON files)"
    )
    parser.add_argument(
        '--exp_results_path',
        type=str,
        required=True,
        help='Path to the JSON file in exp_results (e.g., exp_results/warm_start__pythia-1.4b.json)'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model to use for generation (e.g., gpt-4o-mini, claude-3-haiku-20240307, meta-llama/Meta-Llama-3-8B-Instruct-Turbo)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for generation (default: 0.0)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=1024,
        help='Maximum tokens to generate (default: 1024)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help='Top-p sampling parameter (default: 1.0)'
    )
    parser.add_argument(
        '--timeout_seconds',
        type=int,
        default=60,
        help='Timeout in seconds for each generation (default: 60)'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=None,
        help='Start index (optional, for processing subset)'
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        default=None,
        help='End index (optional, for processing subset)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Custom output directory (optional, default: auto-generated in generation_output/)'
    )

    args = parser.parse_args()

    # Load model pricing
    with open("model_price.json", 'r') as f:
        model_price = json.load(f)

    # Load the JSON file
    print(f"Loading data from: {args.exp_results_path}")
    with open(args.exp_results_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries")

    # Create output directory with reasonable filename
    if args.output_dir:
        output_dir = args.output_dir
    else:
        input_filename = Path(args.exp_results_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = args.model.replace('/', '_').replace('-', '_')
        output_dir = f"generation_output/{input_filename}__{model_safe_name}__{timestamp}"
    check_and_create_directory(output_dir)

    print(f"\nOutput will be saved to: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Top-p: {args.top_p}")
    print("-" * 80)

    # Prepare results storage
    results = {}

    # Determine range
    keys = sorted(data.keys(), key=lambda x: int(x))
    if args.start_idx is not None:
        keys = [k for k in keys if int(k) >= args.start_idx]
    if args.end_idx is not None:
        keys = [k for k in keys if int(k) < args.end_idx]

    print(f"\nProcessing {len(keys)} entries...")
    print()

    # Process each entry with progress bar
    pbar = tqdm(keys, desc="Generating outputs", unit="entry")
    for key in pbar:
        entry = data[key]
        pbar.set_description(f"Processing entry {key}")

        # Extract prompts
        original_prompt = entry.get('original_prompt', '')
        optimized_prompt = entry.get('optimized_prompt', '')

        if not original_prompt or not optimized_prompt:
            pbar.write(f"WARNING: Missing prompts for entry {key}, skipping...")
            continue

        # Generate for original prompt
        pbar.set_postfix_str("original prompt")
        original_output = generate_for_prompt(
            original_prompt,
            args.model,
            args.temperature,
            args.max_tokens,
            args.top_p,
            args.timeout_seconds,
            verbose=False
        )

        # Generate for optimized prompt
        pbar.set_postfix_str("optimized prompt")
        optimized_output = generate_for_prompt(
            optimized_prompt,
            args.model,
            args.temperature,
            args.max_tokens,
            args.top_p,
            args.timeout_seconds,
            verbose=False
        )

        # Store results
        results[key] = {
            'original_prompt': original_prompt,
            'optimized_prompt': optimized_prompt,
            'original_output': original_output,
            'optimized_output': optimized_output,
            'starting_prompt': entry.get('starting_prompt', ''),
            'final_metrics': entry.get('final_metrics', {}),
            'log_file': entry.get('log_file', '')
        }

        # Save intermediate results after each entry
        with open(f"{output_dir}/results_checkpoint.json", 'w') as f:
            json.dump(results, f, indent=2)


        # Small delay to avoid rate limiting
        time.sleep(1)

    pbar.close()
    print()

    # Save final results
    output_json_path = f"{output_dir}/results.json"
    output_pickle_path = f"{output_dir}/results.pickle"

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_pickle_path, 'wb') as f:
        pickle.dump(results, f)

        print()
    print("=" * 80)
    print(f"Generation complete!")
    print(f"  Processed: {len(results)} entries")
    print()

    # Token usage and cost calculation
    if cumulative_count > 0:
        print(f"Token Usage:")
        print(f"  Generated {cumulative_count} outputs")
        print(f"  Input tokens: {cumulative_input_tokens:,}")
        print(f"  Output tokens: {cumulative_output_tokens:,}")
        print()

        # Calculate cost if model is in pricing
        if args.model in model_price:
            input_price = model_price[args.model]["input"]
            output_price = model_price[args.model]["output"]

            input_cost = cumulative_input_tokens / 1000000 * input_price
            output_cost = cumulative_output_tokens / 1000000 * output_price
            total_cost = input_cost + output_cost

            print(f"Cost Estimation:")
            print(f"  Input cost: ${input_cost:.4f}")
            print(f"  Output cost: ${output_cost:.4f}")
            print(f"  Total cost: ${total_cost:.4f}")
            print()
        else:
            print(f"Note: Model '{args.model}' not found in model_price.json")
            print(f"      Cannot calculate cost estimation.")
            print()

    print(f"Results saved to:")
    print(f"  - JSON: {output_json_path}")
    print(f"  - Pickle: {output_pickle_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
