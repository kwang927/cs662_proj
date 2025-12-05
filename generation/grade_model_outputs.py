#!/usr/bin/env python3
"""
Script to grade model outputs using GPT-4.
Reads results from generate_model_outputs.py and grades how well responses answer prompts.
"""

import json
import argparse
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import time
import threading

# Import generation functions
from models.openai_related import gpt_generation
from models.togetherai_related import togetherai_generation

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


GRADING_PROMPT_TEMPLATE = """Please judge if the following response answers the prompt. Use a scale of 3 rating, where: 1 means that the response does not answer the prompt at all, and is completely wrong; 2 means that the response gets the general idea of the prompt and answers it to some extent; and 3 means that the response faithfully answers the prompt.

Prompt: {prompt}

Response: {response}

Please provide your rating (1, 2, or 3) and a brief explanation. Format your response as:
Rating: [1/2/3]
Explanation: [Your explanation here]"""


def call_until_timeout(func, timeout_seconds=30, delay=5, **kwargs):
    """
    Calls the specified function until it succeeds or the timeout is reached.
    """
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            print("Timeout reached, stopping attempts.")
            return None, None
        try:
            result, tok_related = func(**kwargs)

            # Track tokens
            if tok_related:
                input_tokens = tok_related[0] if len(tok_related) > 0 else 0
                output_tokens = tok_related[1] if len(tok_related) > 1 else 0
                update_global_counter(input_tokens, output_tokens)

            return result, tok_related
        except Exception as e:
            print(f"Function call failed with error: {e}")
            remaining_time = timeout_seconds - elapsed_time
            if remaining_time <= 0:
                print(f"Timeout reached, stopping attempts.")
                return None, None
            print(f"Retrying in {min(delay, remaining_time)} seconds...")
            time.sleep(min(delay, remaining_time))


def parse_rating(response_text):
    """Extract rating from GPT-4 response."""
    lines = response_text.split('\n')
    rating = None
    explanation = ""

    for line in lines:
        if line.startswith('Rating:'):
            # Extract number from rating line
            rating_str = line.replace('Rating:', '').strip()
            # Try to extract just the number
            for char in rating_str:
                if char.isdigit():
                    rating = int(char)
                    break
        elif line.startswith('Explanation:'):
            explanation = line.replace('Explanation:', '').strip()

    return rating, explanation


def grade_response(prompt, response, grader_model="gpt-4o", temperature=0, max_tokens=500, timeout_seconds=60):
    """Grade a single response using specified grader model (OpenAI or Together AI)."""

    grading_prompt = GRADING_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

    # Determine which API to use based on model name
    # Together AI models we support
    together_models = [
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "meta-llama/Llama-2-13b-chat-hf"
    ]

    if grader_model in together_models:
        # Use Together AI
        generation_func = togetherai_generation
    else:
        # Use OpenAI by default
        generation_func = gpt_generation

    result, tokens = call_until_timeout(
        func=generation_func,
        timeout_seconds=timeout_seconds,
        prompt=grading_prompt,
        model=grader_model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0
    )

    if result is None:
        return None, "Failed to get grading"

    rating, explanation = parse_rating(result)

    return rating, explanation


def check_and_create_directory(dir_path):
    """Check if a directory exists, and if not, create it."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Grade model outputs using GPT-4. Only grades (original_prompt, optimized_output) pairs."
    )
    parser.add_argument(
        '--results_path',
        type=str,
        required=True,
        help='Path to results.json or results.pickle from generate_model_outputs.py'
    )
    parser.add_argument(
        '--grader_model',
        type=str,
        default='gpt-4o',
        help='Model to use for grading. Supported: OpenAI (gpt-4o, gpt-4, etc.) or Together AI (mistralai/Mistral-Small-24B-Instruct-2501, meta-llama/Llama-2-13b-chat-hf). Default: gpt-4o'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for grading results (optional, default: creates grading_results/ subdirectory next to input)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for grading model (default: 0.0)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=200,
        help='Max tokens for grading response (default: 200)'
    )
    parser.add_argument(
        '--timeout_seconds',
        type=int,
        default=60,
        help='Timeout for each grading call (default: 60)'
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

    args = parser.parse_args()

    # Load model pricing
    with open("model_price.json", 'r') as f:
        model_price = json.load(f)

    # Load results
    print(f"Loading results from: {args.results_path}")
    if args.results_path.endswith('.json'):
        with open(args.results_path, 'r') as f:
            data = json.load(f)
    elif args.results_path.endswith('.pickle'):
        with open(args.results_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError("Results file must be .json or .pickle")

    print(f"Loaded {len(data)} entries")

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create subdirectory in same location as input
        input_dir = Path(args.results_path).parent
        output_dir = str(input_dir / "grading_results")

    check_and_create_directory(output_dir)

    print(f"\nGrading results will be saved to: {output_dir}")
    print(f"Grader model: {args.grader_model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print("-" * 80)

    # Prepare results storage
    grading_results = {}

    # Determine range
    keys = sorted(data.keys(), key=lambda x: int(x))
    if args.start_idx is not None:
        keys = [k for k in keys if int(k) >= args.start_idx]
    if args.end_idx is not None:
        keys = [k for k in keys if int(k) < args.end_idx]

    print(f"\nGrading {len(keys)} entries...")
    print()

    # Process each entry with progress bar
    pbar = tqdm(keys, desc="Grading outputs", unit="entry")

    for key in pbar:
        entry = data[key]
        pbar.set_description(f"Grading entry {key}")

        # Extract data
        original_prompt = entry.get('original_prompt', '')
        optimized_prompt = entry.get('optimized_prompt', '')
        optimized_output = entry.get('optimized_output', '')

        if not original_prompt or not optimized_output:
            pbar.write(f"WARNING: Missing data for entry {key}, skipping...")
            continue

        # Grade optimized output with original prompt
        pbar.set_postfix_str("grading optimized output")
        optimized_rating, optimized_explanation = grade_response(
            original_prompt,  # Use original_prompt for grading
            optimized_output,
            args.grader_model,
            args.temperature,
            args.max_tokens,
            args.timeout_seconds
        )

        # Store grading results
        grading_results[key] = {
            'original_prompt': original_prompt,
            'optimized_prompt': optimized_prompt,
            'optimized_output': optimized_output,
            'optimized_rating': optimized_rating,
            'optimized_explanation': optimized_explanation,
            'starting_prompt': entry.get('starting_prompt', ''),
            'final_metrics': entry.get('final_metrics', {}),
            'log_file': entry.get('log_file', '')
        }

        # Save intermediate results after each entry
        with open(f"{output_dir}/grading_checkpoint.json", 'w') as f:
            json.dump(grading_results, f, indent=2)

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    pbar.close()
    print()

    # Save final results
    output_json_path = f"{output_dir}/grading_results.json"
    output_pickle_path = f"{output_dir}/grading_results.pickle"

    with open(output_json_path, 'w') as f:
        json.dump(grading_results, f, indent=2)

    with open(output_pickle_path, 'wb') as f:
        pickle.dump(grading_results, f)

    # Calculate statistics
    optimized_ratings = [r['optimized_rating'] for r in grading_results.values() if r.get('optimized_rating') is not None]

    print("=" * 80)
    print(f"Grading complete!")
    print(f"  Processed: {len(grading_results)} entries")
    print(f"\nStatistics:")
    if optimized_ratings:
        print(f"  Optimized outputs (graded with original prompts):")
        print(f"    - Average rating: {sum(optimized_ratings) / len(optimized_ratings):.2f}")
        print(f"    - Rating distribution: 1={optimized_ratings.count(1)}, 2={optimized_ratings.count(2)}, 3={optimized_ratings.count(3)}")
    else:
        print(f"  No ratings were successfully generated.")
    print()

    # Token usage and cost calculation
    if cumulative_count > 0:
        print(f"Token Usage:")
        print(f"  Generated {cumulative_count} gradings")
        print(f"  Input tokens: {cumulative_input_tokens:,}")
        print(f"  Output tokens: {cumulative_output_tokens:,}")
        print()

        # Calculate cost if model is in pricing
        if args.grader_model in model_price:
            input_price = model_price[args.grader_model]["input"]
            output_price = model_price[args.grader_model]["output"]

            input_cost = cumulative_input_tokens / 1000000 * input_price
            output_cost = cumulative_output_tokens / 1000000 * output_price
            total_cost = input_cost + output_cost

            print(f"Cost Estimation:")
            print(f"  Input cost: ${input_cost:.4f}")
            print(f"  Output cost: ${output_cost:.4f}")
            print(f"  Total cost: ${total_cost:.4f}")
            print()
        else:
            print(f"Note: Model '{args.grader_model}' not found in model_price.json")
            print(f"      Cannot calculate cost estimation.")
            print()

    print(f"Results saved to:")
    print(f"  - JSON: {output_json_path}")
    print(f"  - Pickle: {output_pickle_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
