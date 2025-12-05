#!/usr/bin/env python3
"""
Script to extract specific prompts from input JSON files based on missing problem IDs.
Usage: python extract_missing_prompts.py <input_json> <output_json> <problem_ids>
"""
import json
import sys

def extract_prompts(input_file, output_file, problem_ids):
    """
    Extract prompts for specific problem IDs from input file.

    Args:
        input_file: Path to input JSON file with all prompts
        output_file: Path to output JSON file for filtered prompts
        problem_ids: List of problem IDs to extract
    """
    # Read input file
    with open(input_file, 'r') as f:
        all_prompts = json.load(f)

    # Extract only the missing problem IDs
    filtered_prompts = {pid: all_prompts[pid] for pid in problem_ids if pid in all_prompts}

    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(filtered_prompts, f, indent=4)

    print(f"Extracted {len(filtered_prompts)} prompts from {input_file}")
    print(f"Saved to {output_file}")
    print(f"Problem IDs: {list(filtered_prompts.keys())}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_missing_prompts.py <input_json> <output_json> <comma_separated_problem_ids>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    problem_ids = sys.argv[3].split(',')

    extract_prompts(input_file, output_file, problem_ids)
