#!/bin/bash

# Script to automatically generate and grade model outputs
# Usage: ./run_generation_and_grading.sh <exp_results_path> <generating_model> [start_idx] [end_idx]
# Note: Always uses GPT-4 for grading

set -e  # Exit on error

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <exp_results_path> <generating_model> [start_idx] [end_idx]"
    echo ""
    echo "Arguments:"
    echo "  exp_results_path    Path to JSON file in exp_results (e.g., ../exp_results/warm_start__vicuna-7b.json)"
    echo "  generating_model    Model to use for generation (e.g., claude-3-haiku-20240307, gpt-4o-mini)"
    echo "  start_idx           Start index for subset (optional)"
    echo "  end_idx             End index for subset (optional)"
    echo ""
    echo "Note: Grading is always done with GPT-4"
    echo ""
    echo "Example:"
    echo "  $0 ../exp_results/warm_start__vicuna-7b.json claude-3-haiku-20240307"
    echo "  $0 ../exp_results/warm_start__pythia-1.4b.json gpt-4o-mini"
    echo "  $0 ../exp_results/warm_start__vicuna-7b.json claude-3-haiku-20240307 1 10"
    exit 1
fi

EXP_RESULTS_PATH="$1"
GENERATING_MODEL="$2"
GRADING_MODEL="gpt-4"  # Always use GPT-4 for grading
START_IDX="${3:-}"
END_IDX="${4:-}"

# Validate that exp_results file exists
if [ ! -f "$EXP_RESULTS_PATH" ]; then
    echo "Error: File not found: $EXP_RESULTS_PATH"
    exit 1
fi

# Extract experiment name (filename without .json extension)
EXP_FILENAME=$(basename "$EXP_RESULTS_PATH")
EXP_NAME="${EXP_FILENAME%.json}"

# Create safe model names for directory (replace / and - with _)
GENERATING_MODEL_SAFE=$(echo "$GENERATING_MODEL" | sed 's/[\/\-]/_/g')
GRADING_MODEL_SAFE=$(echo "$GRADING_MODEL" | sed 's/[\/\-]/_/g')

# Create output directory name
OUTPUT_DIR="generation_output/${EXP_NAME}__${GENERATING_MODEL_SAFE}__${GRADING_MODEL_SAFE}"

echo "=========================================="
echo "AUTOMATED GENERATION AND GRADING"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Exp results:      $EXP_RESULTS_PATH"
echo "  Exp name:         $EXP_NAME"
echo "  Generating model: $GENERATING_MODEL"
echo "  Grading model:    $GRADING_MODEL"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "=========================================="
echo ""

# Step 1: Generate outputs
echo "Step 1: Generating outputs with $GENERATING_MODEL..."
echo ""

# Build command with optional start/end idx
GEN_CMD="python3 generate_model_outputs.py \
    --exp_results_path \"$EXP_RESULTS_PATH\" \
    --model \"$GENERATING_MODEL\" \
    --temperature 0 \
    --max_tokens 1024 \
    --output_dir \"$OUTPUT_DIR\""

if [ -n "$START_IDX" ]; then
    GEN_CMD="$GEN_CMD --start_idx $START_IDX"
fi

if [ -n "$END_IDX" ]; then
    GEN_CMD="$GEN_CMD --end_idx $END_IDX"
fi

eval "$GEN_CMD" || {
    echo "Error: Generation failed!"
    exit 1
}

echo ""
echo "=========================================="
echo ""

# Step 2: Grade outputs
echo "Step 2: Grading outputs with $GRADING_MODEL..."
echo ""

# Build command with optional start/end idx
GRADE_CMD="python3 grade_model_outputs.py \
    --results_path \"$OUTPUT_DIR/results.json\" \
    --grader_model \"$GRADING_MODEL\" \
    --temperature 0 \
    --max_tokens 200 \
    --output_dir \"$OUTPUT_DIR\""

if [ -n "$START_IDX" ]; then
    GRADE_CMD="$GRADE_CMD --start_idx $START_IDX"
fi

if [ -n "$END_IDX" ]; then
    GRADE_CMD="$GRADE_CMD --end_idx $END_IDX"
fi

eval "$GRADE_CMD" || {
    echo "Error: Grading failed!"
    exit 1
}

echo ""
echo "=========================================="
echo "WORKFLOW COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  Output directory: $OUTPUT_DIR/"
echo "  Generation:       $OUTPUT_DIR/results.json"
echo "  Grading:          $OUTPUT_DIR/grading_results.json"
echo ""
echo "Quick stats:"
if [ -f "$OUTPUT_DIR/grading_results.json" ]; then
    python3 << EOF
import json
with open("$OUTPUT_DIR/grading_results.json", "r") as f:
    data = json.load(f)
ratings = [v['optimized_rating'] for v in data.values() if v.get('optimized_rating') is not None]
if ratings:
    avg = sum(ratings) / len(ratings)
    print(f"  Total entries: {len(data)}")
    print(f"  Average rating: {avg:.2f}")
    print(f"  Rating distribution: 1={ratings.count(1)}, 2={ratings.count(2)}, 3={ratings.count(3)}")
else:
    print(f"  Total entries: {len(data)}")
    print(f"  No ratings generated")
EOF
fi
echo ""
echo "=========================================="
