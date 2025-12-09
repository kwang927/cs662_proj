# Evil Twin Prompt Optimization

This repository implements methods for finding "evil twin" prompts - adversarially optimized prompts that elicit similar outputs from language models despite looking very different from the original prompts. The optimization uses Greedy Coordinate Gradient (GCG) based methods.

## Project Structure

```
cs662_proj/
├── data/                          # Prompt datasets and configurations
│   ├── full_gt_prompts.json       # Ground truth prompts (100 prompts)
│   ├── warm_start_prompts_from_original_paper.json
│   ├── prune_warm_start_prompts_from_original_paper.json
│   ├── fluency_warm_start_prompts_from_original_paper.json
│   └── pruned_vicuna_token_ids.json
├── generation/                    # Model generation and grading
│   ├── generate_model_outputs.py  # Generate outputs from optimized prompts
│   ├── grade_model_outputs.py     # Grade outputs using LLM-as-judge
│   ├── run_generation_and_grading.sh
│   └── models/                    # API integrations (OpenAI, Anthropic, Google, etc.)
├── exp_results/                   # Consolidated experiment results
├── optim_logs/                    # Detailed per-prompt optimization logs
├── find_evil_twins.py             # Main optimization script
├── func_from_evil_twins.py        # Core optimization functions
└── util.py                        # Utility functions
```

## Installation

### Dependencies

```bash
pip install torch transformers einops tqdm numpy
```

### Optional Dependencies

```bash
pip install flash-attn  # For Flash Attention 2 speedup
pip install scipy       # For statistical analysis
```

### API Keys (for generation/grading)

Set the following environment variables for model API access:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

## Finding Evil Twins

The main script `find_evil_twins.py` supports four optimization methods:

### Method 1: Cold Start

Start from random initialization ("!" * 15) and optimize from scratch:

```bash
python find_evil_twins.py \
    --original_prompts ./data/full_gt_prompts.json \
    --output_path ./outputs/cold_start_vicuna-7b.json \
    --model_name "lmsys/vicuna-7b-v1.5" \
    --batch_size 4 \
    --n_epochs 50
```

**Characteristics:**
- No starting prompt provided (defaults to "!" repeated 15 times)
- Baseline approach
- Typically produces lower quality optimized prompts

### Method 2: Warm Start

Initialize from semantically-related starting prompts:

```bash
python find_evil_twins.py \
    --original_prompts ./data/full_gt_prompts.json \
    --output_path ./outputs/warm_start__vicuna-7b.json \
    --model_name "lmsys/vicuna-7b-v1.5" \
    --batch_size 4 \
    --n_epochs 50 \
    --starting_prompts ./data/warm_start_prompts_from_original_paper.json
```

**Characteristics:**
- Uses `--starting_prompts` for informed initialization
- Converges faster to better solutions
- Recommended baseline method

### Method 3: Warm Start + Fluency Penalty

Optimize with a fluency constraint to keep prompts more natural:

```bash
python find_evil_twins.py \
    --original_prompts ./data/full_gt_prompts.json \
    --output_path ./outputs/fluency_warm_start__vicuna-7b.json \
    --model_name "lmsys/vicuna-7b-v1.5" \
    --batch_size 4 \
    --n_epochs 50 \
    --starting_prompts ./data/fluency_warm_start_prompts_from_original_paper.json \
    --gamma 0.05
```

**Characteristics:**
- Includes `--gamma` fluency penalty coefficient
- Balances optimization effectiveness with readability
- Results in more human-like optimized prompts

### Method 4: Warm Start + Pruned Vocabulary

Optimize with a restricted token vocabulary:

```bash
python find_evil_twins.py \
    --original_prompts ./data/full_gt_prompts.json \
    --output_path ./outputs/prune_warm_start__vicuna-7b.json \
    --model_name "lmsys/vicuna-7b-v1.5" \
    --batch_size 4 \
    --n_epochs 50 \
    --starting_prompts ./data/prune_warm_start_prompts_from_original_paper.json \
    --pruned_vocab ./data/pruned_vicuna_token_ids.json
```

**Characteristics:**
- Uses `--pruned_vocab` to restrict optimization to allowed tokens
- Results in grammatically more correct prompts
- Useful for safety-conscious applications

## Command Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--original_prompts` | Path to JSON file with original prompts |
| `--output_path` | Path to store optimization results |
| `--model_name` | HuggingFace model name (e.g., "lmsys/vicuna-7b-v1.5") |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--starting_prompts` | None | Path to JSON file with starting prompts (warm start) |
| `--n_epochs` | 50 | Number of optimization epochs |
| `--n_docs` | 100 | Number of document continuations to generate |
| `--doc_len` | 32 | Length of each continuation in tokens |
| `--batch_size` | 50 | Batch size for forward passes |
| `--top_k` | 256 | Top-k tokens to keep from gradients |
| `--gamma` | 0.0 | Fluency penalty coefficient |
| `--pruned_vocab` | None | Path to JSON file with allowed token IDs |
| `--log_dir` | ./optimization_logs | Directory for optimization logs |
| `--kl_every` | 1 | Compute KL divergence every N epochs |
| `--early_stop_kl` | -1e13 | KL threshold for early stopping |
| `--use_flash_attn_2` | False | Enable Flash Attention 2 |
| `--verbose` | False | Print detailed progress |

## Supported Models

The system supports multiple model families:

**Chat Models:**
- Vicuna (lmsys/vicuna-7b-v1.5, lmsys/vicuna-13b-v1.5)
- LLaMA-2-Chat, LLaMA-3-Instruct
- Mistral, Gemma-2-it, Qwen-2-Instruct

**Base Models:**
- Pythia (EleutherAI/pythia-1.4b, pythia-6.9b, etc.)
- OPT, Phi, GPT-2

## Data Format

### Input Prompts (JSON)

```json
{
  "1": "When were atomic weapons first used?",
  "2": "How can cities become more eco-friendly?",
  "3": "Write a customer service response..."
}
```

### Output Results (JSON)

```json
{
  "1": {
    "original_prompt": "When were atomic weapons first used?",
    "starting_prompt": "Explain the events of the first atomic bombing...",
    "optimized_prompt": "Explain oraz events of про firstlices...",
    "optimized_token_ids": [[...]],
    "final_metrics": {
      "best_kl": 8.856,
      "best_std": 0.220,
      "best_loss": 0.405,
      "final_epoch": 50
    },
    "log_file": "optim_logs/vicuna-7b-v1_5__warm_start_prompt_1_log.json"
  }
}
```

## Evaluation Pipeline

After finding evil twins, evaluate them using the generation and grading pipeline:

### 1. Generate Model Outputs

```bash
cd generation
bash run_generation_and_grading.sh ../exp_results/warm_start__vicuna-7b.json gpt-4
```

Or run steps separately:

```bash
python generate_model_outputs.py \
    --exp_results_path ../exp_results/warm_start__vicuna-7b.json \
    --model "gpt-4" \
    --output_dir ./generation_output/my_experiment
```

### 2. Grade Outputs

```bash
python grade_model_outputs.py \
    --results_path ./generation_output/my_experiment/results.json \
    --grader_model "gpt-4" \
    --output_dir ./generation_output/my_experiment
```

### 3. View Results

Use the helper script to inspect results:

```bash
python print_evil_twin_outputs.py \
    generation_output/warm_start__vicuna-7b__gpt_4__gpt_4/results.json \
    --limit 5
```

## Example Workflow

```bash
# 1. Run warm start optimization
python find_evil_twins.py \
    --original_prompts ./data/full_gt_prompts.json \
    --starting_prompts ./data/warm_start_prompts_from_original_paper.json \
    --output_path ./outputs/my_experiment.json \
    --model_name "lmsys/vicuna-7b-v1.5" \
    --n_epochs 50 \
    --batch_size 4

# 2. Generate and grade outputs
cd generation
bash run_generation_and_grading.sh ../outputs/my_experiment.json gpt-4

# 3. Check results
python print_evil_twin_outputs.py \
    generation_output/my_experiment__gpt_4__gpt_4/grading_results.json \
    --summary
```

## Results

Experiment results are stored in `exp_results/` directory:

| File | Method |
|------|--------|
| `cold_start_vicuna-7b-v15.json` | Cold Start |
| `warm_start__vicuna-7b.json` | Warm Start |
| `fluency_warm_start__vicuna-7b.json` | Warm Start + Fluency |
| `prune_warm_start__vicuna-7b.json` | Warm Start + Pruned Vocab |

## Notes

- Prompts with id `[7, 86, 97]` cannot be found in Alpaca's training set.
