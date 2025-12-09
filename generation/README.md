# Generation Module

This module provides API wrappers for generating text from various LLM providers and evaluating model outputs using LLM-as-judge.

## Setup

### 1. Install Dependencies

```bash
pip install openai anthropic google-generativeai together xai-sdk
```

### 2. Configure API Keys

Edit the API key in each model file under `models/`:

| File | Variable | Provider |
|------|----------|----------|
| `openai_related.py` | `openai_api_key` | OpenAI |
| `anthropic_related.py` | `claude_api_key` | Anthropic |
| `google_related.py` | `GOOGLE_API_KEY` | Google |
| `togetherai_related.py` | `together_ai_key` | Together AI |
| `xai_related.py` | `xai_api_key` | xAI |

## API Functions

### OpenAI (`models/openai_related.py`)

| Function | Description |
|----------|-------------|
| `gpt_generation` | Single-turn text generation |
| `gpt_generation_multi_turn` | Multi-turn conversation |
| `gpt_generation_with_image` | Vision models with image input |
| `gpt_reasoning_generation` | Reasoning models (o1, o3, gpt-5-mini) |
| `gpt_reasoning_generation_multi_turn` | Multi-turn reasoning |

### Anthropic (`models/anthropic_related.py`)

| Function | Description |
|----------|-------------|
| `generate_claude_response` | Single-turn text generation |
| `generate_claude_response_multi_turn` | Multi-turn conversation |
| `generate_claude_response_with_image` | Vision with image input |

### Google Gemini (`models/google_related.py`)

| Function | Description |
|----------|-------------|
| `generate_gemini_response` | Pipeline-compatible generation |
| `call_gemini` | Low-level call with system prompt support |

**Note:** The Gemini wrapper includes built-in rate limiting (25 requests/min, 1M tokens/min).

### Together AI (`models/togetherai_related.py`)

| Function | Description |
|----------|-------------|
| `togetherai_generation` | Single-turn text generation |
| `togetherai_generation_multi_turn` | Multi-turn conversation |

### xAI Grok (`models/xai_related.py`)

| Function | Description |
|----------|-------------|
| `generate_grok_response_multi_turn` | Multi-turn conversation |

## Pipeline Scripts

### Generate Model Outputs

Generate responses for both original and optimized (evil twin) prompts:

```bash
python generate_model_outputs.py \
    --exp_results_path ../exp_results/warm_start__vicuna-7b.json \
    --model "gpt-4o" \
    --temperature 0 \
    --max_tokens 1024 \
    --output_dir ./generation_output/my_experiment
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--exp_results_path` | Yes | - | Path to experiment results JSON |
| `--model` | Yes | - | Model name for generation |
| `--temperature` | No | 0.0 | Sampling temperature |
| `--max_tokens` | No | 1024 | Maximum tokens to generate |
| `--top_p` | No | 1.0 | Top-p sampling parameter |
| `--timeout_seconds` | No | 60 | Timeout per generation |
| `--start_idx` | No | None | Start index (for subset) |
| `--end_idx` | No | None | End index (for subset) |
| `--output_dir` | No | Auto | Output directory |

**Output Files:**
- `results.json` - Generation results
- `results.pickle` - Pickle format
- `results_checkpoint.json` - Intermediate checkpoint

### Grade Model Outputs

Grade how well responses answer the original prompts using LLM-as-judge:

```bash
python grade_model_outputs.py \
    --results_path ./generation_output/my_experiment/results.json \
    --grader_model "gpt-4" \
    --temperature 0 \
    --max_tokens 200 \
    --output_dir ./generation_output/my_experiment
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--results_path` | Yes | - | Path to results.json from generation |
| `--grader_model` | No | gpt-4o | Model for grading |
| `--temperature` | No | 0.0 | Sampling temperature |
| `--max_tokens` | No | 200 | Max tokens for grading response |
| `--timeout_seconds` | No | 60 | Timeout per grading |
| `--start_idx` | No | None | Start index (for subset) |
| `--end_idx` | No | None | End index (for subset) |
| `--output_dir` | No | Auto | Output directory |

**Output Files:**
- `grading_results.json` - Grading results with ratings (1-3)
- `grading_results.pickle` - Pickle format
- `grading_checkpoint.json` - Intermediate checkpoint

**Rating Scale:**
- **1**: Response does not answer the prompt at all
- **2**: Response gets the general idea and partially answers
- **3**: Response faithfully answers the prompt

### Automated Pipeline

Run both generation and grading in one command:

```bash
bash run_generation_and_grading.sh <exp_results_path> <generating_model> [start_idx] [end_idx]
```

**Examples:**

```bash
# Full pipeline with GPT-4o
bash run_generation_and_grading.sh ../exp_results/warm_start__vicuna-7b.json gpt-4o

# With Claude
bash run_generation_and_grading.sh ../exp_results/warm_start__pythia-1.4b.json claude-3-haiku-20240307

# Process subset (entries 1-10)
bash run_generation_and_grading.sh ../exp_results/warm_start__vicuna-7b.json gpt-4o-mini 1 10
```

**Note:** Grading is always done with GPT-4 for consistency.

## Viewing Results

Use the helper script to inspect results:

```bash
# View first 5 entries
python print_evil_twin_outputs.py \
    generation_output/warm_start__vicuna-7b__gpt_4__gpt_4/results.json \
    --limit 5

# View specific entry
python print_evil_twin_outputs.py \
    generation_output/warm_start__vicuna-7b__gpt_4__gpt_4/results.json \
    --id 42

# View summary statistics
python print_evil_twin_outputs.py \
    generation_output/warm_start__vicuna-7b__gpt_4__gpt_4/results.json \
    --summary

# Compare original and evil twin outputs
python print_evil_twin_outputs.py \
    generation_output/warm_start__vicuna-7b__gpt_4__gpt_4/results.json \
    --limit 3 --show-original
```

## Supported Models

### OpenAI

**Standard Models:**
- `gpt-4o`, `gpt-4o-mini`
- `gpt-4`, `gpt-4-turbo`, `gpt-4-turbo-preview`
- `gpt-4-32k`, `gpt-4-vision-preview`
- `gpt-3.5-turbo-0125`

**Reasoning Models** (use `gpt_reasoning_generation`):
- `o1`, `o3-mini`
- `gpt-5-mini-2025-08-07`

### Anthropic

- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`, `claude-3-5-sonnet-20240620`
- `claude-3-haiku-20240307`

### Google Gemini

- `gemini-1.0-pro-latest`, `gemini-1.0-pro-vision-latest`
- `gemini-1.5-pro`, `gemini-1.5-pro-latest`
- `gemini-2.0-flash-lite`
- `gemini-2.5-pro-preview`
- `gemini-3-pro-preview`


## Cost Estimation

The scripts automatically calculate cost estimates based on `model_price.json`. Prices are in USD per million tokens.

Example output:
```
Token Usage:
  Generated 200 outputs
  Input tokens: 45,230
  Output tokens: 128,450

Cost Estimation:
  Input cost: $0.2262
  Output cost: $1.9268
  Total cost: $2.1529
```

## Data Format

### Input (exp_results JSON)

```json
{
  "1": {
    "original_prompt": "When were atomic weapons first used?",
    "optimized_prompt": "Explain oraz events of про firstlices...",
    "starting_prompt": "Explain the events of the first atomic bombing...",
    "final_metrics": { ... }
  }
}
```

### Output (results.json)

```json
{
  "1": {
    "original_prompt": "When were atomic weapons first used?",
    "optimized_prompt": "Explain oraz events of про firstlices...",
    "original_output": "Atomic weapons were first used in combat on August 6, 1945...",
    "optimized_output": "It appears that your sentence is a mix of several languages...",
    "starting_prompt": "...",
    "final_metrics": { ... }
  }
}
```

### Output (grading_results.json)

```json
{
  "1": {
    "original_prompt": "When were atomic weapons first used?",
    "optimized_prompt": "Explain oraz events of про firstlices...",
    "optimized_output": "...",
    "optimized_rating": 3,
    "optimized_explanation": "The response accurately answers the prompt...",
    ...
  }
}
```
