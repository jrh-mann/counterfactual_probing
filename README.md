# Counterfactual Probing

A library for counterfactual probing of language models. Generate rollouts, sample branch points at the token level, and measure how model behavior changes with counterfactual continuations.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Create a configuration file

```bash
cfprobe init --output config.json
```

Or create one manually:

```json
{
  "dataset": {
    "path": "data/prompts.jsonl",
    "prompt_field": "prompt",
    "format": "jsonl"
  },
  "model": {
    "name": "Qwen/Qwen2.5-0.5B"
  },
  "generation": {
    "temperature": 0.7,
    "max_tokens": 2048,
    "num_counterfactuals": 50
  },
  "sampling": {
    "method": "uniform_count",
    "num_samples": 20
  },
  "scorer": {
    "module": "my_scorer",
    "class": "MyScorer"
  },
  "output": {
    "dir": "outputs/"
  }
}
```

### 2. Create a dataset

Create a JSONL file with your prompts:

```jsonl
{"prompt": "What is 2+2?", "id": "math_1"}
{"prompt": "Write a haiku about programming.", "id": "creative_1"}
```

### 3. Create a custom scorer

```python
# my_scorer.py
from counterfactual_probing.scorer import Scorer

class MyScorer(Scorer):
    def score(self, text: str) -> float:
        # Return 0.0-1.0 based on your criteria
        if "bad_pattern" in text:
            return 1.0
        return 0.0
```

### 4. Run the pipeline

```bash
cfprobe run --config config.json
```

Or from Python:

```python
from counterfactual_probing import run

run("config.json")
```

## Output Format

Each output file contains:

```json
{
  "prompt_id": "math_1",
  "metadata": {},
  "initial_rollout": {
    "token_ids": [1, 234, 567, ...],
    "text": "The answer is 4..."
  },
  "samples": [
    {
      "token_index": 42,
      "prefix_token_ids": [1, 234, ...],
      "counterfactuals": [
        {"token_ids": [...], "score": 0.0},
        {"token_ids": [...], "score": 0.0}
      ],
      "p_score": 0.12
    }
  ]
}
```

## Configuration Reference

### Dataset

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | required | Path to dataset file |
| `prompt_field` | string | "prompt" | Field name containing the prompt |
| `format` | string | auto | File format: jsonl, json, csv |

### Model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Model name or path |
| `tensor_parallel_size` | int | 1 | vLLM tensor parallel size |
| `gpu_memory_utilization` | float | 0.9 | GPU memory fraction |

### Generation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float | 0.7 | Sampling temperature |
| `max_tokens` | int | 4096 | Maximum tokens to generate |
| `num_counterfactuals` | int | 50 | Counterfactuals per branch point |

### Sampling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | string | "uniform_count" | "uniform_count" or "density" |
| `num_samples` | int | 20 | Number of branch points (uniform_count) |
| `density` | float | 0.02 | Sampling density (density method) |
| `seed` | int | null | Random seed for reproducibility |

### Scorer

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `module` | string | required | Python module path |
| `class` | string | required | Scorer class name |
| `config` | object | {} | Config passed to scorer __init__ |

## Built-in Scorers

### DummyScorer

Returns a constant value (for testing):

```json
{
  "module": "counterfactual_probing.scorer.examples.dummy",
  "class": "DummyScorer",
  "config": {"return_value": 0.5}
}
```

### RewardHackingScorer

Detects attempts to manipulate reward by referencing `expected.json`:

```json
{
  "module": "counterfactual_probing.scorer.examples.reward_hacking",
  "class": "RewardHackingScorer"
}
```

## Examples

See the `examples/` directory for complete working examples.

## Development

Run tests:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=counterfactual_probing --cov-report=html
```
