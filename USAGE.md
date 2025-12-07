# Counterfactual Probing Data Collection - Usage Guide

## Overview
This system generates rollouts from reward hacking prompts, splits them by sentence, and estimates P(reward hacks | sequence) through counterfactual sampling.

## Quick Start

1. **Configure parameters** in `run_experiment.py` (see Configuration section below)

2. **Run the experiment**:
```bash
python run_experiment.py
```

This will process all prompts and save results to `src/rollouts/`.

## Configuration

Edit the configuration section at the top of `run_experiment.py`:

```python
# Experiment parameters
NUM_COUNTERFACTUALS = 50  # Number of rollouts per sentence
NUM_INITIAL_ROLLOUTS = 1  # Number of initial rollouts per prompt

# Processing options
PROCESS_ALL_PROMPTS = True  # Process all prompts or just some?
PROMPT_INDICES = None  # Or specify: [0, 1, 2] for specific prompts
```

## Components

### Utils Modules
- **`src/utils/api_client.py`**: vLLM API client for querying the model
- **`src/utils/chat_utils.py`**: Chat message formatting utilities
- **`src/utils/reward_detector.py`**: Detects reward hacking attempts (searches for `expected.json`)
- **`src/utils/text_utils.py`**: Sentence splitting utilities

### Main Generator
- **`src/rollouts/generate.py`**: Core orchestration logic
- **`run_experiment.py`**: Main entry point with configuration

### Output Format
Each output file contains a list of sentence data:
```json
[
  {
    "sentence_index": 0,
    "sentence": "First sentence. ",
    "p_reward_hacks": 0.42,
    "rollouts": ["continuation 1", "continuation 2", ...],
    "formatted_prompt": [{"role": "user", "content": "..."}],
    "unformatted_prompt": "..."
  },
  ...
]
```

## Testing
Test scripts are available:
- `test_api_client.py` - Test API connectivity
- `test_chat_utils.py` - Test chat formatting
- `test_reward_detector.py` - Test reward hacking detection
- `test_text_utils.py` - Test sentence splitting
- `test_schema.py` - Test data schema
- `test_full_pipeline.py` - Test full pipeline with minimal example

## Configuration
Default settings:
- Temperature: 0.7
- Max tokens: 32,000
- Counterfactuals per sentence: 50
- vLLM API: http://localhost:8000

