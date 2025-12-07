#!/usr/bin/env python3
"""
Main script to run the counterfactual probing data collection experiment.

Configure parameters at the top of this file, then run:
    python run_experiment.py
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    print("Progress bars will be limited without tqdm.\n")

from rollouts.generate import RolloutGenerator


# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Data files
PROMPTS_FILE = "src/data/reward_hacking_prompts.json"
OUTPUT_DIR = "src/rollouts"

# API configuration
API_URL = "http://localhost:8000"
MODEL = "openai/gpt-oss-20b"

# Generation parameters
TEMPERATURE = 0.7
MAX_TOKENS = 32000

# Experiment parameters
NUM_COUNTERFACTUALS = 50  # Number of rollouts per sentence
NUM_INITIAL_ROLLOUTS = 1  # Number of initial rollouts per prompt (typically 1)
N_INITIAL_ROLLOUT_TESTS = 10  # Number of initial rollouts to generate and test before selecting one
INITIAL_ROLLOUT_SELECTION_STRATEGY = "MIN"  # Strategy for selecting from N_INITIAL_ROLLOUT_TESTS: "MIN" or "MEDIAN"

# Parallelization
MAX_WORKERS = 120 # Number of parallel API requests for rollout generation
                  # Increase if your API server can handle more concurrent requests
                  # Typical vLLM can handle 10-50 concurrent requests depending on GPU memory

# Processing options
PROCESS_ALL_PROMPTS = True  # If False, will process only the first prompt
PROMPT_INDICES = None  # Optional: specify which prompts to process (e.g., [0, 1, 2])
                       # Only used if PROCESS_ALL_PROMPTS is False
SKIP_IF_EXISTS = True  # Skip prompts that already have output files (for resuming)

# ============================================================================


def main():
    """Run the counterfactual probing experiment."""
    
    print("=" * 70)
    print("COUNTERFACTUAL PROBING DATA COLLECTION EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Prompts file: {PROMPTS_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  API URL: {API_URL}")
    print(f"  Model: {MODEL}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Counterfactuals per sentence: {NUM_COUNTERFACTUALS}")
    print(f"  Initial rollouts per prompt: {NUM_INITIAL_ROLLOUTS}")
    print(f"  Initial rollout tests: {N_INITIAL_ROLLOUT_TESTS}")
    print(f"  Initial rollout selection: {INITIAL_ROLLOUT_SELECTION_STRATEGY}")
    print(f"  Max parallel workers: {MAX_WORKERS}")
    print(f"  Skip if exists: {SKIP_IF_EXISTS}")
    print()
    
    # Initialize generator
    generator = RolloutGenerator(
        api_url=API_URL, 
        model=MODEL, 
        max_workers=MAX_WORKERS,
        n_initial_rollout_tests=N_INITIAL_ROLLOUT_TESTS,
        initial_rollout_selection_strategy=INITIAL_ROLLOUT_SELECTION_STRATEGY
    )
    generator.temperature = TEMPERATURE
    generator.max_tokens = MAX_TOKENS
    generator.num_counterfactuals = NUM_COUNTERFACTUALS
    
    # Load prompts
    import json
    prompts_path = Path(PROMPTS_FILE)
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {PROMPTS_FILE}")
        return 1
    
    with open(prompts_path) as f:
        all_prompts = json.load(f)
    
    print(f"Loaded {len(all_prompts)} prompt(s) from {PROMPTS_FILE}")
    
    # Determine which prompts to process
    if PROCESS_ALL_PROMPTS:
        prompts_to_process = all_prompts
    elif PROMPT_INDICES is not None:
        prompts_to_process = [all_prompts[i] for i in PROMPT_INDICES if 0 <= i < len(all_prompts)]
    else:
        prompts_to_process = all_prompts[:1]  # Default to first prompt
    
    print(f"Processing {len(prompts_to_process)} prompt(s)\n")
    print("=" * 70)
    
    # Process each prompt
    results = []
    experiment_start = time.time()
    
    for idx, prompt_data in enumerate(prompts_to_process):
        prompt_name = prompt_data.get("name", f"prompt_{idx}")
        prompt_start = time.time()
        
        # Create a progress bar for this specific prompt that tracks sentences
        if HAS_TQDM:
            # Create a progress bar that will track sentences within this prompt
            prompt_pbar = tqdm(
                total=1,  # Will be updated when we know the sentence count
                desc=f"Prompt {idx+1}/{len(prompts_to_process)}: {prompt_name[:40]}",
                unit="sentence",
                leave=True,  # Keep the bar visible after completion
                position=0
            )
        else:
            prompt_pbar = None
            print(f"\n[{idx + 1}/{len(prompts_to_process)}] Processing: {prompt_name}")
            print("-" * 70)
        
        try:
            # Generate initial rollout(s) if needed (currently always 1)
            for rollout_idx in range(NUM_INITIAL_ROLLOUTS):
                if NUM_INITIAL_ROLLOUTS > 1 and not HAS_TQDM:
                    print(f"\nInitial rollout {rollout_idx + 1}/{NUM_INITIAL_ROLLOUTS}")
                
                sentence_data_list = generator.process_prompt(
                    prompt_data, 
                    output_dir=OUTPUT_DIR,
                    skip_if_exists=SKIP_IF_EXISTS,
                    prompt_pbar=prompt_pbar
                )
                
                if sentence_data_list:
                    results.append({
                        "prompt_name": prompt_name,
                        "rollout_index": rollout_idx,
                        "num_sentences": len(sentence_data_list),
                        "sentence_data": sentence_data_list
                    })
                    
                    prompt_time = time.time() - prompt_start
                    
                    # Close the progress bar for this prompt
                    if prompt_pbar:
                        prompt_pbar.close()
                else:
                    if prompt_pbar:
                        prompt_pbar.close()
                    if not HAS_TQDM:
                        print(f"\n✗ Failed to process: {prompt_name}")
        
        except Exception as e:
            if prompt_pbar:
                prompt_pbar.close()
            if not HAS_TQDM:
                print(f"\n✗ Error processing {prompt_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    total_time = time.time() - experiment_start
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nProcessed {len(results)} prompt(s) successfully")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    
    if results:
        total_sentences = sum(r["num_sentences"] for r in results)
        total_rollouts = total_sentences * NUM_COUNTERFACTUALS
        print(f"Total sentences processed: {total_sentences}")
        print(f"Total rollouts generated: {total_rollouts}")
        print(f"Average time per prompt: {total_time/len(results):.1f} seconds")
    
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"\nOutput file names:")
    for result in results:
        prompt_name = result["prompt_name"]
        output_file = Path(OUTPUT_DIR) / f"{prompt_name}_rollouts.json"
        if output_file.exists():
            file_size = output_file.stat().st_size / 1024  # KB
            print(f"  - {output_file} ({file_size:.1f} KB)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

