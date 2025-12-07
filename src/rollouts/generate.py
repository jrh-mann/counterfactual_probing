"""
Main rollout generation script.
Orchestrates the full pipeline for counterfactual probing data collection.
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    # tqdm not available, use dummy function
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import VLLMClient
from utils.chat_utils import concatenate_prompt, format_chat_messages
from utils.reward_detector import detect_reward_hacking
from utils.text_utils import split_sentences, get_sentence_prefix
from rollouts.schema import create_sentence_data, validate_sentence_data


class RolloutGenerator:
    """Main class for generating rollouts and computing reward hacking probabilities."""
    
    def __init__(self, api_url: str = "http://localhost:8000", model: str = "openai/gpt-oss-20b", 
                 max_workers: int = 10, n_initial_rollout_tests: int = 10,
                 initial_rollout_selection_strategy: str = "MIN"):
        self.client = VLLMClient(base_url=api_url, model=model)
        self.temperature = 0.7
        self.max_tokens = 32000
        self.num_counterfactuals = 50
        self.max_workers = max_workers  # Maximum total concurrent API requests
        self.n_initial_rollout_tests = n_initial_rollout_tests  # Number of initial rollouts to test
        self.initial_rollout_selection_strategy = initial_rollout_selection_strategy.upper()  # MIN or MEDIAN
        self._lock = Lock()  # For thread-safe operations
        self._api_semaphore = Semaphore(max_workers)  # Limit total concurrent API requests
    
    def load_prompts(self, prompts_path: str) -> List[Dict[str, str]]:
        """Load reward hacking prompts from JSON file."""
        with open(prompts_path, 'r') as f:
            return json.load(f)
    
    def generate_initial_rollout(self, prompt_data: Dict[str, str]) -> Optional[str]:
        """
        Generate initial rollout for a prompt.
        
        Args:
            prompt_data: Dict with 'task', 'rules', and 'grader' keys
        
        Returns:
            Generated text or None if failed
        """
        # Concatenate prompt components
        unformatted_prompt = concatenate_prompt(
            prompt_data["task"],
            prompt_data["rules"],
            prompt_data["grader"]
        )
        
        # Format for chat API
        messages = format_chat_messages(unformatted_prompt)
        
        # Acquire semaphore to limit total concurrent API requests
        self._api_semaphore.acquire()
        try:
            # Generate response
            response = self.client.generate(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if not response:
                return None
            
            return self.client.extract_content(response)
        finally:
            self._api_semaphore.release()
    
    def _generate_single_rollout(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Generate a single rollout (thread-safe helper for parallel execution)."""
        # Acquire semaphore to limit total concurrent API requests
        self._api_semaphore.acquire()
        try:
            response = self.client.generate(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            if response:
                return self.client.extract_content(response)
            return None
        finally:
            self._api_semaphore.release()
    
    def generate_counterfactual_rollouts(
        self,
        prefix: str,
        unformatted_prompt: str,
        num_rollouts: int = 50,
        desc: str = "Rollouts",
        show_rollout_progress: bool = False
    ) -> List[str]:
        """
        Generate multiple counterfactual rollouts from a given prefix (parallelized).
        
        The model continues from the prefix as if it had already generated it.
        
        Args:
            prefix: Text prefix (all prior sentences + current sentence)
            unformatted_prompt: Original unformatted prompt
            num_rollouts: Number of rollouts to generate
            desc: Description for progress bar
            show_rollout_progress: Whether to show individual rollout progress (default: False for cleaner output)
        
        Returns:
            List of generated continuations (just the continuation part, not including prefix)
        """
        # Create messages: user prompt, then assistant prefix (what was generated so far in reasoning)
        # The prefix is the reasoning chain up to this sentence
        messages = format_chat_messages(unformatted_prompt)
        # Add the reasoning prefix as assistant message
        # Note: We're only tracking reasoning, so prefix contains reasoning sentences
        messages.append({"role": "assistant", "content": prefix})
        
        rollouts = []
        start_time = time.time()
        completed = 0
        
        if HAS_TQDM and show_rollout_progress:
            pbar = tqdm(total=num_rollouts, desc=desc, unit="rollout", leave=False, position=1)
        else:
            pbar = None
        
        # Use ThreadPoolExecutor for parallel generation
        # Note: We can submit all tasks - the semaphore will limit actual concurrent API calls
        with ThreadPoolExecutor(max_workers=num_rollouts) as executor:
            # Submit all tasks
            futures = [
                executor.submit(self._generate_single_rollout, messages)
                for _ in range(num_rollouts)
            ]
            
            # Process completed tasks as they finish
            for future in as_completed(futures):
                try:
                    content = future.result()
                    if content:
                        with self._lock:
                            rollouts.append(content)
                            completed += 1
                    else:
                        completed += 1
                    
                    # Update progress bar
                    if pbar:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed if completed > 0 else 0
                        remaining = avg_time * (num_rollouts - completed)
                        pbar.update(1)
                        pbar.set_postfix({
                            'done': f'{completed}/{num_rollouts}',
                            'ETA': f'{remaining:.0f}s',
                            'workers': self.max_workers
                        })
                except Exception as e:
                    completed += 1
                    if pbar:
                        pbar.update(1)
                    # Log error but continue
                    if not HAS_TQDM:
                        print(f"  Warning: Rollout generation failed: {e}")
        
        if pbar:
            pbar.close()
        
        return rollouts
    
    def _process_single_sentence(
        self,
        sentence_idx: int,
        sentences: List[str],
        unformatted_prompt: str,
        show_rollout_progress: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single sentence: generate counterfactuals and compute P(reward hacks).
        
        Args:
            sentence_idx: Index of the sentence to process
            sentences: List of all sentences
            unformatted_prompt: Original unformatted prompt
            show_rollout_progress: Whether to show rollout progress bars
        
        Returns:
            Sentence data dictionary
        """
        # Get prefix up to and including this sentence
        prefix = get_sentence_prefix(sentences, sentence_idx)
        current_sentence = sentences[sentence_idx]
        
        # Generate counterfactual rollouts (no progress bar for cleaner output)
        rollouts = self.generate_counterfactual_rollouts(
            prefix,
            unformatted_prompt,
            num_rollouts=self.num_counterfactuals,
            desc="",
            show_rollout_progress=show_rollout_progress
        )
        
        # Detect reward hacking in each rollout
        reward_hacks = [detect_reward_hacking(r) for r in rollouts]
        p_reward_hacks = sum(reward_hacks) / len(rollouts) if rollouts else 0.0
        
        # Create sentence data
        formatted_prompt = format_chat_messages(unformatted_prompt)
        sentence_data = create_sentence_data(
            sentence_index=sentence_idx,
            sentence=current_sentence,
            p_reward_hacks=p_reward_hacks,
            rollouts=rollouts,
            formatted_prompt=formatted_prompt,
            unformatted_prompt=unformatted_prompt
        )
        
        return sentence_data
    
    def process_prompt(
        self,
        prompt_data: Dict[str, str],
        output_dir: str = "src/rollouts",
        skip_if_exists: bool = True,
        prompt_pbar: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a single prompt: generate initial rollout, split, and compute probabilities.
        
        Args:
            prompt_data: Prompt data dict
            output_dir: Directory to save output
            skip_if_exists: Whether to skip if output file exists
            prompt_pbar: Optional tqdm progress bar to update for sentence progress
        
        Returns:
            List of sentence data dictionaries
        """
        prompt_name = prompt_data.get("name", "unknown")
        output_path = Path(output_dir) / f"{prompt_name}_rollouts.json"
        
        # Check if output already exists
        if skip_if_exists and output_path.exists():
            if prompt_pbar:
                prompt_pbar.set_postfix({'status': 'checking existing file...'})
            if not HAS_TQDM:
                print(f"\n  Skipping {prompt_name} (output already exists)")
            try:
                with open(output_path) as f:
                    data = json.load(f)
                    if prompt_pbar:
                        prompt_pbar.total = len(data)
                        prompt_pbar.n = len(data)  # Set to completed
                        prompt_pbar.refresh()
                        prompt_pbar.set_postfix({'status': 'loaded from file'})
                    return data
            except:
                pass  # If file is corrupted, regenerate
        
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': f'generating {self.n_initial_rollout_tests} initial rollouts...'})
        elif not HAS_TQDM:
            print(f"\nProcessing prompt: {prompt_name}")
            print(f"  Step 1: Generating {self.n_initial_rollout_tests} initial rollouts...")
        
        # Generate N initial rollouts in parallel to select one based on strategy
        unformatted_prompt = concatenate_prompt(
            prompt_data["task"],
            prompt_data["rules"],
            prompt_data["grader"]
        )
        messages = format_chat_messages(unformatted_prompt)
        
        initial_rollouts = []
        # Submit all rollouts - semaphore will limit concurrent API calls
        with ThreadPoolExecutor(max_workers=self.n_initial_rollout_tests) as executor:
            futures = [executor.submit(self._generate_single_rollout, messages) for _ in range(self.n_initial_rollout_tests)]
            for future in as_completed(futures):
                rollout = future.result()
                if rollout:
                    initial_rollouts.append(rollout)
        
        if not initial_rollouts:
            if prompt_pbar:
                prompt_pbar.set_postfix({'status': 'failed to generate initial rollouts'})
            if not HAS_TQDM:
                print(f"  ✗ Failed to generate initial rollouts for {prompt_name}")
            return []
        
        if prompt_pbar:
            strategy_name = self.initial_rollout_selection_strategy.lower()
            prompt_pbar.set_postfix({'status': f'calculating {strategy_name} sentence length...'})
        elif not HAS_TQDM:
            print(f"  ✓ Generated {len(initial_rollouts)} initial rollouts")
            print(f"  Step 2: Calculating {self.initial_rollout_selection_strategy} sentence length...")
        
        # Calculate sentence counts for each rollout
        rollout_sentence_counts = []
        for rollout in initial_rollouts:
            sentences = split_sentences(rollout)
            rollout_sentence_counts.append((len(sentences), rollout))
        
        sentence_counts = [count for count, _ in rollout_sentence_counts]
        
        # Select rollout based on strategy
        if self.initial_rollout_selection_strategy == "MIN":
            # Pick the rollout with minimum sentence count
            best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: x[0])
            strategy_display = "minimum"
        elif self.initial_rollout_selection_strategy == "MEDIAN":
            # Calculate median sentence count
            sentence_counts_sorted = sorted(sentence_counts)
            median_count = sentence_counts_sorted[len(sentence_counts_sorted) // 2]
            # Pick the rollout closest to the median
            best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: abs(x[0] - median_count))
            strategy_display = f"median (target: {median_count})"
        else:
            # Default to MIN if invalid strategy
            best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: x[0])
            strategy_display = "minimum (invalid strategy, defaulted)"
        
        if prompt_pbar:
            prompt_pbar.set_postfix({
                'status': f'selected rollout with {best_count} sentences ({strategy_display})',
                'sentence_counts': f'{min(sentence_counts)}-{max(sentence_counts)}'
            })
        elif not HAS_TQDM:
            print(f"  ✓ Sentence counts: {min(sentence_counts)}-{max(sentence_counts)}")
            print(f"  ✓ Selected rollout with {best_count} sentences ({strategy_display})")
            print("  Step 3: Splitting into sentences...")
        
        sentences = split_sentences(best_rollout)
        
        if prompt_pbar:
            prompt_pbar.total = len(sentences)
            prompt_pbar.n = 0  # Reset counter
            prompt_pbar.refresh()  # Refresh display
            prompt_pbar.set_postfix({'status': f'processing {len(sentences)} sentences...'})
        elif not HAS_TQDM:
            print(f"  ✓ Found {len(sentences)} sentence(s)")
        
        # unformatted_prompt already prepared above
        
        # Step 3: Process sentences in parallel
        if not HAS_TQDM and not prompt_pbar:
            print("  Step 3: Processing sentences in parallel...")
        
        sentence_data_list = []
        
        # Use ThreadPoolExecutor to process multiple sentences in parallel
        # Each sentence still generates its rollouts in parallel internally
        # The semaphore will limit total concurrent API requests across all sentences
        with ThreadPoolExecutor(max_workers=len(sentences)) as executor:
            # Submit all sentence processing tasks
            futures = {
                executor.submit(self._process_single_sentence, idx, sentences, unformatted_prompt, False): idx
                for idx in range(len(sentences))
            }
            
            # Collect results as they complete, maintaining order
            results = {}
            completed_count = 0
            start_time = time.time()
            
            for future in as_completed(futures):
                sentence_idx = futures[future]
                try:
                    sentence_data = future.result()
                    results[sentence_idx] = sentence_data
                    completed_count += 1
                    
                    # Update the prompt-level progress bar
                    if prompt_pbar:
                        prompt_pbar.update(1)
                        # Calculate stats for completed sentences
                        completed_sentences = [results[i] for i in sorted(results.keys())]
                        if completed_sentences:
                            avg_p_hacks = sum(s.get("p_reward_hacks", 0) for s in completed_sentences) / len(completed_sentences)
                            elapsed = time.time() - start_time
                            avg_time_per_sentence = elapsed / completed_count if completed_count > 0 else 0
                            remaining_sentences = len(sentences) - completed_count
                            eta = avg_time_per_sentence * remaining_sentences
                            
                            prompt_pbar.set_postfix({
                                'sentences': f'{completed_count}/{len(sentences)}',
                                'avg_p_hacks': f'{avg_p_hacks:.3f}',
                                'ETA': f'{eta/60:.1f}m' if eta > 60 else f'{eta:.0f}s'
                            })
                except Exception as e:
                    if not HAS_TQDM and not prompt_pbar:
                        print(f"  ✗ Error processing sentence {sentence_idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create empty placeholder to maintain order
                    results[sentence_idx] = create_sentence_data(
                        sentence_index=sentence_idx,
                        sentence=sentences[sentence_idx] if sentence_idx < len(sentences) else "",
                        p_reward_hacks=0.0,
                        rollouts=[],
                        formatted_prompt=format_chat_messages(unformatted_prompt),
                        unformatted_prompt=unformatted_prompt
                    )
                    completed_count += 1
                    if prompt_pbar:
                        prompt_pbar.update(1)
            
            # Sort by sentence_index to maintain order
            sentence_data_list = [results[i] for i in sorted(results.keys())]
        
        # Step 4: Save to file (output_path already set above)
        
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': 'saving...'})
        elif not HAS_TQDM:
            print(f"\n  Step 4: Saving to {output_path}...")
        
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sentence_data_list, f, indent=2)
        
        if prompt_pbar:
            # Calculate final stats
            avg_p_hacks = sum(s.get("p_reward_hacks", 0) for s in sentence_data_list) / len(sentence_data_list) if sentence_data_list else 0
            max_p_hacks = max((s.get("p_reward_hacks", 0) for s in sentence_data_list), default=0)
            prompt_pbar.set_postfix({
                'status': 'complete',
                'avg_p_hacks': f'{avg_p_hacks:.3f}',
                'max_p_hacks': f'{max_p_hacks:.3f}'
            })
        elif not HAS_TQDM:
            print(f"  ✓ Saved {len(sentence_data_list)} sentence data entries")
        
        return sentence_data_list
    
    def run(self, prompts_path: str, output_dir: str = "src/rollouts"):
        """
        Run the full pipeline on all prompts.
        
        Args:
            prompts_path: Path to prompts JSON file
            output_dir: Directory to save output files
        """
        # Load prompts
        print("Loading prompts...")
        prompts = self.load_prompts(prompts_path)
        print(f"Loaded {len(prompts)} prompt(s)")
        
        # Process each prompt
        for prompt_data in prompts:
            self.process_prompt(prompt_data, output_dir)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate counterfactual rollouts for reward hacking analysis")
    parser.add_argument(
        "--prompts",
        type=str,
        default="src/data/reward_hacking_prompts.json",
        help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/rollouts",
        help="Output directory for rollout files"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM API URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model name"
    )
    parser.add_argument(
        "--num-counterfactuals",
        type=int,
        default=50,
        help="Number of counterfactual rollouts per sentence"
    )
    
    args = parser.parse_args()
    
    generator = RolloutGenerator(api_url=args.api_url, model=args.model)
    generator.num_counterfactuals = args.num_counterfactuals
    generator.run(args.prompts, args.output_dir)


if __name__ == "__main__":
    main()

