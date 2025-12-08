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
from collections import defaultdict
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

from utils.chat_utils import concatenate_prompt, format_chat_messages, messages_to_prompt_string
from utils.reward_detector import detect_reward_hacking
from utils.text_utils import split_sentences, get_sentence_prefix
from rollouts.schema import create_sentence_data, validate_sentence_data


class RolloutGenerator:
    """Main class for generating rollouts and computing reward hacking probabilities."""
    
    def __init__(self, model: str = "openai/gpt-oss-20b", 
                 n_initial_rollout_tests: int = 10,
                 initial_rollout_selection_strategy: str = "MIN"):
        self.model = model
        self.temperature = 0.7
        self.max_tokens = 32000
        self.num_counterfactuals = 50
        self.n_initial_rollout_tests = n_initial_rollout_tests  # Number of initial rollouts to test
        self.initial_rollout_selection_strategy = initial_rollout_selection_strategy.upper()  # MIN or MEDIAN
        self.llm = None  # Will be set from run_experiment.py
        self.sampling_params = None  # Will be set from run_experiment.py
    
    def load_prompts(self, prompts_path: str) -> List[Dict[str, str]]:
        """Load reward hacking prompts from JSON file."""
        with open(prompts_path, 'r') as f:
            return json.load(f)
    
    
    def process_prompt_offline(
        self,
        prompt_data: Dict[str, str],
        output_dir: str = "src/rollouts",
        skip_if_exists: bool = True,
        prompt_pbar: Optional[Any] = None,
        rollout_idx: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a single prompt using offline batched inference.
        
        This is the new batched approach that generates all counterfactuals
        for all sentences in one megabatch call to vLLM.
        
        Args:
            prompt_data: Prompt data dict
            output_dir: Directory to save output
            skip_if_exists: Whether to skip if output file exists
            prompt_pbar: Optional tqdm progress bar to update for sentence progress
        
        Returns:
            List of sentence data dictionaries
        """
        if self.llm is None or self.sampling_params is None:
            raise ValueError("LLM engine and sampling_params must be set before processing prompts")
        
        prompt_name = prompt_data.get("name", "unknown")
        if rollout_idx is not None:
            output_path = Path(output_dir) / f"{prompt_name}_rollout_{rollout_idx}_rollouts.json"
        else:
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
                        prompt_pbar.n = len(data)
                        prompt_pbar.refresh()
                        prompt_pbar.set_postfix({'status': 'loaded from file'})
                    return data
            except:
                pass  # If file is corrupted, regenerate
        
        # Step 1: Generate initial rollouts
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': f'generating {self.n_initial_rollout_tests} initial rollouts...'})
        elif not HAS_TQDM:
            print(f"\nProcessing prompt: {prompt_name}")
            print(f"  Step 1: Generating {self.n_initial_rollout_tests} initial rollouts...")
        
        unformatted_prompt = concatenate_prompt(
            prompt_data["task"],
            prompt_data["rules"],
            prompt_data["grader"]
        )
        messages = format_chat_messages(unformatted_prompt)
        # Get tokenizer from LLM for proper chat formatting (cache it)
        tokenizer = None
        try:
            tokenizer = getattr(self.llm, 'llm_engine', None)
            if tokenizer:
                tokenizer = getattr(tokenizer, 'tokenizer', None)
        except:
            pass
        prompt_string = messages_to_prompt_string(messages, tokenizer=tokenizer)
        
        # Generate N initial rollouts in one batch
        initial_prompts = [prompt_string] * self.n_initial_rollout_tests
        initial_outputs = self.llm.generate(initial_prompts, self.sampling_params)
        
        initial_rollouts = []
        for output in initial_outputs:
            if output.outputs and len(output.outputs) > 0:
                text = output.outputs[0].text
                if text:
                    initial_rollouts.append(text)
        
        if not initial_rollouts:
            if prompt_pbar:
                prompt_pbar.set_postfix({'status': 'failed to generate initial rollouts'})
            if not HAS_TQDM:
                print(f"  ✗ Failed to generate initial rollouts for {prompt_name}")
            return []
        
        # Step 2: Select best rollout based on strategy
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
            best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: x[0])
            strategy_display = "minimum"
        elif self.initial_rollout_selection_strategy == "MEDIAN":
            sentence_counts_sorted = sorted(sentence_counts)
            median_count = sentence_counts_sorted[len(sentence_counts_sorted) // 2]
            best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: abs(x[0] - median_count))
            strategy_display = f"median (target: {median_count})"
        else:
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
        
        # Step 3: Split into sentences
        sentences = split_sentences(best_rollout)
        
        if prompt_pbar:
            prompt_pbar.total = len(sentences)
            prompt_pbar.n = 0
            prompt_pbar.refresh()
            prompt_pbar.set_postfix({'status': f'preparing megabatch for {len(sentences)} sentences...'})
        elif not HAS_TQDM:
            print(f"  ✓ Found {len(sentences)} sentence(s)")
            print("  Step 3: Preparing megabatch for counterfactual generation...")
        
        # Step 4: Prepare the megabatch - ALL counterfactuals for ALL sentences
        all_prompts = []
        metadata_map = []  # Maps prompt index to sentence_idx
        
        # Get tokenizer once (cached for efficiency)
        tokenizer = None
        try:
            tokenizer = getattr(self.llm, 'llm_engine', None)
            if tokenizer:
                tokenizer = getattr(tokenizer, 'tokenizer', None)
        except:
            pass
        
        for sent_idx in range(len(sentences)):
            prefix = get_sentence_prefix(sentences, sent_idx)
            
            # Create messages with prefix for this sentence
            counterfactual_messages = format_chat_messages(unformatted_prompt)
            counterfactual_messages.append({"role": "assistant", "content": prefix})
            counterfactual_prompt = messages_to_prompt_string(counterfactual_messages, tokenizer=tokenizer)
            
            # Add num_counterfactuals copies of this prompt
            for _ in range(self.num_counterfactuals):
                all_prompts.append(counterfactual_prompt)
                metadata_map.append(sent_idx)
        
        # Step 5: The megabatch generation - ONE call to vLLM
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': f'generating {len(all_prompts)} rollouts in megabatch...'})
        elif not HAS_TQDM:
            print(f"  Step 4: Generating {len(all_prompts)} rollouts in megabatch...")
        
        outputs = self.llm.generate(all_prompts, self.sampling_params)
        
        # Step 6: Reassemble data - map outputs back to sentences
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': 'reassembling data...'})
        elif not HAS_TQDM:
            print("  Step 5: Reassembling data...")
        
        # Group rollouts by sentence index
        sentence_rollouts = defaultdict(list)
        for i, output in enumerate(outputs):
            sent_idx = metadata_map[i]
            if output.outputs and len(output.outputs) > 0:
                generated_text = output.outputs[0].text
                if generated_text:
                    sentence_rollouts[sent_idx].append(generated_text)
        
        # Step 7: Create sentence data structures
        sentence_data_list = []
        formatted_prompt = format_chat_messages(unformatted_prompt)
        
        for sent_idx in range(len(sentences)):
            rollouts = sentence_rollouts.get(sent_idx, [])
            
            # Detect reward hacking in each rollout
            reward_hacks = [detect_reward_hacking(r) for r in rollouts]
            p_reward_hacks = sum(reward_hacks) / len(rollouts) if rollouts else 0.0
            
            sentence_data = create_sentence_data(
                sentence_index=sent_idx,
                sentence=sentences[sent_idx],
                p_reward_hacks=p_reward_hacks,
                rollouts=rollouts,
                formatted_prompt=formatted_prompt,
                unformatted_prompt=unformatted_prompt
            )
            sentence_data_list.append(sentence_data)
            
            # Update progress bar
            if prompt_pbar:
                prompt_pbar.update(1)
                completed = len(sentence_data_list)
                if completed > 0:
                    avg_p_hacks = sum(s.get("p_reward_hacks", 0) for s in sentence_data_list) / completed
                    prompt_pbar.set_postfix({
                        'sentences': f'{completed}/{len(sentences)}',
                        'avg_p_hacks': f'{avg_p_hacks:.3f}'
                    })
        
        # Step 8: Save to file
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': 'saving...'})
        elif not HAS_TQDM:
            print(f"\n  Step 6: Saving to {output_path}...")
        
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sentence_data_list, f, indent=2)
        
        if prompt_pbar:
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
            self.process_prompt_offline(prompt_data, output_dir)


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

