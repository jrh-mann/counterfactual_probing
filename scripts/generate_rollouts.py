#!/usr/bin/env python3
"""
Generate rollouts for MATH problems and select shortest for counterfactual generation.

Pipeline:
1. Load problems from JSONL
2. Generate N rollouts per problem using vLLM (n=N in SamplingParams)
3. For each problem, pick shortest valid rollout
4. Sort all problems by their shortest rollout length
5. Select top K shortest
6. Save selected problems for counterfactual generation
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from tqdm import tqdm


@dataclass
class RolloutResult:
    """Result from rollout generation for a single problem."""
    prompt_id: str
    prompt: str
    metadata: dict
    # Best (shortest valid) rollout
    token_ids: list[int]
    text: str
    length: int
    # Validity info
    truncated: bool
    valid: bool
    # Stats from all N rollouts
    all_lengths: list[int]
    num_valid: int
    timestamp: str


def load_problems(path: str) -> list[dict]:
    """Load problems from JSONL file."""
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def generate_rollouts_batch(
    prompts: list[str],
    llm,
    tokenizer,
    n_per_prompt: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 6000,
) -> list[list[dict]]:
    """
    Generate N rollouts per prompt using vLLM.

    Returns:
        List of lists - for each prompt, a list of N rollouts
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_per_prompt,  # Generate N rollouts per prompt
    )

    print(f"Generating {len(prompts)} prompts × {n_per_prompt} = {len(prompts) * n_per_prompt} total rollouts...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        prompt_rollouts = []
        for completion in output.outputs:
            token_ids = list(completion.token_ids)
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            prompt_rollouts.append({
                "token_ids": token_ids,
                "text": text,
                "length": len(token_ids),
            })
        results.append(prompt_rollouts)

    return results


def select_shortest_valid(
    rollouts: list[dict],
    max_tokens: int,
    boundary_token: Optional[int] = None,
) -> tuple[dict, dict]:
    """
    Select the shortest valid rollout from a list.

    Returns:
        (best_rollout, stats_dict)
    """
    valid_rollouts = []
    all_lengths = []

    for r in rollouts:
        all_lengths.append(r["length"])

        # Check validity
        truncated = r["length"] >= max_tokens

        has_boundary = True
        if boundary_token is not None:
            has_boundary = boundary_token in r["token_ids"]

        is_valid = not truncated and has_boundary

        if is_valid:
            valid_rollouts.append(r)

    stats = {
        "all_lengths": all_lengths,
        "num_valid": len(valid_rollouts),
    }

    if valid_rollouts:
        # Pick shortest valid
        best = min(valid_rollouts, key=lambda r: r["length"])
        return best, stats
    else:
        # No valid rollouts - return shortest anyway (will be marked invalid)
        best = min(rollouts, key=lambda r: r["length"])
        return best, stats


def main():
    parser = argparse.ArgumentParser(description="Generate rollouts and select shortest")
    parser.add_argument("--problems", type=str, default="examples/math/problems_2000.jsonl",
                        help="Input problems JSONL")
    parser.add_argument("--output", type=str, default="outputs/rollouts_2000.jsonl",
                        help="Output rollouts JSONL")
    parser.add_argument("--selected", type=str, default="outputs/selected_1000.jsonl",
                        help="Output selected problems JSONL (shortest K)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="Model name")
    parser.add_argument("--n-rollouts", type=int, default=3,
                        help="Number of rollouts per problem")
    parser.add_argument("--select-top", type=int, default=1000,
                        help="Select top K shortest valid rollouts")
    parser.add_argument("--max-tokens", type=int, default=6000,
                        help="Max tokens per rollout")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--boundary-token", type=int, default=None,
                        help="CoT boundary token ID (e.g., 151668 for Qwen3 </think>)")
    parser.add_argument("--gpu-memory", type=float, default=0.9,
                        help="GPU memory utilization")
    args = parser.parse_args()

    # Load problems
    print(f"Loading problems from {args.problems}...")
    problems = load_problems(args.problems)
    print(f"  Loaded {len(problems)} problems")

    # Initialize model
    print(f"\nLoading model {args.model}...")
    from transformers import AutoTokenizer
    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("  Model loaded!")

    # Format prompts
    print("\nFormatting prompts...")
    formatted_prompts = []
    for problem in tqdm(problems, desc="Formatting"):
        messages = [{"role": "user", "content": problem["prompt"]}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted)

    # Generate rollouts
    print(f"\nGenerating {len(formatted_prompts)} × {args.n_rollouts} = {len(formatted_prompts) * args.n_rollouts} rollouts...")
    all_rollouts = generate_rollouts_batch(
        prompts=formatted_prompts,
        llm=llm,
        tokenizer=tokenizer,
        n_per_prompt=args.n_rollouts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print("  Generation complete!")

    # Process each problem - select shortest valid rollout
    print("\nSelecting shortest valid rollout per problem...")
    results = []
    for problem, rollouts in tqdm(zip(problems, all_rollouts), total=len(problems), desc="Processing"):
        best, stats = select_shortest_valid(
            rollouts,
            args.max_tokens,
            args.boundary_token,
        )

        truncated = best["length"] >= args.max_tokens
        valid = stats["num_valid"] > 0

        result = RolloutResult(
            prompt_id=problem["id"],
            prompt=problem["prompt"],
            metadata={
                "id": problem["id"],
                "answer": problem["answer"],
                "level": problem["level"],
                "type": problem["type"],
            },
            token_ids=best["token_ids"],
            text=best["text"],
            length=best["length"],
            truncated=truncated,
            valid=valid,
            all_lengths=stats["all_lengths"],
            num_valid=stats["num_valid"],
            timestamp=datetime.now().isoformat(),
        )
        results.append(result)

    # Save all rollouts
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving all {len(results)} rollouts to {output_path}...")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    # Stats
    valid_results = [r for r in results if r.valid]
    lengths = [r.length for r in results]
    valid_lengths = [r.length for r in valid_results]

    print(f"\nStats:")
    print(f"  Total problems: {len(results)}")
    print(f"  Valid rollouts: {len(valid_results)} ({100*len(valid_results)/len(results):.1f}%)")
    print(f"  All lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")
    if valid_lengths:
        print(f"  Valid lengths: min={min(valid_lengths)}, max={max(valid_lengths)}, mean={sum(valid_lengths)/len(valid_lengths):.0f}")

    # Sort by length and select top K
    print(f"\nSelecting top {args.select_top} shortest valid rollouts...")
    valid_results.sort(key=lambda r: r.length)
    selected = valid_results[:args.select_top]

    if len(selected) < args.select_top:
        print(f"  Warning: Only {len(selected)} valid rollouts available (wanted {args.select_top})")

    # Save selected
    selected_path = Path(args.selected)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(selected)} selected rollouts to {selected_path}...")
    with open(selected_path, "w") as f:
        for r in selected:
            f.write(json.dumps(asdict(r)) + "\n")

    # Selected stats
    selected_lengths = [r.length for r in selected]
    print(f"\nSelected stats:")
    print(f"  Count: {len(selected)}")
    print(f"  Length range: {min(selected_lengths)} - {max(selected_lengths)}")
    print(f"  Mean length: {sum(selected_lengths)/len(selected_lengths):.0f}")

    # Level distribution in selected
    from collections import Counter
    levels = Counter(r.metadata["level"] for r in selected)
    print(f"\nLevel distribution in selected:")
    for level in sorted(levels.keys()):
        print(f"  {level}: {levels[level]} ({100*levels[level]/len(selected):.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
