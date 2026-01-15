#!/usr/bin/env python3
"""
Download MATH dataset from HuggingFace and format for probing pipeline.

Downloads from qwedsacf/competition_math, samples problems stratified by difficulty,
and formats them into our standard JSONL format.
"""

import argparse
import json
import random
import re
from pathlib import Path
from collections import defaultdict, Counter


def load_math_dataset():
    """Load MATH dataset from HuggingFace."""
    from datasets import load_dataset

    print("Loading MATH dataset from HuggingFace...")
    ds = load_dataset('qwedsacf/competition_math', split='train')
    print(f"  Loaded {len(ds)} problems")
    return ds


def format_prompt(problem: str) -> str:
    """Format a MATH problem into our standard prompt format."""
    return (
        "Solve the following math problem. Show your reasoning step by step, "
        f"then provide your final answer in \\boxed{{}}.\n\n"
        f"Problem: {problem}"
    )


def extract_answer(solution: str) -> str:
    """Extract the boxed answer from a MATH solution."""
    # Handle nested braces - find last \boxed{...}
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
    if matches:
        return matches[-1]

    # Fallback: simpler pattern
    match = re.search(r'\\boxed\{([^}]+)\}', solution)
    if match:
        return match.group(1)

    return ""


def sample_problems(
    dataset,
    n: int = 2000,
    prioritize_hard: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    Sample n problems, optionally prioritizing harder levels.

    Args:
        dataset: HuggingFace dataset
        n: Number to sample
        prioritize_hard: If True, oversample Level 3-5 problems
        seed: Random seed
    """
    random.seed(seed)

    # Convert to list of dicts with index
    problems = []
    for i, item in enumerate(dataset):
        problems.append({
            "idx": i,
            "problem": item["problem"],
            "solution": item["solution"],
            "level": item["level"],
            "type": item["type"],
        })

    # Group by level
    by_level = defaultdict(list)
    for p in problems:
        by_level[p["level"]].append(p)

    print(f"\nLevel distribution in dataset:")
    for level in sorted(by_level.keys()):
        print(f"  {level}: {len(by_level[level])}")

    if prioritize_hard:
        # Target distribution: more hard problems
        # Level 1: 10%, Level 2: 15%, Level 3: 20%, Level 4: 25%, Level 5: 30%
        target_pcts = {
            "Level 1": 0.10,
            "Level 2": 0.15,
            "Level 3": 0.20,
            "Level 4": 0.25,
            "Level 5": 0.30,
        }

        sampled = []
        for level, pct in target_pcts.items():
            target_n = int(n * pct)
            available = by_level.get(level, [])

            if len(available) >= target_n:
                sampled.extend(random.sample(available, target_n))
            else:
                # Take all available
                sampled.extend(available)
                print(f"  Warning: Only {len(available)} problems for {level}, wanted {target_n}")

        # Fill remainder randomly from all levels
        remaining = n - len(sampled)
        if remaining > 0:
            sampled_idxs = {p["idx"] for p in sampled}
            available = [p for p in problems if p["idx"] not in sampled_idxs]
            sampled.extend(random.sample(available, min(remaining, len(available))))
    else:
        # Uniform random sampling
        sampled = random.sample(problems, min(n, len(problems)))

    # Shuffle final list
    random.shuffle(sampled)

    return sampled


def format_for_pipeline(problems: list[dict]) -> list[dict]:
    """Format problems for our probing pipeline."""
    formatted = []

    for i, p in enumerate(problems):
        formatted.append({
            "id": f"math_{i:04d}",
            "prompt": format_prompt(p["problem"]),
            "answer": extract_answer(p["solution"]),
            "level": p["level"],
            "type": p["type"],
        })

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Create MATH dataset for probing")
    parser.add_argument("--output", type=str, default="examples/math/problems_2000.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of problems to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--uniform", action="store_true",
                        help="Use uniform sampling instead of prioritizing hard problems")
    args = parser.parse_args()

    # Load dataset
    dataset = load_math_dataset()

    # Sample
    sampled = sample_problems(
        dataset,
        n=args.n,
        prioritize_hard=not args.uniform,
        seed=args.seed,
    )

    print(f"\nSampled {len(sampled)} problems")

    # Check level distribution in sample
    levels = Counter(p["level"] for p in sampled)
    print("Sampled level distribution:")
    for level in sorted(levels.keys()):
        print(f"  {level}: {levels[level]} ({100*levels[level]/len(sampled):.1f}%)")

    # Format for pipeline
    formatted = format_for_pipeline(sampled)

    # Check for empty answers
    empty_answers = sum(1 for p in formatted if not p["answer"])
    if empty_answers:
        print(f"\nWarning: {empty_answers} problems have empty answers")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for p in formatted:
            f.write(json.dumps(p) + "\n")

    print(f"\nSaved {len(formatted)} problems to {output_path}")

    # Show sample
    print("\nSample problem:")
    sample = formatted[0]
    print(f"  ID: {sample['id']}")
    print(f"  Level: {sample['level']}")
    print(f"  Type: {sample['type']}")
    print(f"  Answer: {sample['answer']}")
    print(f"  Prompt (truncated): {sample['prompt'][:200]}...")


if __name__ == "__main__":
    main()
