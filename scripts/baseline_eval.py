#!/usr/bin/env python3
"""
Simple baseline evaluation for math problems.

Generates a single response per problem and checks correctness.
No counterfactual probing - just straightforward accuracy measurement.

Usage:
    python scripts/baseline_eval.py --model Qwen/Qwen3-4B --dataset examples/math/problems_1000.jsonl
    python scripts/baseline_eval.py --model Qwen/Qwen3-4B --dataset examples/math/problems_1000.jsonl --output results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_baseline_eval(
    model_name: str,
    dataset_path: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    gpu_memory_utilization: float = 0.9,
    batch_size: int = 32,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run baseline evaluation on math problems.

    Args:
        model_name: Model to evaluate (e.g., "Qwen/Qwen3-4B")
        dataset_path: Path to JSONL dataset
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        gpu_memory_utilization: GPU memory fraction
        batch_size: Batch size for generation
        limit: Optional limit on number of problems
        verbose: Print progress

    Returns:
        Dict with evaluation results
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from counterfactual_probing.dataset import Dataset
    from counterfactual_probing.scorer.examples.math import MathScorer

    results = {
        "model": model_name,
        "dataset": dataset_path,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "total": 0,
        "correct": 0,
        "accuracy": 0.0,
        "by_type": {},
        "by_level": {},
        "errors": [],
    }

    # Load dataset
    if verbose:
        print(f"Loading dataset: {dataset_path}")
    dataset = Dataset(path=dataset_path, prompt_field="prompt")
    items = list(dataset)
    if limit:
        items = items[:limit]
    results["total"] = len(items)

    if verbose:
        print(f"Loaded {len(items)} problems")
        print(f"Loading model: {model_name}")

    # Initialize model
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Initialize scorer
    scorer = MathScorer(config={"answer_field": "answer"})

    # Sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    # Process in batches
    if verbose:
        print(f"Generating responses (batch_size={batch_size})...")

    start_time = time.time()
    all_responses = []

    for batch_start in tqdm(range(0, len(items), batch_size), desc="Batches", disable=not verbose):
        batch_items = items[batch_start:batch_start + batch_size]

        # Format prompts
        prompts = []
        for item in batch_items:
            messages = [{"role": "user", "content": item["prompt"]}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        # Generate
        outputs = llm.generate(prompts, sampling_params)

        # Extract responses
        for output in outputs:
            text = output.outputs[0].text
            all_responses.append(text)

    generation_time = time.time() - start_time
    results["generation_time_seconds"] = generation_time

    if verbose:
        print(f"Generation complete in {generation_time:.1f}s")
        print("Scoring responses...")

    # Score all responses
    for i, (item, response) in enumerate(zip(items, all_responses)):
        metadata = item["metadata"]

        # Set scorer context
        scorer.set_context(item["prompt"], metadata)

        # Score
        score = scorer.score(response)
        is_correct = score == 1.0

        if is_correct:
            results["correct"] += 1

        # Track by type/level
        prob_type = metadata.get("type", "Unknown")
        prob_level = metadata.get("level", "Unknown")

        if prob_type not in results["by_type"]:
            results["by_type"][prob_type] = {"total": 0, "correct": 0}
        results["by_type"][prob_type]["total"] += 1
        if is_correct:
            results["by_type"][prob_type]["correct"] += 1

        if prob_level not in results["by_level"]:
            results["by_level"][prob_level] = {"total": 0, "correct": 0}
        results["by_level"][prob_level]["total"] += 1
        if is_correct:
            results["by_level"][prob_level]["correct"] += 1

    # Calculate accuracies
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0

    for key in results["by_type"]:
        t = results["by_type"][key]
        t["accuracy"] = t["correct"] / t["total"] if t["total"] > 0 else 0.0

    for key in results["by_level"]:
        t = results["by_level"][key]
        t["accuracy"] = t["correct"] / t["total"] if t["total"] > 0 else 0.0

    # Print summary
    if verbose:
        print()
        print("=" * 60)
        print("BASELINE EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_path}")
        print(f"Total problems: {results['total']}")
        print(f"Correct: {results['correct']}")
        print(f"Accuracy: {results['accuracy']:.1%}")
        print(f"Generation time: {generation_time:.1f}s ({generation_time/len(items):.2f}s/problem)")
        print()

        print("By Problem Type:")
        for prob_type, stats in sorted(results["by_type"].items()):
            print(f"  {prob_type}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
        print()

        print("By Difficulty Level:")
        for level, stats in sorted(results["by_level"].items()):
            print(f"  {level}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simple baseline evaluation for math problems"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate (e.g., Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="examples/math/problems_1000.jsonl",
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generation (default: 32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    results = run_baseline_eval(
        model_name=args.model,
        dataset_path=args.dataset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory,
        batch_size=args.batch_size,
        limit=args.limit,
        verbose=not args.quiet,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
