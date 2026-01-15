#!/usr/bin/env python3
"""
Compare sequential vs batched counterfactual generation on real math problems.

This script demonstrates the speedup from batching all branch points together.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from counterfactual_probing.dataset import Dataset
from counterfactual_probing.generator import (
    create_prefix_prompt,
    generate_all_branch_counterfactuals,
    generate_counterfactuals_batch,
    generate_initial_rollout,
)
from counterfactual_probing.sampler import TokenSampler
from counterfactual_probing.scorer.examples.math import MathScorer


def run_sequential(
    llm,
    tokenizer,
    items: list,
    num_branch_points: int,
    num_counterfactuals: int,
    max_tokens: int,
) -> dict:
    """Run the OLD sequential approach."""
    sampler = TokenSampler(method="uniform_count", num_samples=num_branch_points, seed=42)
    scorer = MathScorer(config={"answer_field": "answer"})

    total_time = 0
    total_generations = 0
    results = []

    for item in tqdm(items, desc="Sequential"):
        prompt = item["prompt"]
        metadata = item["metadata"]
        messages = [{"role": "user", "content": prompt}]

        # Initial rollout
        initial = generate_initial_rollout(
            messages=messages,
            model_name="",
            tokenizer=tokenizer,
            temperature=0.7,
            max_tokens=max_tokens,
            llm=llm,
        )
        token_ids = initial["token_ids"]

        # Sample branch points
        sampler.reset(seed=42)
        branch_points = sampler.sample(token_ids)

        # Sequential generation (OLD approach)
        start = time.perf_counter()
        samples = []
        for token_idx in branch_points:
            prefix_ids = token_ids[:token_idx + 1]
            counterfactuals = generate_counterfactuals_batch(
                prefix_ids=prefix_ids,
                messages=messages,
                model_name="",
                tokenizer=tokenizer,
                num_counterfactuals=num_counterfactuals,
                temperature=0.7,
                max_tokens=max_tokens,
                llm=llm,
            )
            samples.append({
                "token_index": token_idx,
                "counterfactuals": counterfactuals,
            })
            total_generations += len(counterfactuals)

        elapsed = time.perf_counter() - start
        total_time += elapsed

        # Score
        scorer.set_context(prompt, metadata)
        correct_count = 0
        for sample in samples:
            for cf in sample["counterfactuals"]:
                text = tokenizer.decode(cf["token_ids"], skip_special_tokens=True)
                if scorer.score(text) == 1.0:
                    correct_count += 1

        results.append({
            "prompt_id": metadata.get("id"),
            "seq_len": len(token_ids),
            "branch_points": len(branch_points),
            "time": elapsed,
            "correct": correct_count,
        })

    return {
        "approach": "sequential",
        "total_time": total_time,
        "total_generations": total_generations,
        "throughput": total_generations / total_time if total_time > 0 else 0,
        "results": results,
    }


def run_batched(
    llm,
    tokenizer,
    items: list,
    num_branch_points: int,
    num_counterfactuals: int,
    max_tokens: int,
) -> dict:
    """Run the NEW batched approach."""
    sampler = TokenSampler(method="uniform_count", num_samples=num_branch_points, seed=42)
    scorer = MathScorer(config={"answer_field": "answer"})

    total_time = 0
    total_generations = 0
    results = []

    for item in tqdm(items, desc="Batched"):
        prompt = item["prompt"]
        metadata = item["metadata"]
        messages = [{"role": "user", "content": prompt}]

        # Initial rollout
        initial = generate_initial_rollout(
            messages=messages,
            model_name="",
            tokenizer=tokenizer,
            temperature=0.7,
            max_tokens=max_tokens,
            llm=llm,
        )
        token_ids = initial["token_ids"]

        # Sample branch points
        sampler.reset(seed=42)
        branch_points = sampler.sample(token_ids)

        # Batched generation (NEW approach)
        start = time.perf_counter()
        samples = generate_all_branch_counterfactuals(
            initial_token_ids=token_ids,
            branch_points=branch_points,
            messages=messages,
            tokenizer=tokenizer,
            num_counterfactuals=num_counterfactuals,
            temperature=0.7,
            max_tokens=max_tokens,
            llm=llm,
        )
        elapsed = time.perf_counter() - start
        total_time += elapsed

        for sample in samples:
            total_generations += len(sample["counterfactuals"])

        # Score
        scorer.set_context(prompt, metadata)
        correct_count = 0
        for sample in samples:
            for cf in sample["counterfactuals"]:
                text = tokenizer.decode(cf["token_ids"], skip_special_tokens=True)
                if scorer.score(text) == 1.0:
                    correct_count += 1

        results.append({
            "prompt_id": metadata.get("id"),
            "seq_len": len(token_ids),
            "branch_points": len(branch_points),
            "time": elapsed,
            "correct": correct_count,
        })

    return {
        "approach": "batched",
        "total_time": total_time,
        "total_generations": total_generations,
        "throughput": total_generations / total_time if total_time > 0 else 0,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare sequential vs batched generation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default="examples/math/problems_1000.jsonl")
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of prompts to test")
    parser.add_argument("--num-branch-points", type=int, default=10)
    parser.add_argument("--num-counterfactuals", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory", type=float, default=0.9)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = Dataset(path=args.dataset, prompt_field="prompt")
    items = list(dataset)[:args.num_prompts]

    print(f"\n{'='*70}")
    print("COMPARING SEQUENTIAL VS BATCHED COUNTERFACTUAL GENERATION")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(items)}")
    print(f"Branch points per prompt: {args.num_branch_points}")
    print(f"Counterfactuals per branch: {args.num_counterfactuals}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Expected generations per prompt: {args.num_branch_points * args.num_counterfactuals}")
    print()

    # Warmup
    print("Warming up...")
    _ = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    print()

    # Run sequential
    print("=" * 70)
    print("RUNNING SEQUENTIAL APPROACH (OLD)")
    print("=" * 70)
    seq_results = run_sequential(
        llm, tokenizer, items,
        args.num_branch_points, args.num_counterfactuals, args.max_tokens
    )

    # Run batched
    print()
    print("=" * 70)
    print("RUNNING BATCHED APPROACH (NEW)")
    print("=" * 70)
    batch_results = run_batched(
        llm, tokenizer, items,
        args.num_branch_points, args.num_counterfactuals, args.max_tokens
    )

    # Summary
    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Sequential':<15} {'Batched':<15} {'Speedup':<10}")
    print("-" * 70)

    seq_time = seq_results["total_time"]
    batch_time = batch_results["total_time"]
    speedup = seq_time / batch_time if batch_time > 0 else 0

    print(f"{'Total generation time':<30} {seq_time:<15.2f} {batch_time:<15.2f} {speedup:<10.2f}x")
    print(f"{'Total generations':<30} {seq_results['total_generations']:<15} {batch_results['total_generations']:<15}")
    print(f"{'Throughput (gen/s)':<30} {seq_results['throughput']:<15.1f} {batch_results['throughput']:<15.1f} {batch_results['throughput']/seq_results['throughput'] if seq_results['throughput'] > 0 else 0:<10.2f}x")

    print()
    print("Per-prompt breakdown:")
    print(f"{'Prompt ID':<15} {'Seq Time (s)':<15} {'Batch Time (s)':<15} {'Speedup':<10}")
    print("-" * 55)

    for seq_r, batch_r in zip(seq_results["results"], batch_results["results"]):
        s_time = seq_r["time"]
        b_time = batch_r["time"]
        sp = s_time / b_time if b_time > 0 else 0
        print(f"{seq_r['prompt_id']:<15} {s_time:<15.2f} {b_time:<15.2f} {sp:<10.2f}x")

    print()
    print("=" * 70)
    print(f"CONCLUSION: Batched approach is {speedup:.1f}x faster!")
    print("=" * 70)


if __name__ == "__main__":
    main()
