#!/usr/bin/env python3
"""
Profile counterfactual generation to identify bottlenecks.

Compares different approaches:
1. Current: Sequential branch points, batch counterfactuals per branch
2. Optimized: Batch ALL branch points together
3. Using n= parameter vs duplicated prompts
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def profile_current_approach(
    llm, tokenizer, prompt: str, num_branch_points: int, num_counterfactuals: int, max_tokens: int
) -> dict:
    """Profile the current sequential branch point approach."""
    from counterfactual_probing.generator import (
        create_prefix_prompt,
        generate_initial_rollout,
    )

    messages = [{"role": "user", "content": prompt}]
    timings = {"initial_rollout": 0, "branch_points": [], "total_generations": 0}

    # Initial rollout
    start = time.perf_counter()
    initial = generate_initial_rollout(
        messages=messages,
        model_name="",  # Not used when llm provided
        tokenizer=tokenizer,
        temperature=0.7,
        max_tokens=max_tokens,
        llm=llm,
    )
    timings["initial_rollout"] = time.perf_counter() - start

    token_ids = initial["token_ids"]
    seq_len = len(token_ids)

    # Sample branch points uniformly
    if num_branch_points >= seq_len:
        branch_points = list(range(seq_len))
    else:
        branch_points = [0] + sorted(np.random.choice(
            range(1, seq_len - 1),
            min(num_branch_points - 2, seq_len - 2),
            replace=False
        ).tolist()) + [seq_len - 1]

    timings["num_branch_points"] = len(branch_points)
    timings["seq_len"] = seq_len

    # Current approach: sequential branch points
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens, n=1)

    for bp_idx, token_idx in enumerate(branch_points):
        prefix_ids = token_ids[:token_idx + 1]
        prefix_prompt = create_prefix_prompt(prefix_ids, tokenizer, messages)
        prompts = [prefix_prompt] * num_counterfactuals

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start

        timings["branch_points"].append({
            "idx": token_idx,
            "prefix_len": len(prefix_ids),
            "time": elapsed,
            "num_outputs": len(outputs),
        })
        timings["total_generations"] += len(outputs)

    timings["total_branch_time"] = sum(bp["time"] for bp in timings["branch_points"])
    return timings


def profile_batched_approach(
    llm, tokenizer, prompt: str, num_branch_points: int, num_counterfactuals: int, max_tokens: int
) -> dict:
    """Profile batching ALL branch points together."""
    from counterfactual_probing.generator import (
        create_prefix_prompt,
        generate_initial_rollout,
    )

    messages = [{"role": "user", "content": prompt}]
    timings = {"initial_rollout": 0, "total_generations": 0}

    # Initial rollout
    start = time.perf_counter()
    initial = generate_initial_rollout(
        messages=messages,
        model_name="",
        tokenizer=tokenizer,
        temperature=0.7,
        max_tokens=max_tokens,
        llm=llm,
    )
    timings["initial_rollout"] = time.perf_counter() - start

    token_ids = initial["token_ids"]
    seq_len = len(token_ids)

    # Sample branch points
    if num_branch_points >= seq_len:
        branch_points = list(range(seq_len))
    else:
        branch_points = [0] + sorted(np.random.choice(
            range(1, seq_len - 1),
            min(num_branch_points - 2, seq_len - 2),
            replace=False
        ).tolist()) + [seq_len - 1]

    timings["num_branch_points"] = len(branch_points)
    timings["seq_len"] = seq_len

    # Batched approach: create ALL prompts at once
    all_prompts = []
    prompt_metadata = []  # Track which branch point each prompt belongs to

    start_prep = time.perf_counter()
    for token_idx in branch_points:
        prefix_ids = token_ids[:token_idx + 1]
        prefix_prompt = create_prefix_prompt(prefix_ids, tokenizer, messages)
        for _ in range(num_counterfactuals):
            all_prompts.append(prefix_prompt)
            prompt_metadata.append(token_idx)
    timings["prep_time"] = time.perf_counter() - start_prep

    timings["total_prompts"] = len(all_prompts)

    # Single batched generation
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens, n=1)

    start = time.perf_counter()
    outputs = llm.generate(all_prompts, sampling_params)
    timings["batch_generation_time"] = time.perf_counter() - start
    timings["total_generations"] = len(outputs)

    return timings


def profile_n_parameter(
    llm, tokenizer, prompt: str, num_branch_points: int, num_counterfactuals: int, max_tokens: int
) -> dict:
    """Profile using n= parameter instead of duplicated prompts."""
    from counterfactual_probing.generator import (
        create_prefix_prompt,
        generate_initial_rollout,
    )

    messages = [{"role": "user", "content": prompt}]
    timings = {"initial_rollout": 0, "branch_points": [], "total_generations": 0}

    # Initial rollout
    start = time.perf_counter()
    initial = generate_initial_rollout(
        messages=messages,
        model_name="",
        tokenizer=tokenizer,
        temperature=0.7,
        max_tokens=max_tokens,
        llm=llm,
    )
    timings["initial_rollout"] = time.perf_counter() - start

    token_ids = initial["token_ids"]
    seq_len = len(token_ids)

    # Sample branch points
    if num_branch_points >= seq_len:
        branch_points = list(range(seq_len))
    else:
        branch_points = [0] + sorted(np.random.choice(
            range(1, seq_len - 1),
            min(num_branch_points - 2, seq_len - 2),
            replace=False
        ).tolist()) + [seq_len - 1]

    timings["num_branch_points"] = len(branch_points)
    timings["seq_len"] = seq_len

    # Using n= parameter
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens, n=num_counterfactuals)

    for bp_idx, token_idx in enumerate(branch_points):
        prefix_ids = token_ids[:token_idx + 1]
        prefix_prompt = create_prefix_prompt(prefix_ids, tokenizer, messages)

        start = time.perf_counter()
        outputs = llm.generate([prefix_prompt], sampling_params)
        elapsed = time.perf_counter() - start

        timings["branch_points"].append({
            "idx": token_idx,
            "prefix_len": len(prefix_ids),
            "time": elapsed,
            "num_outputs": len(outputs[0].outputs),
        })
        timings["total_generations"] += len(outputs[0].outputs)

    timings["total_branch_time"] = sum(bp["time"] for bp in timings["branch_points"])
    return timings


def profile_batched_with_n(
    llm, tokenizer, prompt: str, num_branch_points: int, num_counterfactuals: int, max_tokens: int
) -> dict:
    """Profile batching branch points WITH n= parameter (best of both)."""
    from counterfactual_probing.generator import (
        create_prefix_prompt,
        generate_initial_rollout,
    )

    messages = [{"role": "user", "content": prompt}]
    timings = {"initial_rollout": 0, "total_generations": 0}

    # Initial rollout
    start = time.perf_counter()
    initial = generate_initial_rollout(
        messages=messages,
        model_name="",
        tokenizer=tokenizer,
        temperature=0.7,
        max_tokens=max_tokens,
        llm=llm,
    )
    timings["initial_rollout"] = time.perf_counter() - start

    token_ids = initial["token_ids"]
    seq_len = len(token_ids)

    # Sample branch points
    if num_branch_points >= seq_len:
        branch_points = list(range(seq_len))
    else:
        branch_points = [0] + sorted(np.random.choice(
            range(1, seq_len - 1),
            min(num_branch_points - 2, seq_len - 2),
            replace=False
        ).tolist()) + [seq_len - 1]

    timings["num_branch_points"] = len(branch_points)
    timings["seq_len"] = seq_len

    # Create one prompt per branch point
    all_prompts = []
    start_prep = time.perf_counter()
    for token_idx in branch_points:
        prefix_ids = token_ids[:token_idx + 1]
        prefix_prompt = create_prefix_prompt(prefix_ids, tokenizer, messages)
        all_prompts.append(prefix_prompt)
    timings["prep_time"] = time.perf_counter() - start_prep

    # Single batched generation with n= parameter
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens, n=num_counterfactuals)

    start = time.perf_counter()
    outputs = llm.generate(all_prompts, sampling_params)
    timings["batch_generation_time"] = time.perf_counter() - start
    timings["total_generations"] = sum(len(o.outputs) for o in outputs)

    return timings


def main():
    parser = argparse.ArgumentParser(description="Profile counterfactual generation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
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

    # Test prompt
    prompt = "Solve the following math problem. Show your reasoning step by step, then provide your final answer in \\boxed{}.\n\nProblem: What is 15 + 27?"

    print(f"\n{'='*70}")
    print("PROFILING COUNTERFACTUAL GENERATION")
    print(f"{'='*70}")
    print(f"Branch points: {args.num_branch_points}")
    print(f"Counterfactuals per branch: {args.num_counterfactuals}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Total generations per approach: {args.num_branch_points * args.num_counterfactuals}")
    print()

    # Warmup
    print("Warming up...")
    _ = llm.generate(["Hello"], SamplingParams(max_tokens=10))

    results = {}

    # Profile each approach
    print("\n1. CURRENT APPROACH (sequential branch points, batched counterfactuals)")
    print("-" * 50)
    np.random.seed(42)
    results["current"] = profile_current_approach(
        llm, tokenizer, prompt, args.num_branch_points, args.num_counterfactuals, args.max_tokens
    )
    print(f"   Initial rollout: {results['current']['initial_rollout']:.2f}s")
    print(f"   Sequence length: {results['current']['seq_len']} tokens")
    print(f"   Branch points processed: {results['current']['num_branch_points']}")
    print(f"   Total branch time: {results['current']['total_branch_time']:.2f}s")
    print(f"   Avg per branch point: {results['current']['total_branch_time']/results['current']['num_branch_points']:.2f}s")
    print(f"   Total generations: {results['current']['total_generations']}")

    print("\n2. BATCHED APPROACH (all branch points in one call, duplicated prompts)")
    print("-" * 50)
    np.random.seed(42)
    results["batched"] = profile_batched_approach(
        llm, tokenizer, prompt, args.num_branch_points, args.num_counterfactuals, args.max_tokens
    )
    print(f"   Initial rollout: {results['batched']['initial_rollout']:.2f}s")
    print(f"   Prep time: {results['batched']['prep_time']:.4f}s")
    print(f"   Total prompts: {results['batched']['total_prompts']}")
    print(f"   Batch generation time: {results['batched']['batch_generation_time']:.2f}s")
    print(f"   Total generations: {results['batched']['total_generations']}")

    print("\n3. N-PARAMETER APPROACH (sequential branch points, n= for counterfactuals)")
    print("-" * 50)
    np.random.seed(42)
    results["n_param"] = profile_n_parameter(
        llm, tokenizer, prompt, args.num_branch_points, args.num_counterfactuals, args.max_tokens
    )
    print(f"   Initial rollout: {results['n_param']['initial_rollout']:.2f}s")
    print(f"   Total branch time: {results['n_param']['total_branch_time']:.2f}s")
    print(f"   Avg per branch point: {results['n_param']['total_branch_time']/results['n_param']['num_branch_points']:.2f}s")
    print(f"   Total generations: {results['n_param']['total_generations']}")

    print("\n4. BATCHED + N-PARAMETER (all branch points batched, n= for counterfactuals)")
    print("-" * 50)
    np.random.seed(42)
    results["batched_n"] = profile_batched_with_n(
        llm, tokenizer, prompt, args.num_branch_points, args.num_counterfactuals, args.max_tokens
    )
    print(f"   Initial rollout: {results['batched_n']['initial_rollout']:.2f}s")
    print(f"   Prep time: {results['batched_n']['prep_time']:.4f}s")
    print(f"   Batch generation time: {results['batched_n']['batch_generation_time']:.2f}s")
    print(f"   Total generations: {results['batched_n']['total_generations']}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (generation time only, excluding initial rollout)")
    print(f"{'='*70}")

    current_time = results["current"]["total_branch_time"]
    batched_time = results["batched"]["batch_generation_time"]
    n_param_time = results["n_param"]["total_branch_time"]
    batched_n_time = results["batched_n"]["batch_generation_time"]

    print(f"   Current (sequential + dup prompts):  {current_time:6.2f}s  (baseline)")
    print(f"   Batched (all prompts at once):       {batched_time:6.2f}s  ({current_time/batched_time:.2f}x speedup)")
    print(f"   N-param (sequential + n=):           {n_param_time:6.2f}s  ({current_time/n_param_time:.2f}x speedup)")
    print(f"   Batched + N-param (best):            {batched_n_time:6.2f}s  ({current_time/batched_n_time:.2f}x speedup)")

    print(f"\n{'='*70}")
    print("THROUGHPUT (generations per second)")
    print(f"{'='*70}")
    total_gens = args.num_branch_points * args.num_counterfactuals
    print(f"   Current:        {total_gens/current_time:6.1f} gen/s")
    print(f"   Batched:        {total_gens/batched_time:6.1f} gen/s")
    print(f"   N-param:        {total_gens/n_param_time:6.1f} gen/s")
    print(f"   Batched+N:      {total_gens/batched_n_time:6.1f} gen/s")


if __name__ == "__main__":
    main()
