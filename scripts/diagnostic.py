#!/usr/bin/env python3
"""
20-minute diagnostic to investigate bottlenecks in counterfactual generation.

Tests:
1. Throughput vs max_tokens (sequence length impact)
2. Throughput vs batch size
3. Initial rollout vs counterfactual generation breakdown
4. Prefix caching effectiveness
5. GPU utilization patterns
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from counterfactual_probing.dataset import Dataset
from counterfactual_probing.generator import (
    create_prefix_prompt,
    generate_all_branch_counterfactuals,
    generate_initial_rollout,
)
from counterfactual_probing.sampler import TokenSampler


def get_gpu_stats():
    """Get GPU utilization and memory stats."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        parts = result.stdout.strip().split(", ")
        return {
            "gpu_util": int(parts[0]),
            "mem_util": int(parts[1]),
            "mem_used_gb": int(parts[2]) / 1024,
            "mem_total_gb": int(parts[3]) / 1024,
        }
    except:
        return None


def monitor_gpu(interval=1.0, results=None, stop_event=None):
    """Background thread to monitor GPU."""
    if results is None:
        results = []
    while not stop_event.is_set():
        stats = get_gpu_stats()
        if stats:
            results.append(stats)
        time.sleep(interval)
    return results


def test_throughput_vs_max_tokens(llm, tokenizer, prompts, num_samples=5):
    """Test how max_tokens affects throughput."""
    print("\n" + "="*70)
    print("TEST 1: Throughput vs max_tokens")
    print("="*70)

    # Use chat template
    messages_list = [[{"role": "user", "content": p}] for p in prompts[:num_samples]]
    formatted_prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                         for m in messages_list]

    results = []
    for max_tokens in [256, 512, 1024, 2048, 4096]:
        # Warmup
        _ = llm.generate(formatted_prompts[:1], SamplingParams(max_tokens=10, temperature=0.7))

        start = time.perf_counter()
        outputs = llm.generate(formatted_prompts, SamplingParams(max_tokens=max_tokens, temperature=0.7))
        elapsed = time.perf_counter() - start

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / elapsed
        avg_len = total_tokens / len(outputs)

        results.append({
            "max_tokens": max_tokens,
            "avg_output_len": avg_len,
            "throughput_tok_s": throughput,
            "time_s": elapsed,
        })
        print(f"max_tokens={max_tokens:4d}: avg_len={avg_len:6.0f}, throughput={throughput:6.0f} tok/s, time={elapsed:.1f}s")

    return results


def test_throughput_vs_batch_size(llm, tokenizer, prompts):
    """Test how batch size affects throughput."""
    print("\n" + "="*70)
    print("TEST 2: Throughput vs batch size")
    print("="*70)

    # Use a single prompt, vary batch size
    messages = [{"role": "user", "content": prompts[0]}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    results = []
    for batch_size in [1, 10, 50, 100, 200, 400]:
        batch = [formatted] * batch_size

        start = time.perf_counter()
        outputs = llm.generate(batch, SamplingParams(max_tokens=512, temperature=0.7))
        elapsed = time.perf_counter() - start

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / elapsed
        per_request = elapsed / batch_size

        results.append({
            "batch_size": batch_size,
            "throughput_tok_s": throughput,
            "time_per_request_s": per_request,
            "total_time_s": elapsed,
        })
        print(f"batch={batch_size:3d}: throughput={throughput:6.0f} tok/s, per_request={per_request:.2f}s, total={elapsed:.1f}s")

    return results


def test_initial_vs_counterfactual(llm, tokenizer, prompts, num_prompts=3):
    """Break down time between initial rollout and counterfactual generation."""
    print("\n" + "="*70)
    print("TEST 3: Initial rollout vs counterfactual generation breakdown")
    print("="*70)

    sampler = TokenSampler(method="uniform_count", num_samples=20, seed=42)
    results = []

    for i, prompt in enumerate(prompts[:num_prompts]):
        messages = [{"role": "user", "content": prompt}]

        # Time initial rollout
        start = time.perf_counter()
        initial = generate_initial_rollout(
            messages=messages,
            model_name="",
            tokenizer=tokenizer,
            temperature=0.7,
            max_tokens=4096,
            llm=llm,
        )
        initial_time = time.perf_counter() - start
        initial_tokens = len(initial["token_ids"])

        # Sample branch points
        sampler.reset(seed=42)
        branch_points = sampler.sample(initial["token_ids"])

        # Time counterfactual generation
        start = time.perf_counter()
        samples = generate_all_branch_counterfactuals(
            initial_token_ids=initial["token_ids"],
            branch_points=branch_points,
            messages=messages,
            tokenizer=tokenizer,
            num_counterfactuals=20,
            temperature=0.7,
            max_tokens=4096,
            llm=llm,
        )
        cf_time = time.perf_counter() - start

        cf_tokens = sum(len(cf["token_ids"]) for s in samples for cf in s["counterfactuals"])
        num_cf = sum(len(s["counterfactuals"]) for s in samples)

        results.append({
            "prompt_idx": i,
            "initial_tokens": initial_tokens,
            "initial_time_s": initial_time,
            "num_counterfactuals": num_cf,
            "cf_tokens": cf_tokens,
            "cf_time_s": cf_time,
            "total_time_s": initial_time + cf_time,
            "initial_pct": 100 * initial_time / (initial_time + cf_time),
        })

        print(f"Prompt {i}: initial={initial_time:.1f}s ({initial_tokens} tok), "
              f"cf={cf_time:.1f}s ({num_cf}x, {cf_tokens} tok), "
              f"initial={100*initial_time/(initial_time+cf_time):.0f}%")

    return results


def test_prefix_caching(llm, tokenizer, prompts):
    """Test if prefix caching is working by comparing repeated vs unique prefixes."""
    print("\n" + "="*70)
    print("TEST 4: Prefix caching effectiveness")
    print("="*70)

    messages = [{"role": "user", "content": prompts[0]}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Test 1: Same prefix repeated (should benefit from caching)
    same_prefix_batch = [formatted] * 100

    start = time.perf_counter()
    outputs1 = llm.generate(same_prefix_batch, SamplingParams(max_tokens=256, temperature=0.7))
    same_prefix_time = time.perf_counter() - start
    same_prefix_tokens = sum(len(o.outputs[0].token_ids) for o in outputs1)

    # Test 2: Different prefixes (less caching benefit)
    diff_messages = [[{"role": "user", "content": p}] for p in prompts[:100]]
    diff_prefix_batch = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                         for m in diff_messages]

    start = time.perf_counter()
    outputs2 = llm.generate(diff_prefix_batch, SamplingParams(max_tokens=256, temperature=0.7))
    diff_prefix_time = time.perf_counter() - start
    diff_prefix_tokens = sum(len(o.outputs[0].token_ids) for o in outputs2)

    same_throughput = same_prefix_tokens / same_prefix_time
    diff_throughput = diff_prefix_tokens / diff_prefix_time

    print(f"Same prefix (100x):  {same_throughput:.0f} tok/s, {same_prefix_time:.1f}s")
    print(f"Diff prefixes (100): {diff_throughput:.0f} tok/s, {diff_prefix_time:.1f}s")
    print(f"Caching benefit: {same_throughput/diff_throughput:.2f}x")

    return {
        "same_prefix_throughput": same_throughput,
        "diff_prefix_throughput": diff_throughput,
        "caching_ratio": same_throughput / diff_throughput,
    }


def test_gpu_utilization_pattern(llm, tokenizer, prompts):
    """Monitor GPU utilization during generation."""
    print("\n" + "="*70)
    print("TEST 5: GPU utilization pattern during generation")
    print("="*70)

    messages = [{"role": "user", "content": prompts[0]}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = [formatted] * 400

    # Start GPU monitoring
    gpu_stats = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(0.5, gpu_stats, stop_event))
    monitor_thread.start()

    # Run generation
    start = time.perf_counter()
    outputs = llm.generate(batch, SamplingParams(max_tokens=1024, temperature=0.7))
    elapsed = time.perf_counter() - start

    # Stop monitoring
    stop_event.set()
    monitor_thread.join()

    if gpu_stats:
        avg_gpu_util = sum(s["gpu_util"] for s in gpu_stats) / len(gpu_stats)
        avg_mem_util = sum(s["mem_util"] for s in gpu_stats) / len(gpu_stats)
        max_mem_used = max(s["mem_used_gb"] for s in gpu_stats)

        print(f"Duration: {elapsed:.1f}s")
        print(f"Avg GPU utilization: {avg_gpu_util:.0f}%")
        print(f"Avg memory bandwidth utilization: {avg_mem_util:.0f}%")
        print(f"Peak memory used: {max_mem_used:.1f} GB")

        return {
            "duration_s": elapsed,
            "avg_gpu_util": avg_gpu_util,
            "avg_mem_util": avg_mem_util,
            "peak_mem_gb": max_mem_used,
            "samples": len(gpu_stats),
        }
    return None


def main():
    print("="*70)
    print("COUNTERFACTUAL PROBING - BOTTLENECK DIAGNOSTIC")
    print("="*70)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Expected duration: ~20 minutes")
    print()

    # Load model
    print("Loading model...")
    model_name = "Qwen/Qwen3-4B"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load dataset
    print("Loading dataset...")
    dataset = Dataset(path="examples/math/problems_1000.jsonl", prompt_field="prompt")
    prompts = [item["prompt"] for item in list(dataset)[:100]]

    print(f"Loaded {len(prompts)} prompts")
    print()

    # Run tests
    results = {}

    results["max_tokens"] = test_throughput_vs_max_tokens(llm, tokenizer, prompts)
    results["batch_size"] = test_throughput_vs_batch_size(llm, tokenizer, prompts)
    results["breakdown"] = test_initial_vs_counterfactual(llm, tokenizer, prompts)
    results["prefix_caching"] = test_prefix_caching(llm, tokenizer, prompts)
    results["gpu_pattern"] = test_gpu_utilization_pattern(llm, tokenizer, prompts)

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print("\n1. THROUGHPUT vs MAX_TOKENS:")
    print("   Longer sequences = lower throughput (more KV cache, more compute)")
    for r in results["max_tokens"]:
        print(f"   {r['max_tokens']:4d} tokens -> {r['throughput_tok_s']:5.0f} tok/s")

    print("\n2. BATCH SIZE SCALING:")
    print("   Larger batches amortize overhead, improve GPU utilization")
    for r in results["batch_size"]:
        print(f"   batch={r['batch_size']:3d} -> {r['throughput_tok_s']:5.0f} tok/s")

    print("\n3. TIME BREAKDOWN:")
    avg_initial_pct = sum(r["initial_pct"] for r in results["breakdown"]) / len(results["breakdown"])
    print(f"   Initial rollout: ~{avg_initial_pct:.0f}% of time")
    print(f"   Counterfactuals: ~{100-avg_initial_pct:.0f}% of time")

    print("\n4. PREFIX CACHING:")
    print(f"   Caching benefit: {results['prefix_caching']['caching_ratio']:.2f}x")
    if results['prefix_caching']['caching_ratio'] > 1.2:
        print("   -> Prefix caching IS helping")
    else:
        print("   -> Prefix caching has minimal effect (different prefixes dominate)")

    print("\n5. GPU UTILIZATION:")
    if results["gpu_pattern"]:
        print(f"   GPU compute: {results['gpu_pattern']['avg_gpu_util']:.0f}%")
        print(f"   Memory bandwidth: {results['gpu_pattern']['avg_mem_util']:.0f}%")
        if results['gpu_pattern']['avg_gpu_util'] > 90:
            print("   -> COMPUTE BOUND (GPU maxed out)")
        elif results['gpu_pattern']['avg_mem_util'] > 90:
            print("   -> MEMORY BANDWIDTH BOUND")
        else:
            print("   -> May have scheduling/overhead bottleneck")

    print("\n" + "="*70)
    print("BOTTLENECK ANALYSIS:")
    print("="*70)

    # Identify primary bottleneck
    if results["gpu_pattern"] and results["gpu_pattern"]["avg_gpu_util"] > 85:
        print("PRIMARY BOTTLENECK: GPU Compute")
        print("- You are extracting most of the GPU's capability")
        print("- Only way to speed up: smaller model, more GPUs, or quantization")

    if results["max_tokens"][-1]["throughput_tok_s"] < results["max_tokens"][0]["throughput_tok_s"] * 0.5:
        print("\nSECONDARY: Sequence length penalty")
        print("- Longer sequences significantly reduce throughput")
        print("- Consider if you really need 4096+ token outputs")

    # Save results
    with open("logs/diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: logs/diagnostic_results.json")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
