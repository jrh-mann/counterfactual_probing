"""
Staged pipeline for efficient counterfactual probing.

Stage 1: Generate initial rollouts for all problems, save to intermediate file
Stage 2: Filter valid rollouts, select shortest N, run counterfactuals

This allows cherry-picking the most efficient rollouts before expensive
counterfactual generation.
"""

import json
import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict
from datetime import datetime

from tqdm import tqdm

from .config import Config, load_config
from .dataset import Dataset
from .generator import generate_initial_rollouts_batch
from .sampler import TokenSampler
from .scorer import Scorer, load_scorer

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Result from initial rollout generation."""
    prompt_id: str
    prompt: str
    metadata: dict
    token_ids: list[int]
    text: str
    length: int
    truncated: bool
    has_boundary: bool
    boundary_position: int | None
    valid: bool
    timestamp: str


def check_rollout_validity(
    token_ids: list[int],
    max_tokens: int,
    boundary_token: int | None,
) -> dict[str, Any]:
    """Check if rollout is valid for counterfactual generation."""
    truncated = len(token_ids) >= max_tokens

    has_boundary = False
    boundary_position = None

    if boundary_token is not None:
        try:
            boundary_position = token_ids.index(boundary_token)
            has_boundary = True
        except ValueError:
            has_boundary = False
    else:
        has_boundary = True  # No boundary required

    valid = not truncated and has_boundary

    return {
        "truncated": truncated,
        "has_boundary": has_boundary,
        "boundary_position": boundary_position,
        "valid": valid,
    }


def run_stage1(
    config_path: str,
    output_path: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Stage 1: Generate initial rollouts for all problems.

    Args:
        config_path: Path to configuration JSON file
        output_path: Path to save rollouts JSONL file
        limit: Optional limit on number of problems to process

    Returns:
        Statistics dict
    """
    config = load_config(config_path)

    # Load dataset
    dataset = Dataset(
        path=config.dataset.path,
        prompt_field=config.dataset.prompt_field,
        format=config.dataset.format,
    )

    # Initialize model and tokenizer
    from transformers import AutoTokenizer
    from vllm import LLM

    print("Loading model...")
    llm = LLM(
        model=config.model.name,
        tensor_parallel_size=config.model.tensor_parallel_size,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=True,
    )

    # Get boundary token
    boundary_token = config.sampling.cot_boundary_token

    # Stats
    stats = {
        "total": 0,
        "valid": 0,
        "truncated": 0,
        "missing_boundary": 0,
        "lengths": [],
    }

    # Process and save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset_list = list(dataset)
    if limit:
        dataset_list = dataset_list[:limit]

    print(f"\nStage 1: Generating {len(dataset_list)} initial rollouts...")
    print(f"Output: {output_path}")
    print()

    # Prepare all prompts for batch generation
    print("Preparing prompts...")
    all_prompts = []
    all_metadata = []
    for idx, item in enumerate(dataset_list):
        prompt = item["prompt"]
        metadata = item["metadata"]
        prompt_id = metadata.get("id", f"prompt_{idx:04d}")

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_prompts.append(formatted_prompt)
        all_metadata.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "metadata": metadata,
        })

    # Generate all rollouts in one batch
    print(f"Generating {len(all_prompts)} rollouts in batch...")
    rollouts = generate_initial_rollouts_batch(
        prompts=all_prompts,
        tokenizer=tokenizer,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        llm=llm,
    )

    # Process and save results
    print("Processing and saving results...")
    with open(output_file, "w") as f:
        for rollout, meta in tqdm(zip(rollouts, all_metadata), total=len(rollouts), desc="Saving"):
            # Check validity
            validity = check_rollout_validity(
                rollout["token_ids"],
                config.generation.max_tokens,
                boundary_token,
            )

            # Create result
            result = RolloutResult(
                prompt_id=meta["prompt_id"],
                prompt=meta["prompt"],
                metadata=meta["metadata"],
                token_ids=rollout["token_ids"],
                text=rollout["text"],
                length=len(rollout["token_ids"]),
                truncated=validity["truncated"],
                has_boundary=validity["has_boundary"],
                boundary_position=validity["boundary_position"],
                valid=validity["valid"],
                timestamp=datetime.now().isoformat(),
            )

            # Write to file
            f.write(json.dumps(asdict(result)) + "\n")

            # Update stats
            stats["total"] += 1
            stats["lengths"].append(result.length)
            if result.valid:
                stats["valid"] += 1
            if result.truncated:
                stats["truncated"] += 1
            if not result.has_boundary:
                stats["missing_boundary"] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print(f"Total rollouts:     {stats['total']}")
    print(f"Valid:              {stats['valid']} ({100*stats['valid']/stats['total']:.1f}%)")
    print(f"Truncated:          {stats['truncated']} ({100*stats['truncated']/stats['total']:.1f}%)")
    print(f"Missing boundary:   {stats['missing_boundary']} ({100*stats['missing_boundary']/stats['total']:.1f}%)")
    if stats["lengths"]:
        print(f"Length range:       {min(stats['lengths'])} - {max(stats['lengths'])} tokens")
        print(f"Length median:      {sorted(stats['lengths'])[len(stats['lengths'])//2]} tokens")
    print(f"\nRollouts saved to: {output_path}")
    print("=" * 60)

    return stats


def run_stage2(
    config_path: str,
    rollouts_path: str,
    select_count: int,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Stage 2: Select shortest valid rollouts and run counterfactuals.

    Args:
        config_path: Path to configuration JSON file
        rollouts_path: Path to rollouts JSONL from stage 1
        select_count: Number of shortest valid rollouts to select
        output_dir: Override output directory (optional)

    Returns:
        Statistics dict
    """
    config = load_config(config_path)

    if output_dir:
        config.output.dir = output_dir

    # Load rollouts
    print(f"Loading rollouts from {rollouts_path}...")
    rollouts = []
    with open(rollouts_path) as f:
        for line in f:
            rollouts.append(json.loads(line))

    print(f"Loaded {len(rollouts)} rollouts")

    # Filter to valid
    valid_rollouts = [r for r in rollouts if r["valid"]]
    print(f"Valid rollouts: {len(valid_rollouts)}")

    if len(valid_rollouts) < select_count:
        print(f"WARNING: Only {len(valid_rollouts)} valid rollouts, requested {select_count}")
        select_count = len(valid_rollouts)

    # Sort by length and select shortest
    valid_rollouts.sort(key=lambda r: r["length"])
    selected = valid_rollouts[:select_count]

    print(f"\nSelected {len(selected)} shortest rollouts:")
    print(f"  Length range: {selected[0]['length']} - {selected[-1]['length']} tokens")
    print(f"  Rejected lengths: {valid_rollouts[select_count]['length'] if len(valid_rollouts) > select_count else 'N/A'} - {valid_rollouts[-1]['length'] if valid_rollouts else 'N/A'} tokens")
    print()

    # Initialize components
    sampler = TokenSampler(
        method=config.sampling.method,
        num_samples=config.sampling.num_samples,
        density=config.sampling.density,
        seed=config.sampling.seed,
        cot_boundary_token=config.sampling.cot_boundary_token,
    )

    scorer: Scorer | None = None
    if config.scorer:
        scorer = load_scorer({
            "module": config.scorer.module,
            "class": config.scorer.class_name,
            "config": config.scorer.config,
        })

    # Initialize model and tokenizer
    from transformers import AutoTokenizer
    from vllm import LLM

    print("Loading model...")
    llm = LLM(
        model=config.model.name,
        tensor_parallel_size=config.model.tensor_parallel_size,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=True,
    )

    # Set up output directory
    output_path = Path(config.output.dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Import generator function
    from .generator import generate_all_branch_counterfactuals

    # Stats
    stats = {
        "total": len(selected),
        "skipped": 0,
        "success": 0,
    }

    print(f"\nStage 2: Running counterfactuals on {len(selected)} rollouts...")
    print(f"Output: {config.output.dir}")
    print()

    for rollout in tqdm(selected, desc="Processing counterfactuals"):
        prompt_id = rollout["prompt_id"]
        output_file = output_path / f"{prompt_id}.json"

        # Skip if exists
        if config.skip_existing and output_file.exists():
            stats["skipped"] += 1
            continue

        messages = [{"role": "user", "content": rollout["prompt"]}]
        token_ids = rollout["token_ids"]

        # Set scorer context
        if scorer:
            scorer.set_context(rollout["prompt"], rollout["metadata"])

        # Sample branch points
        branch_points = sampler.sample(token_ids)

        # Generate counterfactuals
        samples = generate_all_branch_counterfactuals(
            initial_token_ids=token_ids,
            branch_points=branch_points,
            messages=messages,
            tokenizer=tokenizer,
            num_counterfactuals=config.generation.num_counterfactuals,
            temperature=config.generation.temperature,
            max_tokens=config.generation.max_tokens,
            llm=llm,
        )

        # Score counterfactuals
        for sample in samples:
            scores = []
            for cf in sample["counterfactuals"]:
                if scorer:
                    text = tokenizer.decode(cf["token_ids"], skip_special_tokens=True)
                    score = scorer.score(text)
                else:
                    score = 0.0
                cf["score"] = score
                scores.append(score)
            sample["p_score"] = sum(scores) / len(scores) if scores else 0.0

        # Build result
        result = {
            "prompt_id": prompt_id,
            "metadata": rollout["metadata"],
            "initial_rollout": {
                "token_ids": token_ids,
                "text": rollout["text"],
                "length": rollout["length"],
                "boundary_position": rollout["boundary_position"],
            },
            "samples": samples,
        }

        # Save
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        stats["success"] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
    print(f"Total selected:     {stats['total']}")
    print(f"Skipped (existing): {stats['skipped']}")
    print(f"Success:            {stats['success']}")
    print(f"\nOutputs saved to: {config.output.dir}")
    print("=" * 60)

    return stats


def run_both_stages(
    config_path: str,
    num_rollouts: int,
    select_count: int,
    rollouts_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run both stages in sequence.

    Args:
        config_path: Path to configuration JSON file
        num_rollouts: Number of rollouts to generate in stage 1
        select_count: Number of shortest to select for stage 2
        rollouts_path: Path for intermediate rollouts file (optional)
        output_dir: Override output directory (optional)

    Returns:
        Combined statistics
    """
    if rollouts_path is None:
        config = load_config(config_path)
        rollouts_path = str(Path(config.output.dir) / "rollouts.jsonl")

    print("=" * 60)
    print("STAGED COUNTERFACTUAL PROBING")
    print("=" * 60)
    print(f"Stage 1: Generate {num_rollouts} rollouts")
    print(f"Stage 2: Select shortest {select_count} for counterfactuals")
    print("=" * 60)
    print()

    # Stage 1
    stats1 = run_stage1(config_path, rollouts_path, limit=num_rollouts)

    print()

    # Stage 2
    stats2 = run_stage2(config_path, rollouts_path, select_count, output_dir)

    return {"stage1": stats1, "stage2": stats2}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Staged counterfactual probing")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    parser.add_argument("--num-rollouts", type=int, default=800, help="Rollouts to generate")
    parser.add_argument("--select", type=int, default=500, help="Number to select")
    parser.add_argument("--rollouts-path", help="Intermediate rollouts file")
    parser.add_argument("--output-dir", help="Override output directory")

    args = parser.parse_args()

    if args.stage == "1":
        run_stage1(args.config, args.rollouts_path or "rollouts.jsonl", args.num_rollouts)
    elif args.stage == "2":
        run_stage2(args.config, args.rollouts_path or "rollouts.jsonl", args.select, args.output_dir)
    else:
        run_both_stages(args.config, args.num_rollouts, args.select, args.rollouts_path, args.output_dir)
