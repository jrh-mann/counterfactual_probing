"""
Main pipeline orchestration.

This module provides the main entry point for running the counterfactual
probing pipeline with two-phase generation:
1. Generate multiple initial rollouts, select shortest valid one
2. Generate counterfactuals only for valid rollouts
"""

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import Config, load_config
from .dataset import Dataset
from .generator import (
    generate_all_branch_counterfactuals,
    generate_initial_rollout,
)
from .sampler import TokenSampler
from .scorer import Scorer, load_scorer

logger = logging.getLogger(__name__)


class RolloutSelectionError(Exception):
    """Raised when no valid rollout can be selected."""
    pass


def is_rollout_valid(
    token_ids: list[int],
    max_tokens: int,
    boundary_token: int | None,
) -> dict[str, Any]:
    """
    Check if a rollout is valid for counterfactual generation.

    Args:
        token_ids: The generated token IDs
        max_tokens: Maximum tokens that were allowed
        boundary_token: The CoT boundary token (e.g., </think>)

    Returns:
        Dict with:
        - valid: bool
        - truncated: bool (hit max_tokens)
        - has_boundary: bool (contains boundary token)
        - boundary_position: int or None
        - reason: str (if invalid)
    """
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
        # No boundary token configured - consider it valid if not truncated
        has_boundary = True

    valid = not truncated and has_boundary

    reason = None
    if truncated:
        reason = f"Truncated at {len(token_ids)} tokens (max={max_tokens})"
    elif not has_boundary and boundary_token is not None:
        reason = f"Missing boundary token {boundary_token}"

    return {
        "valid": valid,
        "truncated": truncated,
        "has_boundary": has_boundary,
        "boundary_position": boundary_position,
        "length": len(token_ids),
        "reason": reason,
    }


def generate_and_select_rollout(
    messages: list[dict[str, str]],
    model_name: str,
    tokenizer,
    llm,
    num_rollouts: int,
    temperature: float,
    max_tokens: int,
    max_tokens_retry: int | None,
    boundary_token: int | None,
    prompt_id: str,
) -> dict[str, Any] | None:
    """
    Generate multiple rollouts and select the shortest valid one.

    Args:
        messages: Chat messages for the prompt
        model_name: Model name
        tokenizer: Tokenizer
        llm: vLLM instance
        num_rollouts: Number of rollouts to generate
        temperature: Sampling temperature
        max_tokens: Initial max tokens
        max_tokens_retry: Max tokens for retry (if None, uses max_tokens * 1.5)
        boundary_token: CoT boundary token ID (e.g., </think>)
        prompt_id: Identifier for logging

    Returns:
        Dict with selected rollout info, or None if all failed
    """
    if max_tokens_retry is None:
        max_tokens_retry = int(max_tokens * 1.5)

    all_rollouts = []

    # Phase 1: Generate initial rollouts
    print(f"  [{prompt_id}] Generating {num_rollouts} initial rollouts...")

    for i in range(num_rollouts):
        rollout = generate_initial_rollout(
            messages=messages,
            model_name=model_name,
            tokenizer=tokenizer,
            temperature=temperature,
            max_tokens=max_tokens,
            llm=llm,
        )

        validation = is_rollout_valid(
            rollout["token_ids"],
            max_tokens,
            boundary_token,
        )

        all_rollouts.append({
            "rollout": rollout,
            "validation": validation,
            "attempt": "initial",
        })

    # Filter to valid rollouts
    valid_rollouts = [r for r in all_rollouts if r["validation"]["valid"]]

    # Log initial attempt results
    num_truncated = sum(1 for r in all_rollouts if r["validation"]["truncated"])
    num_no_boundary = sum(1 for r in all_rollouts if not r["validation"]["has_boundary"])

    print(f"  [{prompt_id}] Initial: {len(valid_rollouts)}/{num_rollouts} valid "
          f"({num_truncated} truncated, {num_no_boundary} missing boundary)")

    # If none valid, retry with higher max_tokens
    if not valid_rollouts and num_truncated > 0:
        print(f"  [{prompt_id}] Retrying with max_tokens={max_tokens_retry}...")

        for i in range(num_rollouts):
            rollout = generate_initial_rollout(
                messages=messages,
                model_name=model_name,
                tokenizer=tokenizer,
                temperature=temperature,
                max_tokens=max_tokens_retry,
                llm=llm,
            )

            validation = is_rollout_valid(
                rollout["token_ids"],
                max_tokens_retry,
                boundary_token,
            )

            all_rollouts.append({
                "rollout": rollout,
                "validation": validation,
                "attempt": "retry",
            })

        # Re-filter
        valid_rollouts = [r for r in all_rollouts if r["validation"]["valid"]]

        retry_truncated = sum(1 for r in all_rollouts if r["attempt"] == "retry" and r["validation"]["truncated"])
        retry_no_boundary = sum(1 for r in all_rollouts if r["attempt"] == "retry" and not r["validation"]["has_boundary"])

        print(f"  [{prompt_id}] Retry: {len([r for r in valid_rollouts if r['attempt'] == 'retry'])}/{num_rollouts} valid "
              f"({retry_truncated} truncated, {retry_no_boundary} missing boundary)")

    # If still none valid, log loudly and return None
    if not valid_rollouts:
        error_msg = (
            f"FAILED: No valid rollout for {prompt_id} after {len(all_rollouts)} attempts. "
            f"All rollouts either truncated or missing boundary token."
        )
        print(f"\n{'='*60}")
        print(f"  ERROR: {error_msg}")
        print(f"{'='*60}\n")
        logger.error(error_msg)
        return None

    # Select shortest valid rollout
    valid_rollouts.sort(key=lambda r: r["validation"]["length"])
    selected = valid_rollouts[0]

    lengths = [r["validation"]["length"] for r in valid_rollouts]
    print(f"  [{prompt_id}] Selected: {selected['validation']['length']} tokens "
          f"(from {len(valid_rollouts)} valid, lengths: {lengths})")

    return {
        "token_ids": selected["rollout"]["token_ids"],
        "text": selected["rollout"]["text"],
        "selection_info": {
            "num_attempts": len(all_rollouts),
            "num_valid": len(valid_rollouts),
            "selected_length": selected["validation"]["length"],
            "boundary_position": selected["validation"]["boundary_position"],
            "attempt_type": selected["attempt"],
            "all_lengths": [r["validation"]["length"] for r in all_rollouts],
            "all_valid": [r["validation"]["valid"] for r in all_rollouts],
        },
    }


def run(config_path: str) -> None:
    """
    Run the counterfactual probing pipeline.

    Args:
        config_path: Path to configuration JSON file

    Raises:
        FileNotFoundError: If config or dataset file not found
        ValueError: If configuration is invalid
    """
    # Load and validate config
    config = load_config(config_path)

    # Set up output directory
    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = Dataset(
        path=config.dataset.path,
        prompt_field=config.dataset.prompt_field,
        format=config.dataset.format,
    )

    # Initialize sampler
    sampler = TokenSampler(
        method=config.sampling.method,
        num_samples=config.sampling.num_samples,
        density=config.sampling.density,
        seed=config.sampling.seed,
        cot_boundary_token=config.sampling.cot_boundary_token,
    )

    # Load scorer if configured
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

    # Track statistics
    stats = {
        "total": 0,
        "skipped": 0,
        "success": 0,
        "failed_rollout": 0,
    }

    # Process each prompt
    dataset_list = list(dataset)
    for idx, item in enumerate(tqdm(dataset_list, desc="Processing prompts")):
        prompt = item["prompt"]
        metadata = item["metadata"]

        # Determine output filename
        prompt_id = metadata.get("id", f"prompt_{idx:04d}")
        output_path = output_dir / f"{prompt_id}.json"

        stats["total"] += 1

        # Skip if exists and skip_existing is True
        if config.skip_existing and output_path.exists():
            stats["skipped"] += 1
            continue

        # Process this prompt
        result = process_prompt(
            prompt=prompt,
            metadata=metadata,
            prompt_id=prompt_id,
            config=config,
            sampler=sampler,
            scorer=scorer,
            tokenizer=tokenizer,
            llm=llm,
        )

        if result is None:
            stats["failed_rollout"] += 1
            continue

        stats["success"] += 1

        # Save result
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    # Print final statistics
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total prompts:      {stats['total']}")
    print(f"Skipped (existing): {stats['skipped']}")
    print(f"Success:            {stats['success']}")
    print(f"Failed (no valid):  {stats['failed_rollout']}")
    print("=" * 60)


def process_prompt(
    prompt: str,
    metadata: dict[str, Any],
    prompt_id: str,
    config: Config,
    sampler: TokenSampler,
    scorer: Scorer | None,
    tokenizer,
    llm,
) -> dict[str, Any] | None:
    """
    Process a single prompt through the pipeline.

    Args:
        prompt: The prompt text
        metadata: Additional metadata from dataset
        prompt_id: Identifier for this prompt
        config: Pipeline configuration
        sampler: Token sampler for selecting branch points
        scorer: Optional scorer for evaluating completions
        tokenizer: Tokenizer for the model
        llm: vLLM instance

    Returns:
        Result dictionary with initial rollout and samples, or None if failed
    """
    # Create chat messages
    messages = [{"role": "user", "content": prompt}]

    # Phase 1: Generate and select best rollout
    selected_rollout = generate_and_select_rollout(
        messages=messages,
        model_name=config.model.name,
        tokenizer=tokenizer,
        llm=llm,
        num_rollouts=config.generation.num_initial_rollouts,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        max_tokens_retry=config.generation.max_tokens_retry,
        boundary_token=config.sampling.cot_boundary_token,
        prompt_id=prompt_id,
    )

    if selected_rollout is None:
        return None

    initial_token_ids = selected_rollout["token_ids"]

    # Set scorer context (for scorers that need ground truth)
    if scorer:
        scorer.set_context(prompt, metadata)

    # Phase 2: Sample branch points (will respect boundary token)
    branch_points = sampler.sample(initial_token_ids)

    print(f"  [{prompt_id}] Sampling {len(branch_points)} branch points, "
          f"generating {config.generation.num_counterfactuals} counterfactuals each...")

    # Generate counterfactuals for ALL branch points in a single batched call
    samples = generate_all_branch_counterfactuals(
        initial_token_ids=initial_token_ids,
        branch_points=branch_points,
        messages=messages,
        tokenizer=tokenizer,
        num_counterfactuals=config.generation.num_counterfactuals,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        llm=llm,
    )

    # Score all counterfactuals
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

        # Calculate aggregate score
        sample["p_score"] = sum(scores) / len(scores) if scores else 0.0

    # Build result
    result = {
        "prompt_id": prompt_id,
        "metadata": metadata,
        "initial_rollout": {
            "token_ids": initial_token_ids,
            "text": selected_rollout["text"],
            "selection_info": selected_rollout["selection_info"],
        },
        "samples": samples,
    }

    return result


def run_from_config(config: Config) -> None:
    """
    Run the pipeline from a Config object (useful for programmatic use).

    Args:
        config: Pre-loaded Config object
    """
    # Create a temporary config file and run
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config.model_dump(), f)
        temp_path = f.name

    try:
        run(temp_path)
    finally:
        Path(temp_path).unlink()
