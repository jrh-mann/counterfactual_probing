"""
Main pipeline orchestration.

This module provides the main entry point for running the counterfactual
probing pipeline.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm

from .config import load_config, Config
from .dataset import Dataset
from .sampler import TokenSampler
from .scorer import load_scorer, Scorer
from .generator import generate_initial_rollout, generate_counterfactuals_batch


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
    )

    # Load scorer if configured
    scorer: Optional[Scorer] = None
    if config.scorer:
        scorer = load_scorer({
            "module": config.scorer.module,
            "class": config.scorer.class_name,
            "config": config.scorer.config,
        })

    # Initialize model and tokenizer
    from vllm import LLM
    from transformers import AutoTokenizer

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

    # Process each prompt
    for idx, item in enumerate(tqdm(dataset, desc="Processing prompts")):
        prompt = item["prompt"]
        metadata = item["metadata"]

        # Determine output filename
        prompt_id = metadata.get("id", f"prompt_{idx:04d}")
        output_path = output_dir / f"{prompt_id}.json"

        # Skip if exists and skip_existing is True
        if config.skip_existing and output_path.exists():
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

        # Save result
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)


def process_prompt(
    prompt: str,
    metadata: Dict[str, Any],
    prompt_id: str,
    config: Config,
    sampler: TokenSampler,
    scorer: Optional[Scorer],
    tokenizer,
    llm,
) -> Dict[str, Any]:
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
        Result dictionary with initial rollout and samples
    """
    # Create chat messages
    messages = [{"role": "user", "content": prompt}]

    # Generate initial rollout
    initial = generate_initial_rollout(
        messages=messages,
        model_name=config.model.name,
        tokenizer=tokenizer,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        llm=llm,
    )

    initial_token_ids = initial["token_ids"]

    # Sample branch points
    branch_points = sampler.sample(initial_token_ids)

    # Generate counterfactuals at each branch point
    samples = []

    for token_idx in tqdm(branch_points, desc="Branch points", leave=False):
        prefix_ids = initial_token_ids[:token_idx + 1]

        # Generate counterfactuals
        counterfactuals = generate_counterfactuals_batch(
            prefix_ids=prefix_ids,
            messages=messages,
            model_name=config.model.name,
            tokenizer=tokenizer,
            num_counterfactuals=config.generation.num_counterfactuals,
            temperature=config.generation.temperature,
            max_tokens=config.generation.max_tokens,
            llm=llm,
        )

        # Score each counterfactual
        scores = []
        for cf in counterfactuals:
            if scorer:
                text = tokenizer.decode(cf["token_ids"], skip_special_tokens=True)
                score = scorer.score(text)
            else:
                score = 0.0
            cf["score"] = score
            scores.append(score)

        # Calculate aggregate score
        p_score = sum(scores) / len(scores) if scores else 0.0

        samples.append({
            "token_index": token_idx,
            "prefix_token_ids": prefix_ids,
            "counterfactuals": counterfactuals,
            "p_score": p_score,
        })

    # Build result
    result = {
        "prompt_id": prompt_id,
        "metadata": metadata,
        "initial_rollout": {
            "token_ids": initial_token_ids,
            "text": initial["text"],
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
