"""
Token-based counterfactual generation.

This module handles generating initial rollouts and counterfactual
continuations from prefix-conditioned prompts, working entirely with
token IDs for precision.
"""

import warnings
from typing import Any

from transformers import PreTrainedTokenizer


def validate_prefix_roundtrip(
    token_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    skip_special_tokens: bool = True,
) -> dict[str, Any]:
    """
    Validate that token IDs survive a decode -> encode round-trip.

    This is critical for ensuring token-level precision when creating
    prefix-conditioned prompts.

    Args:
        token_ids: Token IDs to validate
        tokenizer: Tokenizer to use for decode/encode
        skip_special_tokens: Whether to skip special tokens in decode

    Returns:
        Dict with:
        - valid: bool - whether round-trip succeeded
        - original_ids: the input token IDs
        - roundtrip_ids: token IDs after decode -> encode
        - decoded_text: the intermediate decoded text
        - mismatch_position: first position where tokens differ (if any)
    """
    if not token_ids:
        return {
            "valid": True,
            "original_ids": [],
            "roundtrip_ids": [],
            "decoded_text": "",
            "mismatch_position": None,
        }

    # Decode
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # Re-encode
    roundtrip_ids = tokenizer.encode(decoded_text, add_special_tokens=False)

    # Compare
    valid = token_ids == roundtrip_ids
    mismatch_position = None

    if not valid:
        # Find first mismatch
        for i, (orig, rt) in enumerate(zip(token_ids, roundtrip_ids)):
            if orig != rt:
                mismatch_position = i
                break
        else:
            # Length mismatch
            mismatch_position = min(len(token_ids), len(roundtrip_ids))

    return {
        "valid": valid,
        "original_ids": token_ids,
        "roundtrip_ids": roundtrip_ids,
        "decoded_text": decoded_text,
        "mismatch_position": mismatch_position,
    }


def decode_preserving_tokens(
    token_ids: list[int],
    tokenizer: PreTrainedTokenizer,
) -> str:
    """
    Decode token IDs to text, preserving ability to re-encode to same tokens.

    This uses skip_special_tokens=False to avoid losing information,
    but may include special token strings in the output.

    Args:
        token_ids: Token IDs to decode
        tokenizer: Tokenizer to use

    Returns:
        Decoded text that should re-encode to same token IDs
    """
    # Don't skip special tokens to preserve information
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def extract_continuation_safe(
    full_ids: list[int],
    prefix_ids: list[int],
) -> dict[str, Any]:
    """
    Extract continuation tokens with validation.

    Unlike extract_continuation(), this function validates the prefix match
    and reports any issues instead of silently ignoring them.

    Args:
        full_ids: Complete token sequence
        prefix_ids: Expected prefix token sequence

    Returns:
        Dict with:
        - prefix_matched: bool - whether prefix matched exactly
        - continuation: list[int] - extracted continuation tokens
        - mismatch_position: int or None - first position of mismatch
    """
    prefix_len = len(prefix_ids)

    # Check prefix match
    actual_prefix = full_ids[:prefix_len]
    prefix_matched = actual_prefix == prefix_ids

    mismatch_position = None
    if not prefix_matched:
        for i, (expected, actual) in enumerate(zip(prefix_ids, actual_prefix)):
            if expected != actual:
                mismatch_position = i
                break
        else:
            mismatch_position = min(len(prefix_ids), len(actual_prefix))

    return {
        "prefix_matched": prefix_matched,
        "continuation": full_ids[prefix_len:],
        "mismatch_position": mismatch_position,
    }


def create_prefix_prompt_tokens(
    prefix_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
) -> list[int]:
    """
    Create prompt token IDs for generating continuations from a prefix.

    This works entirely at the token level to avoid decode/encode drift.

    Args:
        prefix_ids: Token IDs of the prefix to continue from
        tokenizer: Tokenizer for chat template
        messages: List of chat messages (user/system)

    Returns:
        Token IDs for the full prompt including prefix
    """
    # Get the base prompt tokens (without assistant content)
    base_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    base_token_ids = tokenizer.encode(base_prompt, add_special_tokens=False)

    # Append the prefix tokens directly
    if prefix_ids:
        return base_token_ids + prefix_ids
    else:
        return base_token_ids


def create_prefix_prompt(
    prefix_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
) -> str:
    """
    Create a prompt string for generating continuations from a prefix.

    This formats the messages with the chat template and appends the
    prefix as partial assistant content.

    Args:
        prefix_ids: Token IDs of the prefix to continue from
        tokenizer: Tokenizer for decoding and chat template
        messages: List of chat messages (user/system)

    Returns:
        Formatted prompt string ready for generation
    """
    # Decode prefix to text
    if prefix_ids:
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    else:
        prefix_text = ""

    # Build messages with assistant prefix
    full_messages = messages.copy()
    if prefix_text:
        full_messages.append({"role": "assistant", "content": prefix_text})

    # Apply chat template
    if prefix_text:
        # We have assistant content, don't add generation prompt
        formatted = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # No assistant content yet, add generation prompt
        formatted = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return formatted


def validate_token_ids(token_ids: list[int], vocab_size: int) -> dict[str, Any]:
    """
    Validate that token IDs are within vocabulary range.

    Args:
        token_ids: List of token IDs to validate
        vocab_size: Size of the vocabulary

    Returns:
        Dict with 'valid' bool and 'error' message if invalid
    """
    for i, tid in enumerate(token_ids):
        if tid < 0 or tid >= vocab_size:
            return {
                "valid": False,
                "error": f"Token ID {tid} at position {i} is out of range [0, {vocab_size})"
            }
    return {"valid": True, "error": None}


def extract_continuation(
    full_ids: list[int],
    prefix_ids: list[int],
) -> list[int]:
    """
    Extract continuation tokens from full sequence given prefix.

    Args:
        full_ids: Complete token sequence
        prefix_ids: Prefix token sequence

    Returns:
        Token IDs of the continuation (full_ids without prefix_ids)
    """
    prefix_len = len(prefix_ids)

    # Verify prefix matches
    if full_ids[:prefix_len] != prefix_ids:
        # Prefix doesn't match exactly - this can happen with some tokenizers
        # Fall back to returning everything after prefix length
        pass

    return full_ids[prefix_len:]


def generate_counterfactuals(
    prefix_ids: list[int],
    messages: list[dict[str, str]],
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    num_counterfactuals: int = 50,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Generate counterfactual continuations from a prefix.

    Args:
        prefix_ids: Token IDs of the prefix to continue from
        messages: Chat messages (user/system prompts)
        model_name: Name/path of the model
        tokenizer: Tokenizer for the model
        num_counterfactuals: Number of continuations to generate
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        llm: Optional pre-initialized vLLM instance

    Returns:
        List of dicts with 'token_ids' for each counterfactual
    """
    from vllm import LLM, SamplingParams

    # Initialize LLM if not provided
    if llm is None:
        llm = LLM(model=model_name, trust_remote_code=True)

    # Create the prefix prompt
    prompt = create_prefix_prompt(prefix_ids, tokenizer, messages)

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_counterfactuals,
    )

    # Generate
    outputs = llm.generate([prompt], sampling_params)

    # Extract continuations
    counterfactuals = []
    for output in outputs[0].outputs:
        # Get the generated token IDs
        generated_ids = list(output.token_ids)
        counterfactuals.append({
            "token_ids": generated_ids,
        })

    return counterfactuals


def generate_counterfactuals_batch(
    prefix_ids: list[int],
    messages: list[dict[str, str]],
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    num_counterfactuals: int = 50,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Generate counterfactuals using batch prompting for efficiency.

    Instead of using n= parameter, creates duplicate prompts for
    better batching behavior with prefix caching.

    Args:
        prefix_ids: Token IDs of the prefix to continue from
        messages: Chat messages (user/system prompts)
        model_name: Name/path of the model
        tokenizer: Tokenizer for the model
        num_counterfactuals: Number of continuations to generate
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        llm: Optional pre-initialized vLLM instance

    Returns:
        List of dicts with 'token_ids' for each counterfactual
    """
    from vllm import LLM, SamplingParams

    # Initialize LLM if not provided
    if llm is None:
        llm = LLM(model=model_name, trust_remote_code=True)

    # Create the prefix prompt
    prompt = create_prefix_prompt(prefix_ids, tokenizer, messages)

    # Duplicate prompts for batching
    prompts = [prompt] * num_counterfactuals

    # Create sampling params (n=1 since we're using batch)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Extract continuations
    counterfactuals = []
    for output in outputs:
        generated_ids = list(output.outputs[0].token_ids)
        counterfactuals.append({
            "token_ids": generated_ids,
        })

    return counterfactuals


def generate_all_branch_counterfactuals(
    initial_token_ids: list[int],
    branch_points: list[int],
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    num_counterfactuals: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Generate counterfactuals for ALL branch points in a single batched call.

    This is significantly faster than sequential generation because it allows
    vLLM to batch all prompts together for better GPU utilization.

    Args:
        initial_token_ids: Token IDs from the initial rollout
        branch_points: List of token indices to branch from
        messages: Chat messages (user/system prompts)
        tokenizer: Tokenizer for the model
        num_counterfactuals: Number of continuations per branch point
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        llm: Pre-initialized vLLM instance

    Returns:
        List of dicts, one per branch point, each containing:
        - token_index: The branch point index
        - prefix_token_ids: Token IDs up to this branch point
        - counterfactuals: List of generated continuations
    """
    from vllm import SamplingParams

    if llm is None:
        raise ValueError("llm must be provided for batched generation")

    # Build all prompts at once
    all_prompts = []
    prompt_to_branch = []  # Track which branch point each prompt belongs to

    for token_idx in branch_points:
        prefix_ids = initial_token_ids[:token_idx + 1]
        prefix_prompt = create_prefix_prompt(prefix_ids, tokenizer, messages)

        # Add num_counterfactuals copies of this prompt
        for _ in range(num_counterfactuals):
            all_prompts.append(prefix_prompt)
            prompt_to_branch.append(token_idx)

    # Single batched generation call
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    outputs = llm.generate(all_prompts, sampling_params)

    # Organize results by branch point
    results_by_branch: dict[int, list[dict]] = {bp: [] for bp in branch_points}

    for output, branch_idx in zip(outputs, prompt_to_branch):
        generated_ids = list(output.outputs[0].token_ids)
        results_by_branch[branch_idx].append({
            "token_ids": generated_ids,
        })

    # Build final result list in branch point order
    results = []
    for token_idx in branch_points:
        prefix_ids = initial_token_ids[:token_idx + 1]
        results.append({
            "token_index": token_idx,
            "prefix_token_ids": prefix_ids,
            "counterfactuals": results_by_branch[token_idx],
        })

    return results


def generate_initial_rollout(
    messages: list[dict[str, str]],
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    llm: Any | None = None,
) -> dict[str, Any]:
    """
    Generate an initial rollout for a prompt.

    Args:
        messages: Chat messages (user/system prompts)
        model_name: Name/path of the model
        tokenizer: Tokenizer for the model
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        llm: Optional pre-initialized vLLM instance

    Returns:
        Dict with 'token_ids' and 'text' of the generated rollout
    """
    from vllm import LLM, SamplingParams

    # Initialize LLM if not provided
    if llm is None:
        llm = LLM(model=model_name, trust_remote_code=True)

    # Format prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    # Generate
    outputs = llm.generate([prompt], sampling_params)

    # Extract result
    output = outputs[0].outputs[0]
    token_ids = list(output.token_ids)
    text = tokenizer.decode(token_ids, skip_special_tokens=True)

    return {
        "token_ids": token_ids,
        "text": text,
    }


def generate_initial_rollouts_batch(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    llm: Any = None,
) -> list[dict[str, Any]]:
    """
    Generate initial rollouts for multiple prompts in a single batch.

    This is much more efficient than calling generate_initial_rollout
    repeatedly, as vLLM can parallelize the generation.

    Args:
        prompts: List of formatted prompt strings
        tokenizer: Tokenizer for decoding
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate per rollout
        llm: Pre-initialized vLLM instance (required)

    Returns:
        List of dicts, each with 'token_ids' and 'text'
    """
    from vllm import SamplingParams

    if llm is None:
        raise ValueError("llm parameter is required for batch generation")

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    # Generate all at once
    outputs = llm.generate(prompts, sampling_params)

    # Extract results
    results = []
    for output in outputs:
        token_ids = list(output.outputs[0].token_ids)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        results.append({
            "token_ids": token_ids,
            "text": text,
        })

    return results
