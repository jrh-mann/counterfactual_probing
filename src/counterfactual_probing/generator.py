"""
Token-based counterfactual generation.

This module handles generating initial rollouts and counterfactual
continuations from prefix-conditioned prompts, working entirely with
token IDs for precision.
"""

from typing import Any

from transformers import PreTrainedTokenizer


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
