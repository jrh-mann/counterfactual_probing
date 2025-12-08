"""
Utilities for reconstructing sequences and formatting for activation extraction.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer

# Add src directory to path to import from utils
# Path from src/activations/utils.py -> src/
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.text_utils import get_sentence_prefix


def reconstruct_sequence_messages(sentence_data_list: List[Dict[str, Any]], sentence_index: int) -> List[Dict[str, str]]:
    """
    Reconstruct chat messages up to a given sentence index.
    
    Args:
        sentence_data_list: List of sentence data dictionaries from rollout JSON
        sentence_index: Index of the sentence to include (0-based)
    
    Returns:
        List of chat messages: [user_prompt, assistant_response_up_to_index]
    """
    if not sentence_data_list:
        raise ValueError("sentence_data_list cannot be empty")
    
    if sentence_index < 0 or sentence_index >= len(sentence_data_list):
        raise ValueError(f"sentence_index {sentence_index} out of range [0, {len(sentence_data_list)})")
    
    # Get the formatted prompt from the first sentence (should be same for all)
    first_sentence = sentence_data_list[0]
    formatted_prompt = first_sentence.get("formatted_prompt", [])
    
    if not formatted_prompt:
        raise ValueError("formatted_prompt not found in sentence data")
    
    # Extract sentences up to and including sentence_index
    sentences = [s["sentence"] for s in sentence_data_list[:sentence_index + 1]]
    
    # Concatenate sentences to form assistant response
    assistant_content = get_sentence_prefix(sentences, sentence_index)
    
    # Construct messages: [user_prompt, assistant_response]
    messages = formatted_prompt.copy()  # Start with user prompt
    messages.append({"role": "assistant", "content": assistant_content})
    
    return messages


def format_with_chat_template(messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizer) -> str:
    """
    Format chat messages with tokenizer's chat template.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: Tokenizer with apply_chat_template method
    
    Returns:
        Formatted string with chat template applied
    """
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise ValueError("Tokenizer does not have apply_chat_template method")
    
    chat_formatted = tokenizer.apply_chat_template(
        messages,  # type: ignore[arg-type]
        tokenize=False,
        add_generation_prompt=False
    )
    
    if not isinstance(chat_formatted, str):
        raise TypeError(f"Expected string from apply_chat_template, got {type(chat_formatted)}")
    
    return chat_formatted

