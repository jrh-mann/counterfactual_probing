"""
Activation extraction using nnsight.
Based on reference implementation pattern.
"""

import nnsight
import torch
import json
import os
import gc
import logging
import threading
import queue
from typing import List, Dict, Optional, Any
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer

from .utils import reconstruct_sequence_messages, format_with_chat_template

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _save_activations_worker(save_queue: queue.Queue) -> None:
    """Worker function to save activations from queue to disk."""
    while True:
        item = save_queue.get()
        if item is None:  # Sentinel to stop worker
            break
        
        save_path, token_ids, activations = item
        try:
            # Save as tuple of (token_ids, activations)
            torch.save((token_ids, activations), save_path)
            logger.info(f"Saved activations to {save_path}")
        except Exception as e:
            logger.error(f"Error saving activations to {save_path}: {e}")
        finally:
            save_queue.task_done()


def store_activations(
    model_name: str,
    sentence_data_list: List[Dict[str, Any]],
    activations_dir: str,
    tokenizer: PreTrainedTokenizer,
    model: Any,  # nnsight.LanguageModel
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
    token_position: str = "final",
    verbose: bool = False
) -> None:
    """
    Extract and store activations from sentence data.
    
    Args:
        model_name: HuggingFace model identifier (for logging)
        sentence_data_list: List of sentence data dictionaries from rollout JSON
        activations_dir: Directory to save activation tensors
        tokenizer: Pre-loaded tokenizer
        model: Pre-loaded nnsight.LanguageModel
        start_idx: Starting sentence index (defaults to 0)
        end_idx: Ending sentence index (defaults to None, processes all)
        layer_idx: Specific layer index to extract (defaults to None, extracts all layers)
        token_position: Which token to extract (defaults to "final", can be "final" or integer index)
        verbose: Print detailed output for review
    """
    # Create activations directory
    os.makedirs(activations_dir, exist_ok=True)
    logger.info(f"Activations will be saved to: {activations_dir}")
    
    # Determine range to process
    # Note: start_idx and end_idx refer to positions in sentence_data_list, not sentence_index values
    # This is important when using selected sentences format
    end_idx = end_idx if end_idx is not None else len(sentence_data_list)
    sentences_to_process = sentence_data_list[start_idx:end_idx]
    
    # Check if we're using selected sentences format
    has_sentence_index_field = all("sentence_index" in s for s in sentence_data_list)
    if has_sentence_index_field:
        # Get the actual sentence indices that will be processed
        sentence_indices = [s.get("sentence_index") for s in sentences_to_process]
        logger.info(f"Processing {len(sentences_to_process)} sentences (list positions {start_idx} to {end_idx-1})")
        logger.info(f"  Sentence indices: {sentence_indices}")
    else:
        logger.info(f"Processing {len(sentences_to_process)} sentences (indices {start_idx} to {end_idx-1})")
    
    # Parse token_position
    if token_position == "final":
        token_pos = None  # Will use -2 to get final content token (skip EOS at -1)
    else:
        try:
            token_pos = int(token_position)
        except ValueError:
            raise ValueError(f"token_position must be 'final' or an integer, got: {token_position}")
    
    # Create queue for saving activations and start worker thread
    save_queue: queue.Queue = queue.Queue(maxsize=10)  # Limit queue size to prevent excessive memory usage
    save_thread = threading.Thread(target=_save_activations_worker, args=(save_queue,))
    save_thread.start()
    
    try:
        # Process one at a time (simple and reliable)
        with torch.no_grad():
            for i, sentence_data in enumerate(sentences_to_process):
                actual_idx = start_idx + i
                sentence_index = sentence_data.get("sentence_index", actual_idx)
                
                # Clear GPU memory
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    # Reconstruct sequence messages
                    messages = reconstruct_sequence_messages(sentence_data_list, sentence_index)
                    
                    # Format with chat template
                    chat_formatted = format_with_chat_template(messages, tokenizer)
                    
                    if verbose:
                        logger.info(f"\n=== Sentence {sentence_index} ===")
                        logger.info(f"Messages: {messages}")
                        logger.info(f"Chat formatted (first 500 chars): {chat_formatted[:500]}")
                        if len(chat_formatted) > 700:
                            logger.info(f"Chat formatted (last 200 chars): ...{chat_formatted[-200:]}")
                    
                    # Tokenize to get token IDs (do this first, outside trace)
                    token_ids = tokenizer.encode(chat_formatted, return_tensors='pt')[0]
                    
                    if verbose:
                        logger.info(f"Token IDs shape: {token_ids.shape}")
                        logger.info(f"Token IDs (first 20): {token_ids[:20].tolist()}")
                        logger.info(f"Token IDs (last 20): {token_ids[-20:].tolist()}")
                    
                    # Extract activations using nnsight trace
                    with model.trace(chat_formatted):
                        layer_outputs = []
                        
                        if layer_idx is not None:
                            # Extract single layer
                            num_layers = len(model.model.layers)  # type: ignore[attr-defined]
                            if layer_idx >= num_layers or layer_idx < 0:
                                raise ValueError(f"layer_idx {layer_idx} is out of range. Model has {num_layers} layers (0-{num_layers-1})")
                            layer_outputs.append(model.model.layers[layer_idx].output[0])  # type: ignore[attr-defined]
                        else:
                            # Extract all layers
                            for layer in model.model.layers:  # type: ignore[attr-defined]
                                layer_outputs.append(layer.output[0])
                        
                        # Stack: (num_layers, seq_len, d_model)
                        activations = torch.stack(layer_outputs, dim=0)
                        
                        # Extract specified token position
                        if token_pos is None:
                            # Extract final content token (skip EOS token at -1)
                            # The chat template adds <|return|> (EOS) at the end, so we want -2
                            # This gives us the last actual content token, not the EOS token
                            token_activations = activations[:, -2, :]
                            actual_token_pos = -2
                        else:
                            # Extract specific token position
                            if token_pos >= activations.shape[1] or token_pos < -activations.shape[1]:
                                raise ValueError(f"token_pos {token_pos} out of range for sequence length {activations.shape[1]}")
                            token_activations = activations[:, token_pos, :]
                            actual_token_pos = token_pos
                        
                        if verbose:
                            logger.info(f"Full activations shape: {activations.shape}")
                            logger.info(f"Extracted token activations shape: {token_activations.shape}")
                            logger.info(f"Token position extracted: {actual_token_pos} (position -1 is EOS token <|return|>)")
                            # Show what token we're extracting
                            if actual_token_pos < 0:
                                token_idx = len(token_ids) + actual_token_pos
                            else:
                                token_idx = actual_token_pos
                            if 0 <= token_idx < len(token_ids):
                                extracted_token_id = token_ids[token_idx].item()
                                try:
                                    extracted_token_text = tokenizer.decode([extracted_token_id])
                                    logger.info(f"Extracted token ID: {extracted_token_id}, decoded: {repr(extracted_token_text)}")
                                except Exception:
                                    logger.info(f"Extracted token ID: {extracted_token_id} (could not decode)")
                        
                        # Move to CPU and add to save queue
                        token_ids_cpu = token_ids.cpu()
                        activations_cpu = token_activations.cpu()
                        save_path = Path(activations_dir) / f"sentence_{sentence_index}.pt"
                        
                        # Put in queue for background saving as tuple (token_ids, activations)
                        save_queue.put((save_path, token_ids_cpu, activations_cpu))
                        logger.info(f"Queued activations for sentence {sentence_index}")
                    
                except Exception as e:
                    logger.error(f"Error on sentence {sentence_index}: {e}")
                    import traceback
                    if verbose:
                        traceback.print_exc()
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
        
        # Wait for all saves to complete
        save_queue.join()
        logger.info("All activations processed and queued for saving")
        
    finally:
        # Stop the save worker thread
        save_queue.put(None)  # Sentinel to stop worker
        save_thread.join()
    
    logger.info("Activation storage complete!")

