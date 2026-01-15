"""
Rigorous activation extraction for counterfactual probing outputs.

This module extracts activations while preserving tokenization integrity:
1. Uses saved token_ids directly (no text→tokens roundtrip for generation)
2. Properly reconstructs full sequence (prompt + generation)
3. Validates alignment at each step
4. Maps branch point indices correctly
"""

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer


@dataclass
class BranchPoint:
    """A branch point with its position and score."""
    token_index: int  # Index within generation tokens
    p_score: float    # Probability score from counterfactuals


@dataclass
class ActivationData:
    """Extracted activations for a single prompt."""
    activations: torch.Tensor      # (num_layers, gen_len, hidden_dim)
    generation_token_ids: list[int]  # The generation tokens
    branch_points: list[BranchPoint]
    prompt_id: str
    metadata: dict[str, Any]

    # For validation
    prompt_token_count: int
    full_sequence_length: int

    def get_branch_point_activations(
        self,
        layer_idx: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get activations and labels at branch points only.

        Args:
            layer_idx: Specific layer (None = all layers concatenated)

        Returns:
            Tuple of (activations, p_scores)
            - activations: (num_branch_points, hidden_dim) or (num_branch_points, num_layers * hidden_dim)
            - p_scores: (num_branch_points,)
        """
        indices = [bp.token_index for bp in self.branch_points]
        p_scores = np.array([bp.p_score for bp in self.branch_points])

        if layer_idx is not None:
            # Single layer
            acts = self.activations[layer_idx, indices, :].float().numpy()
        else:
            # All layers concatenated
            all_layers = self.activations[:, indices, :]  # (num_layers, num_bp, hidden_dim)
            all_layers = all_layers.permute(1, 0, 2)  # (num_bp, num_layers, hidden_dim)
            acts = all_layers.reshape(len(indices), -1).float().numpy()

        return acts, p_scores


class CounterfactualActivationExtractor:
    """
    Extract activations from counterfactual probing outputs.

    This class handles the full extraction pipeline:
    1. Loading counterfactual outputs
    2. Reconstructing full sequences with proper tokenization
    3. Running forward passes to extract activations
    4. Saving results with validation metadata
    """

    def __init__(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizer | None = None,
        model: Any | None = None,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the extractor.

        Args:
            model_name: HuggingFace model name
            tokenizer: Optional pre-loaded tokenizer
            model: Optional pre-loaded nnsight model
            device: Device for model ("auto", "cuda", "cpu")
            dtype: Model dtype
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        # Load tokenizer
        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        else:
            self.tokenizer = tokenizer

        # Load model (lazy - only when needed)
        self._model = model

    @property
    def model(self):
        """Lazy load nnsight model."""
        if self._model is None:
            from nnsight import LanguageModel
            self._model = LanguageModel(
                self.model_name,
                device_map=self.device,
                dtype=self.dtype,
            )
        return self._model

    def _get_prompt_for_output(
        self,
        output_data: dict[str, Any],
        dataset_path: str | None = None,
    ) -> str:
        """
        Get the original prompt for a counterfactual output.

        First checks metadata, then falls back to dataset lookup.

        Args:
            output_data: Counterfactual output dict
            dataset_path: Optional path to original dataset

        Returns:
            Original prompt string
        """
        metadata = output_data.get("metadata", {})

        # Check if prompt is in metadata
        if "prompt" in metadata:
            return metadata["prompt"]

        # Try dataset lookup
        if dataset_path:
            # Use original prompt_id from metadata (e.g., "math_0929") not the sequential one
            prompt_id = metadata.get("prompt_id") or output_data.get("prompt_id") or metadata.get("id")
            prompt = self._lookup_prompt_in_dataset(dataset_path, prompt_id)
            if prompt:
                return prompt

        # Last resort: try to infer from initial_rollout text
        # This is less reliable but better than nothing
        raise ValueError(
            f"Could not find prompt for {output_data.get('prompt_id')}. "
            f"Provide prompt in metadata or dataset_path."
        )

    def _lookup_prompt_in_dataset(
        self,
        dataset_path: str,
        prompt_id: str,
    ) -> str | None:
        """Lookup a prompt by ID in a JSONL dataset."""
        path = Path(dataset_path)
        if not path.exists():
            return None

        with open(path) as f:
            for line in f:
                item = json.loads(line)
                # Check both 'prompt_id' and 'id' fields
                item_id = item.get("prompt_id") or item.get("id")
                if item_id == prompt_id:
                    return item.get("prompt")

        return None

    def _tokenize_prompt(
        self,
        prompt: str,
    ) -> tuple[list[int], str]:
        """
        Tokenize prompt with chat template.

        Args:
            prompt: Raw prompt text

        Returns:
            Tuple of (prompt_token_ids, formatted_prompt_text)
        """
        messages = [{"role": "user", "content": prompt}]

        # Get formatted text
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        prompt_ids = self.tokenizer.encode(formatted, add_special_tokens=False)

        return prompt_ids, formatted

    def _validate_sequence(
        self,
        prompt_ids: list[int],
        generation_ids: list[int],
        full_ids: list[int],
    ) -> dict[str, Any]:
        """
        Validate that the full sequence is correctly constructed.

        Args:
            prompt_ids: Prompt token IDs
            generation_ids: Generation token IDs (from saved output)
            full_ids: Concatenated full sequence

        Returns:
            Validation result dict
        """
        expected_len = len(prompt_ids) + len(generation_ids)
        actual_len = len(full_ids)

        result = {
            "valid": True,
            "prompt_len": len(prompt_ids),
            "generation_len": len(generation_ids),
            "full_len": actual_len,
            "expected_len": expected_len,
        }

        if actual_len != expected_len:
            result["valid"] = False
            result["error"] = f"Length mismatch: {actual_len} != {expected_len}"

        # Verify concatenation
        if full_ids[:len(prompt_ids)] != prompt_ids:
            result["valid"] = False
            result["error"] = "Prompt prefix mismatch"

        if full_ids[len(prompt_ids):] != generation_ids:
            result["valid"] = False
            result["error"] = "Generation suffix mismatch"

        return result

    def extract_for_output(
        self,
        output_data: dict[str, Any],
        dataset_path: str | None = None,
        layers: list[int] | None = None,
    ) -> ActivationData:
        """
        Extract activations for a single counterfactual output.

        Args:
            output_data: Counterfactual output dict with initial_rollout and samples
            dataset_path: Optional path to dataset for prompt lookup
            layers: Specific layers to extract (None = all)

        Returns:
            ActivationData with extracted activations
        """
        # Get prompt
        prompt = self._get_prompt_for_output(output_data, dataset_path)

        # Tokenize prompt
        prompt_ids, _formatted_prompt = self._tokenize_prompt(prompt)

        # Get generation token IDs (already correctly tokenized by vLLM)
        generation_ids = output_data["initial_rollout"]["token_ids"]

        # Construct full sequence
        full_ids = prompt_ids + generation_ids

        # Validate
        validation = self._validate_sequence(prompt_ids, generation_ids, full_ids)
        if not validation["valid"]:
            raise ValueError(f"Sequence validation failed: {validation['error']}")

        # Run forward pass with token IDs directly (no text roundtrip!)
        full_ids_tensor = torch.tensor([full_ids], dtype=torch.long)

        saved_outputs = []
        with torch.no_grad(), self.model.trace(full_ids_tensor):
            if layers is None:
                for layer in self.model.model.layers:
                    saved_outputs.append(layer.output[0].save())
            else:
                for layer_idx in layers:
                    saved_outputs.append(
                        self.model.model.layers[layer_idx].output[0].save()
                    )

        # Stack layer outputs: (num_layers, seq_len, hidden_dim)
        layer_tensors = [out.squeeze(0) for out in saved_outputs]
        all_activations = torch.stack(layer_tensors, dim=0)

        # Extract only generation portion
        generation_activations = all_activations[:, len(prompt_ids):, :].cpu()

        # Validate extraction
        assert generation_activations.shape[1] == len(generation_ids), (
            f"Extraction mismatch: {generation_activations.shape[1]} != {len(generation_ids)}"
        )

        # Parse branch points
        branch_points = [
            BranchPoint(
                token_index=sample["token_index"],
                p_score=sample["p_score"]
            )
            for sample in output_data.get("samples", [])
        ]

        # Validate branch points are in range
        max_idx = max(bp.token_index for bp in branch_points) if branch_points else 0
        if max_idx >= len(generation_ids):
            raise ValueError(
                f"Branch point index {max_idx} out of range for "
                f"generation length {len(generation_ids)}"
            )

        return ActivationData(
            activations=generation_activations,
            generation_token_ids=generation_ids,
            branch_points=branch_points,
            prompt_id=output_data.get("prompt_id", "unknown"),
            metadata=output_data.get("metadata", {}),
            prompt_token_count=len(prompt_ids),
            full_sequence_length=len(full_ids),
        )

    def extract_all(
        self,
        output_dir: str,
        dataset_path: str | None = None,
        output_path: str | None = None,
        layers: list[int] | None = None,
        skip_existing: bool = True,
    ) -> list[ActivationData]:
        """
        Extract activations for all outputs in a directory.

        Args:
            output_dir: Directory containing counterfactual JSON outputs
            dataset_path: Path to original dataset (for prompt lookup)
            output_path: Optional directory to save individual activation files
            layers: Specific layers to extract
            skip_existing: Skip files that already have activations

        Returns:
            List of ActivationData objects
        """
        output_dir = Path(output_dir)
        json_files = sorted(output_dir.glob("*.json"))

        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

        results = []

        for json_file in tqdm(json_files, desc="Extracting activations"):
            # Skip if already extracted
            if output_path and skip_existing:
                act_file = output_path / f"{json_file.stem}.pt"
                if act_file.exists():
                    continue

            try:
                with open(json_file) as f:
                    output_data = json.load(f)

                activation_data = self.extract_for_output(
                    output_data,
                    dataset_path=dataset_path,
                    layers=layers,
                )

                # Save if output_path provided
                if output_path:
                    act_file = output_path / f"{json_file.stem}.pt"
                    self._save_activation_data(activation_data, act_file)
                else:
                    results.append(activation_data)

                # Free memory
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                continue

        return results

    def _save_activation_data(
        self,
        data: ActivationData,
        path: Path,
    ) -> None:
        """Save ActivationData to disk."""
        save_dict = {
            "activations": data.activations,
            "generation_token_ids": data.generation_token_ids,
            "branch_points": [
                {"token_index": bp.token_index, "p_score": bp.p_score}
                for bp in data.branch_points
            ],
            "prompt_id": data.prompt_id,
            "metadata": data.metadata,
            "prompt_token_count": data.prompt_token_count,
            "full_sequence_length": data.full_sequence_length,
        }
        torch.save(save_dict, path)


def load_activation_data(path: str | Path) -> ActivationData:
    """Load ActivationData from disk."""
    path = Path(path)
    data = torch.load(path, map_location="cpu")

    return ActivationData(
        activations=data["activations"],
        generation_token_ids=data["generation_token_ids"],
        branch_points=[
            BranchPoint(**bp) for bp in data["branch_points"]
        ],
        prompt_id=data["prompt_id"],
        metadata=data["metadata"],
        prompt_token_count=data["prompt_token_count"],
        full_sequence_length=data["full_sequence_length"],
    )


def load_all_activation_data(
    activation_dir: str | Path,
) -> list[ActivationData]:
    """Load all activation files from a directory."""
    activation_dir = Path(activation_dir)
    pt_files = sorted(activation_dir.glob("*.pt"))

    return [load_activation_data(f) for f in tqdm(pt_files, desc="Loading")]


def prepare_probe_data(
    activation_data_list: list[ActivationData],
    layer_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Prepare data for probe training from activation data.

    Args:
        activation_data_list: List of ActivationData objects
        layer_idx: Specific layer to use (None = all layers concatenated)

    Returns:
        Tuple of (X, y, metadata)
        - X: (total_branch_points, feature_dim)
        - y: (total_branch_points,) p_scores
        - metadata: List of dicts with prompt_id, token_index for each sample
    """
    all_X = []
    all_y = []
    all_meta = []

    for data in activation_data_list:
        X, y = data.get_branch_point_activations(layer_idx)
        all_X.append(X)
        all_y.append(y)

        # Track metadata
        for bp in data.branch_points:
            all_meta.append({
                "prompt_id": data.prompt_id,
                "token_index": bp.token_index,
            })

    return np.vstack(all_X), np.concatenate(all_y), all_meta
