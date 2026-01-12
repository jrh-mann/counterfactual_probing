"""Tests for model utilities."""

import pytest
from pathlib import Path

from counterfactual_probing.model_utils import (
    get_model_slug,
    get_model_output_dir,
    get_experiment_paths,
    ensure_experiment_dirs,
)


class TestGetModelSlug:
    """Tests for get_model_slug function."""

    def test_qwen_model(self):
        assert get_model_slug("Qwen/Qwen3-0.6B") == "qwen3-0.6b"
        assert get_model_slug("Qwen/Qwen3-4B") == "qwen3-4b"
        assert get_model_slug("Qwen/Qwen2.5-0.5B") == "qwen2.5-0.5b"

    def test_llama_model(self):
        assert get_model_slug("meta-llama/Llama-3.1-8B-Instruct") == "llama-3.1-8b-instruct"
        assert get_model_slug("meta-llama/Llama-2-7B") == "llama-2-7b"

    def test_mistral_model(self):
        assert get_model_slug("mistralai/Mistral-7B-v0.1") == "mistral-7b-v0.1"

    def test_no_org_prefix(self):
        assert get_model_slug("gpt2") == "gpt2"
        assert get_model_slug("bert-base-uncased") == "bert-base-uncased"

    def test_special_characters(self):
        assert get_model_slug("org/Model_Name") == "model-name"
        assert get_model_slug("org/Model Name") == "model-name"

    def test_empty_string(self):
        assert get_model_slug("") == ""

    def test_preserves_version_dots(self):
        # Dots in version numbers should be preserved
        assert get_model_slug("Qwen/Qwen2.5-0.5B") == "qwen2.5-0.5b"


class TestGetModelOutputDir:
    """Tests for get_model_output_dir function."""

    def test_basic_path(self):
        result = get_model_output_dir("outputs", "Qwen/Qwen3-4B")
        assert result == Path("outputs/qwen3-4b")

    def test_with_trailing_slash(self):
        result = get_model_output_dir("outputs/", "Qwen/Qwen3-4B")
        assert result == Path("outputs/qwen3-4b")


class TestGetExperimentPaths:
    """Tests for get_experiment_paths function."""

    def test_default_paths(self):
        paths = get_experiment_paths("Qwen/Qwen3-4B", "math")

        assert paths["model_slug"] == "qwen3-4b"
        assert paths["outputs_dir"] == Path("outputs/qwen3-4b/math")
        assert paths["activations_dir"] == Path("activations/qwen3-4b/math")
        assert paths["plots_dir"] == Path("plots/qwen3-4b/math")
        assert paths["probes_dir"] == Path("probes/qwen3-4b/math")

    def test_custom_base_dirs(self):
        paths = get_experiment_paths(
            "Qwen/Qwen3-4B",
            "reward_hacking",
            base_outputs="/data/outputs",
            base_activations="/data/activations",
        )

        assert paths["outputs_dir"] == Path("/data/outputs/qwen3-4b/reward_hacking")
        assert paths["activations_dir"] == Path("/data/activations/qwen3-4b/reward_hacking")

    def test_different_experiments(self):
        math_paths = get_experiment_paths("Qwen/Qwen3-4B", "math")
        rh_paths = get_experiment_paths("Qwen/Qwen3-4B", "reward_hacking")

        assert math_paths["outputs_dir"] != rh_paths["outputs_dir"]
        assert "math" in str(math_paths["outputs_dir"])
        assert "reward_hacking" in str(rh_paths["outputs_dir"])


class TestEnsureExperimentDirs:
    """Tests for ensure_experiment_dirs function."""

    def test_creates_directories(self, tmp_path):
        paths = get_experiment_paths(
            "Qwen/Qwen3-4B",
            "math",
            base_outputs=str(tmp_path / "outputs"),
            base_activations=str(tmp_path / "activations"),
            base_plots=str(tmp_path / "plots"),
            base_probes=str(tmp_path / "probes"),
        )

        # Directories shouldn't exist yet
        assert not paths["outputs_dir"].exists()

        ensure_experiment_dirs(paths)

        # Now they should exist
        assert paths["outputs_dir"].exists()
        assert paths["activations_dir"].exists()
        assert paths["plots_dir"].exists()
        assert paths["probes_dir"].exists()

    def test_idempotent(self, tmp_path):
        paths = get_experiment_paths(
            "Qwen/Qwen3-4B",
            "math",
            base_outputs=str(tmp_path / "outputs"),
        )

        # Call twice - should not raise
        ensure_experiment_dirs(paths)
        ensure_experiment_dirs(paths)

        assert paths["outputs_dir"].exists()
