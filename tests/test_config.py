"""
Tests for configuration loading and validation.

The config system uses Pydantic for validation with clear error messages.
"""

import pytest
import json
from pathlib import Path


class TestConfigLoading:
    """Test loading configuration from files."""

    def test_load_valid_config(self, tmp_path, minimal_config):
        """Should load valid config file."""
        from counterfactual_probing.config import load_config

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(minimal_config, f)

        config = load_config(str(config_path))

        assert config is not None
        assert config.dataset.path == "data/prompts.jsonl"

    def test_load_nonexistent_file_raises(self):
        """Loading nonexistent file should raise clear error."""
        from counterfactual_probing.config import load_config

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.json")

    def test_load_invalid_json_raises(self, tmp_path):
        """Loading invalid JSON should raise clear error."""
        from counterfactual_probing.config import load_config

        config_path = tmp_path / "config.json"
        config_path.write_text("{ invalid json }")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config(str(config_path))


class TestRequiredFields:
    """Test validation of required fields."""

    def test_missing_dataset_raises(self, tmp_path):
        """Missing dataset section should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "model": {"name": "test-model"},
            "generation": {"temperature": 0.7},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="dataset"):
            load_config(str(config_path))

    def test_missing_dataset_path_raises(self, tmp_path):
        """Missing dataset.path should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"prompt_field": "prompt"},
            "model": {"name": "test-model"},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="path"):
            load_config(str(config_path))

    def test_missing_model_raises(self, tmp_path):
        """Missing model section should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "generation": {"temperature": 0.7},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="model"):
            load_config(str(config_path))

    def test_missing_model_name_raises(self, tmp_path):
        """Missing model.name should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="name"):
            load_config(str(config_path))


class TestDefaultValues:
    """Test default values for optional fields."""

    def test_dataset_defaults(self, tmp_path):
        """Dataset should have sensible defaults."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = load_config(str(config_path))

        assert loaded.dataset.prompt_field == "prompt"  # Default
        assert loaded.dataset.format == "jsonl"  # Default

    def test_generation_defaults(self, tmp_path):
        """Generation should have sensible defaults."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = load_config(str(config_path))

        assert loaded.generation.temperature == 0.7  # Default
        assert loaded.generation.max_tokens == 4096  # Default
        assert loaded.generation.num_counterfactuals == 50  # Default

    def test_sampling_defaults(self, tmp_path):
        """Sampling should have sensible defaults."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = load_config(str(config_path))

        assert loaded.sampling.method == "random"  # Default (pure random, no forced endpoints)
        assert loaded.sampling.num_samples == 20  # Default

    def test_output_defaults(self, tmp_path):
        """Output should have sensible defaults."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = load_config(str(config_path))

        assert loaded.output.dir == "outputs/"  # Default


class TestSamplingValidation:
    """Test validation of sampling configuration."""

    def test_invalid_sampling_method_raises(self, tmp_path):
        """Invalid sampling method should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "sampling": {"method": "invalid_method"},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="sampling.*method"):
            load_config(str(config_path))

    def test_uniform_count_requires_num_samples(self, tmp_path):
        """uniform_count method needs num_samples."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "sampling": {"method": "uniform_count"},  # no num_samples
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Should use default num_samples, not raise
        loaded = load_config(str(config_path))
        assert loaded.sampling.num_samples > 0

    def test_density_requires_density_value(self, tmp_path):
        """density method needs density value."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "sampling": {"method": "density"},  # no density value
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Should use default density, not raise
        loaded = load_config(str(config_path))
        assert 0 < loaded.sampling.density <= 1

    def test_density_out_of_range_raises(self, tmp_path):
        """Density outside [0, 1] should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "sampling": {"method": "density", "density": 1.5},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="density"):
            load_config(str(config_path))

    def test_negative_num_samples_raises(self, tmp_path):
        """Negative num_samples should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "sampling": {"method": "uniform_count", "num_samples": -5},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="num_samples"):
            load_config(str(config_path))


class TestScorerValidation:
    """Test validation of scorer configuration."""

    def test_scorer_module_required(self, tmp_path):
        """Scorer config needs module field."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "scorer": {"class": "MyScorer"},  # no module
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="module"):
            load_config(str(config_path))

    def test_scorer_class_required(self, tmp_path):
        """Scorer config needs class field."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "scorer": {"module": "my_module"},  # no class
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="class"):
            load_config(str(config_path))

    def test_scorer_config_optional(self, tmp_path):
        """Scorer config.config is optional."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "scorer": {
                "module": "counterfactual_probing.scorer.examples.dummy",
                "class": "DummyScorer",
            },  # no config
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = load_config(str(config_path))
        assert loaded.scorer.config == {}  # Empty dict default


class TestGenerationValidation:
    """Test validation of generation configuration."""

    def test_temperature_out_of_range_raises(self, tmp_path):
        """Temperature outside valid range should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "generation": {"temperature": 5.0},  # Too high
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="temperature"):
            load_config(str(config_path))

    def test_negative_max_tokens_raises(self, tmp_path):
        """Negative max_tokens should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "generation": {"max_tokens": -100},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="max_tokens"):
            load_config(str(config_path))

    def test_zero_counterfactuals_raises(self, tmp_path):
        """Zero counterfactuals should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl"},
            "model": {"name": "test-model"},
            "generation": {"num_counterfactuals": 0},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="num_counterfactuals"):
            load_config(str(config_path))


class TestDatasetFormatValidation:
    """Test validation of dataset format."""

    def test_valid_formats_accepted(self, tmp_path):
        """Valid dataset formats should be accepted."""
        from counterfactual_probing.config import load_config

        for format in ["jsonl", "json", "csv"]:
            config = {
                "dataset": {"path": "data/prompts.jsonl", "format": format},
                "model": {"name": "test-model"},
            }

            config_path = tmp_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            loaded = load_config(str(config_path))
            assert loaded.dataset.format == format

    def test_invalid_format_raises(self, tmp_path):
        """Invalid dataset format should raise error."""
        from counterfactual_probing.config import load_config

        config = {
            "dataset": {"path": "data/prompts.jsonl", "format": "xml"},
            "model": {"name": "test-model"},
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="format"):
            load_config(str(config_path))


class TestConfigDict:
    """Test config as dict operations."""

    def test_config_to_dict(self, tmp_path, minimal_config):
        """Config should be convertible to dict."""
        from counterfactual_probing.config import load_config

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(minimal_config, f)

        loaded = load_config(str(config_path))
        as_dict = loaded.model_dump()

        assert isinstance(as_dict, dict)
        assert "dataset" in as_dict
        assert "model" in as_dict

    def test_config_attribute_access(self, tmp_path, minimal_config):
        """Config fields should be accessible as attributes."""
        from counterfactual_probing.config import load_config

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(minimal_config, f)

        loaded = load_config(str(config_path))

        # Should be able to access nested fields
        assert loaded.dataset.path == "data/prompts.jsonl"
        assert loaded.model.name == "Qwen/Qwen2.5-0.5B"
        assert loaded.sampling.method == "uniform_count"
