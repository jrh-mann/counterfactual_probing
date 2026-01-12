"""
End-to-end tests for the full pipeline.

These tests run the complete pipeline with a small model to verify
everything works together.
"""

import pytest
import json
from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Test the complete pipeline end-to-end."""

    def test_full_pipeline_produces_output(
        self, tmp_path, small_model_name, small_tokenizer
    ):
        """Run complete pipeline and verify output is produced."""
        from counterfactual_probing import run

        # Create config
        config = {
            "dataset": {
                "path": str(tmp_path / "prompts.jsonl"),
                "prompt_field": "prompt",
                "format": "jsonl",
            },
            "model": {
                "name": small_model_name,
            },
            "generation": {
                "temperature": 0.5,
                "max_tokens": 50,
                "num_counterfactuals": 3,
            },
            "sampling": {
                "method": "uniform_count",
                "num_samples": 3,
            },
            "scorer": {
                "module": "counterfactual_probing.scorer.examples.dummy",
                "class": "DummyScorer",
            },
            "output": {
                "dir": str(tmp_path / "outputs"),
            },
        }

        # Create dataset
        prompts_path = tmp_path / "prompts.jsonl"
        with open(prompts_path, "w") as f:
            f.write(json.dumps({"prompt": "What is 2+2?"}) + "\n")

        # Create config file
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Run pipeline
        run(config_path=str(config_path))

        # Verify output exists
        output_dir = tmp_path / "outputs"
        assert output_dir.exists()

        # Should have output files
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) > 0


@pytest.mark.integration
class TestOutputFormat:
    """Test output format correctness."""

    def test_output_has_required_fields(self, tmp_path, small_model_name):
        """Output should have all required fields."""
        from counterfactual_probing import run

        # Setup minimal test
        config = create_minimal_config(tmp_path, small_model_name)
        create_single_prompt_dataset(tmp_path)

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        run(config_path=str(config_path))

        # Read output
        output_files = list((tmp_path / "outputs").glob("*.json"))
        assert len(output_files) > 0

        with open(output_files[0]) as f:
            output = json.load(f)

        # Verify structure
        assert "prompt_id" in output or "metadata" in output
        assert "initial_rollout" in output
        assert "samples" in output

    def test_output_token_ids_present(self, tmp_path, small_model_name):
        """Output should contain token IDs, not just text."""
        from counterfactual_probing import run

        config = create_minimal_config(tmp_path, small_model_name)
        create_single_prompt_dataset(tmp_path)

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        run(config_path=str(config_path))

        output_files = list((tmp_path / "outputs").glob("*.json"))
        with open(output_files[0]) as f:
            output = json.load(f)

        # Initial rollout should have token_ids
        assert "token_ids" in output["initial_rollout"]
        assert isinstance(output["initial_rollout"]["token_ids"], list)

        # Samples should have token_ids
        for sample in output["samples"]:
            assert "token_index" in sample
            assert "prefix_token_ids" in sample
            for cf in sample["counterfactuals"]:
                assert "token_ids" in cf

    def test_output_token_ids_decodable(
        self, tmp_path, small_model_name, small_tokenizer
    ):
        """All token IDs in output should be decodable."""
        from counterfactual_probing import run

        config = create_minimal_config(tmp_path, small_model_name)
        create_single_prompt_dataset(tmp_path)

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        run(config_path=str(config_path))

        output_files = list((tmp_path / "outputs").glob("*.json"))
        with open(output_files[0]) as f:
            output = json.load(f)

        # Decode initial rollout
        initial_ids = output["initial_rollout"]["token_ids"]
        text = small_tokenizer.decode(initial_ids)
        assert isinstance(text, str)

        # Decode each counterfactual
        for sample in output["samples"]:
            prefix_ids = sample["prefix_token_ids"]
            small_tokenizer.decode(prefix_ids)  # Should not raise

            for cf in sample["counterfactuals"]:
                cf_ids = cf["token_ids"]
                small_tokenizer.decode(cf_ids)  # Should not raise


@pytest.mark.integration
class TestScoreCalculation:
    """Test that scores are calculated correctly."""

    def test_p_score_is_mean(self, tmp_path, small_model_name):
        """p_score should equal mean of counterfactual scores."""
        from counterfactual_probing import run

        # Use a scorer that returns predictable values
        config = create_minimal_config(tmp_path, small_model_name)
        config["scorer"] = {
            "module": "counterfactual_probing.scorer.examples.dummy",
            "class": "DummyScorer",
            "config": {"return_value": 0.5},
        }

        create_single_prompt_dataset(tmp_path)

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        run(config_path=str(config_path))

        output_files = list((tmp_path / "outputs").glob("*.json"))
        with open(output_files[0]) as f:
            output = json.load(f)

        for sample in output["samples"]:
            scores = [cf["score"] for cf in sample["counterfactuals"]]
            expected_p_score = sum(scores) / len(scores)
            assert abs(sample["p_score"] - expected_p_score) < 1e-6


@pytest.mark.integration
class TestMultiplePrompts:
    """Test processing multiple prompts."""

    def test_processes_all_prompts(self, tmp_path, small_model_name):
        """Should process all prompts in dataset."""
        from counterfactual_probing import run

        config = create_minimal_config(tmp_path, small_model_name)

        # Create dataset with multiple prompts
        prompts_path = tmp_path / "prompts.jsonl"
        with open(prompts_path, "w") as f:
            for i in range(3):
                f.write(json.dumps({"prompt": f"Question {i}"}) + "\n")

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        run(config_path=str(config_path))

        # Should have output for each prompt
        output_files = list((tmp_path / "outputs").glob("*.json"))
        assert len(output_files) == 3


# Helper functions

def create_minimal_config(tmp_path: Path, model_name: str) -> dict:
    """Create minimal config for testing."""
    return {
        "dataset": {
            "path": str(tmp_path / "prompts.jsonl"),
            "prompt_field": "prompt",
            "format": "jsonl",
        },
        "model": {
            "name": model_name,
        },
        "generation": {
            "temperature": 0.5,
            "max_tokens": 30,
            "num_counterfactuals": 2,
        },
        "sampling": {
            "method": "uniform_count",
            "num_samples": 2,
        },
        "scorer": {
            "module": "counterfactual_probing.scorer.examples.dummy",
            "class": "DummyScorer",
        },
        "output": {
            "dir": str(tmp_path / "outputs"),
        },
    }


def create_single_prompt_dataset(tmp_path: Path):
    """Create a single-prompt dataset for testing."""
    prompts_path = tmp_path / "prompts.jsonl"
    with open(prompts_path, "w") as f:
        f.write(json.dumps({"prompt": "What is 1+1?"}) + "\n")


@pytest.mark.integration
class TestResumeCapability:
    """Test ability to resume interrupted runs."""

    def test_skip_existing_outputs(self, tmp_path, small_model_name):
        """Should skip prompts with existing outputs."""
        from counterfactual_probing import run

        config = create_minimal_config(tmp_path, small_model_name)
        config["skip_existing"] = True

        # Create dataset with 2 prompts
        prompts_path = tmp_path / "prompts.jsonl"
        with open(prompts_path, "w") as f:
            f.write(json.dumps({"prompt": "Question 1", "id": "q1"}) + "\n")
            f.write(json.dumps({"prompt": "Question 2", "id": "q2"}) + "\n")

        # Pre-create output for first prompt
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        (output_dir / "q1.json").write_text("{}")

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        run(config_path=str(config_path))

        # First output should be unchanged (empty)
        q1_output = (output_dir / "q1.json").read_text()
        assert q1_output == "{}"

        # Second output should be created
        assert (output_dir / "q2.json").exists()


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in pipeline."""

    def test_invalid_config_fails_fast(self, tmp_path):
        """Invalid config should fail immediately with clear error."""
        from counterfactual_probing import run

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"invalid": "config"}, f)

        with pytest.raises(ValueError):
            run(config_path=str(config_path))

    def test_missing_dataset_fails_clearly(self, tmp_path, small_model_name):
        """Missing dataset file should fail with clear error."""
        from counterfactual_probing import run

        config = create_minimal_config(tmp_path, small_model_name)
        # Don't create the dataset file

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(FileNotFoundError, match="Dataset"):
            run(config_path=str(config_path))
