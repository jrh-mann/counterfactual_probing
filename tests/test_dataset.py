"""
Tests for the flexible dataset loader.

The Dataset class loads prompts from various formats with configurable
field mapping.
"""

import pytest
import json
import csv


class TestDatasetLoading:
    """Test loading datasets from various formats."""

    def test_load_jsonl_file(self, sample_jsonl_file):
        """Should load JSONL file correctly."""
        from counterfactual_probing.dataset import Dataset

        dataset = Dataset(path=sample_jsonl_file, prompt_field="prompt", format="jsonl")

        items = list(dataset)
        assert len(items) == 3
        assert items[0]["prompt"] == "What is 1+1?"

    def test_load_json_file(self, sample_json_file):
        """Should load JSON file correctly."""
        from counterfactual_probing.dataset import Dataset

        dataset = Dataset(path=sample_json_file, prompt_field="text", format="json")

        items = list(dataset)
        assert len(items) == 2
        assert items[0]["prompt"] == "Question one"

    def test_load_csv_file(self, tmp_path):
        """Should load CSV file correctly."""
        from counterfactual_probing.dataset import Dataset

        csv_path = tmp_path / "prompts.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "category"])
            writer.writeheader()
            writer.writerow({"question": "What is AI?", "category": "tech"})
            writer.writerow({"question": "What is love?", "category": "philosophy"})

        dataset = Dataset(path=str(csv_path), prompt_field="question", format="csv")

        items = list(dataset)
        assert len(items) == 2
        assert items[0]["prompt"] == "What is AI?"


class TestPromptFieldMapping:
    """Test configurable prompt field mapping."""

    def test_custom_prompt_field(self, tmp_path):
        """Should use custom prompt field name."""
        from counterfactual_probing.dataset import Dataset

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"my_custom_field": "Hello"}) + "\n")
            f.write(json.dumps({"my_custom_field": "World"}) + "\n")

        dataset = Dataset(
            path=str(jsonl_path),
            prompt_field="my_custom_field",
            format="jsonl"
        )

        items = list(dataset)
        assert items[0]["prompt"] == "Hello"
        assert items[1]["prompt"] == "World"

    def test_missing_prompt_field_raises(self, tmp_path):
        """Missing prompt field in data should raise error."""
        from counterfactual_probing.dataset import Dataset

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"wrong_field": "Hello"}) + "\n")

        dataset = Dataset(
            path=str(jsonl_path),
            prompt_field="prompt",  # Not in data
            format="jsonl"
        )

        with pytest.raises(KeyError, match="prompt"):
            list(dataset)


class TestMetadataPreservation:
    """Test that metadata is preserved."""

    def test_metadata_included(self, sample_jsonl_file):
        """Non-prompt fields should be in metadata."""
        from counterfactual_probing.dataset import Dataset

        dataset = Dataset(path=sample_jsonl_file, prompt_field="prompt", format="jsonl")

        items = list(dataset)
        assert "metadata" in items[0]
        assert items[0]["metadata"]["category"] == "math"

    def test_prompt_field_not_in_metadata(self, sample_jsonl_file):
        """Prompt field should not be duplicated in metadata."""
        from counterfactual_probing.dataset import Dataset

        dataset = Dataset(path=sample_jsonl_file, prompt_field="prompt", format="jsonl")

        items = list(dataset)
        assert "prompt" not in items[0]["metadata"]


class TestDatasetIteration:
    """Test dataset iteration behavior."""

    def test_iteration_is_repeatable(self, sample_jsonl_file):
        """Should be able to iterate multiple times."""
        from counterfactual_probing.dataset import Dataset

        dataset = Dataset(path=sample_jsonl_file, prompt_field="prompt", format="jsonl")

        items1 = list(dataset)
        items2 = list(dataset)

        assert items1 == items2

    def test_len_returns_count(self, sample_jsonl_file):
        """len(dataset) should return item count."""
        from counterfactual_probing.dataset import Dataset

        dataset = Dataset(path=sample_jsonl_file, prompt_field="prompt", format="jsonl")

        assert len(dataset) == 3

    def test_indexing_supported(self, sample_jsonl_file):
        """Should support indexing by position."""
        from counterfactual_probing.dataset import Dataset

        dataset = Dataset(path=sample_jsonl_file, prompt_field="prompt", format="jsonl")

        assert dataset[0]["prompt"] == "What is 1+1?"
        assert dataset[2]["prompt"] == "Explain gravity."


class TestEmptyDataset:
    """Test handling of empty datasets."""

    def test_empty_file(self, tmp_path):
        """Empty file should result in empty dataset."""
        from counterfactual_probing.dataset import Dataset

        empty_path = tmp_path / "empty.jsonl"
        empty_path.touch()

        dataset = Dataset(path=str(empty_path), prompt_field="prompt", format="jsonl")

        assert len(dataset) == 0
        assert list(dataset) == []


class TestFileNotFound:
    """Test handling of missing files."""

    def test_nonexistent_file_raises(self):
        """Nonexistent file should raise clear error."""
        from counterfactual_probing.dataset import Dataset

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            Dataset(path="/nonexistent/file.jsonl", prompt_field="prompt", format="jsonl")


class TestAutoFormatDetection:
    """Test automatic format detection from extension."""

    def test_auto_detect_jsonl(self, tmp_path):
        """Should auto-detect .jsonl format."""
        from counterfactual_probing.dataset import Dataset

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"prompt": "test"}) + "\n")

        dataset = Dataset(path=str(jsonl_path), prompt_field="prompt")  # No format specified

        items = list(dataset)
        assert len(items) == 1

    def test_auto_detect_json(self, tmp_path):
        """Should auto-detect .json format."""
        from counterfactual_probing.dataset import Dataset

        json_path = tmp_path / "data.json"
        with open(json_path, "w") as f:
            json.dump([{"prompt": "test"}], f)

        dataset = Dataset(path=str(json_path), prompt_field="prompt")  # No format specified

        items = list(dataset)
        assert len(items) == 1

    def test_auto_detect_csv(self, tmp_path):
        """Should auto-detect .csv format."""
        from counterfactual_probing.dataset import Dataset

        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt"])
            writer.writeheader()
            writer.writerow({"prompt": "test"})

        dataset = Dataset(path=str(csv_path), prompt_field="prompt")  # No format specified

        items = list(dataset)
        assert len(items) == 1


class TestInvalidFormat:
    """Test handling of invalid formats."""

    def test_invalid_format_raises(self, tmp_path):
        """Invalid format should raise error."""
        from counterfactual_probing.dataset import Dataset

        txt_path = tmp_path / "data.txt"
        txt_path.write_text("just some text")

        with pytest.raises(ValueError, match="Unsupported format"):
            Dataset(path=str(txt_path), prompt_field="prompt", format="txt")

    def test_unknown_extension_raises(self, tmp_path):
        """Unknown extension without format should raise error."""
        from counterfactual_probing.dataset import Dataset

        unknown_path = tmp_path / "data.xyz"
        unknown_path.write_text("some content")

        with pytest.raises(ValueError, match="Could not determine format"):
            Dataset(path=str(unknown_path), prompt_field="prompt")


class TestDatasetFromList:
    """Test creating dataset from list directly."""

    def test_from_list(self):
        """Should create dataset from list of dicts."""
        from counterfactual_probing.dataset import Dataset

        data = [
            {"prompt": "Question 1", "id": 1},
            {"prompt": "Question 2", "id": 2},
        ]

        dataset = Dataset.from_list(data, prompt_field="prompt")

        items = list(dataset)
        assert len(items) == 2
        assert items[0]["prompt"] == "Question 1"
        assert items[0]["metadata"]["id"] == 1

    def test_from_list_of_strings(self):
        """Should create dataset from list of strings."""
        from counterfactual_probing.dataset import Dataset

        prompts = ["First prompt", "Second prompt"]

        dataset = Dataset.from_strings(prompts)

        items = list(dataset)
        assert len(items) == 2
        assert items[0]["prompt"] == "First prompt"
        assert items[0]["metadata"] == {}
