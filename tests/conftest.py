"""Pytest configuration and shared fixtures."""

import pytest
from typing import List
import os

# Mark for skipping integration tests when model not available
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires model)"
    )


# ---------------------------------------------------------------------------
# Tokenizer fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def small_model_name() -> str:
    """The small model used for testing."""
    return "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="session")
def small_tokenizer(small_model_name):
    """Load tokenizer for tests. Session-scoped for efficiency."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_texts() -> List[str]:
    """Simple test texts of varying complexity."""
    return [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer text with multiple sentences. It contains various punctuation! And questions?",
        "Code example: def foo(): return 42",
        "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
        "",  # Empty string
        "   ",  # Whitespace only
        "Single",  # Single word
    ]


@pytest.fixture
def sample_prompts() -> List[str]:
    """Sample prompts for generation tests."""
    return [
        "Write a haiku about programming.",
        "Explain why the sky is blue in one sentence.",
        "What is 2 + 2?",
    ]


@pytest.fixture
def sample_chat_messages() -> List[dict]:
    """Sample chat messages for template tests."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
    ]


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config() -> dict:
    """Minimal valid configuration."""
    return {
        "dataset": {
            "path": "data/prompts.jsonl",
            "prompt_field": "prompt",
            "format": "jsonl",
        },
        "model": {
            "name": "Qwen/Qwen2.5-0.5B",
        },
        "generation": {
            "temperature": 0.7,
            "max_tokens": 100,
            "num_counterfactuals": 3,
        },
        "sampling": {
            "method": "uniform_count",
            "num_samples": 5,
        },
        "scorer": {
            "module": "counterfactual_probing.scorer.examples.dummy",
            "class": "DummyScorer",
        },
        "output": {
            "dir": "outputs/",
        },
    }


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_jsonl_file(tmp_path) -> str:
    """Create a sample JSONL dataset file."""
    import json

    file_path = tmp_path / "prompts.jsonl"
    prompts = [
        {"prompt": "What is 1+1?", "category": "math"},
        {"prompt": "Write a poem.", "category": "creative"},
        {"prompt": "Explain gravity.", "category": "science"},
    ]

    with open(file_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    return str(file_path)


@pytest.fixture
def sample_json_file(tmp_path) -> str:
    """Create a sample JSON dataset file."""
    import json

    file_path = tmp_path / "prompts.json"
    prompts = [
        {"text": "Question one", "id": 1},
        {"text": "Question two", "id": 2},
    ]

    with open(file_path, "w") as f:
        json.dump(prompts, f)

    return str(file_path)


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Skip integration tests if SKIP_INTEGRATION is set."""
    if os.environ.get("SKIP_INTEGRATION"):
        skip_integration = pytest.mark.skip(reason="SKIP_INTEGRATION is set")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
