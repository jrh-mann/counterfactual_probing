"""
Flexible dataset loader with configurable schema.

Supports loading prompts from JSONL, JSON, and CSV formats with
user-defined field mapping.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any


class Dataset:
    """
    Flexible dataset loader with configurable schema.

    Supports multiple file formats and allows specifying which field
    contains the prompt text.
    """

    def __init__(
        self,
        path: str,
        prompt_field: str = "prompt",
        format: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            path: Path to the dataset file
            prompt_field: Name of the field containing the prompt text
            format: File format ("jsonl", "json", "csv"). If None, auto-detected.

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If format is unsupported or cannot be determined
        """
        self.path = Path(path)
        self.prompt_field = prompt_field

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Auto-detect format from extension if not specified
        if format is None:
            format = self._detect_format()

        if format not in ("jsonl", "json", "csv"):
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Must be 'jsonl', 'json', or 'csv'"
            )

        self.format = format
        self._items: List[Dict[str, Any]] = []
        self._loaded = False

    def _detect_format(self) -> str:
        """Detect format from file extension."""
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            return "jsonl"
        elif suffix == ".json":
            return "json"
        elif suffix == ".csv":
            return "csv"
        else:
            raise ValueError(
                f"Could not determine format from extension '{suffix}'. "
                f"Please specify format explicitly."
            )

    def _load(self) -> None:
        """Load data from file."""
        if self._loaded:
            return

        if self.format == "jsonl":
            self._items = self._load_jsonl()
        elif self.format == "json":
            self._items = self._load_json()
        elif self.format == "csv":
            self._items = self._load_csv()

        self._loaded = True

    def _load_jsonl(self) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        items = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    items.append(json.loads(line))
        return items

    def _load_json(self) -> List[Dict[str, Any]]:
        """Load JSON file (expected to be a list)."""
        with open(self.path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        return data

    def _load_csv(self) -> List[Dict[str, Any]]:
        """Load CSV file."""
        items = []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(dict(row))
        return items

    def _transform_item(self, raw_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a raw item into the standard format.

        Args:
            raw_item: Raw item from the dataset

        Returns:
            Transformed item with 'prompt' and 'metadata' keys

        Raises:
            KeyError: If prompt_field is missing from the item
        """
        if self.prompt_field not in raw_item:
            raise KeyError(
                f"Field '{self.prompt_field}' not found in dataset item. "
                f"Available fields: {list(raw_item.keys())}"
            )

        prompt = raw_item[self.prompt_field]
        metadata = {k: v for k, v in raw_item.items() if k != self.prompt_field}

        return {
            "prompt": prompt,
            "metadata": metadata,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset items."""
        self._load()
        for item in self._items:
            yield self._transform_item(item)

    def __len__(self) -> int:
        """Return number of items in dataset."""
        self._load()
        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item by index."""
        self._load()
        return self._transform_item(self._items[index])

    @classmethod
    def from_list(
        cls,
        data: List[Dict[str, Any]],
        prompt_field: str = "prompt",
    ) -> "Dataset":
        """
        Create a dataset from a list of dictionaries.

        Args:
            data: List of dictionaries containing prompt data
            prompt_field: Name of the field containing the prompt

        Returns:
            Dataset instance
        """
        instance = cls.__new__(cls)
        instance.path = None
        instance.prompt_field = prompt_field
        instance.format = "memory"
        instance._items = data
        instance._loaded = True
        return instance

    @classmethod
    def from_strings(cls, prompts: List[str]) -> "Dataset":
        """
        Create a dataset from a list of prompt strings.

        Args:
            prompts: List of prompt strings

        Returns:
            Dataset instance
        """
        data = [{"prompt": p} for p in prompts]
        return cls.from_list(data, prompt_field="prompt")
