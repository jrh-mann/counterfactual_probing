"""
Checkpoint management for run recovery.

Provides state persistence and recovery for long-running counterfactual probing jobs.
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .exceptions import CheckpointError
from .logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class RunCheckpoint:
    """
    Checkpoint state for a counterfactual probing run.

    Tracks progress and allows resumption of interrupted runs.
    """

    run_id: str
    config_hash: str
    model_name: str
    experiment_name: str
    total_prompts: int
    completed_prompts: list[str] = field(default_factory=list)
    failed_prompts: list[str] = field(default_factory=list)
    skipped_prompts: list[str] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pending_count(self) -> int:
        """Number of prompts still to process."""
        processed = len(self.completed_prompts) + len(self.failed_prompts) + len(self.skipped_prompts)
        return max(0, self.total_prompts - processed)

    @property
    def progress_percent(self) -> float:
        """Progress as a percentage."""
        if self.total_prompts == 0:
            return 100.0
        processed = len(self.completed_prompts) + len(self.failed_prompts)
        return (processed / self.total_prompts) * 100

    def mark_completed(self, prompt_id: str) -> None:
        """Mark a prompt as successfully completed."""
        if prompt_id not in self.completed_prompts:
            self.completed_prompts.append(prompt_id)
        self.last_update = datetime.utcnow().isoformat()

    def mark_failed(self, prompt_id: str, error: str | None = None) -> None:
        """Mark a prompt as failed."""
        if prompt_id not in self.failed_prompts:
            self.failed_prompts.append(prompt_id)
        if error:
            self.metadata.setdefault("errors", {})[prompt_id] = error
        self.last_update = datetime.utcnow().isoformat()

    def mark_skipped(self, prompt_id: str) -> None:
        """Mark a prompt as skipped (already exists)."""
        if prompt_id not in self.skipped_prompts:
            self.skipped_prompts.append(prompt_id)
        self.last_update = datetime.utcnow().isoformat()

    def is_processed(self, prompt_id: str) -> bool:
        """Check if a prompt has been processed (completed, failed, or skipped)."""
        return (
            prompt_id in self.completed_prompts
            or prompt_id in self.failed_prompts
            or prompt_id in self.skipped_prompts
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunCheckpoint":
        """Create from dictionary."""
        return cls(**data)


def compute_config_hash(config: dict[str, Any]) -> str:
    """
    Compute a hash of the configuration for change detection.

    Args:
        config: Configuration dictionary

    Returns:
        SHA256 hash of the config (first 16 chars)
    """
    # Sort keys for deterministic hashing
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class CheckpointManager:
    """
    Manages checkpoint persistence and recovery.

    Checkpoints are saved as JSON files in the output directory.
    """

    CHECKPOINT_FILENAME = "_checkpoint.json"

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / self.CHECKPOINT_FILENAME
        self._checkpoint: RunCheckpoint | None = None

    def create(
        self,
        config: dict[str, Any],
        model_name: str,
        experiment_name: str,
        total_prompts: int,
    ) -> RunCheckpoint:
        """
        Create a new checkpoint for a run.

        Args:
            config: Full configuration dictionary
            model_name: Model name
            experiment_name: Experiment name
            total_prompts: Total number of prompts

        Returns:
            New RunCheckpoint instance
        """
        self._checkpoint = RunCheckpoint(
            run_id=generate_run_id(),
            config_hash=compute_config_hash(config),
            model_name=model_name,
            experiment_name=experiment_name,
            total_prompts=total_prompts,
        )
        self.save()
        logger.info(f"Created checkpoint: {self._checkpoint.run_id}")
        return self._checkpoint

    def load(self) -> RunCheckpoint | None:
        """
        Load existing checkpoint if available.

        Returns:
            RunCheckpoint if exists, None otherwise
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)
            self._checkpoint = RunCheckpoint.from_dict(data)
            logger.info(
                f"Loaded checkpoint: {self._checkpoint.run_id} "
                f"({self._checkpoint.progress_percent:.1f}% complete)"
            )
            return self._checkpoint
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e

    def save(self) -> None:
        """Save current checkpoint to disk."""
        if self._checkpoint is None:
            raise CheckpointError("No checkpoint to save")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write via temp file
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(self._checkpoint.to_dict(), f, indent=2)
            temp_path.replace(self.checkpoint_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def get_or_create(
        self,
        config: dict[str, Any],
        model_name: str,
        experiment_name: str,
        total_prompts: int,
        force_new: bool = False,
    ) -> tuple[RunCheckpoint, bool]:
        """
        Get existing checkpoint or create new one.

        Args:
            config: Configuration dictionary
            model_name: Model name
            experiment_name: Experiment name
            total_prompts: Total prompts
            force_new: Force creation of new checkpoint

        Returns:
            Tuple of (checkpoint, is_resumed)
        """
        if not force_new:
            existing = self.load()
            if existing:
                # Verify config hasn't changed
                new_hash = compute_config_hash(config)
                if existing.config_hash != new_hash:
                    logger.warning(
                        f"Config changed (old={existing.config_hash}, new={new_hash}), "
                        "starting fresh"
                    )
                else:
                    return existing, True

        checkpoint = self.create(config, model_name, experiment_name, total_prompts)
        return checkpoint, False

    @property
    def checkpoint(self) -> RunCheckpoint | None:
        """Get current checkpoint."""
        return self._checkpoint

    def update(self) -> None:
        """Save current checkpoint state."""
        self.save()

    def delete(self) -> None:
        """Delete checkpoint file (call on successful completion)."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Deleted checkpoint (run completed)")
