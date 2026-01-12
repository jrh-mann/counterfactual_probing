"""
Structured logging configuration for counterfactual probing.

Provides consistent logging across all modules with optional JSON output.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, "run_id"):
            log_data["run_id"] = record.run_id
        if hasattr(record, "prompt_id"):
            log_data["prompt_id"] = record.prompt_id
        if hasattr(record, "model"):
            log_data["model"] = record.model

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console output for better readability."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | Path | None = None,
    run_id: str | None = None,
) -> logging.Logger:
    """
    Configure logging for counterfactual probing.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON format for structured logging
        log_file: Optional file path to write logs
        run_id: Optional run ID to include in all log messages

    Returns:
        Configured root logger for the package
    """
    logger = logging.getLogger("counterfactual_probing")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)

    if json_output:
        console_handler.setFormatter(JSONFormatter())
    else:
        # Check if terminal supports colors
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            console_handler.setFormatter(ColoredFormatter(fmt, datefmt="%H:%M:%S"))
        else:
            fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            console_handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    logger.addHandler(console_handler)

    # File handler (always JSON for parseability)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    # Store run_id for use in log records
    if run_id:
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.run_id = run_id
            return record

        logging.setLogRecordFactory(record_factory)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"counterfactual_probing.{name}")


class LogContext:
    """Context manager for adding extra fields to log messages."""

    def __init__(self, logger: logging.Logger, **extra: Any):
        self.logger = logger
        self.extra = extra
        self._old_factory = None

    def __enter__(self):
        self._old_factory = logging.getLogRecordFactory()
        extra = self.extra

        def factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            for key, value in extra.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(factory)
        return self

    def __exit__(self, *args):
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
