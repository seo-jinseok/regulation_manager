"""
Logging configuration for Regulation RAG System.

Provides centralized logging with both console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Default log directory
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "rag.log"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure logging for the RAG system.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional path to log file (default: logs/rag.log).
        console: Whether to log to console (default: True).

    Returns:
        Root logger for the RAG system.
    """
    # Create logs directory if needed
    log_path = log_file or LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get or create logger
    logger = logging.getLogger("rag")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # File handler (always add)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "rag") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (will be prefixed with 'rag.').

    Returns:
        Logger instance.
    """
    if name == "rag":
        return logging.getLogger("rag")
    return logging.getLogger(f"rag.{name}")


# Auto-setup on import (minimal config)
_root_logger = logging.getLogger("rag")
if not _root_logger.handlers:
    # Only set up basic console logging if not already configured
    _root_logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    _root_logger.addHandler(_handler)
