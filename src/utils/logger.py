"""
Centralized logging for OpenMeet
"""
import logging
import sys
from pathlib import Path


def setup_logger(name, log_file=None, level="INFO"):
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to log file
        level: Log level string

    Returns:
        Configured logging.Logger
    """
    # Use root logger so all modules inherit handlers
    logger = logging.getLogger() if name == "root" else logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        logger.addHandler(fh)

    return logger


def get_logger(name):
    """Get an existing logger by name."""
    return logging.getLogger(name)
