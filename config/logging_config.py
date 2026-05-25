"""
Centralized logging configuration.
"""

import logging
import sys
from pathlib import Path

from config.settings import get_settings


def setup_logging(name: str | None = None) -> logging.Logger:
    """
    Configure and return a logger with file and console handlers.

    Parameters
    ----------
    name : str, optional
        Logger name. Uses root logger if None.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    settings = get_settings()
    settings.ensure_directories()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = settings.logs_dir / settings.log_file
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
