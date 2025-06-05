"""Simple logging wrapper supporting optional file logging."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, file_path: str | None = None) -> logging.Logger:
    """Return configured logger, attaching ``file_path`` handler if provided."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            f_handler = logging.FileHandler(file_path, encoding="utf-8")
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)
    return logger
