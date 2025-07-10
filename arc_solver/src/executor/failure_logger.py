from __future__ import annotations

"""Simple JSONL failure tracing utility."""

import json
from pathlib import Path
from typing import Any, Dict


def log_failure(entry: Dict[str, Any], path: str | Path = "logs/failure_log.jsonl") -> None:
    """Append ``entry`` as a JSON line to ``path``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        json.dump(entry, f)
        f.write("\n")


__all__ = ["log_failure"]
