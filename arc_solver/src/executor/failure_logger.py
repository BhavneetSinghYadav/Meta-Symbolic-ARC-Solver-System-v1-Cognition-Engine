from __future__ import annotations

"""Simple JSONL failure tracing utility."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def log_failure(
    *,
    task_id: str | None,
    rule_id: str,
    rule_type: str,
    rule_steps: Iterable[str] | None,
    rejection_stage: str,
    failed_step_index: int | None,
    reason: str,
    color_lineage: Iterable[Iterable[int]] | None = None,
    intermediate_grids: Iterable[Iterable[Iterable[int]]] | None = None,
    path: str | Path = "failure_log.jsonl",
) -> None:
    """Append a failure diagnostic record as a JSON line to ``path``."""

    entry: Dict[str, Any] = {
        "task_id": task_id,
        "rule_id": rule_id,
        "rule_type": rule_type,
        "rule_steps": list(rule_steps or []),
        "rejection_stage": rejection_stage,
        "failed_step_index": failed_step_index,
        "reason": reason,
        "color_lineage": [sorted(set(s)) for s in color_lineage or []],
        "intermediate_grids": list(intermediate_grids or []),
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        json.dump(entry, f)
        f.write("\n")


__all__ = ["log_failure"]
