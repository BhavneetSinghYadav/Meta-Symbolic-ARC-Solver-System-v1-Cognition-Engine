from __future__ import annotations

"""Logging utilities for simulator diagnostics."""

from typing import Dict, List
import json
from collections import Counter

rule_failures_log: List[Dict] = []


def log_rule_failure(
    rule: object,
    *,
    failure_type: str,
    skipped_due_to: object | None = None,
    message: str,
    lineage: List[str] | None = None,
    grid_snapshot: object | None = None,
    task_id: str | None = None,
) -> None:
    """Record a rule failure with optional context."""
    entry = {
        "rule": str(rule),
        "type": failure_type,
        "skipped_due_to": skipped_due_to,
        "message": message,
    }
    if lineage is not None:
        entry["lineage"] = lineage
    if grid_snapshot is not None:
        entry["grid_snapshot"] = grid_snapshot
    if task_id is not None:
        entry["task_id"] = task_id
    rule_failures_log.append(entry)


def summarize_skips_by_type() -> Dict[str, int]:
    """Return a histogram of skip messages."""
    counts = Counter(entry.get("message", "") for entry in rule_failures_log)
    return dict(counts)


def export_failures_json(path: str) -> None:
    """Dump recorded failures to ``path``."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rule_failures_log, f, indent=2)

