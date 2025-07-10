from __future__ import annotations

"""Utility for tracking rule lineage and scoring metadata."""

from typing import Any, Dict, List, Optional


class RuleLineageTracker:
    """Track ancestry and scoring details for symbolic rules."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def add_entry(
        self,
        rule_id: str,
        parent_ids: Optional[List[str]] | None = None,
        *,
        source_task: Optional[str] = None,
        scoring_trace: Optional[Dict[str, Any]] = None,
        **metadata: Any,
    ) -> None:
        """Insert or update ``rule_id`` with lineage information."""

        entry = self._data.setdefault(rule_id, {"parents": []})
        if parent_ids is not None:
            entry["parents"] = list(parent_ids)
        if source_task is not None:
            entry["source"] = source_task
        if scoring_trace is not None:
            entry.update(scoring_trace)
        if metadata:
            entry.setdefault("meta", {}).update(metadata)

    def export(self) -> Dict[str, Dict[str, Any]]:
        """Return a JSON serialisable mapping of all tracked rules."""

        result: Dict[str, Dict[str, Any]] = {}
        for rid, entry in self._data.items():
            exported: Dict[str, Any] = {"parents": entry.get("parents", [])}
            if "source" in entry:
                exported["source"] = entry["source"]
            if "meta" in entry:
                exported["meta"] = entry["meta"]
            for k, v in entry.items():
                if k in {"parents", "source", "meta"}:
                    continue
                exported[k] = v
            result[rid] = exported
        return result
