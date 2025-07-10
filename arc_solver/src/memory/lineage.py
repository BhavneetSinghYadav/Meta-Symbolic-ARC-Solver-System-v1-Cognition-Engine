from __future__ import annotations

"""Utility for tracking rule lineage and scoring metadata."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional


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


class LineageTracker:
    """Record step-by-step lineage information for rule derivations."""

    def __init__(self) -> None:
        self._lineages: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_step(self, rule_id: str, description: str, grid_state: Any) -> None:
        """Append ``description`` and ``grid_state`` snapshot to ``rule_id`` lineage."""

        if hasattr(grid_state, "to_list"):
            grid = grid_state.to_list()
        else:
            grid = grid_state
        self._lineages[rule_id].append({"description": description, "grid": grid})

    def get_lineage(self, rule_id: str) -> List[Dict[str, Any]]:
        """Return the tracked lineage steps for ``rule_id``."""

        return list(self._lineages.get(rule_id, []))

    def to_json(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return a JSON serialisable view of all tracked lineages."""

        return {rid: list(steps) for rid, steps in self._lineages.items()}

    def dump_json(self, path: str | Path) -> None:
        """Write all stored lineages to ``path`` in JSON format."""

        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def from_json(cls, path: str | Path) -> "LineageTracker":
        """Create a tracker from data stored at ``path``."""

        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        tracker = cls()
        for rule_id, steps in data.items():
            tracker._lineages[rule_id] = list(steps)
        return tracker
