from __future__ import annotations

"""Cache of failed rule programs to avoid repetition."""

from typing import Dict, List

from arc_solver.src.symbolic.vocabulary import SymbolicRule


class PolicyCache:
    def __init__(self) -> None:
        self._failed: Dict[str, List[List[SymbolicRule]]] = {}

    def add_failure(self, task_hash: str, ruleset: List[SymbolicRule]) -> None:
        self._failed.setdefault(task_hash, []).append(list(ruleset))

    def is_failed(self, task_hash: str, ruleset: List[SymbolicRule]) -> bool:
        failed = self._failed.get(task_hash, [])
        for rs in failed:
            if len(rs) == len(ruleset) and all(a == b for a, b in zip(rs, ruleset)):
                return True
        return False
