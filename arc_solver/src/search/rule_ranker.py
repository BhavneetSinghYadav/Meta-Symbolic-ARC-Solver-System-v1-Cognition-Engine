from __future__ import annotations

"""Simple heuristics for ranking sets of symbolic rules."""

from typing import List

from arc_solver.src.memory.policy_cache import PolicyCache
from arc_solver.src.symbolic.vocabulary import SymbolicRule


def rank_rule_sets(
    rule_sets: List[List[SymbolicRule]],
    policy_cache: PolicyCache | None = None,
    task_hash: str | None = None,
) -> List[List[SymbolicRule]]:
    """Return ``rule_sets`` sorted by heuristic score."""

    def score(rs: List[SymbolicRule]) -> tuple:
        coverage = len(rs)
        diversity = len({r.transformation.ttype for r in rs})
        abstractness = -sum(len(r.source) + len(r.target) for r in rs)
        reuse = 0
        if policy_cache and task_hash and policy_cache.is_failed(task_hash, rs):
            reuse = -1
        return (reuse, coverage, diversity, abstractness)

    return sorted(rule_sets, key=score, reverse=True)
