"""Feature mapping utilities for symbolic rule ranking."""


from __future__ import annotations

from typing import List

from arc_solver.src.symbolic.vocabulary import SymbolicRule, SymbolType


def map_features(grid):
    """Return a feature vector."""
    return []


def rule_feature_vector(rule: SymbolicRule) -> List[float]:
    """Return simple heuristic features for ``rule``."""
    zone = 1.0 if rule.condition.get("zone") else 0.0
    colors = len({s.value for s in rule.source + rule.target if s.type is SymbolType.COLOR})
    transform = hash(rule.transformation.ttype.value) % 5
    return [zone, float(colors), float(transform)]
