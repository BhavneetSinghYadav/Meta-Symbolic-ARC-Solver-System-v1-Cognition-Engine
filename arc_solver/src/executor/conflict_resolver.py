from __future__ import annotations

"""Heuristics for resolving conflicts between symbolic rules."""

from typing import Dict, List, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolType, SymbolicRule, TransformationType


def _rule_key(rule: SymbolicRule) -> Tuple[str, str]:
    """Return (ttype, src_color) key used for grouping conflicts."""
    if rule.transformation.ttype is not TransformationType.REPLACE:
        return (rule.transformation.ttype.value, "")
    color = next((s.value for s in rule.source if s.type is SymbolType.COLOR), "")
    return ("REPLACE", color)


def _rule_complexity(rule: SymbolicRule) -> int:
    return len(rule.source) + len(rule.target)


def resolve_conflicts(rules: List[SymbolicRule], input_grid: Grid) -> List[SymbolicRule]:
    """Return a new rule list with contradictions removed."""
    groups: Dict[Tuple[str, str], List[SymbolicRule]] = {}
    for rule in rules:
        groups.setdefault(_rule_key(rule), []).append(rule)

    resolved: List[SymbolicRule] = []
    for key, group in groups.items():
        if len(group) == 1 or key[0] != "REPLACE":
            resolved.append(group[0])
            continue

        # choose rule with highest coverage (count of matching source cells)
        src_color = int(key[1]) if key[1] else None
        if src_color is not None:
            coverage = sum(row.count(src_color) for row in input_grid.data)
        else:
            coverage = 0
        group.sort(key=lambda r: (-coverage, _rule_complexity(r)))
        resolved.append(group[0])
    return resolved
