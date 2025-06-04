"""Heuristics for detecting and resolving symbolic rule conflicts."""

from __future__ import annotations

from typing import Dict, List, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.vocabulary import SymbolType, SymbolicRule

Location = Tuple[int, int]
Conflict = Tuple[Location, List[SymbolicRule]]


def detect_conflicts(rule_set: List[SymbolicRule], grid: Grid) -> List[Conflict]:
    """Return a list of conflicting rule groups.

    Each conflict is represented as ``((row, col), [rules])`` where more than one
    rule attempts to write different values to the same cell.
    """
    h, w = grid.shape()
    writes: Dict[Location, Dict[int, List[SymbolicRule]]] = {}

    for rule in rule_set:
        predicted = simulate_rules(grid, [rule])
        for r in range(h):
            for c in range(w):
                new_val = predicted.get(r, c)
                if new_val == grid.get(r, c):
                    continue
                cell = (r, c)
                writes.setdefault(cell, {}).setdefault(new_val, []).append(rule)

    conflicts: List[Conflict] = []
    for loc, by_val in writes.items():
        if len(by_val) > 1:
            rules: List[SymbolicRule] = []
            for lst in by_val.values():
                rules.extend(lst)
            conflicts.append((loc, rules))
    return conflicts


def _specificity(rule: SymbolicRule) -> int:
    """Return a simple specificity score for a rule."""
    score = len(rule.condition)
    for sym in rule.source:
        if sym.type in (SymbolType.ZONE, SymbolType.REGION):
            score += 1
    return score


def resolve_conflicts(
    conflicts: List[Conflict],
    rule_set: List[SymbolicRule] | None = None,
    strategy: str = "prioritize-conditional",
) -> List[SymbolicRule]:
    """Resolve conflicts according to ``strategy`` and return surviving rules."""

    if not conflicts:
        return rule_set[:] if rule_set is not None else []

    to_remove: List[SymbolicRule] = []
    to_keep: List[SymbolicRule] = []

    for _loc, rules in conflicts:
        if strategy == "weighted":
            rules = sorted(rules, key=_specificity, reverse=True)
        else:  # prioritize-conditional or fallback
            rules = sorted(rules, key=_specificity, reverse=True)

        winner = rules[0]
        if winner not in to_keep:
            to_keep.append(winner)
        if strategy != "fallback":
            for r in rules[1:]:
                if r not in to_remove:
                    to_remove.append(r)

    resolved: List[SymbolicRule] = []
    for conf in conflicts:
        for rule in conf[1]:
            if rule in to_remove:
                continue
            if rule not in resolved:
                resolved.append(rule)
    for rule in to_keep:
        if rule not in resolved:
            resolved.append(rule)
    return resolved


def apply_rules_with_resolution(
    grid: Grid, rule_set: List[SymbolicRule], strategy: str = "prioritize-conditional"
) -> Grid:
    """Apply ``rule_set`` on ``grid`` after resolving internal conflicts."""
    conflicts = detect_conflicts(rule_set, grid)
    rules = resolve_conflicts(conflicts, rule_set, strategy=strategy)
    return simulate_rules(grid, rules)


__all__ = ["detect_conflicts", "resolve_conflicts", "apply_rules_with_resolution"]
