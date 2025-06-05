"""Utilities for generalizing and scoring symbolic rules."""

from __future__ import annotations

from typing import List, Tuple
from arc_solver.src.symbolic.rule_language import rule_to_dsl
from arc_solver.src.utils import config_loader

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule, TransformationType


def generalize_rules(rules: List[SymbolicRule]) -> List[SymbolicRule]:
    """Return a deduplicated list of rules.

    For now the generalization step simply removes duplicate rules while
    preserving order.
    """
    unique = remove_duplicate_rules(rules)
    if config_loader.SPARSE_MODE:
        unique.sort(key=rule_cost)
    return unique


def _coverage_for_replace(
    rule: SymbolicRule, input_grid: Grid, output_grid: Grid
) -> float:
    src_color = None
    tgt_color = None
    for sym in rule.source:
        if sym.type is rule.source[0].type and sym.type.name == "COLOR":
            src_color = int(sym.value)
            break
    for sym in rule.target:
        if sym.type.name == "COLOR":
            tgt_color = int(sym.value)
            break
    if src_color is None or tgt_color is None:
        return 0.0

    changed = 0
    explained = 0
    h, w = input_grid.shape()
    for r in range(h):
        for c in range(w):
            src = input_grid.get(r, c)
            tgt = output_grid.get(r, c)
            if src != tgt:
                changed += 1
                if src == src_color and tgt == tgt_color:
                    explained += 1
    return explained / changed if changed else 0.0


def score_rules(
    rules: List[SymbolicRule], input_grid: Grid, output_grid: Grid
) -> List[Tuple[SymbolicRule, float]]:
    """Assign a simple coverage-based score to each rule."""
    scores: List[Tuple[SymbolicRule, float]] = []
    for rule in rules:
        score = 0.0
        if rule.transformation.ttype is TransformationType.REPLACE:
            score = _coverage_for_replace(rule, input_grid, output_grid)
        scores.append((rule, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def normalize_rule_dsl(dsl: str) -> str:
    """Return DSL string normalized for deduplication."""
    return dsl.replace(" ", "").replace("zone=", "Z=").replace("color=", "C=")


def remove_duplicate_rules(rules: List[SymbolicRule]) -> List[SymbolicRule]:
    """Return rule list with semantic duplicates removed."""
    seen_hashes = set()
    deduped: List[SymbolicRule] = []
    for rule in rules:
        h = hash(normalize_rule_dsl(rule_to_dsl(rule)))
        if h not in seen_hashes:
            deduped.append(rule)
            seen_hashes.add(h)
    return deduped


def rule_cost(rule: SymbolicRule) -> float:
    """Return heuristic cost of ``rule`` for sparsity ranking."""
    zone_str = rule.condition.get("zone", "") if rule.condition else ""
    zone_size = len(zone_str) if isinstance(zone_str, str) else len(str(zone_str))
    transform_complexity = len(rule_to_dsl(rule).split("->")[1])
    return 0.5 * zone_size + transform_complexity


__all__ = [
    "generalize_rules",
    "score_rules",
    "normalize_rule_dsl",
    "remove_duplicate_rules",
    "rule_cost",
]
