"""Utilities for generalizing and scoring symbolic rules."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from arc_solver.src.symbolic.rule_language import rule_to_dsl
from arc_solver.src.utils import config_loader

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule, TransformationType
from arc_solver.src.symbolic.rule_language import CompositeRule


def generalize_rules(
    rules: List[SymbolicRule | CompositeRule],
) -> List[SymbolicRule | CompositeRule]:
    """Return a deduplicated list of rules.

    For now the generalization step simply removes duplicate rules while
    preserving order.
    """
    # Attach operator metadata for functional rules so that deduplication
    # distinguishes different operations.  ``rule_to_dsl`` does not encode
    # transformation parameters, therefore we record the functional operator
    # name in ``meta`` before deduplicating.
    enriched: List[SymbolicRule | CompositeRule] = []
    for rule in rules:
        if isinstance(rule, CompositeRule):
            for step in rule.steps:
                if step.transformation.ttype is TransformationType.FUNCTIONAL:
                    op = step.transformation.params.get("op")
                    if op:
                        step.meta.setdefault("op", op)
        else:
            if rule.transformation.ttype is TransformationType.FUNCTIONAL:
                op = rule.transformation.params.get("op")
                if op:
                    rule.meta.setdefault("op", op)
        enriched.append(rule)

    unique = remove_duplicate_rules(enriched)
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


def remove_duplicate_rules(
    rules: List[SymbolicRule | CompositeRule],
) -> List[SymbolicRule | CompositeRule]:
    """Return rule list with semantic duplicates removed."""
    seen_hashes = set()
    deduped: List[SymbolicRule | CompositeRule] = []

    def _meta_value(v: Any) -> str:
        if hasattr(v, "data"):
            return str(getattr(v, "data", v))
        return str(v)

    def _meta_signature(r: SymbolicRule | CompositeRule) -> str:
        """Return string representation of meta information for hashing."""
        def _sig(meta: Dict[str, Any]) -> str:
            return ";".join(f"{k}={_meta_value(v)}" for k, v in sorted(meta.items()))

        if isinstance(r, CompositeRule):
            parts = [_sig(getattr(r, "meta", {}))]
            parts.extend(_sig(getattr(step, "meta", {})) for step in r.steps)
            return "|".join(parts)
        return _sig(getattr(r, "meta", {}))

    for rule in rules:
        if isinstance(rule, CompositeRule):
            proxy = rule.as_symbolic_proxy()
            sig_dsl = rule_to_dsl(proxy)
        else:
            sig_dsl = rule_to_dsl(rule)
        sig = normalize_rule_dsl(sig_dsl) + _meta_signature(rule)
        h = hash(sig)
        if h not in seen_hashes:
            deduped.append(rule)
            seen_hashes.add(h)
    return deduped


def rule_cost(rule: SymbolicRule | CompositeRule) -> float:
    """Return heuristic cost of ``rule`` for sparsity ranking."""
    if isinstance(rule, CompositeRule):
        return sum(rule_cost(step) for step in rule.steps)
    from arc_solver.src.executor.scoring import _op_cost

    op_weight = float(_op_cost(rule))
    if (
        config_loader.SPARSE_MODE
        and rule.transformation.ttype is TransformationType.FUNCTIONAL
    ):
        return op_weight
    zone_str = rule.condition.get("zone", "") if rule.condition else ""
    zone_size = len(zone_str) if isinstance(zone_str, str) else len(str(zone_str))
    transform_complexity = len(rule_to_dsl(rule).split("->")[1])
    return op_weight + 0.5 * zone_size + 0.1 * transform_complexity


__all__ = [
    "generalize_rules",
    "score_rules",
    "normalize_rule_dsl",
    "remove_duplicate_rules",
    "rule_cost",
]
