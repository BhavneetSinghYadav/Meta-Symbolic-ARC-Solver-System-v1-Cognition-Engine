"""Utilities for generalizing and scoring symbolic rules."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
from arc_solver.src.symbolic.rule_language import rule_to_dsl
from arc_solver.src.utils import config_loader

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule, TransformationType
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.generators import (
    generate_mirror_tile_rules,
    generate_draw_line_rules,
    generate_dilate_zone_rules,
    generate_erode_zone_rules,
    generate_zone_remap_rules,
    generate_rotate_about_point_rules,
    generate_morph_remap_composites,
    generate_pattern_fill_rules,
)

OPERATOR_GENERATORS: Dict[str, Callable[[Grid, Grid], List[SymbolicRule | CompositeRule]]] = {
    "mirror_tile": generate_mirror_tile_rules,
    "draw_line": generate_draw_line_rules,
    "dilate_zone": generate_dilate_zone_rules,
    "erode_zone": generate_erode_zone_rules,
    "zone_remap": generate_zone_remap_rules,
    "rotate_about_point": generate_rotate_about_point_rules,
    "morph_remap_composites": generate_morph_remap_composites,
    "pattern_fill": generate_pattern_fill_rules,
}


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


def fallback_composite_rules(
    rules: List[SymbolicRule],
    input_grid: Grid,
    output_grid: Grid,
    *,
    score_threshold: float = 0.6,
) -> List[CompositeRule]:
    """Return composite chains when all ``rules`` score below ``score_threshold``."""

    from itertools import product
    from pathlib import Path
    import json
    from arc_solver.src.executor.scoring import score_rule, rule_cost
    from arc_solver.src.executor.simulator import simulate_composite_safe

    if not rules:
        return []

    base_scores = [score_rule(input_grid, output_grid, r) for r in rules]
    if base_scores and max(base_scores) >= score_threshold:
        return []

    candidates: List[tuple[CompositeRule, float]] = []
    for length in range(2, min(4, len(rules)) + 1):
        for steps in product(rules, repeat=length):
            comp = CompositeRule(list(steps))
            try:
                simulate_composite_safe(input_grid, comp)
            except Exception:
                continue
            score = score_rule(input_grid, output_grid, comp)
            if score <= 0:
                continue
            candidates.append((comp, score))

    seen = set()
    deduped: List[CompositeRule] = []
    for comp, score in candidates:
        signature = tuple(
            getattr(s, "dsl_str", rule_to_dsl(s)) for s in comp.steps
        ) + tuple(s.meta.get("functional") for s in comp.steps)
        if signature in seen:
            continue
        seen.add(signature)
        comp.meta["score"] = score
        deduped.append(comp)

    deduped.sort(
        key=lambda c: score_rule(input_grid, output_grid, c) - 0.05 * rule_cost(c),
        reverse=True,
    )

    try:
        path = Path("logs/fallback_trace.jsonl")
        path.parent.mkdir(exist_ok=True)
        best = deduped[0] if deduped else None
        entry = {
            "trigger_reason": "base_scores_below_threshold",
            "candidate_count": len(deduped),
            "best_summary": rule_to_dsl(best.as_symbolic_proxy()) if best else None,
            "best_score": score_rule(input_grid, output_grid, best) if best else None,
        }
        path.open("a", encoding="utf-8").write(json.dumps(entry) + "\n")
    except Exception:
        pass

    return deduped


def generate_all_rules(
    grid_in: Grid,
    grid_out: Grid,
    *,
    allowlist: List[str] | None = None,
    blocklist: List[str] | None = None,
) -> List[SymbolicRule | CompositeRule]:
    """Return candidate rules from all registered operator generators."""

    allowed = set(allowlist) if allowlist else set(OPERATOR_GENERATORS)
    blocked = set(blocklist or [])
    rules: List[SymbolicRule | CompositeRule] = []
    for name, gen in OPERATOR_GENERATORS.items():
        if name not in allowed or name in blocked:
            continue
        try:
            candidates = gen(grid_in, grid_out)
        except Exception:
            continue
        for r in candidates:
            if not r.is_well_formed():
                continue
            try:
                r.dsl_str = rule_to_dsl(r)
            except Exception:
                continue
            r.meta.setdefault("source", f"generator:{name}")
            rules.append(r)
    return rules


__all__ = [
    "generalize_rules",
    "score_rules",
    "normalize_rule_dsl",
    "remove_duplicate_rules",
    "rule_cost",
    "fallback_composite_rules",
    "generate_all_rules",
    "OPERATOR_GENERATORS",
]
