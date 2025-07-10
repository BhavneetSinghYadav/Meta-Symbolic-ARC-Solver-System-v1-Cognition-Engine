from __future__ import annotations

"""Symbolic mutation utilities for rule exploration."""

from typing import List

from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule

__all__ = ["mutate_rule"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clone_symbolic(rule: SymbolicRule) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(
            rule.transformation.ttype, rule.transformation.params.copy()
        ),
        source=rule.source[:],
        target=rule.target[:],
        nature=rule.nature,
        condition=rule.condition.copy(),
        meta=rule.meta.copy(),
    )


def _clone_composite(rule: CompositeRule) -> CompositeRule:
    return CompositeRule([
        _clone_symbolic(s) for s in rule.steps
    ], nature=rule.nature, meta=rule.meta.copy())


# ---------------------------------------------------------------------------
# Mutation primitives
# ---------------------------------------------------------------------------

def _perturb_translation(rule: SymbolicRule | CompositeRule) -> List[SymbolicRule | CompositeRule]:
    """Return variants with slightly adjusted translation offsets."""

    variants: List[SymbolicRule | CompositeRule] = []
    if isinstance(rule, CompositeRule):
        for idx, step in enumerate(rule.steps):
            for v in _perturb_translation(step):
                steps = rule.steps[:]
                steps[idx] = v if isinstance(v, SymbolicRule) else v.steps[0]
                if isinstance(v, CompositeRule):
                    # flatten nested composite
                    new_steps = steps[:idx] + v.steps + steps[idx + 1 :]
                else:
                    new_steps = steps
                variants.append(
                    CompositeRule(new_steps, nature=rule.nature, meta=rule.meta.copy())
                )
        return variants

    if rule.transformation.ttype is TransformationType.TRANSLATE:
        params = rule.transformation.params
        dx = int(params.get("dx", "0"))
        dy = int(params.get("dy", "0"))
        offsets = {(dx + 1, dy), (dx - 1, dy), (dx, dy + 1), (dx, dy - 1)}
        for ndx, ndy in offsets:
            new_params = {**params, "dx": str(ndx), "dy": str(ndy)}
            variants.append(
                SymbolicRule(
                    transformation=Transformation(
                        TransformationType.TRANSLATE, new_params
                    ),
                    source=rule.source[:],
                    target=rule.target[:],
                    nature=rule.nature,
                    condition=rule.condition.copy(),
                    meta=rule.meta.copy(),
                )
            )
    return variants


def _swap_colours(rule: SymbolicRule | CompositeRule) -> List[SymbolicRule | CompositeRule]:
    """Return variants with source/target colours swapped."""

    variants: List[SymbolicRule | CompositeRule] = []
    if isinstance(rule, CompositeRule):
        for idx, step in enumerate(rule.steps):
            for v in _swap_colours(step):
                steps = rule.steps[:]
                if isinstance(v, CompositeRule):
                    new_steps = steps[:idx] + v.steps + steps[idx + 1 :]
                else:
                    steps[idx] = v
                    new_steps = steps
                variants.append(
                    CompositeRule(new_steps, nature=rule.nature, meta=rule.meta.copy())
                )
        return variants

    if (
        rule.transformation.ttype is TransformationType.REPLACE
        and rule.source
        and rule.target
        and rule.source[0].type is SymbolType.COLOR
        and rule.target[0].type is SymbolType.COLOR
    ):
        src = rule.source[0]
        tgt = rule.target[0]
        variants.append(
            SymbolicRule(
                transformation=Transformation(
                    TransformationType.REPLACE, rule.transformation.params.copy()
                ),
                source=[tgt],
                target=[src],
                nature=rule.nature,
                condition=rule.condition.copy(),
                meta=rule.meta.copy(),
            )
        )
    return variants


def _reorder_composite_steps(rule: CompositeRule) -> List[CompositeRule]:
    """Return variants with step order reversed."""

    if not isinstance(rule, CompositeRule) or len(rule.steps) < 2:
        return []

    reversed_steps = [_clone_symbolic(s) for s in reversed(rule.steps)]
    return [CompositeRule(reversed_steps, nature=rule.nature, meta=rule.meta.copy())]


def _inject_noop(rule: SymbolicRule | CompositeRule) -> List[CompositeRule]:
    """Return variants with a no-op ``SHAPE_ABSTRACT`` step injected."""

    noop = SymbolicRule(
        transformation=Transformation(TransformationType.SHAPE_ABSTRACT),
        source=[Symbol(SymbolType.SHAPE, "A")],
        target=[Symbol(SymbolType.SHAPE, "A")],
    )
    variants: List[CompositeRule] = []
    if isinstance(rule, CompositeRule):
        steps = [_clone_symbolic(s) for s in rule.steps]
        for i in range(len(steps) + 1):
            new_steps = steps[:i] + [noop] + steps[i:]
            variants.append(
                CompositeRule(new_steps, nature=rule.nature, meta=rule.meta.copy())
            )
    else:
        variants.append(CompositeRule([_clone_symbolic(rule), noop]))
        variants.append(CompositeRule([noop, _clone_symbolic(rule)]))
    return variants


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mutate_rule(rule: SymbolicRule | CompositeRule) -> List[SymbolicRule | CompositeRule]:
    """Return a list of symbolic variants derived from ``rule``."""

    candidates: List[SymbolicRule | CompositeRule] = []
    try:
        candidates.extend(_perturb_translation(rule))
    except Exception:
        pass
    try:
        candidates.extend(_swap_colours(rule))
    except Exception:
        pass
    if isinstance(rule, CompositeRule):
        try:
            candidates.extend(_reorder_composite_steps(rule))
        except Exception:
            pass
    try:
        candidates.extend(_inject_noop(rule))
    except Exception:
        pass
    unique: List[SymbolicRule | CompositeRule] = []
    seen = set()
    for cand in candidates:
        key = repr(cand)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique
