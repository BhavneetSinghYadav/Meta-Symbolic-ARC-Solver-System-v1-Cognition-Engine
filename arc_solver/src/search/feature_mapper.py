"""Feature mapping utilities for symbolic rule ranking."""


from __future__ import annotations

from typing import List

from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    SymbolType,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule


def map_features(grid):
    """Return a feature vector."""
    return []


def rule_feature_vector(rule: SymbolicRule | CompositeRule) -> List[float]:
    """Return simple heuristic features for ``rule``.

    The vector encodes:
    ``token_count``      - number of symbols in the rule
    ``zone``             - whether a zone condition is present
    ``color_usage``      - count of color tokens
    ``shape_usage``      - count of shape tokens
    ``position``         - whether a position constraint is present
    ``class_entropy``    - hashed transformation class indicator
    """

    if isinstance(rule, CompositeRule):
        source_syms = rule.get_sources()
        target_syms = rule.get_targets()
        cond = rule.get_condition() or {}
    else:
        source_syms = rule.source
        target_syms = rule.target
        cond = rule.condition

    token_count = float(len(source_syms) + len(target_syms))
    zone = 1.0 if cond and cond.get("zone") else 0.0
    color_usage = float(
        sum(1 for s in source_syms + target_syms if s.type is SymbolType.COLOR)
    )
    shape_usage = float(
        sum(1 for s in source_syms + target_syms if s.type is SymbolType.SHAPE)
    )
    position = 1.0 if cond and cond.get("position") else 0.0

    # Simple hashed representation of the transformation type. This is not a
    # true entropy measure but provides diversity across classes without
    # requiring explicit one-hot encoding.
    if isinstance(rule.transformation.ttype, TransformationType):
        class_entropy = float(hash(rule.transformation.ttype.value) % 7) / 7.0
    else:  # pragma: no cover - defensive
        class_entropy = 0.0

    return [
        token_count,
        zone,
        color_usage,
        shape_usage,
        position,
        class_entropy,
    ]
