"""Feature mapping utilities for symbolic rule ranking."""


from __future__ import annotations

from typing import List

from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    SymbolType,
    TransformationType,
)


def map_features(grid):
    """Return a feature vector."""
    return []


def rule_feature_vector(rule: SymbolicRule) -> List[float]:
    """Return simple heuristic features for ``rule``.

    The vector encodes:
    ``token_count``      - number of symbols in the rule
    ``zone``             - whether a zone condition is present
    ``color_usage``      - count of color tokens
    ``shape_usage``      - count of shape tokens
    ``position``         - whether a position constraint is present
    ``class_entropy``    - hashed transformation class indicator
    """

    token_count = float(len(rule.source) + len(rule.target))
    zone = 1.0 if rule.condition and rule.condition.get("zone") else 0.0
    color_usage = float(
        sum(1 for s in rule.source + rule.target if s.type is SymbolType.COLOR)
    )
    shape_usage = float(
        sum(1 for s in rule.source + rule.target if s.type is SymbolType.SHAPE)
    )
    position = 1.0 if rule.condition and rule.condition.get("position") else 0.0

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
