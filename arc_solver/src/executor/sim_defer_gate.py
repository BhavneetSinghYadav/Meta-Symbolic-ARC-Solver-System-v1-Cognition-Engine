from __future__ import annotations

"""Pruning gate helper for simulation."""

from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.vocabulary import SymbolicRule


def is_composite(rule: SymbolicRule | CompositeRule) -> bool:
    return isinstance(rule, CompositeRule)


def allow_pruning(rule: SymbolicRule | CompositeRule) -> bool:
    """Return ``False`` for composites so pruning is deferred."""
    if is_composite(rule):
        return False
    return True


__all__ = ["is_composite", "allow_pruning"]
