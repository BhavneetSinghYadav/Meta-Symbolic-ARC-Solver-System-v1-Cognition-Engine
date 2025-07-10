from __future__ import annotations

"""Extended proxy utilities for composite rules."""

from typing import Any, Optional

from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.vocabulary import SymbolicRule, Symbol


def as_symbolic_proxy(rule: CompositeRule) -> SymbolicRule:
    """Return a proxy rule including zone information and extent."""
    cond: dict[str, Any] = rule.get_condition() or {}
    zones = {
        step.condition.get("zone")
        for step in rule.steps
        if getattr(step, "condition", None) and "zone" in step.condition
    }
    if len(zones) == 1:
        cond = {**cond, "zone": next(iter(zones))}
    proxy = SymbolicRule(
        transformation=rule.transformation,
        source=rule.steps[0].source,
        target=rule.final_targets(),
        condition=cond,
        nature=rule.nature,
    )
    proxy.meta["input_zones"] = list(zones)
    proxy.meta["step_count"] = len(rule.steps)
    return proxy


__all__ = ["as_symbolic_proxy"]
