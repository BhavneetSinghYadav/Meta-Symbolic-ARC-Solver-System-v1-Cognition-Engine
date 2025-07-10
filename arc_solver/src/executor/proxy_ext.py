from __future__ import annotations

"""Extended proxy utilities for composite rules."""

from typing import Any

from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.vocabulary import SymbolicRule, Symbol


def merge_zones(steps) -> list[str]:
    """Return sorted unique zones present in ``steps``.

    Zones may be specified in ``condition['zone']`` or in ``meta`` under
    ``input_zones`` or ``output_zones``.  Both input and output zones are
    merged into a single list as dependency ordering only cares about the
    overall spatial scope of the composite rule.
    """

    merged: set[str] = set()
    for step in steps:
        cond = getattr(step, "condition", None) or {}
        zone = cond.get("zone")
        if zone:
            if isinstance(zone, str):
                merged.add(zone)
            else:
                merged.update(zone)
        meta = getattr(step, "meta", {})
        for key in ("input_zones", "output_zones"):
            val = meta.get(key)
            if not val:
                continue
            if isinstance(val, str):
                merged.add(val)
            else:
                merged.update(val)
    return sorted(merged)


def as_symbolic_proxy(rule: CompositeRule) -> SymbolicRule:
    """Return a proxy rule describing ``rule`` for dependency sorting."""

    cond: dict[str, Any] = rule.get_condition() or {}
    merged_zones = merge_zones(rule.steps)
    if len(merged_zones) == 1:
        cond = {**cond, "zone": merged_zones[0]}

    last_step = rule.steps[-1]
    proxy = SymbolicRule(
        transformation=last_step.transformation,
        source=rule.steps[0].source,
        target=rule.final_targets(),
        condition=cond,
        nature=rule.nature,
    )

    proxy.meta["input_zones"] = merged_zones
    proxy.meta["output_zones"] = merged_zones
    proxy.meta["step_count"] = len(rule.steps)
    return proxy


__all__ = ["as_symbolic_proxy"]
