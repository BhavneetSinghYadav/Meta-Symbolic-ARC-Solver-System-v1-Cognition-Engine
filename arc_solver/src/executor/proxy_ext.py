from __future__ import annotations

"""Extended proxy utilities for composite rules."""

from typing import Any, List, Tuple

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
    """Return a proxy rule describing ``rule`` for dependency sorting.

    The proxy exposes aggregated zone metadata alongside a ``zone_chain``
    describing the input/output zone transition of each step.  ``zone_chain``
    is used by :func:`sort_rules_by_topology` to build a dependency graph that
    respects how composites move data across zones over time.
    """

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

    zone_chain: List[Tuple[str | None, str | None]] = []
    for step in rule.steps:
        meta = getattr(step, "meta", {})
        in_val = meta.get("input_zones")
        out_val = meta.get("output_zones")

        def _first(val):
            if not val:
                return None
            if isinstance(val, str):
                return val
            return val[0] if len(val) == 1 else None

        in_zone = _first(in_val)
        out_zone = _first(out_val)

        if in_zone is None or out_zone is None:
            zone = (getattr(step, "condition", None) or {}).get("zone")
            if isinstance(zone, str):
                in_zone = in_zone or zone
                out_zone = out_zone or zone

        zone_chain.append((in_zone, out_zone))

    proxy.meta["input_zones"] = merged_zones
    proxy.meta["output_zones"] = merged_zones
    proxy.meta["step_count"] = len(rule.steps)
    proxy.meta["zone_chain"] = zone_chain
    return proxy


__all__ = ["as_symbolic_proxy"]
