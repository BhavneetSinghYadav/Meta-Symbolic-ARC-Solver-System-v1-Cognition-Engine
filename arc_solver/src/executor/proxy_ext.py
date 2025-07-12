from __future__ import annotations

"""Extended proxy utilities for composite rules."""

from typing import Any, List, Tuple

from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Symbol,
    TransformationType,
)


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
    describing the input/output zone transition of each step. ``zone_chain``
    is used by :func:`sort_rules_by_topology` to build a dependency graph that
    respects how composites move data across zones over time. Functional
    operations like ``dilate_zone`` or ``zone_remap`` automatically record
    their zone parameters so spatial dependencies are preserved.
    """

    cond: dict[str, Any] = rule.get_condition() or {}

    last_step = rule.steps[-1]
    proxy = SymbolicRule(
        transformation=last_step.transformation,
        source=rule.steps[0].source,
        target=rule.final_targets(),
        condition=cond,
        nature=rule.nature,
    )

    zone_chain: List[Tuple[str | None, str | None]] = []
    zone_scopes: List[Tuple[List[str], List[str]]] = []
    pivot_chain: List[str | None] = []

    def _to_list(val: Any) -> List[str]:
        if not val:
            return []
        if isinstance(val, str):
            return [val]
        return list(val)

    for step in rule.steps:
        meta = dict(getattr(step, "meta", {}) or {})
        cond = getattr(step, "condition", None) or {}

        if step.transformation.ttype is TransformationType.FUNCTIONAL:
            op = step.transformation.params.get("op")
            if op in {"dilate_zone", "erode_zone"}:
                z = step.transformation.params.get("zone")
                if z is not None:
                    val = str(z)
                    meta.setdefault("input_zones", [val])
                    meta.setdefault("output_zones", [val])
            elif op == "zone_remap":
                mapping = step.transformation.params.get("map") or meta.get("mapping")
                if isinstance(mapping, dict):
                    zones = [str(k) for k in mapping.keys()]
                    meta.setdefault("input_zones", zones)
                    meta.setdefault("output_zones", zones)
        elif step.transformation.ttype is TransformationType.ROTATE:
            cx = step.transformation.params.get("cx")
            cy = step.transformation.params.get("cy")
            if cx is not None and cy is not None:
                meta.setdefault("pivot", f"{cx},{cy}")

        in_list = _to_list(meta.get("input_zones"))
        out_list = _to_list(meta.get("output_zones"))
        cond_list = _to_list(cond.get("zone"))

        if not in_list:
            in_list = cond_list
        if not out_list:
            out_list = cond_list

        def _first(lst: List[str]) -> str | None:
            return lst[0] if lst else None

        zone_chain.append((_first(in_list), _first(out_list)))
        zone_scopes.append((in_list, out_list))
        pivot_chain.append(meta.get("pivot"))

    merged_zones = sorted({z for ins, outs in zone_scopes for z in ins + outs if z})
    if len(merged_zones) == 1:
        cond = {**cond, "zone": merged_zones[0]}

    proxy.meta["input_zones"] = merged_zones
    proxy.meta["output_zones"] = merged_zones
    proxy.meta["step_count"] = len(rule.steps)
    proxy.meta["zone_chain"] = zone_chain
    proxy.meta["zone_scope_chain"] = zone_scopes
    proxy.meta["pivot_chain"] = pivot_chain
    unique_pivots = {p for p in pivot_chain if p}
    if len(unique_pivots) == 1:
        proxy.meta["pivot"] = next(iter(unique_pivots))
    return proxy


__all__ = ["as_symbolic_proxy"]
