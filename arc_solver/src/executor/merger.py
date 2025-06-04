"""Utilities for merging symbolic rule sets into unified programs."""

from __future__ import annotations

from typing import Dict, List, Tuple

from arc_solver.src.symbolic.vocabulary import SymbolicRule


Key = Tuple[str, Tuple, Tuple]


def _rule_key(rule: SymbolicRule) -> Key:
    return (
        rule.transformation.ttype.value,
        tuple(rule.source),
        tuple(rule.target),
    )


def merge_rule_sets(rule_sets: List[List[SymbolicRule]]) -> List[SymbolicRule]:
    """Merge multiple rule sets, collapsing identical behaviors across zones."""
    groups: Dict[Key, Dict[str, any]] = {}
    for rules in rule_sets:
        for rule in rules:
            key = _rule_key(rule)
            info = groups.setdefault(key, {"zones": set(), "rule": rule})
            zone = rule.condition.get("zone")
            if zone:
                info["zones"].add(zone)
    merged: List[SymbolicRule] = []
    for info in groups.values():
        base = info["rule"]
        if info["zones"]:
            condition = {"zone": "|".join(sorted(info["zones"]))}
        else:
            condition = base.condition
        merged.append(
            SymbolicRule(
                transformation=base.transformation,
                source=base.source,
                target=base.target,
                nature=base.nature,
                condition=condition,
            )
        )
    return merged


__all__ = ["merge_rule_sets"]
