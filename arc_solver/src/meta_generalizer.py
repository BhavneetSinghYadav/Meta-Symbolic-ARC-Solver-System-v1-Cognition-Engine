from __future__ import annotations

"""Simple rule program generalization utilities."""

from typing import List

from arc_solver.src.symbolic.vocabulary import SymbolicRule, Symbol, SymbolType
from arc_solver.src.symbolic.vocabulary import TransformationType


def mutate_rule(rule: SymbolicRule, task_signature: str | None = None) -> SymbolicRule:
    """Return a lightly generalized variant of ``rule``."""
    if rule.transformation.ttype is TransformationType.REPLACE and rule.source:
        src = rule.source[0]
        tgt = rule.target[0] if rule.target else src
        # drop explicit source color to generalize
        return SymbolicRule(rule.transformation, source=[], target=[tgt], nature=rule.nature, condition=rule.condition.copy())
    return rule


def generalize_rule_program(rules: List[SymbolicRule], task_signature: str | None = None) -> List[SymbolicRule]:
    """Return a list containing original and generalized rules."""
    out: List[SymbolicRule] = []
    for r in rules:
        out.append(r)
        try:
            out.append(mutate_rule(r, task_signature))
        except Exception:
            continue
    return out


__all__ = ["generalize_rule_program", "mutate_rule"]
