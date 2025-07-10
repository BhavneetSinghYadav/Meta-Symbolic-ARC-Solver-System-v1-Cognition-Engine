from __future__ import annotations

"""Simple rule program generalization utilities."""

from typing import List, Union

from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.mutation import mutate_rule


def generalize_rule_program(
    rules: List[SymbolicRule], task_signature: str | None = None
) -> List[Union[SymbolicRule, CompositeRule]]:
    """Return ``rules`` augmented with mutated variants."""

    out: List[Union[SymbolicRule, CompositeRule]] = []
    for r in rules:
        out.append(r)
        try:
            out.extend(mutate_rule(r))
        except Exception:
            continue
    return out


__all__ = ["generalize_rule_program", "mutate_rule"]
