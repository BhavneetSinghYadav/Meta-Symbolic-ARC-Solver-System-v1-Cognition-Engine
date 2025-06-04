from __future__ import annotations

"""Simple library of prior rule templates."""

from typing import List

from arc_solver.src.symbolic.rule_language import parse_rule
from arc_solver.src.symbolic.vocabulary import SymbolicRule


def load_prior_templates() -> List[List[SymbolicRule]]:
    """Return a list of canned rule programs."""
    try:
        rule = parse_rule("REPLACE [COLOR=0] -> [COLOR=1]")
    except Exception:
        return []
    return [[rule]]


__all__ = ["load_prior_templates"]
