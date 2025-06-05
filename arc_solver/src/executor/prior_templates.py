from __future__ import annotations

"""Simple library of prior rule templates."""

from typing import List

from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.memory.deep_prior_loader import load_prior_templates as _load


def load_prior_templates() -> List[List[SymbolicRule]]:
    """Return a list of canned rule programs."""
    templates = _load()
    return [t["rules"] for t in templates]


__all__ = ["load_prior_templates"]
