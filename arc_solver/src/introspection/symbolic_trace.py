from __future__ import annotations

"""Execution trace utilities for symbolic programs."""

from dataclasses import dataclass
from typing import List, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule


@dataclass
class SymbolicTrace:
    """Record of applying a rule to a grid."""

    rule: SymbolicRule
    before: Grid
    after: Grid
    zone: str | None
    mismatches: List[Tuple[int, int]]


__all__ = ["SymbolicTrace"]
