from __future__ import annotations

"""Penalty based symbolic scoring utilities."""

from dataclasses import dataclass
from typing import Tuple, Dict

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.vocabulary import SymbolicRule


@dataclass
class SymbolicDiffPenaltyEngine:
    """Compute pattern-aware scores for predicted grids."""

    zone_weight: float = 0.3
    symbol_weight: float = 0.1

    def score(self, predicted: Grid, target: Grid) -> Tuple[float, Dict[str, float]]:
        """Return (score, summary) comparing ``predicted`` and ``target``."""
        summary = predicted.diff_summary(target)
        score = predicted.detailed_score(target)
        summary["score"] = score
        return score, summary

    def evaluate_rule(self, rule: SymbolicRule, inp: Grid, tgt: Grid) -> Tuple[float, Dict[str, float]]:
        """Simulate ``rule`` on ``inp`` and evaluate against ``tgt``."""
        pred = simulate_rules(inp, [rule])
        return self.score(pred, tgt)

__all__ = ["SymbolicDiffPenaltyEngine"]
