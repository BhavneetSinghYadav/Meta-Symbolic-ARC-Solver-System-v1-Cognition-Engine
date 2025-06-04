from __future__ import annotations

"""Select the best symbolic rule program for a grid pair."""

from typing import List

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules, score_prediction
from arc_solver.src.executor.conflict_resolver import resolve_conflicts
from arc_solver.src.symbolic.vocabulary import SymbolicRule


def select_best_program(
    input_grid: Grid,
    output_grid: Grid,
    rule_sets: List[List[SymbolicRule]],
) -> List[SymbolicRule]:
    """Return the rule set with the highest simulation score."""
    best_rules: List[SymbolicRule] = []
    best_score = -1.0
    for rules in rule_sets:
        resolved = resolve_conflicts(rules, input_grid)
        predicted = simulate_rules(input_grid, resolved)
        score = score_prediction(predicted, output_grid)
        if score > best_score:
            best_score = score
            best_rules = rules
    return best_rules
