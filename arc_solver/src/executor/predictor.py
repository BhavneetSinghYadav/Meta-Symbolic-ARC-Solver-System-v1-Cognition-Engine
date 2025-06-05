from __future__ import annotations

"""Select the best symbolic rule program for a grid pair."""

from typing import List, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules, score_prediction
from arc_solver.src.executor.conflict_resolver import detect_conflicts, resolve_conflicts
from arc_solver.src.symbolic.vocabulary import SymbolicRule
from arc_solver.src.utils import validate_grid
from arc_solver.src.introspection import (
    build_trace,
    inject_feedback,
    llm_refine_program,
    evaluate_refinements,
)


def select_best_program(
    input_grid: Grid,
    output_grid: Grid,
    rule_sets: List[List[SymbolicRule]],
) -> List[SymbolicRule]:
    """Return the rule set with the highest simulation score.

    Candidates producing invalid grids are discarded. If the best score is
    below ``0.5`` a simple refinement pass is attempted using heuristic
    feedback utilities.
    """

    candidates: List[Tuple[List[SymbolicRule], Grid, float]] = []

    for rules in rule_sets:
        conflicts = detect_conflicts(rules, input_grid)
        resolved = resolve_conflicts(conflicts, rules)
        predicted = simulate_rules(input_grid, resolved)
        if not validate_grid(predicted, expected_shape=output_grid.shape()):
            continue
        score = score_prediction(predicted, output_grid)
        candidates.append((rules, predicted, score))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[2], reverse=True)
    best_rules, best_pred, best_score = candidates[0]

    if best_score < 0.5 and best_rules:
        try:
            trace = build_trace(best_rules[0], input_grid, best_pred, output_grid)
            feedback = inject_feedback(trace)
            cands = llm_refine_program(trace, feedback)
            refined = evaluate_refinements(cands, input_grid, output_grid)
            best_rules = [refined] + list(best_rules[1:])
        except Exception:
            pass

    return list(best_rules)
