from __future__ import annotations

"""High level ARC task solving pipeline."""

from typing import List, Tuple

from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.abstractions.rule_generator import generalize_rules
from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.simulator import simulate_symbolic_program
from arc_solver.src.rank_rule_sets import probabilistic_rank_rule_sets
from arc_solver.src.combine_predictions import combine_predictions
from arc_solver.src.introspection import (
    build_trace,
    inject_feedback,
    llm_refine_program,
    evaluate_refinements,
)


def solve_task(task: dict, *, introspect: bool = False):
    """Solve a single ARC task represented by a JSON dictionary."""
    train_pairs = [
        (Grid(p["input"]), Grid(p["output"])) for p in task.get("train", [])
    ]
    test_inputs = [Grid(p["input"]) for p in task.get("test", [])]
    test_outputs = [Grid(p["output"]) for p in task.get("test", []) if "output" in p]

    rule_sets: List[List] = []
    for inp, out in train_pairs:
        rules = abstract([inp, out])
        rules = generalize_rules(rules)
        rule_sets.append(rules)

    ranked_rules = probabilistic_rank_rule_sets(rule_sets, train_pairs)
    best_rules: List = ranked_rules[0][0] if ranked_rules else []

    # Optional introspection/refinement using first training example
    traces = []
    if introspect and best_rules and train_pairs:
        inp0, out0 = train_pairs[0]
        pred0 = simulate_rules(inp0, best_rules)
        trace = build_trace(best_rules[0], inp0, pred0, out0)
        feedback = inject_feedback(trace)
        candidates = llm_refine_program(trace, feedback)
        refined = evaluate_refinements(candidates, inp0, out0)
        best_rules = [refined]
        traces.append(trace)

    predictions = []
    for g in test_inputs:
        ensemble = []
        for rules, prob in ranked_rules:
            pred = simulate_rules(g, rules)
            ensemble.append((pred, prob))
        final = combine_predictions(ensemble)
        predictions.append(final)

    return predictions, test_outputs, traces, best_rules


def solve_task_iterative(task: dict, *, steps: int = 3, introspect: bool = False):
    """Solve task using multi-step symbolic simulation."""
    preds, outs, traces, rules = solve_task(task, introspect=introspect)
    if not preds:
        return preds, outs, traces, rules

    refined_preds = []
    for pred in preds:
        grid = pred
        for _ in range(1, steps):
            grid = simulate_symbolic_program(grid, rules)
        refined_preds.append(grid)
    return refined_preds, outs, traces, rules

__all__ = ["solve_task", "solve_task_iterative"]
