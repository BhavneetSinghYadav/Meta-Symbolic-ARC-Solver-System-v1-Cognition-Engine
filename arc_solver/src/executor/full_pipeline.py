from __future__ import annotations

"""High level ARC task solving pipeline."""

from typing import List, Tuple

from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.abstractions.rule_generator import generalize_rules
from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.simulator import simulate_symbolic_program
from arc_solver.src.introspection import (
    build_trace,
    inject_feedback,
    llm_refine_program,
    evaluate_refinements,
)


def _score_rules(rules: List, pairs: List[Tuple[Grid, Grid]]) -> float:
    total = 0.0
    for inp, out in pairs:
        pred = simulate_rules(inp, rules)
        total += pred.compare_to(out)
    return total / len(pairs) if pairs else 0.0


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

    # Select best rule set by average training score
    best_rules: List = []
    best_score = -1.0
    for rules in rule_sets:
        score = _score_rules(rules, train_pairs)
        if score > best_score:
            best_score = score
            best_rules = rules

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

    predictions = [simulate_rules(g, best_rules) for g in test_inputs]
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
