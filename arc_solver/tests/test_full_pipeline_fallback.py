from arc_solver.src.executor.full_pipeline import solve_task
from arc_solver.src.core.grid import Grid


def test_pipeline_empty_rules():
    task = {"train": [], "test": [{"input": [[1]]}]}
    preds, outs, traces, rules = solve_task(task)
    assert preds and isinstance(preds[0], Grid)
