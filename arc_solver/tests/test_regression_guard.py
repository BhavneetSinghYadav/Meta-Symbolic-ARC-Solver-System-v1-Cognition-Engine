import json
import json
from pathlib import Path

import pytest

from arc_solver.src.executor.full_pipeline import solve_task
from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType



def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_regression_guard_triggers(tmp_path, monkeypatch):
    task = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[1]], "output": [[2]]}],
    }

    fallback_grid = Grid([[9]])

    monkeypatch.setattr(
        "arc_solver.src.executor.full_pipeline.fallback_predict", lambda g: fallback_grid
    )
    monkeypatch.setattr(
        "arc_solver.src.executor.full_pipeline.abstract",
        lambda pairs, logger=None: [_color_rule(1, 3)],
    )
    monkeypatch.setattr(
        "arc_solver.src.executor.full_pipeline.simulate_rules",
        lambda grid, rules, **kw: grid,
    )
    monkeypatch.setattr(
        "arc_solver.src.regime.decision_controller.DecisionReflexController.decide",
        lambda self: "symbolic",
    )
    log_path = tmp_path / "fail.json"
    monkeypatch.setattr(
        "arc_solver.src.executor.full_pipeline._FAILURE_LOG", log_path
    )

    preds, outs, traces, rules = solve_task(task, debug=True, task_id="t")
    assert preds[0].data == fallback_grid.data
    assert log_path.exists()
    data = json.loads(log_path.read_text())
    assert data and data[0]["task_id"] == "t"
