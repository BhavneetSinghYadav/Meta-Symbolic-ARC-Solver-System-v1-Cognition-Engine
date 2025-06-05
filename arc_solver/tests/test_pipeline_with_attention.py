import json
from pathlib import Path

from arc_solver.src.executor.full_pipeline import solve_task
from arc_solver.src.utils import config_loader


def test_pipeline_runs_with_attention():
    config_loader.set_use_structural_attention(True)
    task = json.loads(Path("arc_solver/tests/sample_task.json").read_text())
    preds, _, _, _ = solve_task(task)
    assert preds and preds[0].shape() == (1, 1)
