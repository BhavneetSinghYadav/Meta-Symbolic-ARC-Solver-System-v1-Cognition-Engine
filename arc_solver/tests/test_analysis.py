import json
from pathlib import Path

from arc_solver.src.core.grid import Grid
from arc_solver.src.evaluation.analysis import (
    grid_diff_heatmap,
    entropy_change,
    save_failure_case,
)
from arc_solver.src.utils.grid_utils import compute_grid_entropy


def test_compute_grid_entropy_values():
    g1 = Grid([[1, 1], [1, 1]])
    g2 = Grid([[1, 2], [3, 4]])
    assert compute_grid_entropy(g1) == 0.0
    ent = compute_grid_entropy(g2)
    assert 1.9 < ent < 2.1


def test_grid_diff_heatmap_data():
    pred = Grid([[1, 2], [3, 4]])
    tgt = Grid([[1, 0], [3, 4]])
    fig, heat = grid_diff_heatmap(pred, tgt, return_data=True)
    assert heat == [[0, 1], [0, 0]]
    fig.clf()


def test_save_failure_case(tmp_path: Path):
    inp = Grid([[1]])
    tgt = Grid([[2]])
    pred = Grid([[3]])
    save_failure_case("t1", 0, inp, tgt, pred, 0.0, out_dir=tmp_path)
    case_dir = tmp_path / "t1.csv"
    assert (case_dir / "input.png").exists()
    assert (case_dir / "target.png").exists()
    assert (case_dir / "pred.png").exists()
    meta = json.loads((case_dir / "metadata.json").read_text())
    assert meta["task_id"] == "t1"

