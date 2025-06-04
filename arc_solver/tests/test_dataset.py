from arc_solver.src.data import ARCDataset, load_arc_task
from pathlib import Path

def test_load_arc_task():
    task = load_arc_task(Path(__file__).parent / "sample_task.json")
    assert task["train"][0]["output"][0][0] == 1

def test_arcdataset_iteration():
    ds = ARCDataset(Path(__file__).parent)
    tasks = list(ds)
    assert len(tasks) >= 1
