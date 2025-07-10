from arc_solver.src.core.grid import Grid
from arc_solver.src.debug.visualizer import visual_diff_report


def test_visual_diff_report_basic():
    pred = Grid([[1, 2], [3, 4]])
    target = Grid([[1, 0], [3, 4]])
    report = visual_diff_report(pred, target)
    assert "Mismatch at (0,1)" in report
    assert "predicted color 2" in report
    assert "expected color 0" in report
    assert "Total errors: 1" in report


def test_visual_diff_report_shape_mismatch():
    pred = Grid([[1, 2], [3, 4]])
    target = Grid([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
    report = visual_diff_report(pred, target)
    assert "Shape mismatch" in report
    assert "Total errors: 5" in report
