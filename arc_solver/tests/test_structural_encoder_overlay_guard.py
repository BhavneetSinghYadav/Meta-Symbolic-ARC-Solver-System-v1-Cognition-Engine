from arc_solver.src.attention.structural_encoder import validate_overlay
from arc_solver.src.core.grid import Grid


def test_validate_overlay_shape_and_type():
    grid = Grid([[0, 0], [0, 0]])
    bad_overlay = [[None], [None]]
    assert not validate_overlay(grid, bad_overlay)
    good_overlay = [[None, 1], ["A", None]]
    assert validate_overlay(grid, good_overlay)
