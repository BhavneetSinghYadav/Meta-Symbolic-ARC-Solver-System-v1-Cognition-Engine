from arc_solver.src.core.grid import Grid
from arc_solver.src.utils import validate_grid


def test_validate_grid_valid():
    grid = Grid([[1, 2], [3, 4]])
    assert validate_grid(grid, expected_shape=(2, 2))


def test_validate_grid_value_range():
    grid = Grid([[1, 11]])
    assert not validate_grid(grid)


def test_validate_grid_shape_mismatch():
    grid = Grid([[1]])
    assert not validate_grid(grid, expected_shape=(2, 2))


def test_validate_grid_all_zero():
    grid = Grid([[0, 0], [0, 0]])
    assert not validate_grid(grid)
