from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.rotate_about_point import rotate_about_point


def test_rotate_l_90():
    grid = Grid([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ])
    center = (1, 1)
    out = rotate_about_point(grid, center, 90)
    expected = [
        [0, 0, 0],
        [0, 0, 1],
        [1, 1, 1],
    ]
    assert out.data == expected


def test_rotate_t_180():
    grid = Grid([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    center = (1, 1)
    out = rotate_about_point(grid, center, 180)
    expected = [
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    assert out.data == expected


def test_rotate_l_270():
    grid = Grid([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ])
    center = (1, 1)
    out = rotate_about_point(grid, center, 270)
    expected = [
        [1, 1, 1],
        [1, 0, 0],
        [0, 0, 0],
    ]
    assert out.data == expected
