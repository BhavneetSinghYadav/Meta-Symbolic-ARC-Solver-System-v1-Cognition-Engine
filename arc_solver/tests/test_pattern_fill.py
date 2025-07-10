from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.pattern_fill_operator import pattern_fill


def test_pattern_fill_single():
    grid = Grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    mask = Grid([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    pattern = Grid([[1, 2], [3, 4]])

    out = pattern_fill(grid, mask, pattern)

    assert out.data == [
        [1, 2, 0],
        [3, 4, 0],
        [0, 0, 0],
    ]


def test_pattern_fill_overlap_order():
    grid = Grid([[0] * 3 for _ in range(3)])
    mask = Grid(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
        ]
    )
    pattern = Grid([[5, 6], [7, 8]])

    out = pattern_fill(grid, mask, pattern)

    # The second mask cell overwrites overlapping area
    assert out.data == [
        [5, 5, 6],
        [7, 7, 8],
        [0, 0, 0],
    ]
