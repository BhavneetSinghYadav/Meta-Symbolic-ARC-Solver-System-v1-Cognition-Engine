from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.operators import mirror_tile


def test_mirror_tile_horizontal():
    g = Grid([[1, 2], [3, 4]])
    out = mirror_tile(g, "horizontal", 3)
    assert out.shape() == (2, 6)
    expected = [
        [1, 2, 2, 1, 1, 2],
        [3, 4, 4, 3, 3, 4],
    ]
    assert out.data == expected
