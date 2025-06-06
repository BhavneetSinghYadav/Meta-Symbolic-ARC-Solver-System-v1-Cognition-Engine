from arc_solver.src.core.grid import Grid


def test_structural_diff_mask():
    g1 = Grid([[1, 2], [3, 4]])
    g2 = Grid([[1, 0], [3, 5]])
    mask = g1.structural_diff(g2)
    assert mask == [[False, True], [False, True]]
