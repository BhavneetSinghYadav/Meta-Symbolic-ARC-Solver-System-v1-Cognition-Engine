from arc_solver.src.abstractions.abstractor import segment_and_overlay
from arc_solver.src.core.grid import Grid


def test_entropy_merging():
    g = Grid([[1, 1], [1, 1]])
    overlay, _ = segment_and_overlay(g, g)
    assert overlay is None
