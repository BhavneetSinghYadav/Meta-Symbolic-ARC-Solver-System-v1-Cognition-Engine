from arc_solver.src.segment.segmenter import zone_overlay
from arc_solver.src.core.grid import Grid


def test_zone_overlay_with_morphology():
    grid = Grid([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    overlay = zone_overlay(grid, use_morphology=True)
    assert overlay[1][1] is not None
    assert overlay[1][1].value.startswith("Skeleton")
