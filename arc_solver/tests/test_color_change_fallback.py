from arc_solver.src.abstractions.abstractor import extract_color_change_rules
from arc_solver.src.core.grid import Grid
from arc_solver.src.segment.segmenter import zone_overlay


def test_zone_fallback():
    grid_in = Grid([[1]*6 for _ in range(6)])
    grid_out = Grid([[1]*6 for _ in range(6)])
    grid_out.set(2, 2, 2)
    overlay = zone_overlay(grid_in)
    rules = extract_color_change_rules(grid_in, grid_out, overlay)
    assert rules
    assert "zone" not in rules[0].condition
