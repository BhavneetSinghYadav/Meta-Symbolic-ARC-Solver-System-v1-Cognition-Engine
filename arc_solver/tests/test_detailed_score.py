from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic import Symbol, SymbolType


def test_diff_summary_basic():
    g1 = Grid([[1, 2], [3, 4]])
    g2 = Grid([[1, 2], [4, 4]])
    overlay = [[Symbol(SymbolType.ZONE, "A"), Symbol(SymbolType.ZONE, "A")],
               [Symbol(SymbolType.ZONE, "B"), Symbol(SymbolType.ZONE, "B")]]
    g1.attach_overlay(overlay)
    g2.attach_overlay(overlay)
    summary = g1.diff_summary(g2)
    assert summary["cell_match_ratio"] == 0.75
    assert summary["zone_coverage_match"] == 1.0
    assert summary["symbol_mismatch_count"] == 0


def test_detailed_score_range():
    g1 = Grid([[1, 0], [0, 1]])
    g2 = Grid([[1, 1], [0, 1]])
    score = g1.detailed_score(g2)
    assert 0.0 <= score <= 1.0
    assert score < 1.0
