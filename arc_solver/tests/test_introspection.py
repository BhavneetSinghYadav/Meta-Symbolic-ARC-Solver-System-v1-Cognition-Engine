from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.introspection import build_trace, validate_trace, narrate_trace


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_validate_trace_metrics():
    grid_in = Grid([[1, 1], [1, 1]])
    rule = _color_rule(1, 2)
    pred = Grid([[2, 1], [2, 1]])
    true = Grid([[2, 2], [2, 2]])
    trace = build_trace(rule, grid_in, pred, true)
    metrics = validate_trace(trace)
    assert metrics["total_cells"] == 4
    assert metrics["correct_cells"] == 2
    assert metrics["symbolic_consistency"] is False
    assert "structural_mismatch" in metrics["conflict_flags"]


def test_narrate_trace_fallback():
    grid = Grid([[1]])
    rule = _color_rule(1, 2)
    pred = Grid([[2]])
    trace = build_trace(rule, grid, pred, pred)
    summary = narrate_trace(trace)
    assert "Rule" in summary
    assert "full coverage" in summary
