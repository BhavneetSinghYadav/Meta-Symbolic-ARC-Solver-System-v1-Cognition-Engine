from arc_solver.src.introspection import build_trace, validate_trace
from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_no_ground_truth():
    grid = Grid([[1]])
    rule = _color_rule(1, 2)
    pred = Grid([[2]])
    trace = build_trace(rule, grid, pred, None)
    m = validate_trace(trace)
    assert m["coverage_score"] == 1.0
    assert m["interpretation_valid"]


def test_repeated_labels_flag():
    grid = Grid([[1]])
    rule = _color_rule(1, 2)
    pred = Grid([[2]])
    trace = build_trace(rule, grid, pred, pred)
    trace.symbolic_context["labels"] = {(0, 0): ["A", "A", "A"]}
    metrics = validate_trace(trace)
    assert "region_repeated" in metrics["conflict_flags"]
