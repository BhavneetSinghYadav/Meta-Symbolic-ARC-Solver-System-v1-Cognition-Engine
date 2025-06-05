from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType
from arc_solver.src.introspection import (
    compute_discrepancy,
    trace_prediction,
    refine_rule,
    run_meta_repair,
)


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_compute_discrepancy_simple():
    pred = Grid([[1, 2]])
    target = Grid([[1, 0]])
    diff = compute_discrepancy(pred, target)
    assert diff == {(0, 1): (2, 0)}


def test_trace_prediction_records_cells():
    grid = Grid([[1]])
    rule = _color_rule(1, 2)
    traces = trace_prediction([rule], grid)
    assert traces[0].affected == [(0, 0)]


def test_refine_rule_adjusts_color():
    bad_rule = _color_rule(1, 3)
    context = {(0, 0): (3, 2)}
    fixed = refine_rule(bad_rule, context)
    assert fixed and fixed.target[0].value == "2"


def test_run_meta_repair_improves_program():
    inp = Grid([[1, 1]])
    true = Grid([[2, 2]])
    bad_rule = _color_rule(1, 3)
    pred = simulate_rules(inp, [bad_rule])
    repaired_pred, fixed_rules = run_meta_repair(inp, pred, true, [bad_rule])
    assert repaired_pred.compare_to(true) == 1.0
    assert fixed_rules[0].target[0].value == "2"
