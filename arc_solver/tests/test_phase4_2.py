from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.introspection import (
    build_trace,
    inject_feedback,
    llm_refine_program,
    evaluate_refinements,
)


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_llm_refinement_improves_rule():
    grid_in = Grid([[1, 1], [1, 1]])
    grid_out = Grid([[2, 2], [2, 2]])
    bad_rule = _color_rule(1, 3)
    pred = simulate_rules(grid_in, [bad_rule])
    trace = build_trace(bad_rule, grid_in, pred, grid_out)
    feedback = inject_feedback(trace)
    candidates = llm_refine_program(trace, feedback)
    best = evaluate_refinements(candidates, grid_in, grid_out)
    before = pred.compare_to(grid_out)
    after = simulate_rules(grid_in, [best]).compare_to(grid_out)
    assert after >= before

