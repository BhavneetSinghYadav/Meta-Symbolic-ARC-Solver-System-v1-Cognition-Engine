from arc_solver.src.core.grid import Grid
from arc_solver.src.executor import (
    simulate_rules,
    score_prediction,
    resolve_conflicts,
    select_best_program,
)
from arc_solver.src.feedback import generate_error_signals
from arc_solver.src.memory import PolicyCache
from arc_solver.src.search import rank_rule_sets
from arc_solver.src.symbolic import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_simulate_replace_and_score():
    grid = Grid([[1, 2], [1, 0]])
    rule = _color_rule(1, 3)
    pred = simulate_rules(grid, [rule])
    assert pred.data[0][0] == 3
    target = Grid([[3, 2], [3, 0]])
    assert score_prediction(pred, target) == 1.0


def test_conflict_resolver_prefers_simple():
    grid = Grid([[1, 1], [2, 2]])
    simple = _color_rule(1, 3)
    complex_rule = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.ZONE, "Z"), Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    resolved = resolve_conflicts([complex_rule, simple], grid)
    assert resolved[0] == simple


def test_select_best_program():
    grid_in = Grid([[1, 1], [0, 0]])
    grid_out = Grid([[2, 2], [0, 0]])
    good = [_color_rule(1, 2)]
    bad = [_color_rule(1, 3)]
    best = select_best_program(grid_in, grid_out, [bad, good])
    assert best == good


def test_generate_error_signals():
    pred = Grid([[1]])
    target = Grid([[2]])
    msgs = generate_error_signals(pred, target)
    assert any("Color mismatch" in m for m in msgs)


def test_policy_cache():
    cache = PolicyCache()
    ruleset = [_color_rule(1, 2)]
    cache.add_failure("task", ruleset)
    assert cache.is_failed("task", ruleset)


def test_rule_ranker():
    rs1 = [_color_rule(1, 2), _color_rule(2, 3)]
    rs2 = [_color_rule(1, 2)]
    ranked = rank_rule_sets([rs2, rs1])
    assert ranked[0] == rs1
