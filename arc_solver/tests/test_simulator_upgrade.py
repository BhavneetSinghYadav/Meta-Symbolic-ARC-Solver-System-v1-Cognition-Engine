from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import (
    simulate_rules,
    validate_rule_application,
    check_symmetry_break,
    ReflexOverrideException,
)
from arc_solver.src.symbolic import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_conflict_resolution():
    grid = Grid([[1]])
    r1 = _color_rule(1, 2)
    r2 = _color_rule(1, 3)
    r3 = _color_rule(1, 2)
    out = simulate_rules(grid, [r1, r2, r3])
    assert out.get(0, 0) == 2


def test_reflex_violation():
    grid = Grid([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
        condition={"zone": "TopLeft"},
    )
    try:
        check_symmetry_break(rule, grid)
        assert False, "ReflexOverrideException was not raised"
    except ReflexOverrideException:
        assert True


def test_valid_write_guard():
    grid = Grid([[1]])
    bad_rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
        condition={"zone": "Nonexistent"},
    )
    out = simulate_rules(grid, [bad_rule])
    assert out.data == grid.data


def test_trace_consistency():
    grid = Grid([[1]])
    rule = _color_rule(1, 2)
    trace: list[dict] = []
    out = simulate_rules(grid, [rule], trace_log=trace)
    assert out.get(0, 0) == 2
    assert trace and trace[0]["effect"] == [(0, 0, 1, 2)]

