from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)


def _rule(src: int, tgt: int, **cond) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
        condition=cond,
    )


def test_prioritize_by_coverage():
    grid = Grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    small = _rule(1, 3, zone="TopLeft")
    big = _rule(1, 2)
    out = simulate_rules(grid, [small, big])
    for r in range(3):
        for c in range(3):
            assert out.get(r, c) == 2


def test_skip_zero_coverage():
    grid = Grid([[1, 1], [1, 1]])
    rule = _rule(1, 2, zone="TopLeft")
    out = simulate_rules(grid, [rule])
    for r in range(2):
        for c in range(2):
            assert out.get(r, c) == 1
