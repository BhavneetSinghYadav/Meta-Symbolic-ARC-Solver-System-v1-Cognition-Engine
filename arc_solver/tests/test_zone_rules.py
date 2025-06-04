from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType


def test_replace_zone_condition():
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
    pred = simulate_rules(grid, [rule])
    assert pred.get(0, 0) == 2
    for r in range(3):
        for c in range(3):
            if (r, c) == (0, 0):
                continue
            assert pred.get(r, c) == 1
