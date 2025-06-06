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
    # Symmetry violation triggers reflex override; grid remains unchanged
    for r in range(3):
        for c in range(3):
            assert pred.get(r, c) == 1


def test_zone_expansion_fallback():
    grid = Grid([[1, 1], [1, 1]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
        condition={"zone": "Missing"},
    )
    pred = simulate_rules(grid, [rule])
    for r in range(2):
        for c in range(2):
            assert pred.get(r, c) == 1
