from __future__ import annotations

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.attention import zone_to_mask
from arc_solver.src.executor.dependency import select_independent_rules


def test_reflex_override_symmetry():
    grid = Grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    rule = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
        condition={"zone": "TopLeft"},
    )
    pred = simulate_rules(grid, [rule])
    assert pred.data == grid.data


def test_attention_mask_zone():
    grid = Grid([[1, 0, 2], [1, 1, 1], [1, 1, 1]])
    mask = zone_to_mask(grid, "TopLeft")
    rule = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "3")],
    )
    pred = simulate_rules(grid, [rule], attention_mask=mask)
    assert pred.get(0, 0) == 3
    for r in range(3):
        for c in range(3):
            if (r, c) == (0, 0):
                continue
            assert pred.get(r, c) == grid.get(r, c)


def test_dependency_graph_select():
    r1 = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    r2 = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "3")],
    )
    selected = select_independent_rules([r1, r2])
    assert len(selected) == 1
