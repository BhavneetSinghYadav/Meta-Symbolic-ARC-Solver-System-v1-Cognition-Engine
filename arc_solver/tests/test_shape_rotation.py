import pytest
from arc_solver.src.abstractions.abstractor import extract_shape_based_rules
from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.executor.simulator import simulate_rules


def test_extract_rotation_rule():
    inp = Grid([[1, 2], [0, 0]])
    out = inp.rotate90(1)
    rules = extract_shape_based_rules(inp, out)
    assert any(r.transformation.ttype is TransformationType.ROTATE90 for r in rules)


def test_extract_shape_rule():
    inp = Grid([[1, 0], [0, 0]])
    out = Grid([[2, 0], [0, 0]])
    rules = extract_shape_based_rules(inp, out)
    assert any(r.transformation.ttype is TransformationType.SHAPE_ABSTRACT for r in rules)


def test_rotate90_application():
    grid = Grid([[1, 0], [2, 0]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.ROTATE90, params={"times": "1"}),
        source=[Symbol(SymbolType.SHAPE, "Any")],
        target=[Symbol(SymbolType.SHAPE, "Any")],
    )
    out = simulate_rules(grid, [rule])
    assert out.data == grid.rotate90(1).data

