from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Transformation,
    TransformationType,
    Symbol,
    SymbolType,
)
from arc_solver.src.symbolic.operators import mirror_tile


def test_repeat():
    inp = Grid([[1]])
    rule = SymbolicRule(
        transformation=Transformation(
            TransformationType.REPEAT, params={"kx": "2", "ky": "2"}
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )
    out = rule.apply(inp)
    assert out.data == [[1, 1], [1, 1]]


def test_recolor():
    inp = Grid([[1, 2], [1, 2]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "3")],
    )
    out = rule.apply(inp)
    assert out.data == [[3, 2], [3, 2]]


def test_rotate90():
    inp = Grid([[1, 2], [3, 4]])
    rule = SymbolicRule(
        transformation=Transformation(
            TransformationType.ROTATE90, params={"times": "1"}
        ),
        source=[Symbol(SymbolType.SHAPE, "A")],
        target=[Symbol(SymbolType.SHAPE, "A")],
    )
    out = rule.apply(inp)
    assert out.data == [[3, 1], [4, 2]]


def test_translate():
    inp = Grid([[1, 2], [3, 4]])
    rule = SymbolicRule(
        transformation=Transformation(
            TransformationType.TRANSLATE, params={"dx": "1", "dy": "0"}
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )
    out = rule.apply(inp)
    assert out.data == [[0, 1], [0, 3]]


def test_shape_abstract_identity():
    inp = Grid([[1, 2], [3, 4]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.SHAPE_ABSTRACT),
        source=[Symbol(SymbolType.SHAPE, "A")],
        target=[Symbol(SymbolType.SHAPE, "A")],
    )
    out = rule.apply(inp)
    assert out.data == inp.data


def test_mirror_tile_horizontal():
    inp = Grid([[1, 2], [3, 4]])
    out = mirror_tile(inp, "horizontal", 2)
    assert out.data == [[1, 2, 2, 1], [3, 4, 4, 3]]
