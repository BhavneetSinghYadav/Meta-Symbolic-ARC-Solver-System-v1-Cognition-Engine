from arc_solver.src.abstractions.rule_generator import remove_duplicate_rules
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Symbol,
    SymbolType,
    Transformation,
    TransformationType,
)
from arc_solver.src.core.grid import Grid


def _rule():
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )


def test_duplicate_removal():
    r1 = _rule()
    r2 = _rule()
    dedup = remove_duplicate_rules([r1, r2])
    assert len(dedup) == 1


def test_duplicate_removal_with_meta():
    grid_a = Grid([[1]])
    grid_b = Grid([[2]])
    r1 = SymbolicRule(
        Transformation(TransformationType.FUNCTIONAL, params={"op": "pattern_fill"}),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        meta={"mask": grid_a, "pattern": grid_b},
    )
    r2 = SymbolicRule(
        Transformation(TransformationType.FUNCTIONAL, params={"op": "pattern_fill"}),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        meta={"mask": grid_b, "pattern": grid_b},
    )

    dedup = remove_duplicate_rules([r1, r2])
    assert len(dedup) == 2
