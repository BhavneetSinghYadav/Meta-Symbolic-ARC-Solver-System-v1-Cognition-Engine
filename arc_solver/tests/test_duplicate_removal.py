from arc_solver.src.abstractions.rule_generator import remove_duplicate_rules
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Symbol,
    SymbolType,
    Transformation,
    TransformationType,
)


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
