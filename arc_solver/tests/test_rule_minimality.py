from arc_solver.src.abstractions.rule_generator import rule_cost
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Symbol,
    SymbolType,
    Transformation,
    TransformationType,
)


def test_rule_minimality():
    simple = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    complex_rule = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1"), Symbol(SymbolType.COLOR, "2")],
        target=[Symbol(SymbolType.COLOR, "3"), Symbol(SymbolType.COLOR, "4")],
        condition={"zone": "A"},
    )
    assert rule_cost(simple) < rule_cost(complex_rule)
