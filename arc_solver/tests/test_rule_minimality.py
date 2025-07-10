from arc_solver.src.abstractions.rule_generator import rule_cost
from arc_solver.src.executor.scoring import _op_cost
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Symbol,
    SymbolType,
    Transformation,
    TransformationType,
)
from arc_solver.src.utils import config_loader


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


def test_rule_cost_functional_sparse():
    rule = SymbolicRule(
        Transformation(TransformationType.FUNCTIONAL, params={"op": "dilate_zone"}),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )
    config_loader.set_sparse_mode(True)
    try:
        assert rule_cost(rule) == _op_cost(rule)
    finally:
        config_loader.set_sparse_mode(False)
