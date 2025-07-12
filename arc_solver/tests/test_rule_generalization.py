import pytest
from arc_solver.src.abstractions.rule_generator import generalize_rules
from arc_solver.src.symbolic import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.utils import config_loader


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_generalize_rules_with_composite():
    comp = CompositeRule([_color_rule(1, 2), _color_rule(2, 3)])
    config_loader.set_sparse_mode(True)
    rules = generalize_rules([comp])
    config_loader.set_sparse_mode(False)
    assert rules == [comp]


def test_generalize_rules_preserves_functional_metadata():
    r1 = SymbolicRule(
        Transformation(TransformationType.FUNCTIONAL, params={"op": "dilate_zone"}),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )
    r2 = SymbolicRule(
        Transformation(TransformationType.FUNCTIONAL, params={"op": "erode_zone"}),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )

    rules = generalize_rules([r1, r2])
    assert len(rules) == 2
    assert rules[0].meta.get("op") == "dilate_zone"
    assert rules[1].meta.get("op") == "erode_zone"

