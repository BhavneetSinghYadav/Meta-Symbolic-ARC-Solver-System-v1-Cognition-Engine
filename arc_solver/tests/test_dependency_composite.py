import pytest
from arc_solver.src.executor.dependency import (
    sort_rules_by_dependency,
    select_independent_rules,
)
from arc_solver.src.symbolic import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_sort_rules_with_composite():
    r1 = _color_rule(1, 2)
    r2 = _color_rule(2, 3)
    comp = CompositeRule([r1])
    ordered = sort_rules_by_dependency([r2, comp])
    assert ordered[0] is comp


def test_select_independent_rules_composite():
    r1 = _color_rule(1, 2)
    r2 = _color_rule(1, 3)
    comp = CompositeRule([r1])
    selected = select_independent_rules([comp, r2])
    assert selected == [r2, comp]
