import pytest
from arc_solver.src.symbolic.vocabulary import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType
from arc_solver.src.symbolic.rule_language import CompositeRule


def _z_rule(src, tgt, zone):
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
        condition={"zone": zone},
        meta={"input_zones": [zone], "output_zones": [zone]},
    )


def test_proxy_merges_zones():
    step1 = _z_rule(1, 2, "TopLeft")
    step2 = _z_rule(2, 3, "BottomRight")
    comp = CompositeRule([step1, step2])

    proxy = comp.as_symbolic_proxy()
    assert set(proxy.meta.get("input_zones", [])) == {"TopLeft", "BottomRight"}
    assert set(proxy.meta.get("output_zones", [])) == {"TopLeft", "BottomRight"}
    assert proxy.transformation.ttype is TransformationType.REPLACE
