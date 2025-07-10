import pytest
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule


def _functional_rule(op, zone=None, mapping=None):
    params = {"op": op}
    if zone is not None:
        params["zone"] = str(zone)
    if mapping is not None:
        params["map"] = mapping
    return SymbolicRule(
        transformation=Transformation(TransformationType.FUNCTIONAL, params=params),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )


def test_functional_zone_proxy():
    step1 = _functional_rule("dilate_zone", zone=1)
    step2 = _functional_rule("zone_remap", mapping={1: 2})
    comp = CompositeRule([step1, step2])

    proxy = comp.as_symbolic_proxy()
    assert proxy.meta.get("zone_chain") == [("1", "1"), ("1", "1")]
    assert proxy.meta.get("zone_scope_chain") == [(["1"], ["1"]), (["1"], ["1"])]
    assert set(proxy.meta.get("input_zones", [])) == {"1"}
    assert set(proxy.meta.get("output_zones", [])) == {"1"}

