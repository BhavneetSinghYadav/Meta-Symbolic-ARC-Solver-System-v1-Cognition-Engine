from arc_solver.src.executor.dependency import sort_rules_by_topology
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule


def _z_rule(src, tgt, in_zone, out_zone=None):
    meta = {"input_zones": [in_zone]}
    if out_zone:
        meta["output_zones"] = [out_zone]
    else:
        meta["output_zones"] = [in_zone]
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
        condition={"zone": in_zone},
        meta=meta,
    )


def test_zone_chain_sorting():
    step1 = _z_rule(1, 2, "A", "B")
    step2 = _z_rule(2, 3, "B", "C")
    comp1 = CompositeRule([step1, step2])

    comp2 = CompositeRule([_z_rule(3, 4, "C", "D")])

    ordered = sort_rules_by_topology([comp2, comp1])
    assert ordered[0] is comp1
    assert ordered[1] is comp2
