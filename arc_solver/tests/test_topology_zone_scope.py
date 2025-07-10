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
    meta = {"input_zones": [in_zone] if isinstance(in_zone, str) else list(in_zone)}
    if out_zone is not None:
        meta["output_zones"] = [out_zone] if isinstance(out_zone, str) else list(out_zone)
    else:
        meta["output_zones"] = meta["input_zones"]
    zone = in_zone if isinstance(in_zone, str) else (in_zone[0] if in_zone else None)
    cond = {"zone": zone} if zone else {}
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
        condition=cond,
        meta=meta,
    )


def test_zone_scope_sorting():
    step1 = _z_rule(1, 2, "A", ["B", "C"])
    step2 = _z_rule(2, 3, "B", "D")
    comp1 = CompositeRule([step1, step2])

    comp2 = CompositeRule([_z_rule(3, 4, "C", "E")])

    ordered = sort_rules_by_topology([comp2, comp1])
    assert ordered[0] is comp1
    assert ordered[1] is comp2
