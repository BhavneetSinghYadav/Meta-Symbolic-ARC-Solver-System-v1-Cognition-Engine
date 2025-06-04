from arc_solver.src.symbolic import parse_rule, SymbolType, TransformationType


def test_parse_rule_basic():
    rule = parse_rule("REPLACE [ZONE=TopLeft, COLOR=Red] -> [COLOR=Blue]")
    assert rule.transformation.ttype is TransformationType.REPLACE
    assert any(sym.type is SymbolType.ZONE and sym.value == "TopLeft" for sym in rule.source)
    assert any(sym.type is SymbolType.COLOR and sym.value == "Blue" for sym in rule.target)
