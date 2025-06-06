from arc_solver.src.search.rule_ranker import rank_rule_sets
from arc_solver.src.symbolic import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType


def _color_rule(src: int, tgt: int, zone: str | None = None) -> SymbolicRule:
    cond = {"zone": zone} if zone else None
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
        condition=cond,
    )


def test_rank_rule_sets_feature_bonus():
    rs1 = [_color_rule(1, 2)]
    rs2 = [_color_rule(1, 2), _color_rule(2, 3, zone="TopLeft")]
    ranked = rank_rule_sets([rs1, rs2])
    assert ranked[0] == rs2
