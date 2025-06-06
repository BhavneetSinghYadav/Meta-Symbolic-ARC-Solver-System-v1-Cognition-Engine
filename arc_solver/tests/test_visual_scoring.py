from arc_solver.src.core.grid import Grid
from arc_solver.src.introspection.visual_scoring import compute_visual_score, rerank_by_visual_score
from arc_solver.src.symbolic import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType


def _rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_compute_visual_score_perfect_match():
    g = Grid([[1, 2], [3, 4]])
    assert compute_visual_score(g, g) == 1.0


def test_rerank_by_visual_score():
    inp = Grid([[1, 1], [1, 1]])
    tgt = Grid([[2, 2], [2, 2]])
    rule1 = _rule(1, 2)
    rule2 = _rule(1, 3)
    ranked = rerank_by_visual_score([[rule2], [rule1]], inp, tgt)
    assert ranked and ranked[0][0].target[0].value == str(2)
