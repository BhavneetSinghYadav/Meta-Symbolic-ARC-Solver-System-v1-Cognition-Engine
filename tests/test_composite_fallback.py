import pytest
from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule, Transformation, TransformationType, Symbol, SymbolType
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.abstractions.rule_generator import fallback_composite_rules
from arc_solver.src.executor.scoring import score_rule


def _replace(src, tgt, **meta):
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
        meta=meta,
    )


def test_fallback_generates_and_deduplicates():
    inp = Grid([[1]])
    out = Grid([[3]])
    r1 = _replace(1, 2, source="A")
    r1_dup = _replace(1, 2, source="B")
    r2 = _replace(2, 3)

    chains = fallback_composite_rules([r1, r1_dup, r2], inp, out, score_threshold=0.9)
    assert chains, "expected fallback chain generation"
    assert len(chains) < 30
    assert any(isinstance(c, CompositeRule) and c.simulate(inp) == out for c in chains)
    scores = [score_rule(inp, out, c) for c in chains]
    assert scores == sorted(scores, reverse=True)


def test_fallback_filters_invalid():
    inp = Grid([[1]])
    out = Grid([[2] * 64])
    invalid = SymbolicRule(
        transformation=Transformation(TransformationType.REPEAT, params={"kx": "65", "ky": "1"}),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )
    r = _replace(1, 2)
    chains = fallback_composite_rules([invalid, r], inp, out, score_threshold=0.9)
    assert all(all(step.transformation.ttype != TransformationType.REPEAT or step.transformation.params.get("kx") != "65" for step in c.steps) for c in chains)
