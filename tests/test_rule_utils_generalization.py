import pytest
from arc_solver.src.symbolic import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType
from arc_solver.src.utils.rule_utils import generalize_rules


def _repl(zone: str) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
        meta={"zone": zone},
    )


def test_generalize_with_merges_meta():
    r1 = _repl("A")
    r2 = _repl("B")
    merged = r1.generalize_with(r2)
    assert merged is not None
    assert merged.meta["zone"] == ["A", "B"]


def test_generalize_rules_groups_by_dsl(tmp_path):
    r1 = _repl("A")
    r2 = _repl("B")
    out = generalize_rules([r1, r2])
    assert len(out) == 1
    assert out[0].meta["zone"] == ["A", "B"]
