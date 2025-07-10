from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
    TransformationNature,
)
from arc_solver.src.executor.scoring import score_rule, preferred_rule_types
from arc_solver.src.scoring.entropy_utils import grid_color_entropy
import pytest


def test_rule_scoring_on_tiling_vs_replace():
    inp = Grid([[1]])
    out = Grid([[1, 1], [1, 1]])

    repeat = SymbolicRule(
        transformation=Transformation(TransformationType.REPEAT, params={"kx": "2", "ky": "2"}),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        nature=TransformationNature.SPATIAL,
    )

    replace = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )

    s_repeat = score_rule(inp, out, repeat)
    s_replace = score_rule(inp, out, replace)
    assert s_repeat > s_replace


def test_grow_rules_prioritization():
    inp = Grid([[1]])
    out = Grid([[1, 1], [1, 1]])
    prefs = preferred_rule_types(inp, out)
    assert "REPEAT" in prefs or "REPEAT→REPLACE" in prefs


def test_score_trace_logging(monkeypatch):
    inp = Grid([[1]])
    out = Grid([[9, 9], [9, 9]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )

    captured = {}

    def fake_log_failure(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "arc_solver.src.executor.scoring.log_failure", fake_log_failure
    )

    trace = score_rule(inp, out, rule, return_trace=True)

    assert trace["final_score"] < 0.2
    assert captured.get("score_trace") == trace


def test_score_rule_expected_value():
    inp = Grid([[1]])
    out = Grid([[2]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )

    score = score_rule(inp, out, rule)
    assert score == pytest.approx(1.244, rel=1e-3)


def test_grid_color_entropy_extremes():
    same = Grid([[1, 1], [1, 1]])
    assert grid_color_entropy(same) == 0.0

    uniform = Grid([[1, 2], [3, 4]])
    assert grid_color_entropy(uniform) == pytest.approx(1.0, rel=1e-6)


def test_grid_color_entropy_intermediate():
    mixed = Grid([[1, 1, 1], [1, 1, 2]])
    ent = grid_color_entropy(mixed)
    assert 0.0 < ent < 1.0
