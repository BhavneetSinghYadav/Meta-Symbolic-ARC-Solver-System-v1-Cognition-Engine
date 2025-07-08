import json
from pathlib import Path

from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.core.grid import Grid


def test_composite_rule_integration():
    data = json.loads(Path("arc-agi_training_challenges.json").read_text())
    pair = data["00576224"]["train"][0]
    inp = Grid(pair["input"])
    out = Grid(pair["output"])
    rules = abstract([inp, out])
    assert any(isinstance(r, CompositeRule) for r in rules)
    simple_rules = [r for r in rules if not isinstance(r, CompositeRule)]
    pred = simulate_rules(inp, simple_rules) if simple_rules else inp
    assert pred.shape() == out.shape()
