import pytest
from arc_solver.src.symbolic.composite_rules import generate_repeat_composite_rules
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.executor.scoring import score_rule
from arc_solver.src.executor.simulator import validate_color_dependencies
from arc_solver.src.core.grid import Grid


def test_repeat_composite_multiple_mappings():
    inp = Grid([[1, 2]])
    out = Grid([[3, 4], [3, 4]])

    rules = generate_repeat_composite_rules(inp, out)
    assert len(rules) == 1

    comp = rules[0]
    assert isinstance(comp, CompositeRule)
    # repeat + two replace steps
    assert len(comp.steps) == 3
    assert comp.simulate(inp).data == out.data

    score = score_rule(inp, out, comp)
    assert score > 0.9

    validated = validate_color_dependencies([comp], inp)
    assert validated == [comp]
