from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.core.grid import Grid


def test_fallback_trigger():
    inp = Grid([[1]])
    out = Grid([[1, 1]])
    rules = abstract([inp, out])
    assert rules and rules[0].meta.get("fallback_reason") == "no_rule_found"
