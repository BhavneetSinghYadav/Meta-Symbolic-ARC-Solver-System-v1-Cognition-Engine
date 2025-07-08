from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic import SymbolicRule


def test_fallback_trigger():
    inp = Grid([[1]])
    out = Grid([[1, 1]])
    rules = abstract([inp, out])
    assert rules
    first_rule = next(
        (r for r in rules if isinstance(r, SymbolicRule) and r.meta.get("fallback_reason")),
        None,
    )
    if first_rule is not None:
        assert first_rule.meta.get("fallback_reason") in {"no_rule_found", "heuristic"}
