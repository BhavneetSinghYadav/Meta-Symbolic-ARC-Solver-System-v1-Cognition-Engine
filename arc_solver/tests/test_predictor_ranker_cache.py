import pytest

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.predictor import select_best_program
from arc_solver.src.search.rule_ranker import rank_rule_sets
from arc_solver.src.memory.policy_cache import PolicyCache
from arc_solver.src.symbolic import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)


def _color_rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_select_best_program_with_conflict():
    grid_in = Grid([[1]])
    grid_out = Grid([[2]])
    correct = [_color_rule(1, 2)]
    conflicting = [_color_rule(1, 3), _color_rule(1, 4)]
    best = select_best_program(grid_in, grid_out, [conflicting, correct])
    assert best == correct


def test_policy_cache_equivalent_ruleset():
    cache = PolicyCache()
    rs = [_color_rule(1, 2)]
    cache.add_failure("t", rs)
    assert cache.is_failed("t", [_color_rule(1, 2)])


def test_policy_cache_distinct_ruleset():
    cache = PolicyCache()
    cache.add_failure("t", [_color_rule(1, 2)])
    assert not cache.is_failed("t", [_color_rule(1, 3)])


def test_rule_ranker_uses_cache():
    cache = PolicyCache()
    bad = [_color_rule(1, 2)]
    good = [_color_rule(1, 3)]
    cache.add_failure("task", bad)
    ranked = rank_rule_sets([bad, good], cache, "task")
    assert ranked[0] == good
