from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.memory.memory_store import (
    validate_memory_program,
    inject_reliable_memory_rules,
)
from arc_solver.src.symbolic.overlay import zone_overlap_score
from arc_solver.src.utils.entropy import compute_zone_entropy_map


def _make_rule(zone: str, reliability: float = 1.0) -> SymbolicRule:
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
        condition={"zone": zone},
    )
    rule.meta["rule_reliability"] = reliability
    return rule


def test_zone_entropy_computation_and_overlap():
    grid = Grid(
        [
            [1, 2, 3, 0, 0, 0],
            [4, 5, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    ent_map = compute_zone_entropy_map(grid)
    assert ent_map
    score = zone_overlap_score("TopLeft", ent_map)
    assert 0.5 < score <= 1.0


def test_validate_memory_program_zone_mismatch():
    grid = Grid(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    rule = _make_rule("TopLeft")
    assert not validate_memory_program(rule, grid)


def test_inject_reliable_memory_rules_filters():
    grid = Grid(
        [
            [1, 2, 3, 0, 0, 0],
            [4, 5, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    good = _make_rule("TopLeft", reliability=0.8)
    bad = _make_rule("TopLeft", reliability=0.4)
    out = inject_reliable_memory_rules([good, bad], grid)
    assert good in out and bad not in out
