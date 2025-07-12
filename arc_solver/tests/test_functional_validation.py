import pytest
from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    SymbolicRule,
    Transformation,
    TransformationType,
    Symbol,
    SymbolType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.executor.simulator import simulate_composite_safe
from arc_solver.src.executor.functional_utils import InvalidParameterError


def _functional_rule(op, params=None, meta=None):
    params = params or {"op": op}
    return SymbolicRule(
        transformation=Transformation(TransformationType.FUNCTIONAL, params=params),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        meta=meta or {},
    )


def test_rotate_about_point_invalid_pivot():
    grid = Grid([[1, 0], [0, 0]])
    rule = _functional_rule(
        "rotate_about_point", params={"op": "rotate_about_point", "cx": "3", "cy": "0", "angle": "90"}
    )
    comp = CompositeRule([rule])
    with pytest.raises(InvalidParameterError):
        simulate_composite_safe(grid, comp)


def test_zone_remap_unknown_zone():
    grid = Grid([[1, 1], [1, 1]])
    rule = _functional_rule(
        "zone_remap",
        params={"op": "zone_remap"},
        meta={"mapping": {3: 2}},
    )
    comp = CompositeRule([rule])
    with pytest.raises(InvalidParameterError):
        simulate_composite_safe(grid, comp)


def test_pattern_fill_missing_pattern():
    grid = Grid([[0, 0], [0, 0]])
    mask = Grid([[1, 0], [0, 1]])
    rule = _functional_rule(
        "pattern_fill",
        params={"op": "pattern_fill"},
        meta={"mask": mask},
    )
    comp = CompositeRule([rule])
    with pytest.raises(InvalidParameterError):
        simulate_composite_safe(grid, comp)
