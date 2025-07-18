import pytest
from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.validator import (
    validate_color_dependencies,
    validate_color_lineage,
)
from arc_solver.simulator import ColorLineageTracker
from arc_solver.src.symbolic.vocabulary import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType
from arc_solver.src.symbolic.rule_language import CompositeRule


def _rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_validate_color_dependencies_pass():
    grid = Grid([[1, 2]])
    chain = [_rule(1, 1), _rule(2, 2)]
    assert validate_color_dependencies(chain, grid)


def test_validate_color_dependencies_fail():
    grid = Grid([[1, 2]])
    chain = [_rule(1, 3), _rule(3, 3)]
    assert not validate_color_dependencies(chain, grid)


def test_validate_color_lineage_recolor_restore():
    grid = Grid([[1]])
    comp = CompositeRule([_rule(1, 2), _rule(2, 1)])
    assert validate_color_lineage(comp, grid)


def test_validate_color_lineage_erase_restore():
    grid = Grid([[1]])
    step1 = _rule(1, 0)
    step2 = _rule(0, 1)
    comp = CompositeRule([step1, step2])
    assert validate_color_lineage(comp, grid)


def test_validate_color_lineage_copy_paste_restore():
    grid = Grid([[1]])
    translate = SymbolicRule(
        transformation=Transformation(
            TransformationType.TRANSLATE, params={"dx": "0", "dy": "0"}
        ),
        source=[],
        target=[],
    )
    comp = CompositeRule([_rule(1, 2), translate, _rule(2, 1)])
    assert validate_color_lineage(comp, grid)


def _repeat_rule(kx: int, ky: int) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(
            TransformationType.REPEAT,
            params={"kx": str(kx), "ky": str(ky)},
        ),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "1")],
    )


def test_training_color_override_accept():
    inp = Grid([[1]])
    comp = CompositeRule([
        _repeat_rule(2, 1),
        _rule(1, 2),
    ])
    tracker = ColorLineageTracker(inp)
    assert validate_color_dependencies(
        [comp],
        inp,
        training_colors={2},
        lineage_tracker=tracker,
    )


def test_training_color_mismatch_allowed_by_zone_remap():
    grid = Grid([[1]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.FUNCTIONAL, params={"op": "zone_remap"}),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        meta={"mapping": {0: 2}},
    )
    comp = CompositeRule([rule])
    assert validate_color_dependencies([comp], grid, training_colors={1})


def test_training_color_mismatch_without_transform_fails():
    grid = Grid([[1]])
    rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    comp = CompositeRule([rule])
    assert not validate_color_dependencies([comp], grid, training_colors={1})

