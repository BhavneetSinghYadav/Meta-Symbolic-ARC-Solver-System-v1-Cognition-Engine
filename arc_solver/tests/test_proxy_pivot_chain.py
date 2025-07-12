from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule


def _mirror_step():
    return SymbolicRule(
        transformation=Transformation(
            TransformationType.FUNCTIONAL,
            params={"op": "mirror_tile", "axis": "horizontal", "repeats": "2"},
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )


def _rotate_step():
    return SymbolicRule(
        transformation=Transformation(
            TransformationType.ROTATE,
            params={"cx": "1", "cy": "1", "angle": "90"},
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )


def _dilate_step():
    return SymbolicRule(
        transformation=Transformation(
            TransformationType.FUNCTIONAL,
            params={"op": "dilate_zone", "zone": "1"},
        ),
        source=[Symbol(SymbolType.ZONE, "1")],
        target=[Symbol(SymbolType.ZONE, "1")],
    )


def _draw_step():
    return SymbolicRule(
        transformation=Transformation(
            TransformationType.FUNCTIONAL,
            params={"op": "draw_line", "p1": "(0,0)", "p2": "(1,1)", "color": "2"},
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
    )


def test_mirror_then_rotate_proxy():
    comp = CompositeRule([_mirror_step(), _rotate_step()])
    proxy = comp.as_symbolic_proxy()
    assert proxy.meta.get("pivot_chain") == [None, "1,1"]


def test_dilate_then_draw_proxy():
    comp = CompositeRule([_dilate_step(), _draw_step()])
    proxy = comp.as_symbolic_proxy()
    assert proxy.meta.get("zone_chain")[0] == ("1", "1")
    assert proxy.meta.get("zone_scope_chain")[0] == (["1"], ["1"])
