from arc_solver.src.executor.dependency import sort_rules_by_dependency
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


def test_dependency_ordering():
    r1 = _color_rule(1, 2)
    r2 = _color_rule(2, 3)
    ordered = sort_rules_by_dependency([r2, r1])
    assert ordered == [r1, r2]
