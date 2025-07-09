from arc_solver.src.symbolic.vocabulary import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.executor.dependency import rule_dependency_graph


def _rule(src: int, tgt: int) -> SymbolicRule:
    return SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, str(src))],
        target=[Symbol(SymbolType.COLOR, str(tgt))],
    )


def test_composite_rule_dependency_pass():
    r1 = _rule(1, 2)
    r2 = _rule(2, 3)
    comp = CompositeRule([r1, r2])
    graph = rule_dependency_graph([comp, r1])
    assert isinstance(graph, dict)
