from arc_solver.src.abstractions.abstractor import (
    abstract,
    extract_color_change_rules,
)
from arc_solver.src.abstractions.rule_generator import generalize_rules
from arc_solver.src.symbolic import (
    rules_to_program,
    program_to_rules,
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.core.grid import Grid

def test_abstract_returns_list():
    assert isinstance(abstract([]), list)


def test_extract_color_change_rules_simple():
    grid_in = Grid([[1, 1], [2, 2]])
    grid_out = Grid([[3, 3], [2, 2]])
    rules = extract_color_change_rules(grid_in, grid_out)
    assert any(
        r.transformation.ttype is TransformationType.REPLACE
        and any(s.value == "1" for s in r.source)
        and any(t.value == "3" for t in r.target)
        for r in rules
    )


def test_generalize_rules_deduplicates():
    rule = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    assert len(generalize_rules([rule, rule])) == 1


def test_dsl_roundtrip():
    rule = SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )
    program = rules_to_program([rule])
    parsed = program_to_rules(program)
    assert parsed == [rule]

