from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.symbolic.mutation import mutate_rule


# --- translation -----------------------------------------------------------
tr_rule = SymbolicRule(
    transformation=Transformation(TransformationType.TRANSLATE, {"dx": "1", "dy": "0"}),
    source=[Symbol(SymbolType.REGION, "All")],
    target=[Symbol(SymbolType.REGION, "All")],
)
mutants = mutate_rule(tr_rule)
assert any(m.transformation.params.get("dx") == "2" for m in mutants)

# --- colour swap ----------------------------------------------------------
rep_rule = SymbolicRule(
    transformation=Transformation(TransformationType.REPLACE),
    source=[Symbol(SymbolType.COLOR, "1")],
    target=[Symbol(SymbolType.COLOR, "2")],
)
mutants = mutate_rule(rep_rule)
assert any(m.source[0].value == "2" and m.target[0].value == "1" for m in mutants)

# --- composite reorder ----------------------------------------------------
r1 = SymbolicRule(
    transformation=Transformation(TransformationType.REPLACE),
    source=[Symbol(SymbolType.COLOR, "1")],
    target=[Symbol(SymbolType.COLOR, "2")],
)
r2 = SymbolicRule(
    transformation=Transformation(TransformationType.REPLACE),
    source=[Symbol(SymbolType.COLOR, "2")],
    target=[Symbol(SymbolType.COLOR, "3")],
)
comp = CompositeRule([r1, r2])
mutants = mutate_rule(comp)
assert any(isinstance(m, CompositeRule) and m.steps[0].target[0].value == "3" for m in mutants)

# --- noop injection -------------------------------------------------------
mutants = mutate_rule(r1)
assert any(isinstance(m, CompositeRule) and len(m.steps) == 2 for m in mutants)
