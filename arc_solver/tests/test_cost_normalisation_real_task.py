import json
from pathlib import Path

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationNature,
    TransformationType,
)
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.executor.scoring import score_rule


def test_cost_normalisation_real_task():
    data = json.loads(Path("arc-agi_training_challenges.json").read_text())
    pair = data["00576224"]["train"][0]
    inp = Grid(pair["input"])
    out = Grid(pair["output"])

    repeat = SymbolicRule(
        transformation=Transformation(
            TransformationType.REPEAT,
            params={"kx": "3", "ky": "3"},
        ),
        source=[Symbol(SymbolType.REGION, "All")],
        target=[Symbol(SymbolType.REGION, "All")],
        nature=TransformationNature.SPATIAL,
    )

    noop = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "1")],
    )

    comp = CompositeRule([repeat, noop])

    single = score_rule(inp, out, repeat, return_trace=True)
    chained = score_rule(inp, out, comp, return_trace=True)

    assert chained["penalty"] > single["penalty"]
    assert chained["op_cost"] >= single["op_cost"]
