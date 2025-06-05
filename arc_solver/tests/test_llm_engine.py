from arc_solver.src.symbolic import Symbol, SymbolType, SymbolicRule, Transformation, TransformationType
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "arc_solver" / "src" / "introspection" / "llm_engine.py"
spec = importlib.util.spec_from_file_location("llm_engine", MODULE_PATH)
llm_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_engine)
local_refine_program = llm_engine.local_refine_program


def _simple_rule() -> SymbolicRule:
    return SymbolicRule(
        Transformation(TransformationType.REPLACE),
        source=[Symbol(SymbolType.COLOR, "1")],
        target=[Symbol(SymbolType.COLOR, "2")],
    )


class DummyTrace:
    def __init__(self, rule):
        self.rule = rule
    def __str__(self) -> str:
        return "dummy"


def test_local_refine():
    rule = _simple_rule()
    trace = DummyTrace(rule)
    improved = local_refine_program(trace, "feedback")
    assert isinstance(improved, list)
