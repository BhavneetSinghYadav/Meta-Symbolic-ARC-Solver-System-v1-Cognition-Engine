from arc_solver.src.introspection.trace_builder import build_trace
from arc_solver.src.introspection.introspective_validator import validate
from arc_solver.src.introspection.narrator_llm import narrate

def test_trace_builder_empty():
    assert build_trace([]) == []


def test_trace_builder_simple():
    trace = build_trace(["a", "b"])
    assert trace == [
        {"step": 0, "action": "a"},
        {"step": 1, "action": "b"},
    ]


def test_validator_accepts_simple_plan():
    assert validate(["a", "b"]) is True


def test_validator_rejects_invalid():
    assert validate(["", "b"]) is False


def test_narrator_fallback():
    assert narrate(["move", "paint"]) == "move -> paint"
