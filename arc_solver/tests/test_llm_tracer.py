from arc_solver.src.introspection.trace_builder import build_trace

def test_trace_builder_empty():
    assert build_trace([]) == []
