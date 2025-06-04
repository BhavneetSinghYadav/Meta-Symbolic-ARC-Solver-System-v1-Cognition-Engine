from arc_solver.src.executor.symbolic_executor import execute

def test_execute_identity():
    assert execute(None, [1]) == [1]
