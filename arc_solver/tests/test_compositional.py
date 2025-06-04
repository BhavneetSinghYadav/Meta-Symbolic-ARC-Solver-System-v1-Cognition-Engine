from arc_solver.src.core.grid import Grid
from arc_solver.src.scoring import run_pipeline
from arc_solver.src.symbolic.vocabulary import TransformationType


def test_run_pipeline_color_replace():
    inp = Grid([[1, 1]])
    out = Grid([[2, 2]])
    rules, just = run_pipeline([(inp, out)])
    assert len(rules) >= 1
    assert rules[0].transformation.ttype is TransformationType.REPLACE
    rid = repr(rules[0])
    assert rid in just
    assert just[rid]["support_count"] == 1
