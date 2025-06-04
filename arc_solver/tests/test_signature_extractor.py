import json
from arc_solver.src.utils.signature_extractor import extract_task_signature, similarity_score


def test_extract_signature():
    task = json.loads(open('arc_solver/tests/sample_task.json').read())
    sig = extract_task_signature(task)
    assert 'colors' in sig
    assert similarity_score(sig, sig) == 1.0
