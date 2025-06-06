import json
from arc_solver.src.memory.memory_store import load_memory, match_signature, get_last_load_stats
from arc_solver.src.utils import config_loader


def test_load_stats(tmp_path):
    path = tmp_path / "mem.json"
    data = [
        {"task_id": "t", "signature": "s", "rules": ["REPLACE [COLOR=0] -> [COLOR=1]", "BAD"], "score": 0.5},
        {"invalid": True}
    ]
    path.write_text(json.dumps(data))
    load_memory(path)
    loaded, discarded = get_last_load_stats()
    assert loaded == 1
    assert discarded >= 1


def test_injection_filter_override(tmp_path):
    path = tmp_path / "mem.json"
    entry = {
        "task_id": "t1",
        "signature": [1, 2, 3],
        "rules": ["REPLACE [COLOR=0] -> [COLOR=1]"],
        "score": 0.5,
        "constraints": {"shape": (2, 2)},
    }
    path.write_text(json.dumps([entry]))
    constraints = {"shape": (3, 3)}
    config_loader.set_ignore_memory_shape_constraint(True)
    matches = match_signature([1, 2, 3], path, threshold=0.8, constraints=constraints)
    assert matches
    config_loader.set_ignore_memory_shape_constraint(False)
