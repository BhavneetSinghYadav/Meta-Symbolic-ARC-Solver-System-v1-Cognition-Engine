from pathlib import Path
from arc_solver.src.memory.memory_store import retrieve_similar_signatures
import json


def test_invalid_rule_skipped(tmp_path):
    mem_path = tmp_path / "mem.json"
    data = [{
        "task_id": "t1",
        "signature": "sig",
        "rules": ["REPLACE [COLOR=10] -> [COLOR=1]"]
    }]
    mem_path.write_text(json.dumps(data))
    retrieved = retrieve_similar_signatures("sig", mem_path)
    assert retrieved == []
