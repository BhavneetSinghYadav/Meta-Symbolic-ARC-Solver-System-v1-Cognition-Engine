from pathlib import Path
import json
from arc_solver.src.memory.memory_store import (
    save_rule_program,
    load_memory,
    match_signature,
    extract_task_constraints,
)
from arc_solver.src.symbolic.rule_language import parse_rule
from arc_solver.src.utils import config_loader


def test_memory_save_and_retrieve(tmp_path):
    mem_path = tmp_path / "mem.json"
    rule = parse_rule("REPLACE [COLOR=0] -> [COLOR=1]")
    save_rule_program("t1", "sigA", [rule], 0.9, mem_path)

    mem = load_memory(mem_path)
    assert mem and mem[0]["task_id"] == "t1"

    retrieved = match_signature("sigA", mem_path)
    assert retrieved
    assert retrieved[0]["rules"][0].transformation.ttype.name == "REPLACE"


def test_load_memory_filters_invalid(tmp_path):
    mem_path = tmp_path / "mem.json"
    data = [
        {
            "task_id": "t1",
            "signature": "sig",
            "rules": [
                "REPLACE [COLOR=0] -> [COLOR=1]",
                "REPLACE [COLOR=11] -> [COLOR=2]",
            ],
            "score": 0.5,
        }
    ]
    mem_path.write_text(json.dumps(data))

    mem = load_memory(mem_path)
    assert mem[0]["rules"] == ["REPLACE [COLOR=0] -> [COLOR=1]"]


def test_corrupted_entry(tmp_path):
    mem_path = tmp_path / "mem.json"
    mem_path.write_text("{bad json]", encoding="utf-8")
    mem = load_memory(mem_path)
    assert mem == []


def test_similarity_matching(tmp_path):
    mem_path = tmp_path / "mem.json"
    entry = {
        "task_id": "t1",
        "signature": [1.0, 0.0, 0.5],
        "rules": ["REPLACE [COLOR=0] -> [COLOR=1]"],
        "score": 0.9,
        "constraints": {"shape": (3, 3)},
    }
    mem_path.write_text(json.dumps([entry]))
    matches = match_signature([1.0, 0.0, 0.5], mem_path, threshold=0.9)
    assert matches and matches[0]["task_id"] == "t1"


def test_injection_filter(tmp_path):
    mem_path = tmp_path / "mem.json"
    entry = {
        "task_id": "t1",
        "signature": [1, 2, 3],
        "rules": ["REPLACE [COLOR=0] -> [COLOR=1]"],
        "score": 0.5,
        "constraints": {"shape": (2, 2)},
    }
    mem_path.write_text(json.dumps([entry]))
    constraints = {"shape": (3, 3)}
    config_loader.set_ignore_memory_shape_constraint(False)
    matches = match_signature([1, 2, 3], mem_path, threshold=0.8, constraints=constraints)
    assert matches == []
    config_loader.set_ignore_memory_shape_constraint(True)


def test_recall_with_lower_threshold(tmp_path):
    """Memory retrieval should succeed with similarity around 0.82."""
    mem_path = tmp_path / "mem.json"
    entry = {
        "task_id": "t_sim",
        "signature": [1.0, 0.0, 0.0],
        "rules": ["REPLACE [COLOR=0] -> [COLOR=1]"],
        "score": 0.9,
    }
    other = {
        "task_id": "t_other",
        "signature": [0.0, 1.0, 0.0],
        "rules": ["REPLACE [COLOR=2] -> [COLOR=3]"],
        "score": 0.4,
    }
    mem_path.write_text(json.dumps([entry, other]))
    matches = match_signature([0.75, 0.5, 0.0], mem_path, threshold=0.8)
    assert matches and matches[0]["task_id"] == "t_sim"

