from pathlib import Path
from arc_solver.src.memory.memory_store import save_rule_program, load_memory, retrieve_similar_signatures
from arc_solver.src.symbolic.rule_language import parse_rule


def test_memory_save_and_retrieve(tmp_path):
    mem_path = tmp_path / "mem.json"
    rule = parse_rule("REPLACE [COLOR=0] -> [COLOR=1]")
    save_rule_program("t1", "sigA", [rule], 0.9, mem_path)

    mem = load_memory(mem_path)
    assert mem and mem[0]["task_id"] == "t1"

    retrieved = retrieve_similar_signatures("sigA", mem_path)
    assert retrieved
    assert retrieved[0]["rules"][0].transformation.ttype.name == "REPLACE"
