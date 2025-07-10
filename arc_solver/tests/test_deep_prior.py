import json
from pathlib import Path

from arc_solver.src.memory.deep_prior_loader import (
    load_prior_templates,
    load_motifs,
    select_motifs,
)
from arc_solver.src.meta_generalizer import generalize_rule_program
from arc_solver.src.executor.full_pipeline import solve_task
from arc_solver.src.utils import config_loader
from arc_solver.src.symbolic.rule_language import rule_to_dsl


def test_load_prior_templates(tmp_path):
    data = [
        {
            "name": "ok",
            "rules": ["REPLACE [COLOR=0] -> [COLOR=1]"],
            "frequency": 1,
        },
        {"name": "bad", "rules": ["BAD"]},
    ]
    p = tmp_path / "priors.yaml"
    p.write_text(json.dumps(data))
    templates = load_prior_templates(p)
    assert len(templates) == 1
    assert rule_to_dsl(templates[0]["rules"][0]) == "REPLACE [COLOR=0] -> [COLOR=1]"


def test_generalize_rule_program():
    from arc_solver.src.symbolic.rule_language import parse_rule
    from arc_solver.src.executor.simulator import simulate_rules
    from arc_solver.src.core.grid import Grid
    rule = parse_rule("REPLACE [COLOR=0] -> [COLOR=1]")
    gens = generalize_rule_program([rule])
    assert gens and len(gens) >= 2
    simulate_rules(Grid([[0]]), gens)


def test_motif_matching():
    motifs = load_motifs(Path(__file__).resolve().parents[1] / "configs" / "motif_db.yaml")
    sel = select_motifs("2-colors_1x1_vsym", motifs)
    assert any("mirror" in m.get("abstract_tag", "") for m in sel)


def test_solve_task_with_priors():
    config_loader.set_reflex_override(True)
    task = json.loads(Path("arc_solver/tests/sample_task.json").read_text())
    preds, _, _, _ = solve_task(task)
    assert preds[0].data == [[0]]
    preds2, _, _, _ = solve_task(task, use_deep_priors=True)
    assert preds2[0].data == [[1]]
