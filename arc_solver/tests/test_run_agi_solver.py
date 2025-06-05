from pathlib import Path
import sys
import json
from arc_solver.scripts.run_agi_solver import main
from arc_solver.src.utils import config_loader


def test_run_agi_solver_creates_submission(tmp_path, monkeypatch):
    data_dir = tmp_path
    out_file = tmp_path / "submission.json"
    # Only underscore filename provided
    src = Path(__file__).with_name("sample_agi-challenges.json")
    (data_dir / "arc-agi_training_challenges.json").write_text(src.read_text())

    argv = [
        "prog",
        "--split",
        "train",
        "--data_dir",
        str(data_dir),
        "--output",
        str(out_file),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "00000001" in data
    assert data["00000001"]["output"] == [[[[0]], [[0]]]]


def test_cli_flag_toggles_attention(tmp_path, monkeypatch):
    data_dir = tmp_path
    out_file = tmp_path / "submission.json"
    src = Path(__file__).with_name("sample_agi-challenges.json")
    (data_dir / "arc-agi_training_challenges.json").write_text(src.read_text())

    argv = [
        "prog",
        "--split",
        "train",
        "--data_dir",
        str(data_dir),
        "--output",
        str(out_file),
        "--use_structural_attention",
        "--attention_weight",
        "0.3",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    config_loader.set_use_structural_attention(False)
    config_loader.set_attention_weight(0.1)
    main()
    assert config_loader.USE_STRUCTURAL_ATTENTION is True
    assert config_loader.STRUCTURAL_ATTENTION_WEIGHT == 0.3

