from pathlib import Path
import sys
import json
from arc_solver.scripts.run_agi_solver import main


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
    assert data["00000001"]["output"] == [[1]]

