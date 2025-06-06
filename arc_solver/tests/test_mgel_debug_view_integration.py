import sys
import subprocess
from pathlib import Path



CASES = [
    ("color_replacement_input.json", "color_replacement_target.json"),
    ("shape_mismatch_input.json", "shape_mismatch_target.json"),
    ("silent_failure_input.json", "silent_failure_target.json"),
    ("multi_step_input.json", "multi_step_target.json"),
    ("memory_generalization_input.json", "memory_generalization_target.json"),
]


def _run_case(tmp_path, case_name, target_name, capsys):
    in_path = Path(__file__).with_name("integration_cases") / case_name
    out_path = Path(__file__).with_name("integration_cases") / target_name
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "mgel_debug_view.py"),
        "--manual_input",
        str(in_path),
        "--manual_target",
        str(out_path),
        "--trace",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    captured = result.stdout
    assert "Prediction Score" in captured or "No symbolic rules" in captured


def test_integration_cases(tmp_path, monkeypatch, capsys):
    for inp, out in CASES:
        _run_case(tmp_path, inp, out, capsys)
