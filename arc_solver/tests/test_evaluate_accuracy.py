from pathlib import Path
from arc_solver.scripts.evaluate_accuracy import load_solutions


def test_load_solutions_dict_format():
    path = Path(__file__).with_name("sample_agi-solutions.json")
    sols = load_solutions(path)
    assert sols == {"00000001": [[[1]]]}


def test_load_solutions_list_format(tmp_path):
    src = Path(__file__).with_name("data").joinpath("sample_agi-solutions-list.json")
    path = tmp_path / "sol.json"
    path.write_text(src.read_text())
    sols = load_solutions(path)
    assert sols == {"00000001": [[[1]]]}
