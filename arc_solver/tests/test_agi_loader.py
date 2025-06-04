from pathlib import Path
from arc_solver.src.data.agi_loader import load_agi_tasks


def test_load_agi_tasks(tmp_path: Path) -> None:
    challenges = tmp_path / "ch.json"
    solutions = tmp_path / "sol.json"
    challenges.write_text(Path(__file__).with_name("sample_agi-challenges.json").read_text())
    solutions.write_text(Path(__file__).with_name("sample_agi-solutions.json").read_text())

    tasks = load_agi_tasks(challenges, solutions)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "00000001"
    assert task.train[0][0].data[0][0] == 0
    assert task.train[0][1].data[0][0] == 1
    assert task.test[0].data[0][0] == 0
    assert task.ground_truth[0].data[0][0] == 1


def test_load_agi_tasks_no_solutions(tmp_path: Path) -> None:
    challenges = tmp_path / "ch.json"
    challenges.write_text(Path(__file__).with_name("sample_agi-challenges.json").read_text())
    tasks = load_agi_tasks(challenges)
    assert tasks[0].ground_truth is None


def test_load_agi_tasks_wrapped_test(tmp_path: Path) -> None:
    challenges = tmp_path / "ch.json"
    challenges.write_text(Path(__file__).with_name("sample_agi-dict-test.json").read_text())
    tasks = load_agi_tasks(challenges)
    assert tasks[0].test[0].data == [[0]]

