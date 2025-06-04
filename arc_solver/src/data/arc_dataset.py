"""Dataset utilities for the Abstraction and Reasoning Corpus (ARC)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from arc_solver.src.core.grid import Grid


def load_arc_task(path: str | Path) -> Dict[str, List[Dict[str, List[List[int]]]]]:
    """Load a single ARC task from ``path`` and return its dictionary structure.

    Parameters
    ----------
    path:
        Path to a ``.json`` file containing an ARC task.

    Returns
    -------
    dict
        Parsed JSON describing the ARC task.
    """

    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


class ARCDataset(Iterable[Dict[str, List[Dict[str, List[List[int]]]]]]):
    """Iterate over all ARC tasks contained in a directory.

    The dataset can also index tasks by their ``task_id`` which corresponds to
    the JSON filename without the extension.  When iterated over each yielded
    task dictionary includes an additional ``"id"`` field containing this
    identifier.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __iter__(self) -> Iterator[Dict[str, List[Dict[str, List[List[int]]]]]]:
        for json_file in sorted(self.root.glob("*.json")):
            task = load_arc_task(json_file)
            task["id"] = json_file.stem
            yield task

    def __getitem__(self, task_id: str) -> Dict[str, List[Dict[str, List[List[int]]]]]:
        """Return the task dictionary for ``task_id``."""
        path = self.root / f"{task_id}.json"
        if not path.exists():
            raise KeyError(task_id)
        task = load_arc_task(path)
        task["id"] = task_id
        return task

    @staticmethod
    def to_grids(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List]:
        """Convert the raw JSON task to one containing :class:`Grid` objects."""

        train = [
            (Grid(pair["input"]), Grid(pair["output"])) for pair in task.get("train", [])
        ]
        test_pairs = task.get("test", [])
        test_inputs = [Grid(p["input"]) for p in test_pairs]
        test_outputs = [Grid(p["output"]) for p in test_pairs if "output" in p]
        return {
            "id": task.get("id"),
            "train": train,
            "test_inputs": test_inputs,
            "test_outputs": test_outputs,
        }
