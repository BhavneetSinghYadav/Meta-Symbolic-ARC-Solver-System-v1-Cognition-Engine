"""Dataset utilities for the Abstraction and Reasoning Corpus (ARC)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


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
    """Iterate over all ARC tasks contained in a directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __iter__(self) -> Iterator[Dict[str, List[Dict[str, List[List[int]]]]]]:
        for json_file in sorted(self.root.glob("*.json")):
            yield load_arc_task(json_file)
