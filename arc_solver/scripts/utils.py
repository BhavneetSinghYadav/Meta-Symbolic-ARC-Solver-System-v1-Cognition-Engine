from __future__ import annotations

"""Helper utilities for command line scripts."""

from pathlib import Path
import sys
from typing import Iterable, Tuple, Dict, Any

from arc_solver.src.data.arc_dataset import load_arc_task


def iter_arc_task_files(directory: Path) -> Iterable[Tuple[str, Dict[str, Any], bool]]:
    """Yield ``(task_id, json_obj, skipped)`` for each task in ``directory``.

    Files missing a ``test`` field or with an empty ``test`` list are skipped and
    logged. ``skipped`` indicates whether the task was ignored.
    """
    for path in sorted(Path(directory).glob("*.json")):
        tid = path.stem
        try:
            task = load_arc_task(path)
        except Exception as exc:
            print(f"[ERROR] Failed to load {tid}: {exc}", file=sys.stderr)
            continue
        if not task.get("test"):
            print(f"[SKIP] Task {tid} — no test set available.", file=sys.stderr)
            yield tid, task, True
            continue
        yield tid, task, False
