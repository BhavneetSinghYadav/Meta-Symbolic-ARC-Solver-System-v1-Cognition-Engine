"""Fallback predictor for unseen tasks.

This policy pads the grid to a square using the most frequent colour but
first attempts a simple transformation (rotation or mirror) learned from
the training dataset.  The transform with the highest frequency is applied
before padding to approximate the target orientation when no rules are
available.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from arc_solver.src.core.grid import Grid


_TRANSFORMS = {
    "identity": lambda g: g,
    "rot90": lambda g: g.rotate90(1),
    "rot180": lambda g: g.rotate90(2),
    "rot270": lambda g: g.rotate90(3),
    "flip_h": lambda g: g.flip_horizontal(),
    "flip_v": lambda g: g.flip_vertical(),
}


def _rank_transforms() -> list[str]:
    """Return dataset frequency ranking of simple transforms."""
    counts: Counter[str] = Counter()
    root = Path(__file__).resolve().parents[2]
    path = root / "arc-agi_training_challenges.json"
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    for task in data.values():
        for pair in task.get("train", []):
            inp = Grid(pair["input"])
            out = Grid(pair["output"])
            if inp.shape() != out.shape():
                continue
            for name, fn in _TRANSFORMS.items():
                try:
                    if fn(inp).data == out.data:
                        counts[name] += 1
                        break
                except Exception:
                    pass
    ranked = [t for t, _ in counts.most_common() if t != "identity"]
    return ranked


_RANKED_TRANSFORMS = _rank_transforms()


def pad_to_expected(grid: Grid, fill: int) -> Grid:
    """Return ``grid`` padded to square shape filled with ``fill``."""
    h, w = grid.shape()
    size = max(h, w)
    new_data = [[fill for _ in range(size)] for _ in range(size)]
    for r in range(h):
        for c in range(w):
            new_data[r][c] = grid.get(r, c)
    return Grid(new_data)


def predict(grid: Grid) -> Grid:
    """Return a naive guess for the output grid."""

    try:
        grid.shape()
    except Exception:
        return grid

    transformed = grid
    if _RANKED_TRANSFORMS:
        tname = _RANKED_TRANSFORMS[0]
        try:
            transformed = _TRANSFORMS[tname](grid)
        except Exception:
            transformed = grid

    counts = transformed.count_colors()
    mode = max(counts, key=counts.get) if counts else 0
    return pad_to_expected(transformed, fill=mode)
