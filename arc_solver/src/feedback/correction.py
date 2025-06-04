from __future__ import annotations

"""Generate high level error signals comparing predicted and target grids."""

from typing import List

from arc_solver.src.core.grid import Grid


def generate_error_signals(predicted: Grid, target: Grid) -> List[str]:
    """Return a list of textual feedback messages."""
    messages: List[str] = []
    if predicted.shape() != target.shape():
        messages.append("Shape mismatch")
        return messages
    h, w = predicted.shape()
    mismatches = 0
    for r in range(h):
        for c in range(w):
            pv = predicted.get(r, c)
            tv = target.get(r, c)
            if pv != tv:
                mismatches += 1
                if r >= h // 2 and c >= w // 2:
                    zone = "bottom right"
                elif r < h // 2 and c < w // 2:
                    zone = "top left"
                elif r < h // 2 and c >= w // 2:
                    zone = "top right"
                else:
                    zone = "bottom left"
                messages.append(f"Color mismatch at ({r},{c}) in {zone} zone")
    if mismatches > (h * w) // 2:
        messages.append("Too many elements added")
    if not mismatches:
        messages.append("Perfect match")
    return messages
