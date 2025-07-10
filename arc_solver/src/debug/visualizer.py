from __future__ import annotations

"""Grid comparison utilities for debugging purposes."""

from typing import List, Optional

from arc_solver.src.core.grid import Grid


def visual_diff_report(pred: Grid, target: Grid) -> str:
    """Return a human-readable report of mismatches between ``pred`` and ``target``.

    The report lists each cell that differs, providing coordinates and color
    values. At the end a summary of total errors and match ratio is appended.
    """

    report_lines: List[str] = []

    shape_pred = pred.shape()
    shape_target = target.shape()
    if shape_pred != shape_target:
        report_lines.append(
            f"Shape mismatch: predicted {shape_pred}, expected {shape_target}"
        )

    h = max(shape_pred[0], shape_target[0])
    w = max(shape_pred[1], shape_target[1])

    errors = 0
    for r in range(h):
        for c in range(w):
            a = pred.get(r, c, None)
            b = target.get(r, c, None)
            if a == b:
                continue

            pred_desc = "empty" if a is None else f"color {a}"
            tgt_desc = "empty" if b is None else f"color {b}"

            zone: Optional[str] = None

            def _zone(sym: Optional[object]) -> Optional[str]:
                if sym is None:
                    return None
                if isinstance(sym, list):
                    for s in sym:
                        if getattr(s, "type", None).__str__() == "ZONE":
                            return str(s.value)
                    return None
                if getattr(sym, "type", None).__str__() == "ZONE":
                    return str(getattr(sym, "value", None))
                return None

            if pred.overlay or target.overlay:
                zone = _zone(pred.overlay[r][c] if pred.overlay else None) or _zone(
                    target.overlay[r][c] if target.overlay else None
                )

            loc = f"({r},{c})"
            if zone:
                loc += f" zone {zone}"
            report_lines.append(
                f"Mismatch at {loc}: predicted {pred_desc}, expected {tgt_desc}"
            )
            errors += 1

    total_cells = h * w
    match_ratio = (total_cells - errors) / total_cells if total_cells else 1.0
    report_lines.append(f"Total errors: {errors}")
    report_lines.append(f"Match ratio: {match_ratio:.2f}")

    return "\n".join(report_lines)


__all__ = ["visual_diff_report"]
