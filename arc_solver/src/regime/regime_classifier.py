from __future__ import annotations

"""Lightweight regime classifier heuristics."""

from enum import Enum, auto
from pathlib import Path
import csv
import math
import logging
from typing import Dict, List, Tuple

from arc_solver.src.core.grid import Grid


class RegimeType(Enum):
    """Enumeration of coarse task regimes."""

    SymbolicallyTractable = auto()
    LikelyConflicted = auto()
    Fragmented = auto()
    RequiresHeuristic = auto()
    Unknown = auto()


_LOG_PATH = Path("logs/regime_log.csv")


def _grid_entropy(grid: Grid) -> float:
    counts = grid.count_colors()
    total = sum(counts.values())
    ent = 0.0
    for v in counts.values():
        if v == 0:
            continue
        p = v / total
        ent -= p * math.log2(p)
    return ent


def compute_task_signature(
    train_pairs: List[Tuple[Grid, Grid]],
    *,
    logger: logging.Logger | None = None,
) -> Dict[str, float]:
    """Return simple statistics summarising the training examples.

    Any malformed or misaligned pairs are skipped with a warning.
    """
    if not train_pairs:
        return {}

    sizes: List[int] = []
    entropies: List[float] = []
    diffs: List[float] = []
    symmetry = 0
    colors = set()

    max_warnings = 5
    warning_count = 0

    for idx, (inp, out) in enumerate(train_pairs):
        try:
            if inp is None or out is None:
                raise ValueError("pair contains None")
            if inp.shape() != out.shape():
                if logger and warning_count < max_warnings:
                    logger.warning(
                        "pair %d shape mismatch %s vs %s, skipping",
                        idx,
                        inp.shape(),
                        out.shape(),
                    )
                    warning_count += 1
                continue
            h, w = inp.shape()
            if h == 0 or w == 0:
                if logger and warning_count < max_warnings:
                    logger.warning("pair %d has empty grid, skipping", idx)
                    warning_count += 1
                continue
            sizes.append(h * w)
            try:
                entropies.append(_grid_entropy(inp))
                entropies.append(_grid_entropy(out))
            except Exception as exc:  # pragma: no cover - safety
                if logger and warning_count < max_warnings:
                    logger.warning("entropy failed for pair %d: %s", idx, exc)
                    warning_count += 1
                entropies.append(0.0)
                entropies.append(0.0)
            if inp.data == inp.flip_horizontal().data or inp.data == inp.flip_horizontal().flip_horizontal().data:
                symmetry += 1
            colors.update(inp.count_colors().keys())
            colors.update(out.count_colors().keys())
            try:
                diff = sum(
                    1
                    for r in range(h)
                    for c in range(w)
                    if inp.get(r, c) != out.get(r, c)
                )
                diffs.append(diff / (h * w))
            except Exception as exc:  # pragma: no cover - safety
                if logger and warning_count < max_warnings:
                    logger.warning("diff failed for pair %d: %s", idx, exc)
                    warning_count += 1
                diffs.append(0.0)
        except Exception as exc:  # pragma: no cover - safety
            if logger and warning_count < max_warnings:
                logger.warning("error processing pair %d: %s", idx, exc)
                warning_count += 1
            continue

    if not sizes:
        return {}

    return {
        "avg_size": sum(sizes) / len(sizes),
        "avg_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "symmetry_ratio": symmetry / len(train_pairs),
        "avg_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "num_colors": len(colors),
    }


def predict_regime_category(signature: Dict[str, float]) -> RegimeType:
    """Return a coarse regime label based on ``signature`` values."""
    if not signature:
        return RegimeType.Unknown
    if signature["avg_size"] <= 9 and signature["avg_entropy"] < 1.0:
        return RegimeType.SymbolicallyTractable
    if signature["avg_entropy"] > 2.5 and signature["avg_size"] >= 400:
        return RegimeType.Fragmented
    if signature["avg_diff"] > 0.5 and signature["num_colors"] > 6:
        return RegimeType.RequiresHeuristic
    if signature["avg_diff"] > 0.4:
        return RegimeType.LikelyConflicted
    return RegimeType.SymbolicallyTractable


def score_abstraction_likelihood(signature: Dict[str, float]) -> float:
    """Return a crude probability that symbolic abstraction will succeed."""
    if not signature:
        return 0.5
    score = 1.0
    score -= min(1.0, signature.get("avg_entropy", 0.0) / 5)
    score -= min(1.0, signature.get("avg_diff", 0.0))
    score -= min(1.0, signature.get("num_colors", 0) / 10)
    return max(0.0, min(1.0, score))


def log_regime(task_id: str, signature: Dict[str, float], regime: RegimeType, score: float) -> None:
    """Append regime statistics to the log csv."""
    _LOG_PATH.parent.mkdir(exist_ok=True)
    headers = ["task_id", "regime", "score"] + sorted(signature)
    row = [task_id, regime.name, f"{score:.3f}"] + [f"{signature[k]:.3f}" for k in sorted(signature)]
    write_header = not _LOG_PATH.exists()
    with _LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)


__all__ = [
    "RegimeType",
    "compute_task_signature",
    "predict_regime_category",
    "score_abstraction_likelihood",
    "log_regime",
]
