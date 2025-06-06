from __future__ import annotations

"""Lightweight regime classifier heuristics with task signature memory."""

from enum import Enum, auto
from pathlib import Path
import csv
import json
import math
import logging
from typing import Dict, List, Tuple, Optional

from arc_solver.src.core.grid import Grid
from arc_solver.src.utils.grid_utils import compute_grid_entropy
from arc_solver.src.segment.segmenter import zone_overlay
from arc_solver.src.abstractions.abstractor import _find_translation
from arc_solver.src.utils import config_loader


class RegimeType(Enum):
    """Enumeration of coarse task regimes."""

    SymbolicallyTractable = auto()
    LikelyConflicted = auto()
    Fragmented = auto()
    RequiresHeuristic = auto()
    Unknown = auto()


_LOG_PATH = Path("logs/regime_log.csv")
_OVERRIDE_LOG = Path("logs/regime_override_log.json")
_SIGNATURE_INDEX = Path("logs/task_signature_index.json")


def _grid_entropy(grid: Grid) -> float:
    """Alias for :func:`compute_grid_entropy` (for backward compatibility)."""
    return compute_grid_entropy(grid)


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
    ent_in: List[float] = []
    ent_out: List[float] = []
    diffs: List[float] = []
    symmetry_in: List[int] = []
    symmetry_out: List[int] = []
    translations: List[int] = []
    zone_scores: List[float] = []
    rotation_scores: List[int] = []
    sparsities: List[float] = []
    dominants: List[float] = []
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
                ent_in.append(_grid_entropy(inp))
                ent_out.append(_grid_entropy(out))
            except Exception as exc:  # pragma: no cover - safety
                if logger and warning_count < max_warnings:
                    logger.warning("entropy failed for pair %d: %s", idx, exc)
                    warning_count += 1
                ent_in.append(0.0)
                ent_out.append(0.0)
            if inp.data == inp.flip_horizontal().data:
                symmetry_in.append(1)
            else:
                symmetry_in.append(0)
            if out.data == out.flip_horizontal().data:
                symmetry_out.append(1)
            else:
                symmetry_out.append(0)
            colors.update(inp.count_colors().keys())
            colors.update(out.count_colors().keys())
            try:
                translations.append(1 if _find_translation(inp, out) else 0)
            except Exception:
                translations.append(0)
            try:
                ov_in = zone_overlay(inp)
                ov_out = zone_overlay(out)
                matches = 0
                total_cells = h * w
                for r in range(h):
                    for c in range(w):
                        if ov_in[r][c] == ov_out[r][c]:
                            matches += 1
                zone_scores.append(matches / total_cells)
            except Exception:
                zone_scores.append(0.0)
            try:
                rot_match = 0
                for t in range(4):
                    if inp.rotate90(t).data == out.data:
                        rot_match = 1
                        break
                rotation_scores.append(rot_match)
            except Exception:
                rotation_scores.append(0)
            try:
                zero = sum(1 for r in range(h) for c in range(w) if out.get(r, c) == 0)
                sparsities.append(zero / (h * w))
                counts = out.count_colors()
                if counts:
                    dominants.append(max(counts.values()) / (h * w))
                else:
                    dominants.append(0.0)
            except Exception:
                sparsities.append(0.0)
                dominants.append(0.0)
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

    raw = {
        "entropy_in": sum(ent_in) / len(ent_in) if ent_in else 0.0,
        "entropy_out": sum(ent_out) / len(ent_out) if ent_out else 0.0,
        "grid_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "translation_score": sum(translations) / len(translations) if translations else 0.0,
        "io_zone_alignment_score": sum(zone_scores) / len(zone_scores) if zone_scores else 0.0,
        "rotation_invariance_score": sum(rotation_scores) / len(rotation_scores) if rotation_scores else 0.0,
        "output_sparsity": sum(sparsities) / len(sparsities) if sparsities else 0.0,
        "dominant_symbol_ratio": sum(dominants) / len(dominants) if dominants else 0.0,
        "symmetry_diff": abs(sum(symmetry_in)/len(symmetry_in) - sum(symmetry_out)/len(symmetry_out)) if symmetry_in and symmetry_out else 0.0,
    }

    values = list(raw.values())
    if values:
        vmin = min(values)
        vmax = max(values)
        if vmax - vmin > 0:
            norm = {k: (v - vmin) / (vmax - vmin) for k, v in raw.items()}
        else:
            norm = {k: 0.0 for k in raw}
    else:
        norm = raw

    return norm


def predict_regime_category(signature: Dict[str, float]) -> RegimeType:
    """Return a coarse regime label based on ``signature`` values."""
    if not signature:
        return RegimeType.Unknown

    th = config_loader.META_CONFIG.get("regime_thresholds", {})
    ent_low = float(th.get("entropy_low", 0.1))
    zone_cut = float(th.get("zone_alignment_cutoff", 0.3))
    sym_cut = float(th.get("symmetry_cutoff", 0.5))

    entropy = signature.get("entropy_in", 0.0)
    io_diff = signature.get("grid_diff", 0.0)
    zone = signature.get("io_zone_alignment_score", 1.0)
    sparsity = signature.get("output_sparsity", 1.0)
    dominant = signature.get("dominant_symbol_ratio", 0.0)
    translation_score = signature.get("translation_score", 0.0)
    sym_diff = signature.get("symmetry_diff", 0.0)

    if entropy < ent_low and io_diff < 0.05:
        label = RegimeType.SymbolicallyTractable
    elif zone < zone_cut or sparsity < 0.2:
        label = RegimeType.RequiresHeuristic
    elif dominant > 0.8 and translation_score > 0.6:
        label = RegimeType.Fragmented
    elif sym_diff > sym_cut:
        label = RegimeType.LikelyConflicted
    else:
        label = RegimeType.Unknown

    conf = regime_confidence_score(signature, label)
    if conf < 0.55:
        return RegimeType.Unknown
    return label


def score_abstraction_likelihood(signature: Dict[str, float]) -> float:
    """Return a crude probability that symbolic abstraction will succeed."""
    if not signature:
        return 0.5
    score = 1.0
    score -= min(1.0, signature.get("avg_entropy", 0.0) / 5)
    score -= min(1.0, signature.get("avg_diff", 0.0))
    score -= min(1.0, signature.get("num_colors", 0) / 10)
    return max(0.0, min(1.0, score))


_REGIME_WEIGHT_MAP: Dict[RegimeType, List[float]] = {
    RegimeType.SymbolicallyTractable: [1.0] * 9,
    RegimeType.RequiresHeuristic: [0.5] * 9,
    RegimeType.Fragmented: [0.7] * 9,
    RegimeType.LikelyConflicted: [0.6] * 9,
    RegimeType.Unknown: [0.0] * 9,
}


def regime_confidence_score(signature: Dict[str, float], regime: RegimeType) -> float:
    """Return a confidence score for ``regime`` given signature vector."""
    vec = list(signature.values())
    weights = _REGIME_WEIGHT_MAP.get(regime, [1.0] * len(vec))
    dot = sum(v * w for v, w in zip(vec, weights))
    return 1.0 / (1.0 + math.exp(-dot))


def log_regime_decision(
    task_id: str,
    signature: Dict[str, float],
    regime: RegimeType,
    confidence: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Record regime decision metadata."""
    if logger:
        logger.info(
            f"Task {task_id}: Signature={signature}, Regime={regime.name}, Confidence={confidence:.2f}"
        )
    log_regime(task_id, signature, regime, confidence)


def audit_regime_correction(task_id: str, regime: RegimeType, final_score: float) -> RegimeType:
    """Return possibly corrected regime based on final solver outcome."""
    if final_score == 0 and regime is RegimeType.SymbolicallyTractable:
        new_regime = RegimeType.RequiresHeuristic
        data = {"task_id": task_id, "regime": new_regime.name}
        try:
            _OVERRIDE_LOG.parent.mkdir(exist_ok=True)
            if _OVERRIDE_LOG.exists():
                entries = json.loads(_OVERRIDE_LOG.read_text())
                if not isinstance(entries, list):
                    entries = []
            else:
                entries = []
            entries.append(data)
            _OVERRIDE_LOG.write_text(json.dumps(entries))
        except Exception:
            pass
        return new_regime
    return regime


def _load_signature_index() -> List[Dict[str, object]]:
    if _SIGNATURE_INDEX.exists():
        try:
            return json.loads(_SIGNATURE_INDEX.read_text())
        except Exception:
            return []
    return []


def _save_signature_index(entries: List[Dict[str, object]]) -> None:
    _SIGNATURE_INDEX.parent.mkdir(exist_ok=True)
    _SIGNATURE_INDEX.write_text(json.dumps(entries))


def add_signature_to_index(task_id: str, signature: Dict[str, float], regime: RegimeType) -> None:
    entries = _load_signature_index()
    entries.append({"task_id": task_id, "signature": list(signature.values()), "regime": regime.name})
    _save_signature_index(entries)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_closest_signature(sig: Dict[str, float], threshold: float = 0.95) -> Optional[str]:
    """Return regime label from memory if a close signature exists."""
    vec = list(sig.values())
    entries = _load_signature_index()
    best_score = 0.0
    best_label: Optional[str] = None
    for e in entries:
        try:
            other = [float(x) for x in e.get("signature", [])]
            score = _cosine(vec, other)
            if score > best_score:
                best_score = score
                best_label = e.get("regime")
        except Exception:
            continue
    if best_score >= threshold:
        return best_label
    return None


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
    "regime_confidence_score",
    "log_regime_decision",
    "audit_regime_correction",
    "add_signature_to_index",
    "find_closest_signature",
    "log_regime",
]
