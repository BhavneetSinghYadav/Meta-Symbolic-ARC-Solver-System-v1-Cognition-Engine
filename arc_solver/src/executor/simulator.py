from __future__ import annotations

"""Simple symbolic rule simulator for ARC grids."""

from typing import List, Optional
import logging
import math

from collections import Counter, defaultdict
from arc_solver.src.utils.logger import get_logger
from arc_solver.simulator import (
    log_rule_failure,
    rule_failures_log,
    summarize_skips_by_type,
    ColorLineageTracker,
)
from arc_solver.src.executor.color_lineage import ColorLineage
from arc_solver.src.executor.failure_logger import log_failure
from arc_solver.src.executor.sim_defer_gate import allow_pruning
from arc_solver.src.symbolic.vocabulary import validate_color_range, MAX_COLOR
from arc_solver.src.symbolic.rule_language import CompositeRule
from arc_solver.src.utils import config_loader
from arc_solver.src.utils.coverage import rule_coverage
from arc_solver.src.executor.validator import get_color_set

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import (
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from arc_solver.src.segment.segmenter import zone_overlay
from arc_solver.src.executor.dependency import (
    sort_rules_by_dependency,
    sort_rules_by_topology,
)
from arc_solver.src.utils.grid_utils import validate_grid


logger = get_logger(__name__)
CONFLICT_POLICY = config_loader.META_CONFIG.get("conflict_resolution", "most_frequent")

# Maximum allowed grid dimension before simulation aborts
MAX_GRID_DIM = 64



def simulate_composite_rule(
    grid: Grid, rule: CompositeRule, *, uncertainty_grid: list[list[int]] | None = None
) -> Grid:
    """Apply a :class:`CompositeRule` to ``grid`` sequentially."""
    out = Grid([row[:] for row in grid.data])
    for step in rule.steps:
        before = out
        out = safe_apply_rule(step, out, perform_checks=False)
        if out.shape() != before.shape() and uncertainty_grid is not None:
            _resize_grid_like(uncertainty_grid, out)
    return out


def grid_growth_forecast(
    step: SymbolicRule | CompositeRule, start_shape: tuple[int, int]
) -> tuple[int, int]:
    """Return expected grid size after applying ``step`` to a dummy grid."""

    h, w = start_shape
    dummy = Grid([[0 for _ in range(w)] for _ in range(h)])
    try:
        if isinstance(step, CompositeRule):
            for sub in step.steps:
                dummy = safe_apply_rule(sub, dummy, perform_checks=False)
        else:
            dummy = safe_apply_rule(step, dummy, perform_checks=False)
    except Exception:
        return start_shape
    return dummy.shape()


def simulate_composite_safe(
    grid: Grid, rule: CompositeRule, *, uncertainty_grid: list[list[int]] | None = None
) -> Grid:
    """Apply composite rule skipping invalid steps and validating size."""

    out = Grid([row[:] for row in grid.data])
    for idx, step in enumerate(rule.steps):
        forecast = grid_growth_forecast(step, out.shape())
        if forecast[0] > MAX_GRID_DIM or forecast[1] > MAX_GRID_DIM:
            log_failure(
                task_id=None,
                rule_id=str(rule),
                rule_type="composite",
                rule_steps=[str(s) for s in rule.steps],
                rejection_stage="simulation",
                failed_step_index=idx,
                reason="grid_expansion_failure",
            )
            raise ValidationError(
                f"grid would expand to {forecast}, exceeding {MAX_GRID_DIM}x{MAX_GRID_DIM}"
            )
        if uncertainty_grid is not None and forecast != out.shape():
            _resize_grid_like(uncertainty_grid, Grid([[0] * forecast[1] for _ in range(forecast[0])]))

        if not validate_rule_application(step, out):
            continue
        out = safe_apply_rule(step, out, perform_checks=False)
    return out


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


class ReflexOverrideException(Exception):
    """Raised when a rule violates a reflex constraint."""


class ValidationError(Exception):
    """Raised when a rule would lead to an invalid grid state."""


def _grid_contains(grid: Grid, value: int) -> bool:
    h, w = grid.shape()
    for r in range(h):
        for c in range(w):
            if grid.get(r, c) == value:
                return True
    return False


def validate_rule_application(rule: SymbolicRule, grid: Grid) -> bool:
    zone = rule.condition.get("zone") if rule.condition else None
    if zone:
        overlay = zone_overlay(grid)
        zones = {sym.value for row in overlay for sym in row if sym}
        if zone not in zones:
            return False
    src = next((s for s in rule.source if s.type is SymbolType.COLOR), None)
    if src is not None:
        try:
            if not _grid_contains(grid, int(src.value)):
                return False
        except Exception:
            return False
    tgt = next((s for s in rule.target if s.type is SymbolType.COLOR), None)
    if tgt is not None:
        try:
            if int(tgt.value) >= MAX_COLOR:
                return False
        except Exception:
            return False
    return True


def _resize_grid_like(base: list[list[int]], grid: Grid) -> None:
    """Resize ``base`` in-place to match the shape of ``grid``."""
    h, w = grid.shape()
    bh = len(base)
    bw = len(base[0]) if base else 0
    if bh < h:
        for _ in range(h - bh):
            base.append([0] * bw)
    if bw < w:
        for row in base:
            row.extend([0] * (w - bw))
    if bh < h and bw < w:
        for r in range(bh, h):
            base[r].extend([0] * (w - bw))


def mark_conflict(
    loc: tuple[int, int],
    uncertainty_grid: list[list[int]] | None = None,
    grid: Grid | None = None,
) -> None:
    """Mark a conflict at ``loc`` in ``uncertainty_grid``.

    If ``grid`` is provided and its shape exceeds ``uncertainty_grid`` in any
    dimension, ``uncertainty_grid`` is resized accordingly. This prevents
    ``IndexError`` when rules expand the working grid during simulation.
    """

    if uncertainty_grid is None:
        return

    r, c = loc

    if grid is not None:
        gh, gw = grid.shape()
        uh = len(uncertainty_grid)
        uw = len(uncertainty_grid[0]) if uh > 0 else 0
        if gh > uh or gw > uw:
            old_shape = (uh, uw)
            new_grid = [[0 for _ in range(gw)] for _ in range(gh)]
            for rr in range(uh):
                for cc in range(uw):
                    new_grid[rr][cc] = uncertainty_grid[rr][cc]
            uncertainty_grid[:] = new_grid
            logger.debug(
                "Resized uncertainty grid from %s to %s", old_shape, (gh, gw)
            )

    try:
        uh = len(uncertainty_grid)
        uw = len(uncertainty_grid[0]) if uh > 0 else 0
        if r >= uh:
            for _ in range(r - uh + 1):
                uncertainty_grid.append([0] * uw)
        if c >= uw:
            for row in uncertainty_grid:
                row.extend([0] * (c - uw + 1))
        uncertainty_grid[r][c] += 1
    except IndexError:  # pragma: no cover - safety fallback
        logger.warning("Failed to mark conflict at %s due to size mismatch", loc)


def visualize_uncertainty(uncertainty_grid: list[list[int]], output_path: str = "uncertainty_map.png") -> None:
    try:
        import matplotlib.pyplot as plt
        plt.imshow(uncertainty_grid, cmap="hot", interpolation="nearest")
        plt.axis("off")
        plt.savefig(output_path)
        plt.close()
    except Exception as exc:  # pragma: no cover - optional
        logger.warning(f"Could not visualize uncertainty: {exc}")


def validate_color_dependencies(
    rules: List[SymbolicRule | CompositeRule],
    grid: Grid,
    *,
    logger: logging.Logger | None = None,
    strict: bool = False,
    lineage_tracker: ColorLineageTracker | None = None,
    step_state: "ColorLineage" | None = None,
    task_id: str | None = None,
) -> List[SymbolicRule | CompositeRule]:
    """Return ``rules`` filtered by available colors.

    The grid is simulated in a copy to track which colors remain after each
    rule application. Rules whose required source colors are missing are
    skipped and a warning is logged. When ``strict`` is ``True`` the function
    aborts on the first invalid rule by raising ``ValueError``.
    """

    working = Grid([row[:] for row in grid.data])
    color_presence = {v for row in working.data for v in row}
    lineage: dict[int, str] = {}
    valid: list[SymbolicRule | CompositeRule] = []
    intermediate_grids: List[List[List[int]]] = []
    color_lineage: List[Set[int]] = []
    if step_state is None:
        step_state = ColorLineage(working)

    for rule in rules:
        if isinstance(rule, CompositeRule):
            preview = Grid([row[:] for row in working.data])
            try:
                for st in rule.steps:
                    preview = safe_apply_rule(st, preview, perform_checks=False)
                final_presence = {v for row in preview.data for v in row}
            except Exception:
                final_presence = {v for row in working.data for v in row}

            step_grid = Grid([row[:] for row in working.data])
            step_presence = {v for row in step_grid.data for v in row}
            valid_chain = True
            if step_state:
                step_state.record(step_grid)
            color_lineage.append(get_color_set(step_grid))
            intermediate_grids.append([row[:] for row in step_grid.data])
            for idx, step in enumerate(rule.steps):
                try:
                    required = {
                        int(s.value)
                        for s in step.source
                        if s.type is SymbolType.COLOR
                    }
                except ValueError:
                    if logger:
                        logger.warning(
                            f"Rule '{rule}' skipped – invalid color value"
                        )
                    if strict:
                        raise
                    valid_chain = False
                    break

                missing = [c for c in required if c not in step_presence]
                if missing:
                    for c in missing:
                        if lineage_tracker and c in lineage_tracker.removed_by:
                            info = f"; removed by prior rule '{lineage_tracker.removed_by[c]}'"
                            trace = lineage_tracker.get_lineage(c)
                            if logger:
                                logger.warning(
                                    f"Rule '{rule}' skipped – source color {c} no longer present{info}. Lineage for {c}: {trace}"
                                )
                        else:
                            info = (
                                f"; removed by prior rule '{lineage[c]}'" if c in lineage else ""
                            )
                            if logger:
                                logger.warning(
                                    f"Rule '{rule}' skipped – source color {c} no longer present{info}"
                                )
                    if final_presence.issuperset(missing):
                        pass
                    else:
                        log_failure(
                            task_id=task_id,
                            rule_id=str(rule),
                            rule_type="composite",
                            rule_steps=[str(s) for s in rule.steps],
                            rejection_stage="validation",
                            failed_step_index=idx,
                            reason="missing_color",
                            color_lineage=color_lineage + [get_color_set(step_grid)],
                            intermediate_grids=intermediate_grids + [[row[:] for row in step_grid.data]],
                        )
                        if strict:
                            raise ValueError(f"Missing colors for rule: {rule}")
                        valid_chain = False
                        break

                before_step = {v for row in step_grid.data for v in row}
                if logger:
                    logger.debug(
                        "validate step %d of %s: require %s, presence %s",
                        idx,
                        rule,
                        required,
                        step_presence,
                    )
                step_grid = safe_apply_rule(step, step_grid, perform_checks=False)
                if logger:
                    logger.debug(
                        "after step %d of %s: grid %s",
                        idx,
                        rule,
                        step_grid.data,
                    )
                step_presence = {v for row in step_grid.data for v in row}
                if step_state:
                    step_state.record(step_grid)
            color_lineage.append(get_color_set(step_grid))
            intermediate_grids.append([row[:] for row in step_grid.data])

            if not valid_chain:
                continue

            grid_before_apply = Grid([row[:] for row in working.data])
            before = {v for row in grid_before_apply.data for v in row}
            working = step_grid
            after = step_presence
        else:
            try:
                required = {
                    int(s.value)
                    for s in rule.source
                    if s.type is SymbolType.COLOR
                }
            except ValueError:
                if logger:
                    logger.warning(f"Rule '{rule}' skipped – invalid color value")
                if strict:
                    raise
                continue

            missing = [c for c in required if c not in color_presence]
            if missing:
                for c in missing:
                    if lineage_tracker and c in lineage_tracker.removed_by:
                        info = f"; removed by prior rule '{lineage_tracker.removed_by[c]}'"
                        trace = lineage_tracker.get_lineage(c)
                        if logger:
                            logger.warning(
                                f"Rule '{rule}' skipped – source color {c} no longer present{info}. Lineage for {c}: {trace}"
                            )
                    else:
                        info = f"; removed by prior rule '{lineage[c]}'" if c in lineage else ""
                        if logger:
                            logger.warning(
                                f"Rule '{rule}' skipped – source color {c} no longer present{info}"
                            )
                log_failure(
                    task_id=task_id,
                    rule_id=str(rule),
                    rule_type="atomic",
                    rule_steps=[str(rule)],
                    rejection_stage="validation",
                    failed_step_index=0,
                    reason="missing_color",
                    color_lineage=color_lineage + [get_color_set(working)],
                    intermediate_grids=intermediate_grids + [[row[:] for row in working.data]],
                )
                if strict:
                    raise ValueError(f"Missing colors for rule: {rule}")
                continue

            grid_before_apply = Grid([row[:] for row in working.data])
            before = {v for row in grid_before_apply.data for v in row}
            try:
                working = safe_apply_rule(rule, working, perform_checks=False)
            except Exception:
                pass
            after = {v for row in working.data for v in row}
            if step_state:
                step_state.record(working)

        removed = before - after
        for col in removed:
            lineage[col] = str(rule)
            if lineage_tracker:
                lineage_tracker.removed_by[col] = str(rule)
        if lineage_tracker:
            lineage_tracker.observe_rule(rule, grid_before_apply, working)
        color_presence = after
        valid.append(rule)
        color_lineage.append(get_color_set(working))
        intermediate_grids.append([row[:] for row in working.data])

    return valid


def check_symmetry_break(rule: SymbolicRule, grid: Grid, attention_mask: Optional[list[list[bool]]] = None) -> Grid:
    after = safe_apply_rule(rule, grid, attention_mask, perform_checks=False)
    if violates_symmetry(after, grid):
        raise ReflexOverrideException("Symmetry violated by rule")
    if breaks_training_constraint(after):
        raise ReflexOverrideException("Training constraint mismatch")
    return after


def _is_vertically_symmetric(grid: Grid) -> bool:
    h, w = grid.shape()
    for r in range(h):
        for c in range(w // 2):
            if grid.get(r, c) != grid.get(r, w - c - 1):
                return False
    return True


def _is_horizontally_symmetric(grid: Grid) -> bool:
    h, w = grid.shape()
    for c in range(w):
        for r in range(h // 2):
            if grid.get(r, c) != grid.get(h - r - 1, c):
                return False
    return True


def violates_symmetry(after: Grid, before: Grid) -> bool:
    return (
        (_is_vertically_symmetric(before) and not _is_vertically_symmetric(after))
        or (
            _is_horizontally_symmetric(before)
            and not _is_horizontally_symmetric(after)
        )
    )


def breaks_training_constraint(after: Grid) -> bool:
    # Placeholder constraint: ensure color values remain within 0-9
    h, w = after.shape()
    for r in range(h):
        for c in range(w):
            val = after.get(r, c)
            if val < 0 or val > 9:
                return True
    return False


def _apply_replace(
    grid: Grid,
    rule: SymbolicRule,
    attention_mask: Optional[List[List[bool]]] = None,
    *,
    lineage_tracker: ColorLineageTracker | None = None,
) -> Grid:
    src_color = None
    tgt_color = None
    for sym in rule.source:
        if sym.type is SymbolType.COLOR:
            try:
                src_color = int(sym.value)
            except ValueError:
                msg = f"Invalid symbol value: {sym.value}, skipping rule"
                logger.warning(msg)
                log_rule_failure(
                    rule,
                    failure_type="REPLACE",
                    skipped_due_to=sym.value,
                    message=msg,
                )
                return grid
            break
    for sym in rule.target:
        if sym.type is SymbolType.COLOR:
            try:
                tgt_color = int(sym.value)
            except ValueError:
                msg = f"Invalid symbol value: {sym.value}, skipping rule"
                logger.warning(msg)
                log_rule_failure(
                    rule,
                    failure_type="REPLACE",
                    skipped_due_to=sym.value,
                    message=msg,
                )
                return grid
            break
    if src_color is None or tgt_color is None:
        log_rule_failure(rule, failure_type="REPLACE", message="missing parameters")
        return grid
    if not validate_color_range(tgt_color):
        log_rule_failure(
            rule,
            failure_type="REPLACE",
            skipped_due_to=tgt_color,
            message="target color out of range",
        )
        return grid
    if not _grid_contains(grid, src_color):
        info = None
        if lineage_tracker and src_color in lineage_tracker.removed_by:
            info = lineage_tracker.get_lineage(src_color)
        msg = f"source color {src_color} not found"
        logger.warning(msg + "; skipping rule")
        log_rule_failure(
            rule,
            failure_type="REPLACE",
            skipped_due_to=src_color,
            message=msg,
            lineage=info,
        )
        return grid

    h, w = grid.shape()
    new_data = [row[:] for row in grid.data]
    zone = rule.condition.get("zone") if rule.condition else None
    overlay = zone_overlay(grid) if zone else None
    for r in range(h):
        for c in range(w):
            if attention_mask and not attention_mask[r][c]:
                continue
            if zone and (overlay[r][c] is None or overlay[r][c].value != zone):
                continue
            if new_data[r][c] == src_color:
                new_data[r][c] = tgt_color
    return Grid(new_data)


def _apply_translate(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]] = None
) -> Grid:
    try:
        dx = int(rule.transformation.params.get("dx", "0"))
        dy = int(rule.transformation.params.get("dy", "0"))
    except ValueError:
        log_rule_failure(
            rule,
            failure_type="TRANSLATE",
            message="invalid translation parameters",
        )
        return grid
    h, w = grid.shape()
    new_data = [[0 for _ in range(w)] for _ in range(h)]
    zone = rule.condition.get("zone") if rule.condition else None
    overlay = zone_overlay(grid) if zone else None
    for r in range(h):
        for c in range(w):
            if attention_mask and not attention_mask[r][c]:
                new_data[r][c] = grid.data[r][c]
                continue
            if zone and (overlay[r][c] is None or overlay[r][c].value != zone):
                new_data[r][c] = grid.data[r][c]
                continue
            nr = r + dy
            nc = c + dx
            if 0 <= nr < h and 0 <= nc < w:
                new_data[nr][nc] = grid.data[r][c]
            else:
                # cells translated outside remain 0
                pass
    return Grid(new_data)


def _apply_repeat(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]] = None
) -> Grid:
    """Tile ``grid`` according to ``rule`` parameters."""
    try:
        kx = int(rule.transformation.params.get("kx", "1"))
        ky = int(rule.transformation.params.get("ky", "1"))
    except ValueError:
        log_rule_failure(
            rule,
            failure_type="REPEAT",
            message="invalid repeat parameters",
        )
        return grid
    from arc_solver.src.symbolic.repeat_rule import repeat_tile

    tiled = repeat_tile(grid, kx, ky)

    replace_map = rule.meta.get("replace_map") if hasattr(rule, "meta") else None
    if replace_map:
        h, w = tiled.shape()
        new_data = [row[:] for row in tiled.data]
        for r in range(h):
            for c in range(w):
                val = new_data[r][c]
                if val in replace_map:
                    new_data[r][c] = replace_map[val]
        tiled = Grid(new_data)

    return tiled


def _apply_conditional(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]] = None
) -> Grid:
    """Apply a simple conditional replace rule."""
    src_color = None
    tgt_color = None
    neighbor_color = rule.transformation.params.get("neighbor")
    for sym in rule.source:
        if sym.type is SymbolType.COLOR:
            try:
                src_color = int(sym.value)
            except Exception:
                msg = f"Invalid symbol value: {sym.value}, skipping rule"
                logger.warning(msg)
                log_rule_failure(
                    rule,
                    failure_type="CONDITIONAL",
                    skipped_due_to=sym.value,
                    message=msg,
                )
                return grid
        elif sym.type is SymbolType.ZONE:
            # zone scoping is handled in _apply_region
            pass
    for sym in rule.target:
        if sym.type is SymbolType.COLOR:
            try:
                tgt_color = int(sym.value)
            except Exception:
                msg = f"Invalid symbol value: {sym.value}, skipping rule"
                logger.warning(msg)
                log_rule_failure(
                    rule,
                    failure_type="CONDITIONAL",
                    skipped_due_to=sym.value,
                    message=msg,
                )
                return grid
    if src_color is None or tgt_color is None:
        log_rule_failure(rule, failure_type="CONDITIONAL", message="missing parameters")
        return grid

    h, w = grid.shape()
    zone = rule.condition.get("zone") if rule.condition else None
    overlay = zone_overlay(grid) if zone else None
    new_data = [row[:] for row in grid.data]
    for r in range(h):
        for c in range(w):
            if attention_mask and not attention_mask[r][c]:
                continue
            if zone and (overlay[r][c] is None or overlay[r][c].value != zone):
                continue
            if new_data[r][c] != src_color:
                continue
            if neighbor_color is not None:
                neigh_match = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid.get(nr, nc) == int(neighbor_color):
                        neigh_match = True
                        break
                if not neigh_match:
                    continue
            new_data[r][c] = tgt_color
    return Grid(new_data)


def _apply_region(
    grid: Grid,
    rule: SymbolicRule,
    attention_mask: Optional[List[List[bool]]] = None,
    *,
    lineage_tracker: ColorLineageTracker | None = None,
) -> Grid:
    """Apply a rule only to cells within a labelled region overlay."""
    if grid.overlay is None:
        log_rule_failure(rule, failure_type="REGION", message="no overlay present")
        return grid
    region = None
    for sym in rule.source:
        if sym.type in (SymbolType.REGION, SymbolType.ZONE):
            region = sym.value
            break
    if region is None:
        log_rule_failure(rule, failure_type="REGION", message="region not specified")
        return grid

    inner_rule = SymbolicRule(
        transformation=Transformation(TransformationType.REPLACE),
        source=[s for s in rule.source if s.type is SymbolType.COLOR],
        target=rule.target,
        nature=rule.nature,
    )

    h, w = grid.shape()
    new_data = [row[:] for row in grid.data]
    for r in range(h):
        for c in range(w):
            if attention_mask and not attention_mask[r][c]:
                continue
            sym = grid.overlay[r][c]
            if sym is None or sym.value != region:
                continue
            cell_grid = Grid([row[:] for row in grid.data])
            cell_grid.set(r, c, grid.get(r, c))
            cell_grid = _apply_replace(
                cell_grid,
                inner_rule,
                lineage_tracker=lineage_tracker,
            )
            new_data[r][c] = cell_grid.get(r, c)
    return Grid(new_data)


def _apply_functional(
    grid: Grid, rule: SymbolicRule, attention_mask: Optional[List[List[bool]]] = None
) -> Grid:
    op = rule.transformation.params.get("op")
    if op == "invert_diagonal":
        h, w = grid.shape()
        new_data = [row[:] for row in grid.data]
        for r in range(h):
            for c in range(w):
                if attention_mask and not attention_mask[r][c]:
                    new_data[r][c] = grid.get(r, c)
                    continue
                if r == c or r == w - c - 1:
                    new_data[r][c] = grid.get(r, c)
                else:
                    new_data[r][c] = grid.get(r, c)
        return Grid(new_data)
    elif op == "flip_horizontal":
        return grid.flip_horizontal()
    log_rule_failure(rule, failure_type="FUNCTIONAL", message=f"unknown op {op}")
    return grid


def safe_apply_rule(
    rule: SymbolicRule,
    grid: Grid,
    attention_mask: Optional[List[List[bool]]] = None,
    perform_checks: bool = True,
    *,
    lineage_tracker: ColorLineageTracker | None = None,
    task_id: str | None = None,
    uncertainty_grid: list[list[int]] | None = None,
) -> Grid:
    """Apply ``rule`` safely, returning ``grid`` unchanged on failure."""
    logger.debug(f"Executing rule: {rule}")
    try:
        return _safe_apply_rule(
            grid,
            rule,
            attention_mask,
            perform_checks,
            lineage_tracker,
            uncertainty_grid,
        )
    except IndexError as exc:
        logger.warning(f"IndexError applying rule {rule}: {exc}")
        log_rule_failure(
            rule,
            failure_type="IndexError",
            message=str(exc),
            grid_snapshot=grid.data,
            task_id=task_id,
        )
        return grid
    except Exception as exc:  # pragma: no cover - catch-all
        logger.warning(f"Rule application failed: {rule} — {exc}")
        log_rule_failure(
            rule,
            failure_type="Exception",
            message=str(exc),
            grid_snapshot=grid.data,
            task_id=task_id,
        )
        return grid


def _safe_apply_rule(
    grid: Grid,
    rule: SymbolicRule,
    attention_mask: Optional[List[List[bool]]] = None,
    perform_checks: bool = True,
    lineage_tracker: ColorLineageTracker | None = None,
    uncertainty_grid: list[list[int]] | None = None,
) -> Grid:
    before = Grid([row[:] for row in grid.data])

    if isinstance(rule, CompositeRule):
        after = simulate_composite_safe(grid, rule, uncertainty_grid=uncertainty_grid)
    elif rule.transformation.ttype is TransformationType.REPLACE:
        try:
            after = _apply_replace(
                grid,
                rule,
                attention_mask,
                lineage_tracker=lineage_tracker,
            )
        except Exception as e:
            logger.warning(f"Rule application failed: {rule} — {e}")
            return grid
    elif rule.transformation.ttype is TransformationType.TRANSLATE:
        after = _apply_translate(grid, rule, attention_mask)
    elif rule.transformation.ttype is TransformationType.REPEAT:
        after = _apply_repeat(grid, rule, attention_mask)
    elif rule.transformation.ttype is TransformationType.COMPOSITE:
        after = _apply_repeat(grid, rule, attention_mask)
        after = _apply_replace(after, rule, attention_mask, lineage_tracker=lineage_tracker)
    elif rule.transformation.ttype is TransformationType.CONDITIONAL:
        after = _apply_conditional(grid, rule, attention_mask)
    elif rule.transformation.ttype is TransformationType.REGION:
        after = _apply_region(
            grid,
            rule,
            attention_mask,
            lineage_tracker=lineage_tracker,
        )
    elif rule.transformation.ttype is TransformationType.FUNCTIONAL:
        after = _apply_functional(grid, rule, attention_mask)
    else:
        after = grid

    if perform_checks:
        if violates_symmetry(after, before):
            raise ReflexOverrideException("Symmetry violation")
        if breaks_training_constraint(after):
            raise ReflexOverrideException("Training constraint mismatch")

    if before.compare_to(after) == 1.0:
        logger.debug(f"Rule had no effect: {rule}")
        return grid

    return after


def simulate_rules(
    input_grid: Grid,
    rules: List[SymbolicRule | CompositeRule],
    *,
    attention_mask: Optional[List[List[bool]]] = None,
    logger: logging.Logger | None = None,
    trace_log: list[dict] | None = None,
    uncertainty_grid: list[list[int]] | None = None,
    conflict_policy: str | None = None,
    strict: bool = False,
    lineage_tracker: ColorLineageTracker | None = None,
) -> Grid:
    """Apply a list of symbolic rules to ``input_grid`` with reflex checks."""
    # Determine execution order based on rule dependencies and spatial topology
    try:
        rules = sort_rules_by_topology(rules)
    except Exception:
        rules = sort_rules_by_dependency(rules)

    lineage_tracker = lineage_tracker or ColorLineageTracker(input_grid)

    if logger:
        total = len(rules)
        comp = sum(1 for r in rules if isinstance(r, CompositeRule))
        ratio = comp / total if total else 0.0
        logger.debug(
            "simulate_rules received %d rules, %d composite (%.2f)", total, comp, ratio
        )

    # Validate color dependencies before simulation
    rules = validate_color_dependencies(
        rules,
        input_grid,
        logger=logger,
        strict=strict,
        lineage_tracker=lineage_tracker,
    )

    grid = Grid([row[:] for row in input_grid.data])
    intermediate_grids: List[List[List[int]]] = [[row[:] for row in grid.data]]
    color_lineage: List[Set[int]] = [get_color_set(grid)]
    # Pre-compute rule coverage and sort rules by descending impact
    coverage_pairs: list[tuple[SymbolicRule | CompositeRule, int]] = []
    for r in rules:
        try:
            if isinstance(r, CompositeRule):
                cov = sum(rule_coverage(step, grid) for step in r.steps)
            else:
                cov = rule_coverage(r, grid)
        except Exception:
            cov = 0
        coverage_pairs.append((r, cov))
    coverage_pairs.sort(key=lambda x: x[1], reverse=True)
    if logger:
        order = [cov for _, cov in coverage_pairs]
        logger.debug(f"Rule coverage order: {order}")
    h0, w0 = grid.shape()
    if uncertainty_grid is None:
        uncertainty_grid = [[0 for _ in range(w0)] for _ in range(h0)]
    else:
        _resize_grid_like(uncertainty_grid, grid)
    write_log: dict[tuple[int, int], list[int]] = defaultdict(list)
    write_vals: dict[tuple[int, int], list[int]] = defaultdict(list)

    for idx, (rule, pre_cov) in enumerate(coverage_pairs):
        if pre_cov == 0 and allow_pruning(rule):
            if logger:
                logger.warning(f"Skipping rule due to invalid context: {rule}")
            continue
        if logger:
            colors = sorted({v for row in grid.data for v in row})
            logger.info(f"Applying rule {idx}: {rule}")
            logger.debug(f"Grid shape={grid.shape()}, colors={colors}")
        if isinstance(rule, CompositeRule):
            valid = all(validate_rule_application(step, grid) for step in rule.steps)
            cond = rule.get_condition() if hasattr(rule, "get_condition") else None
        else:
            valid = validate_rule_application(rule, grid)
            cond = rule.condition
        zone = cond.get("zone") if cond else None
        if not valid:
            if logger:
                logger.warning(f"Skipping rule due to invalid context: {rule}")
            continue
        try:
            tentative = check_symmetry_break(rule, grid, attention_mask)
        except ReflexOverrideException:
            if logger:
                logger.warning(f"Reflex override triggered by rule: {rule}")
            continue

        gh, gw = grid.shape()
        th, tw = tentative.shape()
        max_h = max(gh, th)
        max_w = max(gw, tw)
        changed: list[tuple[int, int, int, int]] = []
        for r in range(max_h):
            for c in range(max_w):
                before_val = grid.get(r, c)
                after_val = tentative.get(r, c)
                if before_val != after_val:
                    write_log[(r, c)].append(idx)
                    write_vals[(r, c)].append(after_val)
                    changed.append((r, c, before_val, after_val))

        if not changed:
            if logger:
                logger.info("Pruning ineffective rule")
            continue
        coverage_ratio = len(changed) / (th * tw)
        rule.meta["coverage_ratio"] = coverage_ratio
        if coverage_ratio < 0.01:
            rule.meta["demoted"] = True
        if zone:
            overlay = zone_overlay(grid)
            zone_cells = [
                (r, c)
                for r in range(gh)
                for c in range(gw)
                if overlay[r][c] is not None and overlay[r][c].value == zone
            ]
            coverage = len(changed) / len(zone_cells) if zone_cells else 0.0
            rule.meta["zone_coverage_ratio"] = coverage
            if logger and 0 < coverage < 1.0:
                logger.info(f"Rule partially applied: coverage={coverage:.2f}")
            if coverage < config_loader.ZONE_COVERAGE_THRESHOLD:
                zone_policy = config_loader.ZONE_PERSISTENCE_POLICY
                if zone_policy == "strict":
                    continue
                elif zone_policy == "sensitive":
                    in_zone_change = any((r, c) in zone_cells for r, c, _, _ in changed)
                    if not in_zone_change:
                        continue
                if zone_policy == "relaxed":
                    alt = SymbolicRule(
                        transformation=rule.transformation,
                        source=rule.source,
                        target=rule.target,
                        nature=rule.nature,
                        condition={k: v for k, v in rule.condition.items() if k != "zone"},
                        meta=rule.meta,
                    )
                    tentative_alt = check_symmetry_break(alt, grid, attention_mask)
                    changed_alt: list[tuple[int, int, int, int]] = []
                    for r in range(gh):
                        for c in range(gw):
                            b = grid.get(r, c)
                            a = tentative_alt.get(r, c)
                            if b != a:
                                changed_alt.append((r, c, b, a))
                    if len(changed_alt) > len(changed):
                        tentative = tentative_alt
                        changed = changed_alt
                        rule = alt
        if trace_log is not None:
            zone_val = cond.get("zone") if cond else None
            trace_log.append({"rule_id": idx, "zone": zone_val, "effect": changed})
        if logger and zone:
            zone_miss = len(zone_cells) - len(changed)
            if zone_miss > 0:
                logger.info(f"Zone mismatch count: {zone_miss}")
        if lineage_tracker:
            lineage_tracker.observe_rule(rule, grid, tentative)
        grid = tentative
        intermediate_grids.append([row[:] for row in grid.data])
        color_lineage.append(get_color_set(grid))
        _resize_grid_like(uncertainty_grid, grid)

    policy = conflict_policy or CONFLICT_POLICY
    conflict_count = 0
    for loc, writers in write_log.items():
        if len(writers) > 1:
            if logger:
                logger.warning(f"Collision at {loc}: rules {writers}")
            mark_conflict(loc, uncertainty_grid, grid)
            vals = write_vals[loc]
            if policy == "first":
                grid.set(loc[0], loc[1], vals[0])
            elif policy == "most_frequent":
                grid.set(loc[0], loc[1], Counter(vals).most_common(1)[0][0])
            conflict_count += 1

    if logger and conflict_count:
        logger.info(f"Conflicting writes: {conflict_count}")

    fh, fw = grid.shape()
    area_h = max(h0, fh)
    area_w = max(w0, fw)
    coverage_score = sum(
        1
        for r in range(area_h)
        for c in range(area_w)
        if input_grid.get(r, c) != grid.get(r, c)
    ) / (area_h * area_w)
    entropy_delta = _grid_entropy(grid) - _grid_entropy(input_grid)
    if logger:
        logger.info(
            "coverage=%.2f entropy_delta=%.3f conflicts=%d",
            coverage_score,
            entropy_delta,
            conflict_count,
        )

    expected = input_grid.shape() if grid.shape() == input_grid.shape() else None
    if not validate_grid(grid, expected_shape=expected):
        if logger:
            logger.warning("simulation produced invalid grid; returning copy")
        log_failure(
            task_id=None,
            rule_id=";".join(str(r) for r, _ in coverage_pairs),
            rule_type="composite" if any(isinstance(r, CompositeRule) for r, _ in coverage_pairs) else "atomic",
            rule_steps=[str(r) for r, _ in coverage_pairs],
            rejection_stage="simulation",
            failed_step_index=len(intermediate_grids) - 1,
            reason="invalid_grid",
            color_lineage=color_lineage,
            intermediate_grids=intermediate_grids,
        )
        grid = Grid([row[:] for row in input_grid.data])

    if rule_failures_log:
        print("SKIP REPORT SUMMARY")
        summary = summarize_skips_by_type()
        for msg, count in list(summary.items())[:3]:
            print(f"{msg}: {count}")
        tasks = {
            entry.get("task_id")
            for entry in rule_failures_log
            if entry.get("task_id") is not None
        }
        if tasks:
            print(f"Affected tasks: {sorted(tasks)}")

    return grid


def score_prediction(predicted: Grid, target: Grid) -> float:
    """Return match ratio between ``predicted`` and ``target``."""
    return predicted.compare_to(target)


def simulate_symbolic_program(grid: Grid, rules: List[SymbolicRule]) -> Grid:
    """Alias of :func:`simulate_rules` for program semantics."""
    return simulate_rules(grid, rules)


def simulate_rules_with_softmask(input_grid, rules, config=None):
    from arc_solver.src.core.grid_utils import compute_conflict_map

    config = config or {}
    max_conflict_radius = config.get("MAX_CONFLICT_RADIUS", 2)

    working_grid = Grid([row[:] for row in input_grid.data])
    h, w = working_grid.shape()
    propagation: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]
    applied_rules = []

    for rule in rules:
        try:
            pred_grid = rule.apply(working_grid)
        except Exception:
            continue

        conflict_map = compute_conflict_map(working_grid, pred_grid)

        if rule.triggers_large_conflict(conflict_map, radius=max_conflict_radius):
            continue

        noisy = False
        for r in range(len(conflict_map)):
            for c in range(len(conflict_map[0])):
                if conflict_map[r][c]:
                    propagation[r][c] += 1
                    if propagation[r][c] > max_conflict_radius:
                        noisy = True
        if noisy:
            continue

        working_grid = pred_grid
        applied_rules.append(rule)

    return working_grid


__all__ = [
    "simulate_rules",
    "simulate_rules_with_softmask",
    "simulate_symbolic_program",
    "score_prediction",
    "ReflexOverrideException",
    "ValidationError",
    "validate_rule_application",
    "check_symmetry_break",
    "visualize_uncertainty",
    "validate_color_dependencies",
    "grid_growth_forecast",
    "simulate_composite_rule",
    "simulate_composite_safe",
]
