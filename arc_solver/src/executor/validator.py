from __future__ import annotations

"""Validation helpers for composite rule colour dependencies."""

from typing import List, Set

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.vocabulary import SymbolicRule, SymbolType
from arc_solver.src.symbolic.rule_language import CompositeRule
from .failure_logger import log_failure


def simulate_step(rule_step: SymbolicRule | CompositeRule, grid: Grid) -> Grid:
    """Return ``grid`` after applying ``rule_step`` without checks."""
    from .simulator import safe_apply_rule  # local import to avoid circularity

    if isinstance(rule_step, CompositeRule):
        out = Grid([row[:] for row in grid.data])
        for st in rule_step.steps:
            out = simulate_step(st, out)
        return out
    return safe_apply_rule(rule_step, grid, perform_checks=False)


def get_color_set(grid: Grid) -> Set[int]:
    """Return set of colors present in ``grid`` excluding background ``0``."""
    return {v for row in grid.data for v in row if v != 0}


def validate_color_dependencies(
    rule_chain: List[SymbolicRule | CompositeRule],
    input_grid: Grid,
    *,
    debug: bool = False,
    rule_id: str | None = None,
    task_id: str | None = None,
) -> bool:
    """Validate that ``rule_chain`` preserves all required colours.

    The chain is simulated step by step and the colour set after each step is
    recorded. Only the final colour set is compared against the colours required
    by the chain's source symbols. Intermediate colour removals are allowed.
    """
    working = Grid([row[:] for row in input_grid.data])
    color_lineage: List[Set[int]] = []
    intermediate_grids: List[List[List[int]]] = []
    required: Set[int] = set()

    for step in rule_chain:
        color_lineage.append(get_color_set(working))
        intermediate_grids.append([row[:] for row in working.data])
        if isinstance(step, CompositeRule):
            sub_steps = step.steps
        else:
            sub_steps = [step]
        for st in sub_steps:
            for sym in st.source:
                if sym.type is SymbolType.COLOR:
                    try:
                        val = int(sym.value)
                        if val != 0:
                            required.add(val)
                    except ValueError:
                        pass
            working = simulate_step(st, working)
    color_lineage.append(get_color_set(working))
    intermediate_grids.append([row[:] for row in working.data])

    final_colors = color_lineage[-1]
    missing = {c for c in required if c not in final_colors}
    if missing:
        divergence = None
        for i in range(len(color_lineage) - 1):
            before = color_lineage[i]
            after = color_lineage[i + 1]
            if any(c in before and c not in after for c in missing):
                divergence = i
                break
        log_failure(
            task_id=task_id,
            rule_id=rule_id or "chain",
            rule_type="composite",
            rule_steps=[str(s) for s in rule_chain],
            rejection_stage="validation",
            failed_step_index=divergence,
            reason="missing_final_colors",
            color_lineage=color_lineage,
            intermediate_grids=intermediate_grids,
        )
        return False

    if debug:
        log_failure(
            task_id=task_id,
            rule_id=rule_id or "chain",
            rule_type="composite",
            rule_steps=[str(s) for s in rule_chain],
            rejection_stage="validation",
            failed_step_index=None,
            reason="debug_lineage",
            color_lineage=color_lineage,
            intermediate_grids=intermediate_grids,
        )
    return True


def _step_explicitly_adds(step: SymbolicRule | CompositeRule, color: int) -> bool:
    """Return True if ``step`` introduces ``color`` via its target."""
    if isinstance(step, CompositeRule):
        return any(_step_explicitly_adds(st, color) for st in step.steps)
    try:
        for sym in step.target:
            if sym.type is SymbolType.COLOR and int(sym.value) == color:
                return True
    except Exception:
        pass
    return False


def validate_color_lineage(
    rule_chain: List[SymbolicRule | CompositeRule] | SymbolicRule | CompositeRule,
    input_grid: Grid,
    *,
    rule_id: str | None = None,
    task_id: str | None = None,
) -> bool:
    """Validate colour removals and reintroductions across ``rule_chain``."""

    if isinstance(rule_chain, (SymbolicRule, CompositeRule)):
        steps: List[SymbolicRule | CompositeRule] = (
            rule_chain.steps if isinstance(rule_chain, CompositeRule) else [rule_chain]
        )
    else:
        steps = list(rule_chain)

    working = Grid([row[:] for row in input_grid.data])
    color_lineage: List[Set[int]] = [get_color_set(working)]
    intermediate_grids: List[List[List[int]]] = [[row[:] for row in working.data]]

    removed_at: dict[int, int] = {}
    restored_at: dict[int, int] = {}

    step_index = 0
    for step in steps:
        before_set = get_color_set(working)
        before_grid = [row[:] for row in working.data]
        working = simulate_step(step, working)
        after_set = get_color_set(working)
        after_grid = [row[:] for row in working.data]

        removed = before_set - after_set
        added = after_set - before_set

        for col in removed:
            removed_at.setdefault(col, step_index)
            log_failure(
                task_id=task_id,
                rule_id=rule_id or "chain",
                rule_type="composite",
                rule_steps=[str(s) for s in steps],
                rejection_stage="validation",
                failed_step_index=step_index,
                reason="color_lineage_event",
                color_lineage=[before_set, after_set],
                intermediate_grids=[before_grid, after_grid],
            )

        for col in added:
            if col in removed_at and _step_explicitly_adds(step, col):
                restored_at[col] = step_index
                removed_at.pop(col, None)
            if col in removed_at or col in restored_at:
                log_failure(
                    task_id=task_id,
                    rule_id=rule_id or "chain",
                    rule_type="composite",
                    rule_steps=[str(s) for s in steps],
                    rejection_stage="validation",
                    failed_step_index=step_index,
                    reason="color_lineage_event",
                    color_lineage=[before_set, after_set],
                    intermediate_grids=[before_grid, after_grid],
                )

        color_lineage.append(after_set)
        intermediate_grids.append(after_grid)
        step_index += 1

    final_colors = color_lineage[-1]
    unresolved = {c: i for c, i in removed_at.items() if c in final_colors}
    if unresolved:
        first = min(unresolved.values())
        log_failure(
            task_id=task_id,
            rule_id=rule_id or "chain",
            rule_type="composite",
            rule_steps=[str(s) for s in steps],
            rejection_stage="validation",
            failed_step_index=first,
            reason="unrestored_color",
            color_lineage=color_lineage,
            intermediate_grids=intermediate_grids,
        )
        return False

    return True


__all__ = [
    "validate_color_dependencies",
    "simulate_step",
    "get_color_set",
    "validate_color_lineage",
]
