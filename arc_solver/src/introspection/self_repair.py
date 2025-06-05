from __future__ import annotations

"""Discrepancy analysis and self-repair utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.symbolic.vocabulary import (
    Symbol,
    SymbolType,
    SymbolicRule,
    Transformation,
    TransformationType,
)
from .trace_builder import build_trace, RuleTrace
import logging
from arc_solver.src.utils.config_loader import OFFLINE_MODE, META_CONFIG
from . import llm_engine

logger = logging.getLogger(__name__)

try:
    from arc_solver.src.introspection.llm_engine import repair_symbolic_rule
except Exception:
    repair_symbolic_rule = None

try:  # pragma: no cover - optional dependency
    import openai
except Exception:  # pragma: no cover - if openai is unavailable
    openai = None


# ---------------------------------------------------------------------------
# Discrepancy detection
# ---------------------------------------------------------------------------

def compute_discrepancy(pred: Grid, target: Grid) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Return mapping of mismatched coordinates to (predicted, true) values."""
    if pred.shape() != target.shape():
        return {}
    h, w = pred.shape()
    diff: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            pv = pred.get(r, c)
            tv = target.get(r, c)
            if pv != tv:
                diff[(r, c)] = (pv, tv)
    return diff


# ---------------------------------------------------------------------------
# Tracing utilities
# ---------------------------------------------------------------------------

@dataclass
class RuleTraceEntry:
    rule: SymbolicRule
    before: Grid
    after: Grid
    affected: List[Tuple[int, int]]
    order: int
    symbolic_label: str = ""


def trace_prediction(rule_set: List[SymbolicRule], input_grid: Grid) -> List[RuleTraceEntry]:
    """Apply ``rule_set`` sequentially and record rule activations."""
    grid = Grid([row[:] for row in input_grid.data])
    traces: List[RuleTraceEntry] = []
    for i, rule in enumerate(rule_set):
        before = grid
        after = simulate_rules(before, [rule])
        rt = build_trace(rule, before, after, None)
        traces.append(
            RuleTraceEntry(
                rule=rule,
                before=before,
                after=after,
                affected=rt.affected_cells,
                order=i,
                symbolic_label=str(rule),
            )
        )
        grid = after
    return traces


# ---------------------------------------------------------------------------
# Mismatch attribution
# ---------------------------------------------------------------------------

@dataclass
class FaultHypothesis:
    rule: SymbolicRule
    cells: List[Tuple[int, int]]
    score: float


def localize_faulty_rule(trace: List[RuleTraceEntry], discrepancy_map: Dict[Tuple[int, int], Tuple[int, int]]) -> List[FaultHypothesis]:
    """Return candidate faulty rules ranked by overlap with discrepancies."""
    hypotheses: List[FaultHypothesis] = []
    total = len(discrepancy_map)
    for entry in trace:
        mismatched = [cell for cell in discrepancy_map if cell in entry.affected]
        if not mismatched:
            continue
        score = len(mismatched) / total if total else 0.0
        hypotheses.append(FaultHypothesis(entry.rule, mismatched, score))
    return sorted(hypotheses, key=lambda h: h.score, reverse=True)


# ---------------------------------------------------------------------------
# Repair heuristics
# ---------------------------------------------------------------------------

def refine_rule(rule: SymbolicRule, context: Dict[str, Tuple[int, int]]) -> Optional[SymbolicRule]:
    """Heuristically adjust ``rule`` parameters using ``context``."""
    if rule.transformation.ttype is TransformationType.REPLACE:
        mismatch = next(iter(context.values()), None)
        if mismatch is not None:
            src_color = None
            for s in rule.source:
                if s.type is SymbolType.COLOR:
                    src_color = s
                    break
            if src_color is not None:
                tgt_color = Symbol(SymbolType.COLOR, str(mismatch[1]))
                return SymbolicRule(
                    transformation=Transformation(TransformationType.REPLACE),
                    source=[src_color],
                    target=[tgt_color],
                    nature=rule.nature,
                    condition=rule.condition.copy(),
                )
    elif rule.transformation.ttype is TransformationType.TRANSLATE:
        params = rule.transformation.params
        dx = int(params.get("dx", "0"))
        dy = int(params.get("dy", "0"))
        if dx != 0:
            dx += -1 if dx > 0 else 1
        if dy != 0:
            dy += -1 if dy > 0 else 1
        new_params = {**params, "dx": str(dx), "dy": str(dy)}
        return SymbolicRule(
            transformation=Transformation(TransformationType.TRANSLATE, new_params),
            source=rule.source,
            target=rule.target,
            nature=rule.nature,
            condition=rule.condition.copy(),
        )
    return None


# ---------------------------------------------------------------------------
# LLM-assisted repair
# ---------------------------------------------------------------------------

def llm_suggest_rule_fix(entry: RuleTraceEntry, discrepancies: Dict[Tuple[int, int], Tuple[int, int]]) -> Optional[SymbolicRule]:
    """Use a local LLM or heuristics to propose a replacement rule."""

    if OFFLINE_MODE:
        try:
            return llm_engine.local_suggest_rule_fix(entry, discrepancies)
        except Exception:
            pass

    if openai is not None:
        before_text = entry.before.to_list()
        after_text = entry.after.to_list()
        disc_text = {
            str(k): {"pred": v[0], "true": v[1]} for k, v in discrepancies.items()
        }
        prompt = (
            "You are analysing a symbolic rule for an ARC solver.\n"
            f"Rule: {entry.rule}\n"
            f"Before: {before_text}\n"
            f"After: {after_text}\n"
            f"Discrepancies: {disc_text}\n"
            "Suggest a corrected rule in DSL format."
        )
        try:  # pragma: no cover - external call
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response["choices"][0]["message"]["content"].strip()
            from arc_solver.src.symbolic.rule_language import parse_rule

            return parse_rule(text)
        except Exception:
            pass

    # Fallback to heuristic refinement
    return refine_rule(entry.rule, discrepancies)


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_repair_candidates(
    candidates: List[SymbolicRule], input_grid: Grid, target: Grid
) -> SymbolicRule:
    """Return the candidate yielding the highest similarity."""
    best = candidates[0]
    best_score = -1.0
    for rule in candidates:
        pred = simulate_rules(input_grid, [rule])
        score = pred.compare_to(target)
        if score > best_score:
            best_score = score
            best = rule
    return best


def run_meta_repair(
    grid_in: Grid,
    predicted: Grid,
    ground_truth: Grid,
    rules: List[SymbolicRule],
) -> Tuple[Grid, List[SymbolicRule]]:
    """Return improved prediction and rule set via discrepancy mining."""

    original_score = predicted.compare_to(ground_truth)
    discrepancy = compute_discrepancy(predicted, ground_truth)
    trace = trace_prediction(rules, grid_in)
    hypotheses = localize_faulty_rule(trace, discrepancy)

    for hyp in hypotheses:
        entry = next((e for e in trace if e.rule is hyp.rule), None)
        if entry is None:
            continue
        fix = refine_rule(hyp.rule, discrepancy)
        if fix is None:
            fix = llm_suggest_rule_fix(entry, discrepancy)
        if fix is None and META_CONFIG.get("llm_mode") == "local" and repair_symbolic_rule:
            try:
                dsl = repair_symbolic_rule(str(entry.rule), str(discrepancy))
                if dsl:
                    from arc_solver.src.symbolic.rule_language import parse_rule

                    fix = parse_rule(dsl)
                    logger.info(f"Repaired rule via local LLM: {dsl}")
            except Exception as e:  # pragma: no cover - handle parse errors
                logger.warning(f"Failed to parse LLM-repaired rule: {e}")
        if fix is None:
            continue
        new_rules = [fix if r is hyp.rule else r for r in rules]
        new_pred = simulate_rules(grid_in, new_rules)
        new_score = new_pred.compare_to(ground_truth)
        if new_score > original_score:
            return new_pred, new_rules

    return predicted, rules


__all__ = [
    "compute_discrepancy",
    "RuleTraceEntry",
    "trace_prediction",
    "FaultHypothesis",
    "localize_faulty_rule",
    "refine_rule",
    "llm_suggest_rule_fix",
    "evaluate_repair_candidates",
    "run_meta_repair",
]
