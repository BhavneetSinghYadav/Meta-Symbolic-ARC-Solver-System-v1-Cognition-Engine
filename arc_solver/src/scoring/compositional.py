from __future__ import annotations

"""Rule scoring and compositional ranking utilities."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.introspection import build_trace, RuleTrace
from arc_solver.src.symbolic.vocabulary import SymbolicRule


@dataclass
class RuleInfo:
    rule: SymbolicRule
    support: set[int]
    applications: List[Tuple[Grid, Grid]]


def extract_all_rules(train_pairs: Iterable[Tuple[Grid, Grid]]) -> Dict[str, RuleInfo]:
    """Return all rules extracted from the training pairs."""
    all_rules: Dict[str, RuleInfo] = {}
    for i, (inp, out) in enumerate(train_pairs):
        rules = abstract([inp, out])
        for rule in rules:
            rid = repr(rule)
            if rid not in all_rules:
                all_rules[rid] = RuleInfo(rule=rule, support={i}, applications=[(inp, out)])
            else:
                info = all_rules[rid]
                info.support.add(i)
                info.applications.append((inp, out))
    return all_rules


def simulate_and_trace(rule: SymbolicRule, inp: Grid, out: Grid) -> RuleTrace:
    """Simulate ``rule`` on ``inp`` and build a trace against ``out``."""
    pred = simulate_rules(inp, [rule])
    trace = build_trace(rule, inp, pred, out)
    return trace


def score_rule(rule: SymbolicRule, applications: Iterable[Tuple[Grid, Grid]]) -> float:
    """Return the average match score for ``rule`` across ``applications``."""
    scores: List[float] = []
    for inp, out in applications:
        trace = simulate_and_trace(rule, inp, out)
        scores.append(trace.match_score)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_on_all_pairs(rules: List[SymbolicRule], pairs: Iterable[Tuple[Grid, Grid]]) -> float:
    """Return average similarity of ``rules`` across all grid pairs."""
    total = 0.0
    count = 0
    for inp, out in pairs:
        pred = simulate_rules(inp, rules)
        total += pred.compare_to(out)
        count += 1
    return total / count if count else 0.0


def compose_programs(rule_dict: Dict[str, RuleInfo], pairs: Iterable[Tuple[Grid, Grid]]) -> List[SymbolicRule]:
    """Search over simple rule combinations and return the best performing list."""
    rule_list = [info.rule for info in rule_dict.values()]
    best_combo: List[SymbolicRule] = []
    best_score = float("-inf")

    combos: List[List[SymbolicRule]] = [[r] for r in rule_list]
    for i, r1 in enumerate(rule_list):
        for j, r2 in enumerate(rule_list):
            if i >= j:
                continue
            combos.append([r1, r2])

    for combo in combos:
        score = evaluate_on_all_pairs(combo, pairs)
        if score > best_score:
            best_score = score
            best_combo = combo
    return best_combo


def justify_selection(rules: List[SymbolicRule], rule_dict: Dict[str, RuleInfo]) -> Dict[str, dict]:
    """Return justification metadata for the selected ``rules``."""
    info: Dict[str, dict] = {}
    for rule in rules:
        rid = repr(rule)
        rinfo = rule_dict.get(rid)
        trace = None
        if rinfo and rinfo.applications:
            trace = simulate_and_trace(rule, *rinfo.applications[0])
        info[rid] = {
            "type": rule.transformation.ttype.value,
            "description": str(rule),
            "support_count": len(rinfo.support) if rinfo else 0,
            "score": score_rule(rule, rinfo.applications) if rinfo else 0.0,
            "sample_trace": trace,
        }
    return info


def run_pipeline(train_pairs: Iterable[Tuple[Grid, Grid]]) -> Tuple[List[SymbolicRule], Dict[str, dict]]:
    """Full scoring and ranking pipeline for a list of training pairs."""
    rule_dict = extract_all_rules(train_pairs)
    best_rules = compose_programs(rule_dict, train_pairs)
    justification = justify_selection(best_rules, rule_dict)
    return best_rules, justification


__all__ = [
    "RuleInfo",
    "extract_all_rules",
    "simulate_and_trace",
    "score_rule",
    "evaluate_on_all_pairs",
    "compose_programs",
    "justify_selection",
    "run_pipeline",
]
