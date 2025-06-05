from __future__ import annotations

"""High level ARC task solving pipeline."""

from typing import List, Tuple

from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.abstractions.rule_generator import generalize_rules
from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.simulator import simulate_symbolic_program
from arc_solver.src.executor.attention import AttentionMask, zone_to_mask
from arc_solver.src.executor.dependency import select_independent_rules
from arc_solver.src.segment.segmenter import zone_overlay
from arc_solver.src.rank_rule_sets import probabilistic_rank_rule_sets
from arc_solver.src.memory.memory_store import (
    load_memory,
    retrieve_similar_signatures,
    save_rule_program,
)
from arc_solver.src.utils.signature_extractor import extract_task_signature
from arc_solver.src.fallback import prioritize, soft_vote
from arc_solver.src.executor.prior_templates import load_prior_templates
from arc_solver.src.executor.fallback_predictor import predict as fallback_predict
from arc_solver.src.introspection import (
    build_trace,
    inject_feedback,
    llm_refine_program,
    evaluate_refinements,
)
from arc_solver.src.symbolic import rules_to_program


def solve_task(
    task: dict,
    *,
    introspect: bool = False,
    use_memory: bool = False,
    use_prior: bool = False,
    task_id: str | None = None,
    debug: bool = False,
    log_dir: str = "logs",
):
    """Solve a single ARC task represented by a JSON dictionary."""
    train_pairs = [
        (Grid(p["input"]), Grid(p["output"])) for p in task.get("train", [])
    ]
    test_inputs = [Grid(p["input"]) for p in task.get("test", [])]
    test_outputs = [Grid(p["output"]) for p in task.get("test", []) if "output" in p]

    from arc_solver.src.utils.logger import get_logger

    log_file = None
    logger = None
    if debug:
        ident = task_id or "task"
        log_file = f"{log_dir}/{ident}.log"
        logger = get_logger(f"solver.{ident}", file_path=log_file)

    rule_sets: List[List] = []
    prior_templates = load_prior_templates()
    simple_fallback = prior_templates[0] if prior_templates else []
    for inp, out in train_pairs:
        try:
            rules = abstract([inp, out], logger=logger)
            rules = generalize_rules(rules)
            rules = select_independent_rules(rules)
        except Exception:
            if logger:
                logger.warning("abstraction exception; using simple fallback")
            rules = simple_fallback
        rule_sets.append(rules)

    ranked_rules = probabilistic_rank_rule_sets(rule_sets, train_pairs)
    best_rules: List = ranked_rules[0][0] if ranked_rules else []

    # Recall programs from memory or priors ---------------------------------
    signature = extract_task_signature(task)
    candidate_sets = [select_independent_rules(rs) for rs, _ in ranked_rules]
    if use_memory:
        recalled = retrieve_similar_signatures(signature)
        for entry in recalled:
            candidate_sets.append(select_independent_rules(entry["rules"]))
    if use_prior:
        candidate_sets.extend(select_independent_rules(rs) for rs in load_prior_templates())

    # Score all candidates on training examples
    def _train_score(rules: List, mask: List[List[bool]] | None = None) -> float:
        if not train_pairs:
            return 0.0
        total = 0.0
        for inp, out in train_pairs:
            try:
                pred = simulate_rules(inp, rules, attention_mask=mask, logger=logger)
            except Exception:
                if logger:
                    logger.warning("training simulation failed; using fallback")
                pred = fallback_predict(inp)
            total += pred.compare_to(out)
        return total / len(train_pairs)

    zones: List[str] = []
    if train_pairs:
        overlay = zone_overlay(train_pairs[0][0])
        for row in overlay:
            for sym in row:
                if sym is not None:
                    zones.append(sym.value)
    zones = sorted(set(zones))
    attn_masks = [zone_to_mask(train_pairs[0][0], z) for z in zones]

    scores = []
    for rs in candidate_sets:
        base = _train_score(rs)
        for m in attn_masks:
            base = max(base, _train_score(rs, m))
        scores.append(base)
        if logger:
            logger.info(f"candidate {rules_to_program(rs)} score={base:.3f}")
    prioritized = prioritize(candidate_sets, scores)
    if prioritized:
        best_rules = prioritized[0]
    if logger:
        for rs, sc in zip(candidate_sets, scores):
            logger.info(f"scored {rules_to_program(rs)} -> {sc:.3f}")
        if prioritized:
            logger.info(f"selected {rules_to_program(best_rules)}")

    # Optional introspection/refinement using first training example
    traces = []
    if introspect and best_rules and train_pairs:
        try:
            inp0, out0 = train_pairs[0]
            pred0 = simulate_rules(inp0, best_rules, logger=logger)
            trace = build_trace(best_rules[0], inp0, pred0, out0)
            feedback = inject_feedback(trace)
            candidates = llm_refine_program(trace, feedback)
            refined = evaluate_refinements(candidates, inp0, out0)
            best_rules = [refined]
            traces.append(trace)
            if logger:
                logger.info("LLM refinement applied")
                logger.info(rules_to_program(best_rules))
        except Exception:
            pass

    predictions = []
    top_sets = prioritized[:3] if prioritized else []
    for g in test_inputs:
        cand_preds = []
        for rs in top_sets:
            try:
                cand_preds.append(simulate_rules(g, rs, logger=logger))
                for m in attn_masks:
                    cand_preds.append(simulate_rules(g, rs, attention_mask=m, logger=logger))
            except Exception:
                if logger:
                    logger.warning("simulation error, skipping rule set")
                continue
        if not cand_preds:
            try:
                cand_preds.append(simulate_rules(g, simple_fallback, logger=logger))
                if logger:
                    logger.info("used prior template fallback")
            except Exception:
                if train_pairs and g.shape() == train_pairs[-1][1].shape():
                    cand_preds.append(train_pairs[-1][1])
                    if logger:
                        logger.info("used copy-train heuristic")
                else:
                    cand_preds.append(fallback_predict(g))
                    if logger:
                        logger.info("used dummy fallback predictor")
        final = soft_vote(cand_preds)
        predictions.append(final)

    # Persist best performing program
    if use_memory and train_pairs and best_rules:
        score = _train_score(best_rules)
        if task_id is not None:
            save_rule_program(task_id, signature, best_rules, score)
    if logger:
        logger.info("final rules: %s", rules_to_program(best_rules))

    return predictions, test_outputs, traces, best_rules


def solve_task_iterative(task: dict, *, steps: int = 3, introspect: bool = False):
    """Solve task using multi-step symbolic simulation."""
    preds, outs, traces, rules = solve_task(task, introspect=introspect)
    if not preds:
        return preds, outs, traces, rules

    refined_preds = []
    for pred in preds:
        grid = pred
        for _ in range(1, steps):
            try:
                grid = simulate_symbolic_program(grid, rules)
            except Exception:
                grid = fallback_predict(grid)
        refined_preds.append(grid)
    return refined_preds, outs, traces, rules

__all__ = ["solve_task", "solve_task_iterative"]
