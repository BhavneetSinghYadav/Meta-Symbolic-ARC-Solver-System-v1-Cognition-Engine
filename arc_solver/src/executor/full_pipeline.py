from __future__ import annotations

"""High level ARC task solving pipeline."""

from typing import List, Tuple
import json
from pathlib import Path

from arc_solver.src.abstractions.abstractor import abstract
from arc_solver.src.abstractions.rule_generator import (
    generalize_rules,
    remove_duplicate_rules,
)
from arc_solver.src.core.grid import Grid
from arc_solver.src.executor.simulator import simulate_rules
from arc_solver.src.executor.simulator import simulate_symbolic_program
from arc_solver.src.executor.attention import AttentionMask, zone_to_mask
from arc_solver.src.executor.dependency import select_independent_rules
from arc_solver.src.segment.segmenter import zone_overlay
from arc_solver.src.rank_rule_sets import probabilistic_rank_rule_sets
from arc_solver.src.attention.fusion_injector import apply_structural_attention
from arc_solver.src.memory.memory_store import (
    load_memory,
    match_signature,
    save_rule_program,
    extract_task_constraints,
)
from arc_solver.src.utils.signature_extractor import extract_task_signature
from arc_solver.src.fallback import prioritize, soft_vote
from arc_solver.src.executor.prior_templates import load_prior_templates
from arc_solver.src.memory.deep_prior_loader import (
    load_prior_templates as deep_load_prior_templates,
    load_motifs,
    match_task_signature_to_prior,
    select_motifs,
)
from arc_solver.src.meta_generalizer import generalize_rule_program
from arc_solver.src.symbolic.rule_language import parse_rule
from arc_solver.src.executor.fallback_predictor import predict as fallback_predict
from arc_solver.src.introspection import (
    build_trace,
    inject_feedback,
    llm_refine_program,
    evaluate_refinements,
    run_meta_repair,
    validate_trace,
    trace_prediction,
)
from arc_solver.src.symbolic import rules_to_program
from arc_solver.src.utils import config_loader

_FAILURE_LOG = Path("logs/failure_log.json")


def solve_task(
    task: dict,
    *,
    introspect: bool = False,
    use_memory: bool = False,
    use_prior: bool = False,
    use_deep_priors: bool = False,
    prior_threshold: float = 0.4,
    motif_file: str | None = None,
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

    # Regime detection -----------------------------------------------------
    from arc_solver.src.regime.regime_classifier import (
        compute_task_signature,
        predict_regime_category,
        score_abstraction_likelihood,
        log_regime,
        RegimeType,
    )
    from arc_solver.src.regime.policy_router import decide_policy
    from arc_solver.src.regime.decision_controller import DecisionReflexController

    signature_stats = compute_task_signature(train_pairs)
    regime = predict_regime_category(signature_stats)
    confidence = score_abstraction_likelihood(signature_stats)
    if logger:
        logger.info(
            f"regime {regime.name} score={confidence:.2f} stats={signature_stats}"
        )
    log_regime(task_id or "unknown", signature_stats, regime, confidence)

    controller = DecisionReflexController(task_id or "unknown", regime, confidence)
    policy = controller.decide()

    if policy == "fallback":
        if logger:
            logger.info("policy=fallback")
        if use_deep_priors and config_loader.PRIOR_INJECTION_ENABLED:
            signature = extract_task_signature(task)
            prior_sets = match_task_signature_to_prior(signature, prior_threshold)
            if prior_sets:
                preds = []
                for g in test_inputs:
                    try:
                        preds.append(simulate_rules(g, prior_sets[0], logger=logger))
                    except Exception:
                        preds.append(fallback_predict(g))
                return preds, test_outputs, [], prior_sets[0]
        predictions = [fallback_predict(g) for g in test_inputs]
        return predictions, test_outputs, [], []

    if policy == "memory_then_fallback":
        if logger:
            logger.info("policy=memory_then_fallback")
        signature = extract_task_signature(task)
        recalled = match_signature(
            signature, constraints=extract_task_constraints(train_pairs[0][0]) if train_pairs else None
        )
        if recalled:
            try_rules = recalled[0]["rules"]
            preds = []
            for g in test_inputs:
                try:
                    preds.append(simulate_rules(g, try_rules, logger=logger))
                except Exception:
                    preds.append(fallback_predict(g))
            return preds, test_outputs, [], try_rules
        predictions = [fallback_predict(g) for g in test_inputs]
        return predictions, test_outputs, [], []

    if policy == "fallback_then_prior":
        if logger:
            logger.info("policy=fallback_then_prior")
        if use_deep_priors and config_loader.PRIOR_INJECTION_ENABLED:
            signature = extract_task_signature(task)
            prior_sets = match_task_signature_to_prior(signature, prior_threshold)
            if prior_sets:
                preds = []
                for g in test_inputs:
                    try:
                        preds.append(simulate_rules(g, prior_sets[0], logger=logger))
                    except Exception:
                        preds.append(fallback_predict(g))
                return preds, test_outputs, [], prior_sets[0]
        predictions = [fallback_predict(g) for g in test_inputs]
        return predictions, test_outputs, [], []

    rule_sets: List[List] = []
    prior_templates = load_prior_templates()
    simple_fallback = prior_templates[0] if prior_templates else []
    for inp, out in train_pairs:
        try:
            rules = abstract([inp, out], logger=logger)
            rules = generalize_rules(rules)
            rules = remove_duplicate_rules(rules)
            rules = select_independent_rules(rules)
        except Exception:
            if logger:
                logger.warning("abstraction exception; using simple fallback")
            rules = simple_fallback
        rule_sets.append(rules)

    if not rule_sets:
        if logger:
            logger.warning("no rules extracted; using fallback")
        predictions = [fallback_predict(g) for g in test_inputs]
        return predictions, test_outputs, [], []

    # Inject motifs before ranking -------------------------------------------
    signature = extract_task_signature(task)
    if use_deep_priors and config_loader.PRIOR_INJECTION_ENABLED:
        motifs = []
        if config_loader.PRIOR_USE_MOTIFS:
            motifs = select_motifs(signature, load_motifs(motif_file) if motif_file else None)
        prior_sets = match_task_signature_to_prior(signature, prior_threshold)
        for m in motifs[: config_loader.PRIOR_MAX_INJECT]:
            try:
                prior_sets.append([parse_rule(m["rule_dsl"])])
            except Exception:
                continue
        for p in prior_sets[: config_loader.PRIOR_MAX_INJECT]:
            rule_sets.append(select_independent_rules(p))

    ranked_rules = probabilistic_rank_rule_sets(rule_sets, train_pairs)
    if config_loader.USE_STRUCTURAL_ATTENTION and train_pairs:
        ranked_rules = apply_structural_attention(
            train_pairs[0][0],
            ranked_rules,
            config_loader.STRUCTURAL_ATTENTION_WEIGHT,
        )
    best_rules: List = ranked_rules[0][0] if ranked_rules else []
    if not best_rules:
        if logger:
            logger.warning("no candidate rules; using fallback predictor")
        predictions = [fallback_predict(g) for g in test_inputs]
        return predictions, test_outputs, [], []

    # Recall programs from memory or priors ---------------------------------
    candidate_sets = [select_independent_rules(rs) for rs, _ in ranked_rules]
    if use_memory:
        constraints = (
            extract_task_constraints(train_pairs[0][0]) if train_pairs else None
        )
        recalled = match_signature(signature, constraints=constraints)
        for entry in recalled:
            generalized = generalize_rule_program(entry["rules"], signature)
            candidate_sets.append(select_independent_rules(generalized))
    if use_prior:
        candidate_sets.extend(
            select_independent_rules(generalize_rule_program(rs, signature)) for rs in load_prior_templates()
        )

    # Score all candidates on training examples
    def _train_score(rules: List, mask: List[List[bool]] | None = None) -> float:
        if not train_pairs:
            return 0.0
        total = 0.0
        for inp, out in train_pairs:
            try:
                if not rules:
                    raise ValueError("empty rules")
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
            if (
                config_loader.REPAIR_ENABLED
                and pred0.compare_to(out0) < config_loader.REPAIR_THRESHOLD
            ):
                pred0, best_rules = run_meta_repair(
                    inp0, pred0, out0, best_rules
                )
            try:
                trace = build_trace(best_rules[0], inp0, pred0, out0)
            except Exception as e:
                if logger:
                    logger.warning(f"Trace failed: {e}")
                return [fallback_predict(g) for g in test_inputs], test_outputs, traces, best_rules
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
    if not top_sets:
        if logger:
            logger.warning("no prioritized rule sets; using fallback predictions")
        return [fallback_predict(g) for g in test_inputs], test_outputs, traces, best_rules
    for g in test_inputs:
        cand_preds = []
        for rs in top_sets:
            try:
                if not rs:
                    raise ValueError("no rules")
                cand_preds.append(simulate_rules(g, rs, logger=logger))
                for m in attn_masks:
                    cand_preds.append(
                        simulate_rules(g, rs, attention_mask=m, logger=logger)
                    )
            except Exception:
                if logger:
                    logger.warning("simulation error, skipping rule set")
                cand_preds.append(fallback_predict(g))
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
        try:
            final = soft_vote(cand_preds)
        except Exception as exc:
            if logger:
                logger.warning("soft vote failed: %s", exc)
            final = fallback_predict(g)
        predictions.append(final)

    # Persist best performing program
    if use_memory and train_pairs and best_rules:
        score = _train_score(best_rules)
        if task_id is not None:
            constraints = extract_task_constraints(train_pairs[0][0])
            save_rule_program(
                task_id, signature, best_rules, score, constraints=constraints
            )
    final_score = None
    if logger:
        logger.info("final rules: %s", rules_to_program(best_rules))
        if test_outputs:
            try:
                final_score = sum(
                    p.compare_to(o) for p, o in zip(predictions, test_outputs)
                ) / len(test_outputs)
                logger.debug(
                    f"[{task_id}] Final prediction score: {final_score:.3f}"
                )
            except Exception:
                pass

    if final_score is not None and final_score < 0.2:
        if logger:
            logger.warning(
                f"score {final_score:.2f} below threshold; using regression guard"
            )
        trace_dump = []
        if train_pairs and best_rules:
            inp0, out0 = train_pairs[0]
            for entry in trace_prediction(best_rules, inp0):
                t = build_trace(entry.rule, entry.before, entry.after, out0)
                metrics = validate_trace(t)
                trace_dump.append(
                    {
                        "rule": str(entry.rule),
                        "metrics": metrics,
                        "context": t.symbolic_context,
                    }
                )
        try:
            data = []
            if _FAILURE_LOG.exists():
                data = json.loads(_FAILURE_LOG.read_text())
            data.append({
                "task_id": task_id,
                "score": final_score,
                "trace": trace_dump,
            })
            _FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
            _FAILURE_LOG.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            if logger:
                logger.error("failed to write failure log: %s", exc)
        predictions = [fallback_predict(g) for g in test_inputs]

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
