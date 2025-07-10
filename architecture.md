## 1. System Overview
The meta‑symbolic ARC solver infers a small program of symbolic rules for each task in the Abstraction and Reasoning Corpus.  Rules are instances of `SymbolicRule` and may be chained into `CompositeRule` programs.  The solver emphasises traceable reasoning over opaque learning: every transformation can be logged and simulated step by step.

Recent updates extend the rule language with `SHAPE_ABSTRACT` and `ROTATE90`
transformation types, enabling explicit reasoning about shape similarity and
grid rotation.

## 2. Execution Pipeline
1. **Input parsing** – Tasks are loaded as `Grid` pairs.  `solve_task` in [`executor/full_pipeline.py`](arc_solver/src/executor/full_pipeline.py) reads the JSON description and converts the examples to `Grid` objects.
2. **Rule abstraction** – For each training pair `abstract()` from [`abstractions/abstractor.py`](arc_solver/src/abstractions/abstractor.py) generates candidate rules.  `generalize_rules()` and `remove_duplicate_rules()` in [`abstractions/rule_generator.py`](arc_solver/src/abstractions/rule_generator.py) clean and deduplicate the list.  Repeat based composites are produced via [`symbolic/repeat_rule.py`](arc_solver/src/symbolic/repeat_rule.py) and [`symbolic/composite_rules.py`](arc_solver/src/symbolic/composite_rules.py).
3. **Rule scoring and selection** – Candidate sets are ranked using `probabilistic_rank_rule_sets()` and `score_rule()` functions.  Zone overlays from `segmenter.zone_overlay()` aid the attention masks.  The top set is optionally refined by visual scoring in [`introspection/visual_scoring.py`](arc_solver/src/introspection/visual_scoring.py).
4. **Conflict resolution** – `simulate_rules()` in [`executor/simulator.py`](arc_solver/src/executor/simulator.py) sorts rules via dependency utilities and applies them while logging conflicts with `mark_conflict()`.  `validate_color_dependencies()` now simulates the entire composite chain and verifies colour sufficiency only after the final step.  When the final colours match the training grid the check passes even if some source colours vanish, and a lineage tracker can record each transition.
5. **Simulation** – Each rule is executed on the working grid.  Composite programs are expanded and validated by `simulate_composite_rule()` before application.
6. **Prediction output** – The best rule program is run on the test grids to produce the final prediction list returned by `solve_task`.  Predictions are stored in `submission.json` for evaluation.

## 3. Composite Rule System
`CompositeRule` (defined in [`symbolic/rule_language.py`](arc_solver/src/symbolic/rule_language.py)) represents a sequence of `SymbolicRule` steps.  They were introduced to capture multi‑step patterns such as repeat tiling followed by recolouring.  `generate_repeat_composite_rules()` in [`symbolic/composite_rules.py`](arc_solver/src/symbolic/composite_rules.py) constructs chains like `REPEAT → REPLACE` by inspecting intermediate grids.  During dependency filtering `CompositeRule.as_symbolic_proxy()` exposes a simplified view so that `select_independent_rules()` can schedule them alongside simple rules.

Scoring aggregates similarity metrics and applies a small complexity penalty based on the number of unique operations in the rule.  Colour validation simulates the full composite chain and only checks colour sufficiency at the final step, so temporary recolouring no longer causes rejection.

## 4. Scoring System
`executor/scoring.py` implements the heuristic formula used by `solve_task`:
```
base = 0.55 * after_pixel + 0.35 * zone_match + 0.1 * shape_bonus
if after_pixel > before_pixel:
    base += 0.25 * (after_pixel - before_pixel)
penalty = 0.006 * unique_ops
bonus = 0.2 if isinstance(rule, CompositeRule) and base >= 0.95 else 0.0
final = base - penalty + bonus
```
Negative scores are no longer clipped, allowing fine-grained ranking.

## 5. Debug and Instrumentation Flow
[`instrument.py`](instrument.py) decorates rule generators and simulator helpers with simple print logging.  When `solve_task` is invoked with `debug=True` a per‑task logger is created (`get_logger()` from [`utils/logger.py`](arc_solver/src/utils/logger.py)) and detailed trace entries are written.  Conflict locations and zone mismatches are recorded through `mark_conflict()` and `rule_failures_log` inside `simulator.py`.

## 6. Known Limitations & Failure Cases
* **Grid expansion overflows** – composite steps now forecast growth and abort when exceeding `64x64`, resizing the uncertainty grid proactively when safe.
* **Divergent predictions** – tasks such as `00576224` in `submission.json` show zero‑valued predictions where rules could not be validated against the training colours.

## 7. Design Philosophy
The solver favours modular symbolic reasoning over opaque models.  Components are kept small and interpretable: each transformation is a pure function on `Grid` objects, and DSL strings describe every rule.  This design supports direct introspection (`build_trace()` and `trace_prediction()`), program memory for reuse, and easier debugging through textual logs.  The absence of heavy neural modules keeps behaviour deterministic and explainable.
