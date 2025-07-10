## 1. System Overview
The meta‑symbolic ARC solver infers a small program of symbolic rules for each task in the Abstraction and Reasoning Corpus.  Rules are instances of `SymbolicRule` and may be chained into `CompositeRule` programs.  The solver emphasises traceable reasoning over opaque learning: every transformation can be logged and simulated step by step.

## 2. Execution Pipeline
1. **Input parsing** – Tasks are loaded as `Grid` pairs.  `solve_task` in [`executor/full_pipeline.py`](arc_solver/src/executor/full_pipeline.py) reads the JSON description and converts the examples to `Grid` objects.
2. **Rule abstraction** – For each training pair `abstract()` from [`abstractions/abstractor.py`](arc_solver/src/abstractions/abstractor.py) generates candidate rules.  `generalize_rules()` and `remove_duplicate_rules()` in [`abstractions/rule_generator.py`](arc_solver/src/abstractions/rule_generator.py) clean and deduplicate the list.  Repeat based composites are produced via [`symbolic/repeat_rule.py`](arc_solver/src/symbolic/repeat_rule.py) and [`symbolic/composite_rules.py`](arc_solver/src/symbolic/composite_rules.py).
3. **Rule scoring and selection** – Candidate sets are ranked using `probabilistic_rank_rule_sets()` and `score_rule()` functions.  Zone overlays from `segmenter.zone_overlay()` aid the attention masks.  The top set is optionally refined by visual scoring in [`introspection/visual_scoring.py`](arc_solver/src/introspection/visual_scoring.py).
4. **Conflict resolution** – `simulate_rules()` in [`executor/simulator.py`](arc_solver/src/executor/simulator.py) sorts rules via dependency utilities and applies them while logging conflicts with `mark_conflict()`.  `validate_color_dependencies()` ensures that required source colours remain available.
5. **Simulation** – Each rule is executed on the working grid.  Composite programs are expanded and validated by `simulate_composite_rule()` before application.
6. **Prediction output** – The best rule program is run on the test grids to produce the final prediction list returned by `solve_task`.  Predictions are stored in `submission.json` for evaluation.

## 3. Composite Rule System
`CompositeRule` (defined in [`symbolic/rule_language.py`](arc_solver/src/symbolic/rule_language.py)) represents a sequence of `SymbolicRule` steps.  They were introduced to capture multi‑step patterns such as repeat tiling followed by recolouring.  `generate_repeat_composite_rules()` in [`symbolic/composite_rules.py`](arc_solver/src/symbolic/composite_rules.py) constructs chains like `REPEAT → REPLACE` by inspecting intermediate grids.  During dependency filtering `CompositeRule.as_symbolic_proxy()` exposes a simplified view so that `select_independent_rules()` can schedule them alongside simple rules.

Scoring aggregates the cost of each step via `rule_cost()` in [`abstractions/rule_generator.py`](arc_solver/src/abstractions/rule_generator.py).  Intermediate colours must remain valid across the chain; otherwise `validate_color_dependencies()` will drop the rule.  Bugs remain around checking colours after each step which can suppress viable composites when intermediate recolouring introduces new colours.

## 4. Scoring System
`executor/scoring.py` implements the heuristic formula used by `solve_task`:
```
score = 0.6 * pixel_similarity + 0.3 * zone_match + 0.1 * shape_bonus
score += 0.2 * improvement - 0.02 * rule_cost
```
When `prefer_composites` is enabled the cost penalty is divided by the square root of the number of steps, reducing bias against long chains.
Common causes of rejection are low coverage ratio and high rule cost which reduce the final score below 0.0.

## 5. Debug and Instrumentation Flow
[`instrument.py`](instrument.py) decorates rule generators and simulator helpers with simple print logging.  When `solve_task` is invoked with `debug=True` a per‑task logger is created (`get_logger()` from [`utils/logger.py`](arc_solver/src/utils/logger.py)) and detailed trace entries are written.  Conflict locations and zone mismatches are recorded through `mark_conflict()` and `rule_failures_log` inside `simulator.py`.

## 6. Known Limitations & Failure Cases
* **Intermediate colour dependency** – composite rules may be discarded if a step introduces colours that are later replaced, because `validate_color_dependencies()` only checks current grid colours.
* **Scoring suppression** – complex chains accumulate high `rule_cost` leading to low scores despite perfect matches.
* **Grid expansion overflows** – applying translations or repeats can enlarge the grid beyond the working area.  `mark_conflict()` resizes the uncertainty grid to avoid `IndexError` but extreme cases still fail.
* **Divergent predictions** – tasks such as `00576224` in `submission.json` show zero‑valued predictions where rules could not be validated against the training colours.

## 7. Design Philosophy
The solver favours modular symbolic reasoning over opaque models.  Components are kept small and interpretable: each transformation is a pure function on `Grid` objects, and DSL strings describe every rule.  This design supports direct introspection (`build_trace()` and `trace_prediction()`), program memory for reuse, and easier debugging through textual logs.  The absence of heavy neural modules keeps behaviour deterministic and explainable.
