# Composite Rule Subsystem

## Overview
Composite rules represent a chain of symbolic transformations executed as a single program. Each `CompositeRule` contains a list of `SymbolicRule` steps. They allow the solver to model patterns that require multiple sequential actions, such as tiling followed by recolouring. Chaining enables more expressive logic than single rules while keeping every step interpretable.

## Generation Pipeline
1. **Repeat rule discovery** – `generate_repeat_rules()` analyses the input and output grids to infer tiling parameters and optionally a colour replace map. When `post_process=True` it also extracts extra rules from the tiled grid to build composites. 【F:arc_solver/src/symbolic/repeat_rule.py†L42-L120】
2. **Composite assembly** – `generate_repeat_composite_rules()` wraps the repeat rule and a subsequent recolour step into one chain if the intermediate tiling matches the output except for a single colour mapping. 【F:arc_solver/src/symbolic/composite_rules.py†L18-L58】
3. **Abstraction post‑processing** – During `abstract()` the candidate composites are simulated and retained only if they improve the match over the intermediate grid. Additional fallback generation handles cases where no well‑formed composite is produced. 【F:arc_solver/src/abstractions/abstractor.py†L540-L620】
4. **Scoring** – `score_rule()` runs the composite, compares its output to the target and subtracts a cost penalty derived from `rule_cost()`. When `prefer_composites` is enabled the penalty is divided by the square root of the chain length. 【F:arc_solver/src/executor/scoring.py†L1-L110】【F:arc_solver/src/abstractions/rule_generator.py†L91-L98】

## Known Issues
* **Proxy translation bug** – earlier versions used `get_targets()` when constructing the proxy for dependency checks. This merged all step targets and broke `final_targets()` which expects only the last step. The fix introduced `final_targets()` and updated `as_symbolic_proxy()` to use it. 【F:arc_solver/src/symbolic/rule_language.py†L180-L203】
* **Dependency misordering** – Because the proxy originally reported merged targets, dependency sorting placed composites after rules that should depend on them. The helper now returns the final step's targets to ensure correct order. 【F:arc_solver/src/executor/dependency.py†L30-L40】
* **Execution skips** – Previous colour validation rejected chains when a step removed colours that never reappeared. The updated checker accepts such chains when the final colours match the training grid and logs lineage information.

## Patch Log
* **Composite rule chaining support** – initial integration enabling multi‑step programs. 【1cdbb2†L28-L30】
* **Heuristic fallback** – added to retain composites even when extraction fails. 【fbd609†L1-L12】
* **Final targets helper** – fixed dependency ordering by exposing `final_targets()` and using it in the dependency graph. 【e346c1†L1-L27】
* **Safe composite simulation** – introduced `simulate_composite_safe()` to skip invalid steps rather than aborting, improving robustness. 【775ed7†L1-L29】
* **Colour dependency refactor** – rewrote validation to check colours after each step and log lineage on failure. 【bec30d†L1-L55】
* **Scoring adjustment** – reduced penalty multiplier and scaled by square root of step count when composites are preferred. 【90274b†L1-L25】
* **Zone-aware proxy** – `as_symbolic_proxy()` now merges `input_zones` and `output_zones`, fixing dependency misordering for multi-zone composites.【F:arc_solver/src/executor/proxy_ext.py†L40-L57】
* **Zone chain proxy** – `as_symbolic_proxy()` records `zone_chain` pairs so `sort_rules_by_topology()` can order composites based on zone transitions.【F:arc_solver/src/executor/proxy_ext.py†L41-L91】【F:arc_solver/src/executor/dependency.py†L69-L111】
* **Color validation update** – composite chains are validated as a whole with colour sufficiency checked only after the final step.【F:arc_solver/src/executor/simulator.py†L212-L340】
* **Scoring overhaul** – penalties depend only on unique operation types and perfect composites receive a +0.2 bonus.【F:arc_solver/src/executor/scoring.py†L60-L109】
* **Grid growth forecast** – `simulate_composite_safe()` now predicts composite size using `grid_growth_forecast()` and aborts when the limit is exceeded.【F:arc_solver/src/executor/simulator.py†L60-L123】
* **Training colour lineage** – `validate_color_dependencies` accepts chains when the final colours match the training grid and records transitions via a lineage tracker.

## Remaining Failure Modes
* Proxy translation for advanced transformations (e.g. rotate within a composite) may misreport zones, leading to ordering mistakes.
* Large grid expansions are validated ahead of time to avoid exceeding safety limits.

## Planned Mitigations
* Further normalise cost for composites by weighting cost per unique transformation type.
* Relax colour validation when subsequent steps explicitly replace the missing colours.
* Expand `as_symbolic_proxy()` to include zone information of each step so dependency ordering can reason about spatial scopes.
* Implemented boundary checks and proactive uncertainty-map resizing when composites grow the grid.

## Stabilisation Suggestions
* Provide unit tests covering longer chains and unusual colour sequences.
* Continue logging lineage information to refine the validation logic.
* Consider a learning-based ranking model that treats composites as first-class programs rather than expensive exceptions.

