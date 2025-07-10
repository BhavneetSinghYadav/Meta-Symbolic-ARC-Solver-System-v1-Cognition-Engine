## 1. Purpose
The scoring system ranks candidate rules by their expected contribution to solving an ARC task.  Every rule receives a heuristic score in the range `[0.0, 1.0]`.  Scores guide rule selection, conflict resolution and composite construction.

## 2. Scoring Formula
`score_rule()` in [`executor/scoring.py`](arc_solver/src/executor/scoring.py) evaluates a rule by simulating it and combining similarity metrics with a complexity penalty:

```
base = 0.6 * after_pixel + 0.3 * zone_match + 0.1 * shape_bonus
base += 0.2 * improvement
penalty = 0.02 * rule_cost
final = base - penalty
```
Lines【F:arc_solver/src/executor/scoring.py†L87-L104】 detail this logic.  `after_pixel` is the raw grid match after applying the rule. `zone_match` measures how well labelled zones align, and `shape_bonus` rewards correct output shape.  `improvement` compares the predicted match against the input grid, giving up to `+0.2` extra credit.

### 2.1 Penalty Terms
`rule_cost()` from [`abstractions/rule_generator.py`](arc_solver/src/abstractions/rule_generator.py) assigns a heuristic cost based on zone length and DSL complexity:

```
zone_size = len(zone_str)
transform_complexity = len(rule_to_dsl(rule).split("->")[1])
return 0.5 * zone_size + transform_complexity
```
【F:arc_solver/src/abstractions/rule_generator.py†L91-L98】
The cost grows for longer conditions and more complex transformations.  Composite rules sum the cost of their steps.

### 2.2 Composite Penalties
By default the penalty multiplies the cost by `0.02`.  If the rule is a composite and `prefer_composites=True`, the penalty is divided by `sqrt(len(steps))` to lessen bias against long chains【F:arc_solver/src/executor/scoring.py†L95-L102】.

### 2.3 Negative Clipping
After penalties, scores are clipped into `[0.0, 1.0]` so negative values are truncated at zero【F:arc_solver/src/executor/scoring.py†L103-L108】.  This prevents runaway penalties but may hide useful signal from otherwise promising rules.

## 3. Scoring Functions
### score_rule()
`score_rule(input_grid, output_grid, rule, prefer_composites=False)` returns the final score.  During abstraction the score may be adjusted by strategy bonuses and chain length penalties as shown around lines【F:arc_solver/src/abstractions/abstractor.py†L608-L623】.

### rule_cost()
`rule_cost(rule)` estimates the complexity for sparsity ranking.  It is used both in `score_rule()` and for deduplication heuristics.

## 4. Score Interpretation
Scores above `0.8` are considered strong matches likely to contribute directly to the final program.  Values around `0.5` indicate partial coverage or high cost.  Anything at `0.0` is pruned during ranking.

## 5. Scoring Logs & Instrumentation
[`instrument.py`](instrument.py) prints detailed score diagnostics.  For each rule it reports the raw similarity, adjusted score and improvement gain:
```
rule <desc> raw=<raw> adj=<score> gain=<delta>
```
Logging lines appear in the range【F:instrument.py†L92-L106】 and help trace why specific rules were kept or discarded.

## 6. Issues Observed
* **Over‑penalisation of multi‑step rules** – long composites accumulate cost faster than their match score grows, leading to low final scores.
* **Bias against composites** – even with `prefer_composites`, chains of more than two steps often score below short single rules.
* **Negative clipping** – truncation at zero masks differences between failed but potentially useful rules, reducing learning signal.

## 7. Refinements & Suggestions
Several refinements were applied or proposed:
* **Dynamic cost adjustment** – reducing the penalty factor when the rule greatly improves the grid.
* **Similarity boosting** – adding an improvement bonus as implemented in lines【F:arc_solver/src/executor/scoring.py†L90-L93】.
* **Composite‑aware score normalization** – dividing the cost by `sqrt(steps)` when composites are preferred.

## 8. Scoring Philosophy
The solver balances precision against generality.  A rule should explain as much of the output as possible without unnecessary complexity.  Penalties discourage overly specific or lengthy programs, while bonuses promote transformations that make substantial progress.  The scoring system therefore aims to select concise yet powerful rules that generalise across training examples.
