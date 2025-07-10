## 1. Purpose
The scoring system ranks candidate rules by their expected contribution to solving an ARC task.  Every rule receives a heuristic score in the range `[0.0, 1.0]`.  Scores guide rule selection, conflict resolution and composite construction.

## 2. Scoring Formula
`score_rule()` in [`executor/scoring.py`](arc_solver/src/executor/scoring.py) evaluates a rule by simulating it and combining similarity metrics with a complexity penalty:

```
base = 0.6 * after_pixel + 0.3 * zone_match + 0.1 * shape_bonus
if after_pixel > before_pixel:
    base += 0.2 * (after_pixel - before_pixel)
penalty = 0.005 * unique_ops
bonus = 0.2 if isinstance(rule, CompositeRule) and base == 1.0 else 0.0
final = base - penalty + bonus
```
Lines【F:arc_solver/src/executor/scoring.py†L60-L109】 detail this logic. `after_pixel` is the raw grid match after applying the rule. `zone_match` measures how well labelled zones align, and `shape_bonus` rewards correct output shape. `unique_ops` counts the distinct operation types in the rule.

### 2.1 Penalty Terms
`unique_ops()` returns the number of distinct transformations used by a rule. This replaces the previous heavy dependence on `rule_cost`.

### 2.2 Composite Bonus
A composite rule that perfectly matches the target receives a `+0.2` bonus.

### 2.3 Score Range
Scores are no longer clipped to zero; negative values propagate to allow accurate ranking of poor candidates.

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
* **Bias against composites** – multi-step programs may still score slightly below atomic rules when similarity is imperfect.

## 7. Refinements & Suggestions
Several refinements were applied or proposed:
* **Dynamic cost adjustment** – the penalty factor now uses the number of unique operations.
* **Similarity boosting** – the improvement bonus remains capped at `+0.2`.
* **Composite bonus** – perfect chains receive an explicit bonus instead of penalty scaling.

## 8. Scoring Philosophy
The solver balances precision against generality.  A rule should explain as much of the output as possible without unnecessary complexity.  Penalties discourage overly specific or lengthy programs, while bonuses promote transformations that make substantial progress.  The scoring system therefore aims to select concise yet powerful rules that generalise across training examples.
