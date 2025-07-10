# Meta-Symbolic ARC Solver System

## 1. Project Overview

This repository contains a symbolic solver for the [Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC).  The solver infers a set of symbolic rules from provided training examples and applies them to unseen test grids.  The design emphasises interpretability: rules are represented as structured `SymbolicRule` objects and can be chained into `CompositeRule` programs.  Recent updates added composite rule scoring, failure logging and a safer conflict tracking mechanism.

## 2. System Architecture

```
Input Grid → Abstraction → Rule Ranking → Simulation → Feedback/Memory
```

* **Grid** – core data structure in [`core/grid.py`](arc_solver/src/core/grid.py)
* **Abstraction** – extracts candidate rules in [`abstractions/abstractor.py`](arc_solver/src/abstractions/abstractor.py)
* **Rule ranking** – heuristics and dependency checks in [`search`](arc_solver/src/search)
* **Simulation** – applies rules with conflict resolution in [`executor`](arc_solver/src/executor)
* **Feedback & memory** – optional refinement and program storage in [`feedback`](arc_solver/src/feedback) and [`memory`](arc_solver/src/memory)

A more detailed walkthrough is available in [architecture.md](architecture.md).

## 3. Setup Instructions

1. Python 3.9 or later is recommended.
2. Install the package in editable mode and fetch dependencies:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```
3. Run the unit tests to verify the environment:
   ```bash
   pytest -q
   ```

## 4. How to Run

The main entrypoint for solving tasks is `arc_solver/scripts/run_solver.py`:
```bash
python arc_solver/scripts/run_solver.py <path_to_arc_tasks>
```
For single-task experimentation or scoring analysis use [`instrument.py`](instrument.py):
```bash
python instrument.py --bundle arc-agi_training_challenges.json --task_id 00000001
```
`full_pipeline.py` under `executor` exposes `solve_task` and `solve_task_iterative` functions used by the scripts.

## 5. Code Modules Breakdown

| Path | Role |
| ---- | ---- |
| [`src/core`](arc_solver/src/core) | Grid utilities and helpers |
| [`src/abstractions`](arc_solver/src/abstractions) | Rule extraction and generalisation |
| [`src/symbolic`](arc_solver/src/symbolic) | DSL parser and rule definitions |
| [`src/executor`](arc_solver/src/executor) | Rule simulation, ranking and pipeline orchestration |
| [`src/scoring`](arc_solver/src/scoring) | Heuristics for rule and program scoring |
| [`src/memory`](arc_solver/src/memory) | Optional rule program cache |
| [`scripts`](arc_solver/scripts) | CLI utilities for evaluation and experimentation |

## 6. Output & Logging

Predicted grids for datasets are written to `submission.json`.  When `solve_task` runs with `debug=True` a detailed log file is created under `logs/` describing extracted rules, conflicts and scoring statistics.  Failures below a score threshold are appended to `logs/failure_log.json` for later inspection.

## 7. Example Tasks & Visualizations

Sample notebooks in [`arc_solver/notebooks`](arc_solver/notebooks) demonstrate zone overlays and trace debugging.  The `docs/` folder contains architecture diagrams.  Generated visualisations and experiment outputs are stored under `arc_solver/experiments`.

## 8. Known Issues / Future Work

* Composite rules are still penalised by rule cost.  The scoring module mitigates this by dividing the penalty by the square root of the chain length when `prefer_composites` is enabled【F:arc_solver/src/executor/scoring.py†L6-L10】【F:arc_solver/src/executor/scoring.py†L88-L103】.
* Conflict marking resizes the uncertainty grid to avoid `IndexError` when rules expand the working grid【F:arc_solver/src/executor/simulator.py†L128-L170】.
* Some advanced scoring functions are placeholders (`TODO` in comments) and require tuning.

## 9. Developer Notes

* Run `pytest` before submitting changes.  All current tests should pass (142 tests)【69a983†L1-L4】.
* Keep new modules under the existing `arc_solver/src` hierarchy.
* Contributions that extend the rule vocabulary or improve the ranking heuristics are welcome; please document new behaviour in this README.

