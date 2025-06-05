# Meta-Symbolic ARC Solver System



Scripts located in `arc_solver/scripts` serve as entrypoints for running the solver, training search models and visualizing traces. Configuration files are stored in `arc_solver/configs` while experiment results and notebooks live in their respective directories.

Basic unit tests are included under `arc_solver/tests` and can be run with `pytest`.

For a high level view of how the solver components interact, see
[docs/architecture.md](docs/architecture.md).

## Repository Structure

```
arc_solver/
  src/            # Library code
  scripts/        # Command line entry points
  tests/          # Unit tests
  experiments/    # Sample experiment output
  notebooks/      # Jupyter notebooks
```

### Key Modules

| Module | Purpose |
| ------ | ------- |
| `core/grid.py` | Defines the :class:`Grid` data structure representing a 2‑D array and utility methods such as rotation, flipping, color counting, and comparison. |
| `symbolic/vocabulary.py` | Contains the symbolic vocabulary (`Symbol`, `SymbolicRule`, `Transformation`, etc.) used across the solver. |
| `abstractions/abstractor.py` | Extracts symbolic rules from input/output grid pairs (e.g. color replacements or translations). |
| `abstractions/rule_generator.py` | Provides helper functions like `generalize_rules` and `score_rules` for deduplicating and scoring extracted rules. |
| `abstractions/transformation_library.py` | Minimal library of transform classes such as `ReplaceColor`. |
| `segment/segmenter.py` | Implements zone‐based and connected‑component segmentation utilities. |
| `executor/simulator.py` | Applies symbolic rules to a grid and computes similarity scores. |
| `executor/conflict_resolver.py` | Removes contradictory rules using simple heuristics. |
| `executor/predictor.py` | Chooses the best rule set by simulating and scoring candidates. |
| `search/rule_ranker.py` | Ranks sets of rules using heuristics and an optional `PolicyCache`. |
| `memory/policy_cache.py` | Tracks failing rule programs to avoid repeated mistakes. |
| `feedback/correction.py` | Generates textual feedback comparing predicted and target grids. |
| `data/arc_dataset.py` | Loads ARC tasks and exposes an iterable dataset. |
| `utils/config_loader.py` | Reads JSON or YAML configuration files. |

Most other modules currently contain placeholders for future expansion (e.g. `feature_mapper.py`, `fallback_predictor.py`).

### Symbolic DSL

Simple programs can also be expressed using a lightweight DSL.  The helper
function ``parse_program_expression`` converts expressions like ``"if color == 3
and in region(Center): replace with 2"`` into :class:`SymbolicRule` objects.
Programs are simulated using ``simulate_symbolic_program`` from
``executor.simulator``.

### Dependencies

The packages depend on each other as follows:

* **Abstractions** import :class:`Grid` from `core` and structures from `symbolic`.
* **Executor** modules use `core.grid` and `symbolic` to simulate and evaluate rule programs.
* **Search** utilities rely on `memory.policy_cache` and symbolic rules to rank candidate programs.
* **Predictor** orchestrates `executor` components to pick the best rule set for a task.
* **Feedback** and **introspection** modules analyse solver outputs, while **segment** supplies zone annotations for abstraction.

For a complete overview of the processing pipeline, see the [architecture document](docs/architecture.md).

## Kaggle Usage

When running inside Kaggle notebooks the workspace is read-only. To persist the
rule memory between sessions, place `rule_memory.json` in a dataset named
`arc-memory` and mount it under `/kaggle/input/arc-memory/`. Calling
`preload_memory_from_kaggle_input()` will copy the file into the working
directory so the solver can update it.


