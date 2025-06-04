# Solver Architecture

This document outlines the main pipeline used by the meta-symbolic ARC solver.
Each stage corresponds to a set of modules under `arc_solver/src`.

1. **Grid** – Problems are represented as a `Grid` data structure.
   See [`core/grid.py`](../arc_solver/src/core/grid.py).
2. **Abstraction** – An `Abstractor` generates symbolic transformation
   rules from example input/output grids. Implementation lives in
   [`abstractions/abstractor.py`](../arc_solver/src/abstractions/abstractor.py).
3. **Rule ranking** – Candidate rules are scored and ordered by the
   heuristic rule ranker found in
   [`search/rule_ranker.py`](../arc_solver/src/search/rule_ranker.py).
4. **Simulation/execution** – Ranked rules are executed on the input
   grid using the basic simulator in
   [`executor/simulator.py`](../arc_solver/src/executor/simulator.py).
5. **Feedback and memory** – Execution results can be corrected and
   stored via the feedback and memory utilities located in
   [`feedback`](../arc_solver/src/feedback) and
   [`memory`](../arc_solver/src/memory).

The overall flow can thus be summarized as:

```
Grid → Abstraction → Rule ranking → Simulation/Execution → Feedback/Memory
```

Each component is intentionally lightweight and meant to be extended with more
sophisticated logic as the solver matures.
