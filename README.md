# Meta-Symbolic ARC Solver System

This repository contains a scaffolding for a meta-symbolic solver aimed at the Abstraction and Reasoning Corpus (ARC). The project is organized into several modules such as segment, abstraction, execution and introspection. Each module lives under `arc_solver/src` and exposes simple placeholder functions or classes.

Scripts located in `arc_solver/scripts` serve as entrypoints for running the solver, training search models and visualizing traces. Configuration files are stored in `arc_solver/configs` while experiment results and notebooks live in their respective directories.

Basic unit tests are included under `arc_solver/tests` and can be run with `pytest`.

For a high level view of how the solver components interact, see
[docs/architecture.md](docs/architecture.md).
