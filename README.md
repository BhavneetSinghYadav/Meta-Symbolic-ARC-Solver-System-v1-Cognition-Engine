# Meta-Symbolic ARC Solver System

This repository contains a scaffolding for a meta-symbolic solver aimed at the Abstraction and Reasoning Corpus (ARC). The project is organized into several modules such as segmentation, abstraction, execution and introspection. Each module lives under `arc_solver/src` and exposes simple placeholder functions or classes.

Scripts located in `arc_solver/scripts` serve as entrypoints for running the solver, training search models and visualizing traces. Configuration files are stored in `arc_solver/configs` while experiment results and notebooks live in their respective directories.

Basic unit tests are included under `arc_solver/tests` and can be run with `pytest`.

The introspection module now provides utilities for constructing execution
traces and validating solver plans.  A lightweight narration helper is also
included which can optionally call the OpenAI API when `OPENAI_API_KEY` is set.
Without an API key the narration falls back to a simple textual summary of the
actions in the plan.
