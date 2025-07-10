"""Unit style checks for the :func:`pattern_fill` operator.

This module is intentionally decoupled from any automated test runner. It
can be executed directly to manually verify behaviour of the symbolic
``pattern_fill`` function.
"""

import os
import sys

# Ensure the project root is on ``sys.path`` when executed directly.
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from arc_solver.src.symbolic.pattern_fill_operator import pattern_fill


def make_grid(lst):
    # Local import to avoid pulling in unused heavy modules
    from arc_solver.src.core.grid import Grid
    return Grid(lst)


pattern_fill_cases = {}

# --- single_tile_center --------------------------------------------------
base_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
mask_center = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
pattern = [[1, 2], [3, 4]]
expected_center = [[1, 2, 0], [3, 4, 0], [0, 0, 0]]

_grid = make_grid(base_grid)
_mask = make_grid(mask_center)
_pat = make_grid(pattern)
original = _grid.to_list()
output = pattern_fill(_grid, _mask, _pat)
assert _grid.to_list() == original
assert output.to_list() == expected_center
pattern_fill_cases["single_tile_center"] = {
    "grid": base_grid,
    "mask": mask_center,
    "pattern": pattern,
    "expected": expected_center,
}

# --- multi_tile_overlap --------------------------------------------------
mask_overlap = [[0, 0, 0], [0, 1, 1], [0, 0, 0]]
pattern_overlap = [[5, 6], [7, 8]]
expected_overlap = [[5, 5, 6], [7, 7, 8], [0, 0, 0]]

_grid = make_grid(base_grid)
_mask = make_grid(mask_overlap)
_pat = make_grid(pattern_overlap)
original = _grid.to_list()
output = pattern_fill(_grid, _mask, _pat)
assert _grid.to_list() == original
assert output.to_list() == expected_overlap
pattern_fill_cases["multi_tile_overlap"] = {
    "grid": base_grid,
    "mask": mask_overlap,
    "pattern": pattern_overlap,
    "expected": expected_overlap,
}

# --- edge_crop_case ------------------------------------------------------
mask_edge = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
pattern_edge = [[9, 8], [7, 6]]
expected_edge = [[6, 0, 0], [0, 0, 0], [0, 0, 0]]

_grid = make_grid(base_grid)
_mask = make_grid(mask_edge)
_pat = make_grid(pattern_edge)
original = _grid.to_list()
output = pattern_fill(_grid, _mask, _pat)
assert _grid.to_list() == original
assert output.to_list() == expected_edge
pattern_fill_cases["edge_crop_case"] = {
    "grid": base_grid,
    "mask": mask_edge,
    "pattern": pattern_edge,
    "expected": expected_edge,
}

if __name__ == "__main__":
    for name, info in pattern_fill_cases.items():
        print(f"{name}: passed")
