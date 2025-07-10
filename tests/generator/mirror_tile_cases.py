"""Synthetic test cases for the mirror_tile operator."""

from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.operators import mirror_tile


# Base grids -------------------------------------------------------------

base_grids = {
    "grid_2x2": Grid([[1, 2], [3, 4]]),
    "grid_3x1": Grid([[1], [2], [3]]),
    "grid_1x2": Grid([[1, 2]]),
}


# Generate test cases ----------------------------------------------------
mirror_tile_testcases = {}

# case 1: 2x2 grid, horizontal tiling, count=2 -> 2x4 grid
case_1_in = base_grids["grid_2x2"]
case_1_out = mirror_tile(case_1_in, "horizontal", 2)
mirror_tile_testcases["case_1"] = {
    "input": case_1_in.to_list(),
    "output": case_1_out.to_list(),
}

# case 2: 2x2 grid, vertical tiling, count=2 -> 4x2 grid
case_2_in = base_grids["grid_2x2"]
case_2_out = mirror_tile(case_2_in, "vertical", 2)
mirror_tile_testcases["case_2"] = {
    "input": case_2_in.to_list(),
    "output": case_2_out.to_list(),
}

# case 3: 3x1 grid, horizontal tiling, count=3 -> 3x3 grid
case_3_in = base_grids["grid_3x1"]
case_3_out = mirror_tile(case_3_in, "horizontal", 3)
mirror_tile_testcases["case_3"] = {
    "input": case_3_in.to_list(),
    "output": case_3_out.to_list(),
}

# case 4: 1x2 grid, vertical tiling, count=3 -> 3x2 grid
case_4_in = base_grids["grid_1x2"]
case_4_out = mirror_tile(case_4_in, "vertical", 3)
mirror_tile_testcases["case_4"] = {
    "input": case_4_in.to_list(),
    "output": case_4_out.to_list(),
}


if __name__ == "__main__":
    # Simple visualisation when run manually
    from pprint import pprint

    for name, pair in mirror_tile_testcases.items():
        print(f"\n{name}:")
        print("Input:")
        for row in pair["input"]:
            print(row)
        print("Output:")
        for row in pair["output"]:
            print(row)

