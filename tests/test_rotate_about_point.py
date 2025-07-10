from arc_solver.src.core.grid import Grid
from arc_solver.src.symbolic.rotate_about_point import rotate_about_point


def grid_from_list(data):
    """Create a Grid from a list of lists."""
    return Grid([row[:] for row in data])


rotate_point_testcases = {}

# case 1: rotate an L shape 90 degrees around the grid center
case_1_input = [
    [1, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
]
case_1_grid = grid_from_list(case_1_input)
case_1_center = (1, 1)
case_1_angle = 90
case_1_expected = [
    [0, 0, 0],
    [0, 0, 1],
    [1, 1, 1],
]
assert rotate_about_point(case_1_grid, case_1_center, case_1_angle).to_list() == case_1_expected
rotate_point_testcases["case_1"] = {
    "input": case_1_input,
    "center": case_1_center,
    "angle": case_1_angle,
    "expected": case_1_expected,
}

# case 2: rotate the same L shape 270 degrees around the center
case_2_input = case_1_input
case_2_grid = grid_from_list(case_2_input)
case_2_center = (1, 1)
case_2_angle = 270
case_2_expected = [
    [1, 1, 1],
    [1, 0, 0],
    [0, 0, 0],
]
assert rotate_about_point(case_2_grid, case_2_center, case_2_angle).to_list() == case_2_expected
rotate_point_testcases["case_2"] = {
    "input": case_2_input,
    "center": case_2_center,
    "angle": case_2_angle,
    "expected": case_2_expected,
}

# case 3: rotate a diagonal pattern 180 degrees around the center
case_3_input = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]
case_3_grid = grid_from_list(case_3_input)
case_3_center = (1, 1)
case_3_angle = 180
case_3_expected = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]
assert rotate_about_point(case_3_grid, case_3_center, case_3_angle).to_list() == case_3_expected
rotate_point_testcases["case_3"] = {
    "input": case_3_input,
    "center": case_3_center,
    "angle": case_3_angle,
    "expected": case_3_expected,
}

# case 4: rotate a 4x2 grid 90 degrees about the top-left corner
case_4_input = [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
]
case_4_grid = grid_from_list(case_4_input)
case_4_center = (0, 0)
case_4_angle = 90
case_4_expected = [
    [1, 3],
    [0, 0],
    [0, 0],
    [0, 0],
]
assert rotate_about_point(case_4_grid, case_4_center, case_4_angle).to_list() == case_4_expected
rotate_point_testcases["case_4"] = {
    "input": case_4_input,
    "center": case_4_center,
    "angle": case_4_angle,
    "expected": case_4_expected,
}

# case 5: rotate the 4x2 grid 270 degrees about the bottom-right corner
case_5_input = case_4_input
case_5_grid = grid_from_list(case_5_input)
case_5_center = (3, 1)
case_5_angle = 270
case_5_expected = [
    [0, 0],
    [0, 0],
    [0, 7],
    [0, 8],
]
assert rotate_about_point(case_5_grid, case_5_center, case_5_angle).to_list() == case_5_expected
rotate_point_testcases["case_5"] = {
    "input": case_5_input,
    "center": case_5_center,
    "angle": case_5_angle,
    "expected": case_5_expected,
}


if __name__ == "__main__":
    from pprint import pprint

    pprint(rotate_point_testcases)
