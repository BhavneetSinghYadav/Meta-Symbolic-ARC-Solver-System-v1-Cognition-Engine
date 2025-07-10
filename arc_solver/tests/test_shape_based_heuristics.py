from arc_solver.src.abstractions.abstractor import extract_shape_based_rules
from arc_solver.src.core.grid import Grid


def _has_op(rules, op):
    return any(r.transformation.params.get("op") == op for r in rules)


def test_dilate_zone_detection():
    inp = Grid([
        [0,0,0],
        [0,1,0],
        [0,0,0],
    ])
    out = Grid([
        [0,1,0],
        [1,1,1],
        [0,1,0],
    ])
    rules = extract_shape_based_rules(inp, out)
    assert _has_op(rules, "dilate_zone")


def test_erode_zone_detection():
    inp = Grid([
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ])
    out = Grid([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])
    rules = extract_shape_based_rules(inp, out)
    assert _has_op(rules, "erode_zone") or _has_op(rules, "dilate_zone")


def test_draw_line_detection():
    inp = Grid([
        [1,0,0],
        [0,0,0],
        [0,0,1],
    ])
    out = Grid([
        [1,0,0],
        [1,1,0],
        [0,1,1],
    ])
    rules = extract_shape_based_rules(inp, out)
    assert _has_op(rules, "draw_line")


def test_rotate_patch_detection():
    inp = Grid([
        [1,0,0],
        [1,0,0],
        [1,1,0],
    ])
    out = Grid([
        [0,0,0],
        [0,0,1],
        [1,1,1],
    ])
    rules = extract_shape_based_rules(inp, out)
    assert _has_op(rules, "rotate_about_point") or any(r.transformation.ttype.name == "ROTATE" for r in rules)
