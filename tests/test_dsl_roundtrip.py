from arc_solver.src.symbolic.vocabulary import get_extended_operators
from arc_solver.src.symbolic.rule_language import parse_rule, rule_to_dsl

SAMPLES = {
    "mirror_tile": ("horizontal", 2),
    "pattern_fill": (None, None),
    "draw_line": ((0, 0), (1, 1), 1),
    "dilate_zone": (1,),
    "erode_zone": (1,),
    "rotate_about_point": ((1, 1), 90),
    "zone_remap": (None, {1: 2}),
}


def test_registry_roundtrip():
    ops = get_extended_operators()
    for name, factory in ops.items():
        args = SAMPLES.get(name, ())
        rule = factory(*args)
        rule.meta = {}
        dsl = rule_to_dsl(rule)
        parsed = parse_rule(dsl)
        assert parsed == rule
