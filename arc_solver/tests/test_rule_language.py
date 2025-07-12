import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
module_name = "arc_solver.src.symbolic.rule_language"
spec = importlib.util.spec_from_file_location(
    module_name,
    ROOT / "arc_solver" / "src" / "symbolic" / "rule_language.py",
)
rule_language = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule_language)  # type: ignore

parse_rule = rule_language.parse_rule
rule_to_dsl = rule_language.rule_to_dsl
validate_dsl_program = rule_language.validate_dsl_program
clean_dsl_string = rule_language.clean_dsl_string
get_extended_operators = rule_language.get_extended_operators
from arc_solver.src.symbolic.vocabulary import SymbolicRule, TransformationType


def test_parse_valid_rule():
    rule = parse_rule("REPLACE [COLOR=1] -> [COLOR=2]")
    assert isinstance(rule, SymbolicRule)


def test_parse_invalid_symbol():
    with pytest.raises(ValueError):
        parse_rule("REPLACE [COLOR=11] -> [COLOR=1]")


def test_roundtrip_consistency():
    rule = parse_rule("REPLACE [COLOR=3] -> [COLOR=4]")
    stringified = rule_to_dsl(rule)
    parsed = parse_rule(stringified)
    assert parsed == rule


def test_edge_tokens():
    assert not validate_dsl_program("NaN")
    assert not validate_dsl_program("None")
    with pytest.raises(ValueError):
        parse_rule(clean_dsl_string("FOO [COLOR=1]->[COLOR=2]"))


def test_parse_new_transforms():
    r1 = parse_rule("ROTATE90 [SHAPE=A] -> [SHAPE=A]")
    r2 = parse_rule("SHAPE_ABSTRACT [SHAPE=A] -> [SHAPE=A]")
    assert r1.transformation.ttype is TransformationType.ROTATE90
    assert r2.transformation.ttype is TransformationType.SHAPE_ABSTRACT


def _roundtrip(dsl: str) -> None:
    obj = parse_rule(dsl)
    out = rule_to_dsl(obj)
    obj2 = parse_rule(out)
    assert rule_to_dsl(obj2) == out
    assert obj2.transformation.params == obj.transformation.params
    assert obj2.meta == obj.meta


def test_operator_registry_keys():
    ops = get_extended_operators()
    expected = {
        "mirror_tile",
        "pattern_fill",
        "rotate_about_point",
        "zone_remap",
        "dilate_zone",
        "erode_zone",
    }
    assert expected <= set(ops)


def test_extended_operator_roundtrip():
    cases = [
        "mirror_tile(axis=horizontal, repeats=2) [REGION=All] -> [REGION=All]",
        "pattern_fill(mapping={1:2}) [REGION=All] -> [REGION=All]",
        "rotate_about_point(pivot=(1,2), angle=90) [REGION=All] -> [REGION=All]",
        "zone_remap(mapping={1:3}) [REGION=All] -> [REGION=All]",
        "dilate_zone(zone_id=1) [ZONE=1] -> [ZONE=1]",
        "erode_zone(zone_id=2) [ZONE=2] -> [ZONE=2]",
    ]
    for dsl in cases:
        _roundtrip(dsl)
