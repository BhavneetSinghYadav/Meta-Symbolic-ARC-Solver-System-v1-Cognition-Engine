from arc_solver.src.symbolic.zone_remap import zone_remap

zone_remap_cases = {}

base = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
overlay = [[1, 1, 2], [1, 2, 2], [1, 2, 2]]
mapping = {1: 3, 2: 5}
expected = [[3, 3, 5], [3, 5, 5], [3, 5, 5]]

result = zone_remap(base, overlay, mapping)
assert result == expected
zone_remap_cases["basic"] = {
    "grid": base,
    "overlay": overlay,
    "mapping": mapping,
    "expected": expected,
}

if __name__ == "__main__":
    from pprint import pprint
    pprint(zone_remap_cases)
