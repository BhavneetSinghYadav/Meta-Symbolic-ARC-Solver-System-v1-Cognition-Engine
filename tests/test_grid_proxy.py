import numpy as np
from arc_solver.src.core.grid_proxy import GridProxy


def test_entropy_and_overlay_cache():
    arr = np.array([
        [1, 1, 0, 0, 2],
        [1, 1, 0, 0, 2],
        [3, 3, 3, 0, 2],
        [3, 0, 0, 0, 2],
        [3, 4, 4, 4, 2],
    ])
    proxy = GridProxy(arr)
    e1 = proxy.get_entropy()
    e2 = proxy.get_entropy()
    assert e1 == e2

    ov1 = proxy.get_zone_overlay()
    ov2 = proxy.get_zone_overlay()
    assert ov1 == ov2


def test_shape_metadata_complex():
    grid = np.zeros((10, 10), dtype=int)
    grid[1:4, 1:4] = 5
    grid[6:9, 6:9] = 6
    grid[0, 0] = 1
    grid[9, 9] = 2
    proxy = GridProxy(grid)
    shapes = proxy.get_shapes()
    assert any(s["zone_id"] == 0 for s in shapes)
    assert all("bbox" in s for s in shapes)
    assert all("center" in s for s in shapes)
