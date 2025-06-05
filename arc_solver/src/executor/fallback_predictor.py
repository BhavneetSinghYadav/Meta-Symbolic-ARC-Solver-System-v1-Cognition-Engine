"""Fallback predictor for unseen tasks.

This simple policy either returns the input grid unchanged or pads it to a
square using the most frequent color. It is intended to keep the pipeline
alive when upstream rule synthesis fails.
"""


from arc_solver.src.core.grid import Grid


def predict(grid: Grid) -> Grid:
    """Return a naive guess for the output grid."""

    try:
        h, w = grid.shape()
    except Exception:
        return grid

    counts = grid.count_colors()
    mode = max(counts, key=counts.get) if counts else 0
    size = max(h, w)
    new_data = [[mode for _ in range(size)] for _ in range(size)]
    for r in range(h):
        for c in range(w):
            new_data[r][c] = grid.get(r, c)
    return Grid(new_data)
