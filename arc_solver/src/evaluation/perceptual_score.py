import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.metrics import structural_similarity as ssim

from arc_solver.src.core.grid import Grid


def grid_to_image(grid: Grid, dpi: int = 100) -> np.ndarray:
    """Render a Grid to an RGB image array."""
    h, w = grid.shape()
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(np.array(grid.data), cmap="tab20", interpolation="nearest")
    ax.axis("off")
    fig.tight_layout(pad=0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    return image


def perceptual_similarity_score(pred: Grid, target: Grid) -> float:
    """Return SSIM-based similarity score between two grids."""
    pred_img = grid_to_image(pred)
    tgt_img = grid_to_image(target)
    if pred_img.shape != tgt_img.shape:
        return 0.0
    h, w, _ = pred_img.shape
    min_dim = min(h, w)
    if min_dim < 7:
        return float(np.mean(np.all(pred_img == tgt_img, axis=-1)))
    win = 7 if min_dim >= 7 else min_dim
    if win % 2 == 0:
        win -= 1
    sim, _ = ssim(pred_img, tgt_img, channel_axis=-1, win_size=win, full=True)
    return float(sim)
