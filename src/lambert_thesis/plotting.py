"""Plot style (matplotlib defaults) and figure saving."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .utils import REPO_ROOT

FIG_DIR = REPO_ROOT / "figures"


def setup(dpi: int = 110):
    """Reset to matplotlib defaults with a readable notebook dpi."""
    mpl.rcdefaults()
    plt.rcParams.update({"figure.dpi": dpi, "savefig.dpi": 300, "savefig.bbox": "tight"})


def savefig(name: str, fig=None):
    """Save to ``figures/<name>.png``, named by figure number, e.g. ``fig04_submarine``."""
    FIG_DIR.mkdir(exist_ok=True)
    path = Path(FIG_DIR) / f"{name}.png"
    (fig or plt.gcf()).savefig(path)
    return path


def grid_posterior(ax, P, x, y, truth=None, mass=0.95, marginals=False, cmap="ocean_r",
                   xlabel="ru0", ylabel="xu", title=None):
    """Plot a 2D grid posterior ``P`` (shape (len(y), len(x))).

    Draws the density, the dashed ``mass`` highest-density region, the ground truth (red cross) and,
    optionally, the 1D marginals along the axes together with the posterior means (green lines).
    """
    import numpy as np

    extent = (x[0], x[-1], y[0], y[-1])
    ax.imshow(P, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    flat = np.sort(P.ravel())
    level = flat[np.argmax(np.cumsum(flat) / flat.sum() > 1 - mass)]
    ax.contour(x, y, P, levels=[level], linestyles="--", colors="k", alpha=0.8)
    ax.plot([], [], "k--", label=f"{mass:.0%} highest-density region")
    if marginals:
        px, py = P.sum(axis=0), P.sum(axis=1)
        mx, my = np.sum(x * px) / px.sum(), np.sum(y * py) / py.sum()
        h = 0.06  # peak height of the marginal curves as a fraction of the axis
        ax.fill_between(x, y[0] + h * (y[-1] - y[0]) * px / px.max(), y[0], color="teal")
        ax.fill_betweenx(y, x[0], x[0] + h * (x[-1] - x[0]) * py / py.max(), color="teal")
        ax.axvline(mx, color="g", alpha=0.4)
        ax.axhline(my, color="g", alpha=0.4, label="posterior mean (green) / marginals (teal)")
    if truth is not None:
        ax.plot(*truth, "x", color="r", markersize=10, markeredgewidth=2, label="ground truth")
    ax.set(xlim=extent[:2], ylim=extent[2:], xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    return ax
