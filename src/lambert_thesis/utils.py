"""Seeding, paths and a tiny cache for expensive computations."""
from pathlib import Path
import os
import random

import numpy as np

SEED = 12345
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Seed every global RNG we might touch and return a fresh numpy Generator.

    Prefer passing the returned ``Generator`` around over relying on global state.
    Note: numba keeps its *own* RNG state; to seed it, call ``np.random.seed``
    inside an ``@njit`` function (see :func:`seed_numba`).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
    seed_numba(seed)
    return np.random.default_rng(seed)


def seed_numba(seed: int = SEED) -> None:
    """Seed numba's internal RNG (only possible from inside jitted code)."""
    try:
        from numba import njit
    except ImportError:
        return

    @njit
    def _seed(s):
        np.random.seed(s)

    _seed(seed)


def cached(name: str, compute, force: bool = False):
    """Load ``results/<name>.npz`` if present, otherwise run ``compute()`` and store it.

    ``compute`` must return a dict of numpy-compatible arrays. Used for expensive runs so
    that notebooks can be re-executed quickly; delete the file (or pass ``force=True``) to recompute.
    """
    path = RESULTS_DIR / f"{name}.npz"
    if path.exists() and not force:
        print(f"[cache] loading {path.name}")
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    print(f"[cache] computing {name} ...")
    out = compute()
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez_compressed(path, **out)
    return out
