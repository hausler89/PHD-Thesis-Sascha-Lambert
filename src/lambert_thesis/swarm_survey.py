"""Chapter 5.5: survey of bridged swarms over the noise level sigma (Figures 36-38).

For every sigma, swarms of ``n`` Ballistic Brownian particles are propagated from the origin to t=1:
unconditioned, with the null bridge (shift = slope = 0), and with bridges of varying shift or slope.
From the endpoints we keep
* the widths (standard deviations) of the rotated endpoint clouds (Figure 37),
* the maximum achievable likelihood of a Gaussian target of width sigma_m (Figure 38),
* 2D endpoint histograms for the noise levels shown in Figure 36.

Results are stored per sigma in ``results/swarm_survey_n<n>/sigma_<i>.npz`` so that a long survey can be
interrupted and resumed, and the resolution ``n`` can be increased step by step.
"""
from pathlib import Path

import numpy as np
from scipy.stats import binned_statistic_2d

from . import bridges
from .utils import RESULTS_DIR

SIGMA_M = (1e-1, 1e-2, 1e-3)


def endpoint_histogram(x, y, rotate=False, edge=0.1, nbins=500):
    """2D histogram of endpoints (counts), on a range padded by ``edge`` of the extent."""
    if rotate:
        x, y = bridges.rotate_to_mean(x, y)
    (x0, x1), (y0, y1) = (x.min(), x.max()), (y.min(), y.max())
    rng_ = ((x0 - edge * (x1 - x0), x1 + edge * (x1 - x0)), (y0 - edge * (y1 - y0), y1 + edge * (y1 - y0)))
    st = binned_statistic_2d(x, y, None, statistic="count", bins=nbins, range=rng_)
    cx = (st.x_edge[1:] + st.x_edge[:-1]) / 2
    cy = (st.y_edge[1:] + st.y_edge[:-1]) / 2
    return st.statistic.T.astype(np.float32), cx, cy


def rotated_std(x, y):
    xr, yr = bridges.rotate_to_mean(x, y)
    return np.std(xr), np.std(yr)


def survey_sigma(i, sigma, n, dt, shifts, slopes, rng, hist_sigmas=(), batch=10_000):
    """All statistics for one noise level ``sigma`` (index ``i``); see module docstring."""
    br = bridges.ConditionedBridge()
    run = lambda shift=None, slope=None: bridges.ballistic_endpoints(sigma, n, dt, rng, shift, slope, br, batch)
    seed = lambda: int(rng.integers(2**31))
    out = {}

    xu, yu = run()                                                        # unconditioned
    out["std_uncond"] = rotated_std(xu, yu)
    out["lik_uncond"] = [bridges.max_swarm_likelihood(xu, yu, s, seed())[1] for s in SIGMA_M]

    xn, yn = run(0.0, 0.0)                                                # null bridge
    out["std_null"] = rotated_std(xn, yn)
    lik_null, lik_ref = [], []
    for s in SIGMA_M:
        centre, lik = bridges.max_swarm_likelihood(xn, yn, s, seed())
        lik_null.append(lik)
        lik_ref.append(bridges.swarm_likelihood(xu, yu, centre, s))     # unconditioned swarm, same target
    out["lik_null"], out["lik_null_ref"] = lik_null, lik_ref
    del xu, yu, xn, yn

    std_shift, lik_shift, std_slope = [], [], []
    for k, shift in enumerate(shifts):
        x, y = run(shift, 0.0)
        std_shift.append(rotated_std(x, y))
        lik_shift.append([bridges.max_swarm_likelihood(x, y, s, seed())[1] for s in SIGMA_M])
        if i in hist_sigmas:
            out[f"hist_shift_{k}"], out[f"hist_shift_{k}_x"], out[f"hist_shift_{k}_y"] = endpoint_histogram(x, y)
            out[f"histrot_shift_{k}"], out[f"histrot_shift_{k}_x"], out[f"histrot_shift_{k}_y"] = \
                endpoint_histogram(x, y, rotate=True)
    for k, slope in enumerate(slopes):
        x, y = run(0.0, slope)
        std_slope.append(rotated_std(x, y))
        if i in hist_sigmas:
            out[f"hist_slope_{k}"], out[f"hist_slope_{k}_x"], out[f"hist_slope_{k}_y"] = endpoint_histogram(x, y)
    out.update(std_shift=std_shift, lik_shift=lik_shift, std_slope=std_slope)
    return {k: np.asarray(v) for k, v in out.items()}


def run_survey(sigmas, n, dt, shifts, slopes, seed, hist_sigmas=(), verbose=True):
    """Survey all noise levels, resuming from per-sigma files in ``results/``. Returns a list of dicts.

    Every sigma gets its own generator (seed + i), so that the results do not depend on which
    noise levels were already computed in an earlier, interrupted run.
    """
    folder = Path(RESULTS_DIR) / f"swarm_survey_n{n}"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for i, sigma in enumerate(sigmas):
        path = folder / f"sigma_{i}.npz"
        if not path.exists():
            if verbose:
                print(f"[survey] sigma {i + 1}/{len(sigmas)} = {sigma:.3g} ...", flush=True)
            res = survey_sigma(i, sigma, n, dt, shifts, slopes, np.random.default_rng(seed + i), hist_sigmas)
            np.savez_compressed(path, **res)
        with np.load(path) as f:
            results.append({k: f[k] for k in f.files})
    return results
