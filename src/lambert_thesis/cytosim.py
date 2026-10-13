"""Chapter 3.6: bond inference on force curves from Cytosim simulations.

Data format (one CSV per bond configuration and polymer stiffness): one column per force curve
(``curve0, curve1, ...``) and a last column with the time; a curve is NaN after its bond broke.
Forces are in pN, k_BT = 4.112 pN nm.

The unbinding model is the Bell-Evans rate of :mod:`lambert_thesis.bonds`; here it is evaluated on
the measured (noisy) forces directly, with the exact per-interval probability exp(-dt r_u).
"""
from pathlib import Path

import numba
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KBT = 4.112   # pN nm
EPS = 1e-14


def load_csv(path):
    """Load a Cytosim force-curve file.

    Returns a dict with ``F`` (n_curves, n_times; NaN after unbinding), ``t``, ``dt``,
    ``events`` (index of the last recorded force = the interval in which the bond broke) and
    ``events_t`` (the corresponding unbinding times).
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    t = data[:, -1]
    F = data[:, :-1].T
    events = np.isnan(F).argmax(axis=1)
    events[events == 0] = t.shape[0]          # curve never breaks within the recorded time
    events -= 1
    return dict(F=F, t=t, dt=np.diff(t).mean(), events=events, events_t=t[events], name=Path(path).stem)


# ---------------------------------------------------------------------------
# Force model and unbinding statistics
# ---------------------------------------------------------------------------
def mean_force_curve(data, rel_additional_time=1.0, n_slope=20):
    """Representative force model F~(t): the mean over all recorded curves, cut where the first
    curve ends, and continued linearly (slope of the last ``n_slope`` points) for another
    ``rel_additional_time`` times its duration."""
    F = np.nanmean(data["F"], axis=0)
    last = np.argmax(np.isnan(F))
    F, t = F[:last - 1], data["t"][:last - 1]
    dt = data["dt"]
    slope = (F[-1] - F[-n_slope - 1]) / (n_slope * dt)
    ext = np.arange(1, int(t[-1] * rel_additional_time / dt))
    return dict(t=np.concatenate((t, t[-1] + ext * dt)), F=np.concatenate((F, F[-1] + ext * dt * slope)), dt=dt)


def simulate_unbinding_times(model, ru0, xu, n, rng, kbT=KBT):
    """Sample ``n`` unbinding times for the force model ``model`` (p_u = dt r_u per step)."""
    pu = model["dt"] * ru0 * np.exp(xu * model["F"] / kbT)
    u = rng.random((n, len(pu)))
    broke = u < pu[None, :]
    idx = np.where(broke.any(axis=1), broke.argmax(axis=1), -1)    # -1: did not break (rare)
    return model["t"][idx]


def unbinding_time_density(model, ru0, xu, kbT=KBT):
    """Expected distribution of unbinding times (eq. 3.21): integrate dp/dt = -p r_u(|F~(t)|) with
    p(t+dt) = p(t) exp(-dt r_u) (stable, unlike an Euler step) and normalise its derivative."""
    dt = model["dt"]
    P = np.cumprod(np.exp(-dt * ru0 * np.exp(xu * np.abs(model["F"]) / kbT)))
    p = np.gradient(P)
    return p / (np.sum(p) * dt)


def rejection_probability(model, ru0, xu, f_thr, kbT=KBT):
    """Marginal probability P_r that a trajectory generated with the force model is rejected,
    i.e. that the bond breaks while the force is still below the threshold ``f_thr``."""
    p_stay = np.exp(-model["dt"] * ru0 * np.exp(xu * model["F"] / kbT))
    survive_before = np.concatenate(([1.0], np.cumprod(p_stay)[:-1]))
    return np.sum(survive_before * (1 - p_stay) * (model["F"] < f_thr))


# ---------------------------------------------------------------------------
# Likelihood and posterior sampling
# ---------------------------------------------------------------------------
@numba.njit(parallel=True, cache=True)
def _loglik(F, events, dt, ru0, xu, kbT):
    total = 0.0
    for s in numba.prange(F.shape[0]):
        acc = 0.0
        for t in range(int(events[s]) + 1):
            p = np.exp(-dt * ru0 * np.exp(F[s, t] * xu / kbT))      # bond survives the interval
            if t == events[s]:
                p = 1.0 - p                                         # bond breaks
            acc += np.log(p + 1e-14)
        total += acc
    return total


def log_likelihood(data, ru0, xu, kbT=KBT):
    """Log-likelihood of all force curves of a data set (eq. 3.5/3.6, exact per-interval probability)."""
    return _loglik(data["F"], data["events"], data["dt"], ru0, xu, kbT)


def log_likelihood_rejection(data, ru0, xu, f_thr, kbT=KBT):
    """Log-likelihood under the rejection protocol (section 3.3.2): every recorded curve passed the
    threshold, so each contributes L(T) / (1 - P_r(ru0, xu)). P_r is computed with the mean force curve
    of the data (continued for three times its duration)."""
    model = mean_force_curve(data, rel_additional_time=3)
    Pr = rejection_probability(model, ru0, xu, f_thr, kbT)
    return log_likelihood(data, ru0, xu, kbT) - len(data["F"]) * np.log(1 - Pr + EPS)


def posterior_kde(samples, x, y, bandwidth=1.0):
    """Density of 2D posterior samples on the grid (x, y), from a linear-kernel KDE in standardised
    coordinates. Returns an array (len(y), len(x)) scaled to a maximum of one."""
    pipe = Pipeline([("scaler", StandardScaler()), ("kde", KernelDensity(kernel="linear", bandwidth=bandwidth))])
    pipe.fit(samples)
    X, Y = np.meshgrid(x, y)
    logp = pipe.score_samples(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return np.exp(logp - np.nanmax(logp))
