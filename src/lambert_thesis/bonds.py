"""Chapter 3: Bayesian inference of (vimentin-microtubule) bond parameters.

Model
-----
* Force-extension curve of a stretched polymer: entropic regime (eq. 3.3, a worm-like-chain type
  relation) followed by a linear "Hookean" regime. :func:`force_curve` returns a fast explicit
  approximation (Chebyshev polynomial) of this curve.
* Bell-Evans unbinding rate  r_u(F) = r_u0 * exp(F x_u / k_BT)                         (eq. 3.2)
* Per observation interval dt the bond breaks with p_u = 1 - exp(-dt r_u(F))           (eq. 3.4),
  evaluated to first order, p_u = min(1, dt r_u(F)), which is accurate for dt r_u << 1.

A trajectory is a 1D array of forces F_1..F_n sampled every ``dt``; by construction the bond
was intact for F_1..F_{n-1} and broke during the last interval (at F_n).

Everything is in arbitrary units with k_BT = 1.
"""
import numba
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy import optimize

KBT = 1.0


# ---------------------------------------------------------------------------
# Polymer force-extension curve
# ---------------------------------------------------------------------------
def entropic_extension(F, Lp=1.5, kbT=KBT):
    """Relative extension x/d_IF of the polymer under force F (eq. 3.3):
    x/d = coth(2 Lp F / kBT) - kBT / (2 Lp F)."""
    a = 2 * Lp * F / kbT
    return 1 / np.tanh(a) - 1 / a


def _d_entropic_extension(F, Lp=1.5, kbT=KBT):
    """Derivative dx/dF of :func:`entropic_extension`."""
    return 1 / (2 * Lp * F**2 / kbT) - 2 * Lp / kbT / np.sinh(2 * Lp * F / kbT) ** 2


def force_curve(k, Lp=1.5, kbT=KBT, x_max=5.0, dx=1e-3, deg_subsampling=16):
    """Explicit force-extension curve F(x) as a Chebyshev polynomial.

    The entropic relation (eq. 3.3) is followed until its slope dF/dx reaches the spring
    constant ``k`` (``k`` ~ 1/d_IF, the "hardness"); beyond that point the force grows linearly
    with slope ``k`` (Hookean regime). The sampled curve is then fitted with a Chebyshev
    polynomial, which is much faster to evaluate than solving eq. 3.3 with a root finder.

    Returns
    -------
    numpy.polynomial.Chebyshev  -- call it with an extension ``x`` to obtain ``F(x)``.
    """
    # transition force F*: where the entropic stiffness dF/dx equals k
    F_star = optimize.root_scalar(lambda f: _d_entropic_extension(f, Lp, kbT) - 1 / k,
                                  bracket=(1e-12, 30)).root
    x_lo, x_hi = entropic_extension(dx, Lp, kbT), entropic_extension(F_star, Lp, kbT)

    # sample the entropic part such that the points are (roughly) equidistant in x
    n = max(int((x_hi - x_lo) / dx), 2)
    F = np.linspace(1e-12, F_star, n)
    for _ in range(5):
        c = np.cumsum(1 / _d_entropic_extension(F, Lp, kbT))
        F = (c - c[0]) / (c[-1] - c[0]) * F_star + dx
    x = entropic_extension(F, Lp, kbT)

    # linear (Hookean) continuation
    xh = np.arange(dx, x_max - x[-1], dx)
    x = np.concatenate(([0.0], x, x[-1] + xh))
    F = np.concatenate(([0.0], F, F[-1] + k * xh))
    return cheb.Chebyshev.fit(x, F, len(x) // deg_subsampling)


# ---------------------------------------------------------------------------
# Unbinding model and synthetic data
# ---------------------------------------------------------------------------
def unbinding_rate(F, ru0, xu, kbT=KBT):
    """Bell-Evans rate r_u = r_u0 exp(F x_u / kBT) (eq. 3.2)."""
    return ru0 * np.exp(F * xu / kbT)


def p_unbind(F, ru0, xu, dt, kbT=KBT):
    """Probability that the bond breaks within one interval dt at (quasi-constant) force F (eq. 3.4),
    to first order in dt: p_u = min(1, dt r_u(F))."""
    return np.clip(dt * unbinding_rate(F, ru0, xu, kbT), 0, 1)


def simulate_trajectory(force, ru0, xu, v, dt, rng, kbT=KBT, max_steps=1_000_000):
    """Sample one force trajectory: strain the polymer with velocity ``v`` (x = v t), check every
    ``dt`` whether the bond breaks. Returns the forces up to and including the breaking interval.

    ``force`` is any callable F(x), e.g. the result of :func:`force_curve`.
    """
    # Same as looping step by step ("evaluate F, draw a uniform number, break?"), but done in
    # vectorised chunks of time steps for speed. Extensions beyond the tabulated range of the
    # force curve (its Chebyshev domain) are clamped to its end.
    x_max = force.domain[1] if hasattr(force, "domain") else np.inf
    strain = lambda steps: np.minimum(v * steps * dt, x_max)
    chunk = 1024
    for start in range(0, max_steps, chunk):
        steps = np.arange(start, start + chunk)
        F = force(strain(steps))
        broke = rng.random(chunk) < p_unbind(F, ru0, xu, dt, kbT)
        if broke.any():
            end = start + np.argmax(broke)
            return force(strain(np.arange(end + 1)))
    raise RuntimeError("bond did not break within max_steps")


def simulate_trajectories(force, ru0, xu, v, dt, rng, n):
    """``n`` trajectories. ``v`` is a number, an array of ``n`` velocities, or a callable ``v(rng)``
    drawing one velocity per trajectory.

    Returns (list of force arrays, array of velocities).
    """
    if callable(v):
        vs = np.array([v(rng) for _ in range(n)], float)
    else:
        vs = np.broadcast_to(np.asarray(v, float), (n,)).copy()
    return [simulate_trajectory(force, ru0, xu, vi, dt, rng) for vi in vs], vs


# ---------------------------------------------------------------------------
# Likelihood and grid posterior
# ---------------------------------------------------------------------------
def split_observations(trajectories, censor_below=None):
    """Collect all 'stayed bound' forces and all 'unbound' forces of a data set.

    ``censor_below``: drop the data points before a trajectory first exceeds this force
    (mitigation 1, section 3.3.1). Trajectories that never exceed it contribute nothing.
    """
    bound, unbound = [], []
    for F in trajectories:
        if censor_below is not None:
            above = np.nonzero(F > censor_below)[0]
            if len(above) == 0:
                continue
            F = F[above[0]:]
        bound.append(F[:-1])
        unbound.append(F[-1:])
    return np.concatenate(bound), np.concatenate(unbound)


def log_likelihood_grid(trajectories, ru0, xu, dt, kbT=KBT, censor_below=None):
    """Log-likelihood (eq. 3.5/3.6) on a grid: ``ru0`` (n_r,) x ``xu`` (n_x,) -> array (n_x, n_r).

    Every observation is independent (memoryless bond), so the log-likelihood is a sum over all
    data points:  stayed bound -> log(1 - p_u(F)),  broke -> log p_u(F),  with p_u = min(1, dt r_u(F)).
    Identical forces (e.g. from trajectories at the same velocity) are evaluated only once.
    """
    Fb, Fu = split_observations(trajectories, censor_below)
    Fb, counts = np.unique(Fb, return_counts=True)
    ru0 = np.asarray(ru0, float)
    xu = np.asarray(xu, float)
    logL = np.empty((len(xu), len(ru0)))
    with np.errstate(divide="ignore", invalid="ignore"):      # p_u = 1 or r_u0 = 0 give log(0) = -inf
        for i, x in enumerate(xu):
            pb = np.clip(dt * ru0[:, None] * np.exp(x * Fb[None, :] / kbT), 0, 1)      # (n_r, n_forces)
            pu = dt * ru0[:, None] * np.exp(x * Fu[None, :] / kbT)                    # (n_r, n_events)
            logL[i] = (np.log1p(-pb) * counts).sum(axis=1) + np.log(pu).sum(axis=1)
    return logL


def normalize_log(logp):
    """exp-normalise a log density on a grid so that it sums to one."""
    p = np.exp(logp - np.max(logp))
    return p / p.sum()


def hdi_threshold(P, mass=0.95):
    """Density level whose super-level set contains ``mass`` of the (grid) probability ``P``.
    Draw ``contour(P, levels=[hdi_threshold(P)])`` to show the highest-density region."""
    flat = np.sort(P.ravel())
    cdf = np.cumsum(flat) / flat.sum()
    return flat[np.argmax(cdf > 1 - mass)]


# ---------------------------------------------------------------------------
# Experimental design (section 3.5)
# ---------------------------------------------------------------------------
@numba.njit(parallel=True, cache=True)
def _loglik_sum_over_grid(forces, offsets, ru0, xu, dt, kbT):
    """For every trajectory (forces[offsets[t]:offsets[t+1]]), the sum of its log-likelihood over all
    grid points (ru0[k], xu[k]). Parallel over trajectories."""
    n_traj = offsets.shape[0] - 1
    out = np.zeros(n_traj)
    for t in numba.prange(n_traj):
        a, b = offsets[t], offsets[t + 1]
        acc = 0.0
        for k in range(ru0.shape[0]):
            for i in range(a, b):
                p = min(dt * ru0[k] * np.exp(forces[i] * xu[k] / kbT), 1.0)
                if i == b - 1:                 # last data point: the bond broke
                    acc += np.log(p + 1e-10)
                else:                          # bond stayed intact
                    acc += np.log(1.0 - p + 1e-10)
        out[t] = acc
    return out


def utility_per_trajectory(trajectories, ru0_grid, xu_grid, dt, kbT=KBT):
    """Utility of each (test) trajectory for the experimental design of section 3.5.

    For a flat prior on the (ru0, xu) grid this is the log-likelihood of the trajectory summed over
    all grid points: the more a trajectory 'cuts away' of the parameter space, the more negative it is.
    Averaging it over trajectories simulated at a design velocity v gives the expected utility EU(v).
    """
    R, X = np.meshgrid(ru0_grid, xu_grid)
    forces = np.concatenate(trajectories).astype(float)
    offsets = np.concatenate(([0], np.cumsum([len(F) for F in trajectories]))).astype(np.int64)
    return _loglik_sum_over_grid(forces, offsets, R.ravel(), X.ravel(), float(dt), float(kbT))
