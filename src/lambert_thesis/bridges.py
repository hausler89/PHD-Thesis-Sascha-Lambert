"""Chapter 5: Brownian bridges for (Ballistic) Brownian motion and the bridged ABP propagator.

Paraxial picture (section 5.3): the vertical deviation of a particle is the double cumulative sum
B^(2) of the rotational noise W; its final velocity is the single cumulative sum B^(1). A
*conditioned bridge* is a noise sequence for which both the endpoint B^(2)_N ("shift") and the final
step B^(1)_N ("slope") take prescribed values. It is obtained by conditioning multivariate Gaussians.
"""
import numba
import numpy as np


class CorrelatedRNG:
    """Gaussian random numbers organised in named blocks.

    A block is created on first request and returned unchanged on later requests (so the same
    noise is reused), until it is re-rolled. Re-rolling can keep a correlation ``corr`` with the old
    values: new = corr * old + sqrt(1 - corr^2) * fresh, as used for correlated pseudo-marginal MCMC.
    """

    def __init__(self, seed=None):
        self.gen = np.random.default_rng(seed)
        self.block, self.old_blocks, self.corr = {}, {}, {}

    def get(self, name, shape=None, corr=0.0):
        self.corr[name] = corr
        if name not in self.block:
            self.block[name] = self.gen.normal(size=(1,) if shape is None else shape)
        elif shape is not None:
            assert self.block[name].shape == tuple(shape), f"block {name} requested with a different shape"
        return self.block[name]

    def reroll_block(self, name, corr=None):
        corr = self.corr[name] if corr is None else corr
        self.old_blocks[name] = self.block[name]
        fresh = self.gen.normal(size=self.block[name].shape)
        self.block[name] = corr * self.block[name] + np.sqrt(1 - corr**2) * fresh

    def reroll(self, corr=None):
        for name in self.block:
            self.reroll_block(name, corr)

    def revert(self):
        self.block.update(self.old_blocks)


def integration_matrix(n):
    """Cumulative summation S_ij = [i >= j] (lower triangular ones): the discrete integral."""
    return np.tril(np.ones((n, n)))


def condition_last(cov):
    """Condition a zero-mean Gaussian with covariance ``cov`` on its last component.

    Returns (conditioned covariance of the remaining components, transformer Sigma_12 Sigma_22^-1):
    the conditioned mean for a value a of the last component is ``transformer * a``.
    """
    S11, S12 = cov[:-1, :-1], cov[:-1, -1:]
    S21, S22 = cov[-1:, :-1], cov[-1:, -1:]
    transformer = S12 @ np.linalg.inv(S22)
    return S11 - transformer @ S21, transformer


class ConditionedBridge:
    """Second-order Brownian bridges with conditioned endpoint and conditioned last step.

    Algorithm (section 5.3): integrate the white-noise covariance once (Sigma^(1)), condition the final
    velocity, integrate again (Sigma^(2)), condition the endpoint, draw from the result. Covariance
    factors are cached per length.
    """

    def __init__(self):
        self.cache = {}

    def _factors(self, L):
        if L not in self.cache:
            S1 = integration_matrix(L - 1) @ np.eye(L - 1) @ integration_matrix(L - 1).T
            S1_bar, T1 = condition_last(S1)
            S2 = integration_matrix(L - 2) @ S1_bar @ integration_matrix(L - 2).T
            S2_bar, T2 = condition_last(S2)
            self.cache[L] = dict(T1=T1, T2=T2, chol=np.linalg.cholesky(S2_bar),
                                 std1=np.sqrt(S1[-1, -1]), std2=np.sqrt(S2[-1, -1]))
        return self.cache[L]

    def get(self, L, endpoint, enddiff, block_name, corr, rng, mean_only=False):
        """Samples B2[i, k], i = 0..L, of k bridges with B2[-1, k] = endpoint[k] and
        B2[-1, k] - B2[-2, k] = enddiff[k]. The noise comes from ``rng`` (a :class:`CorrelatedRNG`)."""
        c = self._factors(L)
        mu1 = c["T1"] * enddiff
        mu2 = np.cumsum(mu1, axis=0)
        mu2_bar = mu2[:-1, :] + c["T2"] @ (endpoint - enddiff - mu2[-1:, -1:])
        w = rng.get(block_name, mu2_bar.shape, corr)
        B2 = (c["chol"] @ w) * (1 - mean_only) + mu2_bar
        B2 = np.concatenate((0 * endpoint[None, :], B2, (endpoint - enddiff)[None, :], endpoint[None, :]))
        return dict(mu=mu2_bar, w=B2, std1=c["std1"], std2=c["std2"])


def bridge_noise(B2):
    """Second finite difference of bridge samples B2 (L+1, k): the noise sequence W (k, L) that, when
    integrated twice, produces the bridge."""
    dw = np.concatenate([B2[[0], :], np.diff(B2, axis=0)])
    return np.diff(dw, axis=0).T


def bridge_prior_cov(N):
    """Joint covariance of (shift, slope) = (B^(2)_N, B^(1)_N) of the *unconditioned* process (section 5.4)."""
    return np.array([[N * (N + 1) * (2 * N + 1) / 6, N * (N + 1) / 2],
                     [N * (N + 1) / 2, N]])


def bridge_logl(shift, slope, N):
    """Log density of a bridge configuration under the unconditioned process (bivariate Gaussian)."""
    cov = bridge_prior_cov(N)
    x = np.array([shift, slope])
    return -0.5 * (np.log(np.linalg.det(cov)) + x @ np.linalg.inv(cov) @ x + 2 * np.log(2 * np.pi))


def gaussian_level_ellipse(cov, level, mean=(0.0, 0.0), n=100):
    """Contour of a bivariate Gaussian at ``level`` times its maximum density, as (2, n) points."""
    r = np.sqrt(-2 * np.log(level))
    t = np.linspace(0, 2 * np.pi, n)
    return np.linalg.cholesky(cov) @ (r * np.array([np.cos(t), np.sin(t)])) + np.asarray(mean)[:, None]


def ballistic_brownian(sigma, n_particles, dt, rng, T=1.0):
    """Ballistic Brownian motion (eq. 5.4): x' = cos phi, y' = sin phi, phi' = sqrt(sigma) zeta,
    started at the origin heading in +x, integrated with Euler-Maruyama until T.

    Returns x, y of shape (n_particles, n_steps + 1).
    """
    n_steps = int(T / dt + 1)
    dt_ = T / n_steps
    w = rng.standard_normal((n_particles, n_steps))
    X = np.zeros((n_particles, 3))
    out = [X.copy()]
    for k in range(n_steps):
        dx, dy = np.cos(X[:, 2]), np.sin(X[:, 2])   # propulsion along the orientation at the start of the step
        X[:, 2] += np.sqrt(dt_ * sigma) * w[:, k]
        X[:, 0] += dt_ * dx
        X[:, 1] += dt_ * dy
        out.append(X.copy())
    out = np.stack(out, -1)
    return out[:, 0], out[:, 1]


# ---------------------------------------------------------------------------
# Bridged Ballistic Brownian motion and swarm statistics (sections 5.5, 5.6)
# ---------------------------------------------------------------------------
class FreshRNG:
    """Same interface as :class:`CorrelatedRNG`, but every request returns new Gaussian numbers."""

    def __init__(self, gen):
        self.gen = gen

    def get(self, name, shape=None, corr=0.0):
        return self.gen.normal(size=(1,) if shape is None else shape)


def ballistic_swarm(sigma, n, dt, rng, shift=None, slope=None, bridge=None, X0=(0.0, 0.0, 0.0),
                    T=1.0, v=1.0, force=None, torque=None, hires=False, mean_only=False, noise_units=False):
    """Propagate ``n`` Ballistic Brownian particles from t=0 to T with Euler-Maruyama.

    If ``shift``/``slope`` are given, the rotational noise is replaced by bridge-conditioned noise
    (a :class:`ConditionedBridge` ``bridge`` is required): the paraxial endpoint (shift) and the final
    rate (slope) are conditioned, scaled to the particle's units via the step size and noise level.

    Optional interactions: ``force(X) -> (fx, fy)`` and ``torque(X, t) -> dphi/dt`` (e.g. a wall or a
    magnetic field). With ``noise_units=True``, shift and slope are taken directly as the bridge conditions
    (endpoint and last step of the double-cumulative noise) without rescaling.
    Returns the final states (n, 3) or, with ``hires``, the full history (n, 3, steps+1).
    """
    n_steps = int(T / dt + 1)
    dt_ = T / n_steps
    if shift is not None:
        assert sigma > 0, "the bridge needs rotational noise"
        if noise_units:
            shiftP, slopeP = shift * np.ones(n), slope * np.ones(n)
        else:
            shiftP = shift / (dt_ * np.sqrt(dt_ * sigma)) * np.ones(n)
            slopeP = slope / np.sqrt(dt_ * sigma) * np.ones(n)
        B2 = bridge.get(n_steps + 1, shiftP, slopeP, "swarm", 0.0, FreshRNG(rng), mean_only)["w"]
        w = bridge_noise(B2)
    else:
        w = rng.standard_normal((n, n_steps))
    X = np.zeros((n, 3)) + np.asarray(X0, float)
    out = [X.copy()] if hires else None
    for k in range(n_steps):
        c, s = np.cos(X[:, 2]), np.sin(X[:, 2])
        dphi = 0.0 if torque is None else torque(X, k * dt_)
        fx, fy = (0.0, 0.0) if force is None else force(X)
        X[:, 2] += np.sqrt(dt_ * sigma) * w[:, k]
        X[:, 0] += dt_ * (v * c + fx)
        X[:, 1] += dt_ * (v * s + fy)
        X[:, 2] += dt_ * dphi
        if hires:
            out.append(X.copy())
    return np.stack(out, -1) if hires else X


def ballistic_endpoints(sigma, n, dt, rng, shift=None, slope=None, bridge=None, batch=1000):
    """Final (x, y) of ``n`` particles, simulated in batches of ``batch`` (fresh noise per batch)."""
    xs, ys = [], []
    for _ in range(max(n // batch, 1)):
        X = ballistic_swarm(sigma, min(batch, n), dt, rng, shift, slope, bridge)
        xs.append(X[:, 0]); ys.append(X[:, 1])
    return np.concatenate(xs), np.concatenate(ys)


def rotate_to_mean(x, y):
    """Rotate the point cloud about the origin so that its centre of mass lies on the x-axis."""
    a = np.arctan2(np.mean(y), np.mean(x))
    return np.cos(-a) * x - np.sin(-a) * y, np.sin(-a) * x + np.cos(-a) * y


def density_quantile_levels(density, quantiles=(0.01, 0.10, 0.5, 0.95)):
    """Density values whose super-level sets *exclude* the given fractions of the mass (the lowest
    contour wraps 99 % of the mass, the highest 5 %)."""
    v = np.sort(density.ravel())
    c = np.cumsum(v) / v.sum()
    return [v[np.argmax(c > q)] for q in quantiles]


@numba.njit(parallel=True, cache=True)
def _mean_gaussian(x, y, mx, my, s):
    acc = np.zeros(x.shape[0])
    norm = 1.0 / (2 * np.pi * s * s)
    for i in numba.prange(x.shape[0]):
        acc[i] = norm * np.exp(-((x[i] - mx) ** 2 + (y[i] - my) ** 2) / (2 * s * s))
    return acc.mean()


def swarm_likelihood(x, y, centre, sigma_m):
    """Mean density of an isotropic Gaussian target of width ``sigma_m`` at ``centre`` over the swarm,
    i.e. the likelihood that the transition produces an observation at ``centre``."""
    return _mean_gaussian(x, y, float(centre[0]), float(centre[1]), float(sigma_m))


def max_swarm_likelihood(x, y, sigma_m, seed, bounds=((0, 1), (-1, 1))):
    """Optimise the location of the Gaussian target (differential evolution) to maximise
    :func:`swarm_likelihood`. Returns (best centre, best likelihood)."""
    from scipy.optimize import differential_evolution

    res = differential_evolution(lambda mu: -swarm_likelihood(x, y, mu, sigma_m), bounds, seed=seed)
    return res.x, -res.fun


def empirical_level_ellipse(x, y, level, n=100):
    """Density contour at ``level`` x maximum of the Gaussian fitted (mean, covariance) to samples (x, y)."""
    return gaussian_level_ellipse(np.cov(x, y), level, mean=(np.mean(x), np.mean(y)), n=n)


def kl_divergence_from_samples(N, x, y):
    """KL divergence between the Gaussian fitted to samples of (shift, slope) and the prior of the bridge
    parameters for bridges of length ``N`` (both bivariate normal)."""
    mu = np.array([np.mean(x), np.mean(y)])
    S = np.cov(x, y)
    P = bridge_prior_cov(N)
    Pi = np.linalg.inv(P)
    return 0.5 * (np.trace(Pi @ S) + mu @ Pi @ mu - 2 + np.log(np.linalg.det(P) / np.linalg.det(S)))
