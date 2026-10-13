"""Chapter 4: Bayesian inference of wall torques for active Brownian particles (and active rods).

Setup (dimensionless units): a flat wall at x = 0, particles start at negative x.
The wall is a WCA potential with cut-off x_c = 2^(1/6) (measured from the wall); its force is
    F(x) = -24 eps (x^6 - 2) / x^13   for x > -2^(1/6),   0 otherwise   (x < 0 is the distance's negative).
ABP (type A):   r' = Pe e(phi) + F e_x + sqrt(2 D_T) xi,    phi' = q2 f(phi) F + sqrt(2 D_R) chi
Rod (type R):   no empirical torque; the WCA force acts on N_S test sites along the rod axis, producing
                torques from their lever arms. The observed position is the rod's tip.
Torque functions f: spectral modes  f(phi) = sum_n alpha_n sin(-n phi)  or a quadratic B-spline.

The likelihood p(Y | theta) of observed positions is estimated with a particle filter over the unobserved
orientation (section 4.4), or approximately with a linear Kalman filter.
"""
import numba
import numpy as np

Q2 = 1.54308184422          # geometry factor of the empirical torque (type A)
WCA_CUT = -2 ** (1 / 6)


# ---------------------------------------------------------------------------
# Rod geometry: diffusion coefficients of spherocylinders from A. Lüders, E. Zander, P. Nielaba, J. Chem. Phys. 155(10) (2021),
# https://doi.org/10.1063/5.0060063
# ---------------------------------------------------------------------------
def ratio_diffusion_rotational(p):
    """D_R / D_T^parallel of a spherocylinder with aspect ratio p."""
    rot = 1 / (p * p) * (np.log(p) - 0.3512 + 0.7804 / p - 0.09801 / (p * p))
    para = 0.5 * (np.log(p) - 0.1404 + 1.034 / p - 0.228 / (p * p))
    return rot / para


def ratio_diffusion_orthogonal(p):
    """D_T^perp / D_T^parallel of a spherocylinder with aspect ratio p."""
    ortho = 0.5 * (np.log(p) + 0.8369 + 0.5551 / p - 0.06066 / (p * p))
    para = np.log(p) - 0.1404 + 1.034 / p - 0.228 / (p * p)
    return ortho / para


# ---------------------------------------------------------------------------
# Torque functions
# ---------------------------------------------------------------------------
@numba.njit(cache=True)
def wca_force(x, eps=1.0):
    if x > WCA_CUT:
        return -24 * eps * (x**6 - 2) / x**13
    return 0.0


@numba.njit(cache=True)
def torque_modes(phi, params):
    """Spectral torque function sum_n alpha_n sin(-n phi), n = 1..len(params)."""
    t = 0.0
    for i in range(params.shape[0]):
        t += params[i] * np.sin(-phi * (i + 1))
    return t


@numba.njit(cache=True)
def torque_single_mode(phi, params):
    """Single mode with amplitude alpha and wavelength parameter k: alpha sin(-phi / k)."""
    return params[0] * np.sin(-phi / params[1])


@numba.njit(cache=True)
def torque_single_mode_deriv(phi, params):
    return -params[0] * np.cos(-phi / params[1]) / params[1]


@numba.njit(cache=True)
def bspline2_kernel(u):
    """Quadratic B-spline basis function on [0, 3]."""
    if u < 0:
        return 0.0
    if u < 1:
        return 0.5 * u**2
    if u < 2:
        return 0.5 * (-2 * u**2 + 6 * u - 3)
    if u < 3:
        return 0.5 * (u - 3) ** 2
    return 0.0


@numba.njit(cache=True)
def bspline2(x, coef):
    """B^(2)(x) = sum_i coef_i kappa((N-3) x - i + 2.5), equidistant knots on [0, 1]."""
    n = coef.shape[0]
    s = 0.0
    for i in range(n):
        s += coef[i] * bspline2_kernel((n - 3) * x - i + 2.5)
    return s


@numba.njit(cache=True)
def torque_bspline(phi, params):
    """Torque function from B-spline coefficients ``params`` for orientations in [0, pi], continued
    antisymmetrically (f(-phi) = -f(phi)) and periodically, with zeros at phi = 0 and pi."""
    n = params.shape[0]
    full = np.zeros(2 * n + 5)
    full[0] = -params[0]
    full[2:2 + n] = params
    full[3 + n:3 + 2 * n] = -params[::-1]
    full[-1] = params[0]
    x = (phi % (2 * np.pi)) / (2 * np.pi)
    return bspline2(x, full)


TORQUES = {"modes": 0, "single_mode": 1, "bspline": 2}


@numba.njit(cache=True)
def _torque(kind, phi, params):
    if kind == 0:
        return torque_modes(phi, params)
    if kind == 1:
        return torque_single_mode(phi, params)
    return torque_bspline(phi, params)


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------
def simulate_abp(n_steps, dt, phi0, Pe, Dt, Dr, params, rng, x0=-2.0, y0=0.0, kind="modes", eps=1.0):
    """Type-A ABP trajectories (one per initial orientation in ``phi0``). Returns x, y, phi (n_traj, n_steps)."""
    n = len(phi0)
    X, Y, P = np.zeros((n, n_steps)), np.zeros((n, n_steps)), np.zeros((n, n_steps))
    X[:, 0], Y[:, 0], P[:, 0] = x0, y0, phi0
    k = TORQUES[kind]
    params = np.asarray(params, float)
    for t in range(1, n_steps):
        F = np.array([wca_force(x, eps) for x in X[:, t - 1]])
        X[:, t] = X[:, t - 1] + np.cos(P[:, t - 1]) * Pe * dt + np.sqrt(2 * Dt * dt) * rng.standard_normal(n) + F * dt
        Y[:, t] = Y[:, t - 1] + np.sin(P[:, t - 1]) * Pe * dt + np.sqrt(2 * Dt * dt) * rng.standard_normal(n)
        tq = np.array([Q2 * _torque(k, p, params) for p in P[:, t - 1]]) * F
        P[:, t] = P[:, t - 1] + tq * dt + np.sqrt(2 * Dr * dt) * rng.standard_normal(n)
    return X, Y, P


def simulate_rod(n_steps, dt, phi0, Pe, Dt, aspect_ratio, rng, n_sites=2, x0=-2.0, y0=0.0, eps=1.0):
    """Type-R rod trajectories: WCA forces on ``n_sites`` test sites spread over the rod's length produce the
    torque. Diffusion follows the spherocylinder parametrisation. Returns tip coordinates x, y and phi."""
    n = len(phi0)
    Dr = Dt * ratio_diffusion_rotational(aspect_ratio)
    X, Y, P = np.zeros((n, n_steps)), np.zeros((n, n_steps)), np.zeros((n, n_steps))
    X[:, 0], Y[:, 0], P[:, 0] = x0, y0, phi0
    off_max = (aspect_ratio - 1) / 2
    offsets = np.linspace(-off_max, off_max, n_sites)
    for t in range(1, n_steps):
        torque, F_total = np.zeros(n), np.zeros(n)
        for off in offsets:
            xs = X[:, t - 1] + off * np.cos(P[:, t - 1])
            for k in range(n):
                F = wca_force(xs[k], eps)
                F_total[k] += F
                torque[k] -= F * off * np.sin(P[k, t - 1])
        X[:, t] = X[:, t - 1] + np.cos(P[:, t - 1]) * Pe * dt + np.sqrt(2 * Dt * dt) * rng.standard_normal(n) + Dt * F_total * dt
        Y[:, t] = Y[:, t - 1] + np.sin(P[:, t - 1]) * Pe * dt + np.sqrt(2 * Dt * dt) * rng.standard_normal(n)
        P[:, t] = P[:, t - 1] + torque * dt + np.sqrt(2 * Dr * dt) * rng.standard_normal(n)
    return X + np.cos(P) * off_max, Y + np.sin(P) * off_max, P


# ---------------------------------------------------------------------------
# Likelihood estimators
# ---------------------------------------------------------------------------
@numba.njit(cache=True)
def _resample(w, u, out):
    """Structured resampling of normalised weights ``w`` with offset ``u`` into ``out``."""
    n = w.shape[0]
    c = np.cumsum(w)
    probes = (u + np.arange(n) / n) % 1.0
    out[:] = np.minimum(np.searchsorted(c, probes), n - 1)


@numba.njit(parallel=True, cache=True)
def loglik_particle_filter(x, y, phi0, dt, Pe, Dt, Dr, eps, kind, q2, params, n_particles, seed,
                           phi0_spread=np.pi / 180, clip_norm=np.inf, keep_phi=False):
    """Particle-filter log-likelihood of the observed positions (n_traj, n_steps) for every parameter set
    (rows of ``params``). Each parameter set is filtered with its own seed (``seed + row``), so results do not
    depend on the thread scheduling. With ``keep_phi`` the particle swarm of the first parameter set is
    returned (n_traj, n_steps, n_particles) for diagnostics."""
    n_sets, (n_traj, n_steps) = params.shape[0], x.shape
    out = np.zeros(n_sets)
    swarm = np.zeros((n_traj, n_steps, n_particles if keep_phi else 1))
    for s in numba.prange(n_sets):
        np.random.seed(seed + s)
        th = params[s]
        if np.sqrt(np.sum(th**2)) > clip_norm:
            out[s] = -np.inf
            continue
        logl = 0.0
        idx = np.zeros(n_particles, np.int64)
        for d in range(n_traj):
            phi = phi0[d] + np.random.randn(n_particles) * phi0_spread
            for t in range(n_steps - 1):
                F = wca_force(x[d, t], eps)
                dx = x[d, t] + np.cos(phi) * Pe * dt - (x[d, t + 1] - F * dt)
                dy = y[d, t] + np.sin(phi) * Pe * dt - y[d, t + 1]
                lw = -0.5 * (dx**2 + dy**2) / (2 * Dt * dt)
                m = np.max(lw)
                w = np.exp(lw - m)
                logl += m + np.log(np.sum(w))
                _resample(w / np.sum(w), np.random.rand(), idx)
                phi = phi[idx]
                if keep_phi and s == 0:
                    swarm[d, t, :] = phi
                for k in range(n_particles):
                    phi[k] += q2 * _torque(kind, phi[k], th) * F * dt + np.sqrt(2 * Dr * dt) * np.random.normal()
        out[s] = logl
    return out, swarm


@numba.njit(parallel=True, cache=True)
def loglik_kalman(x, y, phi0, dt, Pe, Dt, Dr, eps, q2, params, phi0_std=5 * np.pi / 180):
    """Linear Kalman-filter approximation of the log-likelihood for the single-mode torque
    (params rows: alpha, k): the orientation is tracked as a Gaussian, linearised around its mean."""
    n_sets, (n_traj, n_steps) = params.shape[0], x.shape
    out = np.zeros(n_sets)
    for s in numba.prange(n_sets):
        th = params[s]
        logl = 0.0
        for d in range(n_traj):
            mu, sd = phi0[d], phi0_std
            for t in range(n_steps - 1):
                F = wca_force(x[d, t], eps)
                px, py = np.cos(mu) * Pe * dt, np.sin(mu) * Pe * dt
                norm = np.sqrt(px**2 + py**2)
                dx = x[d, t] + px - (x[d, t + 1] - F * dt)
                dy = y[d, t] + py - y[d, t + 1]
                logl += -0.5 * (dx**2 + dy**2) / (2 * Dt * dt + (sd * Pe * dt) ** 2)
                off = (-dy * px / norm + dx * py / norm) / (Pe * dt)
                off_sd = np.sqrt(2 * Dt * dt) / (Pe * dt)
                mu = (sd**2 * (mu + off) + off_sd**2 * mu) / (sd**2 + off_sd**2)
                sd = np.sqrt(1 / (1 / sd**2 + 1 / off_sd**2) + 2 * Dr * dt)
                tq = q2 * torque_single_mode(mu, th) * F
                dtq = q2 * torque_single_mode_deriv(mu, th) * F
                mu += tq * dt
                sd *= np.exp(-dtq * dt)
        out[s] = logl
    return out


def bspline_projection(f_values, phis, n_coef):
    """Least-squares B-spline coefficients (for :func:`torque_bspline`) that best represent the torque values
    ``f_values`` at orientations ``phis`` -- e.g. to centre a sampler on a known or estimated torque function."""
    design = np.zeros((len(phis), n_coef))
    for i in range(n_coef):
        e = np.zeros(n_coef); e[i] = 1.0
        design[:, i] = [torque_bspline(p, e) for p in phis]
    return np.linalg.lstsq(design, np.asarray(f_values, float), rcond=None)[0]
