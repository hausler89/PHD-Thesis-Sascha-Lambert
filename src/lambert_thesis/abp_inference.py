"""Chapter 5.7-5.8: particle-filter inference of an ABP's velocity with (optionally) bridged noise.

State of a particle: X = (x, y, phi). Between two observations the dynamics
    dx = v cos(phi) dt + F_x dt,  dy = v sin(phi) dt + F_y dt,  dphi = sqrt(2 D_r) dW
is integrated with Euler-Maruyama steps of size ~dt. With ``use_bridges`` the rotational noise of a step
is a conditioned Brownian bridge (shift/slope chosen per observation, e.g. by a PPO agent).

Parameters are passed in a dict ``p`` (as in the particle filter's propagator interface):
    logv, logDr, dt            dynamics (log velocity, log rotational diffusion, integration step)
    rng                        :class:`lambert_thesis.bridges.CorrelatedRNG` providing the noise blocks
    bridge                     :class:`lambert_thesis.bridges.ConditionedBridge`
    use_bridges                bool
    target_shift, target_slope bridge conditions for the current step (in units of the dynamics)
    bridge_prior               "joint" (bivariate prior of shift and slope) or "independent"
    force                      optional callable X -> (fx, fy), e.g. a harmonic potential
    hires                      store the intermediate states
"""
import numpy as np

from .bridges import bridge_logl, bridge_noise


def propagate(T0, T1, X0, k, p):
    """Propagate all particles from T0 to T1 (one observation interval)."""
    Dr, v = np.exp(p["logDr"]), np.exp(p["logv"])
    n_steps = int((T1 - T0) / p["dt"] + 1)
    n = X0.shape[0]
    dt_ = (T1 - T0) / n_steps
    logl = reward_logl = 0.0
    if p.get("use_bridges", False):
        slope_factor = 1 / np.sqrt(2 * Dr * dt_)
        shift_factor = 1 / (np.sqrt(2 * Dr * dt_) * v * dt_)
        slopeP = p.get("target_slope", 0.0) * slope_factor * np.ones(n)
        shiftP = p.get("target_shift", 0.0) * shift_factor * np.ones(n)
        res = p["bridge"].get(n_steps + 1, shiftP, slopeP, ("bridge", k, T0, T1), 0.0, p["rng"])
        w = bridge_noise(res["w"])
        if p.get("bridge_prior", "joint") == "joint":
            logl = bridge_logl(np.mean(shiftP), np.mean(slopeP), n_steps + 1) - 2 * np.log(v * np.sqrt(2 * Dr * p["dt"]))
            reward_logl = bridge_logl(np.mean(shiftP), np.mean(slopeP), n_steps + 1) - bridge_logl(0.0, 0.0, n_steps + 1)
        else:
            logl = np.mean(-(slopeP / res["std1"]) ** 2 - (shiftP / res["std2"]) ** 2) / 2 \
                + np.log(slope_factor) + np.log(shift_factor)
    else:
        w = p["rng"].get(("free", k, T0, T1), (n, n_steps), 0.0)
    X = X0.copy()
    hires = [X.copy()] if p.get("hires", False) else None
    force = p.get("force")
    for i in range(n_steps):
        c, s = np.cos(X[:, 2]), np.sin(X[:, 2])
        fx, fy = (0.0, 0.0) if force is None else force(X)
        X[:, 2] += np.sqrt(dt_) * w[:, i] * np.sqrt(2 * Dr)
        X[:, 0] += dt_ * (c * v + fx)
        X[:, 1] += dt_ * (s * v + fy)
        if hires is not None:
            hires.append(X.copy())
    streams = {}
    if hires is not None:
        streams["hires"] = np.stack(hires, -1)
    if p.get("record_bridge_parameters", False):
        B1 = np.cumsum(w, axis=1)
        B2 = np.cumsum(B1, axis=1)
        streams["bridge"] = np.stack([B1[:, -1:], B2[:, -1:]], axis=1)       # (n, 2, 1): slope, shift
    return dict(X1=X, logl=logl, reward_logl=reward_logl, streams=streams)


def log_obs_density(X, obs, sigma):
    """Gaussian observation density of the positions (up to a constant), -sum((X - Y)/sigma)^2.
    Unobserved components of Y are NaN (the orientation)."""
    return np.nansum(-((X - obs["Y"]) / sigma) ** 2, axis=1)


def simulate(n_obs, dt_obs, p, rng_seed, X0=(0.0, 0.0, 0.0), use_bridges=False):
    """Ground-truth trajectory of a single particle, observed every ``dt_obs``. Returns (T, x, y).
    With ``use_bridges`` the noise of every interval is a null bridge (shift = slope = 0)."""
    from .bridges import CorrelatedRNG

    q = dict(p, rng=CorrelatedRNG(rng_seed), use_bridges=use_bridges, target_shift=0.0, target_slope=0.0,
             hires=False)
    X = np.array([X0], float)
    T = np.arange(1, n_obs + 1) * dt_obs
    xs, ys, T_prev = [], [], 0.0
    for k, t in enumerate(T):
        q["rng"].reroll(0)
        X = propagate(T_prev, t, X, k, q)["X1"]
        T_prev = t
        xs.append(X[0, 0]); ys.append(X[0, 1])
    return T, np.array(xs), np.array(ys)


def make_observations(T, x, y, futures=0):
    """List of observation dicts; with ``futures`` > 0 each one also carries the next ``futures``
    observed positions (``future_x``, ``future_y``, starting with itself), and observations without
    enough futures are dropped."""
    obs = [dict(T=t, Y=np.array([xi, yi, np.nan])) for t, xi, yi in zip(T, x, y)]
    if futures:
        for k, o in enumerate(obs):
            o["future_x"] = [obs[min(k + f, len(obs) - 1)]["Y"][0] for f in range(futures + 1)]
            o["future_y"] = [obs[min(k + f, len(obs) - 1)]["Y"][1] for f in range(futures + 1)]
        obs = obs[:len(obs) - futures]
    return obs
