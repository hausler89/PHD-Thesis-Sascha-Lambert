"""Bootstrap particle filter for SDEs, as used in chapter 5.

Algorithm following S. Wiqvist, A. Golightly, A. T. McLean, U. Picchini, Efficient inference for stochastic
differential equation mixed-effects models using correlated particle pseudo-marginal algorithms, Comput. Stat. Data
Anal. 157, 107151 (2021), https://doi.org/10.1016/j.csda.2020.107151 (own implementation).

The filter propagates a swarm of particles from observation to observation, weights them with the
observation density, accumulates the log-likelihood estimate log sum_i w_i per observation and resamples
(structured resampling). Hooks allow a policy (e.g. a PPO agent) to choose the bridge for the next step
and to collect rewards.
"""
import numpy as np


def structured_resample(w, rng):
    """Structured (systematic) resampling: N equidistant probes of the CDF of the weights ``w``,
    starting at a random offset."""
    cdf = np.cumsum(w / np.sum(w))
    cdf[-1] = 1.0
    probes = (rng.random() + np.arange(len(w)) / len(w)) % 1
    return np.minimum(np.searchsorted(cdf, probes, side="left"), len(w) - 1)


def logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m))) if np.isfinite(m) else m


def particle_filter(log_obs_density, propagate, X0, observations, rng, before_step=None, after_step=None,
                    keep_history=False, add_propagator_logl=False):
    """Run the particle filter.

    Parameters
    ----------
    log_obs_density(X, obs) -> (n,) log weights of the particle states ``X`` for observation ``obs``
    propagate(T0, T1, X, k) -> dict with ``X1`` (new states) and optionally ``streams`` (per-particle data
        of this step, e.g. high-resolution paths or bridge parameters, first axis = particle)
    X0 : (n, d) initial particle states (the prior on the state)
    observations : list of dicts with at least the time ``T``
    before_step(k, X, obs) : optional hook called before propagating to observation k
    after_step(k, logsumw, prop) -> bool : optional hook after weighting; returning True stops the filter
        (the remaining likelihood is then -inf)
    keep_history : store states and resampling indices to reconstruct one posterior trajectory
    add_propagator_logl : add ``prop["logl"]`` (e.g. the prior density of the chosen bridge) to the estimate

    Returns dict with ``logl`` and, with ``keep_history``, ``trajectory`` (d, n_obs) and ``streams``
    (each stream concatenated along its last axis), plus ``diversity`` (unique fraction after resampling).
    """
    X = X0.copy()
    T_prev = 0.0
    n = X.shape[0]
    logl = 0.0
    history, diversity = [], []
    for k, obs in enumerate(observations):
        if before_step is not None:
            before_step(k, X, obs)
        prop = propagate(T_prev, obs["T"], X, k)
        X, T_prev = prop["X1"], obs["T"]
        logw = log_obs_density(X, obs)
        lsw = logsumexp(logw)
        logl += lsw + (prop.get("logl", 0.0) if add_propagator_logl else 0.0)
        idx = structured_resample(np.exp(logw - np.max(logw)), rng)
        diversity.append(len(np.unique(idx)) / n)
        if keep_history:
            history.append((X, idx, prop.get("streams", {})))
        X = X[idx]
        if after_step is not None and after_step(k, lsw, prop):
            logl = -np.inf
            break
    out = dict(logl=logl, diversity=np.array(diversity))
    if keep_history and history:
        j = rng.integers(n)
        traj, streams = [], {key: [] for key in history[0][2]}
        for Xk, idx, st in reversed(history):
            j = idx[j]
            traj.append(Xk[j])
            for key in streams:
                streams[key].append(st[key][j])
        out["trajectory"] = np.stack(traj[::-1], axis=-1)
        out["streams"] = {key: np.concatenate([np.atleast_1d(a) for a in v[::-1]], axis=-1) for key, v in streams.items()}
    return out
