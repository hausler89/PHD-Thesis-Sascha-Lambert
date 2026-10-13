"""Metropolis-Hastings and its pseudo-marginal variant (section 2.3)."""
import numpy as np


def metropolis_hastings(logp, x0, n_steps, step, rng):
    """Random-walk Metropolis-Hastings with a Gaussian proposal.

    Parameters
    ----------
    logp : callable x -> log of the (unnormalised) target density
    x0 : initial point, shape (d,)
    n_steps : chain length
    step : proposal std (scalar, vector, or (d,d) covariance matrix)
    rng : numpy Generator

    Returns
    -------
    chain : (n_steps, d) array, acceptance rate
    """
    x = np.atleast_1d(np.asarray(x0, float))
    d = x.size
    step = np.asarray(step, float)
    chol = np.linalg.cholesky(step) if step.ndim == 2 else np.diag(np.broadcast_to(step, d))
    lp = logp(x)
    chain = np.empty((n_steps, d))
    accepted = 0
    for i in range(n_steps):
        prop = x + chol @ rng.standard_normal(d)
        lp_prop = logp(prop)
        # symmetric proposal -> the proposal ratio in eq. (mh_acceptance) cancels
        if np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop
            accepted += 1
        chain[i] = x
    return chain, accepted / n_steps


def pseudo_marginal_mh(loglik_estimate, logprior, x0, n_steps, step, rng):
    """Pseudo-marginal MH: ``loglik_estimate(x, rng)`` is a *noisy but unbiased* (on the
    likelihood scale) estimator, e.g. from a particle filter. The estimate at the current
    state is kept and only re-drawn for the proposal -- this is what makes the scheme exact.
    """
    x = np.atleast_1d(np.asarray(x0, float))
    d = x.size
    step = np.asarray(step, float)
    chol = np.linalg.cholesky(step) if step.ndim == 2 else np.diag(np.broadcast_to(step, d))
    ll = loglik_estimate(x, rng)
    lpr = logprior(x)
    chain = np.empty((n_steps, d))
    accepted = 0
    for i in range(n_steps):
        prop = x + chol @ rng.standard_normal(d)
        lpr_p = logprior(prop)
        if np.isfinite(lpr_p):
            ll_p = loglik_estimate(prop, rng)
            if np.log(rng.random()) < (ll_p + lpr_p) - (ll + lpr):
                x, ll, lpr = prop, ll_p, lpr_p
                accepted += 1
        chain[i] = x
    return chain, accepted / n_steps


def componentwise_metropolis(logp, x0, n_samples, steps, thin, rng):
    """Metropolis-within-Gibbs: in every iteration each coordinate is updated in turn with a
    Gaussian random-walk proposal of width ``steps[i]``. One sample is stored every ``thin``
    iterations (starting with the first), giving ``n_samples`` samples.

    Proposals with an undefined (NaN) log density are rejected.
    """
    x = np.array(x0, float)
    lp = -np.inf
    out = np.empty((n_samples, x.size))
    accepted = 0
    for it in range(n_samples * thin):
        for i in range(x.size):
            prop = x.copy()
            prop[i] += rng.standard_normal() * steps[i]
            lp_prop = logp(prop)
            if rng.random() < np.exp(lp_prop - lp):       # NaN compares False -> rejected
                x, lp = prop, lp_prop
                accepted += 1
        if it % thin == 0:
            out[it // thin] = x
    return out, accepted / (n_samples * thin * x.size)
