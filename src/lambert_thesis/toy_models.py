"""Toy models of chapter 2: coin flips and the lost submarine (grid-based Bayes)."""
import numpy as np
from scipy.special import erf


def normalize(density, *axes):
    """Normalise a density given on a (possibly multi-dimensional) grid.

    ``axes`` are the 1D coordinate arrays (in the order of the array dimensions).
    """
    z = density
    for ax in reversed(range(len(axes))):
        z = np.trapezoid(z, axes[ax], axis=ax)
    return density / z


def bernoulli(x, mu):
    """Bern(x | mu) = mu^x (1-mu)^(1-x) -- likelihood of one coin flip (1 = heads)."""
    return mu**x * (1 - mu) ** (1 - x)


def coin_posterior(flips, mu):
    """Posterior of the coin bias on the grid ``mu`` for a flat prior and a sequence of flips."""
    post = np.ones_like(mu)
    for x in flips:
        post = post * bernoulli(x, mu)
    return normalize(post, mu)


def bubble_likelihood_1d(d, theta, length=10.0):
    """Eq. (2.x): a bubble surfaces uniformly above the vessel -> box of width ``length`` around theta."""
    return (np.abs(d - theta) <= length / 2) / length


def bubble_likelihood_2d(d, T, theta, drift=1.0, diffusivity=0.1, length=10.0):
    """Likelihood of a bubble observed at ``d`` for a vessel at lateral position ``theta`` and depth ``T``.

    The bubble starts uniformly along the vessel, then drifts (``drift * T``) and diffuses
    (width ``diffusivity * T``) while rising: a box convolved with a Gaussian -> difference of erf's.
    Unnormalised (only proportional to the density).
    """
    h = length / 2
    s = T * diffusivity
    return erf((theta + h - d + drift * T) / s) - erf((theta - h - d + drift * T) / s)
