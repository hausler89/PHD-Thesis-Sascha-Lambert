import numpy as np
from numba import njit, prange

@njit(inline='always')
def TORQUE(phi, params):
    alpha1 = params[0]
    alpha2 = params[1]
    return alpha1 * np.sin(-phi * 1) + alpha2 * np.sin(-phi * 2)

@njit(inline='always')
def TORQUE_DERIV(phi, params):
    alpha1 = params[0]
    alpha2 = params[1]
    return -alpha1 * np.cos(-phi * 1) - 2 * alpha2 * np.cos(-phi * 2)