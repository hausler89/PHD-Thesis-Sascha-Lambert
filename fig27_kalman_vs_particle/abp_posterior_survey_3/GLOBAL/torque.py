import numpy as np
from numba import njit, prange

@njit(inline='always')
def TORQUE(phi, params):
    alpha = params[0]
    k = params[1]
    return alpha * np.sin(-phi / k)

@njit(inline='always')
def TORQUE_DERIV(phi, params):
    alpha = params[0]
    k = params[1]
    return -alpha * np.cos(-phi / k) / k