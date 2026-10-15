import numpy as np
from numba import jit, njit, prange, vectorize

@vectorize
def _bspline3_kernel(x):
    res = 0.0
    if x < 0:
        return 0
    elif x < 1:
        return 0.5 * x**2
    elif x < 2:
        return 0.5 * (-2*x**2 + 6*x - 3)
    elif x < 3:
        return 0.5 * (x-3)**2
    else:
        return 0

@njit(inline='always')
def bspline3(X, params):
    N = params.shape[0]
    window_size = N - 3
    X = window_size * X

    res = np.zeros_like(X)
    for i in range(N):
        res += params[i] * _bspline3_kernel(X - i + 2.5)

    return res

@njit(inline='always')
def TORQUE(phi, params):
    n_params = params.shape[0]
    torque = 0. * phi
    for i in range(0, n_params):
        torque += params[i] * np.sin(-phi * (i+1))
    return torque

#@njit(inline='always')
#def TORQUE_DERIV(phi, params):
#    n_params = params.shape[0]
#    torque = 0. * phi
#    for i in range(0, n_params):
#        torque += -i * params[i] * np.cos(-phi * (i+1))
#    return torque

@njit(inline='always')
def TORQUE_BSPLINE(phi, params):
    zero = np.array([0.])
    first = np.array((params[0],))
    last = np.array((params[-1],))
    #params = np.concatenate((params))
    n_params = params.shape[0]
    phi = np.mod(phi, 2*np.pi) / (2*np.pi)
    params_fl = -np.flip(params)
    params = np.concatenate((-first, zero, params, zero, params_fl, zero, first))
    torque = bspline3(phi, params)
    return torque