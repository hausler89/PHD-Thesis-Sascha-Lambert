import numpy as np

_EPS_ = 1e-6 # small number for regularization

DT = 1 # Translational diffusion

PE = 100. # Peclet number

ALPHAS = np.array([10., 10.])

FORCE_STRENGTH = 1. # Prefactor of the force
Q2 = 1.54308184422 # Geometry parameter

X0 = -2
Y0 = 0
spread_deg = 90
n_trajectories = 100
PHI0 = np.linspace(-spread_deg/2, spread_deg/2, n_trajectories) * np.pi / 180

n_steps = 400
dt = 0.0001 # Time step

aspect_ratio = 1.5
n_satellites = 2

def ratio_diffusion_rotational(p):
    rot_lueders = 1/(p * p) * (np.log(p) - 0.3512 + 0.7804 / p - 0.09801 / (p * p))
    para_lueders = 0.5 * (np.log(p) - 0.1404 + 1.034 / p - 0.228 / (p * p))
    return rot_lueders/para_lueders

def ratio_diffusion_orthogonal(p):
    ortho_lueders = 0.5 * (np.log(p) + 0.8369 + 0.5551 / p - 0.06066 / (p * p))
    para_lueders = np.log(p) - 0.1404 + 1.034 / p - 0.228 / (p * p)
    return ortho_lueders/para_lueders

ratio_diff_rot = ratio_diffusion_rotational(aspect_ratio)
ratio_diff_orth = ratio_diffusion_orthogonal(aspect_ratio)

DR = DT * ratio_diff_rot