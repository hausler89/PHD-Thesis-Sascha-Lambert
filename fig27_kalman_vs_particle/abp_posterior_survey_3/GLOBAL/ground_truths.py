import numpy as np

_EPS_ = 1e-6 # small number for regularization

DT = 1e-3 # Translational diffusion
DR = 1e-3 # Rotational diffusion
PE = 1. # Peclet number

ALPHA = 10. # Mode strength
K = 1. # Mode wavenumber

FORCE_STRENGTH = 1. # Prefactor of the force
Q2 = 1.54308184422 # Geometry parameter

X0 = -1.5
Y0 = 0
spread_deg = 90
n_trajectories = 100
PHI0 = np.linspace(-spread_deg/2, spread_deg/2, n_trajectories) * np.pi / 180

n_steps = 250
dt = 0.01 # Time step