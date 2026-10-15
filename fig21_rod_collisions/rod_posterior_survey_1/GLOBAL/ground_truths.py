import numpy as np

from .loader import *

_EPS_ = 1e-6 # small number for regularization

DT = 1 # Translational diffusion
DR = 1.54308184422 # Rotational diffusion
PE = 100. # Peclet number

ALPHAS = np.array([100., 0.])

FORCE_STRENGTH = 4. # Prefactor of the force
Q2 = 1.54308184422 # Geometry parameter

X0 = -2**(1/6) - 0.5
Y0 = 0
spread_deg = 90
n_trajectories = 10
PHI0 = np.linspace(-spread_deg/2, spread_deg/2, n_trajectories) * np.pi / 180

n_steps = 300
#dt = 0.001 # Time step

sim = ABPsim('satellites/', )
satellites = sim.satellites_xy
satellites_x = satellites[::2]
aspect_ratio = 1 + max(satellites_x) - min(satellites_x)
dt = sim.state_dump_time_step
print("Aspect ratio: ", aspect_ratio)

''' C Code of rot_ratio

double ratio_diffusion_rotational(double p)
{
    double rot_lueders = 1/(p * p) * (log(p) - 0.3512 + 0.7804 / p - 0.09801 / (p * p));
    double para_lueders = 0.5 * (log(p) - 0.1404 + 1.034 / p - 0.228 / (p * p));
    return rot_lueders/para_lueders;
}

'''

def ratio_diffusion_rotational(p):
    rot_lueders = 1/(p * p) * (np.log(p) - 0.3512 + 0.7804 / p - 0.09801 / (p * p))
    para_lueders = 0.5 * (np.log(p) - 0.1404 + 1.034 / p - 0.228 / (p * p))
    return rot_lueders/para_lueders

''' C Code of orth_ratio

double ratio_diffusion_orthogonal(double p, size_t n)
{

    double ortho_lueders = n * 0.5 * (log(p) + 0.8369 + 0.5551 / p - 0.06066 / (p * p));
    double para_lueders = log(p) - 0.1404 + 1.034 / p - 0.228 / (p * p);
    return ortho_lueders/para_lueders;
}

'''

def ratio_diffusion_orthogonal(p):
    ortho_lueders = 0.5 * (np.log(p) + 0.8369 + 0.5551 / p - 0.06066 / (p * p))
    para_lueders = np.log(p) - 0.1404 + 1.034 / p - 0.228 / (p * p)
    return ortho_lueders/para_lueders

ratio_diff_rot = ratio_diffusion_rotational(aspect_ratio)
ratio_diff_orth = ratio_diffusion_orthogonal(aspect_ratio)

print("Ratio of rotational diffusion: ", ratio_diff_rot)
print("Ratio of orthogonal diffusion: ", ratio_diff_orth)