import numpy as np
from numba import njit, prange
from scipy.signal import correlate
import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from GLOBAL.ground_truths import *
from GLOBAL.torque import *
from GLOBAL.filters import *

import argparse

parser = argparse.ArgumentParser(description='Calculate a chunk of the posterior.')
parser.add_argument('alpha1_min', type=float, help='Minimum value of alpha1.')
parser.add_argument('alpha1_max', type=float, help='Maximum value of alpha1.')
parser.add_argument('alpha2_min', type=float, help='Minimum value of alpha2.')
parser.add_argument('alpha2_max', type=float, help='Maximum value of alpha2.')
parser.add_argument('resolution', type=int, help='Number of values in total per direction.')
parser.add_argument('chunks', type=int, help='Number of chunks per direction')
parser.add_argument('chunk_alpha1', type=int, help='Chunk number in alpha direction')
parser.add_argument('chunk_alpha2', type=int, help='Chunk number in k direction')

args = parser.parse_args()

with np.load("../../GLOBAL/data.npz") as data:
    x_data=data['x_data']
    y_data=data['y_data']


assert args.resolution % args.chunks == 0, 'Resolution must be divisible by chunks.'
chunk_size = args.resolution // args.chunks

alpha1_grid = np.linspace(args.alpha1_min, args.alpha1_max, args.resolution)
alpha2_grid = np.linspace(args.alpha2_min, args.alpha2_max, args.resolution)


# cut out the correct chunk
alpha1_grid = alpha1_grid[args.chunk_alpha1 * chunk_size:(args.chunk_alpha1 + 1) * chunk_size]
alpha2_grid = alpha2_grid[args.chunk_alpha2 * chunk_size:(args.chunk_alpha2 + 1) * chunk_size]

ALPHA1_GRID, ALPHA2_GRID = np.meshgrid(alpha1_grid, alpha2_grid)
shape_chunk = ALPHA1_GRID.shape
ALPHA1_GRID = ALPHA1_GRID.flatten()
ALPHA2_GRID = ALPHA2_GRID.flatten()

params = np.stack((ALPHA1_GRID, ALPHA2_GRID), axis=1)

print("Type of x_data: ", type(x_data))
print("Type of y_data: ", type(y_data))
print("Type of dt: ", type(dt))
print("Type of PE: ", type(PE))
print("Type of DT: ", type(DT))
print("Type of DR: ", type(DR))
print("Type of FORCE_STRENGTH: ", type(FORCE_STRENGTH))
print("Type of param_: ", type(params))

logl, _, _ = calc_logl_core(x_data, y_data, dt, PE, DT, DR, FORCE_STRENGTH, params, n_particles=1000)
logl = logl.reshape(shape_chunk)
np.savez(f'chunk_pf_{args.chunk_alpha1}_{args.chunk_alpha2}.npz', logl=logl, alpha1_grid=alpha1_grid, alpha2_grid=alpha2_grid)

logl, _, _ = calc_logl_core_kalman(x_data, y_data, dt, PE, DT, DR, FORCE_STRENGTH, params)
logl = logl.reshape(shape_chunk)
np.savez(f'chunk_kf_{args.chunk_alpha1}_{args.chunk_alpha2}.npz', logl=logl, alpha1_grid=alpha1_grid, alpha2_grid=alpha2_grid)