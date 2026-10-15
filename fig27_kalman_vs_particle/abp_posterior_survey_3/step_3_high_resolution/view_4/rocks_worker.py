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
parser.add_argument('alpha_min', type=float, help='Minimum value of alpha.')
parser.add_argument('alpha_max', type=float, help='Maximum value of alpha.')
parser.add_argument('k_min', type=float, help='Minimum value of k.')
parser.add_argument('k_max', type=float, help='Maximum value of k.')
parser.add_argument('resolution', type=int, help='Number of values in total per direction.')
parser.add_argument('chunks', type=int, help='Number of chunks per direction')
parser.add_argument('chunk_alpha', type=int, help='Chunk number in alpha direction')
parser.add_argument('chunk_k', type=int, help='Chunk number in k direction')

args = parser.parse_args()

with np.load("../../GLOBAL/data.npz") as data:
    x_data=data['x_data']
    y_data=data['y_data']


assert args.resolution % args.chunks == 0, 'Resolution must be divisible by chunks.'
chunk_size = args.resolution // args.chunks

alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.resolution)
k_grid = np.linspace(args.k_min, args.k_max, args.resolution)


# cut out the correct chunk
alpha_grid = alpha_grid[args.chunk_alpha * chunk_size:(args.chunk_alpha + 1) * chunk_size]
k_grid = k_grid[args.chunk_k * chunk_size:(args.chunk_k + 1) * chunk_size]

ALPHA_GRID, K_GRID = np.meshgrid(alpha_grid, k_grid)
shape_chunk = ALPHA_GRID.shape
ALPHA_GRID = ALPHA_GRID.flatten()
K_GRID = K_GRID.flatten()

params = np.stack((ALPHA_GRID, K_GRID), axis=1)

print("Type of x_data: ", type(x_data))
print("Type of y_data: ", type(y_data))
print("Type of dt: ", type(dt))
print("Type of PE: ", type(PE))
print("Type of DT: ", type(DT))
print("Type of DR: ", type(DR))
print("Type of FORCE_STRENGTH: ", type(FORCE_STRENGTH))
print("Type of param_: ", type(params))

logl, _, _ = calc_logl_core(x_data, y_data, dt, PE, DT, DR, FORCE_STRENGTH, params, n_particles=2000)
logl = logl.reshape(shape_chunk)
np.savez(f'chunk_pf_{args.chunk_alpha}_{args.chunk_k}.npz', logl=logl, alpha_grid=alpha_grid, k_grid=k_grid)

logl, _, _ = calc_logl_core_kalman(x_data, y_data, dt, PE, DT, DR, FORCE_STRENGTH, params)
logl = logl.reshape(shape_chunk)
np.savez(f'chunk_kf_{args.chunk_alpha}_{args.chunk_k}.npz', logl=logl, alpha_grid=alpha_grid, k_grid=k_grid)
