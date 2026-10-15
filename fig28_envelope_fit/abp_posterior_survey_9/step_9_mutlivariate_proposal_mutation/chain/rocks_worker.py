import numpy as np
from numba import njit, prange
from scipy.signal import correlate
import os
import sys
from uuid import uuid4

import argparse

parser = argparse.ArgumentParser(description='Calculate a chain')
parser.add_argument('mutation', type=float, help='mutation rate')
parser.add_argument('particle_count', type=int, help='particle count')
args = parser.parse_args()

from scipy.ndimage import gaussian_filter

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from GLOBAL.ground_truths import *
from GLOBAL.torque import *
from GLOBAL.filters import *
from GLOBAL.mcmc import *

with np.load("../../GLOBAL/data.npz") as data:
    x_data=data['x_data']
    y_data=data['y_data']

with np.load("../proposal_distribution.npz") as data:
    proposal_cov = data['cov'] * args.mutation
    starting_parameters = data['mean']

print("cov", proposal_cov)
mutation_rate = args.mutation
print("Mutation rate:", mutation_rate)
CHAINS, acceptance = mcmc(x_data, y_data,
                        starting_parameters=starting_parameters,
                        n_particles = args.particle_count,
                        gibbs=False,
                        proposal_cov=proposal_cov,
                        thinning=1,
                        use_tqdm=False,
                        #n_samples=100)
                        max_runtime=60*60)

fname = f"chain_{uuid4()}.npz"
np.savez(fname, CHAINS=CHAINS, acceptance=acceptance, mutation_rate=mutation_rate, particle_count=args.particle_count)
