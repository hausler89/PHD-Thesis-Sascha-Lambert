import numpy as np
from numba import njit, prange
from scipy.signal import correlate
import os
import sys
from uuid import uuid4

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

ALPHA1_CHAIN, ALPHA2_CHAIN = mcmc(x_data, y_data,
                                  alpha_mutations=[1e-0, 1e-0],
                                  n_samples=0, superstep=40,
                                  n_particles = 100,
                                  use_tqdm=False, max_runtime=2*60*60)

fname = f"chain_{uuid4()}.npz"
np.savez(fname, ALPHA1_CHAIN=ALPHA1_CHAIN, ALPHA2_CHAIN=ALPHA2_CHAIN)
