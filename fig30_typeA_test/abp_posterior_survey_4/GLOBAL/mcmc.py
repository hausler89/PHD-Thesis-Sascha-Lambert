import numpy as np
from numba import njit, prange
from tqdm import tqdm

from .ground_truths import *
from .torque import *
from .filters import *

import time

def wall_time_generator(duration):
    start_time = None
    while True:
        if start_time is None:
            start_time = time.time()
            counter = 0
        
        elapsed_time = time.time() - start_time
        if elapsed_time < duration:
            yield counter
            counter += 1
        else:
            return

def mcmc(x_data, y_data, n_samples=1000, alpha_mutations = [1e-1, 1e-1], superstep=100, n_particles=150, use_tqdm=True, max_runtime=None):
    
    ALPHA1_CHAIN = []
    ALPHA2_CHAIN = []

    alpha1 = ALPHA1
    alpha2 = ALPHA2
    
    LOGL = -np.inf # Current chain likelihood
    attempts = 0
    accepts = 0
    pbar = tqdm(range(n_samples * superstep), leave=False) if use_tqdm else range(n_samples * superstep)

    ''' If max_runtime is set, we will run indefinitely until the wall time is reached. '''
    if max_runtime is not None:
        print(f"Running MCMC for {max_runtime} seconds")
        pbar = wall_time_generator(max_runtime)

    for s in pbar:
        # GIBBS ALPHA1
        alpha1_prop = alpha1 + np.random.randn() * alpha_mutations[0]
        params = np.atleast_2d([alpha1_prop, alpha2])
        LOGL_prop, _, _ = calc_logl_core(x_data, y_data, dt, PE, DT, DR, FORCE_STRENGTH, params, n_particles=n_particles)
        attempts += 1
        thresh = np.exp(LOGL_prop - LOGL)
        a = np.random.rand()
        if a < thresh:
            accepts += 1
            LOGL = LOGL_prop
            alpha1 = alpha1_prop
    
        # GIBBS ALPHA2
        alpha2_prop = alpha2 + np.random.randn() * alpha_mutations[1]
        params = np.atleast_2d([alpha1, alpha2_prop])
        LOGL_prop, _, _ = calc_logl_core(x_data, y_data, dt, PE, DT, DR, FORCE_STRENGTH, params, n_particles=n_particles)
        attempts += 1
        thresh = np.exp(LOGL_prop - LOGL)
        a = np.random.rand()
        if a < thresh:
            accepts += 1
            LOGL = LOGL_prop
            alpha2 = alpha2_prop
    
        if not s % superstep:
            ALPHA1_CHAIN.append(alpha1)
            ALPHA2_CHAIN.append(alpha2)
            if use_tqdm:
                pbar.set_description(f"Acceptance: {int(100*accepts/attempts)}%. Progress")

    print(f"Acceptance: {int(100*accepts/attempts)}%. Progress")
    return ALPHA1_CHAIN, ALPHA2_CHAIN