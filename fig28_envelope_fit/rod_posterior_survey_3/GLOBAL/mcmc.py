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

def mcmc(x_data, y_data, starting_parameters, n_samples=1000, alpha_mutations = 1e-2, thinning=100, n_particles=150, use_tqdm=True, max_runtime=None):
    
    n_parameters = len(starting_parameters)
    
    if type(alpha_mutations) is not list:
        alpha_mutations = [alpha_mutations] * n_parameters

    PARAMS_CHAIN = []
    
    LOGL = -np.inf # Current chain likelihood
    attempts = 0
    accepts = 0

    params = np.array(starting_parameters)
    
    ''' If max_runtime is set, we will run indefinitely until the wall time is reached. '''
    if max_runtime is not None:
        print(f"Running MCMC for {max_runtime} seconds")
        pbar = wall_time_generator(max_runtime)
    else:
        pbar = tqdm(range(n_samples * thinning), leave=False) if use_tqdm else range(n_samples * thinning)

    for s in pbar:

        # GIBBS blocks for all parameters
        for p in range(n_parameters):
            proposal = params.copy()
            proposal[p] += np.random.randn() * alpha_mutations[p]
            proposal_np = np.atleast_2d(proposal)
            LOGL_prop, _, _ = calc_logl_core(x_data, y_data, dt, PE, DT, DR, FORCE_STRENGTH, proposal_np, n_particles=n_particles)
            attempts += 1
            thresh = np.exp(LOGL_prop - LOGL)
            a = np.random.rand()
            if a < thresh:
                accepts += 1
                LOGL = LOGL_prop
                params = proposal.copy()
    
        if not s % thinning:
            for p in range(n_parameters):
                PARAMS_CHAIN.append(params.copy())
            if use_tqdm:
                pbar.set_description(f"Acceptance: {int(100*accepts/attempts)}%. Progress")

    print(f"Acceptance: {int(100*accepts/attempts)}%. Progress")
    return PARAMS_CHAIN