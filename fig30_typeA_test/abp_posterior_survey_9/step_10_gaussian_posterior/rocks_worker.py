import numpy as np
from scipy.stats import multivariate_normal
import os
import sys
from uuid import uuid4
import time

print("I'm a worker!", flush=True)

start_time = time.time()

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from GLOBAL.ground_truths import *
from GLOBAL.torque import *
from GLOBAL.filters import *

max_angle = 1000000 * np.pi / 180

with np.load("../GLOBAL/data.npz") as data:
    x_data=data['x_data']
    y_data=data['y_data']

search_size = 2
initial_radius = 3

ndims = 6

sample_count = 256
d0 = 0
d1 = 1
centers = np.pad(ALPHAS, (0, ndims-ALPHAS.shape[0]), 'constant', constant_values=0.)
mean = centers
cov = np.eye(ndims) * initial_radius
COVS = [cov]
ALL_COORDS = np.zeros((0, ndims))
ALL_TARGET_DENSITIES = np.zeros(0)
ALL_LIKELIHOODS = np.zeros(0)

progression = range(30)

print("Starting main loop", flush=True)

for prog in (progression):
    print(prog, flush=True)
    print(mean, flush=True)
    distribution = multivariate_normal(mean=mean, cov=search_size*cov)
    coords = distribution.rvs(sample_count)
    likelihood = distribution.pdf(coords)
    logl, _, _ = calc_logl_core(x_data, y_data, dt,
                                   PE, DT, DR, FORCE_STRENGTH,
                                   coords, torque_limit=max_angle,
                                   n_particles=2000
                                   )
    logl = logl - np.max(logl)
    target_density = np.exp(logl)
    target_density = target_density / np.sum(target_density)

    ALL_COORDS = np.concatenate((ALL_COORDS, coords))
    ALL_TARGET_DENSITIES = np.concatenate((ALL_TARGET_DENSITIES, target_density))
    ALL_LIKELIHOODS = np.concatenate((ALL_LIKELIHOODS, likelihood))

    mean = np.average(ALL_COORDS, axis=0, weights=ALL_TARGET_DENSITIES/ALL_LIKELIHOODS)
    cov = np.cov(ALL_COORDS.T, aweights=ALL_TARGET_DENSITIES/ALL_LIKELIHOODS)

    COVS.append(cov)

print("End of main loop", flush=True)
end_time = time.time()

fname = f"distro_{uuid4()}.npz"
print("Saving to", fname, flush=True)
np.savez(fname, COVS=COVS, ALL_COORDS=ALL_COORDS, ALL_TARGET_DENSITIES=ALL_TARGET_DENSITIES, ALL_LIKELIHOODS=ALL_LIKELIHOODS, start_time=start_time, end_time=end_time)
print("Done. Goodbye!", flush=True)