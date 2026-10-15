import numpy as np
from numba import njit, prange

from .ground_truths import *
from .torque import *

def StructuredResample(w, LUT_accelerator_size = 100):
    
    '''
    Implementing a structured resample.
    We take weights w
    
        w = [w0 w1 ... wN-1]
        
    corresponding to N indices 0..N-1
    
    We then draw N resampled indices, distributed
    according to the provided weights w. The
    resample is structured, meaning that all
    indices are sampled in equidistant steps from
    the CDF.
    
    We start with a random sample <off> and then sample
    in steps from there.
    
    We use a LUT to accelerate finding the appropriate
    region of the CDF faster. The lookup itself is
    implemented in Numba.
    '''
    
    off = np.random.rand() * 1          # First sample offset
    p = w / np.sum(w)                   # Normalized PDF
    cdf = np.cumsum(p)                  # CDF

    idx_rsmpl = np.empty_like(cdf, dtype=int)       # Numba can't allocate memory itself
    LUT = np.empty(LUT_accelerator_size, dtype=int) # """
    
    return __structured_resample_core(cdf, idx_rsmpl, off, LUT)



@njit(inline='always')
def __structured_resample_core(cdf, idx_rsmpl, off, LUT):
    
    ''' Prebuild a LUT for finding the correct position in the CDF faster '''
    nLUT = LUT.shape[0]
    for k in prange(nLUT):
        target_prob = k / nLUT
        idx = 0
        while cdf[idx] < target_prob:
            idx += 1
        LUT[k] = idx
    
    ''' Sample from the CDF '''
    for k in prange(len(cdf)):
        target_prob = (off + k / len(cdf)) % 1
        
        ''' Look up the closest starting (ascending) starting point '''
        idx = LUT[int(target_prob*nLUT)]
        
        ''' Ascend the CDF until we're happy '''
        while cdf[idx] < target_prob:
            idx += 1
        idx_rsmpl[k] = idx

    return idx_rsmpl

# Calculate the log-likelihood using a Kalman filter
np.set_printoptions(precision=10, suppress=False)
@njit(parallel=True, fastmath=False)
def calc_logl_core_kalman(x, y, dt, Pe, Dt, Dr, force_strength, torque_params, torque_limit=np.inf):
    n_paramsets = torque_params.shape[0]

    LOGLS = np.zeros(n_paramsets) # Log-likelihoods for each parameter set
    n_datasets = x.shape[0]
    n_steps = x.shape[1]

    phi_means = 0
    phi_stds = 0

    for param_idx in prange(n_paramsets):
        
        params_ = torque_params[param_idx, :]

        logl = 0

        for data_idx in range(n_datasets):
            
            x_ = x[data_idx, :]
            y_ = y[data_idx, :]

            phi_mean = PHI0[data_idx]
            phi_std = 5.0 * np.pi / 180

            for t in range(0, n_steps-1):
                
                F = 0
                if x_[t] > -2**(1/6):
                    F = -24 * force_strength * (x_[t]**6 - 2) / x_[t]**13
                
                x_projection = np.cos(phi_mean) * Pe * dt
                y_projection = np.sin(phi_mean) * Pe * dt
                projection_normalization = np.sqrt(x_projection**2 + y_projection**2)

                x_projected = x_[t] + x_projection
                y_projected = y_[t] + y_projection

                dx = x_projected - (x_[t+1] - F * dt)
                dy = y_projected - y_[t+1]

                offset2 = dx**2 + dy**2
                diffusion_var = 2 * Dt * dt
                uncertainty_var = (phi_std * Pe * dt)**2
                logl += -0.5 * offset2 / (diffusion_var + uncertainty_var)

                offset = (-dy * x_projection / projection_normalization) + (dx * y_projection / projection_normalization)
                phi_offset_mean = offset / (Pe * dt)
                phi_offset_std = np.sqrt(2 * Dt * dt) / (Pe * dt)
                
                phi_mean = (phi_std**2 * (phi_mean + phi_offset_mean) + phi_offset_std**2 * phi_mean) / (phi_std**2 + phi_offset_std**2)
                phi_std = np.sqrt(1 / (1/phi_std**2 + 1/phi_offset_std**2) + 2 * Dr * dt)
                torque = Q2 * TORQUE_FILTER(phi_mean, params_) * F
                d_torque = Q2 * TORQUE_DERIV_FILTER(phi_mean, params_) * F

                if (np.abs(torque * dt)) > torque_limit:
                    logl = -np.inf
                    
                phi_mean += torque * dt
                phi_std *= np.exp(-d_torque * dt)

            #phi_means.append(phi_mean)
            #phi_stds.append(phi_std)

        
        LOGLS[param_idx] = logl
        #break
    
    return LOGLS, phi_means, phi_stds



# Calculate the log-likelihood using a particle filter
np.set_printoptions(precision=10, suppress=False)
@njit(parallel=False, fastmath=False)
def calc_logl_core(x, y, dt, Pe, Dt, Dr, force_strength, torque_params, n_particles=100, torque_limit=np.inf, track_particles=False):
    n_paramsets = torque_params.shape[0]

    LOGLS = np.zeros(n_paramsets) # Log-likelihoods for each parameter set
    n_datasets = x.shape[0]
    n_steps = x.shape[1]

    #phi_means = list()
    #phi_stds = list()

    PHI = np.zeros((n_paramsets, n_datasets, n_particles, n_steps-1), dtype=np.float64)

    for param_idx in range(n_paramsets):
        
        params_ = torque_params[param_idx, :]

        if np.any(np.abs(params_) > 200):
            LOGLS[param_idx] = -np.inf
            continue

        logl = 0

        for data_idx in range(n_datasets):
            
            x_ = x[data_idx, :]
            y_ = y[data_idx, :]

            phi = PHI0[data_idx] + np.random.randn(n_particles) * 0.1 * np.pi / 180

            #phi_mean = list()
            #phi_std = list()
            PHI[param_idx, data_idx, :, 0] = phi
            for t in range(0, n_steps-1):

                F = 0
                if x_[t] > -2**(1/6):
                    F = -24 * force_strength * (x_[t]**6 - 2) / x_[t]**13
                
                x_projection = np.cos(phi) * Pe * dt
                y_projection = np.sin(phi) * Pe * dt

                x_projected = x_[t] + x_projection
                y_projected = y_[t] + y_projection

                dx = x_projected - (x_[t+1] - F * dt)
                dy = y_projected - y_[t+1]

                offset2 = dx**2 + dy**2
                diffusion_var = 2 * Dt * dt

                particle_logls = -0.5 * offset2 / diffusion_var
                
                idx_max_particle_logl = np.argmax(particle_logls)
                max_particle_logl = particle_logls[idx_max_particle_logl]
                logl_rest = np.delete(particle_logls, [idx_max_particle_logl])
                logl += max_particle_logl + np.log(1 + np.sum(np.exp(logl_rest - max_particle_logl)))

                particle_logls -= particle_logls.max()
                particle_weights = np.exp(particle_logls)
                particle_weights /= particle_weights.sum()

                phi_mean = np.sum(phi * particle_weights)
                phi_var = np.sum((phi - phi_mean)**2 * particle_weights)

                #phi = np.random.normal(phi_mean, np.sqrt(phi_var), n_particles)

                for n in range(n_particles):
                    phi[n] = np.random.randn() * np.sqrt(phi_var) + phi_mean

                off = np.random.rand() * 1
                cdf = np.cumsum(particle_weights)
                resampled_indices = np.zeros(n_particles, dtype=np.uint32)
                LUT = np.zeros(100, dtype=np.uint32)

                __structured_resample_core(cdf, resampled_indices, off, LUT)

                phi = phi[resampled_indices]

                for k in prange(n_particles):
                    torque = Q2 * TORQUE_FILTER(phi[k], params_) * F
                    phi[k] += torque * dt + np.sqrt(2 * Dr * dt) * np.random.normal()
                
                PHI[param_idx, data_idx, :, t] = phi

                #phi_mean.append(np.mean(phi))
                #phi_std.append(np.std(phi))

            #phi_means.append(phi_mean)
            #phi_stds.append(phi_std)

        LOGLS[param_idx] = logl
        #break

    
    return LOGLS, PHI, [0.]