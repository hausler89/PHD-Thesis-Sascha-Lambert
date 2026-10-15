import numpy as np
from numba import njit, prange
import torch
from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.special import expit
import matplotlib.pyplot as plt

#####################################################################################

class HilbertSorter():
    
    ''' 
    Sort nD particles by using a Hilbert Curve.
    
    We prepare a Lookup Table of indices that order
    bins of nD space according to Hilbert indices.

    Points passed to us will be centered, scaled and
    squashed to a [0,1]^d hypercube. The transformed
    points are then getting binned and assigned an
    index from the LUT. Finally, we sort the particles
    by their respective index and return the result.

    Note that the LUT has finite precision and therefore
    particles can get the same index, if they fall into
    the same bin. The order in which particles from the
    same bin get sorted in is effectively random and
    depends on the initial ordering, as well as the
    Numpy sorting algorithm.
    '''

    def __init__(self, depth=3, dims=2):
        
        '''
        Prepare for receiving <dims>-dimensional points.
        Use <depth> recursions of the Hilbert curve to
        generate the LUT. This will lead to
        
            bins = 2 ^ depth
        
        bins per dimension.
        '''
        
        self.depth = depth
        self.dims  = dims
        
        self.HilbertCurve = HilbertCurve(depth, dims) # Index generator
        self.bins         = 2 ** depth                # Number of bins per dimension
        
        i  = np.arange(self.bins)         # Index per coordinate
        coords = np.meshgrid(*[i]*dims)   # Arrays of coordinate grids
        
        pts = np.stack([c.flatten() for c in coords], axis=-1) # Running list of cell centroids
        
        hilbert_idx = self.HilbertCurve.distances_from_points(pts)
        self.LUT = np.reshape(hilbert_idx, [self.bins] * dims)
     
    
    def sort(self, pts, squash_factor = 1, center=None, size=None):
        
        '''
        Sort points along the precalculated Hilbert
        curve approximation. We return a reordered 
        index list that can be used to actually
        bring the points into the correct order.
        
        We expect there to be N points with dim
        dimensions each in the following order:
        
        pts[N, dim]
        '''
        
        if center is None:
            center = np.mean(pts, axis=0)
            
        if size is None:
            size = np.std(pts, axis=0)
        
        # Shift to center, rescale with standard deviation, squash with sigmoid
        pts_clean = (pts - center) / size / squash_factor
        pts_squashed = expit(pts_clean)
        
        # Calculate bin indices, move iterable coordinate to the last position
        pts_bin = np.floor((self.bins - 1) * pts_squashed).astype(int)
        pts_bin = np.moveaxis(pts_bin, 0, -1)
        
        # Fetch Hilbert index from LUT
        hilbert_idx = self.LUT[tuple(pts_bin)]
        
        # Sort and return
        return np.argsort(hilbert_idx)
    
    
#####################################################################################


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



@njit(parallel=True)
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
    

    
#####################################################################################

class crng():
    
    '''
    A prng that keeps track of blocks of random data. Currently,
    only gaussian noise is implemented, with no correlation.
    If a random block is requested, it is generated and stored
    in here. On subsequent requests, the same block of data
    is returned, so that the caller will get the same result.
    
    Blocks can be rerolled, optionally with an correlation
    factor, that keeps the block elementwisely highly correlated
    to the previous state. That way, the caller gets a slightly
    modified version of the previous random data, but will still
    sample from gaussian distributions.
    '''
    
    def __init__(self, seed=None):
        self.gen = np.random.default_rng(seed)
        self.block = dict()
        self.old_blocks = dict()
        self.corr  = dict()
        
    def get(self, block_name, shape=None, corr=0):
        
        self.corr[block_name] = corr
        
        if block_name in self.block:
            if shape is not None:
                assert self.block[block_name].shape == shape, f"Rerequest of block {block_name} ({self.block[block_name].shape}) with different shape ({shape})"
            
            return self.block[block_name]
        
        else:
            shape = (1,) if shape is None else shape
            self.block[block_name] = self.gen.normal(size=shape)
            return self.block[block_name]
        
    def reroll_block(self, block_name, corr=None):
        corr = self.corr[block_name] if corr is None else corr
        self.old_blocks[block_name] = self.block[block_name]
        new_block = self.gen.normal(size=self.block[block_name].shape)
        self.block[block_name] = corr * self.block[block_name] + np.sqrt(1-corr**2) * new_block
        
    def reroll(self, corr=None):
        for b in self.block:
            self.reroll_block(b, corr)
            
    def revert(self):
        self.block.update(self.old_blocks)
            
            
#####################################################################################


class BrownianBridges:
    '''
    Generator for Brownian Bridges.
    
    The cummulative sum of independent normal gaussian variables is
    the discretized equivalent of Brownian motion. Likewise, a second
    cummulative summation corresponds to an integratet Brownian
    motion.
    
    A bridge is a Brownian motion (or any higher order equivalent)
    conditioned on the endpoint. This tool generates samples from
    bridges of any order. Note that the variance of these bridges
    quickly explodes for higher integration orders and bridges lengths, leading
    to error.
    
    Intermediate results are cached, which makes sampling new bridges
    of an already seend length and order much quicker.
    '''
    
    def __init__(self):
        
        self.cache = dict()
        
        
        
    def get(self, count, length, order, RNG_block_name, RNG_correlation, RNG):
        
        ''' Check if data is already present, otherwise generate new '''
        if (length, order) in self.cache:
            SbarCH = self.cache[(length, order)]['SbarCH']
            mubar = self.cache[(length, order)]['mubar']
            
        else:
            # Covariance matrix W0 of independent gaussians
            W0 = np.eye(length - 1)
            
            # Intergration I is a linear operation in discrete world
            I = np.tril(np.ones((length - 1, length - 1)))
            
            # Integrate 'order' times, starting from the base gaussians
            Wn = W0
            for _ in range(order):
                # Transformation Aw of gaussian vector w equates to
                # transforming the covariance W of w as AWA.T
                Wn = I @ Wn @ I.T
            
            # For conditioning, the covariance Wn is divided into
            # 4 subquadrants Sij:
            #
            #  [ S11  S12 ]
            #  [ S21  S22 ]
            #
            # Sections 1 correspond to the unobserved dimensions,
            # sections 2 correspond to the observed dimensions. Since
            # we're only conditioning on the last summand, 2 is only
            # one element long.
            
            S11 = Wn[:-1, :-1]
            S12 = Wn[-1:, :-1]
            S22 = Wn[-1:, -1:]
            S22i = np.linalg.inv(S22)
            S21 = Wn[:-1, -1:]
            
            # Sbar is the new covariance under conditioning on the final sum
            Sbar = S11 - S21 @ S22i @ S12
            SbarCH = np.linalg.cholesky(Sbar)
            
            # mubar are the mean values of the conditioned variables.
            # Here we scale it by the standard deviation of the
            # unconditioned final sum, so that the user can easily
            # sample from the unconditioned process by scaling it
            # with a normal variable and adding it to the bridges samples.
            mubar = S21 @ S22i
            mubar = np.pad(mubar, ((1,1), (0,0)))
            mubar[-1,0] = 1
            mubar *= np.sqrt(S22)
            
            self.cache[(length, order)] = dict(SbarCH=SbarCH, mubar=mubar)
            
        w = RNG.get(RNG_block_name, (length-2, count), RNG_correlation)
        a = np.pad(SbarCH @ w, ((1,1),(0,0)))
        
        return a, mubar.copy()
    

#####################################################################################

class ConditionedBridge:
    
    '''
    A generator for second order Brownian bridges with both the enpoint as well
    as the last step being freely conditioned.
    '''
    
    def __init__(self):
        self.cache = dict()
        
    def condition(self, A):
        A_11 = A[:-1, :-1]
        A_21 = A[-1:, :-1]
        A_12 = A[:-1, -1:]
        A_22 = A[-1:, -1:]
        A_22i = np.linalg.inv(A_22)

        A_transformer = A_12 @ A_22i
        A_bar = A_11 - A_transformer @ A_21
        return A_bar, A_transformer
        
    def get(self, L, endpoint, enddiff, RNG_block_name, RNG_correlation, RNG, mean_only=False):
        
        '''
        Return samples from the brownian bridges B2[k,i] of length i=0..L, satisfying
        the endpoint condition B2[:,-1] = endpoint[:] and the final difference
        condition B2[:,-1]-B2[:,-1] = enddiff[:].
        
        Parameters:
        
        L: ............... Length of the trajectories
        
        endpoint: ........ Conditioned final sum. endpoint.shape=(k,) with k the number of
                           trajectories to generate, each with it's own condition.
                        
        enddiff: ......... Like endpoint, only for the final step height.
        
        RNG_block_name: .. Key for the RNG dictionary.
        
        RNG_correlation: . Correlation for the CRNG
        
        RNG: ............. The CRNG instance to use.
        '''
        
        if L in self.cache:
            SIG1_TRANSF = self.cache[L]['SIG1_TRANSF']
            SIG2_TRANSF = self.cache[L]['SIG2_TRANSF']
            SIG2_BAR_CHOLESKY = self.cache[L]['SIG2_BAR_CHOLESKY']
            STD1 = self.cache[L]['STD1']
            STD2 = self.cache[L]['STD2']
        
        else:
            SIG0 = np.eye(L-1)
            INT  = np.tril(np.ones((L-1, L-1)))
            SIG1 = INT @ SIG0 @ INT.T
            
            STD1 = np.sqrt(SIG1[-1,-1])
            
            SIG1_BAR, SIG1_TRANSF = self.condition(SIG1)
            
            INT  = np.tril(np.ones((L-2, L-2)))
            SIG2 = INT @ SIG1_BAR @ INT.T
            STD2 = np.sqrt(SIG2[-1,-1])
            
            SIG2_BAR, SIG2_TRANSF = self.condition(SIG2)
            SIG2_BAR_CHOLESKY = np.linalg.cholesky(SIG2_BAR)
            
            self.cache[L] = dict(SIG1_TRANSF=SIG1_TRANSF,
                                 SIG2_TRANSF=SIG2_TRANSF,
                                 SIG2_BAR_CHOLESKY=SIG2_BAR_CHOLESKY,
                                 STD1=STD1, STD2=STD2)
        
        MU1_BAR = SIG1_TRANSF * enddiff
        MU2  = np.cumsum(MU1_BAR, axis=0) # Same as INT
        MU2_BAR = MU2[:-1, :] + SIG2_TRANSF @ (endpoint - enddiff - MU2[-1:, -1:])
        w = RNG.get(RNG_block_name, MU2_BAR.shape, RNG_correlation)
        W = (SIG2_BAR_CHOLESKY @ w) * (1 - mean_only)  + MU2_BAR
        W = np.concatenate((0 * endpoint[None, :], W, endpoint[None, :]-enddiff[None, :], endpoint[None, :]))
        
        return dict(mu=MU2_BAR, w=W, std1=STD1, std2=STD2)



def bridge_logl(T_N, D, N):
    """
    Calculate the Log-Likelihood of Bridge Parameters
    T_N: endpoint of double-cummulative sum "shift"
    D:   difference of the last to points "slope"
    N:   Length of the bridge
    """
    # Berechne Varianzen und Kovarianzen
    var_TN = (N * (N + 1) * (2 * N + 1)) / 6
    var_D = N
    cov_TN_D = (N * (N + 1)) / 2

    # Kovarianzmatrix
    cov_matrix = np.array([[var_TN, cov_TN_D],
                           [cov_TN_D, var_D]])

    # Berechne die Determinante und die Inverse der Kovarianzmatrix
    det_cov = np.linalg.det(cov_matrix)
    inv_cov = np.linalg.inv(cov_matrix)

    # Erstelle den Vektor der Beobachtungen
    x = np.array([T_N, D])

    # Berechne den Log-Likelihood-Wert
    d = len(x)  # Dimension der Zufallsvariablen (hier 2)
    log_likelihood_value = -0.5 * (np.log(det_cov) + x.T @ inv_cov @ x + d * np.log(2 * np.pi))

    return log_likelihood_value


#####################################################################################
    

def ParticleFilter(LOGL, SDE, X0, TY, params):
    
    ''' __________________________________________________________________
    
                         Particle filter for SDEs.
    
    Following: [ Wiqvist 2021 ] https://doi.org/10.1016/j.csda.2020.107151
    
    ______________________________________________________________________
    Given a set of observations Y at points in time T, construct
    a likelihood estimation by propagating an ensemble of particles.
    All particles start at time t=0 and will be filtered at all
    observation times T.
    
    
    ______________________________________________________________________
    parameters:
    
        LOGL:  Function of form LOGL(X, Y, params).
               X is a stack of n system states X[n, ...].
               Y is an observation of the state. It may
               contain NaNs, to indicate unobserved variables.
               Returns an unnormalized log-likelihood p(Yi|X) of
               the observations under a system state as a np.array.
               
        SDE:   Propagator for the stochastic differential equation.
               Of the form SDE(T0, T1, X0, params)
               X0 is a stack of n states X[n, ...] at time T0. The
               propagator outputs the new states X1 at time T1,
               given parameters params in a dictionary.
               
        X0:    Initial states of n particles of the form X[n, ...].
               Note that this array defines the number of particles
               that will be used.
               
               The X0 swarm encodes the prior on the state before the
               first observation. Keep that in mind in case there's an
               observation at T=0!
               
        TY:    TY is a list of observations, ordered by observation time.
               Each observation consists of a dictionary defining
               the time "T" of the observation, as well as an np.array
               "Y" with observations of the state. It may contain NaNs
               if parts of the state have not been observed.
               
               Example:
               Coordinates of a 3D particle X = (x, y, z) has been
               observed at t=3.14. No data is present for z. Then
               the relevant part of TY may look like this:
               
               TY = [..., dict(T=3.14, Y=np.array([3.1, 0.0, np.nan])), ...]
                    
        params:  A dictionary of parameters that the SDE, the
                 LKLHD and the filter use.
                 
                 Stuff that the filter looks for in params:
                 
                 "output_trajectory" - A posterior sample of X(T) for
                          all observation times is constructed. (TODO)
                                       
                 "hilbert_sort_coordinates" - before resampling, perform
                          a Hilbert sort on the provided coordinates.
                          This is used in CPMMH to ensure that the correlation
                          of the auxilary variable is less likely to be
                          destroyed by prunes. (TODO)
    
    ______________________________________________________________________
    returns:   A dictionary containing the log-likelihood of the observations,
               as well as possibly a bunch of telemetry.
               
    ______________________________________________________________________ '''
    
    
    output_trajectory = params.get('output_trajectory', False)
    output_bridge_parameters = params['output_bridge_parameters']
    HilbertSort_dims = params.get('HilbertSort_dims', ())
    HilbertSort_useObservationCenter = params.get('HilbertSort_useObservationCenter', False)

    Agent = params.get('Agent', None)
    additional_agent_information = params.get('additional_agent_information', [])
    
    if len(HilbertSort_dims):
        sorter = params['HilbertSorter']
        
    ''' __________________________________________________________________ '''
    
    
    
    # Initiate system state X_ at time T_
    T_ = 0
    X_ = X0.copy()
    Y_last = np.zeros(2)
    n_parts = X0.shape[0]
    
    # Storage
    result = dict(logl = 0, sde_aux_logl=0)  # Output 
    
    if output_trajectory or output_bridge_parameters:
        # If a trajectory is requested, we collect States, and resample indices
        # for all observations. At the end we sample backwards a valid chain
        # that made it to the end. If SDE provides a high-resolution trajectory
        # we also stitch together a trajectory containing intermediate states.
        interm = []

    terminate_early = False
    fresh_episode = True
    total_reward_collected = 0.
    diversities = []
    action0_stresses = []
    action1_stresses = []
    
    # SUCCESSIVE FILTERING for all observations
    for num, ty_ in enumerate(TY):
        
        T = ty_['T']  # Time of current observation
        Y = ty_['Y']  # Current observation
        
        for param_name in params['observation_parameters']:
            params[param_name] = ty_[param_name]

        if Agent:
            old_mean_x = X_[:,0].mean()
            old_mean_y = X_[:,1].mean()
            old_mean_phi = X_[:,2].mean()

            if params['base_on_last_observation']:
                old_mean_x = Y_last[0]
                old_mean_y = Y_last[1]
                dx_ = Y[0] - old_mean_x
                dy_ = Y[1] - old_mean_y
                old_mean_phi = np.arctan2(dy_, dx_)

            if not params['rotate_observations_to_swarm']:
                old_mean_phi = 0.

            c = np.cos(old_mean_phi)
            s = np.sin(old_mean_phi)

            obs = []

            for ifuture in range(params['futures'] + 1):
                
                off_x = ty_['future_x'][ifuture] - old_mean_x
                off_y = ty_['future_y'][ifuture] - old_mean_y

                off_x_t = c * off_x + s * off_y
                off_y_t = -s * off_x + c * off_y

                obs.append(off_x_t)
                obs.append(off_y_t)

            obs = [obs + additional_agent_information]
            if params['provide_absolute_orientation']:
                obs[0] += [c, s]

            if params['provide_absolute_x_location']:
                obs[0] += [old_mean_x]

            if params['provide_absolute_y_location']:
                obs[0] += [old_mean_y]

            if params['provide_swarm_polarity']:
                phi_ = X_[:,2]
                cos_ = np.mean(np.cos(phi_))
                sin_ = np.mean(np.sin(phi_))
                obs[0] += [cos_, sin_]

            if params['deterministic_agent']:
                action = Agent.predict_deterministic(torch.Tensor(obs))
            else:
                action = Agent.predict(torch.Tensor(obs), fresh_episode)
            fresh_episode = False

            action0 = action[0][0].item()
            action1 = action[0][1].item()

            if params['use_bridge_offsets']:
                offsetRNG = params['offsetRNG']
                off_shift, off_slope = offsetRNG.get(num, (2,))
                action0 += off_shift
                action1 += off_slope
            
            target_shift_P = action0 * params['action_range0']
            target_slope_P = action1 * params['action_range1']

            action0_stresses.append(np.abs(action0))
            action1_stresses.append(np.abs(action1))
            
            params['target_shift_PHI'] = target_shift_P
            params['target_slope_PHI'] = target_slope_P
        

        X_old = X_.copy()
        prop = SDE(T_, T, X_, params) # Propagate state to current observation
        X_   = prop['X1']
        T_   = T
        #result['logl'] += prop['logl']
        result['sde_aux_logl'] += prop['logl']
        Y_last = Y.copy()
        
        #if len(HilbertSort_dims):
        #    X_sort = X_[:, HilbertSort_dims]
        #    if HilbertSort_useObservationCenter:
        #        resort_idx = sorter.sort(X_sort, center=Y[:2], size=params['sigma_observation'])
        #    else:
        #        resort_idx = sorter.sort(X_sort)
        #    X = X_[resort_idx, :]
        
        logW = LOGL(X_, Y, params)          # Calculate weights
        W = np.exp(logW)

        max_logW = np.max(logW)
        logsumW = max_logW + np.log(np.sum(np.exp(logW - max_logW)))
        result['logl'] += logsumW
        
        #sumW = np.sum(W)
        #reward = sumW / n_parts * np.exp(prop['reward_logl']) # This can be 1 at most
        
        reward = logsumW + prop['reward_logl'] - np.log(n_parts) # This can be 0 a most
        reward -= np.log(params['reward_range']) # This can be -log(reward_range) at most, which is >0
        reward /= -np.log(params['reward_range']) # This can be 1 at most
        if reward < 0 and Agent and Agent.recording:
            terminate_early = True
            reward = 0

        
        
        
        # Hand out reward for prediction if agent is learning
        if Agent and Agent.recording:
            Agent.reward(reward)

        total_reward_collected += reward
        
        ''' Update the loglikelihood of the estimation
        
            For numerical stability, we use the loglikelihoods of the particles
            to update the total estimate. For this we extract the largest contribution
            and name it max_logW, while the rest gets calles logW_rest.
            
            We can decompose a logarithm of a sum
            
            log (sum_0^N xi) = log(x0) + log(1 + sum_1^N exp(log(xi) - log(x0)))
            
            So the total loglikelihood gets updated by a term that consists of the
            largest loglikelihood plus a correction term that only needs relative
            deltas in the loglikelihood. This is numerically prefered, as directly
            evaluating the exponentials xi would result in zero-floats, if the
            likelihood is vanishingly small.
            
        '''
        
        idx_rsmpl = StructuredResample(W)   # Resample

        n_unique_indices = len(np.unique(idx_rsmpl))
        diversities.append(n_unique_indices / n_parts)
        
        if output_trajectory or output_bridge_parameters:
            # Store resampling and state
            traj_dict = dict(X=X_, idx_rsmpl=idx_rsmpl, per_particle_streams=dict())
            
            # Fetch additional data from the SDE result
            per_particle_streams = prop.get('per_particle_streams', dict())
            traj_dict['per_particle_streams'].update(per_particle_streams)
            interm.append(traj_dict)
        
        X_ = X_[idx_rsmpl, ...]

        if terminate_early:
            result['logl'] += -np.inf
            break
            

    result['total_reward_collected'] = total_reward_collected
    if Agent and Agent.recording:
        
        result['mean_diversity'] = np.mean(diversities)
        result['min_diversity'] = np.min(diversities)
        result['spread_diversity'] = np.std(diversities) / np.mean(diversities)
        result['action0_stress'] = np.mean(np.abs(action0_stresses))
        result['action0_stress_max'] = np.max(np.abs(action0_stresses))
        result['action1_stress'] = np.mean(np.abs(action1_stresses))
        result['action1_stress_max'] = np.max(np.abs(action1_stresses))
    
    if output_trajectory or output_bridge_parameters:

        ''' TODO
            WARNING BUGGY

            This code doesn't work for more than one SINGLE STREAM. I don't know
            why, but it's interleaving all the streams and it's trying to concat
            all of them at the end.

            Good luck finding it, I'm busy trying to finish my thesis ;)
        '''
        
        # Backpropagation of the final trajectory.
        # We take a uniform sample from X_, as it's already
        # filtered by the last observation. We're building
        # the list backwards and revert it at the end.
        
        idx = np.random.randint(n_parts) # The lucky survivor at the end
        rev_traj = []
        pps_keys = interm[0]['per_particle_streams'].keys()

        assert len(pps_keys) <= 1, f"Nope, sorry, a bug prevents us from using more than one stream. Fix it before using this. Items: {pps_keys}"
        rev_pps = dict.fromkeys(pps_keys, [])
        
        for step in reversed(interm):
            idx = step['idx_rsmpl'][idx]
            rev_traj.append(step['X'][idx, ...])
            for key in pps_keys:
                rev_pps[key].append(step['per_particle_streams'][key][idx, ...])
        
        result['trajectory'] = np.stack(list(reversed(rev_traj)), axis=-1)
        result['streams'] = dict()
        for key in pps_keys:
            #print("key", key)
            #print("revppskey", list(reversed(rev_pps[key])))
            #result['streams'][key] = list(reversed(rev_pps[key]))
            result['streams'][key] = np.concatenate(list(reversed(rev_pps[key])), axis=-1)
        
    return result


class ZeroBasedRNG:
    """
    A random number generator that starts all values at 0. 
    Reroll adds Gaussian random values to the existing values with a specified strength (std). 
    Revert restores the previous state.
    """
    
    def __init__(self, seed=None):
        self.gen = np.random.default_rng(seed)
        self.block = dict()  # Stores the current blocks of data
        self.old_blocks = dict()  # Stores the previous state of blocks
        self.strength = dict()  # Stores the reroll strength (std) for each block
    
    def get(self, block_name, shape=None, strength=1.0):
        """
        Retrieve the current state of the block. If the block does not exist,
        it initializes it to zeros with the given shape and sets the reroll strength.
        """
        if block_name in self.block:
            if shape is not None:
                assert self.block[block_name].shape == shape, f"Rerequest of block {block_name} ({self.block[block_name].shape}) with different shape ({shape})"
            return self.block[block_name]
        else:
            shape = (1,) if shape is None else shape
            self.block[block_name] = np.zeros(shape)
            self.strength[block_name] = strength
            return self.block[block_name]
        
    def reroll_block(self, block_name, strength=None):
        """
        Add Gaussian noise to a specific block using the given strength (std).
        """
        if block_name in self.block:
            # Save the current state
            self.old_blocks[block_name] = self.block[block_name].copy()
            # Use the provided strength or fall back to the block-specific strength
            strength = self.strength[block_name] if strength is None else strength
            # Update the block with Gaussian noise
            self.block[block_name] += self.gen.normal(scale=strength, size=self.block[block_name].shape)
        else:
            raise KeyError(f"Block {block_name} does not exist.")
        
    def reroll(self, strength=None):
        """
        Add Gaussian noise to all existing blocks, optionally using a global strength.
        """
        for block_name in self.block:
            self.reroll_block(block_name, strength)
    
    def revert(self):
        """
        Revert all blocks to their previous state.
        """
        for block_name in self.old_blocks:
            self.block[block_name] = self.old_blocks[block_name]
        self.old_blocks.clear()


class ZeroBasedRNG:
    """
    A random number generator that starts all values at 0. 
    Reroll adds Gaussian random values to the existing values with a specified strength (std). 
    Revert restores the previous state.
    """
    
    def __init__(self, seed=None):
        self.gen = np.random.default_rng(seed)
        self.block = dict()  # Stores the current blocks of data
        self.old_blocks = dict()  # Stores the previous state of blocks
        self.strength = dict()  # Stores the reroll strength (std) for each block
    
    def get(self, block_name, shape=None, strength=1.0):
        """
        Retrieve the current state of the block. If the block does not exist,
        it initializes it to zeros with the given shape and sets the reroll strength.
        """
        if block_name in self.block:
            if shape is not None:
                assert self.block[block_name].shape == shape, f"Rerequest of block {block_name} ({self.block[block_name].shape}) with different shape ({shape})"
            return self.block[block_name]
        else:
            shape = (1,) if shape is None else shape
            self.block[block_name] = np.zeros(shape)
            self.strength[block_name] = strength
            return self.block[block_name]
        
    def reroll_block(self, block_name, strength=None):
        """
        Add Gaussian noise to a specific block using the given strength (std).
        """
        if block_name in self.block:
            # Save the current state
            self.old_blocks[block_name] = self.block[block_name].copy()
            # Use the provided strength or fall back to the block-specific strength
            strength = self.strength[block_name] if strength is None else strength
            # Update the block with Gaussian noise
            self.block[block_name] += self.gen.normal(scale=strength, size=self.block[block_name].shape)
        else:
            raise KeyError(f"Block {block_name} does not exist.")
        
    def reroll(self, strength=None):
        """
        Add Gaussian noise to all existing blocks, optionally using a global strength.
        """
        for block_name in self.block:
            self.reroll_block(block_name, strength)
    
    def revert(self):
        """
        Revert all blocks to their previous state.
        """
        for block_name in self.old_blocks:
            self.block[block_name] = self.old_blocks[block_name]
        self.old_blocks.clear()





def effective_sample_size(samples):
    """
    Calculate the Effective Sample Size (ESS) of an MCMC chain.
    
    Parameters:
    samples (list or np.ndarray): A 1D array-like object of MCMC samples.
    
    Returns:
    float: The effective sample size.
    """
    samples = np.asarray(samples)
    n = len(samples)
    
    # Compute mean and variance of the samples
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)
    
    # Normalize samples by subtracting mean
    normalized_samples = samples - sample_mean
    
    # Compute autocorrelations for lag values
    autocorrelations = []
    for lag in range(n):
        # Calculate the autocorrelation for the given lag
        if lag == 0:
            autocorr = 1.0
        else:
            autocorr = np.sum(
                normalized_samples[:n-lag] * normalized_samples[lag:]
            ) / (n * sample_var)
        autocorrelations.append(autocorr)
        
        # Stop if the autocorrelation becomes non-positive
        if autocorr <= 0:
            break
    
    # Compute ESS
    ess = n / (1 + 2 * np.sum(autocorrelations[1:]))
    return ess


def draw_bridge_level(N, L, fill=False, num_points=100, **kwargs):
    """
    Draws the ellipsoid for the density at a given level L (as a fraction of the maximum density) in the current figure.
    Parameters:
    N (int): Length of the bridge.
    L (float): Desired density level as a fraction of the maximum density (e.g., 0.1 for 10%).
    num_points (int): Number of points for drawing the ellipsoid.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Variance and covariance terms
    var_TN = (N * (N + 1) * (2 * N + 1)) / 6
    var_D = N
    cov_TN_D = (N * (N + 1)) / 2

    # Covariance matrix
    cov_matrix = np.array([[var_TN, cov_TN_D],
                           [cov_TN_D, var_D]])

    # Log-likelihood at the origin (center of the ellipsoid)
    d = 2  # Dimension of the random variable
    log_likelihood_center = -0.5 * (np.log(np.linalg.det(cov_matrix)) + d * np.log(2 * np.pi))

    # Scale log-likelihood by the given density level L
    log_likelihood_target = log_likelihood_center + np.log(L)

    # Compute the Mahalanobis distance threshold
    threshold = -2 * log_likelihood_target - d * np.log(2 * np.pi) - np.log(np.linalg.det(cov_matrix))

    if threshold <= 0:
        raise ValueError(f"Density level L={L} leads to an invalid threshold.")

    # Create the ellipsoid boundary in the canonical form
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle = np.array([np.cos(theta), np.sin(theta)])  # Unit circle

    # Apply the covariance matrix for scaling and rotation
    scaling_matrix = np.linalg.cholesky(cov_matrix)
    ellipse = np.sqrt(threshold) * scaling_matrix @ circle

    if fill:
        plt.fill(ellipse[0, :], ellipse[1, :], **kwargs)  # Fill the ellipse
    else:
        plt.plot(ellipse[0, :], ellipse[1, :], **kwargs)  # Just plot the ellipse



def draw_empirical_ellipsoid(x, y, L, fill=False, num_points=100, **kwargs):
    """
    Estimates the mean and covariance of the data (x, y) and draws the corresponding ellipsoid at a given density level L.
    Parameters:
    x (array-like): Data for the x-coordinate.
    y (array-like): Data for the y-coordinate.
    L (float): Desired density level as a fraction of the maximum density (e.g., 0.1 for 10%).
    num_points (int): Number of points for drawing the ellipsoid.
    **kwargs: Additional arguments to pass to plt.plot (e.g., linestyle, color, linewidth).
    """
    # Step 1: Estimate the mean and covariance matrix from the data
    data = np.vstack((x, y))
    mean = np.mean(data, axis=1)
    cov_matrix = np.cov(data)

    # Step 2: Log-likelihood at the origin (center of the ellipsoid)
    d = 2  # Dimension of the random variable
    log_likelihood_center = -0.5 * (np.log(np.linalg.det(cov_matrix)) + d * np.log(2 * np.pi))

    # Step 3: Scale log-likelihood by the given density level L
    log_likelihood_target = log_likelihood_center + np.log(L)

    # Step 4: Compute the Mahalanobis distance threshold
    threshold = -2 * log_likelihood_target - d * np.log(2 * np.pi) - np.log(np.linalg.det(cov_matrix))

    if threshold <= 0:
        raise ValueError(f"Density level L={L} leads to an invalid threshold.")

    # Step 5: Create the ellipsoid boundary in the canonical form (unit circle)
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle = np.array([np.cos(theta), np.sin(theta)])  # Unit circle

    # Step 6: Apply the covariance matrix for scaling and rotation (Cholesky decomposition)
    scaling_matrix = np.linalg.cholesky(cov_matrix)
    ellipse = np.sqrt(threshold) * scaling_matrix @ circle

    # Step 7: Translate the ellipsoid to the estimated mean
    ellipse[0, :] += mean[0]
    ellipse[1, :] += mean[1]

    if fill:
        plt.fill(ellipse[0, :], ellipse[1, :], **kwargs)  # Fill the ellipse
    else:
        plt.plot(ellipse[0, :], ellipse[1, :], **kwargs)  # Just plot the ellipse

import numpy as np

def kl_divergence_from_samples(N, x, y):
    """
    Compute the Kullback-Leibler (KL) divergence between the empirical distribution (estimated from x and y)
    and the theoretical distribution with mean (0, 0) and covariance matrix derived from N.
    
    Parameters:
    - N (int): Length of the bridge, used to compute the covariance of the theoretical distribution.
    - x (array-like): Data for the x-coordinate of the empirical distribution.
    - y (array-like): Data for the y-coordinate of the empirical distribution.
    
    Returns:
    - float: The KL divergence between the two distributions.
    """
    # Estimate the mean and covariance from the empirical samples
    empirical_mean = np.mean([x, y], axis=1)
    empirical_cov = np.cov(x, y)

    # Calculate the covariance of the theoretical distribution based on N
    var_TN = (N * (N + 1) * (2 * N + 1)) / 6
    var_D = N
    cov_TN_D = (N * (N + 1)) / 2

    # Theoretical covariance matrix
    theoretical_cov = np.array([[var_TN, cov_TN_D],
                                 [cov_TN_D, var_D]])

    # Theoretical mean is always (0, 0)
    theoretical_mean = np.array([0, 0])

    # Compute KL divergence between the empirical and theoretical distributions
    d = len(empirical_mean)  # Dimension (2D)
    
    # Inverse of the theoretical covariance matrix
    inv_theoretical_cov = np.linalg.inv(theoretical_cov)
    
    # Trace term: tr( Sigma_theoretical_inv * Sigma_empirical )
    trace_term = np.trace(inv_theoretical_cov @ empirical_cov)
    
    # Mean difference term: (mu_empirical - mu_theoretical)^T * Sigma_theoretical_inv * (mu_empirical - mu_theoretical)
    mean_diff = empirical_mean - theoretical_mean
    mean_diff_term = mean_diff.T @ inv_theoretical_cov @ mean_diff
    
    # Determinant ratio: log(det(Sigma_theoretical) / det(Sigma_empirical))
    det_ratio = np.log(np.linalg.det(theoretical_cov) / np.linalg.det(empirical_cov))
    
    # KL divergence formula
    kl_div = 0.5 * (trace_term + mean_diff_term - d + det_ratio)
    
    return kl_div