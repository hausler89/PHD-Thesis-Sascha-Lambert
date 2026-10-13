"""Chapter 5.8: velocity inference with PPO-learned bridges, and the MSD reference estimator.

The particle filter of :mod:`lambert_thesis.filters` is run with :func:`abp_inference.propagate`.
Before each observation the agent sees the next ``futures + 1`` observations relative to the swarm
(centre of mass, rotated into the swarm's mean orientation) plus the proposed velocity offset and the
current observation relaxation, and returns the bridge (shift, slope). After weighting, it receives the
reward  R = (log sum_i w_i + log p_bridge - log N - log r) / (-log r)  (r = ``reward_range``), and an
episode is truncated as soon as R < 0 during training.
"""
import numpy as np
import torch
from scipy.optimize import curve_fit

from . import abp_inference as abp
from . import filters
from .utils import RESULTS_DIR

REWARD_RANGE = 1e-15


def agent_observation(X, obs, extra):
    """State for the agent: offsets of the next observations relative to the swarm, rotated by the mean orientation."""
    mx, my, mphi = X[:, 0].mean(), X[:, 1].mean(), X[:, 2].mean()
    c, s = np.cos(mphi), np.sin(mphi)
    feat = []
    for fx, fy in zip(obs["future_x"], obs["future_y"]):
        ox, oy = fx - mx, fy - my
        feat += [c * ox + s * oy, -s * ox + c * oy]
    return torch.Tensor([feat + list(extra)])


def run_filter(p, observations, n_parts, sigma_obs, rng, agent=None, deterministic=False, extra=(),
               keep_history=True):
    """One particle-filter run. With an ``agent`` the bridges are chosen by the policy (stochastic while the
    agent is recording, else as requested by ``deterministic``) and rewards are handed to a recording agent.

    Returns the filter result plus ``total_reward`` and ``length`` (number of observations reached).
    """
    state = dict(fresh=True, total_reward=0.0)
    recording = agent is not None and agent.recording

    def before(k, X, o):
        if agent is None:
            return
        f = agent_observation(X, o, extra)
        a = agent.predict_deterministic(f) if deterministic else agent.predict(f, state["fresh"])
        state["fresh"] = False
        p["target_shift"], p["target_slope"] = a[0][0].item(), a[0][1].item()

    def after(k, lsw, prop):
        r = (lsw + prop["reward_logl"] - np.log(n_parts) - np.log(REWARD_RANGE)) / -np.log(REWARD_RANGE)
        stop = False
        if r < 0 and recording:
            stop, r = True, 0.0
        if recording:
            agent.reward(r)
        state["total_reward"] += r
        return stop

    res = filters.particle_filter(lambda X, o: abp.log_obs_density(X, o, sigma_obs),
                                  lambda T0, T1, X, k: abp.propagate(T0, T1, X, k, p),
                                  np.zeros((n_parts, 3)), observations, rng, before_step=before, after_step=after,
                                  keep_history=keep_history)
    res["total_reward"] = state["total_reward"]
    res["length"] = res["trajectory"].shape[1] if "trajectory" in res else 0
    return res


def likelihood_scan(p, observations, v_gt, v_offsets, n_parts, sigma_obs, rng, agent=None, extra_fn=None):
    """Log-likelihood estimates (and posterior bridge parameters per observation) over velocity offsets.
    ``extra_fn(v_offset)`` gives the additional agent inputs. Returns dict(v, logl, reward, slope, shift)."""
    out = dict(v=np.asarray(v_offsets), logl=[], reward=[], slope=[], shift=[])
    for v in v_offsets:
        p["logv"] = np.log(v_gt * v)
        p["rng"].reroll()
        res = run_filter(p, observations, n_parts, sigma_obs, rng, agent=agent, deterministic=True,
                         extra=() if extra_fn is None else extra_fn(v))
        out["logl"].append(res["logl"]); out["reward"].append(res["total_reward"])
        out["slope"].append(res["streams"]["bridge"][0]); out["shift"].append(res["streams"]["bridge"][1])
    return {k: np.asarray(v) for k, v in out.items()}


def train_agent(agent, p, observations, v_gt, v_range, n_parts, sigma_obs, rng, n_training_steps=250,
                batch=1024, relaxation_adoption=0.9995, ent_coef=0.15, log_path=None):
    """PPO training of the bridge agent (section 5.8.2, steps 1-5).

    Each training step collects >= ``batch`` state-action-reward triplets from filter runs at random velocity
    offsets in ``v_range`` and performs a PPO update (minibatches of 16, 4 epochs). The observation precision
    is annealed: it starts relaxed by ``agent.relaxation`` and is tightened by ``relaxation_adoption`` per
    episode once the swarm traverses all observations; exploration (entropy bonus) is switched off then.
    With ``log_path``, one line per training step is appended to that file (episodes seen, mean and maximum
    episode length of the step, observation relaxation) to monitor the progress while training runs.
    Returns the episode history and, for every training step, the trajectory of its last episode
    (NaN-padded to the number of observations) together with the number of episodes seen so far.
    """
    hist = dict(length=[], reward=[], v=[], logl=[])
    snap_traj = np.full((n_training_steps, 3, len(observations)), np.nan)
    snap_episodes = np.zeros(n_training_steps, int)
    full_length = False
    log = open(log_path, "w", buffering=1) if log_path else None
    for step in range(n_training_steps):
        agent.start_recording()
        n_before = len(hist["length"])
        while len(agent.obs) < batch:
            v_off = rng.uniform(*v_range)
            p["logv"] = np.log(v_gt * v_off)
            p["rng"].reroll()
            res = run_filter(p, observations, n_parts, sigma_obs * agent.relaxation, rng, agent=agent,
                             extra=(v_off, np.log(agent.relaxation)))
            if res["length"] >= len(observations) or full_length:
                agent.relaxation = max(agent.relaxation * relaxation_adoption, 1.0)
                agent.ent_coef = 0.0
                full_length = True
            else:
                agent.ent_coef = ent_coef
            hist["length"].append(res["length"]); hist["reward"].append(res["total_reward"])
            hist["v"].append(v_off); hist["logl"].append(res["logl"])
        agent.learn(minibatch_size=16, epochs=4)
        agent.stop_recording()
        snap_traj[step, :, :res["trajectory"].shape[1]] = res["trajectory"]
        snap_episodes[step] = agent.episode_count
        if log:
            L = hist["length"][n_before:]
            log.write(f"step {step:4d}  episodes {agent.episode_count:6d}  length mean {np.mean(L):5.1f} "
                      f"max {np.max(L):3d}  relaxation {agent.relaxation:.3f}" + "\n")
    if log:
        log.close()
    hist = {k: np.asarray(v) for k, v in hist.items()}
    hist.update(snapshot_trajectory=snap_traj, snapshot_episodes=snap_episodes)
    return hist


def bootstrap_mean_std(x, logl, rng, n_bootstrap=1000, confidence=0.95):
    """Mean and standard deviation of the normalised density exp(logl) over the grid ``x``, with bootstrap
    confidence intervals (resampling grid points with their weights)."""
    w = np.exp(logl - np.max(logl)); w /= w.sum()
    mean = np.sum(w * x); std = np.sqrt(np.sum(w * (x - mean) ** 2))
    bm, bs = [], []
    for _ in range(n_bootstrap):
        i = rng.choice(len(x), size=len(x), p=w)
        ww = w[i] / w[i].sum()
        m = np.sum(ww * x[i]); bm.append(m); bs.append(np.sqrt(np.sum(ww * (x[i] - m) ** 2)))
    a = 1 - confidence
    ci = lambda v: (np.percentile(v, 100 * a / 2), np.percentile(v, 100 * (1 - a / 2)))
    return dict(mean=mean, std=std, ci_mean=ci(bm), ci_std=ci(bs))


# ---------------------------------------------------------------------------
# MSD estimator (section 5.8.3)
# ---------------------------------------------------------------------------
def tamsd_bootstrap(x, y, rng):
    """Time-averaged MSD of one trajectory, with the displacements of every lag resampled (one bootstrap draw)."""
    n = len(x)
    out = np.zeros(n - 1)
    for lag in range(1, n):
        d = (x[:-lag] - x[lag:]) ** 2 + (y[:-lag] - y[lag:]) ** 2
        out[lag - 1] = np.mean(rng.choice(d, size=len(d)))
    return out


def abp_msd(lag, v0, Dr, sigma_obs):
    """MSD of a 2D ABP without translational diffusion, plus the static localisation error."""
    return 2 * v0**2 / Dr**2 * (Dr * lag - 1 + np.exp(-Dr * lag)) + 2 * sigma_obs**2


def msd_velocity_fit(x, y, dt_obs, Dr, sigma_obs, rng, n_bootstrap=100):
    """Bootstrapped fit of the log-MSD with lags up to the rotational diffusion time 1/Dr.
    Returns (velocities of all bootstrap fits, lags, bootstrapped TAMSD curves)."""
    max_lag = int(1 / Dr)
    lags = (1 + np.arange(len(x) - 1)) * dt_obs
    sig = np.ones(len(lags)); sig[max_lag:] = np.inf
    vs, curves = [], []
    for _ in range(n_bootstrap):
        t = tamsd_bootstrap(x, y, rng)
        par, _ = curve_fit(lambda lag, v0: np.log(abp_msd(lag, v0, Dr, sigma_obs)), lags, np.log(t), sigma=sig)
        vs.append(par[0]); curves.append(t)
    return np.array(vs), lags, np.array(curves)


# ---------------------------------------------------------------------------
# Complete inference experiments (section 5.8.3)
# ---------------------------------------------------------------------------
GT_DR, GT_V, DT_OBS, DT, N_OBS, SIG_OBS, FUTURES = 0.5, 1.0, 1.0, 0.1, 30, 0.3, 3

SCENARIOS = {
    # free space, observations = ground truth positions (Figures 45-48)
    "free": dict(obs_noise=False, potential=0.0, n_naive=4000, v_naive=(0.85, 1.1), super_res=True,
                 n_agent=4000, training_steps=250, fit_range=False, msd=False),
    # free space, noisy observations, MSD reference (Figure 50)
    "random": dict(obs_noise=True, potential=0.0, n_naive=4000, v_naive=(0.75, 1.25), super_res=True,
                   n_agent=4000, training_steps=200, fit_range=True, msd=True),
    # harmonic potential with prefactor k = 0.5, force -k x (Figure 51)
    "well": dict(obs_noise=True, potential=0.5, n_naive=2000, v_naive=(0.75, 1.25), super_res=False,
                 n_agent=10, training_steps=200, fit_range=True, msd=True),
}


def _params(potential, seed):
    from .bridges import ConditionedBridge, CorrelatedRNG
    p = dict(logv=np.log(GT_V), logDr=np.log(GT_DR), dt=DT, bridge=ConditionedBridge(), rng=CorrelatedRNG(seed),
             record_bridge_parameters=True)
    if potential:
        p["force"] = lambda X, k=potential: (-k * X[:, 0], -k * X[:, 1])
    return p


def run_experiment(kind, seed, gt_seed=None, agent_only=False, training_steps=None):
    """Full inference on one synthetic observation set of scenario ``kind`` (see ``SCENARIOS``).

    ``gt_seed`` selects the observation set (default: ``seed``), ``seed`` all other randomness. With
    ``agent_only`` only the agent is trained and the bridged filter evaluated (no reference filters).
    Returns a dict with the likelihood scans (naive, super-resolution, bridged), posterior summaries
    (bootstrap mean/std) and the training history of the bridge agent.
    """
    gt_seed = seed if gt_seed is None else gt_seed
    from . import ppo
    cfg = SCENARIOS[kind]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)                       # PPO minibatch shuffling
    # ground truth and observations
    gt = _params(cfg["potential"], gt_seed)
    T, gx, gy = abp.simulate(N_OBS + FUTURES + 1, DT_OBS, gt, rng_seed=gt_seed)
    ox, oy = gx.copy(), gy.copy()
    if cfg["obs_noise"]:
        gt_rng = np.random.default_rng(gt_seed)
        ox += gt_rng.normal(scale=SIG_OBS, size=gx.shape) / np.sqrt(2)
        oy += gt_rng.normal(scale=SIG_OBS, size=gy.shape) / np.sqrt(2)
    obs = abp.make_observations(T, ox, oy, futures=FUTURES)[:N_OBS]
    out = dict(gt_x=gx, gt_y=gy, obs_x=ox, obs_y=oy)
    # naive particle filter (and super resolution)
    if agent_only:
        cfg = dict(cfg, super_res=False, msd=False, fit_range=False)
    p = dict(_params(cfg["potential"], seed + 1), use_bridges=False)
    if not agent_only:
        scan = likelihood_scan(p, obs, GT_V, np.linspace(*cfg["v_naive"], 300), cfg["n_naive"], SIG_OBS, rng)
        out.update({f"naive_{k}": v for k, v in scan.items()})
    if cfg["super_res"]:
        p = dict(_params(cfg["potential"], seed + 2), use_bridges=False)
        scan = likelihood_scan(p, obs, GT_V, np.linspace(*cfg["v_naive"], 100), 100_000, SIG_OBS, rng)
        out.update({f"sr_{k}": v for k, v in scan.items()})
    # MSD reference
    if cfg["msd"]:
        vs, lags, curves = msd_velocity_fit(ox[:N_OBS + FUTURES + 1], oy[:N_OBS + FUTURES + 1], DT_OBS, GT_DR, SIG_OBS, rng)
        out.update(msd_v=vs, msd_lags=lags, msd_curves=curves)
    # velocity range for the agent
    v_range = (0.85, 1.1)
    if cfg["fit_range"]:
        l = out["naive_logl"] - out["naive_logl"].max()
        w = np.exp(l)
        (a, b, c), _ = curve_fit(lambda x, a, b, c: a * x**2 + b * x + c, out["naive_v"], l,
                                 sigma=1 / np.maximum(w, 1e-300), absolute_sigma=True)
        m, s = -b / (2 * a), np.sqrt(-1 / (2 * a))
        v_range = (m - 4 * s, m + 4 * s)
    out["v_range"] = np.array(v_range)
    # train the bridge agent
    agent = ppo.PPOAgent(2 * FUTURES + 4, 2, learning_rate=2e-3, hidden=16)
    agent.gamma, agent.relaxation = 0.5, 0.5 / SIG_OBS
    p = dict(_params(cfg["potential"], seed + 3), use_bridges=True)
    log_path = RESULTS_DIR / f"training_{kind}_{gt_seed}_{seed}.log"
    hist = train_agent(agent, p, obs, GT_V, v_range, cfg["n_agent"], SIG_OBS, rng,
                       n_training_steps=training_steps or cfg["training_steps"], log_path=log_path)
    out.update({f"train_{k}": v for k, v in hist.items()})
    out["head_width"] = agent.head_width()
    # bridged particle filter with the deterministic policy
    scan = likelihood_scan(p, obs, GT_V, np.linspace(*v_range, 300), cfg["n_agent"], SIG_OBS * agent.relaxation, rng,
                           agent=agent, extra_fn=lambda v: (v, np.log(agent.relaxation)))
    out.update({f"bridge_{k}": v for k, v in scan.items()})
    # posterior summaries
    for key in ("naive", "sr", "bridge"):
        if f"{key}_logl" in out:
            r = bootstrap_mean_std(out[f"{key}_v"], out[f"{key}_logl"], rng)
            out[f"{key}_mean"], out[f"{key}_std"] = r["mean"], r["std"]
            out[f"{key}_ci_mean"] = np.array(r["ci_mean"])
    rew = -(out["bridge_reward"] - out["bridge_reward"].max()) * np.log(REWARD_RANGE)
    r = bootstrap_mean_std(out["bridge_v"], rew, rng)
    out["reward_mean"], out["reward_std"] = r["mean"], r["std"]
    if cfg["msd"]:
        out["msd_mean"], out["msd_std"] = np.mean(out["msd_v"]), np.std(out["msd_v"])
    return out


def run_experiments(kind, seeds, folder, **kw):
    """Run (or load) several experiments; each result is stored in ``folder/run_<seed>.npz``.
    Keyword arguments are passed to :func:`run_experiment`. Progress (runs done, time per run, estimated end)
    is appended to ``folder/progress.log``."""
    import time
    from datetime import datetime, timedelta
    from pathlib import Path
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if not (folder / f"run_{s}.npz").exists()]
    t_start, n_done = time.time(), 0
    results = []
    for s in seeds:
        path = folder / f"run_{s}.npz"
        if not path.exists():
            print(f"[{kind}] experiment with seed {s} ...", flush=True)
            res = run_experiment(kind, s, **kw)
            np.savez_compressed(path, **{k: np.asarray(v) for k, v in res.items()})
            n_done += 1
            per_run = (time.time() - t_start) / n_done
            eta = datetime.now() + timedelta(seconds=per_run * (len(todo) - n_done))
            with open(folder / "progress.log", "a") as log:
                log.write(f"{datetime.now():%Y-%m-%d %H:%M}  {kind}  seed {s}  done {n_done}/{len(todo)}  "
                          f"{per_run / 60:.1f} min/run  ETA {eta:%H:%M}\n")
        with np.load(path) as f:
            results.append({k: f[k] for k in f.files})
    return results


def plot_training_progress(lengths, ax=None, window=50):
    """Episode lengths during training (dots) with a running mean: the reach first stagnates at the
    first observations, jumps to the first turn, and finally explodes to the full trajectory."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    lengths = np.asarray(lengths, float)
    ax.plot(lengths, ".", ms=1, alpha=0.3, label="episode length")
    if len(lengths) >= window:
        ax.plot(np.arange(window - 1, len(lengths)), np.convolve(lengths, np.ones(window) / window, "valid"),
                "k", lw=1.5, label=f"running mean ({window} episodes)")
    ax.set(xlabel="Episode", ylabel="observations reached")
    ax.legend(fontsize=8)
    return ax
