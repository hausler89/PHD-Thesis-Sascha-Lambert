**This branch serves to archive the original code. Please refer to the `main` branch of this repository for a reworked and cleaned-up version of the results, with improved comments and clearer connections to the thesis.**

# Original programs, sorted by figure

Dissertation: S. Lambert, *Bayesian Inference in Stochastic Biological Systems*, doctoral thesis, Georg-August-Universität
Göttingen (2025), [doi:10.53846/goediss-11359](https://doi.org/10.53846/goediss-11359)
([eDiss](https://ediss.uni-goettingen.de/handle/11858/16080)).

Unmodified copies of the original notebooks, scripts and data behind the figures of the dissertation, one folder per
figure (or group of figures that share code). Files used by several figures are copied into each folder. Every folder
has a README with the original location of each file and notes on what is missing.

Not copied: training logs and model checkpoints (`runs/`, `agents/`), caches (`__pycache__`) and single files larger
than 25 MB; the READMEs list them. Some notebooks therefore do not run as they are.

Removed for privacy: the HTCondor job logs and error outputs (`*.log`, `*.err`) of the cluster runs. Personal paths in
notebook outputs (home directories, cloud folders) are replaced by `~`, and the cluster is referred to as "HPC cluster".
Apart from that, all files are unmodified.

## Legal notes

- **License.** No license has been chosen for this repository yet.
- **Third-party code.** `ppolib.py` contains major parts of the PPO implementation of
  [CleanRL](https://github.com/vwxyzjn/cleanrl) (`cleanrl/ppo_continuous_action.py`,
  https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy), Copyright (c) 2019 CleanRL developers,
  MIT License. The license text is in [`LICENSES/CleanRL-MIT.txt`](LICENSES/CleanRL-MIT.txt). Copies of `ppolib.py`
  are in: `fig36_38_bridge_focusing/`, `fig40_magnetotaxis/`, `fig45_48_bridge_inference/PMMH5 Full Logl/`, `fig47_training_progress/PMMH7 Well Animation/`, `fig49_50_msd_comparison/PMMH7 Randomize Observations/`, `fig49_50_msd_comparison/PMMH7 Well/`, `fig51_harmonic_potential/PMMH7 Well Low particle count/`.
  Reference: S. Huang et al., JMLR 23(274):1–18 (2022), http://jmlr.org/papers/v23/21-1342.html.
- **Data by others.** The Cytosim force curves (`*.csv` and the zip archive in `fig17_20_cytosim/`) were produced by
  Kristian A. Pajanonot and Komal Bhattacharyya, who are the authors of these data. Please credit them when using
  the data.
- **External simulation code (not included).** The rod trajectories of chapter 4 were generated with the ABP simulation
  suite of the group (gitlab.gwdg.de/NLD/activeparticles/abp, branch `sl_rod_paper`); see
  `fig21_rod_collisions/rod_posterior_survey_1/README.md`.
- **Work by others (not included).** The least-squares analysis of Figures 23 and 24 is by M. Duchêne.
- **Methods from the literature** implemented in this code: particle filter and correlated pseudo-marginal MCMC after
  S. Wiqvist et al., Comput. Stat. Data Anal. 157, 107151 (2021), https://doi.org/10.1016/j.csda.2020.107151
  (`inference_tools*.py`); Proximal Policy Optimization after J. Schulman et al., arXiv:1707.06347 (2017);
  diffusion coefficients of spherocylinders after A. Lüders, E. Zander, P. Nielaba, J. Chem. Phys. 155(10) (2021),
  https://doi.org/10.1063/5.0060063.


| Folder | Content |
|---|---|
| `fig01_02_coin` | Coin likelihood and posteriors (Figures 1 and 2). |
| `fig03_submarine_illustration` | Illustration (Inkscape), not computed. |
| `fig04_submarine_2d` | Submarine 2D prior/likelihood/posteriors. |
| `fig05_cytosim_snapshots` | MISSING: rendered Cytosim snapshots (external simulation by K. A. Pajanonot / K. Bhattacharyya); no code here. |
| `fig06_force_regimes` | Force curves of the entropic/Hookean spring model. |
| `fig07_09_simple_model` | Simulated force trajectories (Figures 7, 8) and grid posteriors (Figure 9). |
| `fig10_11_selection_bias` | Selection bias and mitigation by dropping data. |
| `fig12_13_surrogate` | Neural-network surrogate of the stress-strain curve; Figure 13 matches 6_learn_force_model.ipynb exactly. |
| `fig14_noisy_likelihood` | Likelihood of noisy force measurements (sympy). |
| `fig15_velocity_comparison` | Influence of the driving velocity on the posterior. |
| `fig16_experimental_design` | Expected information gain (matches exactly). |
| `fig17_20_cytosim` | Cytosim data, unbinding statistics, posteriors (e5_comparison_all), rejection modelling (e6_dropout_mitigation, matches Figure 20 exactly; n |
| `fig21_rod_collisions` | Rod trajectories: step_1_data_generation/step1_data_generation.ipynb. Rod data come from the external ABP suite (gitlab.gwdg.de/NLD/activepa |
| `fig22_25_illustrations` | Figures 22 and 25 are illustrations (not computed). |
| `fig23_24_least_squares` | MISSING here: least-squares analysis by M. Duchene (bachelor thesis); the code is archived in her university account. |
| `fig26_phi_filter` | Orientation filter with posterior-predictive bands; 3_merles_parametrization_NJIT.ipynb is the latest version, 1/2 are earlier ones, 4 adds  |
| `fig27_kalman_vs_particle` | Survey 3 (100 trajectories, single-mode torque): step_2_coarse_overview (kalman*/particle* images) and step_3_high_resolution (zoomed views) |
| `fig28_envelope_fit` | Gaussian-envelope proposal: step_8_mutlivariate_proposal (ABP) and step_8_envelope_fit (rods). |
| `fig29_bsplines` | step_8_mutlivariate_proposal/bsplines.ipynb (matches exactly). |
| `fig30_typeA_test` | Panels A/B: survey 4 (two fixed modes), step_2/step_3; panel C: survey 9 (B-spline representation), step_8/step_9. |
| `fig31_rod_inference` | step_8_envelope_fit/prepare_proposal.ipynb saves comparison_results.png. |
| `fig32_noise_levels` | Ballistic Brownian motion at different noise levels. |
| `fig33_mvn_conditioning` | 3D normal samples and conditioning (matches). |
| `fig34_finite_differences` | Construction of the doubly conditioned bridge. |
| `fig35_prior_bridges` | Conditioned vs. unconditioned bridge (matches). |
| `fig36_38_bridge_focusing` | Bridged swarms (Figure 36), focusing strength (37) and achievable densities (38); PMMH_5_Precisions-Copy2.ipynb saves 21_std_slope / 22_lklh |
| `fig39_wall_scattering` | Swarms scattering at a wall (matches). |
| `fig40_magnetotaxis` | Magnetotaxis swarms. The saved outputs of PMMH_3_Field.ipynb do not match the printed panels exactly (later re-run with other settings); the |
| `fig41_42_cpmmh_sampling` | CPMMH trajectory sampling: sticky chains (saves sticky_chain*) and MCMC warm-up. |
| `fig43_44_illustrations` | Figures 43 and 44 are illustrations (not computed); source graphic of Figure 44. |
| `fig45_48_bridge_inference` | Velocity inference with PPO-learned bridges: velocity inference.ipynb (template, Figures 45, 46, 48 left), papermill.ipynb runs it as pm_vel |
| `fig47_training_progress` | Training snapshots and progress (Untitled2.ipynb matches the bottom panel). |
| `fig49_50_msd_comparison` | Bootstrapped MSD (testMSD.ipynb, bootstrap_mean_std.py) and estimator accuracy: 'PMMH7 Randomize Observations/Evaluate Scraps.ipynb' collect |
| `fig51_harmonic_potential` | Inference in a harmonic potential: Evaluate Scraps.ipynb collects the 50 runs pm_velocity0-49.ipynb (template: velocity inference.ipynb, lau |
