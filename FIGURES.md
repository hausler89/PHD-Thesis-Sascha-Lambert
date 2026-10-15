# Figure catalog

Figure numbers and pages refer to the printed dissertation (Revision 2025).
Generated images are written to `figures/`, named `figNN...`.

Status: *computed* in the notebook; *illustration* (not computed); *external* simulation render; *not included*: least-squares analysis by M. Duchêne; *replaced*: posterior sampling uses emcee instead (notebook 03).

Notebooks with expensive computations have a `FULL_RESOLUTION` switch (`True`: full settings, hours of computation; `False`: quick settings). Results are cached in `results/`.

| Figure | Page | Content | Notebook | Status |
|---|---|---|---|---|
| 1 | 16 | Coin likelihood | `01_bayes_basics` | computed |
| 2 | 17 | Coin posteriors | `01_bayes_basics` | computed |
| 3 | 19 | Submarine 1D posterior | `01_bayes_basics` | illustration |
| 4 | 21 | Submarine 2D prior/likelihood/posterior | `01_bayes_basics` | computed |
| 5 | 29 | Cytosim snapshots | – | external |
| 6 | 33 | Polymer force regimes | `02a_bonds_basic_inference` | computed |
| 7 | 34 | Entropic-spring force trajectories | `02a_bonds_basic_inference` | computed |
| 8 | 36 | 50 force trajectories, variable v | `02a_bonds_basic_inference` | computed |
| 9 | 37 | Posterior of bond parameters | `02a_bonds_basic_inference` | computed |
| 10 | 38 | Selection bias | `02b_bonds_selection_bias` | computed |
| 11 | 40 | Bias mitigation by dropping data | `02b_bonds_selection_bias` | computed |
| 12 | 43 | Surrogate learning of stress-strain curve | `02b_bonds_selection_bias` | computed |
| 13 | 44 | Surrogate performance | `02b_bonds_selection_bias` | computed |
| 14 | 46 | Noisy force likelihood | `02c_bonds_noise_and_design` | computed |
| 15 | 49 | Influence of design parameter v | `02c_bonds_noise_and_design` | computed |
| 16 | 50 | Expected information gain | `02c_bonds_noise_and_design` | computed |
| 17 | 52 | Cytosim raw data | `02d_bonds_cytosim` | computed |
| 18 | 53 | Unbinding statistics | `02d_bonds_cytosim` | computed |
| 19 | 56 | Posterior of bond parameters (Cytosim) | `02d_bonds_cytosim` | computed |
| 20 | 57 | Bond 2 rejection modelling | `02d_bonds_cytosim` | computed |
| 21 | 61 | Single rod collisions | `03_walltorque_bayes` | computed |
| 22 | 63 | Simulation setup | – | illustration |
| 23 | 64 | Wall-torque least-squares fit (M. Duchêne) | – | not included |
| 24 | 66 | Aspect-ratio study (M. Duchêne) | – | not included |
| 25 | 70 | Inference process | – | illustration |
| 26 | 72 | Orientation filtering | `03_walltorque_bayes` | computed |
| 27 | 74 | Kalman vs particle filter | `03_walltorque_bayes` | computed (coarse grid; printed version computed on an HPC cluster) |
| 28 | 75 | Gaussian envelope fitting | – | replaced |
| 29 | 76 | Quadratic B-splines | `03_walltorque_bayes` | computed |
| 30 | 78 | Bayesian test case (two modes, B-splines) | `03_walltorque_bayes` | computed (panels A/B on a coarse grid; printed version computed on an HPC cluster) |
| 31 | 79 | Posterior torque for active rods | `03_walltorque_bayes` | computed |
| 32 | 86 | Ballistic Brownian motion noise levels | `04a_bridges_construction` | computed |
| 33 | 91 | MVN conditioning | `04a_bridges_construction` | computed |
| 34 | 93 | Construction of conditioned noise | `04a_bridges_construction` | computed |
| 35 | 96 | Conditioned vs unconditioned bridge | `04a_bridges_construction` | computed |
| 36 | 97 | Bridges in non-linear dynamics | `04b_bridges_nonlinear` | computed |
| 37 | 99 | Focusing strength | `04b_bridges_nonlinear` | computed |
| 38 | 100 | Achievable particle densities | `04b_bridges_nonlinear` | computed |
| 39 | 102 | Wall scattering swarms | `04b_bridges_nonlinear` | computed |
| 40 | 103 | Bridges under magnetotaxis | `04b_bridges_nonlinear` | computed |
| 41 | 106 | Sticky chains | `04c_bridges_sampling` | computed |
| 42 | 107 | MCMC warm-up | `04c_bridges_sampling` | computed |
| 43 | 108 | Velocity/bridge correlation | – | illustration |
| 44 | 115 | RL step | – | illustration |
| 45 | 117 | Particle filter likelihood/posterior | `04d_bridges_ppo_inference` | computed |
| 46 | 119 | Bridge parameter posterior | `04d_bridges_ppo_inference` | computed |
| 47 | 120 | PPO training progress | `04d_bridges_ppo_inference` | computed |
| 48 | 121 | PPO bridge posterior approximation | `04d_bridges_ppo_inference` | computed |
| 49 | 123 | Bootstrapped MSD fit | `04d_bridges_ppo_inference` | computed |
| 50 | 123 | Velocity estimator accuracy | `04d_bridges_ppo_inference` | computed |
| 51 | 124 | Harmonic potential inference | `04d_bridges_ppo_inference` | computed |
