# Bayesian Inference in Stochastic Biological Systems: code

Code for the figures of the dissertation

> S. Lambert, *Bayesian Inference in Stochastic Biological Systems*, doctoral thesis, Georg-August-Universität Göttingen
> (2025), [doi:10.53846/goediss-11359](https://doi.org/10.53846/goediss-11359)
> ([eDiss](https://ediss.uni-goettingen.de/handle/11858/16080)).

Entry points are the Jupyter notebooks in `notebooks/`; reusable pieces live in
the small library `src/lambert_thesis` (`from lambert_thesis import toy_models, mcmc, ...`), meant to be read next to the
notebooks. [FIGURES.md](FIGURES.md) lists which notebook produces which figure.

## Setup

```bash
conda env create -f environment.yml      # pinned versions; environment.lock.yml is the full export
conda activate lambert-thesis
python -m ipykernel install --user --name lambert-thesis
```

## Notebooks

| Notebook | Chapter | Figures |
|---|---|---|
| `01_bayes_basics` | Ch. 2 Bayesian inference methods | 1–4 |
| `02a_bonds_basic_inference` | Ch. 3 Vimentin–microtubule bonds: basic inference | 6–9 |
| `02b_bonds_selection_bias` | Ch. 3 selection bias, surrogate model | 10–13 |
| `02c_bonds_noise_and_design` | Ch. 3 noisy forces, experimental design | 14–16 |
| `02d_bonds_cytosim` | Ch. 3 inference on Cytosim data | 17–20 |
| `03_walltorque_bayes` | Ch. 4 wall torques (Bayesian analysis) | 21, 26, 27, 29–31 |
| `04a_bridges_construction` | Ch. 5 Brownian bridges: construction | 32–35 |
| `04b_bridges_nonlinear` | Ch. 5 non-linear dynamics, environments | 36–40 |
| `04c_bridges_sampling` | Ch. 5 sampling trajectories | 41–42 |
| `04d_bridges_ppo_inference` | Ch. 5 inference with PPO-learned bridges | 45–51 |

Library modules (`src/lambert_thesis`): `toy_models`, `mcmc`, `bonds`, `surrogate`, `cytosim`, `walltorque`,
`bridges`, `swarm_survey`, `filters`, `abp_inference`, `ppo`, `bridge_learning`, plus `utils` (seed, cache) and `plotting`.

## Computations on an HPC cluster
The likelihood grids of Figures 27 and 30A/B were computed on an HPC cluster: 1000 × 1000 (Figure 27) and 2000 × 2000
(Figure 30A/B) parameter values, each grid split into 40 × 40 chunks that ran as separate jobs. Notebook
`03_walltorque_bayes` evaluates these grids at coarse resolution with the same filter settings. The job scripts and the
merging of the chunks are in the branch
[`original_scripts`](https://gitlab.gwdg.de/NLD/activeparticles/phd-thesis-sascha-lambert/-/tree/original_scripts)
(folders `fig27_kalman_vs_particle` and `fig30_typeA_test`).

## Data
`data/cytosim/` contains force curves from Cytosim simulations. Authors of these data: Kristian A. Pajanonot and
Komal Bhattacharyya (details in `data/cytosim/PROVENANCE.md`).

## Reproducibility
All randomness is seeded with **12345** (`lambert_thesis.set_seed`). Re-running a notebook gives identical arrays;
numbers may differ slightly from the printed dissertation, which used other seeds.
Expensive notebooks have a `FULL_RESOLUTION` switch (`True`: full settings, hours of computation; `False`: quick settings) and cache intermediate results in `results/`.

## Running
```bash
jupyter lab                                              # interactive
jupyter nbconvert --to notebook --execute --inplace notebooks/01_bayes_basics.ipynb   # headless, one notebook
bash run_all.sh 8                                        # all notebooks in order, 8 threads
```
`run_all.sh` writes one log per notebook to `logs/`. With `FULL_RESOLUTION = True` the complete run takes days on a
workstation; if it is interrupted, run it again: finished notebooks are skipped and cached results are reused.

## License
The code in this repository is released under the MIT License, see [LICENSE](LICENSE). Third-party material keeps its
own terms: `src/lambert_thesis/ppo.py` contains code from CleanRL (MIT License, see below), and the Cytosim data in
`data/cytosim/` are not covered by this license; their authors are named in `data/cytosim/PROVENANCE.md`.

## Third-party code and references

**Code from other projects**
- `src/lambert_thesis/ppo.py` is derived from `cleanrl/ppo_continuous_action.py` of
  [CleanRL](https://github.com/vwxyzjn/cleanrl) ([file](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py),
  [documentation](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy)),
  Copyright (c) 2019 CleanRL developers, MIT License. The license text is in
  [`LICENSES/CleanRL-MIT.txt`](LICENSES/CleanRL-MIT.txt); the changes are listed in the module docstring.
  Reference: S. Huang et al., JMLR 23(274):1–18 (2022), http://jmlr.org/papers/v23/21-1342.html.

**Methods from the literature** (own implementations)
- Particle filter and correlated pseudo-marginal MCMC: S. Wiqvist, A. Golightly, A. T. McLean, U. Picchini,
  Comput. Stat. Data Anal. 157, 107151 (2021), https://doi.org/10.1016/j.csda.2020.107151.
- Proximal Policy Optimization: J. Schulman et al., arXiv:1707.06347 (2017), https://doi.org/10.48550/arXiv.1707.06347.
- Diffusion coefficients of spherocylinders: A. Lüders, E. Zander, P. Nielaba, J. Chem. Phys. 155(10) (2021),
  https://doi.org/10.1063/5.0060063.

**Data**
- Cytosim force curves by Kristian A. Pajanonot and Komal Bhattacharyya (see above); Cytosim: F. Nédélec,
  D. Foethke, New J. Phys. 9, 427 (2007), https://doi.org/10.1088/1367-2630/9/11/427.

**Libraries** (installed via `environment.yml`, not distributed with this repository, each under its own license):
NumPy, SciPy, numba, matplotlib, pandas, scikit-learn, SymPy, emcee, PyMC, ArviZ, PyTorch, tqdm, JupyterLab.

## How this repository was made

The methods, models and results come from the dissertation ([doi:10.53846/goediss-11359](https://doi.org/10.53846/goediss-11359)). This repository restructures
the original research code into notebooks and a library. The restructuring (code organisation, documentation, notebook
text) was done with the help of the AI assistant Claude Code (Anthropic). The author reviewed the results against the
dissertation and is responsible for the content. Deviations from the dissertation are listed in [FIGURES.md](FIGURES.md).

## Original scripts

The original, unmodified notebooks and scripts behind the figures, sorted by figure, are archived in the branch
[`original_scripts`](https://gitlab.gwdg.de/NLD/activeparticles/phd-thesis-sascha-lambert/-/tree/original_scripts).
