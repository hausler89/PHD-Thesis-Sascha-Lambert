# fig30_typeA_test

Panels A/B: survey 4 (two fixed modes), step_2/step_3; panel C: survey 9 (B-spline representation), step_8/step_9.

Copied unchanged from (paths relative to the original working directory):

- `Code\rod_bayes\Bayesian Rods\abp_posterior_survey_4`
- `Code\rod_bayes\Bayesian Rods\abp_posterior_survey_9`

Notes:

- skipped (>25 MB): Code\rod_bayes\Bayesian Rods\abp_posterior_survey_4\step_3_high_resolution\view_1\results.npz (61 MB)
- skipped (>25 MB): Code\rod_bayes\Bayesian Rods\abp_posterior_survey_4\step_3_high_resolution\view_2\results.npz (61 MB)

## Execution on the HPC cluster

Panels A and B: HTCondor jobs on a grid of 2000 x 2000 amplitude pairs (alpha_1, alpha_2), split into 40 x 40 = 1600
chunks of 50 x 50 points, one job per chunk (particle filter with 1000 particles, and Kalman filter).

- `abp_posterior_survey_4/step_3_high_resolution/view_2/` (alpha 1-500, panel A) and `view_1/` (alpha 1-20, panel B).
- In each view: `prepare_run.py` writes the chunk list (`parameters.txt`) and the job description (`job.jdf`);
  `run.sh` starts `rocks_worker.py`, which evaluates one chunk; `clear_and_submit.sh` submits all jobs.
- Merging: `abp_posterior_survey_4/step_3_high_resolution/plot_chunks.ipynb` combines the chunks and plots the fields.

## Execution on the HPC cluster

MCMC chains run as HTCondor jobs, one job per chain (folders `chain/` and `step_10_gaussian_posterior/`):
`prepare_run.py` and `parameters.txt` define the jobs, `job.jdf` is the job description, `run.sh` starts `rocks_worker.py`
(one chain), `submit.sh` / `clear_and_submit.sh` submit them. The chains are combined and analysed in the notebooks of the
same step (e.g. `analyze_chains.ipynb`, `prepare_proposal.ipynb`).
