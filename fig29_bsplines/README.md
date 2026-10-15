# fig29_bsplines

step_8_mutlivariate_proposal/bsplines.ipynb (matches exactly).

Copied unchanged from (paths relative to the original working directory):

- `Code\rod_bayes\Bayesian Rods\abp_posterior_survey_9`

## Execution on the HPC cluster

MCMC chains run as HTCondor jobs, one job per chain (folders `chain/` and `step_10_gaussian_posterior/`):
`prepare_run.py` and `parameters.txt` define the jobs, `job.jdf` is the job description, `run.sh` starts `rocks_worker.py`
(one chain), `submit.sh` / `clear_and_submit.sh` submit them. The chains are combined and analysed in the notebooks of the
same step (e.g. `analyze_chains.ipynb`, `prepare_proposal.ipynb`).
