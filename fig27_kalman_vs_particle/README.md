# fig27_kalman_vs_particle

Survey 3 (100 trajectories, single-mode torque): step_2_coarse_overview (kalman*/particle* images) and step_3_high_resolution (zoomed views).

Copied unchanged from (paths relative to the original working directory):

- `Code\rod_bayes\Bayesian Rods\abp_posterior_survey_3`

## Execution on the HPC cluster

The likelihood fields were computed as HTCondor jobs. Each view is a grid of 1000 x 1000 parameter values (alpha, k),
split into 40 x 40 = 1600 chunks of 25 x 25 points, one job per chunk; every job evaluates the Kalman-filter and the
particle-filter likelihood (2000 particles) for its chunk.

- `abp_posterior_survey_3/step_3_high_resolution/view_3/` (alpha 1-200, k 0.1-15) and `view_1/` (alpha 1-15, k 0.1-1.5)
  are the two views of Figure 27; `view_2`, `view_4`, `view_5` are further zoom levels, `template/` the starting point.
- In each view: `prepare_run.py` writes the chunk list (`parameters.txt`) and the job description (`job.jdf`);
  `run.sh` starts `rocks_worker.py`, which evaluates one chunk; `clear_and_submit.sh` submits all jobs.
- Merging: `step_3_high_resolution/plot_chunks.ipynb` (`stitch_data`) combines the chunk files of a view into
  `results.npz` and plots the fields.
