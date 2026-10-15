#!/bin/bash -l
source ~/.bashrc
conda activate bayes_rod
rm *.err; rm *.csv; rm *.log;
rm chunk_*
rm results.npz
python prepare_run.py
condor_submit job.jdf 
