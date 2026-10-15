#!/bin/bash -l
source ~/.bashrc
conda activate bayes_rod
rm *.err; rm *.csv; rm *.log;
rm chunk_*
rm chain*
rm results.npz
condor_submit job.jdf 
