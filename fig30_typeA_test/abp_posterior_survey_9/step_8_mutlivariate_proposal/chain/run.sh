#!/bin/bash -l
source ~/.bashrc
conda activate bayes_rod
python rocks_worker.py "$@"