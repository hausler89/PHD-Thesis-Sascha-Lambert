#!/usr/bin/env bash
# Run all notebooks in order: fills results/, figures/ and the notebooks' outputs.
#   conda activate lambert-thesis
#   bash run_all.sh [THREADS]          (default: 8 threads)
# Interrupted? Run it again: finished notebooks are skipped, cached results in results/ are reused.
set -u
THREADS=${1:-8}
export NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONHASHSEED=12345
cd "$(dirname "$0")"
mkdir -p logs

for nb in notebooks/*.ipynb; do
    name=$(basename "$nb" .ipynb)
    if [ -f "logs/$name.done" ]; then echo "skip     $name (done)"; continue; fi
    echo "$(date '+%F %T')  start  $name" | tee -a logs/run_all.log
    if jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb" > "logs/$name.log" 2>&1; then
        touch "logs/$name.done"
        echo "$(date '+%F %T')  done   $name" | tee -a logs/run_all.log
    else
        echo "$(date '+%F %T')  FAILED $name, see logs/$name.log" | tee -a logs/run_all.log
        exit 1
    fi
done
echo "$(date '+%F %T')  all notebooks done" | tee -a logs/run_all.log
