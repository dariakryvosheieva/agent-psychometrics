#!/bin/bash
#SBATCH --job-name=tb-binomial
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=output/experiment_a_terminalbench_binomial/slurm-%j.out
#SBATCH --error=output/experiment_a_terminalbench_binomial/slurm-%j.err

# Run the Terminal-Bench 2.0 all-responses (per-attempt binomial) appendix
# table with the main Table 2 protocol (20 shuffled k-fold splits, seed-level
# mean and Bessel-corrected sample SD across seeds). Uses binomial-likelihood
# IRT via the per-attempt response file, evaluating AUC over per-attempt
# expanded observations.
#
# Only one dataset is selected (terminalbench), so the inner fold-seed pool
# is the only parallelism in play. With 20 seeds, 20 CPUs are actively used
# at peak; the rest of the 96 CPUs are headroom for Pyro retries (handled by
# max_train_attempts=3 with seed-bumping in train_irt_split.py).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p output/experiment_a_terminalbench_binomial

python -m experiment_new_tasks.run_all_datasets \
    --datasets terminalbench \
    --responses_path data/terminalbench/responses_per_attempt.jsonl \
    --per_dataset_output_dir output/experiment_a_terminalbench_binomial \
    --output_dir output/experiment_a_terminalbench_binomial \
    --output output/experiment_a_terminalbench_binomial/results.csv \
    --max_seed_workers 20
