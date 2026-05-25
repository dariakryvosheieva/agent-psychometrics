#!/bin/bash
#SBATCH --job-name=info-ablation
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=output/information_ablation/slurm-%j.out
#SBATCH --error=output/information_ablation/slurm-%j.err

# Run the hierarchical (rows 1-4) and leave-one-out (rows 5-6) information
# ablation across all four datasets with 20 shuffled k-fold splits per cell,
# matching the main Table 2 protocol. Produces mean ± std per cell where
# std = Bessel-corrected sample SD across 20 seed-level mean AUCs.
#
# Parallelism: 4 datasets fan out in the dataset pool. Within each dataset,
# 20 fold seeds run in parallel via max_seed_workers, giving up to
# 4 x 20 = 80 concurrent per-seed CV runs (under the 96-CPU normal-QOS cap).
# Each per-seed run writes to a distinct seed{seed}_fold{fold} IRT cache
# subdir, so seeds never collide on cache writes.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Cap per-process thread counts so we don't oversubscribe the 96 cores when
# many workers are running in parallel.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p output/information_ablation

python -m experiment_new_tasks.run_information_ablation \
    --n_fold_seeds 20 \
    --max_seed_workers 20 \
    --output output/information_ablation/results_with_std.csv
