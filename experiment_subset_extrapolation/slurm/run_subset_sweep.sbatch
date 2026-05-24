#!/bin/bash
#SBATCH --job-name=subset-extrap
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=output/experiment_subset_extrapolation/slurm-%j.out
#SBATCH --error=output/experiment_subset_extrapolation/slurm-%j.err

# Run the subset-extrapolation sweep across all datasets, sizes, and seeds.
# Designed for a single node with ~96 CPUs.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Cap per-process thread counts so we don't oversubscribe the 96 cores when
# many workers are running in parallel. The Python runner also sets these in
# its worker initializer, but exporting here covers the parent process too.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Activate the project virtualenv if it exists. Adapt this to your cluster's
# module / conda setup if needed.
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p output/experiment_subset_extrapolation

python -m experiment_subset_extrapolation.run_all_datasets \
    --max_workers 96 \
    --n_seeds 20 \
    --output_root output/experiment_subset_extrapolation \
    --plot
