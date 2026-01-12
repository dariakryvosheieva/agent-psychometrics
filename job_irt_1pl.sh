#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal
#
# Train IRT (1PL) on SWE-bench Verified evaluation results aggregated in:
#   trajectory_data/irt_verified.jsonlines

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship

source /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/activate

which python
python -V
python -c "import sys; print(sys.executable)"

export PYTHONHASHSEED=0

python /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/prep_swebench.py \
       --experiments_dir /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/experiments/evaluation/verified \
       --output_path /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_verified_20251115_full.jsonl

python swebench_irt/train.py \
       --data_path out/chris_irt/swebench_verified_20251115_full.jsonl \
       --dims 1 \
       --model 1pl \
       --output_dir out/chris_irt/swebench_verified_20251115_full \
       --epochs 5000 \
       --seed 0




