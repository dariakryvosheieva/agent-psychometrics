#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1

python -u -m experiment_agent_features.predict_question_difficulty \
  --benchmark terminalbench \
  --method judge
