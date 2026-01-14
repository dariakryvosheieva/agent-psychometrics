#!/bin/bash
#SBATCH --partition=sched_mit_psfc_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=traj_embed_resume
#SBATCH --output=logs/embed_resume_%j.log

# Resume trajectory embedding computation (skips existing .npz files)

set -e

source .venv/bin/activate

python -m experiment_b.compute_trajectory_embeddings \
    --trajectories_dir trajectory_data/unified_trajs \
    --output_dir chris_output/experiment_b/trajectory_embeddings \
    --content_type full \
    --instruction_type difficulty
