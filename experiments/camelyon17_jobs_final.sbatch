#!/bin/bash
#SBATCH --job-name="cam17_5seeds"
#SBATCH -A rahul
#SBATCH -q ml
#SBATCH -p ml
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time="unlimited"
#SBATCH --exclude dgx1
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm_cam17_%A_%a.out
#SBATCH --error=logs/slurm_cam17_%A_%a.err
#SBATCH --array=0-4  # Just 5 seeds

cd /mfs1/u/viet/bayesian_dpddm

source /pkgs/anaconda3/etc/profile.d/conda.sh
conda activate dpddm

SIZE=20  # 👈 Change this to 10, 20, or 50 as needed
SEED=$((57 + SLURM_ARRAY_TASK_ID))

echo "Running camelyon17_best with sample_size=$SIZE and seed=$SEED"
python experiments/run.py --config-name=camelyon17_best seed=$SEED dpddm.data_sample_size=$SIZE
