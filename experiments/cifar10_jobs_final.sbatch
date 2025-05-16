#!/bin/bash
#SBATCH --job-name="cifar10_experiments"
#SBATCH -A rahul
#SBATCH -q ml
#SBATCH -p ml
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time="unlimited"
#SBATCH --open-mode=append
#SBATCH --exclude dgx1
#SBATCH --output=logs/slurm_cifar10_%A_%a.out
#SBATCH --error=logs/slurm_cifar10_%A_%a.err
#SBATCH --array=0-149  # 3 sizes × 50 seeds = 150 jobs

cd /mfs1/u/viet/bayesian_dpddm

source /pkgs/anaconda3/etc/profile.d/conda.sh
conda activate dpddm

SIZES=(10 20 50)
IDX=$((SLURM_ARRAY_TASK_ID / 50))        # Which size: 0, 1, or 2
SEED_OFFSET=$((SLURM_ARRAY_TASK_ID % 50))  # 0–49

SIZE=${SIZES[$IDX]}
SEED=$((57 + SEED_OFFSET))

echo "Running cifar10_best with sample_size=$SIZE and seed=$SEED"
python experiments/run.py --config-name=cifar10_best seed=$SEED dpddm.data_sample_size=$SIZE
