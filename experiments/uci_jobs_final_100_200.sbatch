#!/bin/bash
#SBATCH --job-name="uci_experiments"
#SBATCH -A rahul
#SBATCH -q ml
#SBATCH -p ml
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time="unlimited"
#SBATCH --exclude dgx1
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm_uci_%A_%a.out
#SBATCH --error=logs/slurm_uci_%A_%a.err
#SBATCH --array=0-39  # 2 sizes × 20 seeds

cd /mfs1/u/viet/bayesian_dpddm

source /pkgs/anaconda3/etc/profile.d/conda.sh
conda activate dpddm

SIZES=(20 50)
IDX=$((SLURM_ARRAY_TASK_ID / 20))         # Which size: 0 (100), 1 (200)
SEED_OFFSET=$((SLURM_ARRAY_TASK_ID % 20)) # 0 to 19

SIZE=${SIZES[$IDX]}
SEED=$((57 + SEED_OFFSET))

echo "Running uci_best with sample_size=$SIZE and seed=$SEED"
python experiments/run.py --config-name=uci_best seed=$SEED dpddm.data_sample_size=$SIZE
