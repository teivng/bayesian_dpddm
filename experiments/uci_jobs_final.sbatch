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
#SBATCH --array=0-149  # 3 sizes × 50 seeds

cd /mfs1/u/viet/bayesian_dpddm

source /pkgs/anaconda3/etc/profile.d/conda.sh
conda activate dpddm

SIZES=(10 20 50)
IDX=$((SLURM_ARRAY_TASK_ID / 50))       # Which size: 0 (10), 1 (20), 2 (50)
SEED_OFFSET=$((SLURM_ARRAY_TASK_ID % 50))  # 0 to 49

SIZE=${SIZES[$IDX]}
SEED=$((57 + SEED_OFFSET))

echo "Running uci_best with sample_size=$SIZE and seed=$SEED"
python experiments/run.py --config-name=uci_best seed=$SEED dpddm.data_sample_size=$SIZE
