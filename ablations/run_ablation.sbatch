#!/bin/bash
#SBATCH --job-name="cam17_reghead_abl"
#SBATCH -A rahul
#SBATCH -q ml
#SBATCH -p ml
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time="unlimited"
#SBATCH -w concerto3
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm_%a.out
#SBATCH --error=logs/slurm_%A.err

cd /mfs1/u/viet/bayesian_dpddm

source /pkgs/anaconda3/etc/profile.d/conda.sh
conda activate dpddm


python ablations/temp.py --config-name camelyon17_best dpddm.data_sample_size=100
