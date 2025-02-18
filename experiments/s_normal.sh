#!/bin/bash
#SBATCH --job-name=bdpddm_sweep
#SBATCH -c 16
#SBATCH --qos=normal
#SBATCH --partition=t4v1,t4v2,rtx6000,a40
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm.out
#SBATCH --error=logs/slurm.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

source ~/.bashrc
conda activate bayes
which python
wandb agent opent03-team/bayesian_dpddm/obxprov4
 