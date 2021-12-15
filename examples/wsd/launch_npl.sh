#!/bin/sh
#SBATCH --job-name=TensorLNN
#SBATCH -p npl
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:2
#SBATCH -o output_npl_%j.out
#SBATCH -t 00:10:00

source activate pytorch-env
srun python wsd_main.py
