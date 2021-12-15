#!/bin/sh
#SBATCH --job-name=TensorLNN
#SBATCH -p dcs
#SBATCH -N 32
#SBATCH -n 192
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:6
#SBATCH -o output_dcs_%j.out
#SBATCH -t 02:00:00

source activate wmlce-1.7.0
srun python wsd_main.py




