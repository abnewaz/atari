#!/bin/bash
#SBATCH -J atari_dt
#SBATCH -p matador
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --gpus-per-node=1
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -e

mkdir -p logs
cd /home/annewaz/research_lab/atari

source /home/annewaz/miniforge3/etc/profile.d/conda.sh
conda activate atari

python3 main.py train
python3 main.py eval --checkpoint checkpoints/dt_breakout_best.pt --render