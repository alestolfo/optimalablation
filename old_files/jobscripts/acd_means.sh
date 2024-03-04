#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --job-name=pruning_means
#SBATCH --gpus=1
#SBATCH --mem=40000
#SBATCH -t 0-12:00
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 pruning_means.py
