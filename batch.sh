#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu_test
#SBATCH --job-name=train_sae
#SBATCH --gpus=1
#SBATCH --mem=10000
#SBATCH -t 0-3:00
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 models.py