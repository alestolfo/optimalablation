#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu_requeue
#SBATCH --job-name=1000-edge_pruning
#SBATCH --gpus=1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 edge_pruning_iterative.py 1000