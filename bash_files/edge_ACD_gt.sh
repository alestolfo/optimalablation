#!/bin/bash

for var in "$@"
do
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu_requeue
#SBATCH --job-name=$var-ioi-edge_pruning
#SBATCH --constraint="a100"
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/pre_$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/pre_$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 edge_pruning_gt.py --lamb $var

EOT
done