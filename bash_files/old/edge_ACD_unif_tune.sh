#!/bin/bash

for var in "$@"
do
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p seas_gpu
#SBATCH --constraint="a100"
#SBATCH --job-name=unif-$var-ioi-edge_pruning
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/unif-pre_$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/unif-pre_$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 edge_pruning_unif_bias_calc.py -d ioi -n unif_window --lamb $var

EOT
done