#!/bin/bash

for var in $(seq 1 $1);
do
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --job-name=$var-sens-ioi-edge_pruning
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/sens_$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/sens_$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 edge_pruning.py --subfolder $var

EOT
done