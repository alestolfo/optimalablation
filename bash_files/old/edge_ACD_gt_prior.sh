#!/bin/bash

for strength in "0.3"
do
for prior in "2e-3"
do
for var in "$@"
do
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --job-name=prior-$var-gt-edge_pruning
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/gtprio-pre_$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/gtprio-pre_$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 edge_pruning_vertex_gt_prior.py --lamb $var --prior $prior --strength $strength

EOT
done
done
done