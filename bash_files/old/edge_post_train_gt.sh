#!/bin/bash

# for temp in "0" "0.5" "1"
# do
for var in "$@"
do
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p seas_gpu
#SBATCH --constraint="a100"
#SBATCH --job-name=gt-$var-post-train
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/gt-infer_$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/gt-infer_$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 edge_post_training_gt.py -s gt_edges_unif -t "0" -l $var

EOT
done
# done