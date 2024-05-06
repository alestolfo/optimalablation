#!/bin/bash

# for temp in "0" "0.5" "1"
# do
RUN_NAME=$1
echo $1
shift
echo $@

for var in "$@"
do
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p seas_gpu
#SBATCH --constraint="a100"
#SBATCH --job-name=$var-post-train
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/postrain_$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/postrain_$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 edge_post_training.py -d ioi -n $RUN_NAME -t "0" -l $var

EOT
done