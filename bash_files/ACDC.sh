#!/bin/bash

DATASET=$1
echo $1
shift
ACDC_TYPE=$1
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
#SBATCH --job-name=$var-acdc-post-train
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/$ACDC_TYPE-$DATASET-$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/$ACDC_TYPE-$DATASET-$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 edge_post_training.py -d $DATASET -t "0" -n $ACDC_TYPE -l $var

EOT
done