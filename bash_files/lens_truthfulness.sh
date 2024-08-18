#!/bin/bash

PARTITION=$1
echo $1
shift

if [ $PARTITION == "gpu" ] | [ $PARTITION == "gpu_test" ]
then
CONSTRAINT=""
else
CONSTRAINT='#SBATCH --constraint="a100"'
fi

sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p $PARTITION
$CONSTRAINT
#SBATCH --job-name=lens-truth
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/lens-truth-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/lens-truth-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2

python3 lens_truthfulness.py

EOT