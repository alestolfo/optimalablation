#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p seas_gpu
#SBATCH --constraint="a100"
#SBATCH --job-name=$1-lens
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/$1-lens-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/$1-lens-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 lens_$1.py

EOT