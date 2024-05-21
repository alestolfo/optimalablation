#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --constraint="a100"
#SBATCH --job-name=abl_loss_$1
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/abl-$1-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/abl-$1-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2
python3 ablation_loss.py -e oca -d $1
python3 ablation_loss.py -e oca -d $1

EOT