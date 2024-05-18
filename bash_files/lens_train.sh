#!/bin/bash
PARTITION=$1
echo $1
shift

MODEL=$1
echo $1
shift

COMMANDS=""
for file in $@
do
COMMANDS+="python3 lens_$file.py $MODEL; "
done

echo $COMMANDS

if [ $PARTITION == "gpu" ] 
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
#SBATCH --job-name=$1-lens
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-18:00
#SBATCH -o prog_files/$MODEL-lens-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/$MODEL-lens-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2

$COMMANDS

EOT