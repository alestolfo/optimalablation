#!/bin/bash

PARTITION=$1
echo $1
shift

if [ $PARTITION == "gpu" ] 
then
CONSTRAINT=""
else
CONSTRAINT='#SBATCH --constraint="a100"'
fi

ABLATION=$1
echo $1
shift
DATASET=$1
echo $1
shift
MINW=$1
echo $1
shift
MAXW=$1
echo $1
shift
NCIRC=$1
echo $1
shift

for var in "$@"
do

sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p $PARTITION
$CONSTRAINT
#SBATCH --job-name=c-random-$DATASET-$var
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/c-random-$DATASET-$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/c-random-$DATASET-$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2

python3 circuits_random.py -d $DATASET -e $ABLATION --minwindow $MINW --maxwindow $MAXW -t $NCIRC

EOT

done