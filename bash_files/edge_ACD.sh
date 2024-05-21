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

ACD_TYPE=$1
echo $1
shift
ABLATION=$1
echo $1
shift
DATASET=$1
echo $1
shift
RUN_NAME=$1
echo $1
shift
echo $@

if [ $RUN_NAME == "unif_dynamic" ] 
then
WINDOW="--window"
else
WINDOW=""
fi

for var in "$@"
do

COMMAND1="python3 edge_pruning$ACD_TYPE.py -d $DATASET -l $var -n $RUN_NAME -e $ABLATION $WINDOW"
echo $COMMAND1

COMMAND2="python3 edge_post_training.py -d $DATASET -n $RUN_NAME -e $ABLATION -t "0.5" -l $var"
echo $COMMAND2

sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p $PARTITION
$CONSTRAINT
#SBATCH --job-name=eACD_$DATASET-$var
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-12:00
#SBATCH -o prog_files/eACD_$DATASET-$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/eACD_$DATASET-$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2

$COMMAND1

EOT


# if $COMMAND1; then
#     $COMMAND2
# else
#     echo "failure so did not run post training"
# fi
# $COMMAND2

done