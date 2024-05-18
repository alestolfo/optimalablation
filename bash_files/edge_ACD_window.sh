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
DATASET=$1
echo $1
shift
RUN_NAME=$1
echo $1
shift
echo $@

for window in "0.5" "0.99"
do

if [ $window == "0.5" ]; then
maxwindow="1.5"
elif [ $window == "0.99" ]; then
maxwindow="1.01"
else
maxwindow="2"
fi

echo $minwindow 
echo $maxwindow

for var in "$@"
do
sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p $PARTITION
$CONSTRAINT
#SBATCH --job-name=eACD_$DATASET-$var
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t 0-18:00
#SBATCH -o prog_files/eACD_$DATASET-$var-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/eACD_$DATASET-$var-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2

if python3 edge_pruning$ACD_TYPE.py -d $DATASET -l $var -n $RUN_NAME-$window --minwindow $window --maxwindow $maxwindow; then
    python3 edge_post_training.py -d $DATASET -n $RUN_NAME-$window -t "0" -l $var
else
    echo "failure so did not run post training"
fi

EOT
echo $window $var
done
done