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

for node in "attn" "mlp"
do
for win in 0 2 4
do

COMMAND1="python3 causal_tracing.py $node $win"
echo $COMMAND1

COMMAND2="python3 causal_tracing_eval.py $node $win"
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

if $COMMAND1; then
    $COMMAND2
else
    echo "failure so did not run eval"
fi

EOT


# if $COMMAND1; then
#     $COMMAND2
# else
#     echo "failure so did not run post training"
# fi
# $COMMAND2

done
done