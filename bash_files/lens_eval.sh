#!/bin/bash
PARTITION=$1
echo $1
shift

MODEL=$1
echo $1
shift

COMMANDS=(
    # "vanilla"
    "proj_rand"
    "steer_rand"
    "project tuned"
    "project linear_oa"
    "steer tuned"
    "steer linear_oa"
    "resample tuned"
    "resample linear_oa"
)
TIMELIMITS=(
    # "0-12:00" # "vanilla"
    "0-12:00" # "proj_rand"
    "0-12:00" # "steer_rand"
    "1-00:00" # "project tuned"
    "1-00:00" # "project linear_oa"
    "1-00:00" # "steer tuned"
    "1-00:00" # "steer linear_oa"
    "0-12:00" # "resample tuned"
    "0-12:00" # "resample linear_oa"
)

echo $COMMANDS

if [ $PARTITION == "gpu" ]  
then
CONSTRAINT=""
else
CONSTRAINT='#SBATCH --constraint="a100"'
fi

for i in ${!COMMANDS[@]}
do

COMMANDSPEC=${COMMANDS[$i]}
COMMAND="python3 lens_compare.py $MODEL $COMMANDSPEC" 
echo $COMMAND

CNAME=${COMMANDSPEC// /-}
echo $CNAME

TIMELIMIT=${TIMELIMITS[$i]}
echo $TIMELIMIT

sbatch <<EOT
#!/bin/bash
#SBATCH -c 1
#SBATCH -p $PARTITION
$CONSTRAINT
#SBATCH --job-name=$CNAME-$MODEL-lens
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -t $TIMELIMIT
#SBATCH -o prog_files/$MODEL-lens-%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e prog_files/$MODEL-lens-%j.err  # File to which STDERR will be written, %j inserts jobid

# Your commands here
module load Anaconda2
conda activate take2

$COMMAND

EOT
done