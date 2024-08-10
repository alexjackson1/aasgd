#!/bin/bash
#SBATCH --job-name=solve
#SBATCH --output=/scratch/prj/formalpaca/aasgd/logs/%A/%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --chdir /scratch/prj/formalpaca/aasgd
#SBATCH --array=0-5789
#SBATCH --time=2-00:00:00

echo "Starting task $SLURM_ARRAY_TASK_ID"

# Directories as variables
AF_DIR="/scratch/prj/formalpaca/aasgd/AFs"
APX_DIR="/scratch/prj/formalpaca/aasgd/apx_dir"
OUT_DIR="/scratch/prj/formalpaca/aasgd/full_dataset"
ENV_DIR="/scratch/prj/formalpaca/aasgd/env"
SCRIPT_PATH="/scratch/prj/formalpaca/aasgd/single_solve.py"

# Load the python module
module load python

# Source the environment
source $ENV_DIR/bin/activate

# Select a single file corresponding to the TASK_ID from AF_DIR
FILE=$(ls $AF_DIR/*.tgf | sed -n "$(( ($SLURM_ARRAY_TASK_ID - 1) / 6 + 1 ))p")

# Define the possible values for the parameter
PARAMS=("GR" "PR" "ST" "SST" "STG" "CO")

# Determine the index for the PARAMS array
PARAM_INDEX=$(( ($SLURM_ARRAY_TASK_ID - 1) % 6 ))

# if not 0 then exit 
if [ $PARAM_INDEX -ne 0 ]; then
    exit
fi

# Get the value from the PARAMS array
PARAM=${PARAMS[$PARAM_INDEX]}

# Run the python script with the filename as the first argument, OUT_DIR as the second, and the selected PARAM
python $SCRIPT_PATH $FILE $APX_DIR $OUT_DIR $PARAM 1

# Deactivate the environment
deactivate

echo "Finished task $SLURM_ARRAY_TASK_ID"