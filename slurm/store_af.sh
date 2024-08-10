#!/bin/bash
#SBATCH --job-name=store
#SBATCH --output=/scratch/prj/formalpaca/aasgd/logs/%A/%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --chdir /scratch/prj/formalpaca/aasgd

echo "Starting task $SLURM_ARRAY_TASK_ID"

# Directories as variables
AF_DIR="/scratch/prj/formalpaca/aasgd/AFs"
OUT_DIR="/scratch/prj/formalpaca/aasgd/apx_dir"
ENV_DIR="/scratch/prj/formalpaca/aasgd/env"
SCRIPT_PATH="/scratch/prj/formalpaca/aasgd/store_af.py"

# Load the python module
module load python

# Source the environment
source $ENV_DIR/bin/activate

# Select a single file corresponding to the TASK_ID from AF_DIR
FILE=$(ls $AF_DIR/*.tgf | sed -n "${SLURM_ARRAY_TASK_ID}p")

# Run the python script with the filename as the first argument and OUT_DIR as the second
python $SCRIPT_PATH $FILE $OUT_DIR

# Deactivate the environment
deactivate

echo "Finished task $SLURM_ARRAY_TASK_ID"