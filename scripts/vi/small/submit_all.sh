#!/bin/bash

#SBATCH --job-name=fr-all       # create a short name for your job
#SBATCH --output=./gpu_output/Train_whisper-small-vietmed-all-submission-JOB_ID_%j-%N.log # create output file

#SBATCH --nodes=1                  # node count
#SBATCH --ntasks-per-node=1        #CPUS per node (how many cpu to use withinin 1 node)
#SBATCH --mem=250G
#SBATCH --time=100:00:00               # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu1 --gres=gpu  # number of gpus per node

# List of job scripts to submit
job_scripts=(
    "vietmed-freeE-D0-8.sh"
    "vietmed-freeE-D3-11.sh"
    "vietmed-freeE0-8-D0-8.sh"
    "vietmed-freeE0-8-D3-11.sh"
    "vietmed-freeE0-8.sh"
    "vietmed-freeE3-11.sh"
)

JOB_DIR="scripts/vi/small/"

for job_script in "${job_scripts[@]}"; do
    full_path="$JOB_DIR$job_script"
    echo "$full_path"
    # submit all files
    sbatch "$full_path"
done
