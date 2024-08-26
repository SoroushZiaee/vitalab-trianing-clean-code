#!/bin/bash
#SBATCH --job-name=multinodes
#SBATCH --output="multinodes-%A_%a.out"
#SBATCH --error=multinodes.err
#SBATCH --array=1-5
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:30
#SBATCH --mem=1G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

echo "Hello from the submit_multi_nodes.sh script running on $SLURM_ARRAY_TASK_ID"