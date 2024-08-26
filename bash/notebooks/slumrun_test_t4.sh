#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=result_t4_1_1.out
#SBATCH --error=error_t4_1_1.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:t4:4
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun ./bash/lab.sh