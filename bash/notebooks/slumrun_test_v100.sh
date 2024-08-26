#!/bin/bash
#SBATCH --job-name=v100_jupyter
#SBATCH --output=result_v100_1_1.out
#SBATCH --error=error_v100_1_1.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:v100:8
#SBATCH --mem=50G
#SBATCH --constraint=cascade,v100
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun ./bash/lab.sh