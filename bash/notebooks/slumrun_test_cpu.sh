#!/bin/bash
#SBATCH --job-name=cpu
#SBATCH --output=cpu.out
#SBATCH --error=cpu.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --mem=40G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun ./bash/lab.sh
