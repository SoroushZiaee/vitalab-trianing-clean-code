#!/bin/bash
#SBATCH --job-name=mean_and_std
#SBATCH --output=mean_and_std.out
#SBATCH --error=mean_and_std.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50G

echo "Starting job script..."
srun python /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/generate_mean_and_std_of_lamem.py