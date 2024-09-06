#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --output=tensorboard.out
#SBATCH --error=tensorboard.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=6:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

echo "Start Installing and setup env"
source /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/bash/prepare_env/setup_env_node.sh

module list

pip freeze

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

echo "Installing requirements"
pip install --no-index -r requirements.txt
pip install --no-index tensorboard

echo "Env has been set up"

pip freeze

srun /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/bash/tensorboard/tb.sh /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/experiments