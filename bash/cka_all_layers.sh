#!/bin/bash
#SBATCH --job-name=cka_all_layers
#SBATCH --output="cka_all_layers.out"
#SBATCH --error="cka_all_layers.err"
#SBATCH --array=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:5:00
#SBATCH --mem=10G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

echo "Start Installing and setup env"
source /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/prepare_env/setup_env_node.sh

module list
pip freeze

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

echo "Installing requirements"
pip install --no-index -r requirements.txt

echo "Env has been set up"

srun /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/cka_all_layers.py