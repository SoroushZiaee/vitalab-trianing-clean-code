#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=training.out
#SBATCH --error=training.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=20G
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

echo "Env has been set up"
pip freeze

torchrun --nproc_per_node=2 --node_rank=0 --master_addr="localhost" --master_port=1234 /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/scripts/trainings/classification.py --config /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/config/clf_training_resnet18.yaml