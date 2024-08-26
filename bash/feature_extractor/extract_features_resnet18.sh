#!/bin/bash
#SBATCH --job-name=extract_features_resnet18
#SBATCH --output=extract_features_resnet18.out
#SBATCH --error=extract_features_resnet18.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
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

pip freeze

#srun python extract_layer_features.py --model resnet18 --task lamem
# srun python extract_layer_features.py --model resnet18 --task imagenet
# srun python extract_layer_features.py --model resnet18 --task lamem_shuffle
# srun python extract_layer_features.py --model resnet18 --task lamem_pretrain_freeze
# srun python extract_layer_features.py --model resnet18 --task lamem_pretrain_no_freeze
# srun python extract_layer_features.py --model resnet18 --task lamem_shuffle_pretrain_freeze



