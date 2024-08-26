#!/bin/bash
#SBATCH --job-name=training_array
#SBATCH --output=training_%A_%a.out
#SBATCH --error=training_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=soroush1@yorku.ca
#SBATCH --array=0-8  # This creates 9 jobs, one for each config file

# Array of config files
declare -a configs=(
    "alexnet_config.yaml"
    "efficientnet_v2_s_config.yaml"
    "resnet18_config.yaml"
    "resnet50_config.yaml"
    "resnet101_config.yaml"
    "vgg16_config.yaml"
    "vgg19_config.yaml"
    "vit_b_16_config.yaml"
    "vit_b_32_config.yaml"
)

# Get the current config file
current_config=${configs[$SLURM_ARRAY_TASK_ID]}

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

echo "Running job for config: $current_config"

torchrun --nproc_per_node=2 --node_rank=0 --master_addr="localhost" --master_port=1234 \
    /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/scripts/trainings/classification.py \
    --config /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/config/clf/${current_config}