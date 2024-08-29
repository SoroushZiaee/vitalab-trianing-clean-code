#!/bin/bash
#SBATCH --job-name=combine_array
#SBATCH --output=combine_%A_%a.out
#SBATCH --error=combine_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=20G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=soroush1@yorku.ca
#SBATCH --array=0-8  # This creates 9 jobs, one for each config file

# Array of combined config files
declare -a configs=(
    "combine_alexnet_config.yaml"
    "combine_efficientnet_b0_config.yaml"
    "combine_resnet18_config.yaml"
    "combine_resnet50_config.yaml"
    "combine_resnet101_config.yaml"
    "combine_vgg16_config.yaml"
    "combine_vgg19_config.yaml"
    "combine_vit_b_16_config.yaml"
    "combine_vit_b_32_config.yaml"
)

# Get the current config file
current_config=${configs[$SLURM_ARRAY_TASK_ID]}
# Calculate a unique port number for each job
master_port=$((12345 + $SLURM_ARRAY_TASK_ID))

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

echo "Running combined task job for config: $current_config"

torchrun --nproc_per_node=4 --node_rank=0 --master_addr="localhost" --master_port=$master_port \
    /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/scripts/trainings/combine.py \
    --config /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/config/combine/${current_config}