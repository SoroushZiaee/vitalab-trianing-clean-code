#!/bin/bash
#SBATCH --job-name=classification_array
#SBATCH --output=classification_%A_%a.out
#SBATCH --error=classification_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=20G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=soroush1@yorku.ca
#SBATCH --array=0-1  # This creates 9 jobs, one for each config file

# Array of classification config files
# declare -a configs=(
#     "classification_alexnet_config.yaml"
#     "classification_efficientnet_b0_config.yaml"
#     "classification_resnet18_config.yaml"
#     "classification_resnet50_config.yaml"
#     "classification_resnet101_config.yaml"
#     "classification_vgg16_config.yaml"
#     "classification_vgg19_config.yaml"
#     "classification_vit_b_16_config.yaml"
#     "classification_vit_b_32_config.yaml"
# )

declare -a configs=(
    "classification_vit_b_16_config.yaml"
    "classification_vit_b_32_config.yaml"
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

echo "Running classification job for config: $current_config"
echo "Using master_port: $master_port"

python /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/scripts/prepare_server/datamodule_imagenet_preparation.py

torchrun --nproc_per_node=2 --node_rank=0 --master_addr="localhost" --master_port=$master_port \
    /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/scripts/trainings/classification.py \
    --config /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/config/clf_rnd/${current_config}