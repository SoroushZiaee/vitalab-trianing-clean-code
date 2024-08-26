#!/bin/bash

#SBATCH --account=def-kohitij
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=imagenet_processing
#SBATCH --output=%x-%j.out
#SBATCH --errot=%x-%j.err


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

# Set path variables
IMAGENET_ROOT="/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/imagenet"
SCRIPT_DIR="/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/scripts/test_imagenet_class.py"

# Activate your virtual environment if you're using one
# source /path/to/your/venv/bin/activate

# Run your Python script that uses the ImageNet class
python $SCRIPT_DIR --root $IMAGENET_ROOT --temp_extract
# /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/notebooks/lab.sh


echo "ImageNet processing complete!"