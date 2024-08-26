#!/bin/bash

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

python /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/calculate_cka_copy.py --model_name resnet18 resnet50 resnet101 vgg19 --src_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/pool_layers_pkl --dst_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/cka_pool_layers

