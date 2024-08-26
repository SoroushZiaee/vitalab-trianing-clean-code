#!/bin/bash
#SBATCH --job-name=extraction
#SBATCH --output="extraction_h5%a.out"
#SBATCH --error="extraction_h5%a.err"
#SBATCH --array=1-9
#SBATCH --cpus-per-task=5
#SBATCH --time=00:15:00
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

srun python pickle_to_h5_pooling_layers.py --model_names resnet50 resnet101  --dst_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/pool_layers_h5 --src_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/pool_layers_pkl