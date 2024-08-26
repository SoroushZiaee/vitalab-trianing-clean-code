#!/bin/bash

# Set up the environment
module load StdEnv/2020 python/3.9.6 ipykernel/2022a gcc/9.3.0 cuda/11.4 opencv/4.5.5 scipy-stack/2022a hdf5/1.12.1

# Set up the virtual environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install --no-cache-dir -r requirements.txt

echo "Give access to all users to the virtual environment"
chmod -R 777 venv/