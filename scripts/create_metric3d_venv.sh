#!/bin/bash

# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"

# Create a new conda environment named 'metric3d' with Python 3.10 and Anaconda
conda create -n metric3d python=3.10 anaconda -y

# Activate the newly created environment
conda activate metric3d

# Install the specified packages in the activated environment
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 xformers --index-url https://download.pytorch.org/whl/cu118

pip install -r requiremetns_v2.txt

