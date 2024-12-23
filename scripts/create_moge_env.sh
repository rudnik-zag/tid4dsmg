#!/bin/bash

# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"

# Create a new conda environment named 'moge' with Python 3.10 and Anaconda
conda create -n moge python=3.10 anaconda -y

# Activate the newly created environment
conda activate moge

# Install the specified packages in the activated environment
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt