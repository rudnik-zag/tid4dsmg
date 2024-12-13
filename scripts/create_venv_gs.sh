#!/bin/bash

# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"

# Create a new conda environment named 'gaussian_splatting' with Python 3.10 and Anaconda
conda create -n gaussian_splatting python=3.10 anaconda -y

# Activate the newly created environment
conda activate gaussian_splatting

# Install the specified packages in the activated environment
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Need to have proper compiler version for specific cuda version
# CUDA 11.8 need gcc and g++ 11.x version
# sudo update-alternatives --config gcc
# sudo update-alternatives --config g++
cd external/gspalt

pip install -e . # build gspalt, go to dir