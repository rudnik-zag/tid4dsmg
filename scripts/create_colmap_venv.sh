#!/bin/bash

# Check if any arguments were passed
NO_ARGS=1
CUDA_SUPPORT=0
ADD_DEPS=0
if [ $# -eq 0 ]; then
    echo "No arguments provided."
    NO_ARGS=0
    echo "NO_ARGS: ${NO_ARGS:-'Not provided'}"
fi

# Loop through all arguments and assign them to variables
if [ "$NO_ARGS" -ne 0 ]; then
    for i in "$@"; do
        # Assign each argument to a variable based on its position
        case $i in
            # Assign the first argument to NAME
            -cs)
                CUDA_SUPPORT="$2"
                shift 2  # Shift past the name argument and its value
                ;;
            # Assign the second argument to AGE
            -ad)
                ADD_DEPS="$2"
                shift 2  # Shift past the age argument and its value
                ;;
        esac
    done
fi

# Output the assigned variables
echo "CUDA_SUPPORT: ${CUDA_SUPPORT:-'Not provided'}"
echo "ADD_DEPS: ${ADD_DEPS:-'Not provided'}"

# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"

# Create a new conda environment named 'colmap_env' with Python 3.10 and Anaconda
conda create -n colmap_env python=3.10 anaconda -y

# Activate the newly created environment
conda activate colmap_env

# Install the specified packages in the activated environment
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Need to have proper compiler version for specific cuda version
# CUDA 11.8 need gcc and g++ 11.x version
# sudo update-alternatives --config gcc
# sudo update-alternatives --config g++


# If you did not 
if [ "$ADD_DEPS" -ne 0 ]; then
    sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
fi
# # To compile with CUDA support, also install Ubuntu’s default CUDA package: - ako nije vec instalirana

if [ "$CUDA_SUPPORT" -ne 0 ]; then
sudo apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc
fi

# cd external/colmap

# Configure and compile COLMAP:

cd external/colmap
mkdir build
cd build
cmake .. -GNinja
ninja
sudo ninja install

pip install ./pycolmap/