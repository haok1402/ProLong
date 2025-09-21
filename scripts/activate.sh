#!/bin/bash

# Activate the conda environment.
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ProLong

# Use CUDA 11.8 for compatibility with the codebase.
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Use cluster-wide cache directory for HuggingFace models.
export HF_HUB_CACHE=/data/hf_cache

# Setup directories for datasets and checkpoints.
DATASET_LOCAL=/data/group_data/cx_group/ProLong/datasets
DATASET_MOUNT=$PWD/datasets

mkdir -p $DATASET_LOCAL
if [ ! -e $DATASET_MOUNT ]; then
    ln -s $DATASET_LOCAL $DATASET_MOUNT
fi

CHECKPOINTS_LOCAL=/data/group_data/cx_group/ProLong/checkpoints
CHECKPOINTS_MOUNT=$PWD/checkpoints

mkdir -p $CHECKPOINTS_LOCAL
if [ ! -e $CHECKPOINTS_MOUNT ]; then
    ln -s $CHECKPOINTS_LOCAL $CHECKPOINTS_MOUNT
fi
