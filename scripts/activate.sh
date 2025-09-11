#!/bin/bash

# Specify the dataset mounting paths.
DATASET_LOCAL=/scratch/$USER/ProLong/datasets
DATASET_MOUNT=$PWD/datasets

# Mount from local SSD to current workspace.
mkdir -p $DATASET_LOCAL
if [ ! -e $DATASET_MOUNT ]; then
    ln -s $DATASET_LOCAL $DATASET_MOUNT
fi

# Download the datasets if not already present.
if [ ! -d $DATASET_MOUNT/long-context-65536 ]; then
    git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-64K datasets/long-context-65536
fi
if [ ! -d $DATASET_MOUNT/long-context-524288 ]; then
    git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-512K datasets/long-context-524288
fi
if [ ! -d $DATASET_MOUNT/prolong-ultrachat-64K ]; then
    git clone https://huggingface.co/datasets/princeton-nlp/prolong-ultrachat-64K datasets/prolong-ultrachat-64K
fi
