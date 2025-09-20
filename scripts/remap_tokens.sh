#!/bin/bash
# Launch scripts/remape_tokens.py with SLURM.

SRUN_ARGS=()
SRUN_ARGS+=(--nodes=1 --ntasks=1 --cpus-per-task=64 --mem=256G)
SRUN_ARGS+=(--unbuffered --output=$PWD/checkpoints/remap_tokens.out)

source scripts/activate.sh
srun ${SRUN_ARGS[@]} python3 -u scripts/remap_tokens.py
