#!/bin/bash
#SBATCH --job-name=musk-tokens
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Extract MUSK intermediate-layer features and pooled patch grids. Submit from
# musk_unsummarized/:
#
#   sbatch extract.sbatch <annotations.csv> <images dir> [extra args]
#
# e.g. sbatch extract.sbatch ~/mhist/annotations.csv ~/images
#      sbatch extract.sbatch ~/mhist/annotations.csv ~/images --grid-layers 18,24
#
# Writes ../embeddings_full/musk_tokens.npz and ../embeddings_full/model_structure.txt.
# One forward pass per image, so expect only a few minutes.
set -euo pipefail
mkdir -p logs

module load pytorch-conda/2.8
unset PYTHONNOUSERSITE   # the module hides pip --user installs (musk, fairscale)

ANNOTATIONS="$1"; IMAGES="$2"; shift 2
nvidia-smi --query-gpu=name,memory.total --format=csv
python extract_musk_tokens.py --annotations "$ANNOTATIONS" --images "$IMAGES" "$@"
