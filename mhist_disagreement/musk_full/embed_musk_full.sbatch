#!/bin/bash
#SBATCH --job-name=musk-full
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Embed MHIST with MUSK's full features on one GPU. Submit from musk_full/:
#
#   sbatch embed_musk_full.sbatch <annotations.csv> <images dir> [extra args]
#
# e.g. sbatch embed_musk_full.sbatch ~/mhist/annotations.csv ~/images
#      sbatch embed_musk_full.sbatch ~/mhist/annotations.csv ~/images --variants ms
#
# Writes ../embeddings_full/musk_{ms,ms_tta,retrieval}.csv. Rerun to resume.
# Estimate: a few minutes for ms + retrieval, ~10-15 min more for ms_tta.
set -euo pipefail
mkdir -p logs

module load pytorch-conda/2.8
unset PYTHONNOUSERSITE   # the module hides pip --user installs (musk, fairscale)

ANNOTATIONS="$1"; IMAGES="$2"; shift 2
nvidia-smi --query-gpu=name,memory.total --format=csv
python embed_musk_full.py --annotations "$ANNOTATIONS" --images "$IMAGES" "$@"
