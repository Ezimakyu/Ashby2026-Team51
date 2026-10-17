#!/bin/bash
#SBATCH --job-name=mhist-embed
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Fallback when the hosted server is down (or after it stops Sun 6 pm):
# embed MHIST on one GPU with the ncsa repo's run_embeddings.py. Uses your own
# GPU-hours, so prefer embed_server.py. Submit from mhist_disagreement/:
#
#   sbatch slurm/embed_gpu.sbatch <mhist images dir> embeddings/musk.csv musk
#
# Needs a clone of https://github.com/ncsa/ashby-hackathon-2026 (set
# NCSA_REPO if it isn't at ~/ashby-hackathon-2026), its one-time setup
# (open_clip_torch, conch/musk packages via pip install --user), and
# `hf auth login` with an account approved for the gated model.
set -euo pipefail
mkdir -p logs

IMAGES="$1"
OUT="$(realpath -m "$2")"
MODEL="${3:-musk}"
NCSA_REPO="${NCSA_REPO:-$HOME/ashby-hackathon-2026}"

module load pytorch-conda/2.8
unset PYTHONNOUSERSITE   # the module hides pip --user installs (open_clip, conch, musk)

mkdir -p "$(dirname "$OUT")"
cd "$NCSA_REPO"
python run_embeddings.py --model "$MODEL" --input-dir "$IMAGES" --output "$OUT"
