#!/bin/bash
#SBATCH --job-name=mhist-probe
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Train and evaluate every probe on CPU. Submit from mhist_disagreement/:
#
#   sbatch slurm/probe_cpu.sbatch <annotations.csv> musk=embeddings/musk.csv \
#       [--zero-shot musk=embeddings/musk_zeroshot.csv] [more probe.py args]
#
# Everything after the annotations path is passed to probe.py; bare tag=path
# arguments are treated as --embeddings. Results land in results/<tag>/.
set -euo pipefail
# logs/ is committed (logs/.gitkeep): Slurm opens --output before this script runs.
# Cap BLAS/OpenMP/torch threads at our allocation; by default they size
# themselves to the whole node and oversubscribe the cores we were given.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS

module load pytorch-conda/2.8   # numpy, pandas, scikit-learn, scipy, matplotlib, torch

ANNOTATIONS="$1"; shift
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --*) ARGS+=("$1" "$2"); shift 2 ;;
        *)   ARGS+=(--embeddings "$1"); shift ;;
    esac
done

python probe.py --annotations "$ANNOTATIONS" "${ARGS[@]}"
