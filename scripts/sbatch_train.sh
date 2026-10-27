#!/bin/bash
#SBATCH -J marble_panda_multiscale_10drop   # Format: marble_<dataset>
#SBATCH -o logs/%x.%j.out
#SBATCH -N 1
#SBATCH --gres=gpu:1
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 240:00:00
#SBATCH --mem=64G

# Slurm wrapper: submits training via scripts/train.sh.
# Export the same env vars as for interactive runs (LABELS_CSV, …), then:
#   sbatch scripts/sbatch_train.sh
#
# Or:
#   CONFIG=configs/classification/tcga_nsclc_multiscale_10drop.yaml \
#   LABELS_CSV=... SPLITS_CSV=... MULTI_SCALE_FEATURES_DIR=... \
#   sbatch scripts/sbatch_train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "${REPO_ROOT}/logs"

bash "${REPO_ROOT}/scripts/train.sh"
