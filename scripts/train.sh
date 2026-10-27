#!/bin/bash
# Interactive / non-Slurm training launcher for MARBLE.
# Defaults to the paper main run: PANDA multiscale (5×/20×) with 10% drop.
#
# Example:
#   export LABELS_CSV=/path/to/PANDA.csv
#   export MULTI_SCALE_FEATURES_DIR=/path/to/PANDA/intermediate_results
#   export AUTOGEN_CSV_PATH=/path/to/.../process_list_autogen.csv
#   bash scripts/train.sh
#
# Optional:
#   CONFIG=configs/classification/panda_multiscale_10drop.yaml
#   DATA_ROOT=/path/to/data          # convenience only; not required by this script
#   SPLITS_CSV=/path/to/splits.csv   # required for NSCLC / survival; unused for PANDA
#   DATASET, LEVELS_SUB_DIRS, SCALE_FACTORS, REMOVAL_FRACTION, BEST_MODEL_PATH, EPOCHS, …

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "${REPO_ROOT}/logs"

nvidia-smi || true

# Path placeholders — export before running, or edit below.
# DATA_ROOT is optional convenience for composing other paths; this script does not use it directly.
DATA_ROOT="${DATA_ROOT:-}"
LABELS_CSV="${LABELS_CSV:?Set LABELS_CSV to your slide-level labels CSV}"
MULTI_SCALE_FEATURES_DIR="${MULTI_SCALE_FEATURES_DIR:?Set MULTI_SCALE_FEATURES_DIR to multi-scale feature root}"
AUTOGEN_CSV_PATH="${AUTOGEN_CSV_PATH:-}"
# Optional for PANDA (ignored by the trainer); required for TCGA_NSCLC / TCGA_SURVIVAL.
SPLITS_CSV="${SPLITS_CSV:-}"

CONFIG="${CONFIG:-configs/classification/panda_multiscale_10drop.yaml}"
DATASET="${DATASET:-}"
PATCH_SIZE="${PATCH_SIZE:-}"
SCALE_FACTORS="${SCALE_FACTORS:-}"
LEVELS_SUB_DIRS="${LEVELS_SUB_DIRS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
REMOVAL_FRACTION="${REMOVAL_FRACTION:-}"
BEST_MODEL_PATH="${BEST_MODEL_PATH:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CMD=(
  torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-12350}"
  -m marble
  --config "$CONFIG"
  --labels_csv "$LABELS_CSV"
  --multi_scale_features_dir "$MULTI_SCALE_FEATURES_DIR"
)

if [[ -n "$AUTOGEN_CSV_PATH" ]]; then
  CMD+=(--autogen_csv_path "$AUTOGEN_CSV_PATH")
fi
if [[ -n "$SPLITS_CSV" ]]; then
  CMD+=(--splits_path "$SPLITS_CSV")
fi
if [[ -n "$DATASET" ]]; then
  CMD+=(--dataset "$DATASET")
fi
if [[ -n "$PATCH_SIZE" ]]; then
  CMD+=(--patch_size "$PATCH_SIZE")
fi
if [[ -n "$SCALE_FACTORS" ]]; then
  # shellcheck disable=SC2206
  CMD+=(--scale_factors $SCALE_FACTORS)
fi
if [[ -n "$LEVELS_SUB_DIRS" ]]; then
  # shellcheck disable=SC2206
  CMD+=(--levels_sub_dirs $LEVELS_SUB_DIRS)
fi
if [[ -n "$BATCH_SIZE" ]]; then
  CMD+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "$EPOCHS" ]]; then
  CMD+=(--epochs "$EPOCHS")
fi
if [[ -n "$REMOVAL_FRACTION" ]]; then
  CMD+=(--removal_fraction "$REMOVAL_FRACTION")
fi
if [[ -n "$BEST_MODEL_PATH" ]]; then
  CMD+=(--model_save_path "$BEST_MODEL_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
