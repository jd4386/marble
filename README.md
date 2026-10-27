# MARBLE

**MARBLE** — Multi-scale linear-time encoder for whole-slide image analysis.

- Paper (IEEE ISBI 2026): [https://doi.org/10.1109/isbi61048.2026.11516046](https://doi.org/10.1109/isbi61048.2026.11516046)

---

## Install

```bash
pip install -e .
# or: pip install -r requirements.txt && PYTHONPATH=src ...
```

This registers the `marble` package and the `marble-train` console script (`python -m marble` also works).

**Notes**

- A CUDA-capable GPU and a CUDA build of `mamba-ssm` (with a matching PyTorch CUDA wheel) are required for training.
- Pins in `requirements.txt` / `pyproject.toml` match a known-working training environment (`torch==2.4.0` with CUDA 12.1). Install the appropriate PyTorch / `mamba-ssm` wheels for your CUDA toolkit if needed.

---

## Feature extraction (external)

This repository **does not** extract patch features.

Produce features **outside** this repo with **UNI** via **CLAM** (or an equivalent pipeline that writes the same `h5` layout). Point MARBLE at the resulting tree with `MULTI_SCALE_FEATURES_DIR` / `--multi_scale_features_dir`.

### Expected `h5` layout

```text
$MULTI_SCALE_FEATURES_DIR/
  <level_dir>/                 # e.g. level_1_5x, level_0_20x, level_1_10x, level_0_40x
    clam_uni_features/
      h5_files/
        <slide_id>.h5             # datasets: features (N, D), coords (N, 2)
```

Also provide label and split CSVs (and an autogen list when required):

| Artifact | Role |
|---|---|
| Labels CSV (`LABELS_CSV` / `--labels_csv`) | Slide-level labels. PANDA: `image_id`, `isup_grade`. NSCLC: `slide_id`, `oncotree_code`. Survival: includes `slide_id`, `survival_months`, `censorship`, etc. |
| Splits (`SPLITS_CSV` / `--splits_path`) | **NSCLC:** a single splits CSV. **Survival:** a **directory** containing `splits_0.csv` … `splits_4.csv`. **PANDA:** optional / ignored — the trainer uses an internal stratified 80/10/10 split (`random_state=42`). |
| Autogen CSV (`AUTOGEN_CSV_PATH` / `--autogen_csv_path`) | CLAM `process_list_autogen.csv` for slide filtering (required for PANDA and TCGA survival; optional for NSCLC) |

---

## Path placeholders / environment variables

Set these before running `scripts/train.sh`, or pass the matching CLI flags / YAML overrides. Use `/path/to/...` placeholders only.

| Env var / CLI flag | Meaning | Example placeholder |
|---|---|---|
| `DATA_ROOT` | Optional convenience root for composing paths (not required by `train.sh`) | `/path/to/data` |
| `LABELS_CSV` / `--labels_csv` | Slide labels CSV (`--labels_dir` is a deprecated alias) | `/path/to/PANDA.csv` |
| `SPLITS_CSV` / `--splits_path` | Splits CSV, or survival splits **directory** (unused for PANDA) | `/path/to/TCGA-NSCLC-split.csv` |
| `MULTI_SCALE_FEATURES_DIR` / `--multi_scale_features_dir` | Feature tree root (layout above) | `/path/to/PANDA/intermediate_results` |
| `AUTOGEN_CSV_PATH` / `--autogen_csv_path` | CLAM autogen list (optional for NSCLC) | `/path/to/.../process_list_autogen.csv` |
| `CONFIG` / `--config` | Paper YAML under `configs/` | `configs/classification/panda_multiscale_10drop.yaml` |
| `--levels_sub_dirs` | Resolution subdirs, coarse → fine | see paper setups below |
| `--scale_factors` | Pixel-size ratios vs base level (same length as `--levels_sub_dirs`) | `1 4` |
| `--model_save_path` / `BEST_MODEL_PATH` | Checkpoint `.pth` under `weights/` (summary CSV → sibling `metrics/`) | `checkpoints/classification/weights/MARBLE_....pth` |
| `--dataset` | `PANDA` \| `TCGA_NSCLC` \| `TCGA_SURVIVAL` | — |
| `--removal_fraction` | Fraction of zero-scale segments to drop | `0.1` (10% drop) |
| `--patch_size` | Patch size in pixels | `256` |
| `--epochs` | Training epochs | `100` (script); paper ~30 + early stop |

### Paper multi-scale setups

| Dataset | Magnifications | Typical `--levels_sub_dirs` | Typical `--scale_factors` |
|---|---|---|---|
| PANDA | 5× / 20× | `level_1_5x level_0_20x` | `1 4` |
| TCGA-NSCLC | 10× / 40× | `level_1_10x level_0_40x` | `1 4` |
| Survival (TCGA-KIRP / LUAD / STAD) | 10× / 40× | `level_1_10x level_0_40x` | `1 4` |

Single-scale ablations use one level only (e.g. `level_1_10x` with `--scale_factors 1`).

Paper training hyperparams (also in YAML configs): `lr=3e-5`, `weight_decay=1e-2`, `warmup_epochs=5`, `patience=10`, `removal_fraction=0.1`.

---

## Training

Training uses distributed launch (`torchrun`) and NCCL/GPU. Install the package first (`pip install -e .`), then from the repo root.

`scripts/train.sh` defaults to the paper main classification run: **PANDA multiscale (5×/20×) with 10% drop**, via `configs/classification/panda_multiscale_10drop.yaml`. Override dataset, config, and paths with the env vars / flags above. Paper configs live under `configs/`.

```bash
export CONFIG=configs/classification/panda_multiscale_10drop.yaml   # or another YAML under configs/
export LABELS_CSV=/path/to/labels.csv
export SPLITS_CSV=/path/to/splits.csv                               # unused for PANDA; dir of splits_*.csv for survival
export MULTI_SCALE_FEATURES_DIR=/path/to/intermediate_results
export AUTOGEN_CSV_PATH=/path/to/.../process_list_autogen.csv       # optional for NSCLC

# interactive (preferred)
bash scripts/train.sh

# Slurm (SBATCH directives live in scripts/sbatch_train.sh)
sbatch scripts/sbatch_train.sh

# or equivalent torchrun launch
torchrun --nproc_per_node=1 --master_port=12350 -m marble \
  --config "$CONFIG" \
  --labels_csv "$LABELS_CSV" \
  --splits_path "$SPLITS_CSV" \
  --multi_scale_features_dir "$MULTI_SCALE_FEATURES_DIR" \
  --autogen_csv_path "$AUTOGEN_CSV_PATH"
```

**Epochs note:** the paper trains for ~30 epochs with early stopping (`patience=10`). Scripts / configs default to `--epochs 100` as a higher ceiling; early stopping still applies.

**Survival multi-fold note:** TCGA survival runs 5 folds and **overwrites** the same `--model_save_path` each fold so published checkpoint basenames stay compatible with this repo’s `checkpoints/**` layout (last fold retained on disk). Per-fold metrics are still appended to the sibling `*_summary.csv`.

---

## Checkpoints

Paper checkpoints (weights + metrics) are under the repo-root `checkpoints/` folder, organized as `classification/`, `survival/`, and `ablations/` (each with `weights/` and `metrics/`). Matching paper configs are in `configs/`.

---

## Repository layout

```text
MARBLE/
  README.md, pyproject.toml, requirements.txt
  main.py                       # deprecated shim → marble.cli
  configs/{classification,survival,ablations}/*.yaml
  scripts/train.sh
  scripts/sbatch_train.sh
  checkpoints/{classification,survival,ablations}/{weights,metrics}/
  src/marble/
    __init__.py, __main__.py
    cli.py, config.py, engine.py
    model/marble.py             # MarbleClassifier / MarbleModel
    data/{wsi_dataset,preprocess,labels}.py
    train/{scheduler,loaders,classification,survival,losses}.py
```

---

## Citation

If you use this code or the MARBLE model in your research, please cite:

```bibtex
@inproceedings{dwarampudi2026marble,
  title     = {A Multi-Scale Linear-Time Encoder for Whole-Slide Image Analysis},
  author    = {Dwarampudi, Jagan Mohan Reddy and Wong, Joshua and Van Nguyen, Hien and Banerjee, Tania},
  booktitle = {2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026},
  doi       = {10.1109/ISBI61048.2026.11516046}
}
```
