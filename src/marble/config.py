"""TrainConfig dataclass and YAML / CLI merge helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional, Sequence

import yaml


# Paper hyperparams used by the training loops (also hardcoded at call sites in engine).
PAPER_BASE_LR = 3e-5
PAPER_WEIGHT_DECAY = 1e-2
PAPER_WARMUP_EPOCHS = 5
PAPER_PATIENCE = 10


@dataclass
class TrainConfig:
    """Explicit training configuration (replaces global argparse Namespace).

    Defaults match ``scripts/train.sh`` / the paper PANDA multiscale 10% drop setup.
    """

    labels_csv: str = "data/labels.csv"
    multi_scale_features_dir: str = "data/results"
    patch_size: int = 256
    scale_factors: List[float] = None  # type: ignore[assignment]
    levels_sub_dirs: List[str] = None  # type: ignore[assignment]
    max_children: Optional[int] = None
    num_workers: int = 0
    batch_size: int = 1
    embedding_dim: int = 1024
    epochs: int = 100
    model_save_path: str = "checkpoints/classification/weights/MARBLE_PANDA_multiscale_10drop.pth"
    dataset: str = "PANDA"
    autogen_csv_path: str = ""
    splits_path: str = ""
    removal_fraction: float = 0.1
    # Paper training hyperparams (overridable via YAML; engine passes them through).
    base_lr: float = PAPER_BASE_LR
    weight_decay: float = PAPER_WEIGHT_DECAY
    warmup_epochs: int = PAPER_WARMUP_EPOCHS
    patience: int = PAPER_PATIENCE
    use_amp: bool = False

    def __post_init__(self):
        if self.scale_factors is None:
            self.scale_factors = [1.0, 4.0]
        else:
            self.scale_factors = [float(x) for x in self.scale_factors]
        if self.levels_sub_dirs is None:
            self.levels_sub_dirs = ["level_1_5x", "level_0_20x"]
        else:
            self.levels_sub_dirs = [str(x) for x in self.levels_sub_dirs]

    def validate(self) -> None:
        if len(self.levels_sub_dirs) != len(self.scale_factors):
            raise ValueError("Number of levels and scale factors must match.")
        if len(self.levels_sub_dirs) < 1:
            raise ValueError("At least one level is required for multi-scale processing.")
        if len(self.scale_factors) < 1:
            raise ValueError("At least one scale factor is required for multi-scale processing.")
        if not all(isinstance(x, (int, float)) for x in self.scale_factors):
            raise ValueError("Scale factors must be numeric (int or float).")
        if not all(isinstance(x, str) for x in self.levels_sub_dirs):
            raise ValueError("Level subdirectory names must be strings.")
        if not isinstance(self.patch_size, int) or self.patch_size <= 0:
            raise ValueError("Patch size must be a positive integer.")
        if not isinstance(self.max_children, (int, type(None))) or (
            isinstance(self.max_children, int) and self.max_children <= 0
        ):
            raise ValueError("Max children must be a positive integer or None.")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("Number of workers must be a non-negative integer.")
        if not isinstance(self.labels_csv, str) or not self.labels_csv:
            raise ValueError("labels_csv must be a non-empty string path.")
        if not isinstance(self.multi_scale_features_dir, str) or not self.multi_scale_features_dir:
            raise ValueError("multi_scale_features_dir must be a non-empty string path.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError("Embedding dimension must be a positive integer.")
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("Number of epochs must be a positive integer.")
        if not isinstance(self.model_save_path, str) or not self.model_save_path:
            raise ValueError("model_save_path must be a non-empty string path.")
        if self.dataset not in {"PANDA", "TCGA_NSCLC", "TCGA_SURVIVAL"}:
            raise ValueError("dataset must be PANDA, TCGA_NSCLC, or TCGA_SURVIVAL.")
        if not isinstance(self.removal_fraction, float) or not (0 <= self.removal_fraction <= 1):
            raise ValueError("Removal fraction must be a float between 0 and 1.")
        if self.dataset in {"TCGA_NSCLC", "TCGA_SURVIVAL"} and not self.splits_path:
            raise ValueError(f"splits_path is required for dataset={self.dataset}.")
        if self.dataset in {"PANDA", "TCGA_SURVIVAL"} and not self.autogen_csv_path:
            raise ValueError(f"autogen_csv_path is required for dataset={self.dataset}.")

    def to_dict(self) -> dict:
        return asdict(self)


_FIELD_NAMES = {f.name for f in fields(TrainConfig)}


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def train_config_from_mapping(data: dict) -> TrainConfig:
    """Build TrainConfig from a dict (YAML keys). Accepts legacy ``labels_dir``."""
    cleaned = {}
    for key, value in data.items():
        if key == "labels_dir":
            cleaned["labels_csv"] = value
        elif key in _FIELD_NAMES:
            cleaned[key] = value
        # silently ignore unknown keys so configs can carry comments metadata
    return TrainConfig(**cleaned)


def merge_train_config(
    base: Optional[TrainConfig] = None,
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> TrainConfig:
    """Merge defaults ← YAML ← explicit CLI overrides (None values skipped)."""
    data: dict = {}
    if base is not None:
        data.update(base.to_dict())
    if yaml_path:
        data.update(load_yaml_config(yaml_path))
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None:
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 0:
                continue
            data[key] = value
    return train_config_from_mapping(data)
