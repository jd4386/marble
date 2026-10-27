"""Fold orchestration, model build, and summary CSV writing."""

from __future__ import annotations

import os
import time

import pandas as pd
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split

from marble.config import TrainConfig
from marble.data.labels import (
    get_panda_slide_df,
    get_tcga_nsclc_slide_df,
    get_tcga_survival_slide_df,
)
from marble.model import MarbleClassifier
from marble.train import get_loaders, run_training, run_training_survival


def run(cfg: TrainConfig) -> None:
    """Run the full training pipeline for ``cfg`` (DDP must already be initialized)."""
    print("Parsing arguments...\n")
    _print_config(cfg)

    print("Validating arguments...\n")
    cfg.validate()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"Using device: {device}")

    n_levels = len(cfg.levels_sub_dirs)
    survival = cfg.dataset == "TCGA_SURVIVAL"

    summary_dict = {
        "folds": [],
        "elapsed_time": [],
        "best_epoch": [],
    }

    # 1) Load the full slide DataFrame
    if cfg.dataset == "TCGA_NSCLC":
        slide_df = get_tcga_nsclc_slide_df(cfg.labels_csv)
    elif cfg.dataset == "PANDA":
        slide_df = get_panda_slide_df(cfg.labels_csv, cfg.autogen_csv_path)
    elif cfg.dataset == "TCGA_SURVIVAL":
        slide_df = get_tcga_survival_slide_df(cfg.labels_csv, cfg.autogen_csv_path)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    print(slide_df.head())

    if survival:
        summary_dict.update({"val_c_index": []})
    else:
        summary_dict.update({
            "val_acc": [],
            "val_auc": [],
            "test_acc": [],
            "test_auc": [],
            "val_report": [],
            "test_report": [],
        })

    if survival:
        # Survival: 5 folds. Each fold overwrites model_save_path so the published
        # checkpoint basename stays compatible with README / paper artifact names
        # (last fold wins on disk). Metrics for all folds are still written to the
        # sibling *_summary.csv.
        splits = 5
        for split in range(splits):
            split_path = os.path.join(cfg.splits_path, f"splits_{split}.csv")
            train_split(
                cfg=cfg,
                split=split,
                splits=splits,
                split_path=split_path,
                slide_df=slide_df,
                n_levels=n_levels,
                device=device,
                summary_dict=summary_dict,
                survival=True,
            )
    else:
        # Classification (PANDA / NSCLC): single run.
        # PANDA ignores splits_path and uses an internal stratified 80/10/10 split
        # (random_state=42) for reproducibility with published weights.
        # NSCLC requires splits_path (a CSV with train/val/test columns).
        train_split(
            cfg=cfg,
            split=0,
            splits=1,
            split_path=cfg.splits_path,
            slide_df=slide_df,
            n_levels=n_levels,
            device=device,
            summary_dict=summary_dict,
            survival=False,
        )

    summary_df = pd.DataFrame(summary_dict)

    # Compute average and standard deviation for each metric with folds avg and std
    metrics_dict = {
        "folds": ["avg", "std"],
        "elapsed_time": [summary_df["elapsed_time"].mean(), summary_df["elapsed_time"].std()],
        "best_epoch": [summary_df["best_epoch"].mean(), summary_df["best_epoch"].std()],
    }

    if survival:
        metrics_dict.update({
            "val_c_index": [summary_df["val_c_index"].mean(), summary_df["val_c_index"].std()],
        })
    else:
        metrics_dict.update({
            "val_acc": [summary_df["val_acc"].mean(), summary_df["val_acc"].std()],
            "val_auc": [summary_df["val_auc"].mean(), summary_df["val_auc"].std()],
            "test_acc": [summary_df["test_acc"].mean(), summary_df["test_acc"].std()],
            "test_auc": [summary_df["test_auc"].mean(), summary_df["test_auc"].std()],
            "val_report": [None, None],
            "test_report": [None, None],
        })

    metrics_df = pd.DataFrame(metrics_dict)
    summary_df = pd.concat([summary_df, metrics_df], ignore_index=True)

    print("\nSummary of results across folds:")
    print(summary_df)
    # Prefer weights/ → sibling metrics/ layout when --model_save_path ends with .../weights/*.pth
    save_dir = os.path.dirname(cfg.model_save_path)
    if os.path.basename(save_dir) == "weights":
        metrics_dir = os.path.join(os.path.dirname(save_dir), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        summary_save_path = os.path.join(
            metrics_dir, os.path.basename(cfg.model_save_path).replace(".pth", "_summary.csv")
        )
    else:
        summary_save_path = cfg.model_save_path.replace(".pth", "_summary.csv")
    summary_df.to_csv(summary_save_path, index=False)
    print(f"\nSummary saved to {summary_save_path}")


def train_split(
        cfg: TrainConfig,
        split,
        splits,
        split_path,
        slide_df,
        n_levels,
        device,
        summary_dict,
        survival=False,
):
    """Train the model on a specific split."""

    if survival:
        split_df = pd.read_csv(split_path)
        train_df = slide_df[slide_df["case_id"].isin(split_df["train"])].reset_index(drop=True)
        val_df = slide_df[slide_df["case_id"].isin(split_df["val"])].reset_index(drop=True)
        test_df = slide_df[slide_df["case_id"].isin(split_df["test"])].reset_index(drop=True)
    elif cfg.dataset == "PANDA":
        # PANDA: stratified random split; splits_path is unused (documented in README).
        train_df, remaining_df = train_test_split(
            slide_df, stratify=slide_df["label"], train_size=0.8, random_state=42
        )
        val_df, test_df = train_test_split(
            remaining_df, stratify=remaining_df["label"], train_size=0.5, random_state=42
        )
    else:
        # TCGA_NSCLC (and any future CSV-driven classification): use splits CSV.
        if not split_path:
            raise ValueError("splits_path is required for CSV-driven classification splits.")
        split_df = pd.read_csv(split_path)
        train_df = slide_df[slide_df["slide_id"].isin(split_df["train"])].reset_index(drop=True)
        val_df = slide_df[slide_df["slide_id"].isin(split_df["val"])].reset_index(drop=True)
        test_df = slide_df[slide_df["slide_id"].isin(split_df["test"])].reset_index(drop=True)

    print("Training split {}/{}...".format(split + 1, splits))

    train_loader, val_loader, test_loader, num_classes, train_weights, train_sampler = get_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=cfg,
        n_levels=n_levels,
        survival=survival,
    )

    model = MarbleClassifier(
        dim=cfg.embedding_dim,
        num_scales=n_levels,
        num_classes=num_classes,
        survival=survival,
        pool_all=False,
        removal_fraction=cfg.removal_fraction,
    ).to(device)

    print(f"→ Using {torch.cuda.device_count()} GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    print("Model created.\n")
    print(model)
    print("Training...\n")

    start_time = time.perf_counter()

    if survival:
        best_val_c_index, best_epoch = run_training_survival(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            base_lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
            use_amp=cfg.use_amp,
            patience=cfg.patience,
            model_path=cfg.model_save_path,
            train_weights=train_weights,
            train_sampler=train_sampler,
        )
    else:
        best_val_metrics, test_metrics, best_epoch = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            base_lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
            use_amp=cfg.use_amp,
            patience=cfg.patience,
            model_path=cfg.model_save_path,
            train_weights=train_weights,
            train_sampler=train_sampler,
        )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    summary_dict["folds"].append(split)
    summary_dict["elapsed_time"].append(elapsed_time)
    summary_dict["best_epoch"].append(best_epoch)

    if survival:
        summary_dict["val_c_index"].append(best_val_c_index)
    else:
        summary_dict["val_acc"].append(best_val_metrics.acc)
        summary_dict["val_auc"].append(best_val_metrics.auc)
        summary_dict["test_acc"].append(test_metrics.acc)
        summary_dict["test_auc"].append(test_metrics.auc)
        summary_dict["val_report"].append(best_val_metrics.report)
        summary_dict["test_report"].append(test_metrics.report)


def _print_config(cfg: TrainConfig) -> None:
    print(f"Labels CSV: {cfg.labels_csv}")
    print(f"Multi-scale features directory: {cfg.multi_scale_features_dir}")
    print(f"Patch size: {cfg.patch_size}")
    print(f"Scale factors: {cfg.scale_factors}")
    print(f"Levels subdirectories: {cfg.levels_sub_dirs}")
    print(f"Max children per parent: {cfg.max_children}")
    print(f"Number of workers: {cfg.num_workers}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Embedding dimension: {cfg.embedding_dim}")
    print(f"Number of epochs: {cfg.epochs}")
    print(f"Model save path: {cfg.model_save_path}")
    print(f"Dataset: {cfg.dataset}\n")
    print(f"Autogen CSV path: {cfg.autogen_csv_path}")
    print(f"Splits path: {cfg.splits_path}\n")
    print(f"Removal fraction: {cfg.removal_fraction}\n")
