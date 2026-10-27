"""CLI entrypoint for MARBLE training (``python -m marble`` / ``marble-train``)."""

from __future__ import annotations

import argparse
import warnings

from marble.config import TrainConfig, merge_train_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MARBLE — multi-scale linear-time encoder for WSI analysis",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (see configs/). CLI flags override YAML values.",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="Path to the slide-level labels CSV.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # deprecated alias for --labels_csv
    )
    parser.add_argument(
        "--multi_scale_features_dir",
        type=str,
        default=None,
        help="Base directory containing extracted feature subfolders per level.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Patch size in pixels (same for all levels). Default: 256.",
    )
    parser.add_argument(
        "--scale_factors",
        type=float,
        nargs="+",
        default=None,
        help="Pixel-size ratios per level vs base (e.g. 1 4). Default: 1 4.",
    )
    parser.add_argument(
        "--levels_sub_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Resolution subdirectories, coarse → fine. Default: level_1_5x level_0_20x.",
    )
    parser.add_argument(
        "--max_children",
        type=int,
        default=None,
        help="Maximum number of children per parent patch. If omitted, all children are included.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for data loading. Default: 0.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for data loading. Default: 1.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Dimensionality of the embedding space. Default: 1024.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (paper ~30 + early stop; default ceiling: 100).",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save the trained model (.pth under weights/; summary CSV → sibling metrics/).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["PANDA", "TCGA_NSCLC", "TCGA_SURVIVAL"],
        help="Dataset for which the model is being trained. Default: PANDA.",
    )
    parser.add_argument(
        "--autogen_csv_path",
        type=str,
        default=None,
        help="Path to the CLAM process_list_autogen.csv (required for PANDA and survival).",
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        default=None,
        help=(
            "Splits CSV (NSCLC) or directory of splits_*.csv (survival). "
            "Optional for PANDA (internal stratified split; path is ignored if set)."
        ),
    )
    parser.add_argument(
        "--removal_fraction",
        type=float,
        default=None,
        help="Fraction of zero-scale segments to remove during training. Default: 0.1.",
    )
    return parser


def _cli_overrides_from_args(args: argparse.Namespace) -> dict:
    labels_csv = args.labels_csv
    if args.labels_dir is not None:
        warnings.warn(
            "--labels_dir is deprecated; use --labels_csv instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if labels_csv is None:
            labels_csv = args.labels_dir

    return {
        "labels_csv": labels_csv,
        "multi_scale_features_dir": args.multi_scale_features_dir,
        "patch_size": args.patch_size,
        "scale_factors": args.scale_factors,
        "levels_sub_dirs": args.levels_sub_dirs,
        "max_children": args.max_children,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "model_save_path": args.model_save_path,
        "dataset": args.dataset,
        "autogen_csv_path": args.autogen_csv_path,
        "splits_path": args.splits_path,
        "removal_fraction": args.removal_fraction,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = merge_train_config(
        base=TrainConfig(),
        yaml_path=args.config,
        cli_overrides=_cli_overrides_from_args(args),
    )

    # Lazy imports so ``python -m marble --help`` works without CUDA / mamba-ssm.
    import torch.distributed as dist
    from marble.engine import run

    # DDP init happens here (not at import time) so ``--help`` exits cleanly.
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        print("Running MARBLE...\n")
        run(cfg)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    print("\nFinished!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
