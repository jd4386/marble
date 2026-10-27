import torch
from torch.utils.data import DataLoader

from marble.data.preprocess import MambaInputPreprocessor, MambaWrappedDataset, mamba_collate
from marble.data.wsi_dataset import MarbleWSIDataset


def get_loaders(train_df, val_df, test_df, cfg, n_levels, survival=False):
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("Train label distribution:")
    print(train_df['label'].value_counts(normalize=True))
    if survival:
        print(train_df['censorship'].value_counts())
        print(train_df['survival_months'].describe())
    print("Val label distribution:")
    print(val_df['label'].value_counts(normalize=True))
    if survival:
        print(val_df['censorship'].value_counts())
        print(val_df['survival_months'].describe())
    print("Test label distribution:")
    print(test_df['label'].value_counts(normalize=True))
    if survival:
        print(test_df['censorship'].value_counts())
        print(test_df['survival_months'].describe())

    print('Loading datasets...\n')

    # 3) Create Datasets for each split
    train_ds = MarbleWSIDataset(
        slide_df=train_df,
        results_dir=cfg.multi_scale_features_dir,
        levels=cfg.levels_sub_dirs[::-1],
        scale_factors=cfg.scale_factors[::-1],
        patch_size=cfg.patch_size,
        max_children_per_parent=cfg.max_children,
        survival=survival
    )
    val_ds = MarbleWSIDataset(
        slide_df=val_df,
        results_dir=cfg.multi_scale_features_dir,
        levels=cfg.levels_sub_dirs[::-1],
        scale_factors=cfg.scale_factors[::-1],
        patch_size=cfg.patch_size,
        max_children_per_parent=cfg.max_children,
        survival=survival
    )
    test_ds = MarbleWSIDataset(
        slide_df=test_df,
        results_dir=cfg.multi_scale_features_dir,
        levels=cfg.levels_sub_dirs[::-1],
        scale_factors=cfg.scale_factors[::-1],
        patch_size=cfg.patch_size,
        max_children_per_parent=cfg.max_children,
        survival=survival
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    val_sampler   = torch.utils.data.distributed.DistributedSampler(val_ds)
    test_sampler  = torch.utils.data.distributed.DistributedSampler(test_ds)

    print('Dataset loaded.\n')

    print('Creating DataLoader...\n')

    preproc = MambaInputPreprocessor(
        embedding_dim=cfg.embedding_dim,
        num_scales=n_levels,
        use_concat=False
    )

    mamba_collate_fn = lambda batch: mamba_collate(batch, survival=survival)

    train_loader = DataLoader(
        MambaWrappedDataset(train_ds, preproc, survival=survival),
        batch_size=cfg.batch_size,
        collate_fn=mamba_collate_fn,
        pin_memory=True,
        persistent_workers=False,
        num_workers=cfg.num_workers,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        MambaWrappedDataset(val_ds, preproc, survival=survival),
        batch_size=cfg.batch_size,
        collate_fn=mamba_collate_fn,
        pin_memory=True,
        persistent_workers=False,
        num_workers=cfg.num_workers,
        sampler=val_sampler
     )
    test_loader = DataLoader(
        MambaWrappedDataset(test_ds, preproc, survival=survival),
        batch_size=cfg.batch_size,
        collate_fn=mamba_collate_fn,
        pin_memory=True,
        persistent_workers=False,
        num_workers=cfg.num_workers,
        sampler=test_sampler
    )

    print('DataLoader created.\n')

    label_map = train_ds.label2idx
    class_distribution = train_df['label'].value_counts(normalize=True)

    train_weights = [1 / class_distribution[label] for label, _ in sorted(label_map.items(), key=lambda x: x[1])]
    train_weights = [float(w) for w in train_weights]

    return train_loader, val_loader, test_loader, train_ds.classes, train_weights, train_sampler
