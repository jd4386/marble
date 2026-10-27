import os

import h5py
import torch
from torch.utils.data import Dataset


class MarbleWSIDataset(Dataset):
    """
    Dataset for MARBLE WSI classification supporting multiple resolutions.
    Returns dict with:
      - features: (L, D) float32 tensor
      - coords:   (L, 2)    float32 tensor (normalized [0,1] if enabled)
      - scale:    (L,)      int64 tensor of level indices
      - label:    ()        int64 slide label
    """
    def __init__(self, slide_df, results_dir, levels, scale_factors,
                 patch_size=256, max_children_per_parent=None, survival=False):
        print('Initializing MarbleWSIDataset')

        assert len(levels) == len(scale_factors), "levels & scale_factors must match"
        self.slide_df = slide_df.reset_index(drop=True)
        self.results_dir = results_dir
        self.levels = levels
        self.scale_factors = scale_factors
        self.patch_size = patch_size
        self.max_children = max_children_per_parent
        self.survival = survival

        # label → index map
        unique = sorted(self.slide_df['label'].unique())
        self.classes = len(unique)
        self.label2idx = {lbl: i for i, lbl in enumerate(unique)}

        print(f'Found {len(unique)} unique labels')
        print(f'Label map: {self.label2idx}')

        # cache for computed sequences
        self._cache = {}
        # cache for open HDF5 handles
        self._h5_handles = {}

        print(f"Dataset initialized with {len(self.slide_df)} slides")

    def __getitem__(self, idx):
        row      = self.slide_df.loc[idx]
        slide_id = row['slide_id']
        label    = self.label2idx[row['label']]

        # if we've already built this slide's data, reuse it
        if slide_id not in self._cache:
            feats_list, coords_list = [], []

            # 1) read features & coords at each level, reusing open file handles
            for lvl in self.levels:
                path = os.path.join(self.results_dir, lvl,
                                    "clam_uni_features", "h5_files",
                                    f"{slide_id}.h5")

                # open once per worker and cache
                if path not in self._h5_handles:
                    # note: mode='r' ensures read-only
                    self._h5_handles[path] = h5py.File(path, 'r')

                f = self._h5_handles[path]
                feats  = torch.from_numpy(f["features"][:]).float()  # (Ni, D)
                coords = torch.from_numpy(f["coords"][:]).float()    # (Ni, 2)

                feats_list.append(feats)
                coords_list.append(coords)

            # 2) build the parent→child maps as before...
            children_maps = []
            for i in range(len(self.levels) - 1):
                parent_coords = coords_list[i]
                child_coords  = coords_list[i+1]
                region_size   = self.patch_size * self.scale_factors[i]

                # vectorized mask for all parents vs all children
                px = parent_coords[:, 0].unsqueeze(1)  # (Np, 1)
                py = parent_coords[:, 1].unsqueeze(1)  # (Np, 1)
                cx = child_coords[:, 0].unsqueeze(0)   # (1, Nc)
                cy = child_coords[:, 1].unsqueeze(0)   # (1, Nc)

                within_x = (cx >= px) & (cx < px + region_size)
                within_y = (cy >= py) & (cy < py + region_size)
                mask = within_x & within_y             # (Np, Nc)

                p2c = []
                for parent_idx in range(mask.shape[0]):
                    idxs = torch.nonzero(mask[parent_idx], as_tuple=False).squeeze(1)
                    if self.max_children and idxs.numel() > self.max_children:
                        perm = torch.randperm(idxs.numel())
                        idxs = idxs[perm[:self.max_children]]
                    p2c.append(idxs)
                children_maps.append(p2c)

            self._cache[slide_id] = (feats_list, coords_list, children_maps)

        feats_list, coords_list, children_maps = self._cache[slide_id]

        # 3) traverse to build the sequence…
        seq_feats, seq_pos, seq_scale = [], [], []
        def traverse(level, p_idx):
            seq_feats.append(feats_list[level][p_idx])
            seq_pos.append(coords_list[level][p_idx])
            seq_scale.append(level)
            if level < len(self.levels) - 1:
                for c_idx in children_maps[level][p_idx]:
                    traverse(level + 1, int(c_idx))

        for parent_idx in range(len(feats_list[0])):
            traverse(0, parent_idx)

        X_feats = torch.stack(seq_feats)
        X_pos   = torch.stack(seq_pos)
        X_scale = torch.tensor(seq_scale, dtype=torch.long)
        y       = torch.tensor(label, dtype=torch.long)

        if self.survival:
            event_time = row['survival_months']
            censorship = row['censorship']

            # Create tensors for survival analysis
            event_time = torch.tensor(event_time, dtype=torch.float)
            censorship = torch.tensor(censorship, dtype=torch.float)

            return {
                "features": X_feats,
                "coords":   X_pos,
                "scale":    X_scale,
                "label":    y,
                "event_time": event_time,
                "censorship": censorship
            }

        return {
            "features": X_feats,
            "coords":   X_pos,
            "scale":    X_scale,
            "label":    y
        }

    def __len__(self):
        return len(self.slide_df)

    def __del__(self):
        # close any open HDF5 files when the dataset is garbage-collected
        for f in self._h5_handles.values():
            try:
                f.close()
            except Exception:
                pass
