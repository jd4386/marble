import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MambaInputPreprocessor(nn.Module):
    """
    Preprocesses raw dataset samples (features, coords, scale) into Mamba inputs (L, D).
    Can either sum or concatenate & project embeddings.
    """
    def __init__(self, embedding_dim: int, num_scales: int,
                 pos_mlp_hidden: int = None, use_concat: bool = False):
        """
        Args:
            embedding_dim (int): Dimensionality D of patch features.
            num_scales (int): Number of discrete scales/resolutions.
            pos_mlp_hidden (int, optional): Hidden size for the positional MLP.
                                             Defaults to embedding_dim if None.
            use_concat (bool): If True, concat feat/pos/scale and project; else sum.
        """
        super().__init__()
        hidden = pos_mlp_hidden or embedding_dim

        # Positional MLP: from 2D coords to D-dim
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embedding_dim)
        )

        # Scale embedding lookup
        self.scale_emb = nn.Embedding(num_scales, embedding_dim)
        self.use_concat = use_concat

        # Layer norms for each stream
        self.feat_norm  = nn.LayerNorm(embedding_dim)
        self.pos_norm   = nn.LayerNorm(embedding_dim)
        self.scale_norm = nn.LayerNorm(embedding_dim)

        if use_concat:
            # Fusion layer to project 3*D dims back to D
            self.fusion = nn.Linear(3 * embedding_dim, embedding_dim)

        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, features: torch.Tensor, coords: torch.Tensor, scales: torch.Tensor):
        """
        Args:
            features (torch.Tensor): (L, D) raw patch embeddings.
            coords   (torch.Tensor): (L, 2) patch x/y coords.
            scales   (torch.Tensor): (L,)  integer scale indices.
        Returns:
            mamba_input (torch.Tensor): (L, D) input for Mamba layer.
        """
        # 1) Normalize coords
        max_xy = coords.max(dim=0).values.unsqueeze(0)  # (1,2)
        coords_norm = coords / max_xy

        pos_emb = self.pos_mlp(coords_norm)        # (L, D)
        scale_emb = self.scale_emb(scales)    # (L, D)

        # 2) Embed each stream
        feat_e = self.feat_norm(features)
        pos_e  = self.pos_norm(pos_emb)
        sc_e   = self.scale_norm(scale_emb)

        if self.use_concat:
            # Concatenate and project
            concat = torch.cat([feat_e, pos_e, sc_e], dim=-1)  # (L, 3D)
            x = self.fusion(concat)                            # (L, D)
        else:
            # Elementwise sum
            x = feat_e + pos_e + sc_e                 # (L, D)

        # 3) Final normalization
        mamba_input = self.final_norm(x)          # (L, D)
        return mamba_input


# Preferred public names; Mamba* retained as aliases.
MarbleInputPreprocessor = MambaInputPreprocessor


class MambaWrappedDataset(Dataset):
    """
    Wraps a base dataset producing raw samples and applies the preprocessor
    to yield final inputs for the Mamba model.
    """
    def __init__(
            self,
            base_dataset: Dataset,
            preprocessor: MambaInputPreprocessor,
            survival: bool = False
        ):
        """
        Args:
            base_dataset (Dataset): Yields dicts with keys 'features', 'coords', 'scale', 'label'.
            preprocessor (MambaInputPreprocessor): Module to convert raw to Mamba inputs.
        """
        self.base = base_dataset
        self.preprocessor = preprocessor
        self.survival = survival

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        features = sample['features']  # (L, D)
        coords   = sample['coords']    # (L, 2)
        scales   = sample['scale']     # (L,)
        label    = sample['label']     # scalar

        # Generate Mamba input (and immediately detach so it's a CPU leaf tensor)
        # this prevents DataLoader from trying to pickle a grad‐tracking tensor
        with torch.no_grad():
            mamba_input = self.preprocessor(features, coords, scales).detach()
        # ensure it's on CPU and not requiring grad
        if mamba_input.device != torch.device('cpu'):
            mamba_input = mamba_input.cpu()

        if self.survival:
            return {
                'mamba_input': mamba_input,  # (L, D)
                'label':        label,        # scalar
                'scale':        scales,      # (L,)
                'event_time': sample['event_time'],  # scalar
                'censorship':   sample['censorship']            # scalar
            }

        return {
            'mamba_input': mamba_input,  # (L, D)
            'label':        label,        # scalar
            'scale':        scales,      # (L,)
        }


MarbleWrappedDataset = MambaWrappedDataset


def mamba_collate(batch, survival=False):
    """
    batch: list of dicts, each {
      'mamba_input': (Li, D),
      'scale':        (Li,),      # int64 tensor of scale indices
      'label':        () or (1,)  # scalar or 1‐element tensor
    }
    Returns a dict with:
      - mamba_input: (B, Lmax, D) float32
      - scale:       (B, Lmax)     int64
      - mask:        (B, Lmax)     bool   (True = real token)
      - label:       (B,)          int64
    """
    B = len(batch)
    Ls = [b['mamba_input'].shape[0] for b in batch]
    D  = batch[0]['mamba_input'].shape[1]
    Lmax = max(Ls)

    # allocate padded tensors
    inputs = torch.zeros(B, Lmax, D, dtype=torch.float32)
    scales = torch.full((B, Lmax), fill_value=0, dtype=torch.long)
    mask   = torch.zeros(B, Lmax, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    # For survival
    event_times = torch.zeros(B, dtype=torch.float32)
    censorships = torch.zeros(B, dtype=torch.float32)

    for i, sample in enumerate(batch):
        L = Ls[i]
        inputs[i, :L] = sample['mamba_input']            # (L, D)
        scales[i, :L] = sample['scale']                  # (L,)
        mask[i, :L]   = True
        labels[i]     = sample['label'].item() if sample['label'].ndim==0 else sample['label'][0]
        if survival:
            event_times[i] = sample['event_time'].item() if sample['event_time'].ndim==0 else sample['event_time'][0]
            censorships[i] = sample['censorship'].item() if sample['censorship'].ndim==0 else sample['censorship'][0]

    if survival:
        return {
            'mamba_input': inputs,  # (B, Lmax, D)
            'scale':        scales,  # (B, Lmax)
            'mask':         mask,    # (B, Lmax)
            'label':        labels,  # (B,)
            'event_time':   event_times,  # (B,)
            'censorship':   censorships    # (B,)
        }

    return {
        'mamba_input': inputs,  # (B, Lmax, D)
        'scale':        scales,  # (B, Lmax)
        'mask':         mask,    # (B, Lmax)
        'label':        labels,  # (B,)
    }


marble_collate = mamba_collate
