import torch
import torch.nn as nn
from mamba_ssm import Mamba2 as Mamba
from torch.utils.checkpoint import checkpoint


class MarbleClassifier(nn.Module):
    def __init__(
            self, dim: int,
            num_scales: int,
            num_classes: int,
            survival: bool = False,
            pool_all: bool = False,
            removal_fraction: float = 0.1
        ):
        """
        dim         – feature dimension D
        num_scales  – number of scales = S+1
        num_classes – number of output classes
        """
        super().__init__()
        self.dim        = dim
        self.num_scales = num_scales
        self.S          = num_scales - 1
        self.survival   = survival
        self.pool_all  = pool_all
        self.removal_fraction = removal_fraction

        # one Mamba block per scale
        self.mambas = nn.ModuleList([Mamba(dim) for _ in range(num_scales)])
        # fusion layers for scales 1..S with dropout
        self.dropout = nn.Dropout(0.1)
        self.fuses  = nn.ModuleList([
            nn.Linear(2*dim, dim, bias=True)
            for _ in range(self.S)
        ])

        self.fuse_norms = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(self.S)
        ])

        # attention pooling head for finest-scale outputs
        self.attn       = nn.Linear(dim, 1, bias=False)
        # final classifier mapping pooled feature to logits with a hidden layer and dropout
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self,
        x:      torch.Tensor,      # (B, L, D)
        scales: torch.LongTensor,  # (B, L) values in {0..S}
        mask:   torch.BoolTensor=None  # (B, L), True = real token
    ) -> torch.Tensor:
        if self.training:
            # randomly drop some zero-scale segments
            x, scales, mask = filter_zero_scale_segments_fast(x, scales, mask, removal_fraction=self.removal_fraction)

        B, L, D = x.shape
        device  = x.device

        # placeholder for full sequence outputs (unused directly)
        out = torch.zeros_like(x)

        # default mask: all valid tokens
        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=device)

        # store per-scale outputs and their indices
        scale_outs = {}
        scale_outs_all = []

        # iterate over each scale
        for k in range(self.num_scales):
            # identify positions belonging to scale k
            sel     = (scales == k) & mask             # (B, L)
            lengths = sel.sum(dim=1)                   # (B,)
            max_len = int(lengths.max().item())        # max tokens in batch
            if max_len == 0:
                scale_outs[k] = (None, None)
                continue

            # bring selected tokens to front via argsort
            trues   = sel.to(torch.int)                # (B, L)
            idx_sort= torch.argsort(trues, dim=1, descending=True)  # (B, L)
            idxk    = idx_sort[:, :max_len]            # (B, max_len)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_len)

            # extract subsequence for this scale
            Xk = x[batch_idx, idxk]                    # (B, max_len, D)

            # fuse with previous scale context for k>0
            if k > 0:
                Yprev, idx_prev = scale_outs[k-1]      # (B, Tprev, D), (B, Tprev)
                if idx_prev is None:
                    # no previous context
                    Xpar = torch.zeros((B, max_len, D), device=device)
                else:
                    # compute parent index matrix via searchsorted
                    parent_mat = torch.zeros((B, max_len), dtype=torch.long, device=device)
                    for b in range(B):
                        prev_pos = idx_prev[b]           # sorted positions of scale k-1
                        child_pos= idxk[b]              # positions of current scale
                        loc = torch.searchsorted(prev_pos, child_pos, right=True) - 1
                        parent_mat[b] = loc.clamp(min=0)
                    Xpar = Yprev[batch_idx, parent_mat]  # (B, max_len, D)
                # fuse embedding + parent context
                Xk = self.fuses[k-1](torch.cat([Xk, Xpar], dim=-1))  # (B, max_len, D)
                Xk = self.dropout(Xk)
                Xk = self.fuse_norms[k-1](Xk)            # (B, max_len, D)

            # require gradients for Mamba blocks
            Xk = Xk.clone().detach().requires_grad_(True)  # (B, max_len, D)
            # use checkpointing to save memory
            Yk = checkpoint(self.mambas[k], Xk, use_reentrant=False)         # (B, max_len, D)
            # store for next scale and scatter into out (optional)
            scale_outs[k] = (Yk, idxk)
            scale_outs_all.append((Yk, idxk))
            out[batch_idx, idxk] = Yk.to(out.dtype)

        # ---------------------------------------------------------
        if self.pool_all:
            # pool all scales via attention
            YS = torch.cat([Y for Y, _ in scale_outs_all], dim=1)  # (B, TS, D)
        else:
            # pool only the finest-scale outputs via attention
            YS, idxS = scale_outs[self.S]                 # (B, TS, D), (B, TS)
        # compute attention scores
        scores = self.attn(YS).squeeze(-1)             # (B, TS)
        A      = torch.softmax(scores, dim=1)         # (B, TS)
        # weighted sum pooling
        z      = torch.einsum('bl,bld->bd', A, YS)    # (B, D)

        # final classification
        logits = self.classifier(z)              # (B, num_classes)
        # apply dropout
        logits = self.dropout(logits)

        if self.survival:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S

        return logits, None


# Preferred public name; MarbleClassifier retained for checkpoint / paper docs.
MarbleModel = MarbleClassifier


def filter_zero_scale_segments_fast(
    x: torch.Tensor,                   # (B, L, D)
    scales: torch.LongTensor,          # (B, L)
    mask: torch.BoolTensor = None,     # (B, L)
    removal_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
    B, L, D = x.shape
    device = x.device
    dtype  = x.dtype

    # Precompute an arange for broadcast
    arange = torch.arange(L, device=device).unsqueeze(0)  # (1, L)

    filtered_xs      = []
    filtered_scales  = []
    filtered_masks   = []

    for i in range(B):
        xi = x[i]                                     # (L, D)
        si = scales[i]                                # (L,)
        mi = mask[i] if mask is not None else None     # (L,) or None

        zero_pos = (si == 0).nonzero(as_tuple=True)[0]  # (n_zero,)
        n_zero   = zero_pos.numel()
        k        = int(n_zero * removal_fraction)

        if k <= 0:
            keep = torch.ones(L, dtype=torch.bool, device=device)
        else:
            # pick k zeros at random
            perm = torch.randperm(n_zero, device=device)[:k]
            sel  = zero_pos[perm]                       # (k,)

            # compute “next zero” for each zero position
            next_zero = torch.cat([zero_pos[1:], torch.tensor([L], device=device)])
            # we need the same indices in next_zero as in sel;
            # since perm picks indices into zero_pos, reuse perm:
            remove_starts = zero_pos[perm]               # (k,)
            remove_ends   = next_zero[perm]              # (k,)

            # vectorized “in‐interval” mask: (k,1) vs (1,L) → (k,L)
            starts = remove_starts.unsqueeze(1)          # (k,1)
            ends   = remove_ends.unsqueeze(1)            # (k,1)
            drop_mask = (arange >= starts) & (arange < ends)  # (k, L)
            keep = ~drop_mask.any(dim=0)                 # (L,)

        # slice out & collect
        filtered_xs     .append(xi[keep])
        filtered_scales .append(si[keep])
        filtered_masks  .append(mi[keep] if mi is not None else torch.ones_like(keep))

    # re‑pad to the max length in this batch
    max_len = max(t.size(0) for t in filtered_xs)
    x_out      = torch.zeros((B, max_len, D), dtype=dtype, device=device)
    scales_out = torch.zeros((B, max_len),      dtype=scales.dtype, device=device)
    mask_out   = torch.zeros((B, max_len),      dtype=torch.bool, device=device)

    for i in range(B):
        L_i = filtered_xs[i].size(0)
        x_out[i, :L_i]      = filtered_xs[i]
        scales_out[i, :L_i] = filtered_scales[i]
        mask_out[i, :L_i]   = filtered_masks[i]

    return x_out, scales_out, mask_out
