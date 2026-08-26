"""Shared building blocks for the three stages.

Nothing here is Roman-specific; it is the small vocabulary of layers the three
model files assemble. Kept in one place so a change to, say, the
normalisation scheme cannot silently apply to SR1 and not SR2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Pre-activation 1D residual block with a fixed residual scale.

    ``alpha`` scales the branch rather than the skip, so a stack of these
    starts near the identity: at initialisation SR1 passes the (upsampled) LR
    spectrum through almost unchanged, and learning is spent on the delta.
    GroupNorm rather than BatchNorm because the batch is small and spectra
    within a batch differ wildly in continuum level.
    """

    def __init__(self, channels: int, groups: int = 8, alpha: float = 0.2,
                 p_drop: float = 0.1):
        super().__init__()
        self.alpha = alpha
        # largest divisor of `channels` at or below `groups` -- GroupNorm
        # requires exact divisibility and the hidden widths are swept values
        # (120, 96) that are not always multiples of 8.
        g = 1
        for i in range(groups, 0, -1):
            if channels % i == 0:
                g = i
                break
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=g, num_channels=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.block(x)


def conv_stack(in_ch: int, out_ch: int, num_blocks: int,
               dropout: float, kernel_size: int = 7) -> nn.Sequential:
    """Plain Conv1d/GELU/Dropout stack used by every ZHead variant.

    Wide kernels (7) and no downsampling: the redshift signal is a handful of
    ~5-pixel emission lines somewhere along a 2500-pixel axis, and pooling it
    away early is exactly how the first two ZHead designs collapsed to the
    prior mean.
    """
    layers: list[nn.Module] = []
    c = in_ch
    for _ in range(num_blocks):
        layers += [
            nn.Conv1d(c, out_ch, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        c = out_ch
    return nn.Sequential(*layers)


def build_param_groups(model: nn.Module, lr: float, weight_decay: float):
    """AdamW param groups that decay only conv/linear weight matrices.

    Biases and normalisation parameters are excluded: decaying a GroupNorm
    gain pulls the block toward killing its own signal, and decaying the
    log-variance head's bias fights its deliberate -2.0 initialisation.
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm = ("norm" in n.lower()) or ("groupnorm" in n.lower())
        if (p.ndim >= 2) and ("weight" in n) and (not is_norm):
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]
