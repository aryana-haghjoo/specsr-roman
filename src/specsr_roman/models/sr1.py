"""SR1 --- the coarse super-resolution stage.

A fully convolutional 1D ResNet mapping the (upsampled) LR grism spectrum to a
mean and a heteroscedastic log-variance on the HR grid. Architecture is
deliberately identical to the JWST/JADES version in ``specsr`` so the two can
be compared and warm-started across instruments; only the input channel count
differs (Roman feeds ``[flux, err]``, see below).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ResidualBlock1D

__all__ = ["SuperRes1D"]


class SuperRes1D(nn.Module):
    """LR spectrum -> (mean, log_var) on the same grid.

    Parameters
    ----------
    in_channels
        ``2`` for the canonical Roman models: ``[flux, err]``, both divided by
        the same per-row flux scale so their ratio is the per-pixel S/N. The
        network can then matched-filter rather than guess which bumps are
        noise --- the single largest quality jump in the Roman port. ``1`` is
        the JWST-compatible flux-only input.
    hidden_dim, num_res_blocks, dropout
        Swept on the JWST data and carried over unchanged (120 / 16 / ~0.024).

    Notes
    -----
    Input and output are the *same length*: the LR spectrum is interpolated
    onto the HR grid before it reaches the model. A fully convolutional stack
    with no resampling layers cannot change the axis length, and keeping the
    grids aligned means the residual ``pred - input`` is meaningful pixel by
    pixel.

    The log-variance head is initialised to a constant -2.0 with zero weights,
    so the model starts by predicting a uniform sigma ~ 0.37 and has to earn
    any structure in its uncertainty.
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 96,
                 num_res_blocks: int = 12, dropout: float = 0.02,
                 activation_fn: nn.Module | None = None):
        super().__init__()
        activation_fn = nn.GELU() if activation_fn is None else activation_fn
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2, bias=True),
            activation_fn,
        )
        self.resblocks = nn.Sequential(*[
            ResidualBlock1D(hidden_dim, p_drop=dropout)
            for _ in range(num_res_blocks)
        ])
        self.mean_head = nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True)
        self.log_var_head = nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.log_var_head.weight, 0.0)
        nn.init.constant_(self.log_var_head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.initial(x)
        x = self.resblocks(x)
        return self.mean_head(x), self.log_var_head(x)

    @classmethod
    def from_state_dict(cls, state: dict, hidden_dim: int = 120,
                        num_res_blocks: int = 16) -> SuperRes1D:
        """Build a model whose input width matches a checkpoint, then load it.

        The 1- vs 2-channel input is the one architectural thing that changed
        across Roman SR1 generations, and every downstream stage has to load
        whichever it is handed. Read it off the stem weight rather than making
        the caller remember.
        """
        in_channels = state["initial.0.weight"].shape[1]
        model = cls(in_channels=in_channels, hidden_dim=hidden_dim,
                    num_res_blocks=num_res_blocks)
        model.load_state_dict(state)
        return model
