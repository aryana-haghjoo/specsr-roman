"""The three model stages.

``SuperRes1D`` (SR1) -> ``ZHeadClf`` (redshift) -> ``SR2Attention`` (refiner).
Each is trainable and usable alone; :mod:`specsr_roman.inference` chains them.
"""

from .blocks import ResidualBlock1D, build_param_groups, conv_stack
from .sr1 import SuperRes1D
from .sr2 import SR2Attention, build_line_mask, constrain_delta, line_profiles, topk_modes
from .zhead import (
                  ZHead1D,
                  ZHeadAttn,
                  ZHeadClf,
                  heteroscedastic_nll,
                  load_zhead,
                  make_z_grid,
                  pz_stats,
                  soft_labels,
                  z_metrics,
)

__all__ = [
    "ResidualBlock1D", "conv_stack", "build_param_groups",
    "SuperRes1D",
    "ZHead1D", "ZHeadAttn", "ZHeadClf", "load_zhead",
    "make_z_grid", "soft_labels", "pz_stats", "z_metrics", "heteroscedastic_nll",
    "SR2Attention", "topk_modes", "constrain_delta", "build_line_mask",
    "line_profiles",
]
