"""Training loops for the three stages.

Each ``train(cfg)`` takes the matching dataclass from :mod:`specsr_roman.config`,
returns a summary dict, and writes its best checkpoint to ``cfg.out_dir``.
"""

from . import losses
from .sr1 import train as train_sr1
from .sr2 import train as train_sr2
from .zhead import train as train_zhead

__all__ = ["train_sr1", "train_zhead", "train_sr2", "losses"]
