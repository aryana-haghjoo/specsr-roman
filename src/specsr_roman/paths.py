"""Where things live.

Resolution order for every data root is: explicit argument, then environment
variable, then a sensible default relative to the current working directory.
Nothing here reaches outside the project, and nothing hard-codes a machine.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["data_root", "dataset_dir", "runs_dir", "cache_dir", "grizli_conf_dir"]


def _env_path(var: str, default: str | Path) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()


def data_root() -> Path:
    """Root for downloaded and prepared simulation products (``SPECSR_ROMAN_DATA``)."""
    return _env_path("SPECSR_ROMAN_DATA", Path.cwd() / "data")


def dataset_dir() -> Path:
    """Where built training datasets land (``SPECSR_ROMAN_DATASETS``)."""
    return _env_path("SPECSR_ROMAN_DATASETS", data_root() / "dataset")


def runs_dir() -> Path:
    """Training outputs: checkpoints, predictions, split records (``SPECSR_ROMAN_RUNS``)."""
    return _env_path("SPECSR_ROMAN_RUNS", Path.cwd() / "runs")


def cache_dir() -> Path:
    """Scratch for extraction intermediates (``SPECSR_ROMAN_CACHE``)."""
    return _env_path("SPECSR_ROMAN_CACHE", data_root() / "cache")


def grizli_conf_dir() -> Path:
    """Local grizli ``CONF`` tree. grizli reads ``$GRIZLI``, so honour it first."""
    return _env_path("GRIZLI", Path.cwd() / "grizli_conf")
