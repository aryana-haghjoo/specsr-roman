"""Train/test splits.

The only rule that matters here: **split by object, never by row.** The same
OU2024 galaxy is observed in many visits, each an independent noise
realisation of the same underlying SED. A row-wise split puts realisation A in
train and realisation B in test, and the resulting "held-out" metric is
measuring memorisation.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np

from ..lines import count_strong_lines

__all__ = ["hash_file", "get_or_make_group_split", "get_or_make_split",
           "filter_split_min_lines", "default_split_dir"]


def hash_file(path: str) -> str:
    """MD5 of a whole file --- the identity of a dataset build."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def default_split_dir(dataset_path: str) -> str:
    """``<dataset dir>/splits``. Keeps split records beside the data they index."""
    return os.path.join(os.path.dirname(os.path.abspath(dataset_path)), "splits")


def get_or_make_group_split(dataset_path: str, ids, train_frac: float = 0.8,
                            split_dir: str | None = None, verbose: bool = True):
    """Deterministic split keyed by object id.

    Membership is a pure hash of the id, not a shuffled permutation. Two
    consequences, both deliberate:

    * every stage of the pipeline derives the *same* split from the same ids,
      without having to pass a file around;
    * the split is stable under dataset growth --- adding SCAs or visits never
      moves an existing galaxy across the boundary, so a model trained on the
      old build can still be evaluated on the new test set.

    A record file is written for provenance, but membership never depends on
    reading it back.
    """
    split_dir = split_dir or default_split_dir(dataset_path)
    os.makedirs(split_dir, exist_ok=True)
    ids = np.asarray(ids)
    h = np.array([int(hashlib.md5(str(int(i)).encode()).hexdigest()[:8], 16) % 1000
                  for i in ids])
    test = h >= int(train_frac * 1000)
    train_idx = np.where(~test)[0]
    test_idx = np.where(test)[0]

    ds_hash = hash_file(dataset_path)
    split_path = os.path.join(split_dir, f"group_split_{ds_hash}.npz")
    if not os.path.exists(split_path):
        np.savez(split_path, train_idx=train_idx, test_idx=test_idx,
                 dataset_hash=ds_hash, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if verbose:
        n_tr, n_te = len(np.unique(ids[train_idx])), len(np.unique(ids[test_idx]))
        print(f"group split by object_id: {len(train_idx)}/{len(test_idx)} rows "
              f"({n_tr}/{n_te} galaxies)")
    return train_idx, test_idx, split_path


def get_or_make_split(dataset_path: str, n_rows: int, train_frac: float = 0.8,
                      seed: int = 42, split_dir: str | None = None,
                      verbose: bool = True):
    """Row-wise split, for datasets with no object ids (Wang2022 era).

    Prefer :func:`get_or_make_group_split` whenever ``ids`` exist.
    """
    split_dir = split_dir or default_split_dir(dataset_path)
    os.makedirs(split_dir, exist_ok=True)
    ds_hash = hash_file(dataset_path)
    split_path = os.path.join(split_dir, f"split_{ds_hash}.npz")

    if os.path.exists(split_path):
        arr = np.load(split_path)
        train_idx, test_idx = arr["train_idx"], arr["test_idx"]
        if (train_idx.max() >= n_rows) or (test_idx.max() >= n_rows):
            raise RuntimeError(
                f"saved indices in {split_path} exceed the current dataset size "
                f"({n_rows} rows) -- the dataset changed under a stale split")
        if verbose:
            print(f"loaded split from {split_path}")
        return train_idx, test_idx, split_path

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_train = int(train_frac * n_rows)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    np.savez(split_path, train_idx=train_idx, test_idx=test_idx,
             dataset_hash=ds_hash, N=n_rows,
             created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if verbose:
        print(f"saved new split to {split_path}")
    return train_idx, test_idx, split_path


def filter_split_min_lines(train_idx, test_idx, z_all, wave_hi_aa,
                           min_lines: int, verbose: bool = True):
    """Drop split rows with fewer than ``min_lines`` strong lines in band.

    Applied *after* the shared split, never before: filtering first would
    change which galaxies fall on which side and break the guarantee that all
    three stages see the same partition. ``min_lines=2`` selects the
    line-pair-identifiable population (roughly z > 1, about 53% of the OU2024
    set) --- a fair sample to quote redshift performance on when photometry is
    not available to break the alias.
    """
    if min_lines <= 0:
        return train_idx, test_idx
    n_lines = count_strong_lines(z_all, wave_hi_aa[0], wave_hi_aa[-1])
    keep = n_lines >= int(min_lines)
    tr = train_idx[keep[train_idx]]
    te = test_idx[keep[test_idx]]
    if verbose:
        print(f"min_strong_lines={min_lines}: train {len(tr)}/{len(train_idx)}, "
              f"test {len(te)}/{len(test_idx)} rows kept")
    return tr, te
