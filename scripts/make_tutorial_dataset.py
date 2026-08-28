#!/usr/bin/env python
"""Build the small tutorial subset published alongside the full dataset.

    python scripts/make_tutorial_dataset.py                    # write it locally
    python scripts/make_tutorial_dataset.py --push             # ... and upload

The tutorial notebook has to run on a laptop in a couple of minutes, and the
full dataset is 271 MB. This carves a few hundred rows out of it that a reader
can download in seconds and still get meaningful numbers from.

Two rules govern the sampling, and both matter more than the size:

* **Test split only.** The subset is drawn from the held-out side of the
  canonical object-id split, so the redshift and line-recovery numbers a
  reader computes in the tutorial are honest out-of-sample numbers rather
  than a demonstration of memorisation.
* **Uniform random within it.** No cherry-picking on S/N, brightness or
  redshift: the subset carries the population's own mix of recoverable and
  unrecoverable lines, which is the whole point of the recoverability-binned
  metric the tutorial teaches. A hand-picked set of pretty spectra would make
  the model look better than it is.

The row selection is a seeded permutation, so re-running reproduces the same
subset byte for byte.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from specsr_roman.data import get_or_make_group_split  # noqa: E402

#: Columns copied straight through, row-subset for the per-row ones.
ROW_KEYS = ("flux_low", "flux_low_err", "flux_high", "redshift", "ids",
            "phot", "ab_h158", "snr", "visit", "sca")
SHARED_KEYS = ("wavelength_low", "wavelength_high", "phot_bands")

DEFAULT_REPO = "aryana-haghjoo/romansr-data"
HUB_PATH = "tutorial/ou2024_h10307_tutorial.npz"


def build(source: str, out: str, n_rows: int, seed: int) -> str:
    data = np.load(source, allow_pickle=True)
    ids = np.asarray(data["ids"])
    _, test_idx, _ = get_or_make_group_split(source, ids)

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.permutation(test_idx)[:n_rows])

    payload = {k: np.asarray(data[k])[rows] for k in ROW_KEYS}
    payload.update({k: np.asarray(data[k]) for k in SHARED_KEYS})
    # Provenance: which rows of the full dataset these are, so anything found
    # here can be traced back rather than re-derived from a seed.
    payload["source_row"] = rows.astype(np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **payload)

    z = payload["redshift"]
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"  {len(rows)} spectra, {len(np.unique(payload['ids']))} galaxies")
    print(f"  z {z.min():.3f} - {z.max():.3f} (median {np.median(z):.3f}), "
          f"{(z > 1).mean():.0%} above z = 1")
    print(f"  AB(H158) {payload['ab_h158'].min():.1f} - "
          f"{payload['ab_h158'].max():.1f}, median extraction S/N "
          f"{np.median(payload['snr']):.2f}")
    return out


def push(path: str, repo_id: str) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_file(path_or_fileobj=path, path_in_repo=HUB_PATH,
                        repo_id=repo_id, repo_type="dataset")
    print(f"pushed -> https://huggingface.co/datasets/{repo_id}/blob/main/{HUB_PATH}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/dataset/ou2024_h10307_dataset.npz")
    p.add_argument("--out", default="data/dataset/ou2024_h10307_tutorial.npz")
    p.add_argument("--n-rows", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--push", action="store_true", help=f"upload to {DEFAULT_REPO}")
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    args = p.parse_args()

    out = build(args.source, args.out, args.n_rows, args.seed)
    if args.push:
        push(out, args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
