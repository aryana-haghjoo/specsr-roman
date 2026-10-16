#!/usr/bin/env python3
"""Compare response conventions on the same valid source-factor pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(primary, fixed):
    result = {}
    for stage in ("sr1", "sr2"):
        a, b = np.asarray(primary[stage]), np.asarray(fixed[stage])
        if not all(np.isin(x[:, 0], [0.5, 2.0]).all() for x in (a, b)):
            raise ValueError("matching expects the declared half/double perturbations")
        ka = a[:, 3].astype(np.int64) * 2 + (a[:, 0] > 1)
        kb = b[:, 3].astype(np.int64) * 2 + (b[:, 0] > 1)
        _, ia, ib = np.intersect1d(ka, kb, return_indices=True)
        if not len(ia):
            raise ValueError("no common responses")
        if not np.allclose(a[ia, 1], b[ib, 1]):
            raise ValueError("source recoverability differs between audits")
        out = {
            "n_responses": len(ia),
            "n_sources": len(np.unique(a[ia, 3])),
            "primary_median": float(np.median(a[ia, 2])),
            "fixed_median": float(np.median(b[ib, 2])),
            "by_snr": [],
        }
        for tag, lo, hi in [("marginal", 2, 3), ("good", 3, 6), ("strong", 6, np.inf)]:
            mask = (a[ia, 1] >= lo) & (a[ia, 1] < hi)
            if mask.any():
                out["by_snr"].append(
                    {
                        "bin": tag,
                        "n_responses": int(mask.sum()),
                        "primary_median": float(np.median(a[ia[mask], 2])),
                        "fixed_median": float(np.median(b[ib[mask], 2])),
                    }
                )
        result[stage] = out
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--primary", default="outputs/validation_v2/lr_restored/response_roman3.records.npz"
    )
    p.add_argument("--fixed", default="outputs/validation_v2/response_roman3.records.npz")
    p.add_argument("--out", default="paper/validation/normalization_control.json")
    args = p.parse_args()
    with np.load(args.primary) as a, np.load(args.fixed) as b:
        result = compare(a, b)
    result.update(
        primary=args.primary,
        fixed=args.fixed,
        note="Matched valid source-factor pairs; target_fixed is a normalized-output control.",
    )
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(result["sr2"])


if __name__ == "__main__":
    main()
