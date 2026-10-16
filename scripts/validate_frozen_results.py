#!/usr/bin/env python3
"""Diagnose existing paper caches without modifying them or the manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from specsr_roman.data.splits import evaluation_split
from specsr_roman.evaluation.validation import (
    pz_calibration,
    spectral_calibration,
    unsupported_emission,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="outputs/pred_cache.npz")
    parser.add_argument("--pz-cache", default="outputs/pz_cache.npz")
    parser.add_argument("--data", default="data/dataset/ou2024_h10307_dataset.npz")
    parser.add_argument("--out", default="outputs/validation_v2/calibration.json")
    args = parser.parse_args()
    c = np.load(args.cache)
    if "object_id" in c:
        ids = c["object_id"]
    else:
        with np.load(args.data) as d:
            keep = (
                np.isfinite(d["snr"])
                & (np.isfinite(d["flux_low"]).mean(1) > 0.7)
                & (np.isfinite(d["flux_high"]).mean(1) > 0.9)
            )
            _, rows, _ = evaluation_split(args.data, d["ids"][keep])
            if not np.allclose(d["redshift"][keep][rows], c["z_true"]):
                raise ValueError("prediction cache does not match dataset rows")
            ids = d["ids"][keep][rows]
    out = {
        "cache": args.cache,
        "cache_sha256": hashlib.sha256(Path(args.cache).read_bytes()).hexdigest(),
        "data": args.data,
        "evaluation_status": "legacy validation, used for checkpoint selection",
        "spectral_uncertainty": spectral_calibration(
            c["sr2"], c["hr"], c["sigma"], c["line_snr"], c["z_true"], c["wl_um"], ids
        ),
        "unsupported_emission": unsupported_emission(
            c["sr2"], c["hr"], c["line_snr"], c["z_true"], c["wl_um"]
        ),
    }
    if "config" in c:
        cfg = json.loads(str(c["config"]))
        out["cache_config"] = cfg
        out["evaluation_status"] = f"{cfg['split_mode']} {cfg['partition']}"
    if "pz" in c:
        pz, centers = c["pz"], c["z_centers"]
    elif args.pz_cache:
        p = np.load(args.pz_cache)
        if (
            not np.array_equal(p["object_id"], ids)
            or not np.allclose(p["z_true"], c["z_true"])
            or not np.allclose(p["z_modes"][:, 0], c["z_pred"], atol=1e-4)
        ):
            raise ValueError("P(z) cache does not match prediction identity and point estimates")
        pz, centers = p["pz"], p["z_centers"]
    if "pz" in c or args.pz_cache:
        out["redshift_uncertainty"] = pz_calibration(pz, centers, c["z_true"])
    else:
        out["redshift_uncertainty"] = None
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
