#!/usr/bin/env python3
"""Rerun all response audits into a separate directory using local weights."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def run_one(row):
    import torch

    from specsr_roman.evaluation.prior_dominance import PriorDominanceConfig, run_prior_dominance

    (key, zname, sname, tier, spectrum, limit, outdir, threads, output_scale,
     runs, data, split_mode, partition) = row
    torch.set_num_threads(threads)

    def checkpoint(stage, name):
        return name if runs == "hub" else str(Path(runs) / stage / f"{name}_best.pth")

    cfg = PriorDominanceConfig(
        data=data,
        sr1_ckpt=checkpoint("sr1", "sr1_ou2024_v6_independent"),
        zhead_ckpt=checkpoint("zhead", zname),
        sr2_ckpt=checkpoint("sr2", sname),
        split_mode=split_mode,
        partition=partition,
        phot_tier=tier,
        zhead_use_spectrum=spectrum,
        max_sources=limit,
        output_scale=output_scale,
        out_json=str(Path(outdir) / f"response_{key}.json"),
    )
    result = run_prior_dominance(cfg)
    return key, result["n_sources"], result["sr2"]["overall_median_r"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-sources", type=int, default=100000)
    p.add_argument("--outdir", default="outputs/audits_v060/lr_restored")
    p.add_argument("--runs", default="runs",
                   help="local stage directory, or 'hub' to use the released weights")
    p.add_argument("--data", default="data/dataset/ou2024_h10307_dataset.npz")
    p.add_argument("--split-mode", choices=("legacy", "three_way"), default="three_way")
    p.add_argument("--partition", choices=("validation", "test"), default="test")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output-scale", choices=("target_fixed", "legacy_lr"), default="legacy_lr")
    args = p.parse_args()
    rows = [
        (
            "roman3",
            "zhead_ou2024_roman_med3_noisy_independent",
            "sr2_ou2024_v5_romanonly_independent",
            "medium", True,
        ),
        (
            "rubin_y1",
            "zhead_ou2024_roman_med3_rubin_y1_independent",
            "sr2_ou2024_v5_rubin_y1_independent",
            "medium_rubin",
            True,
        ),
        (
            "rubin_y10",
            "zhead_ou2024_roman_med3_rubin_y10_independent",
            "sr2_ou2024_v5_rubin_y10_independent",
            "medium_rubin",
            True,
        ),
        (
            "photonly",
            "zhead_ou2024_photonly_rubin_y10_independent",
            "sr2_ou2024_v5_photonly_rubin_y10_independent",
            "medium_rubin",
            False,
        ),
    ]
    rows = [(*r, args.max_sources, args.outdir, args.threads, args.output_scale,
             args.runs, args.data, args.split_mode, args.partition) for r in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_one, rows):
            print("COMPLETE", *result, flush=True)


if __name__ == "__main__":
    main()
