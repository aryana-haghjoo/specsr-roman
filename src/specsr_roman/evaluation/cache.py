"""Frozen test-split predictions.

Every figure and every number quoted about this pipeline is computed from one
cache file rather than from a live model, for two reasons: the full chain over
the test split takes minutes on a GPU that is usually busy, and --- more
importantly --- a cached prediction set cannot drift. Regenerate it
deliberately when the chain changes; never regenerate it by accident while
tuning a plot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from ..checkpoints import CANONICAL_CHAIN, load_sr1, load_sr2, load_zhead_ckpt
from ..data import RomanFixedGridDataset, apply_phot_noise, get_or_make_group_split
from ..inference.pipeline import build_sr2_input
from ..lines import LINE_LIST_REST_AA, angstrom_to_micron
from ..models import constrain_delta

__all__ = ["CacheConfig", "build_prediction_cache", "load_prediction_cache"]


@dataclass
class CacheConfig:
    data: str = "data/dataset/ou2024_h10307_dataset.npz"
    sr1_ckpt: str = CANONICAL_CHAIN["sr1"]
    zhead_ckpt: str = CANONICAL_CHAIN["zhead"]
    sr2_ckpt: str = CANONICAL_CHAIN["sr2"]
    phot_tier: str | None = "medium"
    #: Photometric noise applied at evaluation. Must match training, and must
    #: not be zero: a metric measured on noiseless truth photometry is not a
    #: metric. Seeded so the cache is reproducible.
    eval_mag_err: float = 0.05
    noise_seed: int = 0
    batch_size: int = 64
    delta_cap: float = 40.0
    sigma_base_um: float = 0.005
    z_topk: int = 3
    out: str = "outputs/pred_cache.npz"


def build_prediction_cache(cfg: CacheConfig) -> str:
    """Run the full chain over the test split and cache every array a figure needs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zhead = load_zhead_ckpt(cfg.zhead_ckpt, device=device)
    has_phot = getattr(zhead, "n_phot", 0) > 0

    ds = RomanFixedGridDataset(cfg.data, with_phot=has_phot,
                               phot_tier=cfg.phot_tier if has_phot else None)
    _, test_idx, _ = get_or_make_group_split(os.path.abspath(cfg.data), ds.ids)

    sr1 = load_sr1(cfg.sr1_ckpt, device=device)
    wl_um = ds.wave_hi.astype(np.float32) * 1e-4
    line_rest = angstrom_to_micron([w for _, w in LINE_LIST_REST_AA])
    sr2 = load_sr2(cfg.sr2_ckpt, device=device, line_rest_um=line_rest,
                   wave_hi_um=wl_um)

    # z normalisation from the TRAIN split, matching how the stages were trained.
    train_rows = np.setdiff1d(np.arange(len(ds)), test_idx)
    ztr = ds.z[train_rows].numpy()
    zmean, zstd = float(ztr.mean()), float(ztr.std())
    zminn, zmaxn = (ztr.min() - zmean) / zstd, (ztr.max() - zmean) / zstd
    run_cfg = {"z_topk": cfg.z_topk, "sigma_base_um": cfg.sigma_base_um}

    cols: dict[str, list] = {k: [] for k in
                             ("lr", "lr_err", "sr1", "sr2", "hr", "sigma",
                              "z_true", "z_pred", "line_snr")}
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed)
    with torch.no_grad():
        for s in range(0, len(test_idx), cfg.batch_size):
            rows = [int(i) for i in test_idx[s:s + cfg.batch_size]]
            batch = [ds[i] for i in rows]
            x_low = torch.stack([b[0] for b in batch]).to(device)
            x_high = torch.stack([b[1] for b in batch])
            z_true = torch.stack([b[3] for b in batch]).to(device)
            snr = torch.stack([b[6] for b in batch])
            phot = None
            if has_phot:
                phot = torch.stack([b[7] for b in batch]).to(device)
                phot = apply_phot_noise(phot, cfg.eval_mag_err, gen)

            x_in, sr1_mean, z_modes, z_w, _ = build_sr2_input(
                x_low, sr1, zhead, wl_um, line_rest, run_cfg, device,
                phot=phot, z_mean=zmean, z_std=zstd,
                z_min_n=zminn, z_max_n=zmaxn)
            delta, logvar, _ = sr2(x_in, z_modes, z_w)
            sr2_mean = sr1_mean + constrain_delta(delta, cfg.delta_cap)

            cols["lr"].append(x_low[:, 0].cpu().numpy())
            cols["lr_err"].append(x_low[:, 1].cpu().numpy())
            cols["sr1"].append(sr1_mean[:, 0].cpu().numpy())
            cols["sr2"].append(sr2_mean[:, 0].cpu().numpy())
            cols["hr"].append(x_high.numpy())
            cols["sigma"].append(
                torch.exp(0.5 * logvar.clamp(-8, 6))[:, 0].cpu().numpy())
            cols["z_true"].append(z_true.cpu().numpy())
            cols["z_pred"].append(z_modes[:, 0].cpu().numpy())
            cols["line_snr"].append(snr.numpy())
            print(f"  cached {s + len(rows)}/{len(test_idx)}", end="\r")
    print()

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    np.savez_compressed(cfg.out, wl_um=wl_um,
                        **{k: np.concatenate(v) for k, v in cols.items()})
    print(f"wrote {cfg.out}")
    return cfg.out


def load_prediction_cache(path: str, rebuild: bool = False,
                          cfg: CacheConfig | None = None):
    if rebuild or not os.path.exists(path):
        build_prediction_cache(cfg or CacheConfig(out=path))
    return np.load(path, allow_pickle=True)
