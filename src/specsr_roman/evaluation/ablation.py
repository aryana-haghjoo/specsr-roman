"""Photometry ablation --- how much of the redshift accuracy is the spectrum?

This module exists because a result was too good. A ZHead fed all 14 OU2024
bands scored NMAD 0.0035 with **0% catastrophic outliers**, which is not
possible for a single-line grism: with one in-band line the identification is
alias-degenerate and the information floor is around 37%.

The ablation found the cause without retraining anything. Photometry enters
the head standardised as ``(log10(flux) - phot_mu) / phot_sig`` with the
statistics baked into the checkpoint, so "drop a band" is exactly "feed it its
training mean", which standardises to 0. Masking all 14 bands drops the head
to NMAD 0.18 and **46% catastrophic** --- the single-line floor. The spectrum
was carrying almost none of the accuracy; the photometry was an effectively
complete SED, and a noiseless one at that.

Two caveats the numbers depend on:

* masking a *partial* subset is out of distribution and non-monotonic. A
  masked "Roman-3" mis-scores around 0.37, roughly 50x worse than the truth
  --- which is why the deployable middle point had to come from an actual
  retrain rather than from masking.
* the noise sweep perturbs with the same multiplicative log-normal jitter used
  in training, so small values are in-distribution and the degradation is real
  rather than an artefact of an unfamiliar input. Each level is drawn from a
  generator reseeded to the same value, so the sweep varies only sigma.

**Why there is no "zero the spectrum" row.** The obvious complement --- keep the
photometry, blank the spectrum, see what the colours alone can do --- does not
work here, and reporting it would be worse than reporting nothing. Masking a
band to its training mean is *in distribution*: it standardises to exactly 0,
a value the network sees constantly. There is no equivalent for the spectral
channels. Feeding SR1 a zero array produces a reconstruction and an uncertainty
map unlike anything in training, and the head then reads nonsense from two of
its four channels. Measured, that configuration scores *worse* than masking
every photometric band --- which tells you the input was out of distribution,
not what the photometry contributes. The `grism only` row answers the
answerable half of the question; the other half needs a head trained without
the spectral channels.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..checkpoints import load_sr1, load_zhead_ckpt
from ..data import RomanFixedGridDataset, get_or_make_group_split
from ..grids import ROMAN_MEDIUM_BANDS
from ..models import pz_stats, z_metrics

__all__ = ["AblationConfig", "run_ablation"]

#: Photometric noise levels for the sweep, in magnitudes.
NOISE_MAG = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


@dataclass
class AblationConfig:
    data: str = "data/dataset/ou2024_h10307_dataset.npz"
    sr1_ckpt: str = "sr1_ou2024_v6"
    #: The 14-band head. Kept on the Hub precisely so this audit stays
    #: reproducible --- it is the leak, not a model to deploy.
    zhead_full_ckpt: str = "zhead_ou2024_v4"
    #: The retrained, deployable Roman Medium-tier head.
    zhead_roman_ckpt: str = "zhead_ou2024_roman_med3_noisy"
    roman_bands: tuple[int, ...] = ROMAN_MEDIUM_BANDS
    eval_mag_err: float = 0.05
    noise_seed: int = 0
    batch_size: int = 64
    out_dir: str = "outputs"
    noise_levels: tuple[float, ...] = NOISE_MAG


def _make_input(sr1, x_low, device, null_spectrum: bool = False):
    """Assemble the 4-channel ZHead input from the LR spectrum and frozen SR1.

    ``null_spectrum`` blanks the spectral channels. It is exposed for
    experimentation but is **not** a valid ablation --- see the module
    docstring --- because a zero spectrum is out of distribution for SR1.
    """
    lr = x_low.to(device, non_blocking=True)          # (B, 2, L) [flux, err]
    if null_spectrum:
        lr = torch.zeros_like(lr)
    with torch.no_grad():
        m, lv = sr1(lr)
    return torch.cat([lr, m, 0.5 * lv], dim=1)        # 4 channels for clf


def _mask_phot(phot, keep, phot_mu, n_bands: int):
    """Set every band NOT in ``keep`` to its training mean (standardises to 0)."""
    out = phot.clone()
    drop = [b for b in range(n_bands) if b not in keep]
    if drop:
        out[:, drop] = torch.pow(10.0, phot_mu[drop]).to(out.dtype)
    return out


@torch.no_grad()
def _evaluate(sr1, zhead, loader, device, keep=None, null_spectrum=False,
              noise_mag=0.0, gen=None, columns=None):
    """Score a head. ``keep`` masks bands; ``columns`` slices them instead.

    Masking is for the full-band head (its ``n_phot`` is fixed at 14);
    slicing is for a retrained few-band head, where feeding only the kept
    columns is fully in-distribution.
    """
    preds, trues = [], []
    for batch in loader:
        x_low, z = batch[0], batch[3]
        phot = batch[7].to(device, non_blocking=True)
        if columns is not None:
            phot = phot[:, list(columns)]
        if noise_mag > 0:
            dm = noise_mag * torch.randn(phot.shape, generator=gen,
                                         device=phot.device)
            phot = phot * torch.pow(10.0, -0.4 * dm)
        if keep is not None:
            phot = _mask_phot(phot, keep, zhead.phot_mu, zhead.n_phot)
        x_in = _make_input(sr1, x_low, device, null_spectrum)
        probs = torch.softmax(zhead(x_in, phot=phot), dim=-1)
        zhat, _ = pz_stats(probs, zhead.z_centers, zhead.refine_window)
        preds.append(zhat.cpu().numpy())
        trues.append(z.numpy())
    return z_metrics(np.concatenate(preds), np.concatenate(trues))


def run_ablation(cfg: AblationConfig) -> list[dict]:
    """Run the ablation and write ``phot_ablation.csv``. Returns the rows."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    ds = RomanFixedGridDataset(cfg.data, with_phot=True)   # all 14 bands
    if ds.phot is None:
        raise SystemExit("dataset has no `phot` array")
    _, test_idx, _ = get_or_make_group_split(os.path.abspath(cfg.data), ds.ids)
    loader = DataLoader(Subset(ds, test_idx), batch_size=cfg.batch_size,
                        shuffle=False)

    sr1 = load_sr1(cfg.sr1_ckpt, device=device)
    zfull = load_zhead_ckpt(cfg.zhead_full_ckpt, device=device)
    zroman = load_zhead_ckpt(cfg.zhead_roman_ckpt, device=device)
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed)

    rows: list[dict] = []

    def record(label, met, note=""):
        rows.append({"config": label, **met, "note": note})
        print(f"{label:32s} nmad {met['dz_nmad']:.4f}  "
              f"cat {met['catastrophic_frac']:.3f}  {note}")

    # 1. grism only: every band masked to its training mean.
    gen.manual_seed(cfg.noise_seed)
    record("grism only (all bands masked)",
           _evaluate(sr1, zfull, loader, device, keep=[]),
           "the honest single-line floor")

    # 2. the deployable retrained head, in distribution, with realistic noise.
    gen.manual_seed(cfg.noise_seed)
    record("grism + Roman-3 (retrained)",
           _evaluate(sr1, zroman, loader, device, columns=cfg.roman_bands,
                     noise_mag=cfg.eval_mag_err, gen=gen),
           "deployable")

    # 3. the leaked headline, for reference only.
    gen.manual_seed(cfg.noise_seed)
    record("grism + 14-band truth photometry",
           _evaluate(sr1, zfull, loader, device, keep=list(range(zfull.n_phot))),
           "LEAK -- not reproducible on real data")

    # 4. noise sweep on the full-band head.
    for mag in cfg.noise_levels:
        gen.manual_seed(cfg.noise_seed)
        record(f"14-band, +{mag:.2f} mag noise",
               _evaluate(sr1, zfull, loader, device,
                         keep=list(range(zfull.n_phot)),
                         noise_mag=mag, gen=gen))

    out_csv = os.path.join(cfg.out_dir, "phot_ablation.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    return rows
