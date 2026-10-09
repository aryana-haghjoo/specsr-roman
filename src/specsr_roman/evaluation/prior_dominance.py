"""The inverse-crime audit: is the model reading the data or reciting the prior?

The training targets are model SEDs. A network can score well on every
reconstruction metric by learning the simulation's manifold --- its fixed line
ratios, its single dust law --- rather than by measuring anything. No
reconstruction metric can tell the two apart, because on the manifold they
give the same answer.

This test forces them apart. Take a source whose line the model does recover.
Scale that line in the *truth* by a factor ``f`` --- deliberately off the
manifold, a line ratio the simulation never produces --- forward-model the
*difference* through a Gaussian LSF at grism resolution onto the observed LR
spectrum, and re-run the model. Then measure the response exponent

    r = log(L_pred_perturbed / L_pred_original) / log(f)

``r = 1`` means the model tracked the change: it read the line strength from
the data. ``r = 0`` means it produced the same line regardless: it recited the
prior.

Reading the result requires care. Where the injected change is genuinely below
the noise, a low ``r`` is the *correct* behaviour --- falling back on the prior
is what a well-calibrated model should do when the data says nothing. So bin
by the detectability of the injected change, and judge ``r`` only where the
information is physically present. Aggregate ``r`` is dominated by
unrecoverable cases and understates a good model.

Measured on this project, on the 936 held-out sources with a line above
S/N 5: 0.30 for the published SR1 and 0.30 for the full pipeline, rising from
0.22 at S/N 5-6 to 0.41 above S/N 8. SR2 does not change the response.
Anti-prior augmentation raised an earlier SR1 from 0.14 to 0.51 at fixed
detectability, at the cost of absolute line recovery, which is why the
published SR1 is the unaugmented one and this remains open work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from ..checkpoints import load_sr1, load_sr2, load_zhead_ckpt
from ..data import (
                      RomanFixedGridDataset,
                      apply_phot_noise,
                      get_or_make_group_split,
                      normalize,
)
from ..grids import GRISM_FWHM_AA

__all__ = ["PriorDominanceConfig", "run_prior_dominance"]

# np.trapz was renamed in NumPy 2.0; support both so the package does not
# force a NumPy major version on its users.
_trapz = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201


@dataclass
class PriorDominanceConfig:
    data: str = "data/dataset/ou2024_h10307_dataset.npz"
    sr1_ckpt: str = "sr1_ou2024_v6"
    snr_min: float = 5.0            # only sources with a usable spectrum
    min_recovered_frac: float = 0.2  # line must be recovered at all
    factors: tuple[float, ...] = (0.5, 2.0)
    max_sources: int = 500
    fwhm_aa: float = GRISM_FWHM_AA
    #: Give both to audit the full pipeline as well as SR1. The redshift stage
    #: re-reads the perturbed spectrum; the photometry is drawn once per source
    #: and held fixed, so the response is to the spectrum alone.
    zhead_ckpt: str | None = None
    sr2_ckpt: str | None = None
    phot_tier: str = "medium"
    eval_mag_err: float = 0.05
    noise_seed: int = 0
    delta_cap: float = 40.0
    sigma_base_um: float = 0.005
    z_topk: int = 3


def _find_strongest_line(flux_hi, smooth_px: int = 101):
    """Contiguous segment around the strongest emission line, or ``None``."""
    from scipy.ndimage import gaussian_filter1d
    cont = gaussian_filter1d(flux_hi, smooth_px)
    resid = flux_hi - cont
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    if sigma <= 0:
        return None
    mask = resid > 8 * sigma
    if not mask.any():
        return None
    peak = int(np.argmax(resid))
    if not mask[peak]:
        return None
    lo = hi = peak
    while lo > 0 and resid[lo - 1] > 2 * sigma:
        lo -= 1
    while hi < len(resid) - 1 and resid[hi + 1] > 2 * sigma:
        hi += 1
    seg = np.zeros_like(mask)
    seg[lo:hi + 1] = True
    return seg, cont


def _line_flux(spec, seg, wave):
    """Continuum-subtracted line flux over a segment, local continuum removed."""
    from scipy.ndimage import gaussian_filter1d
    cont = gaussian_filter1d(spec, 101)
    return _trapz((spec - cont)[seg], wave[seg])


def run_prior_dominance(cfg: PriorDominanceConfig) -> dict:
    """Run the audit on SR1, and on the full pipeline when it is configured.

    Returns the response exponents per stage (``summary["sr1"]``,
    ``summary["sr2"]``), each per factor, overall and by best-line S/N. The
    SR1 entries are repeated at the top level.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr1 = load_sr1(cfg.sr1_ckpt, device=device)

    chain = bool(cfg.zhead_ckpt and cfg.sr2_ckpt)
    zhead = load_zhead_ckpt(cfg.zhead_ckpt, device=device) if chain else None
    has_phot = chain and getattr(zhead, "n_phot", 0) > 0
    ds = RomanFixedGridDataset(cfg.data, with_phot=has_phot,
                               phot_tier=cfg.phot_tier if has_phot else None)
    _, test_idx, _ = get_or_make_group_split(os.path.abspath(cfg.data), ds.ids)

    wh = ds.wave_hi
    sig_hr_px = (cfg.fwhm_aa / 2.355) / np.median(np.diff(wh))

    # Candidates: bright enough that the line is genuinely in the data.
    best_snr = ds.line_snr.max(dim=1).values.numpy()
    cand = [i for i in test_idx if best_snr[i] > cfg.snr_min][: cfg.max_sources]
    print(f"{len(cand)} candidate test sources with a recoverable line")

    if chain:
        from ..inference.pipeline import build_sr2_input
        from ..lines import LINE_LIST_REST_AA, angstrom_to_micron
        from ..models import constrain_delta
        wl_um = wh.astype(np.float32) * 1e-4
        line_rest = angstrom_to_micron([w for _, w in LINE_LIST_REST_AA])
        sr2 = load_sr2(cfg.sr2_ckpt, device=device, line_rest_um=line_rest,
                       wave_hi_um=wl_um)
        # z normalisation from the train split, as in the prediction cache
        ztr = ds.z[np.setdiff1d(np.arange(len(ds)), test_idx)].numpy()
        zmean, zstd = float(ztr.mean()), float(ztr.std())
        zminn, zmaxn = (ztr.min() - zmean) / zstd, (ztr.max() - zmean) / zstd
        run_cfg = {"z_topk": cfg.z_topk, "sigma_base_um": cfg.sigma_base_um}
        gen = torch.Generator(device=device).manual_seed(cfg.noise_seed)

    def run_models(lr_hr_grid, err_hr_grid, phot=None):
        """``{stage: prediction}`` in the units of the input spectrum."""
        xn, m, s = normalize(lr_hr_grid)
        err_n = err_hr_grid / max(s, 1e-25)
        x = torch.tensor(np.stack([xn, err_n])[None].astype(np.float32),
                         device=device)
        with torch.no_grad():
            if not chain:
                pred, _ = sr1(x)
                return {"sr1": pred[0, 0].cpu().numpy() * s + m}
            x_in, sr1_mean, z_modes, z_w, _ = build_sr2_input(
                x, sr1, zhead, wl_um, line_rest, run_cfg, device, phot=phot,
                z_mean=zmean, z_std=zstd, z_min_n=zminn, z_max_n=zmaxn)
            delta, _, _ = sr2(x_in, z_modes, z_w)
            sr2_mean = sr1_mean + constrain_delta(delta, cfg.delta_cap)
        return {"sr1": sr1_mean[0, 0].cpu().numpy() * s + m,
                "sr2": sr2_mean[0, 0].cpu().numpy() * s + m}

    from scipy.ndimage import gaussian_filter1d

    def calibration_scale(hr, lr):
        """Scalar taking SED units into extraction units.

        The Diffsky SEDs carry an internal flux scale (~1e-20 here) that has
        nothing to do with the extraction's units, and the dataset hides the
        mismatch by normalising each spectrum at load time. An injected delta
        computed in SED units is therefore numerically invisible once added to
        the LR spectrum -- which silently pins the response exponent at exactly
        zero and makes every model look like it recites the prior.

        The bridge is the least-squares scale between the LSF-smoothed truth
        and the observation: the empirical flux-calibration ratio, absorbing
        both the unit difference and the known ~1.7x aperture-loss offset.
        """
        hr_s = gaussian_filter1d(hr, sig_hr_px)
        ok = np.isfinite(hr_s) & np.isfinite(lr)
        denom = float(np.dot(hr_s[ok], hr_s[ok]))
        if denom <= 0:
            return None
        return float(np.dot(hr_s[ok], lr[ok])) / denom

    stages = ("sr1", "sr2") if chain else ("sr1",)
    # per stage: (factor, best-line S/N, r) for every usable source
    records: dict[str, list[tuple[float, float, float]]] = {k: [] for k in stages}
    n_used = 0
    for i in cand:
        hr = np.nan_to_num(ds.hi_raw[i]).astype(np.float64)
        lr = ds.lo_raw[i].astype(np.float64)
        err = ds.err_raw[i].astype(np.float64)

        found = _find_strongest_line(hr)
        if found is None:
            continue
        seg, _ = found

        phot = None
        if has_phot:
            phot = apply_phot_noise(torch.tensor(ds.phot[i])[None].to(device),
                                    cfg.eval_mag_err, gen, zhead.phot_sigma)

        pred0 = run_models(lr, err, phot)
        L_pred0 = {k: _line_flux(v, seg, wh) for k, v in pred0.items()}
        L_true0 = _line_flux(hr, seg, wh)
        # If a model does not recover the line at all, the ratio below is
        # measuring noise and the test is undefined. Every stage has to pass,
        # so the stages are compared on the same sources.
        if L_true0 <= 0 or any(v < cfg.min_recovered_frac * L_true0 or v <= 0
                               for v in L_pred0.values()):
            continue

        scale = calibration_scale(hr, lr)
        if scale is None or not np.isfinite(scale) or scale == 0:
            continue

        cont = gaussian_filter1d(hr, 101)
        for f in cfg.factors:
            hr_p = hr.copy()
            hr_p[seg] = cont[seg] + f * (hr[seg] - cont[seg])
            # Forward-model the truth change into the observation, so the
            # perturbed pair stays physically consistent -- LSF-smoothed to
            # grism resolution and converted into the extraction's units.
            delta = scale * gaussian_filter1d(hr_p - hr, sig_hr_px)
            pred_p = run_models(lr + delta, err, phot)
            for k in stages:
                L_pred_p = _line_flux(pred_p[k], seg, wh)
                if L_pred_p <= 0:
                    continue
                r = np.log(L_pred_p / L_pred0[k]) / np.log(f)
                if np.isfinite(r):
                    records[k].append((float(f), float(best_snr[i]), float(r)))
        n_used += 1

    print(f"usable sources (line recovered above "
          f"{cfg.min_recovered_frac:.0%} of truth): {n_used}")
    summary: dict = {"n_sources": n_used}
    for k in stages:
        rec = np.array(records[k]).reshape(-1, 3)
        out: dict = {"per_factor": {}, "by_snr": []}
        print(f"\n{k.upper()}")
        for f in cfg.factors:
            arr = rec[rec[:, 0] == f, 2]
            if not arr.size:
                continue
            out["per_factor"][f] = {
                "n": int(arr.size), "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }
            print(f"  f = {f}: N={arr.size}  response exponent r: "
                  f"median {np.median(arr):.3f}  "
                  f"p25/p75 = {np.percentile(arr, 25):.3f}/"
                  f"{np.percentile(arr, 75):.3f}")
        out["overall_median_r"] = (float(np.median(rec[:, 2])) if len(rec)
                                   else float("nan"))
        print(f"  OVERALL response exponent: {out['overall_median_r']:.3f} "
              "(1 = reads the data, 0 = recites the prior)")
        # The aggregate mixes lines the data constrain well with lines they
        # barely constrain. Split by detectability, in equal-count bins.
        if len(rec) >= 30:
            edges = np.quantile(rec[:, 1], [0.0, 1 / 3, 2 / 3, 1.0])
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                m = (rec[:, 1] >= lo) & (rec[:, 1] <= hi)
                row = {"snr_lo": float(lo), "snr_hi": float(hi),
                       "n": int(m.sum()), "median": float(np.median(rec[m, 2]))}
                out["by_snr"].append(row)
                print(f"  best-line S/N {lo:5.1f} - {hi:5.1f}: N={row['n']:4d}  "
                      f"median r {row['median']:.3f}")
        summary[k] = out
    # the SR1 numbers stay at the top level, where earlier callers read them
    summary.update(summary["sr1"])
    return summary
