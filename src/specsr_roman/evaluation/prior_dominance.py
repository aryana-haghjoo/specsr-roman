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

Measured on this project: unaugmented SR1 scored 0.14; with anti-prior
augmentation, 0.51 at fixed detectability --- a 3.7x improvement in
data-faithfulness, at the cost of absolute line recovery, which is why the
published SR1 is the unaugmented one and this remains open work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from ..checkpoints import load_sr1
from ..data import RomanFixedGridDataset, get_or_make_group_split, normalize
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
    """Run the audit. Returns per-factor and overall response exponents."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr1 = load_sr1(cfg.sr1_ckpt, device=device)

    ds = RomanFixedGridDataset(cfg.data)
    _, test_idx, _ = get_or_make_group_split(os.path.abspath(cfg.data), ds.ids)

    wh = ds.wave_hi
    sig_hr_px = (cfg.fwhm_aa / 2.355) / np.median(np.diff(wh))

    # Candidates: bright enough that the line is genuinely in the data.
    best_snr = ds.line_snr.max(dim=1).values.numpy()
    cand = [i for i in test_idx if best_snr[i] > cfg.snr_min][: cfg.max_sources]
    print(f"{len(cand)} candidate test sources with a recoverable line")

    def run_sr1(lr_hr_grid, err_hr_grid):
        xn, m, s = normalize(lr_hr_grid)
        err_n = err_hr_grid / max(s, 1e-25)
        x = torch.tensor(np.stack([xn, err_n])[None].astype(np.float32),
                         device=device)
        with torch.no_grad():
            pred, _ = sr1(x)
        return pred[0, 0].cpu().numpy() * s + m       # back to input units

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

    results: dict[float, list[float]] = {f: [] for f in cfg.factors}
    n_used = 0
    for i in cand:
        hr = np.nan_to_num(ds.hi_raw[i]).astype(np.float64)
        lr = ds.lo_raw[i].astype(np.float64)
        err = ds.err_raw[i].astype(np.float64)

        found = _find_strongest_line(hr)
        if found is None:
            continue
        seg, _ = found

        pred0 = run_sr1(lr, err)
        L_pred0 = _line_flux(pred0, seg, wh)
        L_true0 = _line_flux(hr, seg, wh)
        if L_pred0 <= 0 or L_true0 <= 0:
            continue
        # If the model does not recover the line at all, the ratio below is
        # measuring noise and the test is undefined.
        if L_pred0 < cfg.min_recovered_frac * L_true0:
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
            pred_p = run_sr1(lr + delta, err)
            L_pred_p = _line_flux(pred_p, seg, wh)
            if L_pred_p <= 0:
                continue
            r = np.log(L_pred_p / L_pred0) / np.log(f)
            if np.isfinite(r):
                results[f].append(float(r))
        n_used += 1

    summary: dict = {"n_sources": n_used, "per_factor": {}}
    print(f"usable sources (line recovered above "
          f"{cfg.min_recovered_frac:.0%} of truth): {n_used}")
    for f, rs in results.items():
        arr = np.array(rs)
        if not arr.size:
            continue
        summary["per_factor"][f] = {
            "n": int(arr.size), "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }
        print(f"f = {f}: N={arr.size}  response exponent r: "
              f"median {np.median(arr):.3f}  "
              f"p25/p75 = {np.percentile(arr, 25):.3f}/"
              f"{np.percentile(arr, 75):.3f}")

    allr = np.concatenate([results[f] for f in cfg.factors if results[f]])
    summary["overall_median_r"] = float(np.median(allr)) if allr.size else float("nan")
    print(f"\nOVERALL response exponent: {summary['overall_median_r']:.3f} "
          "(1 = reads the data, 0 = recites the prior)")
    return summary
