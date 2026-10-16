"""Calibration and unsupported-emission diagnostics on frozen predictions.

These describe the supplied partition. Evaluating a legacy validation cache
never upgrades it to an independent test, even when no parameter is fitted.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..lines import SR1_LINES_AA
from .metrics import detected_line_mask


def pz_calibration(pz, centers, truth):
    """Central-interval coverage and reproducible randomized PIT for binned P(z)."""
    p = np.asarray(pz, dtype=float)
    centers, truth = np.asarray(centers), np.asarray(truth)
    if (
        p.ndim != 2
        or p.shape != (len(truth), len(centers))
        or len(centers) < 2
        or np.any(np.diff(centers) <= 0)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p.sum(1) <= 0).any()
    ):
        raise ValueError("invalid P(z) dimensions, probability mass, or grid")
    p = p / p.sum(1, keepdims=True)
    mid = (centers[:-1] + centers[1:]) / 2
    edges = np.r_[centers[0] - (mid[0] - centers[0]), mid, centers[-1] + (centers[-1] - mid[-1])]
    inside = np.isfinite(truth) & (truth >= edges[0]) & (truth < edges[-1])
    k = np.clip(np.searchsorted(edges, truth, side="right") - 1, 0, len(centers) - 1)
    cdf = p.cumsum(1)
    before = np.c_[np.zeros(len(p)), cdf[:, :-1]][np.arange(len(p)), k]
    mass = p[np.arange(len(p)), k]
    # Mid-PIT determines central interval membership without Monte Carlo noise.
    mid_pit = before + 0.5 * mass
    pit = before + np.random.default_rng(0).uniform(size=len(p)) * mass
    coverage = {
        str(level): float(
            np.mean(inside & (mid_pit >= (1 - level) / 2) & (mid_pit <= (1 + level) / 2))
        )
        for level in (0.5, 0.68, 0.95, 0.99)
    }
    fraction = np.clip((truth - edges[k]) / (edges[k + 1] - edges[k]), 0, 1)
    continuous_pit = before + fraction * mass
    continuous_coverage = {
        str(level): float(
            np.mean(
                inside & (continuous_pit >= (1 - level) / 2) & (continuous_pit <= (1 + level) / 2)
            )
        )
        for level in (0.5, 0.68, 0.95, 0.99)
    }
    width = np.sqrt(
        np.maximum((p * centers[None, :] ** 2).sum(1) - (p * centers[None, :]).sum(1) ** 2, 0)
    )
    hist, _ = np.histogram(pit[inside], bins=np.linspace(0, 1, 11))
    # Quantization is retained: no claim of sub-bin calibration follows from PIT.
    return {
        "n": len(p),
        "n_outside_grid": int((~inside).sum()),
        "central_bin_midpoint_coverage": coverage,
        "central_uniform_bin_coverage": continuous_coverage,
        "median_pdf_std": float(np.median(width)),
        "randomized_pit_histogram": hist.tolist(),
        "mean_truth_bin_nll": float(-np.log(np.maximum(mass[inside], 1e-300)).mean()),
        "note": "Reports midpoint and uniform-within-bin central coverage; "
        "randomized PIT ignores truth positions within bins.",
    }


def grouped_interval(values, ids, seed=0, draws=1000):
    """Equal-galaxy mean and bootstrap interval, keeping repeated visits together."""
    values, ids = np.asarray(values), np.asarray(ids)
    _, inverse = np.unique(ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    means = sums / counts
    rng = np.random.default_rng(seed)
    boots = np.array(
        [rng.choice(means, size=len(means), replace=True).mean() for _ in range(draws)]
    )
    return {
        "galaxy_balanced_mean": float(means.mean()),
        "n_galaxies": len(means),
        "bootstrap_95_interval": np.quantile(boots, [0.025, 0.975]).tolist(),
    }


def spectral_calibration(pred, truth, sigma, line_snr, z, wave, ids):
    """Marginal pixel coverage, separated into detected-line and other pixels."""
    pred, truth, sigma = map(np.asarray, (pred, truth, sigma))
    if (
        pred.shape != truth.shape
        or sigma.shape != truth.shape
        or not all(np.isfinite(a).all() for a in (pred, truth, sigma))
        or np.any(sigma <= 0)
    ):
        raise ValueError("prediction, target and positive sigma must have matching shapes")
    residual = np.abs(pred - truth) / sigma
    det = detected_line_mask(wave, z, line_snr)
    out = {}
    for tag, mask in [("all", np.ones_like(det)), ("detected_line", det), ("other_pixels", ~det)]:
        counts = mask.sum(1)
        valid = counts > 0
        row_coverage = np.sum((residual <= 1) & mask, axis=1)[valid] / counts[valid]
        out[tag] = {
            "n_pixels": int(mask.sum()),
            "coverage": {
                str(q): float(np.mean(residual[mask] <= q)) if mask.any() else None
                for q in (1.0, 1.95996398454, 3.0)
            },
            "one_sigma_by_galaxy": grouped_interval(row_coverage, np.asarray(ids)[valid])
            if valid.any()
            else None,
        }
    return out


def unsupported_emission(pred, truth, line_snr, z, wave, threshold=5.0):
    """Truth-position emission proxies; these are not blind detection false positives.

    Uses the training Gaussian window (sigma=0.005 um) and smooth continuum.
    Reports both intrinsically weak target windows and present-but-undetectable
    lines. Off-grid windows are excluded, not counted as true negatives.
    """
    pred, truth = np.asarray(pred), np.asarray(truth)
    wave = np.asarray(wave)
    ph = pred - gaussian_filter1d(pred, 101, axis=1)
    th = truth - gaussian_filter1d(truth, 101, axis=1)
    drawn, target, inside = [], [], []
    for rest in np.asarray(SR1_LINES_AA) * 1e-4:
        center = (1 + np.asarray(z)) * rest
        profile = np.exp(-0.5 * ((wave[None, :] - center[:, None]) / 0.005) ** 2)
        drawn.append(np.maximum((ph * profile).sum(1), 0))
        target.append(np.maximum((th * profile).sum(1), 0))
        inside.append((center > wave[0] + 0.02) & (center < wave[-1] - 0.02))
    drawn, target, inside = map(lambda x: np.array(x).T, (drawn, target, inside))
    masks = {
        "intrinsically_weak_proxy": inside & (target <= threshold),
        "present_but_snr_below_1": inside & (target > threshold) & (np.asarray(line_snr) < 1),
        "present_but_snr_below_2": inside & (target > threshold) & (np.asarray(line_snr) < 2),
    }
    out = {
        "window_sigma_um": 0.005,
        "emission_sum_threshold": threshold,
        "note": (
            "Correlated/blended truth-centered windows; not a blind-search false-positive rate."
        ),
    }
    for tag, mask in masks.items():
        n = int(mask.sum())
        out[tag] = {
            "n_windows": n,
            "n_drawn_above_threshold": int(((drawn > threshold) & mask).sum()),
            "fraction_above_threshold": float((drawn[mask] > threshold).mean()) if n else None,
            "median_drawn_sum": float(np.median(drawn[mask])) if n else None,
        }
        if tag != "intrinsically_weak_proxy":
            out[tag]["median_recovered_fraction"] = (
                float(np.median(drawn[mask] / target[mask])) if n else None
            )
    return out
