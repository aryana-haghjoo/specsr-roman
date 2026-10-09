"""Evaluation metrics.

Two rules decide what the numbers this pipeline reports are computed on.

**Sample statistics use every held-out spectrum.** Redshift accuracy, the
residuals and the power spectra are measured on the whole test split, with no
cut on signal-to-noise.

**The lines are judged only where they are detected.** A line enters the
line-level results only if its integrated S/N in the grism data reaches
:data:`MIN_BEST_LINE_SNR` (:func:`detected_line_mask`), and the
recoverability bins start there. Below it the data carry no strong signal, so
nothing is concluded about what the model draws: neither that it recovers a
line nor that it invents one.

Two amplitude metrics are provided because they answer different questions.
:func:`line_amplitude_recovery` is the published one --- per *row*, total
predicted flux over total true flux across all line pixels, binned by the
row's best line. :func:`per_line_amplitude_recovery` scores each line
separately in a Gaussian window, which is more diagnostic when you want to
know *which* line a model gets wrong. They do not produce the same numbers and
should not be compared to each other.
"""

from __future__ import annotations

import numpy as np

from ..grids import PHOTPLAM_F158
from ..lines import SR1_LINES_AA

__all__ = ["RECOVERABILITY_BINS", "line_amplitude_recovery",
           "per_line_amplitude_recovery", "redshift_summary",
           "line_snr_from_spectrum", "MIN_BEST_LINE_SNR",
           "DETECTED_LINE_HALF_AA", "detected_line_mask",
           "true_line_flux_cgs",
           "hlss_cosmology_sample", "HLSS_LINE_FLUX_LIMIT",
           "HALPHA_VAC_AA", "OIII_VAC_AA"]

#: Line-flux limit of the reference HLSS galaxy redshift sample
#: [erg s^-1 cm^-2], its 6.5 sigma depth (Wang et al. 2022, sections 1-2).
HLSS_LINE_FLUX_LIMIT: float = 1.0e-16

# Vacuum wavelengths: the Diffsky SEDs place their lines there, and a core
# window of a few Angstrom does not forgive the 1.8 A air/vacuum offset.
HALPHA_VAC_AA: float = 6564.61
OIII_VAC_AA: float = 5008.24

# H158 band limits [A] over which the extraction normalises every SED, so the
# band mean of the noiseless target is what the catalogue magnitude calibrates.
_H158_LO, _H158_HI = 13900.0, 17700.0

#: A line below this integrated S/N in the grism data is not a detection, and
#: the line-level results do not include it. Sample statistics (redshifts,
#: residuals) are not cut on it, and the models are trained on every spectrum.
MIN_BEST_LINE_SNR: float = 2.0

#: Integrated line S/N in the LR input, and what each range means. The bins
#: start at :data:`MIN_BEST_LINE_SNR`; ``strong`` is open-ended above 6.
RECOVERABILITY_BINS: dict[str, tuple[float, float]] = {
    "marginal": (MIN_BEST_LINE_SNR, 3.0),
    "good": (3.0, 6.0),
    "strong": (6.0, np.inf),
}

#: Rest-frame half-width [A] of the region that belongs to a detected line.
#: Wide enough to hold [N II] with H-alpha, which the grism measures as one
#: feature; narrow enough to keep [O III] 4959 and H-beta apart from 5007.
DETECTED_LINE_HALF_AA: float = 30.0

def detected_line_mask(wave_um, z, line_snr,
                       min_snr: float = MIN_BEST_LINE_SNR,
                       line_rest_aa=SR1_LINES_AA,
                       half_rest_aa: float = DETECTED_LINE_HALF_AA) -> np.ndarray:
    """Pixels that belong to a line detected in the grism data, ``(N, L)``.

    A pixel is kept when it lies within ``half_rest_aa`` (rest frame) of a
    labelled line whose own integrated S/N reaches ``min_snr``. Everything a
    spectrum shows outside these regions is a line the data did not detect,
    and is left out of the line metrics. ``line_snr`` is ``(N, K)`` in the
    order of ``line_rest_aa``.
    """
    rest_um = np.asarray(wave_um)[None, :] / (1.0 + np.asarray(z)[:, None])
    det = np.asarray(line_snr) >= min_snr
    mask = np.zeros(rest_um.shape, dtype=bool)
    for k, lam in enumerate(line_rest_aa):
        near = np.abs(rest_um - lam * 1e-4) < half_rest_aa * 1e-4
        mask |= near & det[:, k, None]
    return mask


def line_amplitude_recovery(pred, truth, line_snr, line_thresh: float = 5.0,
                            bins: dict[str, tuple[float, float]] | None = None,
                            z=None, wave_um=None) -> dict[str, dict]:
    """Median recovered line-flux fraction per row, binned by recoverability.

    This is the metric the published SR1 -> SR2 numbers are quoted from.

    For each row, "line pixels" are those where the noiseless target exceeds
    ``line_thresh`` in normalised flux. The score is the *summed* predicted
    flux over the summed true flux across those pixels, which measures whether
    the line complex carries the right total amplitude without being fooled by
    a sub-pixel centroid error. Rows are binned by their *best* line's
    integrated S/N, because a row's recoverability is set by the line the data
    actually shows.

    With ``z`` and ``wave_um`` the line pixels are further restricted to the
    lines that are themselves detected (:func:`detected_line_mask`). This is
    the reported form. Without them every line pixel of a row is scored,
    including lines the grism did not detect.

    ``pred`` and ``truth`` are ``(N, L)``; ``line_snr`` is ``(N, K)``.
    """
    bins = bins or RECOVERABILITY_BINS
    pred, truth = np.asarray(pred), np.asarray(truth)
    best_snr = np.asarray(line_snr).max(axis=1)
    detected = (detected_line_mask(wave_um, z, line_snr)
                if z is not None and wave_um is not None else None)

    out: dict[str, dict] = {}
    for name, (lo, hi) in bins.items():
        rows = np.where((best_snr >= lo) & (best_snr < hi))[0]
        ratios = []
        for i in rows:
            line = truth[i] > line_thresh
            if detected is not None:
                line &= detected[i]
            total = truth[i][line].sum()
            if line.any() and total != 0:
                ratios.append(pred[i][line].sum() / total)
        out[name] = {
            "median": float(np.median(ratios)) if ratios else float("nan"),
            "mean": float(np.mean(ratios)) if ratios else float("nan"),
            "n": len(ratios),
        }
    return out


def per_line_amplitude_recovery(pred, truth, line_snr, z, wave_um,
                                line_rest_um=None, sigma_um: float = 0.005,
                                present_thresh: float = 3.0,
                                bins: dict[str, tuple[float, float]] | None = None,
                                ) -> dict[str, dict]:
    """Median integrated predicted/true flux **per line**, binned by that line's S/N.

    Diagnostic companion to :func:`line_amplitude_recovery`: it attributes a
    failure to a specific transition rather than to a row. Only lines present
    in the target are scored, and the bins start at
    :data:`MIN_BEST_LINE_SNR`, so only detected lines are.
    """
    bins = bins or RECOVERABILITY_BINS
    if line_rest_um is None:
        line_rest_um = np.asarray(SR1_LINES_AA, dtype=np.float64) * 1e-4
    line_rest_um = np.asarray(line_rest_um, dtype=np.float64)

    centers = line_rest_um[None, :] * (1.0 + np.asarray(z)[:, None])
    d2 = (np.asarray(wave_um)[None, None, :] - centers[..., None]) ** 2
    prof = np.exp(-0.5 * d2 / (sigma_um ** 2 + 1e-12))          # (N, K, L)

    f_true = (np.asarray(truth)[:, None, :] * prof).sum(-1)
    f_pred = (np.asarray(pred)[:, None, :] * prof).sum(-1)
    ratio = f_pred / np.clip(f_true, 1e-6, None)
    present = f_true > present_thresh

    out = {}
    for name, (lo, hi) in bins.items():
        m = present & (line_snr >= lo) & (line_snr < hi)
        vals = ratio[m]
        out[name] = {
            "median": float(np.median(vals)) if vals.size else float("nan"),
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "n": int(m.sum()),
        }
    return out


def redshift_summary(z_pred, z_true) -> dict[str, float]:
    r"""NMAD, median :math:`|\Delta z|/(1+z)`, catastrophic fraction, and N.

    Report all of them. NMAD alone describes only the well-behaved core, and a
    model can shrink it while pushing more objects past the catastrophic
    threshold.
    """
    z_pred, z_true = np.asarray(z_pred), np.asarray(z_true)
    dz = (z_pred - z_true) / (1 + z_true)
    return {
        "nmad": float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
        "median_abs_dz": float(np.median(np.abs(dz))),
        "catastrophic_frac": float(np.mean(np.abs(dz) > 0.15)),
        "n": int(len(dz)),
    }


def line_snr_from_spectrum(wave_um, flux, lam_obs_um, half: float = 0.045,
                           core: float = 0.012, sbgap: float = 0.015,
                           sbw: float = 0.03) -> float:
    """Local S/N of one line, measured off a spectrum using local sidebands.

    Used to compare the S/N a line has in the LR input against the S/N it has
    after super-resolution --- the "did this become measurable" question, which
    is distinct from "was its amplitude right".
    """
    wave_um, flux = np.asarray(wave_um), np.asarray(flux)
    near = np.abs(wave_um - lam_obs_um) < half
    if near.sum() < 10:
        return float("nan")
    core_m = np.abs(wave_um - lam_obs_um) < core
    side = ((np.abs(wave_um - lam_obs_um) > sbgap)
            & (np.abs(wave_um - lam_obs_um) < sbgap + sbw))
    if core_m.sum() < 2 or side.sum() < 5:
        return float("nan")
    cont = np.median(flux[side])
    noise = 1.4826 * np.median(np.abs(flux[side] - cont))
    signal = float(np.sum(flux[core_m] - cont))
    return signal / max(noise * np.sqrt(core_m.sum()), 1e-30)


def true_line_flux_cgs(wave_aa, flux_high, ab_h158, z, rest_aa: float,
                       core_aa: float = 9.0,
                       side_aa: tuple[float, float] = (45.0, 100.0)) -> np.ndarray:
    """True integrated flux of one line per row [erg s^-1 cm^-2].

    The noiseless targets carry the simulation's internal flux scale, so each
    row is first calibrated by its catalogue magnitude: the mean of the target
    over the H158 band is set to the f_lambda that ``ab_h158`` implies. The
    line flux is then the continuum-subtracted integral over a rest-frame core
    of ``+/-core_aa``, narrow enough to keep [N II] out of H-alpha and
    [O III] 4959 out of 5007. The continuum is the median of two sidebands
    ``side_aa`` away from the line on either side, which clear the same
    neighbours.

    Rows whose core or sidebands leave the grid, and rows with no usable H158
    flux, get zero: a line that is not in the band has no flux to select on.
    """
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    flux_high = np.asarray(flux_high, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    f_h158 = (10.0 ** (-0.4 * (np.asarray(ab_h158, dtype=np.float64) + 48.6))
              * 2.99792458e18 / PHOTPLAM_F158 ** 2)
    band = (wave_aa > _H158_LO) & (wave_aa < _H158_HI)
    dlam = float(np.median(np.diff(wave_aa)))
    out = np.zeros(len(flux_high))
    for i, (f, z_i) in enumerate(zip(flux_high, z, strict=True)):
        s_band = np.mean(f[band])
        if not (np.isfinite(s_band) and s_band > 0 and np.isfinite(f_h158[i])):
            continue
        d_rest = wave_aa / (1.0 + z_i) - rest_aa
        if d_rest[0] > -side_aa[1] or d_rest[-1] < side_aa[1]:
            continue
        core = np.abs(d_rest) < core_aa
        side = (np.abs(d_rest) > side_aa[0]) & (np.abs(d_rest) < side_aa[1])
        line = np.sum(f[core] - np.median(f[side])) * dlam
        out[i] = max(line, 0.0) * f_h158[i] / s_band
    return out


def hlss_cosmology_sample(z, ha_flux, oiii_flux,
                          flux_limit: float = HLSS_LINE_FLUX_LIMIT) -> np.ndarray:
    """Rows that pass the selection of the HLSS galaxy redshift sample.

    Wang et al. (2022): H-alpha emitters at ``1 < z < 2`` and [O III] emitters
    at ``2 < z < 3``, each above the line-flux limit.
    """
    z = np.asarray(z)
    return (((z > 1.0) & (z < 2.0) & (np.asarray(ha_flux) > flux_limit))
            | ((z > 2.0) & (z < 3.0) & (np.asarray(oiii_flux) > flux_limit)))
