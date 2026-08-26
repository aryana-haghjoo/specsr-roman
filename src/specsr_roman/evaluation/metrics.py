"""Evaluation metrics.

The one thing to understand before reading any number this pipeline produces:
**line recovery must be reported split by recoverability.** A single median
amplitude ratio mixes lines the data clearly shows with lines it cannot
possibly show, and a model can improve that average either by getting better
or by hallucinating harder. The bins below separate the two, and the
``unrecoverable`` bin is the control: a well-behaved model scores near zero
there. A model scoring 0.3 in that bin is inventing lines, however good its
``strong`` number looks.

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

from ..lines import SR1_LINES_AA

__all__ = ["RECOVERABILITY_BINS", "line_amplitude_recovery",
           "per_line_amplitude_recovery", "redshift_summary",
           "line_snr_from_spectrum"]

#: Integrated line S/N in the LR input, and what each range means. Edges match
#: the published evaluation; ``strong`` is open-ended above 6.
RECOVERABILITY_BINS: dict[str, tuple[float, float]] = {
    "unrecoverable": (0.0, 1.0),
    "marginal": (1.0, 3.0),
    "good": (3.0, 6.0),
    "strong": (6.0, np.inf),
}


def line_amplitude_recovery(pred, truth, line_snr, line_thresh: float = 5.0,
                            bins: dict[str, tuple[float, float]] | None = None,
                            ) -> dict[str, dict]:
    """Median recovered line-flux fraction per row, binned by recoverability.

    This is the metric the published SR1 -> SR2 numbers are quoted from.

    For each row, "line pixels" are those where the noiseless target exceeds
    ``line_thresh`` in normalised flux. The score is the *summed* predicted
    flux over the summed true flux across those pixels, which measures whether
    the line complex carries the right total amplitude without being fooled by
    a sub-pixel centroid error. Rows are binned by their *best* line's
    integrated S/N, because a row's recoverability is set by the line the data
    actually shows.

    ``pred`` and ``truth`` are ``(N, L)``; ``line_snr`` is ``(N, K)``.
    """
    bins = bins or RECOVERABILITY_BINS
    pred, truth = np.asarray(pred), np.asarray(truth)
    best_snr = np.asarray(line_snr).max(axis=1)

    out: dict[str, dict] = {}
    for name, (lo, hi) in bins.items():
        rows = np.where((best_snr >= lo) & (best_snr < hi))[0]
        ratios = []
        for i in rows:
            line = truth[i] > line_thresh
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
    failure to a specific transition rather than to a row. Only lines actually
    present in the target are scored --- absent lines are the hallucination
    test, which the ``unrecoverable`` bin already covers.
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
