"""Evaluation metrics --- including the split that keeps them honest."""

from __future__ import annotations

import numpy as np
import pytest

from specsr_roman.evaluation import (
    HLSS_LINE_FLUX_LIMIT,
    MIN_BEST_LINE_SNR,
    RECOVERABILITY_BINS,
    detected_line_mask,
    hlss_cosmology_sample,
    line_amplitude_recovery,
    per_line_amplitude_recovery,
    redshift_summary,
    true_line_flux_cgs,
)
from specsr_roman.grids import PHOTPLAM_F158, WAVE_HR


def test_recoverability_bins_start_at_the_cut_and_leave_no_gaps():
    ordered = list(RECOVERABILITY_BINS.values())
    assert ordered[0][0] == MIN_BEST_LINE_SNR    # nothing below the cut is binned
    assert ordered[-1][1] == np.inf
    for (_, hi), (lo, _) in zip(ordered, ordered[1:], strict=False):
        assert hi == lo                      # contiguous, no S/N falls through


def test_redshift_summary_reports_all_four_numbers():
    z_true = np.linspace(0.5, 2.5, 200)
    z_pred = z_true + 0.001 * (1 + z_true)
    # Planted outliers, offset far enough that |dz|/(1+z) > 0.15 at every z in
    # the range (a fixed wrong value would fall under the threshold for the
    # low-z end of the sample).
    z_pred[:20] = z_true[:20] + 0.5 * (1 + z_true[:20])
    met = redshift_summary(z_pred, z_true)
    assert set(met) == {"nmad", "median_abs_dz", "catastrophic_frac", "n"}
    assert met["n"] == 200
    assert met["catastrophic_frac"] == pytest.approx(0.10, abs=0.01)
    # NMAD is robust: 10% outliers must not blow up the core scatter.
    assert met["nmad"] < 0.01


def test_line_amplitude_recovery_scores_a_perfect_model_at_one():
    n, length = 40, 500
    truth = np.zeros((n, length))
    truth[:, 240:260] = 10.0                 # a "line" above the threshold
    line_snr = np.tile(np.array([[0.5, 2.0, 4.0, 20.0]]), (n, 1))
    out = line_amplitude_recovery(truth.copy(), truth, line_snr)
    scored = [v for v in out.values() if v["n"]]
    assert scored, "no rows were scored"
    for v in scored:
        assert v["median"] == pytest.approx(1.0)


def test_line_amplitude_recovery_detects_a_half_amplitude_model():
    n, length = 40, 500
    truth = np.zeros((n, length))
    truth[:, 240:260] = 10.0
    line_snr = np.full((n, 4), 20.0)
    out = line_amplitude_recovery(0.5 * truth, truth, line_snr)
    assert out["strong"]["median"] == pytest.approx(0.5)


def test_per_line_recovery_attributes_failure_to_one_line():
    """The diagnostic companion metric: which transition went wrong."""
    wave_um = np.linspace(1.0, 1.93, 800)
    z = np.full(10, 1.0)
    rest = np.array([6563.0, 5007.0]) * 1e-4
    truth = np.zeros((10, 800))
    for r in rest:
        truth += 40.0 * np.exp(-0.5 * ((wave_um[None, :] - r * 2.0) / 0.005) ** 2)
    # Model reproduces the target exactly -> ratio 1 wherever a line is scored.
    line_snr = np.full((10, 2), 20.0)
    out = per_line_amplitude_recovery(truth, truth, line_snr, z, wave_um,
                                      line_rest_um=rest)
    scored = [v for v in out.values() if v["n"]]
    assert scored
    for v in scored:
        assert v["median"] == pytest.approx(1.0, rel=1e-3)



def _spectrum_with_halpha(z, flux_cgs, ab=21.0, scale=37.0):
    """Flat continuum at ``ab`` plus an H-alpha + [N II] 6585 pair, in arbitrary units."""
    cont = 10 ** (-0.4 * (ab + 48.6)) * 2.99792458e18 / PHOTPLAM_F158 ** 2
    f = np.full(len(WAVE_HR), cont)
    for rest, amp in ((6564.61, flux_cgs), (6585.27, 0.5 * flux_cgs)):
        sig = 3.0 * (1 + z)
        f += amp * np.exp(-0.5 * ((WAVE_HR - rest * (1 + z)) / sig) ** 2) \
            / (sig * np.sqrt(2 * np.pi))
    return f * scale


@pytest.mark.parametrize("z", [1.05, 1.5, 1.9])
def test_true_line_flux_is_calibrated_and_excludes_nii(z):
    # The arbitrary `scale` must drop out through the H158 anchor, and the
    # neighbouring [N II] line, half as bright, must not leak into the core.
    truth = 2.0e-16
    f = _spectrum_with_halpha(z, truth)
    got = true_line_flux_cgs(WAVE_HR, f[None], [21.0], [z], 6564.61)[0]
    assert got == pytest.approx(truth, rel=0.03)


def test_true_line_flux_is_zero_for_a_line_outside_the_band():
    f = _spectrum_with_halpha(1.5, 2.0e-16)
    assert true_line_flux_cgs(WAVE_HR, f[None], [21.0], [0.3], 6564.61)[0] == 0.0
    assert true_line_flux_cgs(WAVE_HR, f[None], [21.0], [2.5], 6564.61)[0] == 0.0


def test_hlss_selection_uses_the_right_line_in_each_redshift_range():
    lim = HLSS_LINE_FLUX_LIMIT
    z = np.array([0.8, 1.5, 1.5, 2.5, 2.5])
    ha = np.array([5, 5, 0.5, 5, 0]) * lim
    oiii = np.array([5, 0, 5, 0, 5]) * lim
    assert hlss_cosmology_sample(z, ha, oiii).tolist() == [
        False, True, False, False, True]


def test_detected_line_mask_keeps_only_lines_above_the_cut():
    wave_um = np.linspace(1.0, 1.93, 2500)
    z = np.array([1.5, 1.5])
    rest = (5007.0, 6563.0)
    line_snr = np.array([[0.5, 5.0],         # only H-alpha detected
                         [1.0, 1.9]])        # nothing detected
    mask = detected_line_mask(wave_um, z, line_snr, line_rest_aa=rest)
    at = lambda lam: int(np.argmin(np.abs(wave_um - lam * 1e-4 * 2.5)))  # noqa: E731
    assert mask[0, at(6563.0)] and mask[0, at(6583.0)]   # [N II] rides with H-alpha
    assert not mask[0, at(5007.0)]
    assert not mask[1].any()


def test_line_amplitude_recovery_ignores_undetected_lines_when_given_redshifts():
    # Two lines per row. The model nails the detected one and misses the
    # undetected one entirely; only the first may enter the score.
    wave_um = np.linspace(1.0, 1.93, 2500)
    z = np.full(8, 1.5)
    centre = {lam: lam * 1e-4 * 2.5 for lam in (5007.0, 6563.0)}
    line = lambda lam: 40.0 * np.exp(                      # noqa: E731
        -0.5 * ((wave_um - centre[lam]) / 0.0008) ** 2)
    truth = np.tile(line(5007.0) + line(6563.0), (8, 1))
    pred = np.tile(line(6563.0), (8, 1))
    line_snr = np.zeros((8, 10))
    line_snr[:, 3] = 8.0                                   # H-alpha column
    out = line_amplitude_recovery(pred, truth, line_snr, z=z, wave_um=wave_um)
    assert out["strong"]["median"] == pytest.approx(1.0, abs=0.01)
    blind = line_amplitude_recovery(pred, truth, line_snr)
    assert blind["strong"]["median"] == pytest.approx(0.5, abs=0.02)

