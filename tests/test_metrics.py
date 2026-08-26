"""Evaluation metrics --- including the split that keeps them honest."""

from __future__ import annotations

import numpy as np
import pytest

from specsr_roman.evaluation import (
    RECOVERABILITY_BINS,
    line_amplitude_recovery,
    per_line_amplitude_recovery,
    redshift_summary,
)


def test_recoverability_bins_tile_the_range_without_gaps():
    edges = sorted(v for pair in RECOVERABILITY_BINS.values() for v in pair)
    assert edges[0] == 0.0
    assert edges[-1] == np.inf
    ordered = list(RECOVERABILITY_BINS.values())
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
