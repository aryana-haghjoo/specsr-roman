"""Scientific invariants for independent splits, audit units and calibration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from specsr_roman.config import SR1Config
from specsr_roman.data.splits import (
    evaluation_split,
    get_or_make_group_split,
    get_or_make_three_way_split,
    training_split,
)
from specsr_roman.evaluation.prior_dominance import recovery_is_usable, restore_prediction
from specsr_roman.evaluation.validation import (
    grouped_interval,
    pz_calibration,
    spectral_calibration,
    unsupported_emission,
)
from specsr_roman.training.common import require_split_provenance, write_split_provenance


def test_reserved_galaxies_never_enter_training_or_selection(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"dataset")
    ids = np.repeat(np.arange(500), 3)
    tr, va, te, _ = get_or_make_three_way_split(str(path), ids, verbose=False)
    assert len(tr) + len(va) + len(te) == len(ids)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not np.intersect1d(ids[a], ids[b]).size
    legacy_tr, legacy_rest, _ = get_or_make_group_split(str(path), ids, verbose=False)
    np.testing.assert_array_equal(tr, legacy_tr)
    np.testing.assert_array_equal(np.sort(np.r_[va, te]), legacy_rest)
    selected = training_split(str(path), ids, len(ids), "three_way")
    np.testing.assert_array_equal(selected[1], va)
    test = evaluation_split(str(path), ids, "three_way", "test")
    np.testing.assert_array_equal(test[1], te)
    grown = np.r_[ids, np.arange(1000, 1500)]
    _, va2, te2, _ = get_or_make_three_way_split(str(path), grown, verbose=False)
    assert set(ids[te]) <= set(grown[te2])
    assert set(ids[va]) <= set(grown[va2])


def test_legacy_checkpoint_cannot_be_labeled_independent_test(tmp_path):
    with pytest.raises(ValueError, match="no independent test"):
        evaluation_split("unused", np.arange(100), "legacy", "test")
    path = tmp_path / "weights.pth"
    path.write_bytes(b"checkpoint")
    data = tmp_path / "dataset"
    data.write_bytes(b"data")
    with pytest.raises(ValueError, match="missing three-way"):
        require_split_provenance(str(path), str(data), "three_way")
    write_split_provenance(path, str(data), SR1Config(split_mode="three_way"), "record")
    require_split_provenance(str(path), str(data), "three_way")
    record = json.loads(path.with_suffix(".provenance.json").read_text())
    assert record["evaluation_partition_used_for_selection"] == "validation"
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        require_split_provenance(str(path), str(data), "three_way")


def test_recovery_gate_is_invariant_to_consistent_units():
    pred = {"sr1": 0.3, "sr2": 0.5}
    assert recovery_is_usable(pred, 1, 1, 0.2)
    # A huge arbitrary extraction-to-SED scale must not change eligibility.
    assert recovery_is_usable({k: v * 1e20 for k, v in pred.items()}, 1, 1e20, 0.2)
    assert not recovery_is_usable({"sr1": 0.1}, 1, 1, 0.2)
    assert not recovery_is_usable(pred, 1, -1, 0.2)
    assert not recovery_is_usable({"sr1": np.nan}, 1, 1, 0.2)


def test_unchanged_network_output_has_no_scale_induced_response():
    normalized = np.array([0.0, 1.0, 0.0])
    a = restore_prediction(normalized, (1.0, 2.0), (10.0, 3.0))
    b = restore_prediction(normalized, (1.0, 5.0), (10.0, 3.0))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(
        restore_prediction(normalized, (1.0, 2.0), (10.0, 3.0), "legacy_lr"),
        restore_prediction(normalized, (1.0, 5.0), (10.0, 3.0), "legacy_lr"),
    )


def test_pz_coverage_distinguishes_calibrated_and_biased_predictions():
    centers = np.linspace(0.005, 0.995, 100)
    truth = np.tile(centers, 10)
    p = np.ones((len(truth), len(centers)))
    result = pz_calibration(p, centers, truth)
    assert result["central_bin_midpoint_coverage"]["0.68"] == pytest.approx(0.68)
    p[:] = 0
    p[:, 0] = 1
    bad = pz_calibration(p, centers, truth)
    assert bad["central_bin_midpoint_coverage"]["0.68"] < 0.02
    with pytest.raises(ValueError):
        pz_calibration(-p, centers, truth)


def test_spectral_coverage_detects_misstated_variance():
    rng = np.random.default_rng(1)
    truth = np.zeros((200, 500))
    pred = rng.normal(size=truth.shape)
    wave = np.linspace(1, 1.93, 500)
    z = np.ones(200)
    snr = np.zeros((200, 10))
    out = spectral_calibration(pred, truth, np.ones_like(truth), snr, z, wave, np.arange(200))
    assert out["all"]["coverage"]["1.0"] == pytest.approx(0.6827, abs=0.01)
    narrow = spectral_calibration(
        pred, truth, np.ones_like(truth) * 0.1, snr, z, wave, np.arange(200)
    )
    assert narrow["all"]["coverage"]["1.0"] < 0.1


def test_grouped_coverage_does_not_overweight_repeated_visits():
    out = grouped_interval(np.array([0.0, 0.0, 0.0, 1.0]), np.array([1, 1, 1, 2]))
    assert out["galaxy_balanced_mean"] == 0.5


def test_unsupported_emission_detects_planted_line_on_flat_target():
    wave = np.linspace(1, 1.93, 2500)
    truth = np.zeros((1, 2500))
    pred = truth.copy()
    pred[0] = 10 * np.exp(-0.5 * ((wave - 1.3126) / 0.001) ** 2)
    out = unsupported_emission(pred, truth, np.zeros((1, 10)), np.array([1.0]), wave)
    assert out["intrinsically_weak_proxy"]["n_drawn_above_threshold"] >= 1
    blank = unsupported_emission(truth, truth, np.zeros((1, 10)), np.array([1.0]), wave)
    assert blank["intrinsically_weak_proxy"]["n_drawn_above_threshold"] == 0


def test_identity_pipeline_tracks_physical_flux_with_lr_restoration():
    """Rescaling is part of the deployed estimator, not itself a network error.

    A fixed-target control intentionally removes this contribution and must
    not be confused with the end-to-end physical response of an identity map.
    """
    from specsr_roman.data.transforms import normalize
    from specsr_roman.evaluation.prior_dominance import _line_flux

    wave = np.linspace(1, 1.93, 2500)
    line = np.exp(-0.5 * ((wave - 1.31) / 0.002) ** 2)
    base = 1 + line
    changed = 1 + 2 * line
    seg = np.abs(wave - 1.31) < 0.01
    x, m, std = normalize(base)
    xp, mp, stdp = normalize(changed)
    a = restore_prediction(x, (m, std), (m, std), "legacy_lr")
    b = restore_prediction(xp, (mp, stdp), (m, std), "legacy_lr")
    response = np.log(_line_flux(b, seg, wave) / _line_flux(a, seg, wave)) / np.log(2)
    assert response == pytest.approx(1)
    fixed = restore_prediction(xp, (mp, stdp), (m, std), "target_fixed")
    assert _line_flux(fixed, seg, wave) / _line_flux(a, seg, wave) == pytest.approx(1)


def test_response_bootstrap_keeps_visits_and_factors_within_galaxies(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from summarize_release_audits import grouped_median_interval

    # One galaxy, two visits and two perturbations: resampling that galaxy
    # cannot alter its visit/factor-pooled median. IID row resampling would.
    result = grouped_median_interval([0.0, 0.1, 0.9, 1.0], [7, 7, 7, 7], draws=100)
    assert result['n_galaxies'] == 1
    assert result['median'] == 0.5
    assert result['bootstrap_95_interval'] == [0.5, 0.5]
