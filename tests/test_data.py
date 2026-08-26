"""Dataset, splits and the resampling primitives."""

from __future__ import annotations

import numpy as np
import pytest

from specsr_roman.data import (
    RomanFixedGridDataset,
    filter_split_min_lines,
    fluxconserve_resample,
    get_or_make_group_split,
    interp_ascending,
    normalize,
    select_bands,
)
from specsr_roman.grids import ROMAN_MEDIUM_BANDS


# --- transforms ----------------------------------------------------------
def test_normalize_is_zero_mean_unit_std():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    xn, m, s = normalize(x)
    assert xn.mean() == pytest.approx(0.0)
    assert xn.std() == pytest.approx(1.0)
    assert (xn * s + m) == pytest.approx(x)


def test_normalize_floors_a_constant_row():
    xn, _, s = normalize(np.ones(10))
    assert np.isfinite(xn).all()
    assert s > 0


@pytest.mark.parametrize("offset", [0.0, 2.5])
def test_fluxconserve_resample_preserves_a_narrow_line(offset):
    """The property the Diffsky SEDs need: a line must survive a coarse rebin.

    Plain interpolation onto a grid coarser than the line gets the flux wrong
    by a factor that depends on where the samples happen to land --- roughly
    doubling it when a sample sits on the peak (``offset=0``), and losing most
    of it when the samples straddle the line (``offset=2.5``, half a bin). Both
    are tested, because either error is invisible downstream: the target simply
    has the wrong line strength and the model dutifully learns it.
    """
    wave = np.linspace(10000, 11000, 20001)              # 0.05 A sampling
    center = 10500.0 + offset
    flux = np.exp(-0.5 * ((wave - center) / 1.0) ** 2)   # 1 A sigma line
    coarse = np.linspace(10000, 11000, 201)              # 5 A sampling

    truth = np.trapezoid(flux, wave)
    rebinned = np.trapezoid(fluxconserve_resample(wave, flux, coarse), coarse)
    assert rebinned == pytest.approx(truth, rel=0.02)

    naive = np.trapezoid(np.interp(coarse, wave, flux), coarse)
    assert abs(naive - truth) / truth > 0.5


def test_interp_ascending_handles_a_descending_grid():
    """Roman's DLDP_A_1 is negative, so extracted wavelengths run backwards.

    np.interp does not check, and silently returns the edge value everywhere,
    producing a flat 'spectrum' that carries no information at all.
    """
    wave_desc = np.linspace(19000, 10000, 500)
    flux = np.linspace(0.0, 1.0, 500)
    target = np.array([11000.0, 15000.0, 18000.0])

    good = interp_ascending(target, wave_desc, flux)
    assert np.all(np.diff(good) < 0)                 # tracks the real trend
    assert len(np.unique(good)) == 3

    naive = np.interp(target, wave_desc, flux)
    assert len(np.unique(naive)) == 1                # the bug: constant output


# --- splits --------------------------------------------------------------
def test_group_split_is_deterministic_and_id_keyed(synthetic_dataset, tmp_path):
    d = np.load(synthetic_dataset)
    a = get_or_make_group_split(synthetic_dataset, d["ids"],
                                split_dir=str(tmp_path / "a"), verbose=False)
    b = get_or_make_group_split(synthetic_dataset, d["ids"],
                                split_dir=str(tmp_path / "b"), verbose=False)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # No object may appear on both sides -- that is the leak this exists to stop.
    assert not set(d["ids"][a[0]]) & set(d["ids"][a[1]])


def test_group_split_membership_is_stable_under_growth(synthetic_dataset, tmp_path):
    """Adding rows must not move an existing galaxy across the split.

    Membership is a pure hash of the id, so a model trained on an older build
    can still be evaluated on the newer test set.
    """
    d = np.load(synthetic_dataset)
    ids = d["ids"]
    train_a, test_a, _ = get_or_make_group_split(
        synthetic_dataset, ids, split_dir=str(tmp_path / "a"), verbose=False)
    grown = np.concatenate([ids, np.arange(9000, 9050, dtype=np.int64)])
    train_b, test_b, _ = get_or_make_group_split(
        synthetic_dataset, grown, split_dir=str(tmp_path / "b"), verbose=False)

    was_test = set(ids[test_a].tolist())
    now_test = set(grown[test_b].tolist())
    assert was_test <= now_test


def test_filter_min_lines_only_removes(synthetic_dataset):
    d = np.load(synthetic_dataset)
    tr = np.arange(len(d["ids"]))
    te = np.arange(len(d["ids"]))
    tr2, te2 = filter_split_min_lines(tr, te, d["redshift"], d["wavelength_high"],
                                      2, verbose=False)
    assert set(tr2) <= set(tr) and set(te2) <= set(te)
    # min_lines=0 is a no-op
    tr3, te3 = filter_split_min_lines(tr, te, d["redshift"],
                                      d["wavelength_high"], 0, verbose=False)
    assert np.array_equal(tr3, tr)


# --- dataset -------------------------------------------------------------
def test_dataset_item_shapes_and_channels(synthetic_dataset):
    ds = RomanFixedGridDataset(synthetic_dataset, with_phot=True,
                               phot_tier="medium", verbose=False)
    low, high, high_err, z, m, s, line_snr, phot = ds[0]
    assert low.shape == (2, 2500)                    # [flux, err] on the HR grid
    assert high.shape == (2500,)
    assert line_snr.shape == (10,)                   # one per SR1_LINES_AA
    assert phot.shape == (3,)                        # Roman Medium tier
    assert ds.n_phot == 3
    # Targets are noiseless simulated SEDs; the error column is structurally zero.
    assert float(high_err.abs().max()) == 0.0


def test_dataset_error_channel_shares_the_flux_scale(synthetic_dataset):
    """Channel ratio must be per-pixel S/N.

    Normalising the error channel independently would destroy exactly the
    quantity SR1 reads to distinguish a weak line from a noise bump.
    """
    ds = RomanFixedGridDataset(synthetic_dataset, verbose=False)
    low, *_ = ds[0]
    flux_n, err_n = low[0].numpy(), low[1].numpy()
    raw_scale = np.nanstd(ds.lo_raw[0])
    assert err_n.mean() == pytest.approx(ds.err_raw[0].mean() / raw_scale, rel=1e-4)
    assert flux_n.std() == pytest.approx(1.0, rel=1e-3)


def test_dataset_rejects_information_free_input(tmp_path, synthetic_dataset):
    """Constant LR rows mean the extraction is broken; training must refuse.

    Left alone the model learns to emit the prior mean and every loss curve
    looks healthy, which is the worst possible failure mode.
    """
    d = dict(np.load(synthetic_dataset, allow_pickle=True))
    d["flux_low"] = np.ones_like(d["flux_low"])
    bad = tmp_path / "constant.npz"
    np.savez(bad, **d)
    with pytest.raises(RuntimeError, match="constant"):
        RomanFixedGridDataset(str(bad), verbose=False)


def test_select_bands_slices_the_requested_tier():
    phot = np.arange(14 * 3, dtype=float).reshape(3, 14)
    out, keep = select_bands(phot, "medium")
    assert keep == ROMAN_MEDIUM_BANDS
    assert out.shape == (3, 3)
    assert np.array_equal(out[0], phot[0, list(ROMAN_MEDIUM_BANDS)])
