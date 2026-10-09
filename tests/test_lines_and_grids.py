"""Constants that other code indexes into positionally."""

from __future__ import annotations

import numpy as np
import pytest

from specsr_roman.grids import (
    MAX_PHOT_BANDS,
    N_HR,
    N_LR,
    PHOT_BANDS,
    PHOT_TIERS,
    ROMAN_MEDIUM_BANDS,
    WAVE_HR,
    WAVE_LR,
    phot_flux_sigma,
    resolve_phot_tier,
)
from specsr_roman.lines import (
    LINE_LIST_REST_AA,
    SR1_LINES_AA,
    STRONG_LINES_AA,
    angstrom_to_micron,
    count_strong_lines,
)


def test_grid_shapes_and_monotonicity():
    assert (N_LR, N_HR) == (len(WAVE_LR), len(WAVE_HR)) == (864, 2500)
    assert np.all(np.diff(WAVE_LR) > 0)
    assert np.all(np.diff(WAVE_HR) > 0)
    # Both grids span the same band, so an LR row interpolates onto HR without
    # extrapolating at either end.
    assert WAVE_LR[0] == WAVE_HR[0] == 10000.0
    assert WAVE_HR[-1] == 19300.0


def test_line_list_has_no_duplicate_wavelengths():
    # Duplicates would waste SR2 attention capacity on identical tokens.
    waves = [w for _, w in LINE_LIST_REST_AA]
    assert len(waves) == len(set(waves))


def test_sr1_line_order_is_ascending():
    # The dataset's line_snr columns follow this tuple and the losses index it
    # positionally, so a reorder silently mislabels every recoverability weight.
    assert list(SR1_LINES_AA) == sorted(SR1_LINES_AA)


def test_strong_lines_are_a_subset_of_sr1_lines():
    assert set(STRONG_LINES_AA) <= set(SR1_LINES_AA)


def test_count_strong_lines_matches_band_coverage():
    lo, hi = WAVE_HR[0], WAVE_HR[-1]
    # z = 0: every strong line is blueward of 1 micron, so none are in band.
    assert count_strong_lines([0.0], lo, hi)[0] == 0
    # z = 2: Ha at 1.97 um has left the band; [OIII], Hb, [OII] are in it.
    assert count_strong_lines([2.0], lo, hi)[0] == 3
    # z = 1: Ha and [OIII] in band -> the line-pair-identifiable regime.
    assert count_strong_lines([1.0], lo, hi)[0] == 2


def test_angstrom_to_micron_is_float32():
    out = angstrom_to_micron([10000.0])
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(1.0)


def test_phot_tiers_use_only_survey_bands():
    assert len(PHOT_BANDS) == 14
    assert all(PHOT_BANDS[i].startswith("roman_") for i in ROMAN_MEDIUM_BANDS)
    # The Rubin tier is the Medium tier plus ugrizy and nothing else: the
    # five Roman bands no grism tier delivers stay out.
    rubin = PHOT_TIERS["medium_rubin"]
    assert set(rubin) - set(ROMAN_MEDIUM_BANDS) == set(range(6))
    assert len(rubin) == MAX_PHOT_BANDS == 9


def test_depth_noise_is_rubin_only_and_scales_with_depth():
    names = [PHOT_BANDS[i] for i in PHOT_TIERS["medium_rubin"]]
    y10, y1 = phot_flux_sigma(names), phot_flux_sigma(names, 1.25)
    assert (y10[:6] > 0).all() and (y10[6:] == 0).all()
    assert y1[:6] / y10[:6] == pytest.approx(10 ** 0.5, rel=1e-4)
    assert not phot_flux_sigma([PHOT_BANDS[i] for i in ROMAN_MEDIUM_BANDS]).any()


def test_resolve_phot_tier_forms():
    assert resolve_phot_tier("medium") == ROMAN_MEDIUM_BANDS
    assert resolve_phot_tier("8,9,11") == (8, 9, 11)
    assert resolve_phot_tier("medium_rubin") == (0, 1, 2, 3, 4, 5, 8, 9, 11)
    assert resolve_phot_tier(None) is None
    # More bands than any survey tier delivers with the grism is refused.
    with pytest.raises(ValueError, match="ceiling"):
        resolve_phot_tier(",".join(str(b) for b in range(MAX_PHOT_BANDS + 1)))
    with pytest.raises(ValueError):
        resolve_phot_tier("not-a-tier")
