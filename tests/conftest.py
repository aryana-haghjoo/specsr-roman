"""Shared fixtures.

Tests that need real data or network access are marked and skipped by default,
so ``pytest`` is fast and offline on a clean checkout. Point
``SPECSR_ROMAN_TEST_DATA`` at a built dataset to enable the data-backed tests, and
``--runslow``/network markers for the rest.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from specsr_roman.grids import N_HR, N_LR, WAVE_HR, WAVE_LR


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory, rng):
    """A small, valid dataset npz with the same schema as a real build.

    Emission lines are placed at real rest wavelengths and redshifted, so the
    per-line S/N labels exercise the same code path as production data rather
    than a degenerate one.
    """
    n = 24
    z = rng.uniform(0.4, 2.4, n)
    hr = np.zeros((n, N_HR), dtype=np.float32)
    lo = np.zeros((n, N_LR), dtype=np.float32)
    err = np.full((n, N_LR), 0.05, dtype=np.float32)

    for i in range(n):
        cont = 1.0 + 0.1 * np.sin(WAVE_HR / 3000.0)
        for rest in (4861.0, 5007.0, 6563.0):
            lam = rest * (1 + z[i])
            cont = cont + 6.0 * np.exp(-0.5 * ((WAVE_HR - lam) / 25.0) ** 2)
        hr[i] = cont
        lo[i] = np.interp(WAVE_LR, WAVE_HR, cont) + rng.normal(0, 0.05, N_LR)

    path = tmp_path_factory.mktemp("data") / "synthetic.npz"
    np.savez(path,
             ids=np.arange(1000, 1000 + n, dtype=np.int64),
             redshift=z.astype(np.float32),
             flux_low=lo, flux_low_err=err, flux_high=hr,
             phot=rng.uniform(1e-3, 1.0, (n, 14)),
             ab_h158=rng.uniform(19, 22.5, n),
             snr=rng.uniform(1, 10, n),
             visit=np.zeros(n, dtype=np.int32), sca=np.ones(n, dtype=np.int16),
             wavelength_low=WAVE_LR, wavelength_high=WAVE_HR,
             phot_bands=np.array([f"b{i}" for i in range(14)]))
    return str(path)


@pytest.fixture(scope="session")
def real_dataset():
    path = os.environ.get("SPECSR_ROMAN_TEST_DATA")
    if not path or not os.path.exists(path):
        pytest.skip("set SPECSR_ROMAN_TEST_DATA to a built dataset npz")
    return path
