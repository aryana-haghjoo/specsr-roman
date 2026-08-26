"""End-to-end tests against the published checkpoints.

Marked ``needs_hub``: they download ~11 MB of weights on first run. Enable
with ``pytest -m needs_hub``.
"""

from __future__ import annotations

import numpy as np
import pytest

from specsr_roman.grids import WAVE_HR, WAVE_LR

pytestmark = pytest.mark.needs_hub


@pytest.fixture(scope="module")
def pipeline():
    from specsr_roman.inference import RomanPipeline
    return RomanPipeline.from_pretrained(device="cpu")


def _fake_spectrum(z: float = 1.0, amplitude: float = 5.0, seed: int = 0):
    """A flat continuum with Halpha and [OIII] placed at redshift ``z``."""
    rng = np.random.default_rng(seed)
    flux = np.ones_like(WAVE_LR)
    for rest in (6563.0, 5007.0):
        lam = rest * (1 + z)
        flux = flux + amplitude * np.exp(-0.5 * ((WAVE_LR - lam) / 12.0) ** 2)
    err = np.full_like(WAVE_LR, 0.1)
    return flux + rng.normal(0, 0.1, len(flux)), err


def test_pipeline_runs_on_a_single_spectrum(pipeline):
    flux, err = _fake_spectrum()
    phot = np.array([1.0, 1.1, 1.2])          # Roman Medium tier
    out = pipeline.predict(flux, err, phot=phot)
    assert out.flux_sr.shape == WAVE_HR.shape
    assert np.isfinite(out.flux_sr).all()
    assert np.isfinite(out.flux_sr_err).all()
    assert 0.0 <= out.z <= 3.1
    assert out.pz is not None and out.pz.sum() == pytest.approx(1.0, rel=1e-4)
    assert len(out.line_names) == out.presence.shape[0]


def test_pipeline_batches(pipeline):
    flux = np.stack([_fake_spectrum(z, seed=i)[0]
                     for i, z in enumerate((0.8, 1.2, 1.6))])
    err = np.full_like(flux, 0.1)
    phot = np.ones((3, 3))
    outs = pipeline.predict(flux, err, phot=phot)
    assert len(outs) == 3
    assert all(np.isfinite(o.flux_sr).all() for o in outs)


def test_pipeline_runs_without_photometry(pipeline):
    """Grism-only is much weaker but must still work --- it is the honest floor."""
    flux, err = _fake_spectrum()
    out = pipeline.predict(flux, err, phot=None)
    assert np.isfinite(out.flux_sr).all()


def test_wrong_photometry_width_is_rejected_clearly(pipeline):
    """A silent band-count mismatch would mis-scale every colour."""
    flux, err = _fake_spectrum()
    with pytest.raises(ValueError, match="photometric bands"):
        pipeline.predict(flux, err, phot=np.ones(14))


def test_sr2_changes_something_relative_to_sr1(pipeline):
    """SR2 is initialised as the identity on SR1; a trained one must not be."""
    flux, err = _fake_spectrum(amplitude=8.0)
    out = pipeline.predict(flux, err, phot=np.array([1.0, 1.1, 1.2]))
    assert not np.allclose(out.flux_sr, out.flux_sr1)


def test_canonical_checkpoints_resolve():
    from specsr_roman.checkpoints import CANONICAL_CHAIN, resolve_checkpoint
    for name in CANONICAL_CHAIN.values():
        assert resolve_checkpoint(name).exists()


def test_output_is_on_the_input_flux_scale(pipeline):
    """De-normalisation must restore the mean, not just the scale.

    `_prepare` standardises each spectrum to zero mean and unit variance.
    Undoing that with the standard deviation alone returns a spectrum offset
    from the input by its continuum level -- it hovers near zero and looks
    like the model destroyed the continuum. Regression test for that.
    """
    flux, err = _fake_spectrum(z=1.0, amplitude=3.0)
    flux = flux * 7.0 + 40.0          # arbitrary offset and scale
    out = pipeline.predict(flux, err * 7.0, phot=np.array([1.0, 1.1, 1.2]))

    in_med = float(np.median(flux))
    for name, arr in (("flux_sr", out.flux_sr), ("flux_sr1", out.flux_sr1)):
        got = float(np.median(arr))
        assert got == pytest.approx(in_med, rel=0.35), (
            f"{name} median {got:.3g} is not on the input scale {in_med:.3g}")


def test_output_scales_with_the_input(pipeline):
    """Doubling the input must double the output; the pipeline is scale-free."""
    flux, err = _fake_spectrum(z=1.0, amplitude=3.0)
    phot = np.array([1.0, 1.1, 1.2])
    a = pipeline.predict(flux, err, phot=phot)
    b = pipeline.predict(flux * 3.0, err * 3.0, phot=phot)
    assert np.allclose(b.flux_sr, a.flux_sr * 3.0, rtol=1e-3, atol=1e-4)
    assert b.z == pytest.approx(a.z, abs=1e-6)


def test_uncertainty_is_offset_invariant(pipeline):
    """An additive offset changes the level, never the error bar."""
    flux, err = _fake_spectrum(z=1.0, amplitude=3.0)
    phot = np.array([1.0, 1.1, 1.2])
    a = pipeline.predict(flux, err, phot=phot)
    b = pipeline.predict(flux + 100.0, err, phot=phot)
    assert np.allclose(b.flux_sr_err, a.flux_sr_err, rtol=1e-3, atol=1e-5)
