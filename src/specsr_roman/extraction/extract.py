"""Contamination-subtracted optimal extraction of a single target."""

from __future__ import annotations

import numpy as np

from ..data.transforms import fluxconserve_resample, interp_ascending
from ..grids import GRIZLI_BEAM_SIZE, WAVE_HR, WAVE_LR

__all__ = ["extract_target", "ExtractionFailure"]


class ExtractionFailure(Exception):
    """A target could not be extracted usefully. Skip it and move on."""


def extract_target(flt, scene, noisy, err2d, compact_id: int, mag: float,
                   spectrum_1d, min_finite: int = 100):
    """Extract one target, subtracting everything else in the scene.

    Returns ``(wave, flam, flam_err)`` on the native grism sampling.

    Contamination is handled exactly, not modelled approximately: the target
    is re-dispersed alone with its own true SED, and that 2D model is
    subtracted from the full scene. Whatever remains inside the beam is
    genuinely other galaxies' light --- which is the whole point of simulating
    a slitless survey rather than isolated sources.

    Flux calibration comes from a second, flat-f_lambda pass through the same
    beam: dividing the extracted counts by the extracted flat model converts
    to f_lambda while cancelling the trace, the sensitivity curve and the
    optimal-extraction profile in one step. Pixels where the flat model falls
    below 5% of its peak are set to NaN --- the band edges, where that division
    is unstable.
    """
    out = flt.compute_model_orders(id=compact_id, mag=mag,
                                   size=GRIZLI_BEAM_SIZE, compute_size=False,
                                   spectrum_1d=spectrum_1d, is_cgs=False,
                                   store=False, in_place=False)
    # grizli may return either the model array or a (status, array) tuple
    # depending on version -- unpack defensively.
    if isinstance(out, (list, tuple)):
        out = out[1]
    own2d = np.asarray(out, dtype=np.float32).reshape(scene.shape)

    beams = flt.compute_model_orders(id=compact_id, mag=mag,
                                     size=GRIZLI_BEAM_SIZE, compute_size=False,
                                     store=False, in_place=False,
                                     get_beams=["A"])
    beam = beams["A"]
    beam.compute_model()                     # flat f_lambda -> calibration beam
    slx, sly = beam.slx_parent, beam.sly_parent

    sci_c = noisy[sly, slx] - (scene - own2d)[sly, slx]
    ivar = 1.0 / np.clip(err2d[sly, slx], 1e-10, None) ** 2
    beam.init_optimal_profile()
    wave, fnum, ferr = beam.optimal_extract(sci_c, ivar=ivar)
    _, flat_c, _ = beam.optimal_extract(beam.model, ivar=ivar)
    calib = np.where(flat_c > 0.05 * np.nanmax(flat_c), flat_c, np.nan)
    flam, flam_err = fnum / calib, ferr / calib

    ok = np.isfinite(flam) & np.isfinite(flam_err) & (flam_err > 0)
    if ok.sum() < min_finite:
        raise ExtractionFailure(f"only {ok.sum()} usable pixels")
    return wave[ok], flam[ok], flam_err[ok]


def to_fixed_grids(wave, flam, flam_err, sed_wave, sed_flux,
                   wave_lr=None, wave_hr=None, min_finite: int = 100):
    """Native extraction + true SED -> the shared LR and HR grids.

    ``interp_ascending`` rather than ``np.interp``: Roman's ``DLDP_A_1`` is
    negative so ``wave`` comes out descending, and plain ``np.interp`` would
    return a constant. See :func:`specsr_roman.data.transforms.interp_ascending`.
    """
    wave_lr = WAVE_LR if wave_lr is None else wave_lr
    wave_hr = WAVE_HR if wave_hr is None else wave_hr

    lr = interp_ascending(wave_lr, wave, flam)
    lr_err = interp_ascending(wave_lr, wave, flam_err)
    lr_ok = np.isfinite(lr)
    if lr_ok.sum() < min_finite or np.nanstd(lr[lr_ok]) == 0:
        raise ExtractionFailure("resampled spectrum is empty or constant")
    hr = fluxconserve_resample(sed_wave, sed_flux, wave_hr)
    return lr, lr_err, hr
