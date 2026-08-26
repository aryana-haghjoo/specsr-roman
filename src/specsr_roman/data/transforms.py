"""Resampling and normalisation primitives.

Small functions, but each encodes a correctness lesson that cost a debugging
session, so they live in one place with the reasoning attached.
"""

from __future__ import annotations

import numpy as np

from ..grids import GRISM_RESOLUTION

__all__ = ["normalize", "fluxconserve_resample", "smooth_to_grism",
           "interp_ascending"]


def normalize(x: np.ndarray, eps: float = 1e-25):
    """Per-spectrum standardisation -> ``(normalised, mean, std)``.

    Every spectrum is normalised individually rather than by a global scale.
    That is not just conditioning: the grizli extraction is systematically
    ~1.7x brighter than the input SED (aperture losses), so an absolute flux
    scale would teach the model a calibration error. Normalising per row
    removes it and makes the task purely about *shape*.

    ``std`` is floored because a genuinely constant row would otherwise divide
    by zero --- though such rows should never reach here; see
    :func:`specsr_roman.data.datasets.RomanFixedGridDataset`.
    """
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if std < eps:
        std = eps
    return (x - mean) / std, mean, std


def fluxconserve_resample(wave: np.ndarray, flux: np.ndarray,
                          new_wave: np.ndarray) -> np.ndarray:
    """Rebin via the cumulative integral, conserving integrated flux.

    Required for the Diffsky SEDs, whose wavelength grid is *adaptive*:
    sub-Angstrom bins at the emission lines and very coarse sampling
    elsewhere. Point-interpolating that onto a uniform grid drops or
    duplicates line flux depending on where the bins land --- an emission line
    can simply vanish. Integrating and differencing preserves it exactly.
    """
    cum = np.concatenate([[0.0], np.cumsum(np.diff(wave) *
                                           0.5 * (flux[1:] + flux[:-1]))])
    edges = np.concatenate([[new_wave[0] - (new_wave[1] - new_wave[0]) / 2],
                            0.5 * (new_wave[1:] + new_wave[:-1]),
                            [new_wave[-1] + (new_wave[-1] - new_wave[-2]) / 2]])
    cum_e = np.interp(edges, wave, cum)
    return np.diff(cum_e) / np.diff(edges)


def smooth_to_grism(wave_obs: np.ndarray, flux: np.ndarray,
                    resolution: float = GRISM_RESOLUTION) -> np.ndarray:
    """Degrade a spectrum to Roman grism resolution.

    R(lambda) = 461 * lambda[um], so FWHM = lambda/R = 1/461 um = 21.7 A ---
    near-constant in wavelength, hence a constant-sigma Gaussian in lambda
    rather than a varying kernel. (This constancy is also why the ZHead needs
    an explicit wavelength ramp: unlike the JWST prism, line width here says
    nothing about where in the band you are.)
    """
    from scipy.ndimage import gaussian_filter1d
    fwhm_A = 1.0e4 / resolution
    dlam = np.median(np.diff(wave_obs))
    return gaussian_filter1d(flux, (fwhm_A / 2.355) / dlam)


def interp_ascending(new_wave: np.ndarray, wave: np.ndarray, flux: np.ndarray,
                     left=np.nan, right=np.nan) -> np.ndarray:
    """``np.interp`` that refuses to be fooled by a descending input grid.

    Roman's ``DLDP_A_1`` is negative, so grizli's ``optimal_extract`` returns
    wavelengths in *descending* order. ``np.interp`` does not check, and for
    unsorted ``xp`` it silently returns the edge value everywhere --- producing
    a perfectly flat "spectrum" that carries zero information and trains a
    model straight into the prior mean. This bug cost one full dataset build.
    """
    order = np.argsort(wave)
    return np.interp(new_wave, wave[order], flux[order], left=left, right=right)
