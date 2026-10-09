"""Broadband photometry handling for the redshift stage.

Two operations, both of which have been got wrong at least once in this
project's history and both of which change the headline number by an order of
magnitude.
"""

from __future__ import annotations

import numpy as np
import torch

from ..grids import PHOT_BANDS, resolve_phot_tier

__all__ = ["select_bands", "apply_phot_noise", "band_names",
           "standardization_stats"]


def band_names(indices) -> list[str]:
    """Index list -> OU2024 column names, for logging and model cards."""
    return [PHOT_BANDS[i] for i in indices]


def select_bands(phot: np.ndarray, tier: str | None) -> tuple[np.ndarray, tuple[int, ...] | None]:
    """Keep only the bands that ship with the grism.

    ``tier`` is ``"medium"`` (Roman F106/F129/F158 --- what the HLWAS grism
    actually comes with), ``"medium_rubin"`` (those three plus Rubin
    *ugrizy*), an explicit ``"8,9,11"``, or ``None`` to keep everything.

    Keeping everything on OU2024 means all fourteen catalogue bands: an
    effectively complete SED, from which the redshift can be read without the
    spectrum contributing anything. It is a valid diagnostic and an invalid
    model.
    """
    keep = resolve_phot_tier(tier)
    if keep is None:
        return phot, None
    return phot[:, list(keep)], keep


def apply_phot_noise(phot: torch.Tensor, mag_err: float,
                     generator: torch.Generator | None = None,
                     sigma_flux: torch.Tensor | None = None) -> torch.Tensor:
    """Multiplicative log-normal flux error of ``mag_err`` magnitudes, plus
    additive per-band depth noise where ``sigma_flux`` is non-zero.

    The multiplicative term is a calibration-like error that is the same
    fraction of every source. ``sigma_flux`` (one value per band, catalogue
    flux units, from :func:`specsr_roman.grids.phot_flux_sigma`) is the
    sky-limited term of a coadd of finite depth: it is what makes a faint band
    a non-detection. The result may be negative; the head floors it at one
    sigma, so "consistent with zero at this depth" reaches the network as a
    single well-defined value.

    Catalogue photometry in a simulation is noiseless truth. Training on it
    teaches the head to trust colours far beyond what a real measurement
    supports, and --- worse --- *evaluating* on it reports an accuracy nobody
    will reproduce. Apply this at train and validation both; pass a seeded
    ``generator`` for validation so checkpoint selection is not comparing
    epochs across different noise draws.
    """
    def draw():
        if generator is None:
            return torch.randn_like(phot)
        return torch.randn(phot.shape, generator=generator,
                           device=phot.device, dtype=phot.dtype)

    # The multiplicative draw comes first and is the only one when no band has
    # depth noise, so a Roman-only head sees the same numbers it always has.
    if mag_err > 0:
        phot = phot * torch.pow(10.0, -0.4 * mag_err * draw())
    if sigma_flux is not None and bool((sigma_flux > 0).any()):
        phot = phot + sigma_flux.to(phot) * draw()
    return phot


def standardization_stats(phot_train: np.ndarray,
                          sigma_flux: np.ndarray | None = None,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """``log10`` mean/std from the TRAIN split only -> ZHead buffers.

    ``sigma_flux`` floors each band at its one-sigma depth first, matching
    what the head does to its input.

    Computing these over the full set leaks test-set information into the
    input normalisation. It is a small leak next to feeding the whole SED, but
    it is free to avoid.
    """
    floor = 1e-12 if sigma_flux is None else np.clip(sigma_flux, 1e-12, None)
    pm = np.log10(np.maximum(phot_train, floor))
    return pm.mean(0), np.clip(pm.std(0), 1e-6, None)
