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
    actually comes with), an explicit ``"8,9,11"``, or
    ``None`` to keep everything.

    Using ``"all"`` on OU2024 means feeding LSST *ugrizy* plus all eight Roman
    bands: an effectively complete SED, from which the redshift can be read
    without the spectrum contributing anything. It is a valid diagnostic and an
    invalid model.
    """
    keep = resolve_phot_tier(tier)
    if keep is None:
        return phot, None
    return phot[:, list(keep)], keep


def apply_phot_noise(phot: torch.Tensor, mag_err: float,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Multiplicative log-normal flux error of ``mag_err`` magnitudes.

    Catalogue photometry in a simulation is noiseless truth. Training on it
    teaches the head to trust colours far beyond what a real measurement
    supports, and --- worse --- *evaluating* on it reports an accuracy nobody
    will reproduce. Apply this at train and validation both; pass a seeded
    ``generator`` for validation so checkpoint selection is not comparing
    epochs across different noise draws.
    """
    if mag_err <= 0:
        return phot
    if generator is None:
        dm = mag_err * torch.randn_like(phot)
    else:
        dm = mag_err * torch.randn(phot.shape, generator=generator,
                                   device=phot.device, dtype=phot.dtype)
    return phot * torch.pow(10.0, -0.4 * dm)


def standardization_stats(phot_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``log10`` mean/std from the TRAIN split only -> ZHead buffers.

    Computing these over the full set leaks test-set information into the
    input normalisation. It is a small leak next to feeding the whole SED, but
    it is free to avoid.
    """
    pm = np.log10(np.clip(phot_train, 1e-12, None))
    return pm.mean(0), np.clip(pm.std(0), 1e-6, None)
