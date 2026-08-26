"""specsr_roman --- physics-informed super-resolution of Roman grism spectra.

Roman's High Latitude Spectroscopic Survey will deliver slitless grism spectra
(R ~ 461 lambda/um over 1--1.93 um) for millions of emission-line galaxies. At
that resolving power the diagnostic complexes stay blended --- Ha+[NII],
[OIII]+Hbeta --- across wide redshift ranges, biasing redshifts and everything
derived from them.

This package super-resolves those spectra in three stages, and is built around
one constraint that shapes every design decision in it: **a model that invents
plausible emission lines is worse than useless in a survey.** So the pipeline
reports what the data can support and declines to draw what it cannot.

    from specsr_roman import RomanPipeline

    pipe = RomanPipeline.from_pretrained()
    out = pipe.predict(flux_low, flux_low_err, phot=roman_medium_fluxes)
    out.flux_sr, out.z, out.pz

Stages
------
``SR1``    coarse super-resolution with a heteroscedastic uncertainty.
``ZHead``  P(z) over a redshift grid, conditioned on the grism spectrum and
           the Roman imaging that ships with it.
``SR2``    line-token attention refinement, gated on line presence.

See ``specsr_roman.evaluation`` for the metrics and the two audits --- the
photometry ablation and the inverse-crime test --- that keep the reported
numbers honest.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import SR1Config, SR2Config, ZHeadConfig
from .grids import PHOT_BANDS, ROMAN_DEEP_BANDS, ROMAN_MEDIUM_BANDS, WAVE_HR, WAVE_LR
from .lines import LINE_LIST_REST_AA, SR1_LINES_AA, STRONG_LINES_AA

__all__ = [
    "__version__",
    "RomanPipeline", "PipelineResult",
    "SuperRes1D", "ZHeadClf", "SR2Attention",
    "RomanFixedGridDataset",
    "SR1Config", "ZHeadConfig", "SR2Config",
    "WAVE_LR", "WAVE_HR", "PHOT_BANDS",
    "ROMAN_MEDIUM_BANDS", "ROMAN_DEEP_BANDS",
    "LINE_LIST_REST_AA", "SR1_LINES_AA", "STRONG_LINES_AA",
]


def __getattr__(name: str):
    """Import torch-dependent objects lazily.

    Keeps ``import specsr_roman`` cheap --- reading ``specsr_roman.WAVE_HR`` or a config
    should not pay for importing torch.
    """
    if name in ("RomanPipeline", "PipelineResult"):
        from . import inference
        return getattr(inference, name)
    if name in ("SuperRes1D", "ZHeadClf", "SR2Attention"):
        from . import models
        return getattr(models, name)
    if name == "RomanFixedGridDataset":
        from .data import RomanFixedGridDataset
        return RomanFixedGridDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
