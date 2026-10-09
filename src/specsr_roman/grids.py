"""Fixed wavelength grids, instrument constants, and photometric band maps.

Every spectrum in the dataset lives on one of two shared grids, so a model is
fully convolutional over a fixed-length axis and any two rows are directly
comparable. The grids mirror the JWST/JADES convention of the companion
project (``specsr``) --- same npz keys, same "LR interpolated onto the HR grid
at load time" contract --- which is what made the cross-instrument
warm-start experiments possible.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Wavelength grids
# --------------------------------------------------------------------------
# LR: the Roman grism's native sampling (~10.764 A/pix) over the useful band.
# Extractions land here directly, so no information is created or destroyed by
# the gridding step.
WAVE_LR: np.ndarray = np.arange(10000.0, 19300.0, 10.764)

# HR: the target grid for the ground-truth SEDs. 2500 points is ~4.3x the LR
# sampling --- fine enough that a grism-resolution line profile is well
# sampled, coarse enough that the network stays small.
WAVE_HR: np.ndarray = np.linspace(10000.0, 19300.0, 2500)

N_LR = len(WAVE_LR)   # 864
N_HR = len(WAVE_HR)   # 2500

WAVE_HR_UM: np.ndarray = WAVE_HR * 1e-4
WAVE_LR_UM: np.ndarray = WAVE_LR * 1e-4

# --------------------------------------------------------------------------
# Roman WFI grism / imaging constants
# --------------------------------------------------------------------------
GRISM_RESOLUTION = 461.0     # R = lambda/dlambda per micron (Wang+2022 G150)
GRISM_FWHM_AA = 21.7         # point-source LSF FWHM at band centre [A]
GRISM_EXPTIME = 301.0        # one HLSS grism exposure [s]
GRISM_BKG = 0.57             # zodiacal background [e-/s/pix]
READ_NOISE = 8.5             # effective read noise [e-]

DIRECT_EXPTIME_H158 = 139.8  # OU2024 RomanWAS H158 exposure [s]

# Wang+2022 F158 photometric calibration. OU2024 images are calibrated to the
# real Roman zeropoint (AB ~ 26.4 per e-/s, verified on an AB ~ 14 index
# galaxy realising 1.26e7 e- in 139.8 s), so the same constants apply to both
# simulation families.
PHOTFLAM_F158 = 1.1e-20
PHOTPLAM_F158 = 15800.0

# grizli geometry for the Roman conf. Both are larger than grizli's adaptive
# defaults and both are required: the trace sits up to ~66 px from the source
# on det1 (~162 px on det4), which overflows the default cutout, and edge
# sources need padding or their beams fall out of the array.
GRIZLI_BEAM_SIZE = 85
GRIZLI_PAD = (120, 900)

N_PIX_SCA = 4088             # Roman SCA side, Wang2022 released frames

# --------------------------------------------------------------------------
# Photometry
# --------------------------------------------------------------------------
# OU2024 `galaxy_flux_<healpix>.parquet` column order. The `phot` array in the
# dataset follows this exactly, and every band-subset flag (`--phot-keep`)
# indexes into it.
#
# NAMING TRAP: OU2024 uses the OLD Roman band names, which collide with the
# current WFI scheme --- OU2024 `R062` is 0.62 um (current F062), and OU2024
# `W146` is the current wide filter R062. Always map by central wavelength,
# never by name.
PHOT_BANDS: tuple[str, ...] = (
    "lsst_flux_u", "lsst_flux_g", "lsst_flux_r",          # 0, 1, 2
    "lsst_flux_i", "lsst_flux_z", "lsst_flux_y",          # 3, 4, 5
    "roman_flux_R062", "roman_flux_Z087",                 # 6, 7
    "roman_flux_Y106", "roman_flux_J129",                 # 8, 9
    "roman_flux_W146", "roman_flux_H158",                 # 10, 11
    "roman_flux_F184", "roman_flux_K213",                 # 12, 13
)

# Band pivot wavelengths [A], for plotting photometry as f_lambda.
BAND_PIVOT: dict[str, float] = {
    "lsst_flux_u": 3671, "lsst_flux_g": 4827, "lsst_flux_r": 6223,
    "lsst_flux_i": 7546, "lsst_flux_z": 8691, "lsst_flux_y": 9712,
    "roman_flux_R062": 6200, "roman_flux_Z087": 8700,
    "roman_flux_Y106": 10600, "roman_flux_J129": 12900,
    "roman_flux_W146": 14600, "roman_flux_H158": 15800,
    "roman_flux_F184": 18400, "roman_flux_K213": 21300,
}

# The photometry that ships *with the grism* in the real survey: the HLWAS
# grism comes with Medium-tier Roman imaging in F106/F129/F158. This is the
# published tier and the one that needs nothing but Roman.
ROMAN_MEDIUM_BANDS: tuple[int, ...] = (8, 9, 11)          # F106 / F129 / F158

# Rubin *ugrizy*. External to Roman, but the HLWAS footprint lies inside the
# Rubin wide survey, so a coadd exists wherever the grism observes.
RUBIN_BANDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

# Roman Medium tier plus Rubin: the nine bands a grism spectrum has once the
# Rubin coadd is in hand. The five remaining Roman bands stay out --- no survey
# tier delivers them alongside the grism.
ROMAN_MEDIUM_RUBIN_BANDS: tuple[int, ...] = RUBIN_BANDS + ROMAN_MEDIUM_BANDS

#: Hard ceiling on how many bands may be fed to a model: the widest set a
#: survey really delivers with a grism spectrum. Rubin bands only count as
#: delivered when they carry their coadd depth noise (see
#: :func:`phot_flux_sigma`); a head trained on them noiseless is reading the
#: simulation's SED, not a measurement.
MAX_PHOT_BANDS: int = len(ROMAN_MEDIUM_RUBIN_BANDS)

PHOT_TIERS: dict[str, tuple[int, ...]] = {
    "medium": ROMAN_MEDIUM_BANDS,
    "medium_rubin": ROMAN_MEDIUM_RUBIN_BANDS,
}

# AB anchor for OU2024 H158: images carry counts = f_cat * 10^(0.4 * ZPTMAG),
# so AB_H158 = AB_ANCHOR - 2.5 log10(f_cat_H158).
AB_ANCHOR_H158 = 14.96
OU2024_ZPTMAG = 16.8009


# OU2024 catalogue fluxes are photon rates through each bandpass
# [ph / s / cm^2], so every band has its own AB zero point,
# AB = ZP - 2.5 log10(f). The H158 anchor above is that number for H158 and
# must not be reused for another band: doing so puts Rubin u two magnitudes
# too faint.
#
# These are measured from the simulation (scripts/derive_zeropoints.py): the
# SED of a galaxy gives its AB colours through the filter curves, H158 sets
# the scale, and the zero point is what turns the catalogue flux into that
# magnitude. The scatter over 400 galaxies is 0.005-0.04 mag. The Rubin values
# are 0.2-0.5 mag below what the lsst/throughputs v1.9 total throughputs give
# (12.655, 14.690, 14.560, 14.379, 13.994, 13.019), which were used until
# 2026-10-08: the catalogue was not computed with those curves. With the old
# values the Rubin magnitudes came out too faint and the depth noise too
# large by the same amount.
AB_ZEROPOINT: dict[str, float] = {
    "lsst_flux_u": 12.458, "lsst_flux_g": 14.348, "lsst_flux_r": 14.198,
    "lsst_flux_i": 13.871, "lsst_flux_z": 13.497, "lsst_flux_y": 12.659,
    "roman_flux_R062": 15.176, "roman_flux_Z087": 14.842,
    "roman_flux_Y106": 14.899, "roman_flux_J129": 14.933,
    "roman_flux_W146": 16.162, "roman_flux_H158": AB_ANCHOR_H158,
    "roman_flux_F184": 14.512, "roman_flux_K213": 14.468,
}

# Rubin 5-sigma point-source depth of the ten-year coadd (Ivezic et al. 2019,
# Table 2). Galaxies are extended, so this is the optimistic end.
RUBIN_M5_Y10: dict[str, float] = {
    "lsst_flux_u": 26.1, "lsst_flux_g": 27.4, "lsst_flux_r": 27.5,
    "lsst_flux_i": 26.8, "lsst_flux_z": 26.1, "lsst_flux_y": 24.9,
}

#: Depth lost going from the ten-year coadd to a single year, in magnitudes.
RUBIN_Y1_OFFSET: float = 1.25


def phot_flux_sigma(band_names, depth_offset: float = 0.0) -> np.ndarray:
    """Per-band 1-sigma flux noise in catalogue units; 0 where none is modelled.

    Rubin bands get the sky-limited noise of a coadd ``depth_offset``
    magnitudes shallower than the ten-year one. Roman Medium-tier imaging is
    several magnitudes deeper than the grism sample, so its bands carry only
    the multiplicative error applied elsewhere and return 0 here.
    """
    sig = np.zeros(len(band_names), dtype=np.float32)
    for i, name in enumerate(band_names):
        if name in RUBIN_M5_Y10:
            m5 = RUBIN_M5_Y10[name] - depth_offset
            sig[i] = 10.0 ** (-0.4 * (m5 - AB_ZEROPOINT[name])) / 5.0
    return sig


def resolve_phot_tier(spec: str | None) -> tuple[int, ...] | None:
    """``"medium"``, ``"medium_rubin"`` or an explicit ``"8,9,11"`` -> band indices.

    ``None`` means "use every band the dataset file carries" and is a *loader*
    convenience --- OU2024 stores 14 columns whatever a model consumes. It is
    not a model configuration: every training config names a tier, and
    :data:`MAX_PHOT_BANDS` caps what a model can be handed.
    """
    if spec is None or spec == "":
        return None
    key = spec.strip().lower()
    if key in PHOT_TIERS:
        return PHOT_TIERS[key]
    try:
        bands = tuple(int(b) for b in spec.split(","))
    except ValueError as exc:  # pragma: no cover - argparse-level guard
        raise ValueError(
            f"phot tier {spec!r} is neither a named tier "
            f"({'/'.join(PHOT_TIERS)}) nor a comma-separated index list"
        ) from exc
    if len(bands) > MAX_PHOT_BANDS:
        raise ValueError(
            f"phot tier {spec!r} asks for {len(bands)} bands; the ceiling is "
            f"{MAX_PHOT_BANDS} (Roman Medium tier plus Rubin ugrizy). More bands "
            "than a survey delivers with the grism turn the redshift into a "
            "photometric one measured on simulated colours -- see "
            "specsr_roman.evaluation.ablation")
    return bands
