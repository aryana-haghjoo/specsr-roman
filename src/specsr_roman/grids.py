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

# Photometric tiers that ship *with the grism* in the real survey. These are
# the only band sets a deployable model may use: the HLWAS grism comes with
# Roman imaging in the same tier, while LSST optical coverage over the
# footprint is external, partial, and not guaranteed at first data release.
# Training on LSST bands inflates the redshift accuracy with data that will
# not be there.
ROMAN_MEDIUM_BANDS: tuple[int, ...] = (8, 9, 11)          # F106 / F129 / F158
ROMAN_DEEP_BANDS: tuple[int, ...] = (7, 8, 9, 10, 11, 12, 13)
ALL_BANDS: tuple[int, ...] = tuple(range(14))

PHOT_TIERS: dict[str, tuple[int, ...]] = {
    "medium": ROMAN_MEDIUM_BANDS,
    "deep": ROMAN_DEEP_BANDS,
    "all": ALL_BANDS,
}

# AB anchor for OU2024 H158: images carry counts = f_cat * 10^(0.4 * ZPTMAG),
# so AB_H158 = AB_ANCHOR - 2.5 log10(f_cat_H158).
AB_ANCHOR_H158 = 14.96
OU2024_ZPTMAG = 16.8009


def resolve_phot_tier(spec: str | None) -> tuple[int, ...] | None:
    """``"medium"`` / ``"deep"`` / ``"all"`` / ``"8,9,11"`` -> band indices.

    ``None`` means "use every band the dataset carries", which is only
    appropriate for ablations --- see :data:`ROMAN_MEDIUM_BANDS`.
    """
    if spec is None or spec == "":
        return None
    key = spec.strip().lower()
    if key in PHOT_TIERS:
        return PHOT_TIERS[key]
    try:
        return tuple(int(b) for b in spec.split(","))
    except ValueError as exc:  # pragma: no cover - argparse-level guard
        raise ValueError(
            f"phot tier {spec!r} is neither a named tier "
            f"({'/'.join(PHOT_TIERS)}) nor a comma-separated index list"
        ) from exc
