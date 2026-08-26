"""Turning simulation products into grizli-ready frames.

OpenUniverse2024 ships direct images and truth catalogues but **no grism
images**, so this package disperses the scene itself with grizli (the same
approach as Guo et al. 2025, arXiv:2512.09993). That is not a workaround --- it
is what makes the training pairs self-consistent: the same configuration file
disperses the scene and extracts it back, so any residual is the fault of
noise, blending and the extraction, not of a mismatched instrument model.

What this module does is write the two FITS files grizli expects: a direct
image in e-/s with a photometric calibration, and an empty grism shell sharing
its WCS, tagged ``INSTRUME='WFI'`` so grizli picks up ``Roman.G150.conf``.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from ..grids import DIRECT_EXPTIME_H158, GRISM_EXPTIME, PHOTFLAM_F158, PHOTPLAM_F158

__all__ = ["prepare_frames"]


def prepare_frames(img_path: str, out_dir: str,
                   grism_exptime: float = GRISM_EXPTIME,
                   overwrite: bool = False) -> tuple[str, str]:
    """OU2024 ``simple_model`` image -> ``(direct_fits, grism_fits)``.

    The direct image is converted from counts to e-/s with the flat sky level
    removed, because grizli's source model works in rate units and the
    zodiacal background is added back explicitly at the noise stage.

    ``PHOTFLAM``/``PHOTPLAM`` are the Wang2022 F158 values, and they apply to
    OU2024 unchanged: those images are calibrated to the real Roman zeropoint
    (verified --- an AB ~ 14 index galaxy realises 1.26e7 e- in 139.8 s, exactly
    ``10^(-0.4 (14 - 26.4)) * 139.8``).

    A caveat that bites anything reading the pixels directly: setting
    ``PHOTFLAM`` makes grizli rescale the SCI array into f_lambda units
    internally. Downstream code must be unit-agnostic or read the header.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(img_path).replace(".fits.gz", "").replace(".fits", "")
    direct_out = os.path.join(out_dir, stem + "_direct.fits")
    grism_out = os.path.join(out_dir, stem + "_grism.fits")
    if not overwrite and os.path.exists(direct_out) and os.path.exists(grism_out):
        return direct_out, grism_out

    with fits.open(img_path) as h:
        hsci = h["SCI"].header
        sci = h["SCI"].data.astype(np.float32)
        err = h["ERR"].data.astype(np.float32)
        exptime = float(h[0].header.get("EXPTIME",
                                        hsci.get("EXPTIME", DIRECT_EXPTIME_H158)))
        sky = float(hsci.get("SKY_MEAN", 0.0))

    sci_rate = (sci - sky) / exptime
    err_rate = err / exptime

    def write(out, data, err_data, filt, t):
        h0 = fits.Header()
        h0["TELESCOP"] = "ROMAN"
        h0["INSTRUME"] = "WFI"      # -> $GRIZLI/CONF/Roman.G150.conf
        h0["FILTER"] = filt
        h0["EXPTIME"] = t
        hs = hsci.copy()
        hs["INSTRUME"] = "WFI"
        hs["FILTER"] = filt
        hs["EXPTIME"] = t
        hs["BUNIT"] = "ELECTRONS/S"
        if filt.startswith("F"):
            hs["PHOTFLAM"] = PHOTFLAM_F158
            hs["PHOTPLAM"] = PHOTPLAM_F158
        fits.HDUList([
            fits.PrimaryHDU(header=h0),
            fits.ImageHDU(data=data, header=hs, name="SCI"),
            fits.ImageHDU(data=err_data, header=hs, name="ERR"),
            fits.ImageHDU(data=np.zeros_like(data, dtype=np.int16), name="DQ"),
        ]).writeto(out, overwrite=True)

    write(direct_out, sci_rate, err_rate, "F158W", exptime)
    write(grism_out, np.zeros_like(sci_rate), np.zeros_like(err_rate),
          "G150", grism_exptime)
    return direct_out, grism_out
