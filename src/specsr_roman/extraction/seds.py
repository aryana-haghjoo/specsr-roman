"""Ground-truth SEDs from the OU2024 (Diffsky) catalogue.

These are the super-resolution *targets*, so how they are read matters as much
as how the spectra are extracted. Two properties of the format decide
everything downstream:

* the wavelength grid is **adaptive** --- sub-Angstrom bins at emission lines,
  very coarse elsewhere. Point interpolation onto a uniform grid loses line
  flux, so every rebin here is flux-conserving;
* the absolute flux scale is internal to the simulation and meaningless.
  Only the *shape* is used: grizli renormalises to the direct-image counts,
  and the training dataset normalises per spectrum.
"""

from __future__ import annotations

import numpy as np

from ..data.transforms import fluxconserve_resample

__all__ = ["SEDLibrary", "H158_NORM_LO", "H158_NORM_HI"]

# Band used to normalise an SED before handing it to grizli. `is_cgs=False`
# treats spectrum_1d as a multiplier against a flat f_lambda == 1 spectrum, so
# the SED must have unity mean over the direct-image band for the image counts
# to set the amplitude. Skip this and raw Diffsky values (~1e-24) disperse an
# effectively empty scene -- a silent, total failure.
H158_NORM_LO, H158_NORM_HI = 13800.0, 17700.0


class SEDLibrary:
    """Lazily opened handle on a ``galaxy_sed_<healpix>.hdf5`` file.

    The file is ~14 GB per healpix and is read thousands of times per SCA, so
    it is opened once and the rest-frame wavelength grid cached. Layout is
    skyCatalogs': ``galaxy/<gid // 100000>/<gid>`` holding ``(3 components,
    n_wave)``, with ``meta/wave_list`` the rest-frame wavelengths in Angstrom.
    """

    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._wave = None

    def _ensure_open(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.path, "r")
            self._wave = np.asarray(self._file["meta/wave_list"], dtype=np.float64)
        return self._file, self._wave

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def observed(self, gid: int, z: float, dlam: float = 5.0):
        """``(observed wavelength [A], f_lambda)`` on a uniform ``dlam`` grid.

        The three Diffsky components are summed, redshifted, and rebinned
        flux-conservingly so the line spikes survive every later
        interpolation. Raises ``KeyError`` for a galaxy with no SED entry.
        """
        f, wave = self._ensure_open()
        sed = np.asarray(f[f"galaxy/{gid // 100000}/{gid}"],
                         dtype=np.float64).sum(axis=0)
        wave_obs = wave * (1.0 + z)
        uniform = np.arange(2500.0, min(wave_obs[-1], 25000.0), dlam)
        return uniform, fluxconserve_resample(wave_obs, sed, uniform)

    def grizli_spectrum(self, gid: int, z: float):
        """``([wave, flux_normalised], (wave, flux_raw))`` for a grizli dispersal.

        Returns ``None`` if the galaxy has no usable flux in the normalising
        band. The second element is the un-normalised SED, which is what
        becomes the training target.
        """
        wave_obs, flam = self.observed(gid, z)
        m = (wave_obs > H158_NORM_LO) & (wave_obs < H158_NORM_HI)
        norm = float(np.mean(flam[m])) if m.any() else 0.0
        if norm <= 0:
            return None
        spec = [wave_obs.astype(np.float64), (flam / norm).astype(np.float64)]
        return spec, (wave_obs, flam)
