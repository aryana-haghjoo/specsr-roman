"""The training dataset.

One class, because all three stages consume the same rows: SR1 needs the
spectra, the ZHead needs the spectra plus photometry, SR2 needs both plus the
per-line recoverability labels. Building them once and returning a wide tuple
keeps the three stages exactly aligned on which row is which.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..grids import PHOT_BANDS, resolve_phot_tier
from ..lines import SR1_LINES_AA
from .augment import SpectrumAugmentor, find_line_segments
from .transforms import normalize

__all__ = ["RomanFixedGridDataset"]

# Window used for the per-line integrated S/N labels: +/-4 native LR pixels,
# about +/-43 A, which is ~2 resolution elements -- wide enough to capture the
# line, narrow enough that the noise term is not dominated by continuum.
_LINE_SNR_HALF_PX = 4

# H158 band limits [A], used to put the noiseless target into the same units
# as the extraction (which normalises every SED to unit mean over this band).
_H158_LO, _H158_HI = 13900.0, 17700.0


class RomanFixedGridDataset(Dataset):
    """Roman grism LR/HR pairs on the shared fixed grids.

    Loads a dataset npz, applies quality cuts, interpolates ``flux_low`` onto
    the HR grid, normalises per spectrum, and precomputes the per-line
    recoverability labels the losses need.

    Returns per item::

        (low_2ch, high, high_err, z, high_mean, high_std, line_snr[, phot])

    ``low_2ch`` is ``[flux, err]``, both divided by the *same* per-row flux
    scale so that the channel ratio is the per-pixel S/N. That shared scaling
    is the point --- normalising the error channel separately would destroy the
    very quantity the network reads.

    ``high_err`` is zeros: the targets are noiseless simulated SEDs. It is
    still carried through so the loss signature matches the JWST version,
    where the targets are real grating spectra with real errors.

    Parameters
    ----------
    npz_path
        Dataset built by :mod:`specsr_roman.extraction`.
    min_finite_low, min_finite_high
        Minimum finite fraction for a row to survive.
    augment
        Apply :class:`~specsr_roman.data.augment.SpectrumAugmentor` on the fly.
        Train split only --- build a second instance for validation.
    with_phot
        Return the photometry column as an eighth tuple element.
    phot_tier
        Band subset (see :func:`specsr_roman.grids.resolve_phot_tier`). Applied at
        load time so the dataset and the model it feeds cannot disagree about
        the band count.
    """

    def __init__(self, npz_path: str, min_finite_low: float = 0.7,
                 min_finite_high: float = 0.9, augment: bool = False,
                 with_phot: bool = False, phot_tier: str | None = None,
                 verbose: bool = True):
        from scipy.ndimage import gaussian_filter1d

        self.with_phot = with_phot
        self.path = npz_path
        data = np.load(npz_path, allow_pickle=True)
        flux_lo = data["flux_low"]
        flux_hi = data["flux_high"]
        wave_lo = data["wavelength_low"]
        wave_hi = data["wavelength_high"]
        z = data["redshift"]
        snr = data["snr"]
        err_lo = data["flux_low_err"] if "flux_low_err" in data else np.ones_like(flux_lo)

        keep = (np.isfinite(snr)
                & (np.isfinite(flux_lo).mean(axis=1) > min_finite_low)
                & (np.isfinite(flux_hi).mean(axis=1) > min_finite_high))
        self.keep_mask = keep
        flux_lo, flux_hi, z = flux_lo[keep], flux_hi[keep], z[keep]
        err_lo = err_lo[keep]
        if verbose:
            print(f"quality cuts: kept {keep.sum()} / {len(keep)} rows")

        # A constant LR spectrum carries zero information. More than a trickle
        # of them means the extraction is broken -- the classic cause is
        # np.interp fed grizli's descending Roman wavelength grid, which
        # returns the edge value everywhere. Left alone, super-resolution
        # would silently learn to emit the prior mean, and every metric would
        # look plausible. Refuse to train instead.
        lr_std = np.nanstd(flux_lo, axis=1)
        const_frac = float((lr_std < 1e-8).mean())
        if const_frac > 0.01:
            raise RuntimeError(
                f"{const_frac:.1%} of flux_low rows are constant -- refusing to "
                "train on an information-free input; re-extract the dataset "
                "(see specsr_roman.data.transforms.interp_ascending)")

        # object ids for the group split, and catalogue photometry for the
        # redshift head.
        self.ids = np.asarray(data["ids"])[keep] if "ids" in data else None
        self.phot = (np.asarray(data["phot"])[keep].astype(np.float32)
                     if "phot" in data else None)
        self.phot_bands: tuple[str, ...] | None = None
        if self.phot is not None:
            bands = (tuple(str(b) for b in data["phot_bands"])
                     if "phot_bands" in data else PHOT_BANDS)
            keep_idx = resolve_phot_tier(phot_tier)
            if keep_idx is not None:
                self.phot = self.phot[:, list(keep_idx)]
                bands = tuple(bands[i] for i in keep_idx)
                if verbose:
                    print(f"phot tier {phot_tier!r}: {len(bands)} bands "
                          f"{list(bands)}")
            self.phot_bands = bands

        h158 = (wave_hi > _H158_LO) & (wave_hi < _H158_HI)
        lines = np.array(SR1_LINES_AA)

        lo_raw, hi_raw, err_raw = [], [], []
        mean_list, std_list, snr_rows = [], [], []
        # strict=True: all four are the same length by construction (the
        # keep mask above was applied to each), so a mismatch is a bug.
        for f_lo, f_hi, e_lo, z_i in zip(flux_lo, flux_hi, err_lo, z, strict=True):
            ok = np.isfinite(f_lo)
            # LR -> HR grid. Same-length input and output is what lets a fully
            # convolutional model map one to the other; np.interp extends the
            # edge values across the NaN margins.
            f_lo_hr = np.interp(wave_hi, wave_lo[ok], f_lo[ok])
            oke = np.isfinite(e_lo) & (e_lo > 0)
            e_lo_hr = (np.interp(wave_hi, wave_lo[oke], e_lo[oke])
                       if oke.any() else np.ones_like(f_lo_hr))

            f_hi = f_hi.copy()
            bad = ~np.isfinite(f_hi)
            if bad.any():
                okh = ~bad
                f_hi[bad] = np.interp(wave_hi[bad], wave_hi[okh], f_hi[okh])
            _, m_hi, s_hi = normalize(f_hi)

            snr_rows.append(self._line_snr_row(
                f_hi, e_lo, wave_lo, wave_hi, z_i, lines, h158, gaussian_filter1d))
            lo_raw.append(f_lo_hr.astype(np.float32))
            err_raw.append(e_lo_hr.astype(np.float32))
            hi_raw.append(f_hi.astype(np.float32))
            mean_list.append(m_hi)
            std_list.append(s_hi)

        self.wave_hi = wave_hi
        self.wave_lo = wave_lo
        self.lo_raw = np.array(lo_raw)
        self.err_raw = np.array(err_raw)
        self.hi_raw = np.array(hi_raw)
        self.line_snr = torch.tensor(np.array(snr_rows), dtype=torch.float32)
        self.z = torch.tensor(np.asarray(z, dtype=np.float32))
        self.high_mean = torch.tensor(np.array(mean_list), dtype=torch.float32)
        self.high_std = torch.tensor(np.array(std_list), dtype=torch.float32)

        if verbose:
            best = self.line_snr.max(dim=1).values.numpy()
            print(f"line-S/N labels: {np.mean(best > 3):.1%} of rows have a "
                  f">3sigma recoverable line (median best-line S/N "
                  f"{np.median(best):.2f})")

        self.augment = augment
        self.augmentor = None
        if augment:
            self.augmentor = SpectrumAugmentor(wave_hi)
            if verbose:
                print("precomputing line segments for augmentation...")
            self.seg_labels = np.zeros_like(self.hi_raw, dtype=np.int16)
            self.n_seg = np.zeros(len(self.hi_raw), dtype=np.int16)
            for i in range(len(self.hi_raw)):
                labels, n, _ = find_line_segments(self.hi_raw[i])
                self.seg_labels[i] = labels
                self.n_seg[i] = n

    @staticmethod
    def _line_snr_row(f_hi, e_lo, wave_lo, wave_hi, z_i, lines, h158, smooth):
        """Integrated S/N of each :data:`SR1_LINES_AA` line, on the native LR grid.

        This label is what makes the whole "recoverability" machinery work.
        Signal is the true line flux --- the continuum-subtracted *noiseless*
        target, converted into LR units via the unit-H158-mean convention the
        extraction uses --- and noise is the quadrature sum of the extraction's
        own error array over the same window. So it answers, per line and per
        row: could this line have been seen in this spectrum at all?

        Lines that could not be seen are exactly the ones a model must not
        draw, and the ones whose absence it must not be punished for.
        """
        row_snr = np.zeros(len(lines), dtype=np.float32)
        s_h158 = float(np.mean(f_hi[h158]))
        if s_h158 <= 0:
            return row_snr
        resid = f_hi - smooth(f_hi, 101)
        half = _LINE_SNR_HALF_PX
        for k, lc in enumerate((1.0 + z_i) * lines):
            j = int(np.searchsorted(wave_lo, lc))
            if j < half + 1 or j > len(wave_lo) - half - 2:
                continue
            win = slice(j - half, j + half + 1)
            if not (np.isfinite(e_lo[win]) & (e_lo[win] > 0)).all():
                continue
            signal = np.interp(wave_lo[win], wave_hi, resid).sum() / s_h158
            noise = np.sqrt(np.sum(e_lo[win] ** 2))
            row_snr[k] = max(signal, 0.0) / max(noise, 1e-30)
        return row_snr

    @property
    def n_phot(self) -> int:
        return 0 if self.phot is None else int(self.phot.shape[1])

    def __len__(self) -> int:
        return len(self.lo_raw)

    def __getitem__(self, idx):
        lr, hr = self.lo_raw[idx], self.hi_raw[idx]
        if self.augment:
            rng = np.random.default_rng()
            lr, hr = self.augmentor(lr, hr, float(self.z[idx]),
                                    self.seg_labels[idx], int(self.n_seg[idx]), rng)
        lr_n, _, s_lo = normalize(lr)
        # the error channel shares the flux scale, so channel ratio = S/N
        err_n = self.err_raw[idx] / max(s_lo, 1e-25)
        hr_n, m_hi, s_hi = normalize(hr)
        low_2ch = torch.from_numpy(np.stack([lr_n, err_n]).astype(np.float32))
        out = (low_2ch,
               torch.tensor(hr_n, dtype=torch.float32),
               torch.zeros(len(hr_n), dtype=torch.float32),
               self.z[idx],
               torch.tensor(m_hi, dtype=torch.float32),
               torch.tensor(s_hi, dtype=torch.float32),
               self.line_snr[idx])
        if self.with_phot:
            return out + (torch.tensor(self.phot[idx]),)
        return out
