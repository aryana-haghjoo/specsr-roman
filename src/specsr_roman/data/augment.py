"""Anti-prior augmentation.

The training targets are model SEDs (Galacticus for Wang2022, Diffsky for
OU2024). A network can score well on them by learning the *manifold* --- fixed
line ratios, one dust law, one star-formation history family --- rather than by
reading line strengths out of the data. That is the classic inverse-crime
failure, and it is invisible to every reconstruction metric.

The audit that detects it (``specsr_roman.evaluation.prior_dominance``) injects an
off-manifold change to a line, forward-models it into the LR input, and
measures a *response exponent* r: 1 means the model read the change from the
data, 0 means it recited the prior regardless. The unaugmented Roman SR1
scored r = 0.14.

These augmentations attack the regularities directly:

* per-line strength jitter, independent per line, so line *ratios* vary;
* Calzetti dust jitter, because the simulations use one fixed Av;
* an LR-only smooth calibration tilt, teaching invariance to flux-calibration
  error rather than to physics.

Every HR perturbation is forward-modelled onto the LR input through a Gaussian
LSF at grism resolution (width jittered to stand in for morphological
broadening), so the pair stays physically consistent --- the augmented input is
what the augmented truth would actually have produced.

With augmentation, r rose to 0.51 at fixed detectability. It also suppressed
absolute line recovery, which is why the canonical SR1 does **not** use it:
the honest fix is a redesign that jitters only recoverable information, or a
broader simulation. See the limitations discussion in the README.
"""

from __future__ import annotations

import numpy as np

from ..grids import GRISM_FWHM_AA

__all__ = ["calzetti_k", "find_line_segments", "SpectrumAugmentor"]


def calzetti_k(lam_rest_um: np.ndarray) -> np.ndarray:
    """Calzetti (2000) attenuation curve k(lambda), Rv = 4.05."""
    lam = np.clip(lam_rest_um, 0.12, 2.2)
    k_red = 2.659 * (-1.857 + 1.040 / lam) + 4.05
    k_blue = (2.659 * (-2.156 + 1.509 / lam - 0.198 / lam ** 2
                       + 0.011 / lam ** 3) + 4.05)
    return np.where(lam >= 0.63, k_red, k_blue)


def find_line_segments(flux_hi: np.ndarray, smooth_px: int = 101,
                       thresh_sig: float = 4.0, grow: int = 2):
    """Label contiguous emission-line segments in a noiseless HR spectrum.

    Returns ``(labels, n_segments, continuum)``. Segment labels let the
    augmentor rescale one line without touching its neighbours, which is what
    makes the line *ratios* vary rather than the overall line strength.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter1d, label
    cont = gaussian_filter1d(flux_hi, smooth_px)
    resid = flux_hi - cont
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-30
    mask = resid > thresh_sig * sigma
    if grow:
        mask = binary_dilation(mask, iterations=grow)
    labels, n = label(mask)
    return labels, n, cont


class SpectrumAugmentor:
    """On-the-fly ``(lr, hr) -> (lr', hr')`` augmentation in raw flux space.

    Applied before normalisation, and only to the train split --- validation
    must always see the unaugmented distribution or the metric drifts with the
    augmentation settings.
    """

    def __init__(self, wave_hi: np.ndarray, fwhm_A: float = GRISM_FWHM_AA,
                 line_jitter: tuple[float, float] = (0.4, 2.5),
                 dav_range: float = 0.4, tilt_amp: float = 0.10,
                 lsf_width_jitter: tuple[float, float] = (1.0, 2.0)):
        from scipy.ndimage import gaussian_filter1d
        self._smooth = gaussian_filter1d
        self.wave = wave_hi
        self.sig_px = (fwhm_A / 2.355) / np.median(np.diff(wave_hi))
        self.line_jitter = line_jitter
        self.dav_range = dav_range
        self.tilt_amp = tilt_amp
        self.lsf_width_jitter = lsf_width_jitter

    def __call__(self, lr, hr, z, seg_labels, n_seg, rng):
        hr_a = hr.copy()
        # 1) independent per-line strength jitter (log-uniform)
        if n_seg > 0:
            cont = self._smooth(hr, 101)
            for s in range(1, n_seg + 1):
                m = seg_labels == s
                f = np.exp(rng.uniform(np.log(self.line_jitter[0]),
                                       np.log(self.line_jitter[1])))
                hr_a[m] = cont[m] + f * (hr[m] - cont[m])
        # 2) dust jitter, applied to both sides (smooth in wavelength)
        dav = rng.uniform(-self.dav_range, self.dav_range)
        dust = 10.0 ** (-0.4 * dav * calzetti_k(self.wave / (1 + z) / 1e4) / 4.05)
        hr_a *= dust
        # forward-model the HR change onto the LR input
        width = rng.uniform(*self.lsf_width_jitter)
        delta = self._smooth(hr_a - hr, self.sig_px * width)
        lr_a = lr + delta
        # 3) LR-only calibration tilt: the target is deliberately unchanged,
        #    so the model learns that a smooth multiplicative error in the
        #    input is not a feature of the source.
        x = np.linspace(-1, 1, len(lr))
        tilt = 1.0 + self.tilt_amp * (rng.uniform(-1, 1) * 0.4 +
                                      rng.uniform(-1, 1) * 0.4 * x +
                                      rng.uniform(-1, 1) * 0.2 * x ** 2)
        lr_a *= tilt
        return lr_a, hr_a
