"""Running the three stages as one pipeline.

Two entry points:

:func:`build_sr2_input`
    Assembles SR2's input channels and redshift hypotheses from frozen SR1 and
    a ZHead. Shared by training, evaluation and inference so all three see
    byte-identical inputs --- a mismatch here is the kind of bug that shows up
    only as "the published metric does not reproduce".

:class:`RomanPipeline`
    The user-facing object: give it a low-resolution grism spectrum (and
    photometry, if you have it) and get back a super-resolved spectrum with an
    uncertainty and a redshift PDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..checkpoints import CANONICAL_CHAIN, load_sr1, load_sr2, load_zhead_ckpt
from ..data.transforms import normalize
from ..grids import WAVE_HR, WAVE_LR
from ..lines import LINE_LIST_REST_AA, angstrom_to_micron
from ..models import ZHeadClf, build_line_mask, constrain_delta, pz_stats, topk_modes

__all__ = ["build_sr2_input", "RomanPipeline", "PipelineResult"]


def build_sr2_input(x_low, sr1, zhead, wave_hi_um, line_rest_um, cfg,
                    device, phot=None, z_mean: float = 0.0, z_std: float = 1.0,
                    z_min_n: float = -3.0, z_max_n: float = 3.0,
                    sr1_out=None):
    """``(x_in, sr1_mean, z_modes, z_weights, line_mask)``.

    ``cfg`` needs ``z_topk`` and ``sigma_base_um``. Pass ``sr1_out`` as a
    precomputed ``(mean, log_var)`` to reuse a forward pass the caller has
    already made rather than running SR1 again.

    For the P(z) head, M = ``z_topk`` distinct modes are extracted and the mask
    channel is their mass-weighted union. Giving SR2 several hypotheses rather
    than one point estimate is what makes it robust to alias errors: when the
    top mode is wrong, the correct identification is usually the second or
    third, and the line branch still draws there at reduced weight. For a
    regression head, M = 1 and there is nothing to be robust with.
    """
    L = len(wave_hi_um)
    sr1_mean, sr1_logvar = sr1(x_low) if sr1_out is None else sr1_out
    sr1_log_sigma = 0.5 * sr1_logvar

    if isinstance(zhead, ZHeadClf):
        z_in = torch.cat([x_low, sr1_mean, sr1_log_sigma], dim=1)
        probs = torch.softmax(zhead(z_in, phot=phot), dim=-1)
        z_modes, z_w = topk_modes(probs, zhead.z_centers, cfg["z_topk"])
        z_modes = torch.nan_to_num(z_modes, nan=float(z_mean))
    else:
        z_in = torch.cat([sr1_mean, sr1_log_sigma], dim=1)
        mu_raw, _ = zhead(z_in)
        mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
        zhat = torch.nan_to_num((mu_n.squeeze(-1) * z_std + z_mean).reshape(-1),
                                nan=float(z_mean))
        z_modes = zhat[:, None]
        z_w = torch.ones_like(z_modes)

    wave_t = torch.as_tensor(wave_hi_um, device=device)
    w_norm = z_w / z_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    line_mask = 0.0
    for m in range(z_modes.shape[1]):
        mask_m = build_line_mask(wave_t, z_modes[:, m], line_rest_um,
                                 sigma_base_um=cfg["sigma_base_um"])
        line_mask = line_mask + w_norm[:, m, None, None] * mask_m
    line_mask = line_mask.clamp(0, 1)

    chans = [x_low, sr1_mean, torch.exp(sr1_log_sigma).clamp_min(1e-6), line_mask,
             z_modes[:, 0][:, None, None].expand(-1, 1, L)]
    return torch.cat(chans, dim=1), sr1_mean, z_modes, z_w, line_mask


@dataclass
class PipelineResult:
    """What the pipeline returns, in the caller's own flux units.

    ``flux_sr`` and ``flux_sr_err`` are de-normalised back onto the input's
    scale, so they can be plotted against the input directly. ``pz`` is the
    full redshift PDF over ``z_grid`` --- keep it: for an alias-ambiguous
    source the second mode is real information that ``z`` alone discards.
    """

    wavelength: np.ndarray        # observed wavelength [A], the HR grid
    flux_sr: np.ndarray
    flux_sr_err: np.ndarray
    flux_sr1: np.ndarray
    z: float
    z_err: float
    pz: np.ndarray | None
    z_grid: np.ndarray | None
    presence: np.ndarray | None
    line_names: list[str]


class RomanPipeline:
    """SR1 -> ZHead -> SR2, ready to run on a single spectrum or a batch.

    >>> pipe = RomanPipeline.from_pretrained()          # doctest: +SKIP
    >>> out = pipe.predict(flux_low, flux_low_err, phot=phot)   # doctest: +SKIP

    ``phot`` must be the same band set the ZHead was trained on --- three
    Roman Medium-tier fluxes (F106, F129, F158) for the published checkpoint,
    in that order, in any consistent linear flux unit.

    Passing ``None`` drops the colour prior and costs most of the redshift
    accuracy --- 26% catastrophic outliers on the held-out split against 5%
    with the imaging --- because a single in-band line is alias-degenerate.
    It is not a grism-only *model*, though: this head was trained with
    photometry, so it receives its training-mean colours rather than nothing,
    which is mean imputation on an out-of-distribution input.
    """

    def __init__(self, sr1, zhead, sr2, device="cpu",
                 wave_lr=None, wave_hr=None, z_topk: int = 3,
                 sigma_base_um: float = 0.005, delta_cap: float = 40.0):
        self.sr1, self.zhead, self.sr2 = sr1, zhead, sr2
        self.device = torch.device(device)
        self.wave_lr = WAVE_LR if wave_lr is None else np.asarray(wave_lr)
        self.wave_hr = WAVE_HR if wave_hr is None else np.asarray(wave_hr)
        self.wave_hi_um = self.wave_hr.astype(np.float32) * 1e-4
        self.line_rest_um = angstrom_to_micron([w for _, w in LINE_LIST_REST_AA])
        self.line_names = [n for n, _ in LINE_LIST_REST_AA]
        self.cfg = {"z_topk": z_topk, "sigma_base_um": sigma_base_um}
        self.delta_cap = delta_cap

    @classmethod
    def from_pretrained(cls, sr1: str | None = None, zhead: str | None = None,
                        sr2: str | None = None, device: str = "cpu",
                        repo_id: str | None = None, **kwargs) -> RomanPipeline:
        """Load the published chain (or named alternatives) from the Hub."""
        return cls(
            sr1=load_sr1(sr1 or CANONICAL_CHAIN["sr1"], device=device, repo_id=repo_id),
            zhead=load_zhead_ckpt(zhead or CANONICAL_CHAIN["zhead"], device=device,
                                  repo_id=repo_id),
            sr2=load_sr2(sr2 or CANONICAL_CHAIN["sr2"], device=device, repo_id=repo_id),
            device=device, **kwargs)

    # -- preprocessing ----------------------------------------------------
    def _prepare(self, flux_low, flux_low_err, wave_low):
        """LR spectrum -> the normalised 2-channel HR-grid tensor the model wants.

        Mirrors :class:`~specsr_roman.data.RomanFixedGridDataset` exactly, including
        the shared flux scale on the error channel. Returns the tensor plus
        the ``(mean, std)`` per row needed to undo the normalisation --- both
        of them: dropping the mean leaves the output offset from the input by
        the continuum level, which looks like a model failure and is not one.
        """
        wave_low = self.wave_lr if wave_low is None else np.asarray(wave_low)
        flux_low = np.asarray(flux_low, dtype=np.float64)
        if flux_low_err is None:
            flux_low_err = np.ones_like(flux_low)
        flux_low_err = np.asarray(flux_low_err, dtype=np.float64)
        if flux_low.ndim == 1:
            flux_low = flux_low[None, :]
            flux_low_err = flux_low_err[None, :]

        rows, means, scales = [], [], []
        for f, e in zip(flux_low, flux_low_err, strict=True):
            ok = np.isfinite(f)
            if ok.sum() < 2:
                raise ValueError("fewer than two finite pixels in an input spectrum")
            f_hr = np.interp(self.wave_hr, wave_low[ok], f[ok])
            oke = np.isfinite(e) & (e > 0)
            e_hr = (np.interp(self.wave_hr, wave_low[oke], e[oke])
                    if oke.any() else np.ones_like(f_hr))
            f_n, m, s = normalize(f_hr)
            rows.append(np.stack([f_n, e_hr / max(s, 1e-25)]))
            means.append(m)
            scales.append(s)
        x = torch.tensor(np.asarray(rows, dtype=np.float32), device=self.device)
        return x, np.asarray(means), np.asarray(scales)

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def predict(self, flux_low, flux_low_err=None, phot=None,
                wave_low=None) -> PipelineResult | list[PipelineResult]:
        """Run the chain. Accepts one spectrum or a batch of them."""
        single = np.asarray(flux_low).ndim == 1
        x_low, means, scales = self._prepare(flux_low, flux_low_err, wave_low)

        phot_t = None
        if phot is not None:
            phot_arr = np.atleast_2d(np.asarray(phot, dtype=np.float32))
            phot_t = torch.tensor(phot_arr, device=self.device)
            expected = getattr(self.zhead, "n_phot", 0)
            if expected and phot_t.shape[1] != expected:
                raise ValueError(
                    f"this ZHead was trained on {expected} photometric bands but "
                    f"{phot_t.shape[1]} were given; the published checkpoint "
                    "expects Roman F106/F129/F158 in that order")

        # One SR1 forward pass, reused by the SR2 input assembly and the
        # redshift read-out below.
        sr1_out = self.sr1(x_low)
        sr1_mean, sr1_logvar = sr1_out
        x_in, sr1_mean, z_modes, z_w, _ = build_sr2_input(
            x_low, self.sr1, self.zhead, self.wave_hi_um, self.line_rest_um,
            self.cfg, self.device, phot=phot_t, sr1_out=sr1_out)
        delta, logvar, presence = self.sr2(x_in, z_modes, z_w)
        sr2_mean = sr1_mean + constrain_delta(delta, self.delta_cap)
        sr2_sigma = torch.exp(0.5 * logvar.clamp(-8.0, 6.0))

        # The point estimate and the PDF come from the same forward pass that
        # produced the modes above; `build_sr2_input` returns the modes, but
        # the caller also wants the full P(z), so read it out here.
        pz = z_grid = None
        if isinstance(self.zhead, ZHeadClf):
            z_in = torch.cat([x_low, sr1_mean, 0.5 * sr1_logvar], dim=1)
            probs = torch.softmax(self.zhead(z_in, phot=phot_t), dim=-1)
            zhat, sig = pz_stats(probs, self.zhead.z_centers,
                                 self.zhead.refine_window)
            pz = probs.cpu().numpy()
            z_grid = self.zhead.z_centers.cpu().numpy()
        else:
            zhat = z_modes[:, 0]
            sig = torch.zeros_like(zhat)

        results = []
        for i, (mean, scale) in enumerate(zip(means, scales, strict=True)):
            # Full inverse of the per-spectrum normalisation: x = x_n * std +
            # mean. The uncertainty takes the scale only -- an offset does not
            # shift an error bar.
            results.append(PipelineResult(
                wavelength=self.wave_hr,
                flux_sr=sr2_mean[i, 0].cpu().numpy() * scale + mean,
                flux_sr_err=sr2_sigma[i, 0].cpu().numpy() * scale,
                flux_sr1=sr1_mean[i, 0].cpu().numpy() * scale + mean,
                z=float(zhat[i]),
                z_err=float(sig[i]),
                pz=None if pz is None else pz[i],
                z_grid=z_grid,
                presence=presence[i].cpu().numpy() if presence is not None else None,
                line_names=self.line_names,
            ))
        return results[0] if single else results
