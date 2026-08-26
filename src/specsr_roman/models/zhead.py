"""ZHead --- the redshift stage.

Three architectures live here because the first two failed in instructive
ways and the failures explain the third:

``ZHead1D``
    The direct JWST port: a Gaussian regression head over SR1's
    ``(mean, log_sigma)``. On Roman it collapsed toward the prior mean
    (corr 0.44, predicted spread 0.27 against a true 0.60). Translation-
    equivariant convolutions plus global pooling carry no absolute positional
    signal, and unlike the JWST prism --- whose varying resolution makes line
    *width* a wavelength cue --- Roman's near-constant R offers nothing to
    substitute. A normalised wavelength ramp channel (``use_position``)
    restores it.

``ZHeadAttn``
    Feature-driven attention pooling. The earlier heads pooled with weights
    derived from SR1's smooth sigma channel, which averaged a ~5-pixel
    emission line over 2500 pixels. Here saliency comes from the flux
    features themselves and each head also pools the wavelength ramp, so the
    attended ramp value *is* a line centroid: the redshift read out
    explicitly.

``ZHeadClf`` (canonical)
    P(z) classification over a fixed redshift grid. Redshift from a grism is
    a line-*identification* problem and is intrinsically multimodal --- one
    observed line is consistent with Ha, [OIII], [OII] or Lya. A single
    Gaussian must average between aliases, which is precisely what produced
    the ~40% catastrophic-outlier floor of the regression heads. A softmax
    over a grid carries the whole P(z): the point estimate is the *mode*, and
    ambiguity survives as PDF width instead of becoming a wrong answer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .blocks import conv_stack

__all__ = [
    "ZHead1D", "ZHeadAttn", "ZHeadClf",
    "make_z_grid", "soft_labels", "pz_stats", "z_metrics",
    "heteroscedastic_nll", "load_zhead",
]


class ZHead1D(nn.Module):
    """SR1 ``(mean, log_sigma)`` -> ``(mu_z, log_var_z)``. Superseded by
    :class:`ZHeadClf`; kept so v1/v2 checkpoints stay loadable."""

    def __init__(self, in_channels: int = 2, hidden_dim: int = 64,
                 num_blocks: int = 4, dropout: float = 0.1,
                 use_position: bool = True):
        super().__init__()
        self.use_uncertainty = (in_channels == 2)
        self.use_position = use_position
        flux_in = 2 if use_position else 1
        if self.use_uncertainty:
            self.flux_net = conv_stack(flux_in, hidden_dim, num_blocks, dropout)
            self.sigma_net = conv_stack(1, hidden_dim // 2, num_blocks, dropout)
            self.confidence_conv = nn.Conv1d(1, 1, kernel_size=15, padding=7)
            combined = hidden_dim + hidden_dim // 2
        else:
            self.flux_net = conv_stack(in_channels + int(use_position),
                                       hidden_dim, num_blocks, dropout)
            combined = hidden_dim
        self.mu = nn.Linear(combined, 1)
        self.log_var = nn.Linear(combined, 1)
        nn.init.constant_(self.log_var.weight, 0.0)
        nn.init.constant_(self.log_var.bias, -2.0)

    # kept as a static method: older checkpoints and external code call it.
    _blocks = staticmethod(conv_stack)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, L = x.shape
        if self.use_position:
            ramp = torch.linspace(-1.0, 1.0, L, device=x.device)
            ramp = ramp.view(1, 1, L).expand(B, 1, L)
        if self.use_uncertainty and C == 2:
            flux, log_sigma = x[:, 0:1, :], x[:, 1:2, :]
            sigma = torch.exp(log_sigma).clamp(min=1e-3)
            confidence = 1.0 / (sigma + 1e-6)
            confidence = torch.sigmoid(self.confidence_conv(confidence))
            confidence = confidence / (confidence.sum(dim=-1, keepdim=True) + 1e-6)
            flux_in = torch.cat([flux, ramp], dim=1) if self.use_position else flux
            h = torch.cat([self.flux_net(flux_in), self.sigma_net(log_sigma)], dim=1)
            h = (h * confidence).sum(dim=-1)
        else:
            x_in = torch.cat([x, ramp], dim=1) if self.use_position else x
            h = self.flux_net(x_in).mean(dim=-1)
        return self.mu(h).squeeze(-1), self.log_var(h).squeeze(-1)


class ZHeadAttn(nn.Module):
    """Feature-driven attention pooling with an explicit centroid readout."""

    def __init__(self, hidden_dim: int = 64, num_blocks: int = 4,
                 dropout: float = 0.1, n_heads: int = 4):
        super().__init__()
        self.flux_net = conv_stack(1, hidden_dim, num_blocks, dropout)
        self.sigma_net = conv_stack(1, hidden_dim // 2, num_blocks, dropout)
        feat = hidden_dim + hidden_dim // 2
        self.saliency = nn.Conv1d(feat, n_heads, kernel_size=1)
        self.n_heads = n_heads
        head_in = n_heads * (feat + 1)  # +1: attended wavelength ramp
        self.mlp = nn.Sequential(nn.Linear(head_in, 128), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU())
        self.mu = nn.Linear(64, 1)
        self.log_var = nn.Linear(64, 1)
        nn.init.constant_(self.log_var.weight, 0.0)
        nn.init.constant_(self.log_var.bias, -2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, L = x.shape
        flux, log_sigma = x[:, 0:1, :], x[:, 1:2, :]
        h = torch.cat([self.flux_net(flux), self.sigma_net(log_sigma)], dim=1)
        attn = torch.softmax(self.saliency(h), dim=-1)          # (B, K, L)
        ramp = torch.linspace(-1.0, 1.0, L, device=x.device).view(1, 1, L)
        pooled_feat = torch.einsum("bcl,bkl->bkc", h, attn)     # (B, K, C')
        pooled_pos = (attn * ramp).sum(dim=-1, keepdim=True)    # (B, K, 1)
        z_in = torch.cat([pooled_feat, pooled_pos], dim=-1).reshape(B, -1)
        out = self.mlp(z_in)
        return self.mu(out).squeeze(-1), self.log_var(out).squeeze(-1)


class ZHeadClf(nn.Module):
    """P(z) over a fixed redshift grid. The canonical redshift stage.

    Parameters
    ----------
    centers
        Bin centres of the redshift grid, from :func:`make_z_grid`. Stored as
        a buffer so a checkpoint is self-describing --- SR2 reads the grid off
        the ZHead it is handed rather than being told it.
    in_channels
        ``4`` for the canonical models: ``[LR flux, LR err, SR1 mean,
        SR1 log-sigma]``. The raw LR channels matter more than they look:
        SR1 is the conservative stage and recovers only a few percent of line
        flux, so a head reading its near-line-free output alone was starved.
        The LR spectrum still holds the line at its true observed wavelength.
    n_phot
        Number of broadband fluxes fed to the photometry branch, or 0. Colours
        over 0.35--2.1 um break the single-line alias degeneracy --- exactly
        the information the grism band lacks. Fluxes go in raw; the branch
        applies ``log10`` then standardises with train-split statistics
        carried in the ``phot_mu`` / ``phot_sig`` buffers.

        Feed it only bands that ship with the grism (see
        :data:`specsr_roman.grids.ROMAN_MEDIUM_BANDS`). A head trained on all 14
        OU2024 bands of noiseless truth photometry scores NMAD 0.003 with 0%
        catastrophic outliers, which is not skill --- it is the redshift
        leaking in through an effectively complete SED.
    """

    def __init__(self, centers: torch.Tensor, in_channels: int = 3,
                 hidden_dim: int = 64, num_blocks: int = 4, dropout: float = 0.1,
                 n_heads: int = 8, refine_window: int = 8, n_phot: int = 0):
        super().__init__()
        n_bins = centers.numel()
        self.register_buffer("z_centers", centers)
        self.refine_window = refine_window
        self.feat_net = conv_stack(in_channels + 1, hidden_dim,
                                   num_blocks, dropout)   # +1: wavelength ramp
        self.saliency = nn.Conv1d(hidden_dim, n_heads, kernel_size=1)
        self.n_heads = n_heads
        head_in = n_heads * (hidden_dim + 1)              # +1: attended ramp
        self.n_phot = n_phot
        if n_phot:
            self.register_buffer("phot_mu", torch.zeros(n_phot))
            self.register_buffer("phot_sig", torch.ones(n_phot))
            self.phot_net = nn.Sequential(nn.Linear(n_phot, 64), nn.GELU(),
                                          nn.Linear(64, 64), nn.GELU())
            head_in += 64
        self.mlp = nn.Sequential(nn.Linear(head_in, 256), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(256, 256), nn.GELU())
        self.logits = nn.Linear(256, n_bins)

    def forward(self, x: torch.Tensor, phot: torch.Tensor | None = None) -> torch.Tensor:
        B, C, L = x.shape
        ramp = torch.linspace(-1.0, 1.0, L, device=x.device).view(1, 1, L)
        h = self.feat_net(torch.cat([x, ramp.expand(B, 1, L)], dim=1))
        attn = torch.softmax(self.saliency(h), dim=-1)        # (B, K, L)
        pooled_feat = torch.einsum("bcl,bkl->bkc", h, attn)   # (B, K, hidden)
        pooled_pos = (attn * ramp).sum(dim=-1, keepdim=True)  # (B, K, 1)
        z_in = torch.cat([pooled_feat, pooled_pos], dim=-1).reshape(B, -1)
        if self.n_phot:
            if phot is None:  # e.g. deliberately zeroed inside SR2's z-loss
                p = torch.zeros(B, self.n_phot, device=x.device)
            else:
                p = (torch.log10(phot.clamp_min(1e-12)) - self.phot_mu) / self.phot_sig
            z_in = torch.cat([z_in, self.phot_net(p)], dim=-1)
        return self.logits(self.mlp(z_in))                    # (B, n_bins)

    @torch.no_grad()
    def predict_z(self, x: torch.Tensor, phot: torch.Tensor | None = None,
                  window: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """logits -> ``(zhat, sigma_z)``."""
        probs = torch.softmax(self.forward(x, phot=phot), dim=-1)
        return pz_stats(probs, self.z_centers,
                        self.refine_window if window is None else window)

    @classmethod
    def from_state_dict(cls, state: dict, **kwargs) -> ZHeadClf:
        """Rebuild the exact head a checkpoint describes.

        Grid, photometry width and input channel count are all recoverable
        from the saved tensors, so nothing about the head has to be
        remembered by the caller. ``feat_net.0`` sees one extra channel (the
        wavelength ramp) that the constructor adds itself, hence the -1.
        """
        n_phot = int(state["phot_mu"].numel()) if "phot_mu" in state else 0
        in_channels = int(state["feat_net.0.weight"].shape[1]) - 1
        head = cls(state["z_centers"], in_channels=in_channels,
                   n_phot=n_phot, **kwargs)
        head.load_state_dict(state)
        return head


def load_zhead(state: dict, **kwargs) -> nn.Module:
    """Build whichever ZHead variant a checkpoint holds.

    SR2 and the evaluation scripts accept any generation, so the dispatch
    lives here rather than being repeated at each call site.
    """
    if "logits.weight" in state:
        return ZHeadClf.from_state_dict(state, **kwargs)
    if "saliency.weight" in state:
        head = ZHeadAttn(**kwargs)
    else:
        head = ZHead1D(**kwargs)
    head.load_state_dict(state)
    return head


# ---------------------------------------------------------------------------
# P(z) helpers
# ---------------------------------------------------------------------------
def make_z_grid(z_lo: float, z_hi: float, n_bins: int,
                device: str | torch.device = "cpu") -> torch.Tensor:
    """Bin centres of a uniform redshift grid."""
    edges = torch.linspace(z_lo, z_hi, n_bins + 1, device=device)
    return 0.5 * (edges[:-1] + edges[1:])


def soft_labels(z: torch.Tensor, centers: torch.Tensor,
                label_sigma: float) -> torch.Tensor:
    """Gaussian-smoothed target over the grid (ordinal-soft cross-entropy).

    A one-hot target would make adjacent bins as wrong as a distant alias.
    Smoothing tells the head that being one bin off is nearly right, which is
    what lets the mode-local refinement below reach sub-bin accuracy.
    """
    d = (z[:, None] - centers[None, :]) / label_sigma
    t = torch.exp(-0.5 * d * d)
    return t / t.sum(dim=1, keepdim=True).clamp_min(1e-12)


def pz_stats(probs: torch.Tensor, centers: torch.Tensor,
             window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mode-based point estimate plus full-grid PDF width.

    ``zhat`` is the probability-weighted mean within +/- ``window`` bins of the
    mode --- sub-bin accuracy without ever averaging across a distant alias.
    ``sigma_z`` is the standard deviation of the *whole* P(z), so it inflates
    precisely when the head is torn between two line identifications. The
    asymmetry is deliberate: the estimate should be decisive, the error bar
    should be honest about the ambiguity.
    """
    n = probs.shape[1]
    mode = probs.argmax(dim=1)
    idx = torch.arange(n, device=probs.device)[None, :]
    m = (idx >= (mode - window).clamp(0, n - 1)[:, None]) & \
        (idx <= (mode + window).clamp(0, n - 1)[:, None])
    w = probs * m
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    zhat = (w * centers[None, :]).sum(dim=1)
    mean = (probs * centers[None, :]).sum(dim=1)
    var = (probs * (centers[None, :] - mean[:, None]) ** 2).sum(dim=1)
    return zhat, var.clamp_min(0).sqrt()


def z_metrics(z_pred, z_true) -> dict[str, float]:
    """The three numbers every redshift result is judged on.

    ``dz_nmad`` is the robust scatter of the non-outliers;
    ``catastrophic_frac`` is the fraction beyond the conventional
    ``|dz|/(1+z) > 0.15``. Report both --- a model can trade one for the other,
    and NMAD alone hides an alias problem.
    """
    z_pred, z_true = np.asarray(z_pred), np.asarray(z_true)
    dz = (z_pred - z_true) / (1 + z_true)
    return {
        "z_mae": float(np.mean(np.abs(z_pred - z_true))),
        "dz_nmad": float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
        "catastrophic_frac": float(np.mean(np.abs(dz) > 0.15)),
    }


def heteroscedastic_nll(mu: torch.Tensor, log_var: torch.Tensor,
                        y: torch.Tensor, var_floor: float = 1e-6) -> torch.Tensor:
    """Gaussian NLL for the regression heads."""
    var = torch.exp(log_var).clamp_min(var_floor)
    return 0.5 * (torch.log(var) + (y - mu) ** 2 / var).mean()
