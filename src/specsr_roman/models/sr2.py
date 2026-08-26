"""SR2 --- the line-token attention refiner.

SR1 produces a smooth, conservative reconstruction; the ZHead produces a
redshift PDF. SR2 spends its whole capacity on the delta between the two and
the truth, through two branches:

*Line branch.* One token per rest-frame feature in
:data:`specsr_roman.lines.LINE_LIST_REST_AA`. Each token reads a local window of
the input at that line's predicted observed position, the tokens cross-attend
(so [OIII]4959/5007 and Ha/[NII] can agree on a consistent picture), and each
decodes to a gated Gaussian --- amplitude x presence --- scattered back onto
the wavelength axis as a sparse delta.

*CNN branch.* A residual stack for the continuum and everything between lines.

The line branch runs once per redshift hypothesis and the deltas are combined
weighted by P(z) mode mass. That is what makes the stage alias-robust: the
correct line identification is almost always among the top few modes even when
the point estimate is a catastrophic outlier, so the right placement still
gets drawn, just at reduced weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SR2Attention", "topk_modes", "constrain_delta", "build_line_mask",
           "line_profiles"]


class SR2Attention(nn.Module):
    """``(input channels, z hypotheses) -> (delta, log_var, presence)``.

    Parameters
    ----------
    in_channels
        6 for the canonical models: LR flux, LR err, SR1 mean, SR1 sigma,
        line-position mask, broadcast zhat.
    line_rest_um, wave_hi_um
        Rest-frame line list and the observed grid, both in microns. Held as
        buffers so a checkpoint carries the line list it was trained with.
    window_half
        Half-width in pixels of the window each line token reads. 25 px on the
        HR grid is ~93 A --- several resolution elements, enough to see the
        local continuum a line sits on.
    """

    def __init__(self, in_channels: int, line_rest_um, wave_hi_um,
                 line_dim: int = 128, num_attn_heads: int = 4,
                 num_attn_layers: int = 4, window_half: int = 25,
                 cnn_dim: int = 96, num_cnn_blocks: int = 6, dropout: float = 0.02):
        super().__init__()
        self.K = len(line_rest_um)
        self.L = len(wave_hi_um)
        self.window_half = window_half
        self.W = 2 * window_half + 1
        self.register_buffer("line_rest_um",
                             torch.as_tensor(line_rest_um, dtype=torch.float32))
        self.register_buffer("wave_hi_um",
                             torch.as_tensor(wave_hi_um, dtype=torch.float32))

        self.line_embed = nn.Embedding(self.K, line_dim)
        self.line_encoder = nn.Sequential(
            nn.Conv1d(in_channels, line_dim, 5, padding=2), nn.GELU(),
            nn.Conv1d(line_dim, line_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(4))
        self.line_proj = nn.Linear(line_dim * 4, line_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=line_dim, nhead=num_attn_heads, dim_feedforward=line_dim * 2,
            dropout=dropout, batch_first=True)
        self.line_attn = nn.TransformerEncoder(enc_layer, num_layers=num_attn_layers)
        self.amp_head = nn.Linear(line_dim, 1)
        self.logw_head = nn.Linear(line_dim, 1)
        self.offset_head = nn.Linear(line_dim, 1)
        self.presence_head = nn.Linear(line_dim, 1)
        # Start quiet and sceptical: near-zero amplitudes, presence prior
        # sigmoid(-2) ~ 0.12, and a line width of e^1 ~ 2.7 px so the first
        # gradients see a resolved profile rather than a delta function.
        nn.init.normal_(self.amp_head.weight, std=0.01)
        nn.init.constant_(self.amp_head.bias, 0.0)
        nn.init.constant_(self.presence_head.bias, -2.0)
        nn.init.constant_(self.logw_head.bias, 1.0)

        g = min(8, cnn_dim)
        self.cnn_in = nn.Sequential(
            nn.Conv1d(in_channels, cnn_dim, 5, padding=2), nn.GELU())
        self.cnn_blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(g, cnn_dim), nn.GELU(),
                nn.Conv1d(cnn_dim, cnn_dim, 3, padding=1, bias=False),
                nn.GroupNorm(g, cnn_dim), nn.GELU(),
                nn.Conv1d(cnn_dim, cnn_dim, 3, padding=1, bias=False),
                nn.Dropout(dropout))
            for _ in range(num_cnn_blocks)])
        # zero-initialised output: SR2 starts as the identity on SR1.
        self.cnn_out = nn.Conv1d(cnn_dim, 1, 1)
        nn.init.constant_(self.cnn_out.weight, 0.0)
        nn.init.constant_(self.cnn_out.bias, 0.0)

        self.logvar_head = nn.Sequential(
            nn.Conv1d(in_channels, cnn_dim, 5, padding=2), nn.GELU(),
            nn.Conv1d(cnn_dim, 1, 1))
        nn.init.constant_(self.logvar_head[-1].weight, 0.0)
        nn.init.constant_(self.logvar_head[-1].bias, -2.0)

    # -- geometry ---------------------------------------------------------
    def _line_positions(self, zhat: torch.Tensor):
        """Fractional pixel position of every line at ``zhat``, plus in-band flag."""
        lam_obs = self.line_rest_um[None, :] * (1.0 + zhat[:, None])
        wave = self.wave_hi_um
        lam_c = lam_obs.clamp(wave[0], wave[-1])
        idx = torch.searchsorted(wave, lam_c).clamp(1, self.L - 1)
        frac = (lam_c - wave[idx - 1]) / (wave[idx] - wave[idx - 1] + 1e-12)
        pos = (idx - 1).float() + frac
        in_range = (lam_obs >= wave[0]) & (lam_obs <= wave[-1])
        return pos, in_range

    def _extract_windows(self, x: torch.Tensor, positions: torch.Tensor):
        B, C, L = x.shape
        h = self.window_half
        pos_int = positions.round().long().clamp(h, L - h - 1)
        offsets = torch.arange(-h, h + 1, device=x.device)
        idx = (pos_int.unsqueeze(-1) + offsets[None, None, :]).clamp(0, L - 1)
        idx_flat = idx.reshape(B, -1).unsqueeze(1).expand(-1, C, -1)
        return torch.gather(x, 2, idx_flat).reshape(
            B, C, self.K, self.W).permute(0, 2, 1, 3)

    def _reconstruct_gaussians(self, amp, width, offset, positions):
        """Scatter per-line Gaussians back onto the full wavelength axis."""
        B, device, h = amp.shape[0], amp.device, self.window_half
        centers = positions + offset
        center_int = centers.round().long().clamp(h, self.L - h - 1)
        off_grid = torch.arange(-h, h + 1, device=device, dtype=torch.float32)
        pix = center_int.unsqueeze(-1).float() + off_grid[None, None, :]
        diff = pix - centers.unsqueeze(-1)
        sigma = width.unsqueeze(-1).clamp_min(0.5)
        profiles = torch.exp(-0.5 * (diff / sigma) ** 2)
        weighted = (amp.unsqueeze(-1) * profiles).reshape(B, -1)
        idx = (center_int.unsqueeze(-1)
               + off_grid[None, None, :].long()).clamp(0, self.L - 1).reshape(B, -1)
        delta = torch.zeros(B, self.L, device=device)
        delta.scatter_add_(1, idx, weighted)
        return delta.unsqueeze(1)

    # -- branches ---------------------------------------------------------
    def _line_branch(self, x: torch.Tensor, zhat: torch.Tensor):
        B = x.shape[0]
        positions, in_range = self._line_positions(zhat)
        windows = self._extract_windows(x, positions)
        enc = self.line_encoder(windows.reshape(B * self.K, -1, self.W))
        feat = self.line_proj(enc.reshape(B * self.K, -1)).reshape(B, self.K, -1)
        feat = feat + self.line_embed.weight[None, :, :]
        feat = self.line_attn(feat)
        amp = self.amp_head(feat).squeeze(-1)
        width = torch.exp(self.logw_head(feat).squeeze(-1).clamp(-2, 3))
        offset = self.offset_head(feat).squeeze(-1).clamp(-10, 10)
        presence = torch.sigmoid(self.presence_head(feat).squeeze(-1))
        amp = amp * presence * in_range.float()
        return self._reconstruct_gaussians(amp, width, offset, positions), presence

    def forward(self, x: torch.Tensor, zhat: torch.Tensor,
                z_weight: torch.Tensor | None = None):
        """``zhat`` is ``(B,)`` for a single hypothesis or ``(B, M)`` for the
        top-M P(z) modes with matching ``z_weight``.

        Always returns ``(delta, log_var, presence)`` --- presence in eval too,
        because every evaluation of this stage asks what it thought was there.
        """
        if zhat.dim() == 1:
            line_delta, presence = self._line_branch(x, zhat)
        else:
            w = z_weight / z_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
            line_delta, presence = 0.0, 0.0
            for m in range(zhat.shape[1]):
                d_m, p_m = self._line_branch(x, zhat[:, m])
                line_delta = line_delta + w[:, m, None, None] * d_m
                presence = presence + w[:, m, None] * p_m
        h = self.cnn_in(x)
        for blk in self.cnn_blocks:
            h = h + 0.5 * blk(h)
        cnn_delta = self.cnn_out(h)
        delta = line_delta + cnn_delta
        logvar = self.logvar_head(x)
        return delta, logvar, presence


# ---------------------------------------------------------------------------
# Helpers shared by SR2 training, inference and evaluation
# ---------------------------------------------------------------------------
def topk_modes(probs: torch.Tensor, centers: torch.Tensor, k: int,
               suppress: int = 15, refine: int = 8):
    """Top-k *distinct* modes of P(z) -> ``(z (B, k), weight (B, k))``.

    Iterative argmax with +/- ``suppress``-bin suppression, so the k hypotheses
    are separate line-alias candidates rather than adjacent bins of one peak.
    Each is refined to a mode-local weighted mean and weighted by the
    probability mass inside its suppression window.
    """
    p = probs.clone()
    B, n = p.shape
    idx = torch.arange(n, device=p.device)[None, :]
    zs, ws = [], []
    for _ in range(k):
        mode = p.argmax(dim=1)
        near = (idx >= (mode - refine)[:, None]) & (idx <= (mode + refine)[:, None])
        wnear = probs * near
        zs.append((wnear * centers[None, :]).sum(1) / wnear.sum(1).clamp_min(1e-12))
        wide = (idx >= (mode - suppress)[:, None]) & (idx <= (mode + suppress)[:, None])
        ws.append((probs * wide * (p > 0)).sum(1))
        p = p.masked_fill(wide, 0.0)
    return torch.stack(zs, dim=1), torch.stack(ws, dim=1)


def constrain_delta(delta: torch.Tensor, cap: float) -> torch.Tensor:
    """Soft-clip the SR2 delta.

    The cap must exceed the tallest line the model has to reach: normalised
    SED lines peak above 30, and an early Roman run capped at 3 made them
    literally unreachable, saturating the gradient into "predict flat". 40 is
    the working value.
    """
    return torch.tanh(delta / cap) * cap if cap > 0 else delta


def build_line_mask(wave_hi_um: torch.Tensor, zhat: torch.Tensor, line_rest_um,
                    sigma_base_um: float = 0.005) -> torch.Tensor:
    """``(B,)`` redshifts -> ``(B, 1, L)`` Gaussian line-position mask."""
    device = zhat.device
    wave = wave_hi_um.to(device)
    line_rest = torch.as_tensor(line_rest_um, device=device, dtype=torch.float32)
    centers = line_rest[None, :] * (1.0 + zhat.reshape(-1, 1).clamp_min(0.0))
    d2 = (wave[None, None, :] - centers[..., None]) ** 2
    return torch.exp(-0.5 * d2 / (sigma_base_um ** 2 + 1e-12)).sum(1).clamp(0, 1)[:, None, :]


def line_profiles(z: torch.Tensor, wave_um: torch.Tensor, line_rest_um,
                  sigma_um: float = 0.005) -> torch.Tensor:
    """``(B, K, L)`` per-line Gaussian windows at the given redshifts."""
    device = z.device
    rest = torch.as_tensor(line_rest_um, device=device, dtype=torch.float32)
    centers = rest[None, :] * (1.0 + z.reshape(-1, 1).clamp_min(0.0))
    d2 = (wave_um[None, None, :] - centers[..., None]) ** 2
    return torch.exp(-0.5 * d2 / (sigma_um ** 2 + 1e-12))
