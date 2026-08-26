"""Loss functions.

The losses are where the Roman port differs most from its JWST ancestor, and
the differences are not cosmetic --- they follow from one fact about the data:
**the targets are noiseless simulated SEDs.**

The JWST version got several scales for free from target noise. A robust MAD
over the target's derivative was a meaningful normaliser because the target
had noise; the high-pass magnitude of a real grating spectrum was O(1) because
it had noise. Feed those same terms a noiseless SED and the denominators
collapse toward zero, and terms that were balanced against the NLL at O(1)
arrive at 60x or 3500x its magnitude. Each such term is rescaled here by a
quantity measured *inside the line regions*, which is O(1) in either regime.

The second theme is recoverability. About two thirds of Roman grism rows carry
no line the data could possibly reveal. Under a plain reconstruction loss the
optimal policy on those rows is to hedge --- draw nothing, predict the prior
mean --- and that policy, averaged over the majority of the training set, is
what SR1 converged to before the weighting below was introduced. Two terms
express the fix: weight the reward for drawing a line by whether it is
detectable, and separately *penalise* drawn amplitude where it is not.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "robust_mad", "masked_mad", "smooth1d_avgpool", "highpass",
    "finite_diff", "finite_diff2", "keep_only_wide",
    "make_line_mask_from_smoothed",
    "line_flux_loss_weighted", "line_hallucination_loss", "loss_deblend_gated",
    "line_flux_loss", "presence_labels", "sr2_reconstruction_loss",
]


# ---------------------------------------------------------------------------
# Signal primitives
# ---------------------------------------------------------------------------
def robust_mad(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    med = x.median(dim=dim, keepdim=True).values
    mad = (x - med).abs().median(dim=dim, keepdim=True).values
    return 1.4826 * mad + eps


def masked_mad(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Robust MAD over masked pixels only, NaN-safe for empty masks.

    This is the Roman replacement for a global MAD: over a noiseless
    continuum the global value is ~0 and dividing by it detonates the
    sharpness term. Inside the line mask there is real structure in any noise
    regime.
    """
    xm = torch.where(mask > 0, x, torch.full_like(x, float("nan")))
    med = xm.nanmedian(dim=-1, keepdim=True).values
    mad = (xm - med).abs().nanmedian(dim=-1, keepdim=True).values
    s = 1.4826 * mad + eps
    return torch.nan_to_num(s, nan=1.0)


def smooth1d_avgpool(x: torch.Tensor, k: int = 31) -> torch.Tensor:
    """Odd-kernel moving average with reflect padding (length-preserving)."""
    k = int(k)
    L = x.shape[-1]
    k = min(k, max(3, L - 1))
    if k % 2 == 0:
        k -= 1
    k = max(k, 3)
    pad = k // 2
    xpad = F.pad(x, (pad, pad), mode="reflect")
    return F.avg_pool1d(xpad, kernel_size=k, stride=1)


def highpass(x: torch.Tensor, k: int = 51) -> torch.Tensor:
    """Continuum-subtracted spectrum: what is left is lines."""
    return x - smooth1d_avgpool(x, k=k)


def finite_diff(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., 1:] - x[..., :-1]
    return F.pad(dx, (0, 1), mode="replicate")


def finite_diff2(x: torch.Tensor) -> torch.Tensor:
    return finite_diff(finite_diff(x))


def keep_only_wide(mask: torch.Tensor, min_width: int = 7) -> torch.Tensor:
    """Erase mask features narrower than ``min_width`` contiguous pixels.

    Removes single-pixel noise spikes from the line mask. The value is
    simulation-dependent: 7 works on Galacticus, but Diffsky (OU2024) lines
    are narrower --- about 5 px above threshold --- and 7 erases *every* mask,
    which silently disables the sharpness term. Use 3 for OU2024.
    """
    min_width = max(int(min_width), 1)
    w = torch.ones(1, 1, min_width, device=mask.device)
    counts = F.conv1d(mask, w, padding=min_width // 2)
    return (counts >= float(min_width)).float()


def make_line_mask_from_smoothed(x_high_raw: torch.Tensor, smooth_k: int = 121,
                                 thresh_mad: float = 7.5, dilate: int = 11,
                                 min_width: int = 7) -> torch.Tensor:
    """Data-driven line mask: high-pass, threshold at ``thresh_mad`` MAD, clean up.

    Derived from the target rather than from a redshift, so it is available
    even when the redshift is not, and it flags whatever structure is
    genuinely there.
    """
    x_smooth = smooth1d_avgpool(x_high_raw, k=smooth_k)
    hp = x_high_raw - x_smooth
    scale = robust_mad(hp, dim=-1)
    z = hp.abs() / scale
    mask = (z > float(thresh_mad)).float()
    mask = keep_only_wide(mask, min_width=min_width)
    if dilate and int(dilate) > 1:
        k = int(dilate)
        pad = k // 2
        mask = F.max_pool1d(F.pad(mask, (pad, pad), mode="replicate"),
                            kernel_size=k, stride=1)
    return mask


# ---------------------------------------------------------------------------
# SR1
# ---------------------------------------------------------------------------
def line_flux_loss_weighted(mean, x_high, z_true, wave_um, line_rest_um,
                            line_snr, sigma_um: float = 0.005,
                            floor: float = 5.0, snr0: float = 2.0,
                            present_thresh: float = 5.0) -> torch.Tensor:
    """Integrated per-line flux L1 at true line positions, weighted by recoverability.

    Plain reconstruction losses saturate: a 30-sigma amplitude miss and a
    3-sigma one look similar once averaged over 2500 pixels, and the NLL can
    always be bought off by inflating the predicted variance at line pixels.
    Integrating the residual over each line window gives an unsaturated,
    per-line gradient that actually pushes a drawn line from 15% to 100%
    amplitude.

    The weighting is what stops the hedging failure. Each *present* line is
    weighted ``snr/(snr + snr0)``, so lines the data cannot support contribute
    almost no gradient; *absent* lines keep weight 1, so drawing flux where
    there is none stays fully penalised. Asymmetric on purpose: we are
    permissive about failing to find the invisible, and strict about inventing.
    """
    device = mean.device
    rest = torch.as_tensor(line_rest_um, device=device, dtype=torch.float32)
    centers = rest[None, :] * (1.0 + z_true.reshape(-1, 1).clamp_min(0.0))
    d2 = (wave_um[None, None, :] - centers[..., None]) ** 2
    prof = torch.exp(-0.5 * d2 / (sigma_um ** 2 + 1e-12))        # (B, K, L)
    resid = ((mean - x_high) * prof).sum(-1)                     # (B, K)
    f_true = (x_high * prof).sum(-1)
    w_rec = line_snr / (line_snr + float(snr0))
    w = torch.where(f_true > float(present_thresh), w_rec, torch.ones_like(w_rec))
    return (w * resid.abs() / (f_true.abs() + float(floor))).mean()


def line_hallucination_loss(mean, z_true, wave_um, line_rest_um, line_snr,
                            sigma_um: float = 0.005, floor: float = 5.0,
                            snr_h0: float = 1.0, smooth_k: int = 101) -> torch.Tensor:
    """Penalise drawn line amplitude the data cannot support.

    Recoverability weighting alone was not enough: given 300 epochs, SR1 still
    learned to recite the prior, drawing ~23% amplitude on lines that were
    present in the target but had integrated LR S/N below 1 --- indistinguishable,
    to a user, from a real weak detection. This term measures the model's *own*
    drawn flux (high-pass of the prediction, integrated over each line window)
    and penalises positive bumps by a non-recoverability weight.

    The weight's shape matters more than its size. A first attempt used the
    complement of the recovery weight, ``snr0/(snr + snr0)``, which is still
    0.25 at S/N 6 --- a quarter-strength drag on exactly the lines we want
    drawn --- and it crushed recoverable recovery from 0.75 to 0.20. The
    squared knee used here, ``snr_h0^2 / (snr^2 + snr_h0^2)``, is 0.5 at
    ``snr_h0``, 0.1 at 3x and 0.03 at 6x: recoverable lines are left alone
    while undetectable ones are driven back to the continuum.

    Only emission (positive) bumps are penalised, so real absorption is safe.
    One caveat: high-passing a deep absorption line leaves positive wings, and
    a neighbouring line's window can fall inside one (within roughly
    ``smooth_k`` pixels). The induced penalty is around two orders of magnitude
    below the emission case, so it does not drive training, but it is not
    identically zero.
    """
    device = mean.device
    rest = torch.as_tensor(line_rest_um, device=device, dtype=torch.float32)
    centers = rest[None, :] * (1.0 + z_true.reshape(-1, 1).clamp_min(0.0))
    d2 = (wave_um[None, None, :] - centers[..., None]) ** 2
    prof = torch.exp(-0.5 * d2 / (sigma_um ** 2 + 1e-12))        # (B, K, L)
    mean_hp = mean - smooth1d_avgpool(mean, k=smooth_k)          # (B, 1, L)
    drawn = (mean_hp * prof).sum(-1).clamp_min(0.0)              # (B, K)
    h0 = float(snr_h0) ** 2
    w_unrec = h0 / (line_snr ** 2 + h0)
    return (w_unrec * drawn / float(floor)).mean()


def loss_deblend_gated(mean, log_var, x_high, x_high_err,
                       logvar_reg: float = 3.4e-6,
                       mask_smooth_k: int = 121, mask_thresh_mad: float = 7.5,
                       mask_dilate: int = 11, mask_min_width: int = 7,
                       lam_d1: float = 0.11, lam_d2: float = 0.0102,
                       gate_min_frac: float = 0.015, gate_temp: float = 0.05,
                       score_w_recon: float = 0.2, score_w_line: float = 2.0,
                       row_w=None, eps: float = 1e-12):
    """SR1's main objective: Gaussian NLL plus a gated, line-masked sharpness term.

    The sharpness term compares first and second derivatives inside the line
    mask, which is what teaches deblending --- matching a blended complex's
    *shape* rather than only its integral. It is gated by how much of the row
    is masked, so rows with no lines do not get pushed toward spurious
    structure, and optionally by ``row_w`` (recoverability), so rows whose best
    line is buried in noise stop teaching the term to prefer flat outputs.

    Returns ``(total, components)``; the components dict is what the training
    loop logs.
    """
    model_var = torch.exp(log_var)
    total_var = (model_var + (x_high_err ** 2)).clamp_min(1e-8)
    nll = 0.5 * (torch.log(total_var + eps) + (mean - x_high) ** 2 / (total_var + eps))
    base_loss = nll.mean()

    reg = float(logvar_reg) * (log_var ** 2).mean()

    line_mask = make_line_mask_from_smoothed(
        x_high_raw=x_high, smooth_k=mask_smooth_k, thresh_mad=mask_thresh_mad,
        dilate=mask_dilate, min_width=mask_min_width)

    frac = line_mask.mean(dim=-1, keepdim=True)
    gate = torch.sigmoid((frac - float(gate_min_frac)) / float(gate_temp)).detach()

    d1_pred, d1_tgt = finite_diff(mean), finite_diff(x_high)
    d2_pred, d2_tgt = finite_diff2(mean), finite_diff2(x_high)
    # Roman adaptation: the JWST version normalised by the MAD over the whole
    # target derivative, whose scale came from target noise. Noiseless SED
    # targets make that ~0 and the sharpness term explodes (~3500x the NLL).
    # Normalise by the derivative MAD *within the line mask* instead: O(1) in
    # any noise regime.
    s1 = masked_mad(d1_tgt, line_mask)
    s2 = masked_mad(d2_tgt, line_mask)
    denom = (line_mask.sum(dim=-1) + 1e-6)
    sharp1 = (((d1_pred - d1_tgt).abs() / s1) * line_mask).sum(dim=-1) / denom
    sharp2 = (((d2_pred - d2_tgt).abs() / s2) * line_mask).sum(dim=-1) / denom
    gate_row = gate.squeeze(-1)
    if row_w is not None:
        gate_row = gate_row * row_w.detach().reshape(gate_row.shape)
    sharp_loss = (gate_row * (float(lam_d1) * sharp1 + float(lam_d2) * sharp2)).mean()

    total = float(score_w_recon) * base_loss + float(score_w_line) * sharp_loss + reg

    with torch.no_grad():
        resid = mean - x_high
        comps = {
            "loss_base_nll": base_loss.detach().item(),
            "loss_logvar_reg": reg.detach().item(),
            "loss_sharp": sharp_loss.detach().item(),
            "loss_total": total.detach().item(),
            "mask_frac_mean": frac.mean().detach().item(),
            "gate_mean": gate.mean().detach().item(),
            "resid_rms": resid.pow(2).mean().sqrt().detach().item(),
            "total_var_p50": total_var.median().detach().item(),
        }
    return total, comps


# ---------------------------------------------------------------------------
# SR2
# ---------------------------------------------------------------------------
def line_flux_loss(sr2_mean, x_high, z_true, wave_um, line_rest_um,
                   sigma_um: float = 0.005, floor: float = 5.0) -> torch.Tensor:
    """Unweighted integrated per-line flux L1 at true line positions.

    SR2's version of the teacher term. Lines absent from the target integrate
    to ~0, so drawing flux there is penalised by the same expression --- which
    is how the stage learned to stop misallocating flux to Mg b and [NI].
    ``floor`` turns strong lines into a relative error while keeping the
    absent-line penalty finite.
    """
    device = sr2_mean.device
    rest = torch.as_tensor(line_rest_um, device=device, dtype=torch.float32)
    centers = rest[None, :] * (1.0 + z_true.reshape(-1, 1).clamp_min(0.0))
    d2 = (wave_um[None, None, :] - centers[..., None]) ** 2
    prof = torch.exp(-0.5 * d2 / (sigma_um ** 2 + 1e-12))        # (B, K, L)
    resid = ((sr2_mean - x_high) * prof).sum(-1)                 # (B, K)
    scale = (x_high * prof).sum(-1).abs() + floor
    return (resid.abs() / scale).mean()


def presence_labels(x_high, prof, hp_k: int = 101, thresh: float = 3.0) -> torch.Tensor:
    """Ground-truth per-line presence from the noiseless HR target.

    We *know* which lines each target has, so the presence head is supervised
    with BCE against these labels rather than left to discover them under a
    sparsity prior --- which it never did: presence collapsed to zero in two
    successive SR2 generations, and with it every line the stage was meant to
    draw.
    """
    resid = x_high - smooth1d_avgpool(x_high, k=hp_k)
    f_line = (resid * prof).sum(-1)                              # (B, K)
    return (f_line > thresh).float()


def sr2_reconstruction_loss(*, sr2_mean, sr2_logvar, x_high, x_high_err,
                            line_mask, presence, lam_hp_in: float,
                            lam_hp_out: float, hp_k: int, lam_sparse: float,
                            var_floor: float = 1e-8):
    """SR2 NLL plus in-line and out-of-line high-pass matching.

    Splitting the high-pass term by the line mask lets the two regions carry
    different weights: sharpen hard where lines are, stay quiet elsewhere.
    """
    model_var = torch.exp(sr2_logvar)
    total_var = (model_var + x_high_err ** 2).clamp_min(var_floor)
    nll = 0.5 * (torch.log(total_var + 1e-12)
                 + (sr2_mean - x_high) ** 2 / (total_var + 1e-12))
    loss = nll.mean()

    hp_sr2, hp_hr = highpass(sr2_mean, k=hp_k), highpass(x_high, k=hp_k)
    # Roman adaptation: noiseless targets make |highpass(HR)| ~0 over the
    # continuum, so a global median collapses to the 1e-3 floor and hp_diff
    # explodes (~60x the NLL). Scale by the high-pass magnitude *within* line
    # regions -- the JWST version got a sane scale for free from target noise.
    hp_in_line = (hp_hr.detach().abs() * line_mask).sum() / line_mask.sum().clamp_min(1e-8)
    hp_scale = hp_in_line.clamp_min(1e-3)
    hp_diff = (hp_sr2 - hp_hr) / hp_scale
    hp_loss = F.smooth_l1_loss(hp_diff, torch.zeros_like(hp_diff),
                               reduction="none", beta=0.1)
    m = line_mask
    if lam_hp_in > 0:
        loss = loss + lam_hp_in * (m * hp_loss).sum() / m.sum().clamp_min(1e-8)
    if lam_hp_out > 0:
        loss = loss + lam_hp_out * ((1 - m) * hp_loss).sum() / (1 - m).sum().clamp_min(1e-8)
    if lam_sparse > 0:
        loss = loss + lam_sparse * presence.mean()
    return loss, {"nll": float(nll.mean().detach()),
                  "presence_mean": float(presence.mean().detach())}
