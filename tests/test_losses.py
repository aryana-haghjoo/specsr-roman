"""Loss behaviour --- the properties that make the pipeline honest.

These are not shape tests. Each one encodes a claim about what the loss
rewards, and each claim was learned by watching a model exploit its absence.
"""

from __future__ import annotations

import pytest
import torch

from specsr_roman.lines import SR1_LINES_AA
from specsr_roman.training.losses import (
    line_flux_loss_weighted,
    line_hallucination_loss,
    loss_deblend_gated,
    make_line_mask_from_smoothed,
    masked_mad,
    presence_labels,
    smooth1d_avgpool,
)

L = 2500
WAVE_UM = torch.linspace(1.0, 1.93, L)
LINES_UM = [w * 1e-4 for w in SR1_LINES_AA]


def _spectrum_with_line(z: float, rest_aa: float, amplitude: float = 30.0):
    """One Gaussian emission line on a flat continuum, on the HR grid."""
    lam = rest_aa * (1 + z) * 1e-4
    x = torch.zeros(1, 1, L)
    x[0, 0] = amplitude * torch.exp(-0.5 * ((WAVE_UM - lam) / 0.0025) ** 2)
    return x


def test_masked_mad_survives_a_noiseless_continuum():
    """The Roman adaptation, in one assertion.

    A global MAD over a noiseless target is ~0, and dividing the sharpness
    term by it made that term ~3500x the NLL. Restricting to the line mask
    keeps it O(1).
    """
    x = _spectrum_with_line(1.0, 6563.0)
    mask = torch.zeros_like(x)
    mask[..., 1000:1100] = 1.0
    out = masked_mad(x, mask)
    assert torch.isfinite(out).all()
    # An all-empty mask must not produce NaN either.
    assert torch.isfinite(masked_mad(x, torch.zeros_like(x))).all()


def test_line_mask_finds_a_line_and_ignores_flat_continuum():
    line = _spectrum_with_line(1.0, 6563.0) + 1.0
    flat = torch.ones(1, 1, L)
    m_line = make_line_mask_from_smoothed(line, min_width=3)
    m_flat = make_line_mask_from_smoothed(flat, min_width=3)
    assert float(m_line.sum()) > 0
    assert float(m_flat.sum()) == 0.0


def test_hallucination_penalty_targets_undetectable_lines_only():
    """The squared knee is the whole design.

    An earlier linear weight was still 0.25 at S/N 6 --- a quarter-strength drag
    on lines we want drawn --- and it crushed recoverable recovery from 0.75 to
    0.20. This asserts the replacement leaves recoverable lines essentially
    alone while still penalising undetectable ones.
    """
    z = torch.tensor([1.0])
    drawn = _spectrum_with_line(1.0, 6563.0, amplitude=30.0)

    def penalty(snr):
        s = torch.zeros(1, len(LINES_UM))
        s[0, 3] = snr                                # index 3 == Halpha
        return float(line_hallucination_loss(drawn, z, WAVE_UM, LINES_UM, s))

    undetectable, marginal, recoverable = penalty(0.0), penalty(1.0), penalty(6.0)
    assert undetectable > marginal > recoverable
    assert marginal == pytest.approx(0.5 * undetectable, rel=0.05)
    assert recoverable < 0.05 * undetectable         # ~0.027 at S/N 6


def test_hallucination_penalty_ignores_absorption():
    """Only positive bumps are penalised, so real absorption is not punished.

    The penalty at the absorbed line itself is exactly zero. A small residual
    survives because high-passing a deep line leaves positive wings, and a
    neighbouring line's window can sit in one (here [SII] 6725, 0.032 um from
    Halpha, inside the 101-pixel smoothing kernel). The effect is bounded and
    two orders of magnitude below the emission case, which is the property
    that matters.
    """
    z = torch.tensor([1.0])
    snr = torch.zeros(1, len(LINES_UM))
    emission = _spectrum_with_line(1.0, 6563.0, amplitude=30.0)
    absorption = -emission

    p_emission = float(line_hallucination_loss(emission, z, WAVE_UM, LINES_UM, snr))
    p_absorption = float(line_hallucination_loss(absorption, z, WAVE_UM, LINES_UM, snr))
    assert p_emission > 0
    assert p_absorption < 0.01 * p_emission

    # At the absorbed line's own token the penalty is identically zero.
    rest = torch.tensor(LINES_UM)
    centers = rest[None, :] * (1.0 + z.reshape(-1, 1))
    d2 = (WAVE_UM[None, None, :] - centers[..., None]) ** 2
    prof = torch.exp(-0.5 * d2 / (0.005 ** 2))
    hp = absorption - smooth1d_avgpool(absorption, k=101)
    drawn = (hp * prof).sum(-1).clamp_min(0.0)
    assert float(drawn[0, 3]) == 0.0                 # index 3 == Halpha


def test_line_flux_loss_downweights_unrecoverable_but_not_absent_lines():
    """Permissive about missing the invisible; strict about inventing.

    A present-but-undetectable line the model misses should cost little. Flux
    drawn where the target has none should cost full price regardless of S/N.
    """
    z = torch.tensor([1.0])
    target = _spectrum_with_line(1.0, 6563.0, amplitude=30.0)
    missed = torch.zeros_like(target)                # model drew nothing

    def cost(pred, tgt, snr_val):
        s = torch.zeros(1, len(LINES_UM))
        s[0, 3] = snr_val
        return float(line_flux_loss_weighted(pred, tgt, z, WAVE_UM, LINES_UM, s))

    miss_undetectable = cost(missed, target, 0.0)
    miss_recoverable = cost(missed, target, 20.0)
    assert miss_undetectable < miss_recoverable      # forgiven when invisible

    # Invented flux against an empty target: weight 1 regardless of S/N.
    empty = torch.zeros_like(target)
    invented = _spectrum_with_line(1.0, 6563.0, amplitude=30.0)
    assert cost(invented, empty, 0.0) == pytest.approx(cost(invented, empty, 20.0))


def test_deblend_loss_returns_components_and_is_finite():
    x = _spectrum_with_line(1.0, 6563.0) + 1.0
    pred = x + 0.01 * torch.randn_like(x)
    total, comps = loss_deblend_gated(pred, torch.full_like(x, -2.0), x,
                                      torch.zeros_like(x), mask_min_width=3)
    assert torch.isfinite(total)
    assert {"loss_base_nll", "loss_sharp", "resid_rms"} <= set(comps)


def test_presence_labels_are_read_off_the_target():
    """SR2's presence head is supervised, not discovered.

    Two earlier generations left it to a sparsity prior and presence collapsed
    to zero, taking every line the stage was meant to draw with it.
    """
    z = torch.tensor([1.0])
    target = _spectrum_with_line(1.0, 6563.0, amplitude=30.0)
    rest = torch.tensor(LINES_UM)
    centers = rest[None, :] * (1.0 + z.reshape(-1, 1))
    d2 = (WAVE_UM[None, None, :] - centers[..., None]) ** 2
    prof = torch.exp(-0.5 * d2 / (0.005 ** 2))
    labels = presence_labels(target, prof)
    assert float(labels[0, 3]) == 1.0                # Halpha present
    assert float(labels[0, 0]) == 0.0                # [OII] absent


def test_smooth1d_avgpool_preserves_length_for_any_kernel():
    x = torch.randn(2, 1, 101)
    for k in (3, 4, 31, 500):
        assert smooth1d_avgpool(x, k=k).shape == x.shape
