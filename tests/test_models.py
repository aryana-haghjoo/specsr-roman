"""Model shapes, initialisation invariants, and checkpoint round-trips."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from specsr_roman.grids import WAVE_HR
from specsr_roman.lines import LINE_LIST_REST_AA, angstrom_to_micron
from specsr_roman.models import (
    SR2Attention,
    SuperRes1D,
    ZHeadClf,
    constrain_delta,
    load_zhead,
    make_z_grid,
    pz_stats,
    soft_labels,
    topk_modes,
    z_metrics,
)

L = 2500


@pytest.fixture(scope="module")
def line_rest_um():
    return angstrom_to_micron([w for _, w in LINE_LIST_REST_AA])


# --- SR1 -----------------------------------------------------------------
def test_sr1_preserves_length_and_returns_two_heads():
    m = SuperRes1D(in_channels=2, hidden_dim=32, num_res_blocks=2)
    mean, log_var = m(torch.randn(3, 2, L))
    assert mean.shape == log_var.shape == (3, 1, L)


def test_sr1_starts_near_identity_in_its_uncertainty():
    """The log-variance head is deliberately initialised to a constant -2.0.

    A model must earn structure in its uncertainty rather than start with it.
    """
    m = SuperRes1D(in_channels=2, hidden_dim=32, num_res_blocks=2).eval()
    _, log_var = m(torch.randn(2, 2, L))
    assert torch.allclose(log_var, torch.full_like(log_var, -2.0), atol=1e-5)


def test_sr1_from_state_dict_infers_input_width():
    src = SuperRes1D(in_channels=2, hidden_dim=32, num_res_blocks=2)
    rebuilt = SuperRes1D.from_state_dict(src.state_dict(), hidden_dim=32,
                                         num_res_blocks=2)
    assert rebuilt.initial[0].in_channels == 2
    for a, b in zip(src.state_dict().values(),
                    rebuilt.state_dict().values(), strict=True):
        assert torch.equal(a, b)


# --- ZHead ---------------------------------------------------------------
def test_zhead_clf_outputs_logits_over_the_grid():
    centers = make_z_grid(0.0, 3.1, 310)
    h = ZHeadClf(centers, in_channels=4, n_phot=3)
    logits = h(torch.randn(2, 4, L), phot=torch.rand(2, 3) + 0.1)
    assert logits.shape == (2, 310)


def test_zhead_tolerates_missing_photometry():
    """SR2's coupled z-loss calls the head with phot=None deliberately."""
    centers = make_z_grid(0.0, 3.1, 64)
    h = ZHeadClf(centers, in_channels=4, n_phot=3)
    assert h(torch.randn(2, 4, L), phot=None).shape == (2, 64)


def test_zhead_round_trips_through_load_zhead():
    centers = make_z_grid(0.0, 3.1, 64)
    src = ZHeadClf(centers, in_channels=4, n_phot=3)
    rebuilt = load_zhead(src.state_dict())
    assert isinstance(rebuilt, ZHeadClf)
    assert rebuilt.n_phot == 3
    assert rebuilt.z_centers.numel() == 64


def test_pz_stats_takes_the_mode_not_the_mean():
    """The whole reason for a classification head.

    Given a bimodal P(z) -- one observed line consistent with two different
    identifications -- a Gaussian head must answer somewhere in between, which
    is wrong for both. The mode-local estimate picks one, and the reported
    sigma widens to say the answer is contested.
    """
    centers = make_z_grid(0.0, 3.0, 300)
    probs = torch.zeros(1, 300)
    probs[0, 50] = 0.6      # z ~ 0.5
    probs[0, 250] = 0.4     # z ~ 2.5
    probs = probs / probs.sum()
    zhat, sig = pz_stats(probs, centers, window=8)
    assert float(zhat) == pytest.approx(0.505, abs=0.02)   # the mode, not ~1.3
    assert float(sig) > 0.9                                # ambiguity is reported


def test_soft_labels_normalise_and_peak_at_the_truth():
    centers = make_z_grid(0.0, 3.0, 300)
    t = soft_labels(torch.tensor([1.5]), centers, 0.03)
    assert float(t.sum()) == pytest.approx(1.0)
    assert float(centers[t.argmax()]) == pytest.approx(1.5, abs=0.01)


def test_topk_modes_returns_distinct_aliases():
    centers = make_z_grid(0.0, 3.0, 300)
    probs = torch.zeros(1, 300)
    for b in (50, 150, 250):
        probs[0, b] = 1.0
    probs = probs / probs.sum()
    z, w = topk_modes(probs, centers, k=3)
    assert z.shape == w.shape == (1, 3)
    # Suppression must stop three peaks collapsing into three adjacent bins.
    assert float(z[0].sort().values.diff().min()) > 0.5


def test_z_metrics_flags_catastrophic_outliers():
    z_true = np.full(100, 1.0)
    z_pred = z_true.copy()
    z_pred[:10] = 2.0                       # dz/(1+z) = 0.5 -> catastrophic
    met = z_metrics(z_pred, z_true)
    assert met["catastrophic_frac"] == pytest.approx(0.10)


# --- SR2 -----------------------------------------------------------------
def test_sr2_forward_shapes(line_rest_um):
    wl = WAVE_HR.astype(np.float32) * 1e-4
    sr2 = SR2Attention(6, line_rest_um, wl)
    z = torch.tensor([[1.0, 0.5, 2.0], [1.2, 0.3, 1.8]])
    delta, logvar, presence = sr2(torch.randn(2, 6, L), z, torch.ones(2, 3))
    assert delta.shape == logvar.shape == (2, 1, L)
    assert presence.shape == (2, len(line_rest_um))


def test_sr2_starts_as_the_identity_on_sr1(line_rest_um):
    """The CNN output layer is zero-initialised on purpose.

    SR2 must begin by changing nothing, so any delta it later produces is
    something it learned rather than initialisation noise.
    """
    wl = WAVE_HR.astype(np.float32) * 1e-4
    sr2 = SR2Attention(6, line_rest_um, wl).eval()
    assert float(sr2.cnn_out.weight.detach().abs().max()) == 0.0
    assert float(sr2.cnn_out.bias.detach().abs().max()) == 0.0


def test_constrain_delta_caps_without_clipping_small_values():
    d = torch.tensor([0.1, 5.0, 1e6])
    out = constrain_delta(d, cap=40.0)
    assert float(out[0]) == pytest.approx(0.1, rel=1e-3)   # small: untouched
    assert float(out[2]) <= 40.0                           # large: capped
    # cap=0 disables the constraint entirely
    assert torch.equal(constrain_delta(d, cap=0.0), d)


def test_delta_cap_reaches_real_line_amplitudes():
    """Normalised SED lines peak above 30; an early cap of 3 made them
    unreachable and saturated the gradient into 'predict flat'."""
    strong = torch.tensor([35.0])
    assert float(constrain_delta(strong, cap=40.0)) > 20.0
    assert float(constrain_delta(strong, cap=3.0)) < 3.1
