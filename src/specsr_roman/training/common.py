"""Shared training plumbing: seeding, devices, W&B, validation plots.

The plotting helpers are not a nicety. A scalar loss curve cannot tell you
that a super-resolution model has quietly converged to the prior mean --- the
loss looks fine, and only a picture of a predicted spectrum next to its target
shows the flat line. Every run in this project logs example spectra and
residual histograms for that reason.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

__all__ = ["set_seed", "pick_device", "init_wandb", "wandb_log", "finish_wandb",
           "log_example_spectrum", "log_residual_histograms", "log_z_plots",
           "ensure_dir", "configure_sdpa_backend"]


def set_seed(seed: int = 42) -> None:
    """Seed every RNG a training run touches.

    The legacy global NumPy seed is deliberate, not an oversight: the
    augmentor and several third-party calls inside the loop draw from it, and
    a Generator here would leave those unseeded.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- see docstring
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure_sdpa_backend() -> None:
    """Force the math scaled-dot-product-attention kernel.

    The fused flash / mem-efficient kernels return NaN in backward for SR2's
    line-token transformer on Blackwell (RTX 5090, torch cu128). The
    transformer is 98 tokens wide, so the math kernel's speed cost is
    negligible and the numerical stability is not optional.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------
def init_wandb(project: str, name: str, config: dict, tags=None,
               mode: str | None = None):
    """Start a W&B run, or return ``None`` if W&B is unavailable/disabled.

    Training must not die because a metrics service is unreachable, so every
    call site treats the return value as optional.
    """
    try:
        import wandb
    except ImportError:
        print("wandb not installed -- metrics will not be synced")
        return None
    return wandb.init(project=project, name=name, config=config,
                      tags=list(tags or []),
                      mode=mode or os.environ.get("WANDB_MODE", "online"))


def wandb_log(run, payload: dict) -> None:
    if run is not None:
        run.log(payload)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()


def _log_figure(run, key: str, fig, epoch: int) -> None:
    if run is None:
        import matplotlib.pyplot as plt
        plt.close(fig)
        return
    import wandb
    run.log({key: wandb.Image(fig), "epoch": epoch})
    import matplotlib.pyplot as plt
    plt.close(fig)


def log_example_spectrum(run, wave_um, x_low_norm, x_high_norm, pred_norm,
                         sigma_norm, z, epoch, mean_hr, std_hr,
                         key: str = "val_example_spectrum") -> None:
    """De-normalised target / prediction / LR overlay with a 1-sigma band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_hr, std_hr = float(mean_hr), float(std_hr)
    x_high_dn = x_high_norm * std_hr + mean_hr
    pred_dn = pred_norm * std_hr + mean_hr
    sig_dn = sigma_norm * std_hr
    low_dn = x_low_norm * std_hr + mean_hr   # same grid, shared scale

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(wave_um, low_dn, color="0.75", lw=0.8, label="LR input (upsampled)")
    ax.plot(wave_um, x_high_dn, color="crimson", lw=1.2, label="Target (input SED)")
    ax.plot(wave_um, pred_dn, color="k", lw=1.0, label="prediction")
    ax.fill_between(wave_um, pred_dn - sig_dn, pred_dn + sig_dn,
                    color="k", alpha=0.15, label=r"$\pm1\sigma$")
    ax.set_xlabel("observed wavelength [µm]")
    ax.set_ylabel(r"$f_\lambda$ (de-normalized)")
    ax.set_title(f"validation example — epoch {epoch + 1}, z = {float(z):.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _log_figure(run, key, fig, epoch)


def log_residual_histograms(run, residuals, pred_sigma, epoch) -> None:
    """Residual and predicted-sigma distributions --- the calibration check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(residuals, bins=60)
    ax1.set_title("Residuals (SR - HR) [normalized]")
    ax1.set_xlabel("residual")
    ax1.set_ylabel("count")
    _log_figure(run, "val_residual_hist", fig1, epoch)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(pred_sigma, bins=60)
    ax2.set_title("Predicted sigma [normalized]")
    ax2.set_xlabel("sigma")
    ax2.set_ylabel("count")
    _log_figure(run, "val_uncertainty_hist", fig2, epoch)


def log_z_plots(run, z_true, z_pred, sig_z, epoch) -> None:
    """z_pred vs z_true and the dz/(1+z) distribution.

    The scatter panel is where alias structure shows up: catastrophic
    outliers land on characteristic diagonal stripes (one line mistaken for
    another), which a single NMAD number hides completely.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([0, 3.2], [0, 3.2], "k--", lw=0.8)
    axes[0].scatter(z_true, z_pred, s=3, alpha=0.25)
    axes[0].set_xlabel("true z")
    axes[0].set_ylabel("predicted z")
    axes[0].set_title(f"epoch {epoch + 1}")
    dz = (np.asarray(z_pred) - np.asarray(z_true)) / (1 + np.asarray(z_true))
    axes[1].hist(np.clip(dz, -0.5, 0.5), bins=80)
    axes[1].set_xlabel(r"$\Delta z / (1+z)$")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    _log_figure(run, "val_z_scatter", fig, epoch)

    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sig_z, bins=60)
    ax.set_xlabel(r"predicted $\sigma_z$")
    ax.set_ylabel("count")
    fig2.tight_layout()
    _log_figure(run, "val_sigz_hist", fig2, epoch)
