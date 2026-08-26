"""Typed configuration for the three training stages.

Every hyperparameter of the published models lives in ``configs/*.yaml`` and
is loaded into one of the dataclasses below. Two reasons this is not just
argparse:

* a run is reproducible from a file you can diff, rather than from a shell
  history line;
* the defaults here *are* the canonical chain. Constructing ``SR1Config()``
  with no arguments gives the settings that produced the published SR1
  checkpoint, so "what was this trained with" has a readable answer.

Values carried over from the JWST sweep are marked; they were not re-swept on
Roman, and re-sweeping them is open work.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

__all__ = ["SR1Config", "ZHeadConfig", "SR2Config", "load_config", "to_dict"]

T = TypeVar("T")


@dataclass
class SR1Config:
    """SR1: coarse super-resolution.

    Architecture and optimiser values are the JWST sweep optima, deliberately
    unchanged so the two instruments' models stay comparable. Everything under
    "recoverability" is Roman-specific and was tuned here.
    """

    # data
    data: str = "data/dataset/ou2024_h10307_dataset.npz"
    augment: bool = False          # see specsr_roman.data.augment for why not
    min_strong_lines: int = 0

    # architecture (JWST sweep)
    hidden_dim: int = 120
    num_res_blocks: int = 16
    dropout: float = 0.023538492919758583
    in_channels: int = 2           # [flux, err]

    # optimiser (JWST sweep)
    epochs: int = 200
    batch_size: int = 32
    lr: float = 8.23706977169561e-05
    weight_decay: float = 4.953557427559904e-05
    grad_clip: float = 0.5

    # predicted-variance handling
    use_var_clamp: bool = True
    var_clamp_min: float = 0.1
    var_clamp_max: float = 30.0
    logvar_reg: float = 1.5590946894260903e-06

    # line mask. mask_min_width 7 is the published value; Diffsky lines are
    # narrow (~5 px above threshold) so 3 is worth trying, but 7 is what the
    # released checkpoint used.
    mask_smooth_k: int = 161
    mask_thresh_mad: float = 8.0   # Roman: noiseless targets need a higher
                                   # threshold than the JWST sweep value 2.946,
                                   # which flagged 18% of all pixels
    mask_dilate: int = 11
    mask_min_width: int = 7

    # gated sharpness term
    lam_d1: float = 0.11
    lam_d2: float = 0.01
    gate_min_frac: float = 0.01282614837763313
    gate_temp: float = 0.2962018702777486
    score_w_line: float = 0.50334848260373
    score_w_recon: float = 0.2

    # sharpness warm-down: the term is most useful early, and left at full
    # strength it eventually trades calibration for contrast.
    sharp_wd_start_epoch: int = 25
    sharp_wd_rate: float = 0.008
    sharp_wd_floor: float = 0.15

    # recoverability (Roman-specific; see specsr_roman.training.losses)
    lam_lineflux: float = 1.0
    lineflux_snr0: float = 2.0
    lam_hallu: float = 0.5
    hallu_snr0: float = 1.0

    # checkpoint monitor smoothing
    ema_alpha: float = 0.9

    # run plumbing
    init_checkpoint: str | None = None
    out_prefix: str = "sr1_roman"
    out_dir: str = "runs/sr1"
    run_name: str | None = None
    seed: int = 42
    num_workers: int = 4
    progress: bool = False
    wandb_project: str = "roman-spectral-superresolution"
    wandb_mode: str | None = None
    push_to_hub: bool = False
    hub_repo: str | None = None
    export_predictions: bool = True


@dataclass
class ZHeadConfig:
    """ZHead: P(z) over a redshift grid, conditioned on grism + photometry."""

    # data
    data: str = "data/dataset/ou2024_h10307_dataset.npz"
    sr1_ckpt: str = "sr1_ou2024_v6"
    min_strong_lines: int = 0

    # architecture
    arch: str = "clf"              # clf | attn | conv (clf is canonical)
    hidden_dim: int = 64
    num_blocks: int = 4
    dropout: float = 0.1
    n_heads: int = 8

    # P(z) grid
    n_bins: int = 310
    z_lo: float = 0.0
    z_hi: float = 3.1
    label_sigma: float = 0.03
    refine_window: int = 8

    # photometry. `medium` = Roman F106/F129/F158, the bands that actually
    # ship with the HLWAS grism. `all` reproduces the leaked ablation.
    use_phot: bool = True
    phot_tier: str | None = "medium"
    phot_mag_err: float = 0.05
    phot_eval_mag_err: float = 0.05   # never score against noiseless truth

    # optimiser
    epochs: int = 150
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-5
    z_var_floor: float = 1e-6

    # run plumbing
    out_prefix: str = "zhead_roman"
    out_dir: str = "runs/zhead"
    run_name: str | None = None
    seed: int = 42
    num_workers: int = 4
    wandb_project: str = "roman-spectral-superresolution"
    wandb_mode: str | None = None
    push_to_hub: bool = False
    hub_repo: str | None = None


@dataclass
class SR2Config:
    """SR2: line-token attention refinement on top of frozen SR1 + ZHead."""

    # data + upstream stages
    data: str = "data/dataset/ou2024_h10307_dataset.npz"
    sr1_ckpt: str = "sr1_ou2024_v6"
    zhead_ckpt: str = "zhead_ou2024_roman_med3_noisy"
    phot_tier: str | None = "medium"   # must match how the ZHead was trained
    min_strong_lines: int = 0
    augment: bool = False

    # optimiser
    epochs: int = 150
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 2e-5
    grad_clip: float = 0.5

    # delta shaping
    delta_cap: float = 40.0        # normalised SED lines peak above 30
    sigma_base_um: float = 0.005
    z_topk: int = 3

    # reconstruction terms
    lam_hp_in: float = 3.0
    lam_hp_out: float = 0.3
    hp_k: int = 51
    lam_sparse: float = 0.0        # superseded by supervised presence
    lam_lineflux: float = 1.0
    lam_hallu: float = 1.0
    hallu_snr0: float = 1.0
    lam_presence: float = 0.3
    presence_thresh: float = 3.0
    label_sigma: float = 0.03

    # coupled z-loss. OFF in the canonical chain: it rewards drawing every
    # line at the predicted redshift, and with a photometry-fed ZHead already
    # supplying z it bought nothing while inflating hallucination amplitude to
    # 2.6x truth during warm-up.
    lam_z: float = 0.0
    lam_z_warmup: int = 5
    zhead_finetune: bool = True
    zhead_lr_mult: float = 0.1

    # run plumbing
    out_prefix: str = "sr2_roman"
    out_dir: str = "runs/sr2"
    run_name: str | None = None
    seed: int = 42
    num_workers: int = 4
    wandb_project: str = "roman-spectral-superresolution"
    wandb_mode: str | None = None
    push_to_hub: bool = False
    hub_repo: str | None = None
    smoke: bool = False


def to_dict(cfg) -> dict[str, Any]:
    """Dataclass -> plain dict, for W&B config and checkpoint cards."""
    return dataclasses.asdict(cfg)


def load_config(cls: type[T], path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> T:
    """Build a config from an optional YAML file plus optional overrides.

    Unknown keys are an error rather than a shrug: a typo'd hyperparameter
    that silently does nothing is the worst possible failure mode for a
    training run you will not look at again for six hours.
    """
    values: dict[str, Any] = {}
    if path is not None:
        import yaml
        with open(path) as fh:
            values.update(yaml.safe_load(fh) or {})
    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown config key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(known)}")
    return cls(**values)
