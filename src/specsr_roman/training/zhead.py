"""ZHead training loop."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..checkpoints import load_sr1, push_checkpoint
from ..config import ZHeadConfig, to_dict
from ..data import (
    RomanFixedGridDataset,
    apply_phot_noise,
    filter_split_min_lines,
    get_or_make_group_split,
    get_or_make_split,
    standardization_stats,
)
from ..models import (
    ZHead1D,
    ZHeadAttn,
    ZHeadClf,
    heteroscedastic_nll,
    make_z_grid,
    pz_stats,
    soft_labels,
    z_metrics,
)
from .common import (
    ensure_dir,
    finish_wandb,
    init_wandb,
    log_z_plots,
    pick_device,
    set_seed,
    wandb_log,
)

__all__ = ["train"]


def train(cfg: ZHeadConfig) -> dict:
    """Train the redshift head against a frozen SR1.

    Checkpoint selection is on **NMAD, not the loss**. On Roman the
    cross-entropy bottoms out early --- driven by how well the PDF width is
    calibrated --- while the point estimates keep improving for another
    hundred epochs. Selecting on the loss ships a worse model that is better
    at saying how unsure it is.
    """
    set_seed(cfg.seed)
    device = pick_device()
    out_dir = ensure_dir(cfg.out_dir)
    dataset_path = os.path.abspath(cfg.data)

    sr1 = load_sr1(cfg.sr1_ckpt, device=device)
    for p in sr1.parameters():
        p.requires_grad = False
    print(f"frozen SR1 loaded from {cfg.sr1_ckpt}")

    full_ds = RomanFixedGridDataset(dataset_path, with_phot=cfg.use_phot,
                                    phot_tier=cfg.phot_tier)
    if cfg.use_phot and full_ds.phot is None:
        raise SystemExit("use_phot is set but the dataset has no `phot` array")

    if full_ds.ids is not None:
        train_idx, test_idx, _ = get_or_make_group_split(dataset_path, full_ds.ids)
    else:
        train_idx, test_idx, _ = get_or_make_split(dataset_path, len(full_ds))
    train_idx, test_idx = filter_split_min_lines(
        train_idx, test_idx, full_ds.z.numpy(), full_ds.wave_hi,
        cfg.min_strong_lines)

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(Subset(full_ds, test_idx), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=max(1, cfg.num_workers // 2),
                            pin_memory=True)

    z_train = full_ds.z[train_idx].numpy()
    z_mean, z_std = float(z_train.mean()), float(z_train.std())
    z_min_n = (float(z_train.min()) - z_mean) / z_std
    z_max_n = (float(z_train.max()) - z_mean) / z_std
    print(f"z_mean={z_mean:.4f} z_std={z_std:.4f} "
          f"range=[{z_train.min():.3f},{z_train.max():.3f}]")

    is_clf = (cfg.arch == "clf")
    use_phot = bool(cfg.use_phot)
    if use_phot and not is_clf:
        raise SystemExit("photometry conditioning is only supported for arch='clf'")

    if is_clf:
        centers = make_z_grid(cfg.z_lo, cfg.z_hi, cfg.n_bins, device)
        n_phot = full_ds.n_phot if use_phot else 0
        # in_channels=4: [LR flux, LR err, SR1 mean, SR1 log-sigma]
        zhead = ZHeadClf(centers, in_channels=4, hidden_dim=cfg.hidden_dim,
                         num_blocks=cfg.num_blocks, dropout=cfg.dropout,
                         n_heads=cfg.n_heads, refine_window=cfg.refine_window,
                         n_phot=n_phot).to(device)
        if use_phot:
            mu, sig = standardization_stats(full_ds.phot[train_idx])
            zhead.phot_mu.copy_(torch.tensor(mu, device=device))
            zhead.phot_sig.copy_(torch.tensor(sig, device=device))
            print(f"photometry branch: {n_phot} bands {list(full_ds.phot_bands)}, "
                  f"train noise {cfg.phot_mag_err} mag, "
                  f"eval noise {cfg.phot_eval_mag_err} mag")
        print(f"P(z) grid: {cfg.n_bins} bins over [{cfg.z_lo},{cfg.z_hi}] "
              f"(dz={(cfg.z_hi - cfg.z_lo) / cfg.n_bins:.4f}), "
              f"label_sigma={cfg.label_sigma}")
    elif cfg.arch == "attn":
        zhead = ZHeadAttn(hidden_dim=cfg.hidden_dim, num_blocks=cfg.num_blocks,
                          dropout=cfg.dropout, n_heads=cfg.n_heads).to(device)
    else:
        zhead = ZHead1D(in_channels=2, hidden_dim=cfg.hidden_dim,
                        num_blocks=cfg.num_blocks, dropout=cfg.dropout).to(device)

    opt = torch.optim.AdamW(zhead.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    run = init_wandb(cfg.wandb_project, cfg.run_name or cfg.out_prefix,
                     to_dict(cfg) | {"z_mean": z_mean, "z_std": z_std},
                     tags=["zhead", "roman"], mode=cfg.wandb_mode)

    def make_input(x_low):
        """ZHead input channels from the LR spectrum plus frozen SR1.

        The classification head sees ``[LR flux, LR err, SR1 mean,
        SR1 log-sigma]``. Keeping the raw LR channels is the fix that made the
        head work: SR1 is conservative and smooths lines down to a few percent
        of their flux, so a head reading only SR1's output had almost nothing
        to locate. The LR spectrum still carries the line at its true observed
        wavelength, noise and all.
        """
        lr = x_low.to(device, non_blocking=True)               # (B, 2, L)
        with torch.no_grad():
            m, lv = sr1(lr)
        feat = torch.cat([m, 0.5 * lv], dim=1)                 # (B, 2, L)
        return torch.cat([lr, feat], dim=1) if is_clf else feat

    def clf_step(x_in, z, phot=None):
        logits = zhead(x_in, phot=phot)
        target = soft_labels(z, zhead.z_centers, cfg.label_sigma)
        loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=1).mean()
        probs = torch.softmax(logits.detach(), dim=-1)
        zhat, sig = pz_stats(probs, zhead.z_centers, cfg.refine_window)
        return loss, zhat.cpu().numpy(), z.cpu().numpy(), sig.cpu().numpy()

    def reg_step(x_in, z, phot=None):
        z_n = (z - z_mean) / z_std
        mu_raw, logvar_n = zhead(x_in)
        logvar_n = torch.clamp(logvar_n, min=-12.0, max=12.0)
        mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
        loss = heteroscedastic_nll(mu_n, logvar_n, z_n, cfg.z_var_floor)
        z_pred = (mu_n.detach().cpu().numpy() * z_std) + z_mean
        sig = np.exp(0.5 * logvar_n.detach().cpu().numpy()) * z_std
        return loss, z_pred, z.cpu().numpy(), sig

    step = clf_step if is_clf else reg_step

    # Validation photometric noise is drawn from a generator reseeded to the
    # same value every epoch, so the held-out metric is deterministic and
    # checkpoint selection is not comparing different noise draws.
    eval_gen = torch.Generator(device=device).manual_seed(1234)

    def batch_phot(batch, train: bool):
        if not use_phot:
            return None
        phot = batch[7].to(device, non_blocking=True)
        sigma = cfg.phot_mag_err if train else cfg.phot_eval_mag_err
        return apply_phot_noise(phot, sigma, None if train else eval_gen)

    best_nmad = float("inf")
    best_path = out_dir / f"{cfg.out_prefix}_best.pth"
    met: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        zhead.train()
        tr = 0.0
        for batch in train_loader:
            x_in = make_input(batch[0])
            loss, *_ = step(x_in, batch[3].to(device), batch_phot(batch, True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(zhead.parameters(), 1.0)
            opt.step()
            tr += loss.item()
        tr /= max(1, len(train_loader))

        zhead.eval()
        va = 0.0
        zp_all, zt_all, sig_all = [], [], []
        eval_gen.manual_seed(1234)                # identical val noise each epoch
        with torch.no_grad():
            for batch in val_loader:
                x_in = make_input(batch[0])
                loss, zp, zt, sg = step(x_in, batch[3].to(device),
                                        batch_phot(batch, False))
                va += loss.item()
                zp_all.append(zp)
                zt_all.append(zt)
                sig_all.append(sg)
        va /= max(1, len(val_loader))
        z_pred, z_true = np.concatenate(zp_all), np.concatenate(zt_all)
        sig_z = np.concatenate(sig_all)
        met = z_metrics(z_pred, z_true)

        wandb_log(run, {"train_loss": tr, "val_loss": va, "epoch": epoch, **met})
        log_z_plots(run, z_true, z_pred, sig_z, epoch)
        print(f"epoch {epoch + 1}/{cfg.epochs}  train {tr:.4f}  val {va:.4f}  "
              f"z_mae {met['z_mae']:.4f}  nmad {met['dz_nmad']:.4f}  "
              f"cat {met['catastrophic_frac']:.3f}", flush=True)

        if met["dz_nmad"] < best_nmad:
            best_nmad = met["dz_nmad"]
            torch.save(zhead.state_dict(), best_path)
            print(f"  saved best (nmad {best_nmad:.4f}) -> {best_path}", flush=True)

    summary = {"best_nmad": best_nmad, "best_checkpoint": str(best_path),
               "z_mean": z_mean, "z_std": z_std, **met}

    if cfg.push_to_hub:
        try:
            push_checkpoint(best_path, cfg.run_name or cfg.out_prefix,
                            meta={**summary, **to_dict(cfg),
                                  "wandb_url": run.get_url() if run else None},
                            repo_id=cfg.hub_repo)
        except Exception as exc:
            print(f"hub push failed (checkpoint is safe locally): {exc}", flush=True)

    finish_wandb(run)
    return summary
