"""SR1 training loop."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..checkpoints import push_checkpoint, resolve_checkpoint
from ..config import SR1Config, to_dict
from ..data import (
    RomanFixedGridDataset,
    filter_split_min_lines,
    get_or_make_group_split,
    get_or_make_split,
)
from ..lines import SR1_LINES_AA
from ..models import SuperRes1D, build_param_groups
from .common import (
    ensure_dir,
    finish_wandb,
    init_wandb,
    log_example_spectrum,
    log_residual_histograms,
    pick_device,
    set_seed,
    wandb_log,
)
from .losses import line_flux_loss_weighted, line_hallucination_loss, loss_deblend_gated

__all__ = ["train"]


def _clamp_logvar(log_var, cfg: SR1Config):
    if not cfg.use_var_clamp:
        return log_var
    return torch.clamp(log_var,
                       min=float(np.log(cfg.var_clamp_min)),
                       max=float(np.log(cfg.var_clamp_max)))


def train(cfg: SR1Config) -> dict:
    """Train SR1 and return a summary dict (best monitor value, paths).

    The checkpoint monitor is worth reading closely --- it is not the
    validation loss. Total val loss is dominated by the NLL, which *rises* as
    the model commits to line amplitudes: a confident half-amplitude line
    costs more than a hedged flat one. An early run had every line-recovery
    metric improving to epoch 150 while the loss picked a mid-run "best". The
    monitor here is the line-flux L1 (recovery) plus the hallucination penalty
    (so a checkpoint that buys recovery by reciting the prior is rejected),
    with residual RMS as a continuum tiebreak, smoothed by an EMA so a single
    lucky epoch cannot win.
    """
    set_seed(cfg.seed)
    device = pick_device()
    out_dir = ensure_dir(cfg.out_dir)
    dataset_path = os.path.abspath(cfg.data)

    run = init_wandb(cfg.wandb_project,
                     cfg.run_name or cfg.out_prefix,
                     to_dict(cfg),
                     tags=["sr1", "roman"],
                     mode=cfg.wandb_mode)

    # Validation always sees unaugmented data; a second, augmented instance
    # feeds the train split when requested.
    full_dataset = RomanFixedGridDataset(dataset_path)
    if full_dataset.ids is not None:
        train_idx, test_idx, _ = get_or_make_group_split(dataset_path,
                                                         full_dataset.ids)
    else:
        train_idx, test_idx, _ = get_or_make_split(dataset_path, len(full_dataset))
    train_idx, test_idx = filter_split_min_lines(
        train_idx, test_idx, full_dataset.z.numpy(), full_dataset.wave_hi,
        cfg.min_strong_lines)

    train_source = (RomanFixedGridDataset(dataset_path, augment=True)
                    if cfg.augment else full_dataset)
    train_dataset = Subset(train_source, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=max(1, cfg.num_workers // 2),
                             pin_memory=True)

    model = SuperRes1D(in_channels=cfg.in_channels, hidden_dim=cfg.hidden_dim,
                       num_res_blocks=cfg.num_res_blocks,
                       dropout=cfg.dropout).to(device)
    if cfg.init_checkpoint:
        state = torch.load(resolve_checkpoint(cfg.init_checkpoint),
                           map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"warm-start from {cfg.init_checkpoint}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected keys")

    wave_um = torch.tensor(full_dataset.wave_hi.astype(np.float32) * 1e-4,
                           device=device)
    sr1_lines_um = [w * 1e-4 for w in SR1_LINES_AA]

    optimizer = torch.optim.AdamW(
        build_param_groups(model, lr=cfg.lr, weight_decay=cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)

    best_path = out_dir / f"{cfg.out_prefix}_best.pth"
    # Separate stream for picking which validation spectrum to plot, so
    # changing the plotting cadence cannot perturb training.
    plot_rng = np.random.default_rng(cfg.seed)
    mon_smooth = None
    best_mon = float("inf")

    def loss_for(batch, lam_d1, lam_d2, collect=False):
        x_low, x_high, x_high_err, z_true = (batch[0].to(device, non_blocking=True),
                                             batch[1].unsqueeze(1).to(device, non_blocking=True),
                                             batch[2].unsqueeze(1).to(device, non_blocking=True),
                                             batch[3].to(device, non_blocking=True))
        line_snr = batch[6].to(device, non_blocking=True)
        best_snr = line_snr.max(dim=1).values
        row_w = best_snr / (best_snr + cfg.lineflux_snr0)

        mean, log_var = model(x_low)
        log_var = _clamp_logvar(log_var, cfg)
        loss, comps = loss_deblend_gated(
            mean, log_var, x_high, x_high_err,
            logvar_reg=cfg.logvar_reg,
            mask_smooth_k=cfg.mask_smooth_k, mask_thresh_mad=cfg.mask_thresh_mad,
            mask_dilate=cfg.mask_dilate, mask_min_width=cfg.mask_min_width,
            lam_d1=lam_d1, lam_d2=lam_d2,
            gate_min_frac=cfg.gate_min_frac, gate_temp=cfg.gate_temp,
            score_w_recon=cfg.score_w_recon, score_w_line=cfg.score_w_line,
            row_w=row_w)
        if cfg.lam_lineflux > 0:
            lf = line_flux_loss_weighted(mean, x_high, z_true, wave_um,
                                         sr1_lines_um, line_snr,
                                         snr0=cfg.lineflux_snr0)
            loss = loss + cfg.lam_lineflux * lf
            comps["lineflux"] = float(lf.detach())
        if cfg.lam_hallu > 0:
            lh = line_hallucination_loss(mean, z_true, wave_um, sr1_lines_um,
                                         line_snr, snr_h0=cfg.hallu_snr0)
            loss = loss + cfg.lam_hallu * lh
            comps["hallu"] = float(lh.detach())
        if collect:
            comps["recon_mse"] = float(((mean - x_high) ** 2).mean().detach())
            comps["_mean"], comps["_logvar"], comps["_high"] = mean, log_var, x_high
        return loss, comps

    for epoch in range(cfg.epochs):
        # Sharpness warm-down: the term does its work early, and holding it at
        # full strength for 200 epochs trades calibration for contrast.
        if epoch < cfg.sharp_wd_start_epoch:
            sharp_scale = 1.0
        else:
            sharp_scale = max(cfg.sharp_wd_floor,
                              1.0 - cfg.sharp_wd_rate * (epoch - cfg.sharp_wd_start_epoch))
        lam_d1, lam_d2 = cfg.lam_d1 * sharp_scale, cfg.lam_d2 * sharp_scale

        model.train()
        total = 0.0
        loop = train_loader
        if cfg.progress:
            from tqdm import tqdm
            loop = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        for batch in loop:
            loss, _ = loss_for(batch, lam_d1, lam_d2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            total += loss.item()
        avg_train = total / max(1, len(train_loader))

        model.eval()
        vtot = vrecon = vresid = vlf = vhallu = 0.0
        resid_sample = sigma_sample = None
        with torch.no_grad():
            for batch in test_loader:
                vloss, comps = loss_for(batch, lam_d1, lam_d2, collect=True)
                vtot += vloss.item()
                vrecon += comps["recon_mse"]
                vresid += comps["resid_rms"]
                vlf += comps.get("lineflux", 0.0)
                vhallu += comps.get("hallu", 0.0)
                if resid_sample is None:
                    resid_sample = (comps["_mean"] - comps["_high"]).flatten().cpu().numpy()
                    sigma_sample = torch.exp(0.5 * comps["_logvar"]).flatten().cpu().numpy()

            if run is not None:
                try:
                    j = int(plot_rng.integers(0, len(test_dataset)))
                    x_low_j, x_high_j, _, z_j, mean_hr_j, std_hr_j = test_dataset[j][:6]
                    pm, plv = model(x_low_j.unsqueeze(0).to(device))
                    plv = _clamp_logvar(plv, cfg)
                    log_example_spectrum(
                        run, full_dataset.wave_hi / 1e4, x_low_j[0].numpy(),
                        x_high_j.numpy(), pm[0, 0].cpu().numpy(),
                        torch.exp(0.5 * plv)[0, 0].cpu().numpy(),
                        z_j, epoch, mean_hr_j.item(), std_hr_j.item())
                    if (epoch % 5) == 0 and resid_sample is not None:
                        log_residual_histograms(run, resid_sample, sigma_sample, epoch)
                except Exception as exc:                      # plots are never fatal
                    print(f"plot logging failed: {exc}", flush=True)

        n_val = max(1, len(test_loader))
        avg_val = vtot / n_val
        if cfg.lam_lineflux > 0:
            mon_raw = float((vlf + cfg.lam_hallu * vhallu + 0.01 * vresid) / n_val)
        else:
            mon_raw = float(avg_val)
        mon_smooth = (mon_raw if mon_smooth is None
                      else cfg.ema_alpha * mon_smooth + (1.0 - cfg.ema_alpha) * mon_raw)
        scheduler.step(mon_smooth)

        wandb_log(run, {
            "avg_train_loss": float(avg_train),
            "avg_val_loss_raw": float(avg_val),
            "ckpt_monitor_raw": mon_raw,
            "ckpt_monitor_smooth": float(mon_smooth),
            "recon_error_val": float(vrecon / n_val),
            "val_resid_rms": float(vresid / n_val),
            "val_lineflux": float(vlf / n_val),
            "val_hallu": float(vhallu / n_val),
            "learning_rate": float(scheduler.optimizer.param_groups[0]["lr"]),
            "sharp_scale": float(sharp_scale),
            "epoch": epoch,
        })
        print(f"epoch {epoch + 1}/{cfg.epochs}  train {avg_train:.4f}  "
              f"val {mon_raw:.4f}  (smooth {mon_smooth:.4f})", flush=True)

        if mon_smooth < best_mon:
            best_mon = float(mon_smooth)
            torch.save(model.state_dict(), best_path)
            print(f"  saved best -> {best_path} (monitor {best_mon:.6f})", flush=True)

    summary = {"best_monitor": best_mon, "best_checkpoint": str(best_path),
               "n_train": len(train_idx), "n_test": len(test_idx)}

    if cfg.export_predictions:
        summary["predictions"] = str(_export_predictions(
            model, test_loader, test_idx, out_dir, cfg, device))
    torch.save(model.state_dict(), out_dir / f"{cfg.out_prefix}_final.pth")

    if cfg.push_to_hub:
        try:
            push_checkpoint(best_path, cfg.run_name or cfg.out_prefix,
                            meta={"best_monitor": best_mon, **to_dict(cfg),
                                  "wandb_url": run.get_url() if run else None},
                            repo_id=cfg.hub_repo)
        except Exception as exc:
            print(f"hub push failed (checkpoint is safe locally): {exc}", flush=True)

    finish_wandb(run)
    return summary


def _export_predictions(model, test_loader, test_idx, out_dir: Path,
                        cfg: SR1Config, device) -> Path:
    """Freeze the test-split predictions so downstream evaluation is reproducible."""
    model.eval()
    preds, sigs, lows, highs, zs, means_hr, stds_hr = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_low, x_high, _, z, mean_hr, std_hr = batch[:6]
            mean, log_var = model(x_low.to(device))
            log_var = _clamp_logvar(log_var, cfg)
            preds.append(mean[:, 0].cpu().numpy())
            sigs.append(torch.exp(0.5 * log_var)[:, 0].cpu().numpy())
            lows.append(x_low[:, 0].numpy())
            highs.append(x_high.numpy())
            zs.append(z.numpy())
            means_hr.append(mean_hr.numpy())
            stds_hr.append(std_hr.numpy())

    out_npz = out_dir / f"{cfg.out_prefix}_test_predictions.npz"
    np.savez(out_npz,
             test_indices=np.asarray(test_idx, dtype=np.int64),
             flux_super=np.concatenate(preds),
             flux_super_err=np.concatenate(sigs),
             flux_low=np.concatenate(lows),
             flux_high=np.concatenate(highs),
             z=np.concatenate(zs),
             mean_high=np.concatenate(means_hr),
             std_high=np.concatenate(stds_hr))
    print(f"saved predictions: {out_npz}")
    return out_npz
