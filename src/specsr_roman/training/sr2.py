"""SR2 training loop."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..checkpoints import load_sr1, load_state_dict, push_checkpoint
from ..config import SR2Config, to_dict
from ..data import (
    RomanFixedGridDataset,
    filter_split_min_lines,
    get_or_make_group_split,
    get_or_make_split,
)
from ..inference.pipeline import build_sr2_input
from ..lines import LINE_LIST_REST_AA, SR1_LINES_AA, angstrom_to_micron
from ..models import (
    SR2Attention,
    ZHeadClf,
    build_line_mask,
    constrain_delta,
    line_profiles,
    load_zhead,
    soft_labels,
)
from .common import (
    configure_sdpa_backend,
    ensure_dir,
    finish_wandb,
    init_wandb,
    pick_device,
    set_seed,
    wandb_log,
)
from .losses import (
    line_flux_loss,
    line_hallucination_loss,
    presence_labels,
    sr2_reconstruction_loss,
)

__all__ = ["train"]


def train(cfg: SR2Config) -> dict:
    """Train SR2 on top of a frozen SR1 and ZHead.

    Checkpoint selection deserves the same warning as SR1's, only more so.
    Plain validation NLL is continuum-dominated and reliably selects the SR2
    that draws nothing. An earlier goal based on the line-region MSE ratio was
    no better: diluted across 98 line windows it is blind to sharpening, and
    it picked the most timid epoch available (strong-line amplitude 0.60,
    *worse* than the SR1 it was meant to improve).

    The goal used here is an amplitude metric --- ``-recov_amp + lam_hallu *
    hallu_amp`` --- integrated predicted-over-true flux on *recoverable*
    strong lines against the same quantity on *undetectable* ones. It selects
    the sharpest epoch that is not yet hallucinating, which on the published
    run is epoch 4. An early best epoch here is the design working, not a
    truncated run: hallucination amplitude climbs from 0.26 to 0.62 by epoch
    150 while recoverable amplitude barely moves.
    """
    set_seed(cfg.seed)
    configure_sdpa_backend()
    device = pick_device()
    out_dir = ensure_dir(cfg.out_dir)
    dataset_path = os.path.abspath(cfg.data)
    run_cfg = {"sigma_base_um": cfg.sigma_base_um, "z_topk": cfg.z_topk}

    # The dataset must expose photometry when, and only when, the ZHead
    # consumes it -- detect that from the checkpoint before building anything.
    zsd = load_state_dict(cfg.zhead_ckpt, map_location=device)
    zhead_has_phot = "phot_mu" in zsd
    full = RomanFixedGridDataset(dataset_path, with_phot=zhead_has_phot,
                                 phot_tier=cfg.phot_tier if zhead_has_phot else None)
    if zhead_has_phot and full.phot is None:
        raise SystemExit("the ZHead expects photometry but the dataset has none")

    if full.ids is not None:
        train_idx, test_idx, _ = get_or_make_group_split(dataset_path, full.ids)
    else:
        train_idx, test_idx, _ = get_or_make_split(dataset_path, len(full))
    train_idx, test_idx = filter_split_min_lines(
        train_idx, test_idx, full.z.numpy(), full.wave_hi, cfg.min_strong_lines)

    wave_hi_um = full.wave_hi.astype(np.float32) * 1e-4
    line_rest_um = angstrom_to_micron([w for _, w in LINE_LIST_REST_AA])
    # The anti-hallucination penalty covers only the ten redshift-carrying
    # strong lines -- the ones where a hallucinated delta would masquerade as a
    # real detection. Weak lines are already handled by line_flux_loss's
    # absent-line term.
    sr1_lines_um = angstrom_to_micron(list(SR1_LINES_AA))
    wave_t = torch.tensor(wave_hi_um, device=device)

    sr1 = load_sr1(cfg.sr1_ckpt, device=device)
    for p in sr1.parameters():
        p.requires_grad = False

    zhead = load_zhead(zsd).to(device).eval()
    zhead_is_clf = isinstance(zhead, ZHeadClf)
    if zhead_has_phot and zhead.n_phot != full.n_phot:
        raise SystemExit(
            f"the ZHead expects {zhead.n_phot} photometric bands but the dataset "
            f"supplies {full.n_phot}; set phot_tier to match how it was trained "
            f"(e.g. 'medium' for the Roman Medium tier)")
    for p in zhead.parameters():
        p.requires_grad = False
    print(f"ZHead loaded ({'clf/P(z)' if zhead_is_clf else 'regression'}) "
          f"from {cfg.zhead_ckpt}")

    # Optionally unfreeze the ZHead readout inside the coupled z-loss. Kept in
    # eval mode regardless, so dropout never perturbs the line placement.
    zhead_ft_params: list[torch.nn.Parameter] = []
    if cfg.zhead_finetune and cfg.lam_z > 0:
        mods = ([zhead.mlp, zhead.logits] if zhead_is_clf
                else [zhead.mlp, zhead.mu, zhead.log_var])
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad = True
                zhead_ft_params.append(p)
        print(f"ZHead readout fine-tuned: "
              f"{sum(p.numel() for p in zhead_ft_params)} params "
              f"at {cfg.zhead_lr_mult}x lr")

    ztr = full.z[train_idx].numpy()
    z_mean, z_std = float(ztr.mean()), float(ztr.std())
    z_min_n, z_max_n = (ztr.min() - z_mean) / z_std, (ztr.max() - z_mean) / z_std

    in_channels = 6  # LR flux, LR err, SR1 mean, SR1 sigma, line mask, zhat
    sr2 = SR2Attention(in_channels, line_rest_um, wave_hi_um).to(device)
    print(f"SR2: {sum(p.numel() for p in sr2.parameters()) / 1e6:.2f}M params, "
          f"K={sr2.K} line tokens")

    def run_batch(batch, lam_z_eff: float = 0.0, collect: bool = False):
        x_low = batch[0].to(device)                    # (B, 2, L) flux + err
        x_high = batch[1].unsqueeze(1).to(device)
        x_high_err = batch[2].unsqueeze(1).to(device)
        z_true = batch[3].to(device).float()
        line_snr = batch[6].to(device)                 # (B, K_strong)
        phot = batch[7].to(device) if zhead_has_phot else None

        with torch.no_grad():
            x_in, sr1_mean, z_modes, z_w, _ = build_sr2_input(
                x_low, sr1, zhead, wave_hi_um, line_rest_um, run_cfg, device,
                phot=phot, z_mean=z_mean, z_std=z_std,
                z_min_n=z_min_n, z_max_n=z_max_n)
            # Loss and metric masks use the TRUE redshift: a teacher signal,
            # loss-only. The model's own input keeps the predicted-z mask from
            # build_sr2_input, so nothing about inference sees the truth.
            line_mask = build_line_mask(wave_t, z_true, line_rest_um,
                                        sigma_base_um=cfg.sigma_base_um)

        delta, logvar, presence = sr2(x_in, z_modes, z_w)
        # Clamp before exp() in the NLL: the SR2 log-variance head is
        # unclamped and overflows to inf within one epoch otherwise, taking
        # the whole loss to NaN. SR1 clamps the same way.
        logvar = logvar.clamp(-8.0, 6.0)
        delta = constrain_delta(delta, cfg.delta_cap)
        sr2_mean = sr1_mean + delta

        loss, comps = sr2_reconstruction_loss(
            sr2_mean=sr2_mean, sr2_logvar=logvar, x_high=x_high,
            x_high_err=x_high_err, line_mask=line_mask, presence=presence,
            lam_hp_in=cfg.lam_hp_in, lam_hp_out=cfg.lam_hp_out, hp_k=cfg.hp_k,
            lam_sparse=cfg.lam_sparse)

        if cfg.lam_lineflux > 0:
            lf = line_flux_loss(sr2_mean, x_high, z_true, wave_t, line_rest_um,
                                sigma_um=cfg.sigma_base_um)
            loss = loss + cfg.lam_lineflux * lf
            comps["lineflux"] = float(lf.detach())
        if cfg.lam_hallu > 0:
            lh = line_hallucination_loss(sr2_mean, z_true, wave_t, sr1_lines_um,
                                         line_snr, snr_h0=cfg.hallu_snr0)
            loss = loss + cfg.lam_hallu * lh
            comps["hallu"] = float(lh.detach())
        if cfg.lam_presence > 0:
            with torch.no_grad():
                prof = line_profiles(z_true, wave_t, line_rest_um,
                                     sigma_um=cfg.sigma_base_um)
                labels = presence_labels(x_high, prof, thresh=cfg.presence_thresh)
            pres_bce = F.binary_cross_entropy(presence.clamp(1e-6, 1 - 1e-6), labels)
            loss = loss + cfg.lam_presence * pres_bce
            comps["presence_bce"] = float(pres_bce.detach())
        comps["presence_max"] = float(presence.max(dim=1).values.mean().detach())

        if lam_z_eff > 0:
            # Coupled z-loss: the ZHead re-reads SR2's own output and is
            # penalised against the true redshift, so SR2 is rewarded for
            # drawing lines that make z readable. The LR channels are ZEROED
            # here on purpose -- left in, the head reads z straight off the raw
            # LR spectrum and ignores SR2 entirely, which pins the term at the
            # head's own floor and applies zero line-drawing pressure.
            sr2_log_sigma = 0.5 * logvar
            if zhead_is_clf:
                z_in2 = torch.cat([torch.zeros_like(x_low), sr2_mean,
                                   sr2_log_sigma], dim=1)
                logits = zhead(z_in2)
                target = soft_labels(z_true, zhead.z_centers, cfg.label_sigma)
                z_loss = -(target * torch.log_softmax(logits, dim=-1)).sum(1).mean()
            else:
                z_in2 = torch.cat([sr2_mean, sr2_log_sigma], dim=1)
                mu_raw, _ = zhead(z_in2)
                mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
                z_pred_ft = mu_n.squeeze(-1) * z_std + z_mean
                dzn = (z_pred_ft - z_true) / (1 + z_true.abs())
                z_loss = F.smooth_l1_loss(dzn, torch.zeros_like(dzn), beta=0.1)
            loss = loss + lam_z_eff * z_loss
            comps["z_loss"] = float(z_loss.detach())

        if collect:
            with torch.no_grad():
                comps.update(_val_metrics(
                    sr2_mean, sr1_mean, x_high, x_low, logvar, z_true, line_mask,
                    line_snr, sr1_lines_um, wave_t, zhead, zhead_is_clf, cfg,
                    z_mean, z_std, z_min_n, z_max_n))
        return loss, comps

    if cfg.smoke:
        loader = DataLoader(Subset(full, train_idx[:64]), batch_size=32)
        sr2.train()
        for b in loader:
            loss, comps = run_batch(b, lam_z_eff=cfg.lam_z, collect=True)
            loss.backward()
            comps = {k: v for k, v in comps.items() if not k.startswith("z_p")
                     and not k.startswith("z_t")}
            print(f"smoke OK: loss {loss.item():.4f} {comps}")
            break
        return {"smoke": True}

    train_source = (RomanFixedGridDataset(
                        dataset_path, augment=True, with_phot=zhead_has_phot,
                        phot_tier=cfg.phot_tier if zhead_has_phot else None)
                    if cfg.augment else full)
    train_loader = DataLoader(Subset(train_source, train_idx),
                              batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(full, test_idx), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=max(1, cfg.num_workers // 2),
                            pin_memory=True)

    param_groups = [{"params": list(sr2.parameters()), "lr": cfg.lr}]
    if zhead_ft_params:
        param_groups.append({"params": zhead_ft_params,
                             "lr": cfg.lr * cfg.zhead_lr_mult})
    opt = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    run = init_wandb(cfg.wandb_project, cfg.run_name or cfg.out_prefix,
                     to_dict(cfg), tags=["sr2", "roman"], mode=cfg.wandb_mode)

    best_goal = float("inf")
    best_path = out_dir / f"{cfg.out_prefix}_best.pth"
    zheadft_path = out_dir / f"{cfg.out_prefix}_zheadft.pth"
    stats: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        lam_z_eff = cfg.lam_z * min(1.0, (epoch + 1) / max(1, cfg.lam_z_warmup))
        sr2.train()
        tr = 0.0
        for b in train_loader:
            loss, _ = run_batch(b, lam_z_eff)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(sr2.parameters(), cfg.grad_clip)
            if zhead_ft_params:
                nn.utils.clip_grad_norm_(zhead_ft_params, cfg.grad_clip)
            opt.step()
            tr += loss.item()
        tr /= max(1, len(train_loader))

        sr2.eval()
        acc = {k: 0.0 for k in ("va", "presence_mean", "presence_max", "hallu",
                                "line_se_sr1", "line_se_sr2", "line_w",
                                "recov_sum", "recov_n", "hallu_sum", "hallu_n")}
        zps, zts = [], []
        with torch.no_grad():
            for b in val_loader:
                loss, comps = run_batch(b, lam_z_eff, collect=True)
                acc["va"] += loss.item()
                for k in list(acc)[1:]:
                    acc[k] += comps.get(k, 0.0)
                zps.append(comps["z_pred"])
                zts.append(comps["z_true"])

        n = max(1, len(val_loader))
        va = acc["va"] / n
        npres, pmax = acc["presence_mean"] / n, acc["presence_max"] / n
        line_ratio = ((acc["line_se_sr2"] / max(acc["line_w"], 1e-8))
                      / max(acc["line_se_sr1"] / max(acc["line_w"], 1e-8), 1e-12))
        recov_amp = acc["recov_sum"] / max(acc["recov_n"], 1.0)
        hallu_amp = acc["hallu_sum"] / max(acc["hallu_n"], 1.0)
        zt = np.concatenate(zts)
        dz = (np.concatenate(zps) - zt) / (1 + zt)
        z_cat = float(np.mean(np.abs(dz) > 0.15))
        goal = -recov_amp + cfg.lam_hallu * hallu_amp

        stats = {"val_loss": va, "val_line_mse_ratio": line_ratio,
                 "val_recov_amp": recov_amp, "val_hallu_amp": hallu_amp,
                 "val_z_cat_sr2": z_cat}
        wandb_log(run, {"train_loss": tr, "presence_mean": npres,
                        "presence_max": pmax, "val_hallu": acc["hallu"] / n,
                        "ckpt_goal": goal, "epoch": epoch, **stats})
        print(f"epoch {epoch + 1}/{cfg.epochs}  train {tr:.4f}  val {va:.4f}  "
              f"presence {npres:.3f}/{pmax:.3f}  recov {recov_amp:.3f}  "
              f"hallu {hallu_amp:.3f}  z_cat {z_cat:.3f}  goal {goal:.4f}",
              flush=True)

        if goal < best_goal:
            best_goal = goal
            torch.save(sr2.state_dict(), best_path)
            if zhead_ft_params:
                torch.save(zhead.state_dict(), zheadft_path)
            print(f"  saved best (goal {goal:.4f}) -> {best_path}", flush=True)

    summary = {"best_goal": best_goal, "best_checkpoint": str(best_path), **stats}
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


def _val_metrics(sr2_mean, sr1_mean, x_high, x_low, logvar, z_true, line_mask,
                 line_snr, sr1_lines_um, wave_t, zhead, zhead_is_clf, cfg,
                 z_mean, z_std, z_min_n, z_max_n) -> dict:
    """Validation-only line-recovery and redshift-readability metrics.

    ``recov`` and ``hallu`` split *present* strong lines by whether the data
    could have revealed them (the dataset's ``line_snr`` labels): recoverable
    at S/N > 3, undetectable below 1. Integrated predicted-over-true flux in
    the first bin should rise above SR1's; in the second it should stay near
    zero. Those two numbers are the whole thesis of the stage.
    """
    m = line_mask
    out = {
        "line_se_sr2": float(((sr2_mean - x_high) ** 2 * m).sum()),
        "line_se_sr1": float(((sr1_mean - x_high) ** 2 * m).sum()),
        "line_w": float(m.sum()),
    }
    prof_s = line_profiles(z_true, wave_t, sr1_lines_um,
                           sigma_um=cfg.sigma_base_um)     # (B, Ks, L)
    f_true = (x_high * prof_s).sum(-1)
    f_pred = (sr2_mean * prof_s).sum(-1)
    ratio = f_pred / f_true.clamp_min(1e-6)
    present = f_true > cfg.presence_thresh
    recov = present & (line_snr > 3.0)
    unrec = present & (line_snr < 1.0)
    out["recov_sum"] = float((ratio * recov).sum())
    out["recov_n"] = float(recov.sum())
    out["hallu_sum"] = float((ratio.clamp_min(0.0) * unrec).sum())
    out["hallu_n"] = float(unrec.sum())

    # Redshift read back from SR2's own output, LR channels zeroed -- see the
    # coupled z-loss comment for why zeroing matters.
    zeroed = torch.cat([torch.zeros_like(x_low), sr2_mean, 0.5 * logvar], dim=1)
    if zhead_is_clf:
        zp, _ = zhead.predict_z(zeroed)
    else:
        mu_raw, _ = zhead(torch.cat([sr2_mean, 0.5 * logvar], dim=1))
        mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
        zp = mu_n.squeeze(-1) * z_std + z_mean
    out["z_pred"] = zp.reshape(-1).cpu().numpy()
    out["z_true"] = z_true.cpu().numpy()
    return out
