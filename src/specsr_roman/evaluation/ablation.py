"""Photometry ablation --- how much of the redshift accuracy is the spectrum?

The redshift head takes two inputs: the grism spectrum (raw plus SR1's
reconstruction) and three Roman Medium-tier colours. A number from that head
is only interpretable if you know which input produced it, and the way to find
out is to take the photometry away.

Photometry enters the head standardised as
``(log10(flux) - phot_mu) / phot_sig``, with the statistics baked into the
checkpoint, so "remove the colours" is exactly "feed every band its training
mean", which standardises to zero. That is the same vector the head receives
from :meth:`RomanPipeline.predict` when it is called with ``phot=None``, so
the ``grism only`` row below is a measurement of the deployed model in that
mode rather than of a hypothetical one.

**Read the ``grism only`` row as an upper bound, not as an information floor.**
This head was *trained* with colours. Handing it a mean-imputed colour vector
tells you what the deployed chain does when photometry is missing; it does not
tell you how well a head trained without colours would do, because the two
differ by everything the network learned to delegate to the photometry branch.
A grism-only head is a separate experiment and has not been run. The physical
floor is set by the alias degeneracy --- with a single line in band, H-alpha,
[O III] and [O II] are mutually consistent --- and no architecture removes it.

**Why there is no "zero the spectrum" row.** The obvious complement --- keep the
photometry, blank the spectrum, see what the colours alone can do --- does not
work here, and reporting it would be worse than reporting nothing. Masking a
band to its training mean is *in distribution*: it standardises to exactly 0,
a value the network sees constantly. There is no equivalent for the spectral
channels. Feeding SR1 a zero array produces a reconstruction and an uncertainty
map unlike anything in training, and the head then reads nonsense from two of
its four channels. Measured, that configuration scores *worse* than removing
the photometry --- which tells you the input was out of distribution, not what
the photometry contributes. The ``grism only`` row answers the answerable half
of the question; the other half needs a head trained without the spectrum.

The noise sweep is the quantitative version of the same question. It perturbs
the three colours with the multiplicative log-normal jitter used in training,
so the small levels are in distribution and the degradation is real rather
than an artefact of an unfamiliar input. Each level is drawn from a generator
reseeded to the same value, so the sweep varies only sigma.

Everything above uses the three Roman Medium-tier bands that ship with the
HLWAS grism, and nothing else.

**The band-set comparison** asks the opposite question: what does adding Rubin
*ugrizy* to those three bands buy? Each row is a separately trained head ---
same architecture, same frozen SR1, same split, differing only in the
photometry it is given --- scored with the noise it was trained on. The Rubin
bands carry the sky-limited noise of a coadd of stated depth on top of the
0.05 mag every band gets; the depth is stored in the checkpoint
(``phot_sigma``), so the evaluation cannot disagree with the training about
it. The comparison is broken down by redshift and by best-line S/N, because
those two are confounded and a gain quoted for the whole sample is an average
over very different populations. Any colour gain measured on OpenUniverse2024
is an upper bound: the photometry and the spectrum come from the same SED.

**The leak controls** put a number on how much of that gain survives
photometry that is not perfectly consistent with the spectrum. Two
perturbations, applied at evaluation to heads that never saw them: a fixed
zero-point offset per Rubin band (a calibration error between the two
surveys, the same for every source), and a per-source factor common to the
six Rubin bands (ground-based and space-based fluxes measured in different
apertures). Both leave the Roman bands and the spectrum alone. A head that
falls apart at a few hundredths of a magnitude was reading the simulation's
internal consistency, not the colours.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..checkpoints import load_sr1, load_zhead_ckpt
from ..data import RomanFixedGridDataset, get_or_make_group_split
from ..grids import PHOT_BANDS, ROMAN_MEDIUM_BANDS, ROMAN_MEDIUM_RUBIN_BANDS
from ..models import pz_stats, z_metrics
from .metrics import MIN_BEST_LINE_SNR

__all__ = ["AblationConfig", "run_ablation", "plot_ablation",
           "run_band_comparison", "plot_band_comparison", "run_leak_controls"]

#: Photometric noise levels for the sweep, in magnitudes. 0.05 is the level
#: the deployable head was trained and evaluated at.
NOISE_MAG = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


@dataclass
class AblationConfig:
    data: str = "data/dataset/ou2024_h10307_dataset.npz"
    sr1_ckpt: str = "sr1_ou2024_v6"
    #: The deployable Roman Medium-tier head. There is deliberately no
    #: wider-band head here: a model fed bands the survey does not deliver
    #: alongside the grism reads the redshift off an effectively complete SED
    #: and measures the simulation rather than the instrument.
    zhead_ckpt: str = "zhead_ou2024_roman_med3_noisy"
    roman_bands: tuple[int, ...] = ROMAN_MEDIUM_BANDS
    eval_mag_err: float = 0.05
    noise_seed: int = 0
    batch_size: int = 64
    out_dir: str = "outputs"
    noise_levels: tuple[float, ...] = NOISE_MAG
    #: Heads trained with Rubin *ugrizy* added to the three Roman bands, as
    #: ``(label, checkpoint)``. A checkpoint that is not on disk is skipped.
    rubin_heads: tuple[tuple[str, str], ...] = (
        ("+ Rubin, 1 yr", "runs/zhead/zhead_ou2024_roman_med3_rubin_y1_best.pth"),
        ("+ Rubin, 10 yr", "runs/zhead/zhead_ou2024_roman_med3_rubin_y10_best.pth"),
    )
    rubin_bands: tuple[int, ...] = ROMAN_MEDIUM_RUBIN_BANDS
    #: The control: the same nine bands with the spectrum blanked at train
    #: and eval. What it cannot do is what the grism contributes.
    photonly_heads: tuple[tuple[str, str], ...] = (
        ("9 bands, no spectrum",
         "runs/zhead/zhead_ou2024_photonly_rubin_y10_best.pth"),
    )


@torch.no_grad()
def _zhead_inputs(sr1, loader, device):
    """Run SR1 once and cache the 4-channel ZHead input for the whole split.

    Every configuration below shares these channels --- only the photometry
    changes --- so SR1 runs once rather than once per row. That is what keeps
    the sweep cheap enough to run on a CPU.
    """
    xs, phots, zs, snrs = [], [], [], []
    for batch in loader:
        x_low, z, phot = batch[0], batch[3], batch[7]
        snrs.append(batch[6].max(dim=1).values)
        lr = x_low.to(device, non_blocking=True)      # (B, 2, L) [flux, err]
        m, lv = sr1(lr)
        xs.append(torch.cat([lr, m, 0.5 * lv], dim=1).cpu())
        phots.append(phot)
        zs.append(z)
    return (torch.cat(xs), torch.cat(phots), torch.cat(zs).numpy(),
            torch.cat(snrs).numpy())


@torch.no_grad()
def _predict(zhead, x_in, phot, device, *, drop_phot=False,
             noise_mag=0.0, gen=None, batch_size=256):
    """Point redshifts. ``drop_phot`` replaces the colours with training means.

    A head trained with depth noise (non-zero ``phot_sigma``) gets it here
    too, after the multiplicative draw, exactly as in training.
    """
    preds = []
    for i in range(0, len(x_in), batch_size):
        xb = x_in[i:i + batch_size].to(device)
        if drop_phot:
            # None -> the head substitutes a standardised zero vector, i.e.
            # every band at its training mean. Identical to pipeline
            # predict(phot=None); see the module docstring.
            pb = None
        else:
            pb = phot[i:i + batch_size].to(device)
            if noise_mag > 0:
                dm = noise_mag * torch.randn(pb.shape, generator=gen,
                                             device=pb.device)
                pb = pb * torch.pow(10.0, -0.4 * dm)
            if bool((zhead.phot_sigma > 0).any()):
                pb = pb + zhead.phot_sigma * torch.randn(
                    pb.shape, generator=gen, device=pb.device)
        probs = torch.softmax(zhead(xb, phot=pb), dim=-1)
        zhat, _ = pz_stats(probs, zhead.z_centers, zhead.refine_window)
        preds.append(zhat.cpu().numpy())
    return np.concatenate(preds)


def _score(zhead, x_in, phot, z_true, device, **kwargs):
    return z_metrics(_predict(zhead, x_in, phot, device, **kwargs), z_true)


#: Bins for the band-set breakdown. Redshift edges follow which strong lines
#: the grism holds (H-alpha enters at 0.52 and leaves at 1.94); S/N edges
#: separate spectra without a detected line from the recoverability bins.
Z_EDGES = (0.0, 0.52, 1.0, 1.5, 1.94, 3.1)
SNR_EDGES = (0.0, MIN_BEST_LINE_SNR, 3.0, 6.0, np.inf)


def _breakdown(label, z_pred, z_true, best_snr):
    rows = []
    for axis, val, edges in (("z", z_true, Z_EDGES),
                             ("best_line_snr", best_snr, SNR_EDGES)):
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (val >= lo) & (val < hi)
            if m.sum() < 20:
                continue
            rows.append({"config": label, "axis": axis, "lo": lo, "hi": hi,
                         "n": int(m.sum()), **z_metrics(z_pred[m], z_true[m])})
    return rows


def plot_band_comparison(rows: list[dict], brk: list[dict],
                         out_dir: str = "outputs") -> str:
    """Render ``phot_band_comparison.png``: what Rubin adds, and what the grism does.

    Left, the outlier rate over the whole split. Middle and right, scatter and
    outlier rate against best-line S/N. The scatter panel is the one that
    separates the heads with a spectrum from the photometry-only control: the
    colours remove the outliers, the grism sets the precision, and it does so
    in proportion to how detectable the line is.
    """
    from .figures import COLOR_HR, COLOR_LR, COLOR_SR, FigureStyle, _save

    labels = [r["config"] for r in rows]
    colors = dict(zip(labels, [COLOR_LR, COLOR_SR, COLOR_HR, "0.6"], strict=False))
    sub = [b for b in brk if b["axis"] == "best_line_snr"]
    bins = sorted({(b["lo"], b["hi"]) for b in sub})
    ticks = [f">{lo:g}" if not np.isfinite(hi) else f"<{hi:g}" if lo == 0
             else f"{lo:g}–{hi:g}" for lo, hi in bins]
    w = 0.8 / len(labels)

    def grouped(ax, key, scale):
        for j, k in enumerate(labels):
            vals = {(b["lo"], b["hi"]): scale * b[key] for b in sub
                    if b["config"] == k}
            ax.bar(np.arange(len(bins)) + (j - (len(labels) - 1) / 2) * w,
                   [vals.get(bn, np.nan) for bn in bins], width=w,
                   color=colors[k], label=k, zorder=3)
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels(ticks)
        ax.set_xlabel("best-line S/N")
        ax.grid(axis="y", alpha=0.25, zorder=0)

    with FigureStyle():
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(14.5, 4.2))

        y = [100 * r["catastrophic_frac"] for r in rows]
        xs = np.arange(len(rows))
        a.bar(xs, y, width=0.6, color=[colors[k] for k in labels], zorder=3)
        for x, yy in zip(xs, y, strict=True):
            a.text(x, yy + 0.02 * max(y), f"{yy:.1f}%", ha="center",
                   va="bottom", fontsize=10)
        a.set_xticks(xs)
        a.set_xticklabels([k.replace(", ", "\n").replace("Roman ", "Roman\n")
                           for k in labels], fontsize=9)
        a.set_ylabel("catastrophic outliers [%]")
        a.set_ylim(0, max(y) * 1.15)
        a.grid(axis="y", alpha=0.25, zorder=0)
        a.set_title("Outliers, whole test split", fontsize=12)

        grouped(b, "dz_nmad", 1.0)
        b.set_yscale("log")
        b.set_ylabel(r"$\sigma_{\rm NMAD}$")
        b.set_title("Scatter by line detectability", fontsize=12)
        b.legend(frameon=False, fontsize=9, loc="upper right")

        grouped(c, "catastrophic_frac", 100.0)
        c.set_ylabel("catastrophic outliers [%]")
        c.set_title("Outliers by line detectability", fontsize=12)

        fig.tight_layout()
        return _save(fig, "phot_band_comparison.png", outdir=out_dir)


#: Panel titles of the redshift scatter figure, by configuration label.
_ZPRED_TITLES = {
    "Roman Y106/J129/H158": "Roman Y106, J129, H158",
    "+ Rubin, 1 yr": "+ Rubin $ugrizy$, 1 yr",
    "+ Rubin, 10 yr": "+ Rubin $ugrizy$, 10 yr",
    "9 bands, no spectrum": "9 bands, no spectrum",
}


def plot_band_zpred(z_true, preds: dict, out_dir: str = "outputs") -> str:
    """Render ``phot_band_zpred.png``: predicted against true redshift, one
    panel per photometric configuration, on shared axes and one colour scale.
    """
    from .figures import FigureStyle, _save

    hi = max(z_true.max(), max(p.max() for p in preds.values())) * 1.02
    with FigureStyle():
        fig, axes = plt.subplots(1, len(preds), figsize=(4.3 * len(preds) + 0.8, 4.6),
                                 sharex=True, sharey=True, squeeze=False,
                                 constrained_layout=True)
        vmax = len(z_true) / 15
        for ax, (label, zp) in zip(axes[0], preds.items(), strict=True):
            met = z_metrics(zp, z_true)
            n_out = int(round(met["catastrophic_frac"] * len(zp)))
            hb = ax.hexbin(z_true, zp, gridsize=60, bins="log", cmap="viridis",
                           mincnt=1, extent=(0, hi, 0, hi), vmin=1, vmax=vmax)
            ax.plot([0, hi], [0, hi], "-", color="#1f77b4", lw=1.2)
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_title(_ZPRED_TITLES.get(label, label), fontsize=12)
            ax.set_xlabel("True redshift")
            ax.text(0.04, 0.96,
                    f"NMAD: {met['dz_nmad']:.4f}\n"
                    f"Outliers (>0.15): {n_out} of {len(zp):,}",
                    transform=ax.transAxes, va="top", fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6",
                              alpha=0.9))
        axes[0, 0].set_ylabel("Predicted redshift")
        cb = fig.colorbar(hb, ax=axes[0], fraction=0.02, pad=0.01)
        cb.set_label("Count per hex (log)", fontsize=9)
        return _save(fig, "phot_band_zpred.png", outdir=out_dir)


def run_band_comparison(cfg: AblationConfig, x_in, phot_all, z_true, best_snr,
                        zhead, device) -> tuple[list[dict], list[dict]]:
    """Roman Medium tier against the same tier plus Rubin *ugrizy*.

    Writes ``phot_band_comparison.csv`` (whole split), ``..._breakdown.csv``
    (by redshift and by best-line S/N) and the figure. Returns both row lists.
    """
    heads = [("Roman Y106/J129/H158", zhead, cfg.roman_bands)]   # as published
    for label, ckpt in cfg.rubin_heads:
        if not os.path.exists(ckpt):
            print(f"band comparison: {ckpt} not found, skipping {label!r}")
            continue
        head = load_zhead_ckpt(ckpt, device=device)
        if head.n_phot != len(cfg.rubin_bands):
            raise SystemExit(f"{ckpt} takes {head.n_phot} bands, expected "
                             f"{len(cfg.rubin_bands)}")
        heads.append((label, head, cfg.rubin_bands))
    if len(heads) == 1:
        return [], []
    blank = set()
    for label, ckpt in cfg.photonly_heads:
        if os.path.exists(ckpt):
            heads.append((label, load_zhead_ckpt(ckpt, device=device),
                          cfg.rubin_bands))
            blank.add(label)

    gen = torch.Generator(device=device)
    rows, brk, preds = [], [], {}
    for label, head, bands in heads:
        gen.manual_seed(cfg.noise_seed)
        x_head = torch.zeros_like(x_in) if label in blank else x_in
        # Same batch size as the prediction cache: the generator is consumed
        # batch by batch, so this draws the noise the published numbers used.
        z_pred = _predict(head, x_head, phot_all[:, list(bands)], device,
                          noise_mag=cfg.eval_mag_err, gen=gen,
                          batch_size=cfg.batch_size)
        preds[label] = z_pred
        met = z_metrics(z_pred, z_true)
        rows.append({"section": "band_set", "config": label, **met,
                     "note": "+".join(PHOT_BANDS[i].split("_")[-1] for i in bands)})
        brk += _breakdown(label, z_pred, z_true, best_snr)
        print(f"{label:38s} nmad {met['dz_nmad']:.4f}  "
              f"cat {met['catastrophic_frac']:.3f}")

    for name, table in (("phot_band_comparison.csv", rows),
                        ("phot_band_comparison_breakdown.csv", brk)):
        path = os.path.join(cfg.out_dir, name)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f"wrote {path}")
    plot_band_comparison(rows, brk, out_dir=cfg.out_dir)
    # per-spectrum redshifts of every configuration, for the scatter panels
    path = os.path.join(cfg.out_dir, "phot_band_zpred.npz")
    np.savez_compressed(path, z_true=z_true, best_snr=best_snr,
                        configs=np.array(list(preds)),
                        z_pred=np.stack(list(preds.values())))
    print(f"wrote {path}")
    plot_band_zpred(z_true, preds, out_dir=cfg.out_dir)
    return rows, brk



def plot_ablation(rows: list[dict], out_dir: str = "outputs") -> str:
    """Render ``phot_ablation.png``: what the colours buy, and how fast.

    Left, the two configurations that matter --- the deployed chain and the
    same chain with its colours removed. Right, the noise sweep, which is the
    same question asked continuously: the operating point is marked, and the
    y-axis is the outlier rate because that, not the scatter, is what a
    survey pipeline pays for.
    """
    from .figures import COLOR_LR, COLOR_SR, FigureStyle, _save

    pair = [r for r in rows if r["section"] == "redshift_vs_phot"]
    sweep = [r for r in rows if r["section"] == "phot_noise"]
    sig = np.array([float(r["config"].split("=")[1].replace("mag", ""))
                    for r in sweep])
    cat = np.array([100 * r["catastrophic_frac"] for r in sweep])
    nmad = np.array([r["dz_nmad"] for r in sweep])

    with FigureStyle():
        fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.0))

        # -- left: with and without the colours ---------------------------
        labels = ["grism only\n(photometry removed)", "grism + three bands\n(Y106, J129, H158)"]
        order = [pair[1], pair[0]]
        y = [100 * r["catastrophic_frac"] for r in order]
        n = [r["dz_nmad"] for r in order]
        xs = np.arange(2)
        a.bar(xs, y, width=0.55, color=[COLOR_LR, COLOR_SR], zorder=3)
        for x, yy, nn in zip(xs, y, n, strict=True):
            a.text(x, yy + 1.0, f"{yy:.1f}%\n$\\sigma_{{\\rm NMAD}}={nn:.4f}$",
                   ha="center", va="bottom", fontsize=10)
        a.set_xticks(xs)
        a.set_xticklabels(labels)
        a.set_ylabel("catastrophic outliers [%]")
        a.set_ylim(0, max(y) * 1.42)
        a.grid(axis="y", alpha=0.25, zorder=0)
        a.set_title("With and without the three Roman bands", fontsize=12)

        # -- right: the noise sweep ---------------------------------------
        b.plot(sig, cat, "o-", color=COLOR_SR, lw=1.8, ms=5, zorder=3,
               label="outlier rate")
        b.axvline(0.05, color="0.35", ls="--", lw=1.0, zorder=2)
        # Lower-right is the only region the two curves leave clear.
        b.annotate("operating point\n0.05 mag", xy=(0.05, cat[2]),
                   xytext=(0.42, max(cat) * 0.16), fontsize=9, ha="left",
                   arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9,
                                   connectionstyle="arc3,rad=0.15"))
        b.set_xlabel("photometric noise added [mag]")
        b.set_ylabel("catastrophic outliers [%]")
        b.grid(alpha=0.25, zorder=0)
        b.set_title("Degrading the photometry", fontsize=12)

        c = b.twinx()
        c.plot(sig, nmad, "s--", color=COLOR_LR, lw=1.3, ms=4, alpha=0.85,
               label=r"$\sigma_{\rm NMAD}$")
        c.set_yscale("log")
        c.set_ylabel(r"$\sigma_{\rm NMAD}$")
        h1, l1 = b.get_legend_handles_labels()
        h2, l2 = c.get_legend_handles_labels()
        b.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=10)

        fig.tight_layout()
        return _save(fig, "phot_ablation.png", outdir=out_dir)


#: Leak-control levels, in magnitudes.
ZP_OFFSET_MAG = (0.0, 0.01, 0.02, 0.05, 0.1)
APERTURE_MAG = (0.0, 0.05, 0.1, 0.2, 0.3)


def run_leak_controls(cfg: AblationConfig, x_in, phot_all, z_true, device,
                      n_draws: int = 5) -> list[dict]:
    """Zero-point and aperture perturbations of the Rubin bands.

    Writes ``phot_leak_controls.csv``. Each row is the mean over ``n_draws``
    realisations of the perturbation; ``cat_max`` is the worst of them.
    """
    heads = [(label, ckpt, False) for label, ckpt in cfg.rubin_heads]
    heads += [(label, ckpt, True) for label, ckpt in cfg.photonly_heads]
    heads = [h for h in heads if os.path.exists(h[1])]
    if not heads:
        return []
    bands = list(cfg.rubin_bands)
    n_rubin = sum(PHOT_BANDS[i].startswith("lsst_") for i in bands)
    phot = phot_all[:, bands]
    gen = torch.Generator(device=device)
    rng = np.random.default_rng(cfg.noise_seed)
    # One set of unit draws, scaled per level, so a sweep varies only sigma.
    zp_unit = rng.standard_normal((n_draws, n_rubin)).astype(np.float32)
    ap_unit = rng.standard_normal((n_draws, len(phot))).astype(np.float32)

    rows = []
    for label, ckpt, blank in heads:
        head = load_zhead_ckpt(ckpt, device=device)
        x_head = torch.zeros_like(x_in) if blank else x_in
        for kind, levels in (("zeropoint", ZP_OFFSET_MAG),
                             ("aperture", APERTURE_MAG)):
            for mag in levels:
                mets = []
                for d in range(n_draws if mag > 0 else 1):
                    pert = phot.clone()
                    if kind == "zeropoint":
                        dm = torch.tensor(mag * zp_unit[d])[None, :]
                    else:
                        dm = torch.tensor(mag * ap_unit[d])[:, None]
                    pert[:, :n_rubin] = pert[:, :n_rubin] * 10.0 ** (-0.4 * dm)
                    gen.manual_seed(cfg.noise_seed)
                    mets.append(z_metrics(_predict(
                        head, x_head, pert, device,
                        noise_mag=cfg.eval_mag_err, gen=gen,
                        batch_size=cfg.batch_size), z_true))
                rows.append({
                    "config": label, "perturbation": kind, "mag": mag,
                    "dz_nmad": float(np.mean([m["dz_nmad"] for m in mets])),
                    "catastrophic_frac": float(np.mean(
                        [m["catastrophic_frac"] for m in mets])),
                    "cat_max": float(max(m["catastrophic_frac"] for m in mets)),
                })
                print(f"{label:22s} {kind:9s} {mag:.2f} mag  "
                      f"nmad {rows[-1]['dz_nmad']:.4f}  "
                      f"cat {rows[-1]['catastrophic_frac']:.3f}")
    path = os.path.join(cfg.out_dir, "phot_leak_controls.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
    return rows


def run_ablation(cfg: AblationConfig) -> list[dict]:
    """Run the ablation and write ``phot_ablation.csv``. Returns the rows."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    ds = RomanFixedGridDataset(cfg.data, with_phot=True)
    if ds.phot is None:
        raise SystemExit("dataset has no `phot` array")
    _, test_idx, _ = get_or_make_group_split(os.path.abspath(cfg.data), ds.ids)
    loader = DataLoader(Subset(ds, test_idx), batch_size=cfg.batch_size,
                        shuffle=False)

    sr1 = load_sr1(cfg.sr1_ckpt, device=device)
    zhead = load_zhead_ckpt(cfg.zhead_ckpt, device=device)
    if zhead.n_phot != len(cfg.roman_bands):
        raise SystemExit(
            f"the ablated head must be the Roman Medium-tier one: it takes "
            f"{zhead.n_phot} bands, expected {len(cfg.roman_bands)}")

    x_in, phot_all, z_true, best_snr = _zhead_inputs(sr1, loader, device)
    phot = phot_all[:, list(cfg.roman_bands)]       # 3 Roman Medium bands
    gen = torch.Generator(device=device).manual_seed(cfg.noise_seed)

    rows: list[dict] = []

    def record(section, label, met, note=""):
        rows.append({"section": section, "config": label, **met, "note": note})
        print(f"{label:38s} nmad {met['dz_nmad']:.4f}  "
              f"cat {met['catastrophic_frac']:.3f}  {note}")

    # 1. the deployable configuration.
    gen.manual_seed(cfg.noise_seed)
    record("redshift_vs_phot",
           f"grism + Roman-3 ({cfg.eval_mag_err:.2f} mag noise)",
           _score(zhead, x_in, phot, z_true, device,
                  noise_mag=cfg.eval_mag_err, gen=gen),
           "deployable")

    # 2. the same chain with the colours removed.
    record("redshift_vs_phot", "grism only (colours mean-imputed)",
           _score(zhead, x_in, phot, z_true, device, drop_phot=True),
           "upper bound, not an information floor")

    # 3. how fast the answer degrades as the colours get worse.
    for mag in cfg.noise_levels:
        gen.manual_seed(cfg.noise_seed)
        record("phot_noise", f"sigma={mag:.2f}mag",
               _score(zhead, x_in, phot, z_true, device,
                      noise_mag=mag, gen=gen),
               "in-dist" if mag <= 0.1 else "extrap")

    out_csv = os.path.join(cfg.out_dir, "phot_ablation.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    plot_ablation(rows, out_dir=cfg.out_dir)
    run_band_comparison(cfg, x_in, phot_all, z_true, best_snr, zhead, device)
    run_leak_controls(cfg, x_in, phot_all, z_true, device)
    return rows
