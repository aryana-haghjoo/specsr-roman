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

Every configuration here uses the three Roman Medium-tier bands that ship with
the HLWAS grism, and nothing else.
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
from ..grids import ROMAN_MEDIUM_BANDS
from ..models import pz_stats, z_metrics

__all__ = ["AblationConfig", "run_ablation", "plot_ablation"]

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


@torch.no_grad()
def _zhead_inputs(sr1, loader, device):
    """Run SR1 once and cache the 4-channel ZHead input for the whole split.

    Every configuration below shares these channels --- only the photometry
    changes --- so SR1 runs once rather than once per row. That is what keeps
    the sweep cheap enough to run on a CPU.
    """
    xs, phots, zs = [], [], []
    for batch in loader:
        x_low, z, phot = batch[0], batch[3], batch[7]
        lr = x_low.to(device, non_blocking=True)      # (B, 2, L) [flux, err]
        m, lv = sr1(lr)
        xs.append(torch.cat([lr, m, 0.5 * lv], dim=1).cpu())
        phots.append(phot)
        zs.append(z)
    return torch.cat(xs), torch.cat(phots), torch.cat(zs).numpy()


@torch.no_grad()
def _score(zhead, x_in, phot, z_true, device, *, drop_phot=False,
           noise_mag=0.0, gen=None, batch_size=256):
    """Score the head. ``drop_phot`` replaces the colours with training means."""
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
        probs = torch.softmax(zhead(xb, phot=pb), dim=-1)
        zhat, _ = pz_stats(probs, zhead.z_centers, zhead.refine_window)
        preds.append(zhat.cpu().numpy())
    return z_metrics(np.concatenate(preds), z_true)



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
        labels = ["grism only\n(colours removed)", "grism + Roman-3\n(F106/F129/F158)"]
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
        a.set_title("What three broadband colours buy", fontsize=12)

        # -- right: the noise sweep ---------------------------------------
        b.plot(sig, cat, "o-", color=COLOR_SR, lw=1.8, ms=5, zorder=3,
               label="catastrophic rate")
        b.axvline(0.05, color="0.35", ls="--", lw=1.0, zorder=2)
        # Lower-right is the only region the two curves leave clear.
        b.annotate("operating point\n0.05 mag", xy=(0.05, cat[2]),
                   xytext=(0.42, max(cat) * 0.16), fontsize=9, ha="left",
                   arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9,
                                   connectionstyle="arc3,rad=0.15"))
        b.set_xlabel("photometric noise added [mag]")
        b.set_ylabel("catastrophic outliers [%]")
        b.grid(alpha=0.25, zorder=0)
        b.set_title("Degrading the three colours", fontsize=12)

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
            f"this audit is Roman Medium-tier only: the head takes "
            f"{zhead.n_phot} bands, expected {len(cfg.roman_bands)}")

    x_in, phot_all, z_true = _zhead_inputs(sr1, loader, device)
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
    return rows
