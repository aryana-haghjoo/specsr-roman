"""The SR2 evaluation, once per redshift configuration.

SR2 places its lines at the redshifts the redshift stage hands it, so every
statement about SR2 is a statement about one redshift configuration. The
band comparison has four of them (three Roman bands
the same with Rubin at
one-year and at ten-year depth
nine bands and no spectrum), and each has its
own SR2, trained on that configuration's redshifts. The figures here draw one
panel per configuration, from one prediction cache each.

The evaluations follow the JWST companion paper: example spectra, residual
maps against a reference scatter, power spectra, line S/N, and integrated
line flux against the target. The line-flux comparison is the one the others
cannot replace: S/N never looks at the target, so only flux against the
target says whether a line is *right*.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..grids import PHOTPLAM_F158
from . import figures as F
from .linefit import ROMAN_FIT, measure_line_fluxes
from .metrics import MIN_BEST_LINE_SNR, RECOVERABILITY_BINS, detected_line_mask

__all__ = ["BandConfig", "BANDS", "load_band_caches", "measure_band_metrics",
           "make_band_figures", "plot_response_bands"]


@dataclass(frozen=True)
class BandConfig:
    """One redshift configuration and the chain trained on it."""

    key: str
    label: str
    zhead: str
    sr2: str
    phot_tier: str = "medium"
    zhead_use_spectrum: bool = True

    @property
    def cache(self) -> str:
        # the baseline's cache is the one every single-configuration figure reads
        if self.key == "roman3":
            return "outputs/pred_cache.npz"
        return f"outputs/pred_cache_{self.key}.npz"

    @property
    def response(self) -> str:
        return f"outputs/prior_dominance_{self.key}.json"


BANDS: tuple[BandConfig, ...] = (
    BandConfig("roman3", "Roman Y106, J129, H158",
               "zhead_ou2024_roman_med3_noisy", "sr2_ou2024_v5_romanonly"),
    BandConfig("rubin_y1", r"+ Rubin $ugrizy$, 1 yr",
               "zhead_ou2024_roman_med3_rubin_y1", "sr2_ou2024_v5_rubin_y1",
               phot_tier="medium_rubin"),
    BandConfig("rubin_y10", r"+ Rubin $ugrizy$, 10 yr",
               "zhead_ou2024_roman_med3_rubin_y10", "sr2_ou2024_v5_rubin_y10",
               phot_tier="medium_rubin"),
    BandConfig("photonly", "9 bands, no spectrum",
               "zhead_ou2024_photonly_rubin_y10",
               "sr2_ou2024_v5_photonly_rubin_y10",
               phot_tier="medium_rubin", zhead_use_spectrum=False),
)

# (name, rest wavelength in micron, column of the line_snr labels)
FLUX_LINES = ((r"[O III] $\lambda$5007", 0.50076, 2), (r"H$\alpha$", 0.65628, 3))
_C_AA = 2.99792458e18


def load_band_caches(bands=BANDS) -> dict:
    """``{BandConfig: cache}`` for the configurations whose cache exists."""
    out = {}
    for b in bands:
        if os.path.exists(b.cache):
            out[b] = _with_scales(np.load(b.cache, allow_pickle=True))
        else:
            print(f"no cache for {b.key} ({b.cache}); panel left out")
    if not out:
        raise SystemExit("no band caches found: run scripts/run_eval.sh")
    return out


def _with_scales(c, data: str = "data/dataset/ou2024_h10307_dataset.npz"):
    """The cache with its de-normalisation columns, added if it predates them.

    They depend only on the dataset and the split, so no model is run.
    """
    if "ab_h158" in c.files:
        return c
    from ..data import RomanFixedGridDataset, get_or_make_group_split, normalize
    ds = RomanFixedGridDataset(data, verbose=False)
    _, test_idx, _ = get_or_make_group_split(os.path.abspath(data), ds.ids,
                                             verbose=False)
    if (len(test_idx) != len(c["z_true"])
            or not np.allclose(ds.z[test_idx].numpy(), c["z_true"])):
        raise RuntimeError("cache was not built from the test split of " + data)
    hr = np.array([normalize(ds.hi_raw[i])[1:] for i in test_idx])
    lr = np.array([normalize(ds.lo_raw[i])[1:] for i in test_idx])
    out = dict(c)
    out.update(hr_mean=hr[:, 0], hr_std=hr[:, 1], lr_mean=lr[:, 0],
               lr_std=lr[:, 1],
               ab_h158=np.asarray(ds.ab_h158[test_idx], dtype=np.float64))
    return out


def _titled(axes, caches):
    for ax, b in zip(axes, caches, strict=True):
        ax.set_title(b.label, fontsize=12)


# ---------------------------------------------------------------------------
# Line recovery, S/N
# ---------------------------------------------------------------------------
def plot_recovery_bands(caches):
    fig, axes = plt.subplots(1, len(caches), figsize=(3.9 * len(caches) + 0.8, 4.4),
                             sharey=True, squeeze=False)
    xs = np.arange(len(RECOVERABILITY_BINS))
    for ax, (_, c) in zip(axes[0], caches.items(), strict=True):
        best = c["line_snr"].max(axis=1)
        det = detected_line_mask(c["wl_um"], c["z_true"], c["line_snr"])
        counts = []
        for pred, lab, col, mk, off in ((c["sr1"], "SR1", F.COLOR_LR, "o", -0.09),
                                        (c["sr2"], "SR2 (full pipeline)",
                                         F.COLOR_SR, "s", +0.09)):
            med, lo, hi, counts = [], [], [], []
            for blo, bhi in RECOVERABILITY_BINS.values():
                r = F._recovery_ratios(pred, c["hr"],
                                       np.where((best >= blo) & (best < bhi))[0], det)
                q = np.percentile(r, [16, 50, 84]) if r.size else [np.nan] * 3
                med.append(q[1])
                lo.append(q[1] - q[0])
                hi.append(q[2] - q[1])
                counts.append(r.size)
            ax.errorbar(xs + off, med, yerr=[lo, hi], fmt=mk + "-", color=col,
                        lw=1.6, ms=7, capsize=3.5, label=lab, zorder=3)
        ax.axhline(1.0, color="0.35", ls="--", lw=1.0, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{F._BIN_LABELS[k]}\n$n$ = {n:,}" for k, n in
                            zip(RECOVERABILITY_BINS, counts, strict=True)],
                           fontsize=9)
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        ax.set_ylim(0, 1.3)
    _titled(axes[0], caches)
    axes[0, 0].set_ylabel("Recovered / true line flux")
    axes[0, 0].legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.supxlabel("Line recoverability", fontsize=13)
    fig.tight_layout()
    return F._save(fig, "line_recovery_bands.png")


def plot_sn_bands(caches, min_points=20):
    first = next(iter(caches.values()))
    lines = [(n, lam, k) for n, lam, k in FLUX_LINES
             if (first["line_snr"][:, k] >= MIN_BEST_LINE_SNR).sum() >= min_points]
    fig, axes = plt.subplots(len(lines), len(caches), squeeze=False, sharex=True,
                             sharey=True,
                             figsize=(3.7 * len(caches) + 1.0, 3.6 * len(lines)))
    lo, hi = 0.5, 5000.0
    last = None
    for j, (_, c) in enumerate(caches.items()):
        wl, zt = c["wl_um"], c["z_true"]
        for i, (name, lam, k) in enumerate(lines):
            ax = axes[i, j]
            lam_obs = (1 + zt) * lam
            keep = ((lam_obs > wl[0] + 0.05) & (lam_obs < wl[-1] - 0.05)
                    & (c["line_snr"][:, k] >= MIN_BEST_LINE_SNR))
            a = np.array([F._line_snr(wl, c["lr"][r], lam_obs[r])
                          for r in np.where(keep)[0]])
            b = np.array([F._line_snr(wl, c["sr2"][r], lam_obs[r])
                          for r in np.where(keep)[0]])
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = np.clip(a[ok], lo, hi), np.clip(b[ok], lo, hi)
            last = ax.hexbin(a, b, gridsize=36, xscale="log", yscale="log",
                             extent=(np.log10(lo), np.log10(hi)) * 2,
                             bins="log", cmap="viridis", mincnt=1)
            ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.text(0.96, 0.05, f"$n$ = {a.size:,}\nf(S/N > 10)\n"
                    f"LR: {np.mean(a > 10):.2f}   SR: {np.mean(b > 10):.2f}",
                    transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6",
                              alpha=0.9))
            if j == 0:
                ax.set_ylabel(f"{name}\nS/N (SR)")
            if i == len(lines) - 1:
                ax.set_xlabel("S/N (LR input)")
    _titled(axes[0], caches)
    fig.tight_layout()
    cb = fig.colorbar(last, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label(r"$\log_{10}$(count)", fontsize=9)
    return F._save(fig, "sn_comparison_bands.png")


# ---------------------------------------------------------------------------
# Residual maps, power spectra
# ---------------------------------------------------------------------------
def plot_river_bands(caches):
    n_b = len(caches)
    fig = plt.figure(figsize=(4.0 * n_b + 1.2, 7))
    gs = GridSpec(2, n_b + 1, figure=fig, height_ratios=[4, 1],
                  width_ratios=[1] * n_b + [0.045], hspace=0.06, wspace=0.07)
    res = {}
    for b, c in caches.items():
        order = np.argsort(c["z_true"])
        res[b] = (order, (c["sr2"] - c["hr"])[order], (c["lr"] - c["hr"])[order])
    img = {b: F._highpass(r[1], sigma_px=40) for b, r in res.items()}
    vmax = np.percentile(np.abs(np.concatenate([v.ravel()[::7] for v in img.values()])), 97)
    top = []
    for j, (b, c) in enumerate(caches.items()):
        wl = c["wl_um"]
        order, r_sr, r_lr = res[b]
        zt = c["z_true"][order]
        n = len(zt)
        ax = fig.add_subplot(gs[0, j])
        top.append(ax)
        im = ax.imshow(img[b], aspect="auto", origin="lower",
                       extent=[wl[0], wl[-1], 0, n], cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ztk = [z for z in (0, 0.5, 1, 1.5, 2, 2.5, 3) if zt[0] <= z <= zt[-1]]
        ax.set_yticks([min(int(np.searchsorted(zt, z)), n - 1) for z in ztk])
        ax.set_yticklabels([str(z) for z in ztk] if j == 0 else [])
        plt.setp(ax.get_xticklabels(), visible=False)
        wt = np.linspace(wl[0], wl[-1], 300)
        for name, lam in F.LINES.items():
            zt_track = wt / lam - 1
            idx = np.interp(zt_track, zt, np.arange(n), left=-1, right=n + 1)
            m = (zt_track >= zt[0]) & (zt_track <= zt[-1])
            if m.sum() < 2:
                continue
            ax.plot(wt[m], idx[m], "k--", lw=0.6, alpha=0.45)
            if j == 0:
                p = int(0.5 * (m.sum() - 1))
                ax.text(wt[m][p], idx[m][p], name, fontsize=8, ha="center",
                        va="bottom", alpha=0.9)
        low = fig.add_subplot(gs[1, j], sharex=ax)
        low.plot(wl, F._robust_sigma_cols(r_sr), lw=1.6, color="#1f77b4",
                 label=r"$\sigma_{\rm SR-HR}(\lambda)$")
        low.plot(wl, F._robust_sigma_cols(r_lr), lw=1.3, ls="--", color="0.4",
                 label=r"$\sigma_{\rm LR-HR}(\lambda)$")
        low.set_xlim(wl[0], wl[-1])
        low.set_ylim(0, None)
        low.set_xlabel(r"Wavelength [$\mu$m]")
        if j == 0:
            low.set_ylabel(r"$\sigma$")
            low.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
        else:
            plt.setp(low.get_yticklabels(), visible=False)
    ymax = max(a.get_ylim()[1] for a in fig.axes[1::2][:n_b])
    for a in fig.axes[1::2][:n_b]:
        a.set_ylim(0, ymax)
    _titled(top, caches)
    top[0].set_ylabel(r"Redshift $z$")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, n_b]))
    cb.set_label(r"SR $-$ HR target (norm. flux)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return F._save(fig, "residual_river_bands.png")


def plot_psd_bands(caches):
    fig, axes = plt.subplots(2, len(caches), squeeze=False, sharex=True,
                             sharey="row", figsize=(4.0 * len(caches) + 0.8, 8))
    for j, (_, c) in enumerate(caches.items()):
        wl, lr, s2, hr = c["wl_um"], c["lr"], c["sr2"], c["hr"]
        k = np.fft.rfftfreq(len(wl), d=(wl[-1] - wl[0]) / (len(wl) - 1))
        for X, lab, col in ((lr, "LR", "#1f77b4"), (s2, "SR", "#f5a623"),
                            (hr, "HR", "#2ca02c")):
            mu, lo, up = F._psd(X)
            axes[0, j].plot(k[1:], mu[1:], color=col, lw=1.5, label=lab)
            axes[0, j].fill_between(k[1:], lo[1:], up[1:], color=col, alpha=0.18)
        for A, B, lab, col in ((lr, hr, "LR $-$ HR", "#1f77b4"),
                               (lr, s2, "LR $-$ SR", "#f5a623"),
                               (s2, hr, "SR $-$ HR", "#2ca02c")):
            mu, lo, up = F._psd(A - B, hp=False)
            axes[1, j].plot(k[1:], mu[1:], color=col, lw=1.5, label=lab)
            axes[1, j].fill_between(k[1:], lo[1:], up[1:], color=col, alpha=0.18)
        kcut = 1.0 / (2 * (wl[-1] - wl[0]) / 864)
        for ax in axes[:, j]:
            ax.axvline(kcut, color="0.5", ls="--", lw=1)
            ax.set_yscale("log")
            ax.set_xlim(0, k[-1])
        axes[1, j].set_xlabel(r"Frequency $k$ (cycles / $\mu$m)")
    _titled(axes[0], caches)
    axes[0, 0].set_ylabel("High-pass signal PSD (arb.)")
    axes[1, 0].set_ylabel("Residual PSD (arb.)")
    axes[0, -1].legend(fontsize=9.5, frameon=False)
    axes[1, -1].legend(fontsize=9.5, frameon=False)
    fig.tight_layout()
    return F._save(fig, "psd_bands.png")


# ---------------------------------------------------------------------------
# Example spectra
# ---------------------------------------------------------------------------
def plot_spectra_bands(caches):
    """One galaxy, one panel per configuration.

    The galaxy is the best-ranked example of the first configuration. It is
    kept fixed, so the panels differ only in the redshift SR2 was given.
    """
    first = next(iter(caches.values()))
    order = F._rank_line_examples(first)
    if not order:
        raise SystemExit("no examples pass the line-fidelity cuts")
    i = order[0]
    fig, axes = plt.subplots(len(caches), 1, figsize=(13, 3.3 * len(caches)),
                             squeeze=False, sharex=True, constrained_layout=True)
    for ax, (b, c) in zip(axes[:, 0], caches.items(), strict=True):
        F._draw_example(ax, c, i, set(), sr_label="SR (ours)")
        ax.set_title(b.label, loc="left", fontsize=11)
    axes[-1, 0].set_xlabel(r"Observed wavelength [$\mu$m]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=3, frameon=False,
               fontsize=11)
    return F._save(fig, "spectra_examples_bands.png")


# ---------------------------------------------------------------------------
# Integrated line flux against the target
# ---------------------------------------------------------------------------
def _physical_scales(c):
    """Per-row factors from normalised flux to erg/s/cm2/A, for LR and SR/HR.

    Each spectrum is put on the scale of its own galaxy's H158 magnitude: its
    mean over the H158 band is set to the f_lambda that magnitude implies. The
    output shares the target's normalisation, so the two share a factor.
    """
    wl = c["wl_um"] * 1e4
    band = (wl > 13800.0) & (wl < 17700.0)
    f_h = 10.0 ** (-0.4 * (c["ab_h158"] + 48.6)) * _C_AA / PHOTPLAM_F158 ** 2
    hr_raw = c["hr"][:, band] * c["hr_std"][:, None] + c["hr_mean"][:, None]
    lr_raw = c["lr"][:, band] * c["lr_std"][:, None] + c["lr_mean"][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        k_hr = f_h / hr_raw.mean(axis=1)
        k_lr = f_h / lr_raw.mean(axis=1)
    return c["lr_std"] * k_lr, c["hr_std"] * k_hr


def line_fluxes(c):
    """Gaussian-fit fluxes [erg/s/cm2] of the detected lines, LR / SR / HR.

    A line enters where it is detected in the grism data
    (:data:`~specsr_roman.evaluation.metrics.MIN_BEST_LINE_SNR`), as everywhere
    else. The fit is the JWST companion paper's: a Gaussian on a linear
    continuum, with the flux ``sqrt(2 pi) A sigma``.
    """
    s_lr, s_hr = _physical_scales(c)
    out = {}
    for name, lam, k in FLUX_LINES:
        rows = np.where(c["line_snr"][:, k] >= MIN_BEST_LINE_SNR)[0]
        if not rows.size:
            continue
        m = measure_line_fluxes(
            c["wl_um"], c["z_true"][rows],
            {"LR": c["lr"][rows], "SR": c["sr2"][rows], "HR": c["hr"][rows]},
            {"LR": s_lr[rows], "SR": s_hr[rows], "HR": s_hr[rows]},
            [lam], ["line"], **ROMAN_FIT)
        # the fit integrates over micron; fluxes are per Angstrom
        out[name] = {kind: m[f"line_flux_{kind}"] * 1e4 for kind in ("LR", "SR", "HR")}
        out[name].update(rows=rows,
                         sigma={kind: m[f"line_sigma_{kind}"] for kind in ("LR", "SR", "HR")})
    return out


def flux_stats(pred, ref, outlier_factor=2.0):
    """Numbers of one flux panel: how a product's line flux compares with truth."""
    ok = np.isfinite(pred) & np.isfinite(ref) & (ref > 0)
    pred, ref = pred[ok], ref[ok]
    ratio = pred / ref
    pos = pred > 0
    out = (~pos) | (ratio > outlier_factor) | (ratio < 1.0 / outlier_factor)
    return {"n": int(ok.sum()), "median_ratio": float(np.median(ratio)),
            "sum_ratio": float(pred.sum() / ref.sum()),
            "outlier_frac": float(out.mean()), "nonpositive_frac": float((~pos).mean())}


def plot_line_flux(caches, fluxes=None):
    """Integrated line flux against the target: the input, then each SR2."""
    fluxes = fluxes or {b: line_fluxes(c) for b, c in caches.items()}
    first = next(iter(fluxes.values()))
    names = [n for n, _, _ in FLUX_LINES if n in first]
    cols = [("Grism input", None)] + [(b.label, b) for b in caches]
    fig, axes = plt.subplots(len(names), len(cols), squeeze=False,
                             figsize=(3.5 * len(cols) + 0.9, 3.7 * len(names)))
    last = None
    for i, name in enumerate(names):
        ref = first[name]["HR"]
        good = np.isfinite(ref) & (ref > 0)
        lo = 10 ** np.floor(np.log10(np.percentile(ref[good], 1)) - 0.5)
        hi = 10 ** np.ceil(np.log10(np.percentile(ref[good], 99.5)) + 0.3)
        for j, (label, b) in enumerate(cols):
            ax = axes[i, j]
            f = first[name] if b is None else fluxes[b][name]
            pred, ref = (f["LR"] if b is None else f["SR"]), f["HR"]
            st = flux_stats(pred, ref)
            ok = np.isfinite(pred) & np.isfinite(ref) & (ref > 0) & (pred > 0)
            last = ax.hexbin(ref[ok], np.clip(pred[ok], lo, hi), gridsize=30,
                             xscale="log", yscale="log", bins="log", mincnt=1,
                             cmap="viridis",
                             extent=(np.log10(lo), np.log10(hi)) * 2)
            ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.text(0.04, 0.96,
                    f"$n$ = {st['n']:,}\nmedian = {st['median_ratio']:.2f}\n"
                    rf"$\Sigma/\Sigma_{{\rm HR}}$ = {st['sum_ratio']:.2f}" "\n"
                    f"outliers: {100 * st['outlier_frac']:.0f}%\n"
                    f"non-positive: {100 * st['nonpositive_frac']:.0f}%",
                    transform=ax.transAxes, va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6",
                              alpha=0.9))
            if i == 0:
                ax.set_title(label, fontsize=11.5)
            if j == 0:
                ax.set_ylabel(f"{name}\nline flux (erg s$^{{-1}}$ cm$^{{-2}}$)")
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if i == len(names) - 1:
                ax.set_xlabel("True line flux (erg s$^{-1}$ cm$^{-2}$)")
    fig.tight_layout()
    cb = fig.colorbar(last, ax=axes, fraction=0.016, pad=0.03)
    cb.set_label(r"$\log_{10}$(count)", fontsize=9)
    return F._save(fig, "line_flux_bands.png")


# ---------------------------------------------------------------------------
# Response to the input
# ---------------------------------------------------------------------------
def plot_response_bands(bands=BANDS, outdir: str | None = None):
    """Response exponent per recoverability bin, one panel per configuration."""
    found = [(b, json.load(open(b.response))) for b in bands
             if os.path.exists(b.response)]
    if not found:
        raise SystemExit("no prior_dominance_<key>.json found")
    with F.FigureStyle():
        fig, axes = plt.subplots(1, len(found), sharey=True, squeeze=False,
                                 figsize=(3.9 * len(found) + 0.8, 4.4))
        for ax, (b, s) in zip(axes[0], found, strict=True):
            labels = None
            for key, lab, col, mk, off in (("sr1", "SR1", F.COLOR_LR, "o", -0.09),
                                           ("sr2", "SR2 (full pipeline)",
                                            F.COLOR_SR, "s", +0.09)):
                rows = s[key]["by_snr"]
                xs = np.arange(len(rows))
                med = np.array([r["median"] for r in rows])
                ax.errorbar(xs + off, med,
                            yerr=[med - [r["p25"] for r in rows],
                                  [r["p75"] for r in rows] - med],
                            color=col, marker=mk, ms=7, lw=1.6, capsize=3.5,
                            label=lab)
                labels = [f"{r['bin']}\n$n$ = {r['n']:,}" for r in rows]
            ax.axhline(1.0, color="0.3", lw=1.0, ls="--")
            ax.axhline(0.0, color="0.3", lw=1.0, ls=":")
            ax.set_xticks(np.arange(len(labels)), labels, fontsize=9)
            ax.set_xlim(-0.5, len(labels) - 0.5)
            ax.set_ylim(-0.15, 1.25)
            ax.set_title(b.label, fontsize=12)
        axes[0, 0].set_ylabel(r"Response exponent $r$")
        axes[0, 0].legend(frameon=False, fontsize=9.5, loc="upper left",
                          bbox_to_anchor=(0.0, 0.80))
        axes[0, -1].text(0.98, 1.03, "follows the input", fontsize=9, color="0.3",
                         ha="right", transform=axes[0, -1].get_yaxis_transform())
        axes[0, -1].text(0.98, -0.10, "ignores the input", fontsize=9, color="0.3",
                         ha="right", transform=axes[0, -1].get_yaxis_transform())
        fig.supxlabel("Line recoverability", fontsize=13)
        fig.tight_layout()
        return F._save(fig, "response_bands.png", outdir=outdir)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def measure_band_metrics(caches, fluxes=None) -> dict:
    """Per-configuration numbers behind the figures, for the paper's macros."""
    from .metrics import line_amplitude_recovery, redshift_summary

    fluxes = fluxes or {b: line_fluxes(c) for b, c in caches.items()}
    out = {}
    for b, c in caches.items():
        rec = line_amplitude_recovery(c["sr2"], c["hr"], c["line_snr"],
                                      z=c["z_true"], wave_um=c["wl_um"])
        m = {"label": b.label,
             "redshift": redshift_summary(c["z_pred"], c["z_true"]),
             "recovery": {k: v["median"] for k, v in rec.items()},
             "flux": {}, "flux_input": {}, "fwhm_aa": {}}
        dz = np.abs(c["z_pred"] - c["z_true"]) / (1 + c["z_true"])
        for name, f in fluxes[b].items():
            m["flux"][name] = flux_stats(f["SR"], f["HR"])
            m["flux_input"][name] = flux_stats(f["LR"], f["HR"])
            # the same comparison where the redshift landed, and where it did not
            near = dz[f["rows"]] < 0.002
            m["flux"][name]["median_ratio_z_ok"] = flux_stats(
                f["SR"][near], f["HR"][near])["median_ratio"]
            m["flux"][name]["frac_z_ok"] = float(near.mean())
            if (~near).sum() >= 10:
                m["flux"][name]["median_ratio_z_off"] = flux_stats(
                    f["SR"][~near], f["HR"][~near])["median_ratio"]
            m["fwhm_aa"][name] = {kind: float(2.3548e4 * np.nanmedian(f["sigma"][kind]))
                                  for kind in ("LR", "SR", "HR")}
        rms_lr = np.sqrt(np.mean((c["lr"] - c["hr"]) ** 2, axis=1))
        rms_sr = np.sqrt(np.mean((c["sr2"] - c["hr"]) ** 2, axis=1))
        m["rms_gain"] = float(np.median(rms_lr / rms_sr))
        out[b.key] = m
    return out


def make_band_figures(bands=BANDS, outdir: str = "outputs/figures",
                      metrics_json: str = "outputs/band_metrics.json") -> dict:
    """Render every four-panel figure and write the numbers behind them."""
    caches = load_band_caches(bands)
    previous, F.OUTDIR = F.OUTDIR, outdir
    try:
        with F.FigureStyle():
            fluxes = {b: line_fluxes(c) for b, c in caches.items()}
            plot_recovery_bands(caches)
            plot_line_flux(caches, fluxes)
            plot_sn_bands(caches)
            plot_psd_bands(caches)
            plot_river_bands(caches)
            plot_spectra_bands(caches)
        metrics = measure_band_metrics(caches, fluxes)
    finally:
        F.OUTDIR = previous
    with open(metrics_json, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"wrote {metrics_json}")
    return metrics
