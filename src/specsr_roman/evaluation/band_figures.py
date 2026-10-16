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

from ..grids import (
                     AB_ZEROPOINT,
                     BAND_PIVOT,
                     PHOTPLAM_F158,
                     ROMAN_MEDIUM_BANDS,
                     RUBIN_BANDS,
                     phot_flux_sigma,
)
from . import figures as F
from .dataset_figures import _stretch, build_cutout_cache
from .linefit import ROMAN_FIT, measure_line_fluxes
from .metrics import MIN_BEST_LINE_SNR, RECOVERABILITY_BINS, detected_line_mask

__all__ = ["BandConfig", "BANDS", "load_band_caches", "measure_band_metrics",
           "nii_halpha_ratio",
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
               "zhead_ou2024_roman_med3_noisy_independent", "sr2_ou2024_v5_romanonly_independent"),
    BandConfig("rubin_y1", r"+ Rubin $ugrizy$, 1 yr",
               "zhead_ou2024_roman_med3_rubin_y1_independent", "sr2_ou2024_v5_rubin_y1_independent",
               phot_tier="medium_rubin"),
    BandConfig("rubin_y10", r"+ Rubin $ugrizy$, 10 yr",
               "zhead_ou2024_roman_med3_rubin_y10_independent",
               "sr2_ou2024_v5_rubin_y10_independent",
               phot_tier="medium_rubin"),
    BandConfig("photonly", "9 bands, no spectrum",
               "zhead_ou2024_photonly_rubin_y10_independent",
               "sr2_ou2024_v5_photonly_rubin_y10_independent",
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
        # Read every array once. An NpzFile decompresses a whole array on each
        # access, so indexing it row by row inside a loop takes hours.
        return {k: c[k] for k in c.files}
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
        med, lo, hi, counts = [], [], [], []
        for blo, bhi in RECOVERABILITY_BINS.values():
            r = F._recovery_ratios(c["sr2"], c["hr"],
                                   np.where((best >= blo) & (best < bhi))[0], det)
            q = np.percentile(r, [16, 50, 84]) if r.size else [np.nan] * 3
            med.append(q[1])
            lo.append(q[1] - q[0])
            hi.append(q[2] - q[1])
            counts.append(r.size)
        ax.errorbar(xs, med, yerr=[lo, hi], fmt="s-", color=F.COLOR_SR,
                    lw=1.6, ms=7, capsize=3.5, zorder=3)
        ax.axhline(1.0, color="0.35", ls="--", lw=1.0, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{F._BIN_LABELS[k]}\n$n$ = {n:,}" for k, n in
                            zip(RECOVERABILITY_BINS, counts, strict=True)],
                           fontsize=9)
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        ax.set_ylim(0, 1.3)
    _titled(axes[0], caches)
    axes[0, 0].set_ylabel("Recovered / true line flux")
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
    """Residual maps against the target: the input, then each SR2 output.

    The input row is the same spectra in every column (the grism spectrum does
    not depend on the redshift configuration); it is drawn in each so that
    every output map sits under the map it started from. Both rows share one
    colour scale, set by the input row: the figure compares the two, and on
    separate scales the output's line tracks look stronger than the input's.
    """
    n_b = len(caches)
    fig = plt.figure(figsize=(4.0 * n_b + 1.2, 11.5))
    gs = GridSpec(3, n_b + 1, figure=fig, height_ratios=[4, 4, 1.1],
                  width_ratios=[1] * n_b + [0.045], hspace=0.07, wspace=0.07)
    res = {}
    for b, c in caches.items():
        order = np.argsort(c["z_true"])
        res[b] = (order, (c["sr2"] - c["hr"])[order], (c["lr"] - c["hr"])[order])
    img_sr = {b: F._highpass(r[1], sigma_px=40) for b, r in res.items()}
    img_lr = {b: F._highpass(r[2], sigma_px=40) for b, r in res.items()}
    vmax = np.percentile(np.abs(next(iter(img_lr.values())).ravel()[::7]), 97)
    top, lows = [], []
    for j, (b, c) in enumerate(caches.items()):
        wl = c["wl_um"]
        order, r_sr, r_lr = res[b]
        zt = c["z_true"][order]
        n = len(zt)
        ztk = [z for z in (0, 0.5, 1, 1.5, 2, 2.5, 3) if zt[0] <= z <= zt[-1]]
        wt = np.linspace(wl[0], wl[-1], 300)
        for row, img in enumerate((img_lr[b], img_sr[b])):
            ax = fig.add_subplot(gs[row, j])
            im = ax.imshow(img, aspect="auto", origin="lower",
                           extent=[wl[0], wl[-1], 0, n], cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_yticks([min(int(np.searchsorted(zt, z)), n - 1) for z in ztk])
            ax.set_yticklabels([str(z) for z in ztk] if j == 0 else [])
            plt.setp(ax.get_xticklabels(), visible=False)
            for name, lam in F.LINES.items():
                zt_track = wt / lam - 1
                idx = np.interp(zt_track, zt, np.arange(n), left=-1, right=n + 1)
                m = (zt_track >= zt[0]) & (zt_track <= zt[-1])
                if m.sum() < 2:
                    continue
                ax.plot(wt[m], idx[m], "k--", lw=0.6, alpha=0.45)
                if j == 0 and row == 0:
                    p = int(0.5 * (m.sum() - 1))
                    ax.text(wt[m][p], idx[m][p], name, fontsize=8, ha="center",
                            va="bottom", alpha=0.9)
            if j == 0:
                ax.set_ylabel(("LR grism $-$ HR target" if row == 0
                               else "SR $-$ HR target") + "\nRedshift $z$")
            if row == 0:
                top.append(ax)
        low = fig.add_subplot(gs[2, j])
        lows.append(low)
        low.plot(wl, F._robust_sigma_cols(r_sr), lw=1.6, color="#1f77b4",
                 label=r"$\sigma_{\rm SR-HR}(\lambda)$")
        low.plot(wl, F._robust_sigma_cols(r_lr), lw=1.3, ls="--", color="0.4",
                 label=r"$\sigma_{\rm LR-HR}(\lambda)$")
        low.set_xlim(wl[0], wl[-1])
        low.set_xlabel(r"Wavelength [$\mu$m]")
        if j == 0:
            low.set_ylabel(r"$\sigma$")
            low.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
        else:
            plt.setp(low.get_yticklabels(), visible=False)
    ymax = max(a.get_ylim()[1] for a in lows)
    for a in lows:
        a.set_ylim(0, ymax)
    _titled(top, caches)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:2, n_b]))
    cb.set_label("Residual (norm. flux)", fontsize=9)
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
#: The spectra of the example figure: hard cases that the pipeline gets right,
#: chosen by hand. A galaxy is observed in several exposures,
#: so a spectrum is named by object id, visit and detector. Each entry then
#: gives the line complex its zooms frame (a key of ``figures.LINE_GROUPS``),
#: the ``line_snr`` column and name of the line its title quotes, and the case.
HARD_CASES = (
    (10307100430361, 16189, 15, "halpha_nii", 3, r"H$\alpha$",
     "A marginal detection"),
    (10307101343410, 16188, 17, "o3hb", 2, r"[O III] $\lambda5007$",
     "A faint doublet"),
    (10307000644988, 16189, 7, "halpha_nii", 3, r"H$\alpha$",
     "A single line and a wrong redshift"),
    (10307101714760, 16188, 8, "halpha_nii", 3, r"H$\alpha$",
     "A marginal line at the red end of the band"),
)
HARD_CASES_CACHE = "outputs/hard_cases_cache.npz"
_DATASET = "data/dataset/ou2024_h10307_dataset.npz"


def _test_rows(cases, data: str = _DATASET) -> list[int]:
    """Row of the prediction cache for each ``(object id, visit, sca, ...)``."""
    from ..data import get_or_make_group_split
    d = np.load(data, allow_pickle=True)
    _, test_idx, _ = get_or_make_group_split(os.path.abspath(data), d["ids"],
                                             verbose=False)
    ids, visit, sca = (d[k][test_idx] for k in ("ids", "visit", "sca"))
    rows = []
    for oid, v, s, *_ in cases:
        (row,) = np.where((ids == oid) & (visit == v) & (sca == s))[0]
        rows.append(int(row))
    return rows


def _draw_stamp(ax, cut, p, half=30):
    x0, _, y0, _ = cut[p + "region_box"]
    xc, yc = cut[p + "xy"]
    img = cut[p + "region_direct"]
    ext = (x0 - xc, x0 - xc + img.shape[1], y0 - yc, y0 - yc + img.shape[0])
    ax.imshow(img, extent=ext, aspect="equal", **_stretch(img, 20, 99.9))
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xticks([])
    ax.set_yticks([])
    # 1 arcsec at the 0.11 arcsec pixels of the WFI
    bar = 1.0 / 0.11
    ax.plot([-half + 4, -half + 4 + bar], [-half + 5] * 2, color="k", lw=2)
    ax.text(-half + 4 + bar / 2, -half + 7, '1"', ha="center", va="bottom",
            fontsize=9)
    ax.text(0.04, 0.96, "H158", transform=ax.transAxes, va="top", fontsize=10)


def _draw_photometry(ax, cut, p, wl):
    """The nine bands of the redshift stage, as AB magnitudes."""
    names = [str(n) for n in cut["phot_bands"]]
    sig = phot_flux_sigma(names)
    for idx, col, mk, lab in ((RUBIN_BANDS, F.COLOR_LR, "o", "Rubin"),
                              (ROMAN_MEDIUM_BANDS, F.COLOR_HR, "s", "Roman")):
        lam = np.array([BAND_PIVOT[names[i]] for i in idx]) * 1e-4
        f = np.array([cut[p + "phot"][i] for i in idx], dtype=float)
        zp = np.array([AB_ZEROPOINT[names[i]] for i in idx])
        # the errors the redshift stage is given: 0.05 mag on every band, and
        # the depth of the ten-year coadd on the Rubin ones
        err = np.hypot(0.05, 1.0857 * sig[list(idx)] / f)
        ax.errorbar(lam, zp - 2.5 * np.log10(f), yerr=err, fmt=mk, color=col,
                    ms=5, lw=1.1, capsize=2, label=lab)
    ax.axvspan(wl[0], wl[-1], color="0.90", lw=0, zorder=0)
    ax.text(0.5 * (wl[0] + wl[-1]), 0.05, "grism", ha="center", fontsize=8.5,
            color="0.35", transform=ax.get_xaxis_transform())
    ax.set_xlim(0.25, 2.0)
    ax.margins(y=0.3)
    ax.invert_yaxis()
    ax.set_xlabel(r"Wavelength [$\mu$m]", fontsize=10)
    ax.set_ylabel("AB magnitude", fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.2,
              borderpad=0.1)


def _draw_two_d(ax, cut, p, wl, half_h=9):
    lam = cut[p + "beam_lam"].astype(np.float64) * 1e-4
    in_band = np.where((lam > wl[0]) & (lam < wl[-1]))[0]
    own = cut[p + "beam_own"]
    trace = int(np.argmax(own[:, in_band].sum(axis=1)))
    r0, r1 = max(trace - half_h, 0), min(trace + half_h, own.shape[0])
    order = np.argsort(lam)
    two_d = cut[p + "beam_clean"][r0:r1][:, order]
    ax.imshow(two_d, extent=(lam[order][0], lam[order][-1], r0 - trace, r1 - trace),
              aspect="auto", **_stretch(two_d, 5, 99.8))
    ax.set_xlim(wl[0], wl[-1])
    ax.set_yticks([])
    ax.set_ylabel("2D", fontsize=10)
    plt.setp(ax.get_xticklabels(), visible=False)


def _draw_hard_case(fig, cell, caches, cut, case, row, letter):
    oid, _, _, group, snr_col, line_name, what = case
    p = f"{oid}_"
    first = next(iter(caches.values()))
    wl, hr, lr = first["wl_um"], first["hr"][row], first["lr"][row]
    z = float(first["z_true"][row])
    snr = float(first["line_snr"][row, snr_col])

    card = cell.subgridspec(1, 2, width_ratios=[1, 4.7], wspace=0.11)
    left = card[0].subgridspec(2, 1, height_ratios=[1.25, 1], hspace=0.12)
    right = card[1].subgridspec(2, 1, height_ratios=[1.75, 1.25], hspace=0.66)
    top = right[0].subgridspec(2, 1, height_ratios=[0.3, 1], hspace=0.04)
    zooms = right[1].subgridspec(1, len(caches), wspace=0.1)

    _draw_stamp(fig.add_subplot(left[0]), cut, p)
    _draw_photometry(fig.add_subplot(left[1]), cut, p, wl)

    a2 = fig.add_subplot(top[0])
    _draw_two_d(a2, cut, p, wl)
    a2.set_title(rf"$\bf{{({letter})}}$ {what}: {line_name} at "
                 rf"$\mathrm{{S/N}}={snr:.1f}$", loc="left", fontsize=12.5)
    a2.set_title(rf"$z={z:.3f}$,  $m_{{\rm H158}}={float(cut[p + 'ab']):.1f}$",
                 loc="right", fontsize=11)

    # -- the whole band, with the output of the first configuration --
    ax = fig.add_subplot(top[1], sharex=a2)
    sr = first["sr2"][row]
    ax.plot(wl, hr, color=F.COLOR_HR, lw=1.0, alpha=0.8, label="HR target",
            zorder=1)
    ax.plot(wl, lr, color=F.COLOR_LR, lw=1.0, ls="--", alpha=0.85,
            label="LR grism", zorder=2)
    ax.plot(wl, sr, color=F.COLOR_SR, lw=1.5, alpha=0.95, label="SR (ours)",
            zorder=3)
    ax.set_ylabel(r"$F_\lambda$ (norm.)", fontsize=10.5)
    ax.tick_params(labelsize=9)
    y_lo, y_hi = F._robust_ylims(np.concatenate([hr, lr, sr]))
    y_hi = max(y_hi, 1.05 * max(hr.max(), sr.max()))
    rep = F._line_report(wl, hr, sr, z, first["line_snr"][row])
    marks = [(name, rest * (1 + z)) for name, rest in F.MAIN_LABELS
             if any(abs(r["lam_rest"] - rest) < 0.003 for r in rep)]
    F._annotate_lines(ax, marks, y_lo, y_hi, head=0.16, fontsize=9)
    x0, x1 = (v * (1 + z) for v in F.LINE_GROUPS[group]["rest"])
    ax.axvspan(x0, x1, color=F.COLOR_SR, alpha=0.16, lw=0, zorder=0)

    # -- the line complex, one panel per configuration --
    m = (wl >= x0) & (wl <= x1)
    stack = np.concatenate([hr[m], lr[m]] + [c["sr2"][row][m]
                                             for c in caches.values()])
    zy_lo = F._robust_ylims(stack, p_lo=1, p_hi=99, pad_frac=0.10)[0]
    zy_hi = float(stack.max()) + 0.06 * float(np.ptp(stack))
    zmarks = [(r["name"], r["lam_obs"]) for r in rep if x0 <= r["lam_obs"] <= x1]
    # the redshift goes on the side of the panel that the labelled lines leave free
    z_left = bool(zmarks) and np.mean([lam for _, lam in zmarks]) > 0.5 * (x0 + x1)
    for q, (b, c) in enumerate(caches.items()):
        az = fig.add_subplot(zooms[q])
        az.plot(wl[m], hr[m], color=F.COLOR_HR, lw=1.2, alpha=0.85, zorder=1)
        az.plot(wl[m], lr[m], color=F.COLOR_LR, lw=1.1, ls="--", alpha=0.85,
                zorder=2)
        az.plot(wl[m], c["sr2"][row][m], color=F.COLOR_SR, lw=1.8, zorder=3)
        az.set_xlim(x0, x1)
        az.xaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
        az.tick_params(labelsize=8.5)
        if q:
            plt.setp(az.get_yticklabels(), visible=False)
        else:
            az.set_ylabel(r"$F_\lambda$ (norm.)", fontsize=10.5)
        zp = float(c["z_pred"][row])
        bad = abs(zp - z) / (1 + z) > 0.15
        az.set_title(b.label, fontsize=10)
        # the redshift this configuration handed to SR2, red when it is wrong
        az.text(0.03 if z_left else 0.97, 0.70, rf"$\hat z={zp:.3f}$",
                transform=az.transAxes,
                ha="left" if z_left else "right", va="top", fontsize=10, zorder=8,
                color="#c1272d" if bad else "0.15",
                bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.85))
        F._annotate_lines(az, zmarks, zy_lo, zy_hi, head=0.17, fontsize=8)
    return ax


def plot_spectra_bands(caches, cutouts: str = HARD_CASES_CACHE):
    """Hard cases, one block per galaxy.

    Each block shows what the pipeline is given (the H158 image, the nine
    bands of photometry, the two-dimensional spectrum and its extraction) and
    what it returns: the output over the whole band for the first
    configuration, then the line complex once per configuration. The
    two-dimensional arrays are not in the dataset, so the detector images of
    these galaxies are dispersed again the first time the figure is made.
    """
    rows = _test_rows(HARD_CASES)
    if not os.path.exists(cutouts):
        build_cutout_cache({c[0]: c[1:3] for c in HARD_CASES}, cutouts)
    cut = np.load(cutouts, allow_pickle=True)
    n = len(HARD_CASES)
    fig = plt.figure(figsize=(11, 2.9 * n))
    outer = GridSpec(n, 1, figure=fig, hspace=0.36, left=0.055, right=0.99,
                     top=1 - 0.25 / n, bottom=0.2 / n)
    for k, case in enumerate(HARD_CASES):
        ax = _draw_hard_case(fig, outer[k], caches, cut, case, rows[k],
                             "abcdefgh"[k])
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.6, 1.0), borderaxespad=0.2)
    fig.supxlabel(r"Observed wavelength [$\mu$m]", fontsize=12, x=0.6)
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
def plot_response_bands(bands=BANDS, outdir: str | None = None,
                        response_dir: str | None = None):
    """Response exponent per recoverability bin, the configurations side by side.

    One panel: the configurations give the same exponents, and a panel for
    each would be three points in an empty frame, four times.
    """
    paths = [(b, os.path.join(response_dir, f"response_{b.key}.json")
              if response_dir else b.response) for b in bands]
    found = [(b, json.load(open(path))) for b, path in paths if os.path.exists(path)]
    methods = {(r.get("method_version", 1), r.get("config", {}).get("output_scale"))
               for _, r in found}
    if len(methods) > 1:
        raise ValueError("cannot mix response methods in one figure")
    if not found:
        raise SystemExit("no prior_dominance_<key>.json found")
    colors = [F.COLOR_LR, F.COLOR_SR, F.COLOR_HR, "0.5"]
    markers = ["o", "s", "D", "^"]
    step = 0.17
    with F.FigureStyle():
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for q, (b, s) in enumerate(found):
            rows = s["sr2"]["by_snr"]
            xs = np.arange(len(rows)) + (q - (len(found) - 1) / 2) * step
            med = np.array([r["median"] for r in rows])
            ax.errorbar(xs, med,
                        yerr=[med - [r["p25"] for r in rows],
                              [r["p75"] for r in rows] - med],
                        color=colors[q % len(colors)],
                        marker=markers[q % len(markers)], ms=6.5, lw=1.4,
                        capsize=3, label=b.label)
        bins = [r["bin"] for r in found[0][1]["sr2"]["by_snr"]]
        size = [np.mean([s["sr2"]["by_snr"][k]["n"] for _, s in found])
                for k in range(len(bins))]
        ax.axhline(1.0, color="0.3", lw=1.0, ls="--")
        ax.axhline(0.0, color="0.3", lw=1.0, ls=":")
        ax.set_xticks(np.arange(len(bins)),
                      [f"{b}\n$n\\approx{round(n, -1):,.0f}$"
                       for b, n in zip(bins, size, strict=True)])
        ax.set_xlim(-0.5, len(bins) - 0.5)
        ax.set_ylim(-0.15, 1.12)
        ax.set_ylabel(r"Response exponent $r$")
        ax.set_xlabel("Line recoverability")
        ax.text(0.985, 1.02, "follows the input", fontsize=9, color="0.3",
                ha="right", transform=ax.get_yaxis_transform())
        ax.text(0.985, -0.10, "ignores the input", fontsize=9, color="0.3",
                ha="right", transform=ax.get_yaxis_transform())
        ax.legend(frameon=False, fontsize=9.5, loc="upper left",
                  bbox_to_anchor=(0.0, 0.93))
        fig.tight_layout()
        return F._save(fig, "response_bands.png", outdir=outdir)


# ---------------------------------------------------------------------------
# [N II] / H-alpha
# ---------------------------------------------------------------------------
def nii_halpha_ratio(c, half_aa: float = 7.0,
                     side_aa: tuple[float, float] = (45.0, 100.0)) -> dict:
    """[N II] 6583 / H-alpha in the target and in the output, where H-alpha is detected.

    Both lines are integrated over a rest-frame window of ``+/-half_aa`` above
    a sideband continuum, which separates them in the target and in the
    output (it cannot in the grism spectrum, where they are one feature).

    The result is a statement about the simulation as much as about the
    model: in OpenUniverse2024 the ratio is nearly the same for every galaxy,
    so there is no galaxy-to-galaxy variation for a model to recover, and the
    scatter of the output about the truth is larger than the scatter of the
    truth itself.
    """
    wl = np.asarray(c["wl_um"]) * 1e4
    z, hr, sr = np.asarray(c["z_true"]), np.asarray(c["hr"]), np.asarray(c["sr2"])

    def flux(spec, zi, rest):
        d = wl / (1.0 + zi) - rest
        if d[0] > -side_aa[1] or d[-1] < side_aa[1]:
            return np.nan
        core = np.abs(d) < half_aa
        side = (np.abs(d) > side_aa[0]) & (np.abs(d) < side_aa[1])
        return float(np.sum(spec[core] - np.median(spec[side])))

    rows = np.where(np.asarray(c["line_snr"])[:, 3] >= MIN_BEST_LINE_SNR)[0]
    f = np.array([[flux(a[i], z[i], r) for a in (hr, sr) for r in (6562.8, 6583.5)]
                  for i in rows])
    ok = np.isfinite(f).all(axis=1) & (f[:, 0] > 0) & (f[:, 1] > 0)
    f = f[ok]
    true = f[:, 1] / f[:, 0]
    good = (f[:, 2] > 0) & (f[:, 3] > 0)
    out_ratio = f[good, 3] / f[good, 2]
    lt, lo = np.log10(true[good]), np.log10(out_ratio)
    return {"n": int(len(true)), "true_median": float(np.median(true)),
            "true_p05": float(np.percentile(true, 5)),
            "true_p95": float(np.percentile(true, 95)),
            "true_scatter_dex": float(np.std(np.log10(true))),
            "usable_frac": float(good.mean()),
            "bias": float(np.median(out_ratio / true[good])),
            "scatter_dex": float(np.std(lo - lt)),
            "corr": float(np.corrcoef(lo, lt)[0, 1])}


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
