"""Publication figures.

Every figure reads the frozen prediction cache (:mod:`specsr_roman.evaluation.cache`)
rather than a live model, so a plotting tweak can never quietly change a
result.

The five figures, and what each is for:

``spectra``
    HR / LR / SR2 overlay with a zoom inset on the blended complex. The inset
    is the point --- whether [OIII]4959/5007 and Ha/[NII] separate is what
    "super-resolution" means here, and it is invisible at full-band scale.
``river``
    Residual maps sorted by redshift, with rest-frame line tracks. Systematic
    failures show up as structure along a track; noise does not.
``sn``
    Per-line S/N, SR2 against the LR input. Answers "did this become
    measurable", which is distinct from "was the amplitude right".
``redshift``
    z_pred against z_true. Read the off-diagonal stripes: they are alias
    families, and a scalar NMAD hides them completely.
``psd``
    Signal and residual power spectra --- where in spatial frequency the
    reconstruction adds information, and where it only adds noise.

The palette is shared with the JWST companion paper so HR/LR/SR mean the same
colour in both, and it survives greyscale printing.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

__all__ = ["FigureStyle", "plot_spectra", "plot_river", "plot_sn",
           "plot_redshift", "plot_psd", "make_figures", "FIGURES"]

# Strong lines (rest-frame micron) for tracks, S/N and insets.
LINES = {r"[OII]": 0.3727, r"H$\beta$": 0.4861, r"[OIII]": 0.5007,
         r"H$\alpha$": 0.6563, r"HeI": 1.0830, r"Pa$\beta$": 1.2818}

# Annotated on the example-spectra panels. Finer than LINES: the doublets are
# split, because whether the SR separates them is what the panels exist to show.
LABEL_LINES = [
    (r"[OII]", 0.37275),
    (r"H$\beta$", 0.48613),
    (r"[OIII]4959", 0.49590),
    (r"[OIII]5007", 0.50076),
    (r"H$\alpha$", 0.65628),
    (r"[NII]6583", 0.65853),
    (r"[SII]6716,31", 0.67236),
    (r"[SIII]9069", 0.90691),
    (r"[SIII]9531", 0.95332),
    (r"HeI", 1.08303),
    (r"Pa$\beta$", 1.28216),
]

# What the wide panel labels. The fine doublets above are for the zoom inset;
# at 0.93 um across a 13-inch axis their names collide into an unreadable
# smear, so the main panel names the complex and the inset splits it.
MAIN_LABELS = [
    (r"[OII]", 0.37275),
    (r"H$\beta$", 0.48613),
    (r"[OIII]", 0.50076),
    (r"H$\alpha$+[NII]", 0.65628),
    (r"[SII]", 0.67236),
    (r"[SIII]", 0.95332),
    (r"HeI", 1.08303),
    (r"Pa$\beta$", 1.28216),
]

# Rest-frame zoom windows, redshifted per object so panels at different z show
# the same physical range.
LINE_GROUPS = {
    "o3hb": {"rest": (0.4800, 0.5120), "weight": 1.15},
    "halpha_nii": {"rest": (0.6505, 0.6625), "weight": 1.00},
    "sulfur": {"rest": (0.8950, 0.9650), "weight": 0.70},
}

_PLASMA = plt.get_cmap("plasma")
COLOR_HR = _PLASMA(0.18)
COLOR_LR = _PLASMA(0.52)
COLOR_SR = _PLASMA(0.72)

OUTDIR = "outputs/figures"


class FigureStyle:
    """Apply the shared rcParams. Use as a context manager or call ``apply()``."""

    PARAMS = {
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 0.9,
        "figure.dpi": 110,
    }

    @classmethod
    def apply(cls) -> None:
        plt.rcParams.update(cls.PARAMS)

    def __enter__(self):
        self._saved = dict(plt.rcParams)
        self.apply()
        return self

    def __exit__(self, *exc):
        plt.rcParams.update(self._saved)


def _save(fig, name: str, outdir: str | None = None) -> str:
    # Read the module global at call time, not at def time: `make_figures`
    # redirects OUTDIR for the duration of a render, and a default argument
    # bound at import would ignore it.
    outdir = OUTDIR if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {path}")
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _robust_ylims(y, p_lo=0.5, p_hi=99.5, pad_frac=0.06):
    """Percentile limits a single bad pixel cannot blow out."""
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    lo, hi = np.percentile(y, [p_lo, p_hi])
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _line_report(wl, hr, sr, z, lines=LABEL_LINES, core_um=0.0018, side_um=0.035):
    """Per-line amplitude of HR and SR above a local continuum.

    Returns one record per line that falls in the band, with the HR
    significance and the worst SR undershoot next to the line -- the two
    numbers the example picker needs.
    """
    out = []
    for name, lam_rest in lines:
        lam_obs = lam_rest * (1.0 + z)
        if not (wl[0] + 0.03 < lam_obs < wl[-1] - 0.03):
            continue
        core = np.abs(wl - lam_obs) <= core_um
        side = np.abs(wl - lam_obs) <= side_um
        near = np.abs(wl - lam_obs) <= 0.010
        if core.sum() < 3 or side.sum() < 20:
            continue
        c_hr = np.median(hr[side])
        noise = 1.4826 * np.median(np.abs(hr[side] - c_hr)) + 1e-6
        c_sr = np.median(sr[side])
        out.append({
            'name': name, 'lam_rest': lam_rest, 'lam_obs': lam_obs,
            'a_hr': float(hr[core].max() - c_hr),
            'a_sr': float(sr[core].max() - c_sr),
            'snr': float((hr[core].max() - c_hr) / noise),
            'undershoot': float(c_sr - sr[near].min()),
        })
    return out


def _peak_shift(wl, hr, sr, lam_obs, half_um=0.010):
    """Offset between the HR and SR peak of one line, in microns."""
    m = np.abs(wl - lam_obs) <= half_um
    if m.sum() < 5:
        return 0.0
    return float(abs(wl[m][hr[m].argmax()] - wl[m][sr[m].argmax()]))


def _profile_err(wl, hr, sr, lam_obs, half_um=0.006, side_um=0.035):
    """RMS SR-HR difference across one line profile, in units of the line.

    Peak amplitude and peak position each miss half the story -- a line can
    match in height and still be too broad, or sit right and be the wrong
    shape. This is the single number that covers all three, and it is what
    "the line detail is right" actually means.
    """
    m = np.abs(wl - lam_obs) <= half_um
    if m.sum() < 5:
        return np.inf
    side = np.abs(wl - lam_obs) <= side_um
    a = hr[m] - np.median(hr[side])
    b = sr[m] - np.median(sr[side])
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-6))


def _rank_line_examples(c, snr_min=6.0, n_lines_min=2, amp_tol=0.30,
                        ring_tol=0.12, dz_tol=0.01, shift_tol=0.0009,
                        prof_tol=0.45, bright_amp_tol=0.20):
    """Examples where the SR gets the *line detail* right, not just the peak.

    The old picker ranked on raw line S/N and one peak-amplitude check, which
    selected bright single-line objects -- including ones where the SR fired a
    huge spike with a deep ringing undershoot beside it. This asks for what the
    figure is meant to demonstrate: several detected lines per object, each
    reproduced to within ``amp_tol`` in amplitude, no ringing deeper than
    ``ring_tol`` of the brightest line, and a redshift that lands.
    """
    wl, hr, sr = c['wl_um'], c['hr'], c['sr2']
    zt, zp = c['z_true'], c['z_pred']
    dz = np.abs(zp - zt) / (1 + zt)
    scored = []
    for i in np.where(dz < dz_tol)[0]:
        rep = [r for r in _line_report(wl, hr[i], sr[i], float(zt[i]))
               if r['snr'] > snr_min]
        if len(rep) < n_lines_min:
            continue
        errs = [abs(r['a_sr'] - r['a_hr']) / (abs(r['a_hr']) + 1e-6) for r in rep]
        peak = max(r['a_hr'] for r in rep)
        ring = max(r['undershoot'] for r in rep) / (peak + 1e-6)
        if max(errs) > amp_tol or ring > ring_tol:
            continue
        # amplitude alone lets through spectra whose SR line sits a resolution
        # element off the HR one, which reads as a wavelength error in the zoom
        shift = max(_peak_shift(wl, hr[i], sr[i], r['lam_obs']) for r in rep)
        if shift > shift_tol:
            continue
        # judged on the brightest line: it is the one the zoom inset shows, and
        # the faint end of the list is profile-dominated by its own noise
        bright = max(rep, key=lambda r: r['a_hr'])
        prof = _profile_err(wl, hr[i], sr[i], bright['lam_obs'])
        if prof > prof_tol:
            continue
        # the eye compares peak heights before anything else, so hold the line
        # the zoom is built around to a tighter amplitude tolerance than the rest
        if abs(bright['a_sr'] - bright['a_hr']) / abs(bright['a_hr']) > bright_amp_tol:
            continue
        scored.append((len(rep), prof, int(i)))
    # most lines first, then the truest line profile among those
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[2] for t in scored]


def _pick_inset_group(wl, hr, z, rep, avoid=()):
    """Which line complex the zoom should frame.

    The brightest complex in band, except that a complex already zoomed in
    another panel is passed over when a second one is resolved here -- two
    panels showing the same Ha/[NII] pair make one point twice, where
    Ha/[NII] plus the [OIII] doublet shows the deblending at two separations.
    """
    cands = []
    for key, g in LINE_GROUPS.items():
        lo, hi = g['rest'][0] * (1 + z), g['rest'][1] * (1 + z)
        if hi < wl[0] or lo > wl[-1]:
            continue
        inw = [r for r in rep if lo <= r['lam_obs'] <= hi and r['a_hr'] > 0]
        if not inw:
            continue
        cands.append((key, g['weight'] * sum(r['a_hr'] for r in inw),
                      sum(1 for r in inw if r['snr'] > 8)))
    if not cands:
        return None
    fresh = [c for c in cands if c[0] not in avoid and c[2] >= 2]
    return max(fresh or cands, key=lambda c: c[1])[0]


def _annotate_lines(ax, marks, y_lo, y_hi, head=0.16, fontsize=8,
                    mask_band=True):
    """Dotted markers on the lines, names in a clear band along the top.

    Rotated labels centred on their line run straight through the peak they
    name. Instead reserve a band above the data, paint it clear, and set the
    names horizontally inside it. Labels near an edge are anchored to that edge
    so nothing overhangs the frame, and any two that would overprint are
    resolved by measuring the rendered text -- estimating the width from the
    character count gets mathtext (``$\\alpha$``, ``$\\beta$``) badly wrong.
    """
    band = head * (y_hi - y_lo)
    ax.set_ylim(y_lo, y_hi + band)
    if not marks:
        return
    if mask_band:
        ax.axhspan(y_hi, y_hi + band, facecolor='white', edgecolor='none',
                   zorder=4)
        for spine in ax.spines.values():
            spine.set_zorder(7)
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    texts = []
    for name, lam in sorted(marks, key=lambda t: t[1]):
        ax.axvline(lam, ymax=1.0 / (1.0 + head), color='0.30', ls=':',
                   lw=0.9, alpha=0.85, zorder=5)
        frac = (lam - x0) / span
        # keep the text inside the frame: anchor edge labels to the edge
        ha = 'left' if frac < 0.06 else ('right' if frac > 0.94 else 'center')
        texts.append(ax.annotate(
            name, xy=(lam, 1.0), xycoords=('data', 'axes fraction'),
            xytext=(0, -3), textcoords='offset points',
            ha=ha, va='top', fontsize=fontsize, color='0.15', zorder=6,
            annotation_clip=False))

    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    kept = []
    for t in texts:
        bb = t.get_window_extent(rend).expanded(1.12, 1.0)
        if any(bb.overlaps(k) for k in kept):
            t.remove()          # marker stays; the crowded name goes
            continue
        kept.append(bb)


def _inset_slot(ax, wl, curves, zoom_x, width=0.40, height=0.51, bottom=0.34):
    """Where to park the zoom inset so it hides as little as possible.

    A fixed upper-right box works until the panel's brightest line happens to
    sit under it -- which it often does, because the brightest line is usually
    the one being zoomed. Try a few positions, throw out any that would sit on
    top of the stretch the inset is showing, and keep the one the spectra poke
    into least.
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    span, yspan = x_hi - x_lo, y_hi - y_lo
    best, best_cost = None, np.inf
    for left in (0.045, 0.30, 0.575, 0.595):
        bx0 = x_lo + left * span
        bx1 = x_lo + (left + width) * span
        if bx0 <= zoom_x[1] and zoom_x[0] <= bx1:     # would cover its own zoom
            continue
        by = y_lo + bottom * yspan
        m = (wl >= bx0) & (wl <= bx1)
        cost = sum(float(np.clip(y[m] - by, 0, None).sum()) for y in curves)
        if cost < best_cost:
            best, best_cost = left, cost
    if best is None:                                   # zoom spans the panel
        best = 0.575
    return [best, bottom, width, height]


def plot_spectra(c, n_examples=2):
    wl, lr, s2, hr = c['wl_um'], c['lr'], c['sr2'], c['hr']
    zt, zp = c['z_true'], c['z_pred']

    order = _rank_line_examples(c)
    if not order:
        raise SystemExit('no examples pass the line-fidelity cuts')
    # spread the picks over the ranked list so the panels are not near-duplicate
    # objects at the same redshift
    picks, seen_z = [], []
    for i in order:
        if any(abs(zt[i] - z) < 0.05 for z in seen_z):
            continue
        picks.append(i); seen_z.append(float(zt[i]))
        if len(picks) == n_examples:
            break

    used_groups = set()
    fig, axes = plt.subplots(n_examples, 1, figsize=(13, 3.6 * n_examples),
                             squeeze=False, sharex=True,
                             constrained_layout=True)
    for ax, i in zip(axes[:, 0], picks, strict=False):
        z = float(zt[i])
        rep = _line_report(wl, hr[i], s2[i], z)
        ax.plot(wl, hr[i], color=COLOR_HR, lw=1.0, alpha=0.75,
                label='HR target', zorder=1)
        ax.plot(wl, lr[i], color=COLOR_LR, lw=1.05, ls='--', alpha=0.85,
                label='LR grism', zorder=2)
        ax.plot(wl, s2[i], color=COLOR_SR, lw=1.5, alpha=0.95,
                label='SR (ours)', zorder=3)
        ax.set_xlim(wl[0], wl[-1])
        ax.set_ylabel(r'$F_\lambda$ (normalized)')

        y_lo, y_hi = _robust_ylims(np.concatenate([hr[i], lr[i], s2[i]]))
        y_hi = max(y_hi, 1.05 * max(hr[i].max(), s2[i].max()))
        # the wide panel names the complex; the inset splits the doublet
        det = {r['lam_rest']: r for r in rep}
        marks = []
        for name, lam_rest in MAIN_LABELS:
            r = min(det.values(), key=lambda q: abs(q['lam_rest'] - lam_rest),
                    default=None)
            if r is None or abs(r['lam_rest'] - lam_rest) > 0.02 or r['snr'] < 5:
                continue
            marks.append((name, lam_rest * (1 + z)))
        _annotate_lines(ax, marks, y_lo, y_hi, head=0.13, fontsize=8.5)

        # ---- zoom inset on the strongest line complex in band ----
        key = _pick_inset_group(wl, hr[i], z, rep, avoid=used_groups)
        slot, x0, x1, m = None, None, None, None
        if key is not None:
            rest_lo, rest_hi = LINE_GROUPS[key]['rest']
            x0 = max(rest_lo * (1 + z), float(wl[0]))
            x1 = min(rest_hi * (1 + z), float(wl[-1]))
            m = (wl >= x0) & (wl <= x1)
            if m.sum() >= 10:
                used_groups.add(key)
                slot = _inset_slot(ax, wl, (hr[i], s2[i]), (x0, x1))
            else:
                key = None

        # the redshift label takes whichever top corner the inset left free
        z_right = slot is not None and slot[0] < 0.5
        ax.text(0.988 if z_right else 0.012, 0.865,
                rf'$z_{{\rm true}}={z:.3f}\ \ \hat z={zp[i]:.3f}$',
                transform=ax.transAxes, va='top',
                ha='right' if z_right else 'left', fontsize=11, zorder=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6',
                          alpha=0.9))
        if key is None:
            continue

        zoom = np.concatenate([hr[i][m], lr[i][m], s2[i][m]])
        zy_lo = _robust_ylims(zoom, p_lo=1, p_hi=99, pad_frac=0.10)[0]
        zy_hi = float(np.nanmax(zoom))
        zy_hi += 0.06 * (zy_hi - zy_lo)
        # bracket the zoomed range on the wide panel
        for xb in (x0, x1):
            ax.axvline(xb, color='k', ls=':', lw=1.1, alpha=0.85,
                       ymax=1.0 / 1.13, zorder=5)

        # the inset stops below the label band so the two never overlap
        axin = ax.inset_axes(slot)
        axin.plot(wl[m], hr[i][m], color=COLOR_HR, lw=1.1, alpha=0.8, zorder=1)
        axin.plot(wl[m], lr[i][m], color=COLOR_LR, lw=1.1, ls='--', alpha=0.85,
                  zorder=2)
        axin.plot(wl[m], s2[i][m], color=COLOR_SR, lw=1.6, alpha=0.95, zorder=3)
        axin.set_xlim(x0, x1)
        axin.tick_params(labelsize=7.5)
        axin.set_facecolor('white')
        zmarks = [(r['name'], r['lam_obs']) for r in rep
                  if x0 <= r['lam_obs'] <= x1 and r['snr'] > 4]
        _annotate_lines(axin, zmarks, zy_lo, zy_hi, head=0.19, fontsize=7.5)

    axes[-1, 0].set_xlabel(r'Observed wavelength [$\mu$m]')
    # one legend for the whole figure, outside the axes: an in-panel legend has
    # to share the upper right with the inset and the upper left with the z box
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='outside upper center', ncol=3, frameon=False,
               fontsize=11)
    _save(fig, 'spectra_examples.png')


# ----------------------------- 2. residual river -----------------------------
def _robust_sigma_cols(R, min_count=30):
    L = R.shape[1]; out = np.full(L, np.nan)
    for j in range(L):
        col = R[:, j]; col = col[np.isfinite(col)]
        if col.size < min_count:
            continue
        med = np.median(col)
        out[j] = 1.4826 * np.median(np.abs(col - med)) + 1e-12
    return out


def plot_river(c):
    wl, lr, s2, hr, zt = c['wl_um'], c['lr'], c['sr2'], c['hr'], c['z_true']
    lre, sig = c['lr_err'], c['sigma']
    keep = zt >= 0
    lr, s2, hr, zt = lr[keep], s2[keep], hr[keep], zt[keep]
    lre, sig = lre[keep], sig[keep]
    order = np.argsort(zt); zt = zt[order]
    R0raw = (lr - s2)[order]; R1raw = (s2 - hr)[order]
    # high-pass each residual row FOR THE IMAGE: removes the broad continuum
    # offset from independent per-spectrum normalization (SR and HR have
    # different line strengths -> different normalized continuum levels),
    # leaving the line-scale structure that actually matters.
    R0 = _highpass(R0raw, sigma_px=40)
    R1 = _highpass(R1raw, sigma_px=40)
    n = len(zt)
    vmax = np.percentile(np.abs(np.concatenate([R0.ravel(), R1.ravel()])), 97)
    ext = [wl[0], wl[-1], 0, n]

    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[4, 1],
                  width_ratios=[1, 1, 0.045], hspace=0.06, wspace=0.08)
    a0, a1, acb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    s0 = fig.add_subplot(gs[1, 0], sharex=a0); s1 = fig.add_subplot(gs[1, 1], sharex=a1)

    a0.imshow(R0, aspect='auto', origin='lower', extent=ext, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    im = a1.imshow(R1, aspect='auto', origin='lower', extent=ext,
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    a0.set_title(r'LR grism $-$ SR'); a1.set_title(r'SR $-$ HR target')
    a0.set_ylabel(r'Redshift $z$')

    ztk = [z for z in (0, 0.5, 1, 1.5, 2, 2.5, 3) if zt[0] <= z <= zt[-1]]
    pos = [min(int(np.searchsorted(zt, z)), n - 1) for z in ztk]
    a0.set_yticks(pos); a0.set_yticklabels([str(z) for z in ztk])
    a1.set_yticks(pos); a1.set_yticklabels([])
    plt.setp(a0.get_xticklabels(), visible=False)
    plt.setp(a1.get_xticklabels(), visible=False)
    cb = fig.colorbar(im, cax=acb); cb.set_label('Residual (norm. flux)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    wt = np.linspace(wl[0], wl[-1], 300)
    # fractional position of each label along its track (avoid crowding at top)
    lab_frac = {r'[OII]': 0.30, r'H$\beta$': 0.40, r'[OIII]': 0.58,
                r'H$\alpha$': 0.5, r'HeI': 0.5, r'Pa$\beta$': 0.5}
    for name, lam in LINES.items():
        zt_track = wt / lam - 1
        idx = np.interp(zt_track, zt, np.arange(n), left=-1, right=n + 1)
        m = (zt_track >= zt[0]) & (zt_track <= zt[-1])
        if m.sum() < 2:
            continue
        for ax in (a0, a1):
            ax.plot(wt[m], idx[m], 'k--', lw=0.7, alpha=0.5)
        p = int(lab_frac.get(name, 0.5) * (m.sum() - 1))
        a0.text(wt[m][p], idx[m][p], name, fontsize=8.5, ha='center',
                va='bottom', alpha=0.9)

    # bottom panels: residual scatter (from RAW residuals) vs a reference
    #   left  : LR-SR residual vs the LR noise floor (median norm flux_low_err)
    #   right : SR-HR residual vs the LR-HR residual (how much closer to truth
    #           SR is than the raw grism; HR is noiseless so there is no HR
    #           noise floor to compare against)
    sig0 = _robust_sigma_cols(R0raw); sig1 = _robust_sigma_cols(R1raw)
    floor_lr = np.median(np.abs(lre), axis=0)
    sig_lrhr = _robust_sigma_cols(lr - hr)
    panels = ((s0, sig0, floor_lr, r'$\tilde\sigma_{\rm LR}(\lambda)$'),
              (s1, sig1, sig_lrhr, r'$\sigma_{\rm LR-HR}(\lambda)$'))
    for ax, sg, ref, flab in panels:
        ax.plot(wl, sg, lw=1.8, color='#1f77b4', label=r'$\sigma_{\rm resid}(\lambda)$')
        ax.plot(wl, ref, lw=1.5, ls='--', color='0.4', label=flab)
        ax.set_xlim(wl[0], wl[-1]); ax.set_xlabel(r'Wavelength [$\mu$m]')
        ax.legend(fontsize=8, frameon=False, loc='upper right', ncol=2)
        fin = np.concatenate([sg[np.isfinite(sg)], ref[np.isfinite(ref)]])
        if fin.size:
            ax.set_ylim(0, np.percentile(fin, 99) * 1.2)
    s0.set_ylabel(r'$\sigma$')
    _save(fig, 'residual_river.png')


# ----------------------------- 3. per-line S/N -----------------------------
def _line_snr(wl, flux, lam_obs, half=0.045, core=0.012, sbgap=0.015, sbw=0.03):
    """Amplitude/continuum-noise S/N from a local peak + sideband estimate."""
    m = np.abs(wl - lam_obs) <= half
    if m.sum() < 8:
        return np.nan
    x, y = wl[m], flux[m]
    side = (np.abs(x - lam_obs) >= sbgap) & (np.abs(x - lam_obs) <= sbgap + sbw)
    if side.sum() < 5:
        return np.nan
    cont = np.median(y[side])
    noise = 1.4826 * np.median(np.abs(y[side] - cont)) + 1e-6
    corem = np.abs(x - lam_obs) <= core
    amp = np.max(y[corem]) - cont
    return amp / noise


def plot_sn(c):
    wl, lr, s2, zt = c['wl_um'], c['lr'], c['sr2'], c['z_true']
    panels = [(r'[OII] $\lambda$3727', 0.3727), (r'H$\beta$', 0.4861),
              (r'[OIII] $\lambda$5007', 0.5007), (r'H$\alpha$', 0.6563)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), squeeze=False)
    last = None
    for ax, (name, lam) in zip(axes[0], panels, strict=False):
        lam_obs = (1 + zt) * lam
        inb = (lam_obs > wl[0] + 0.05) & (lam_obs < wl[-1] - 0.05)
        idx = np.where(inb)[0]
        sn_lr, sn_sr = [], []
        for i in idx:
            a = _line_snr(wl, lr[i], lam_obs[i]); b = _line_snr(wl, s2[i], lam_obs[i])
            if np.isfinite(a) and np.isfinite(b):
                sn_lr.append(a); sn_sr.append(b)
        sn_lr, sn_sr = np.array(sn_lr), np.clip(np.array(sn_sr), 0, None)
        sn_lr = np.clip(sn_lr, 0, None)
        hb = ax.hexbin(sn_lr, sn_sr, gridsize=40, extent=(0, 70, 0, 70),
                       bins='log', cmap='viridis', mincnt=1)
        ax.plot([0, 70], [0, 70], '--', color='0.4', lw=1)
        ax.set_xlim(0, 70); ax.set_ylim(0, 70)
        ax.set_title(name); ax.set_xlabel('S/N (LR input)')
        if name.startswith('[OII]'):
            ax.set_ylabel('S/N (SR)')
        f_lr = np.mean(sn_lr > 10) if sn_lr.size else 0
        f_sr = np.mean(sn_sr > 10) if sn_sr.size else 0
        ax.text(0.04, 0.95, f'f(S/N>10): LR={f_lr:.2f}, SR={f_sr:.2f}',
                transform=ax.transAxes, va='top', fontsize=9)
        last = hb
    cb = fig.colorbar(last, ax=axes[0, -1], fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10}$(count)', fontsize=9)
    fig.tight_layout()
    _save(fig, 'sn_comparison.png')


# ----------------------------- 4. redshift -----------------------------
def plot_redshift(c):
    zt, zp = c['z_true'], c['z_pred']
    m = np.isfinite(zt) & np.isfinite(zp) & (zt >= 0)
    zt, zp = zt[m], zp[m]
    dz = (zp - zt) / (1 + zt)
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    outl = np.mean(np.abs(dz) > 0.15)
    hi = max(zt.max(), zp.max()) * 1.02

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    hb = ax.hexbin(zt, zp, gridsize=60, bins='log', cmap='viridis', mincnt=1,
                   extent=(0, hi, 0, hi))
    ax.plot([0, hi], [0, hi], '-', color='#1f77b4', lw=1.2)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel('True redshift'); ax.set_ylabel('Predicted redshift')
    ax.set_title('ZHead (super-res spectrum + Roman 3-band photometry)')
    ax.text(0.04, 0.96,
            f'NMAD: {nmad:.4f}\nMed |dz|/(1+z): {np.median(np.abs(dz)):.4f}\n'
            f'Outlier (>0.15): {100*outl:.2f}%\nN = {len(zt):,}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6', alpha=0.9))
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Count per hex (log)', fontsize=9)
    fig.tight_layout()
    _save(fig, 'redshift_zpred.png')


# ----------------------------- 5. PSD -----------------------------
def _highpass(X, sigma_px=25):
    from scipy.ndimage import gaussian_filter1d
    return X - gaussian_filter1d(X, sigma_px, axis=1, mode='nearest')


def _psd(X, hp=True):
    Xd = _highpass(X) if hp else X - X.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(Xd, axis=1)) ** 2
    return P.mean(0), np.percentile(P, 16, 0), np.percentile(P, 84, 0)


def plot_psd(c):
    wl, lr, s2, hr = c['wl_um'], c['lr'], c['sr2'], c['hr']
    dl = (wl[-1] - wl[0]) / (len(wl) - 1)
    k = np.fft.rfftfreq(len(wl), d=dl)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.6))
    for X, lab, col in ((lr, 'LR', '#1f77b4'), (s2, 'SR', '#f5a623'), (hr, 'HR', '#2ca02c')):
        mu, lo, up = _psd(X)
        axL.plot(k[1:], mu[1:], color=col, lw=1.6, label=lab)
        axL.fill_between(k[1:], lo[1:], up[1:], color=col, alpha=0.18)
    for A, B, lab, col in ((lr, hr, 'LR $-$ HR', '#1f77b4'),
                           (lr, s2, 'LR $-$ SR', '#f5a623'),
                           (s2, hr, 'SR $-$ HR', '#2ca02c')):
        mu, lo, up = _psd(A - B, hp=False)
        axR.plot(k[1:], mu[1:], color=col, lw=1.6, label=lab)
        axR.fill_between(k[1:], lo[1:], up[1:], color=col, alpha=0.18)
    # grism resolution cutoff (native LR sampling: 864 px over the band)
    kcut = 1.0 / (2 * (wl[-1] - wl[0]) / 864)
    for ax in (axL, axR):
        ax.axvline(kcut, color='0.5', ls='--', lw=1)
        ax.set_yscale('log'); ax.set_xlim(0, k[-1])
        ax.set_xlabel(r'Frequency $k$ (cycles / $\mu$m)')
        ax.legend(fontsize=10, frameon=False)
    axL.set_ylabel('High-pass signal power spectral density (arb.)')
    axR.set_ylabel('Residual power spectral density (arb.)')
    fig.tight_layout()
    _save(fig, 'psd.png')




# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
FIGURES = {
    "spectra": plot_spectra,
    "redshift": plot_redshift,
    "river": plot_river,
    "psd": plot_psd,
    "sn": plot_sn,
}


def make_figures(cache, which=None, outdir: str = OUTDIR) -> list[str]:
    """Render the named figures from a prediction cache.

    ``which`` is a list of keys from :data:`FIGURES`, or ``None`` for all.
    Ordering is deliberate --- the cheap panels render first, so a broken cache
    fails in seconds rather than after the residual river.
    """
    global OUTDIR
    previous, OUTDIR = OUTDIR, outdir
    try:
        keys = list(FIGURES) if which in (None, "all") else list(which)
        unknown = set(keys) - set(FIGURES)
        if unknown:
            raise ValueError(f"unknown figure(s) {sorted(unknown)}; "
                             f"choose from {sorted(FIGURES)}")
        with FigureStyle():
            for k in keys:
                FIGURES[k](cache)
        return keys
    finally:
        OUTDIR = previous
