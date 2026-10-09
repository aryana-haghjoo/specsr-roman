"""Figures of the OpenUniverse2024 dataset itself.

These describe the training data, not the model, so unlike
:mod:`specsr_roman.evaluation.figures` they do not read the prediction cache.

``example``
    One galaxy followed through the simulation: the OpenUniverse2024 H158
    image, the grism exposure we disperse from it, the two-dimensional spectrum
    of the target, and the extracted spectrum against its noiseless target.
``parent``
    The galaxies of the dataset against the parent catalogue: redshift,
    H158 magnitude and stellar mass.

The example needs the two-dimensional arrays, which the dataset file does not
keep. :func:`build_example_cache` re-runs the dispersion of one detector image
with the noise seed of the batch extraction, so the cutouts are the pixels the
dataset spectrum was extracted from, and stores a handful of candidates.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..grids import (
                      AB_ANCHOR_H158,
                      PHOT_BANDS,
                      WAVE_HR,
                      WAVE_LR,
)
from .figures import COLOR_HR, COLOR_LR, COLOR_SR, MAIN_LABELS, FigureStyle, _save

__all__ = ["build_example_cache", "plot_example", "plot_parent_sample",
           "make_dataset_figures", "EXAMPLE_CACHE", "PARENT_STATS"]

EXAMPLE_CACHE = "outputs/ou2024_example_cache.npz"
PARENT_STATS = "outputs/ou2024_parent_stats.json"
#: The galaxy the paper shows: H-alpha and [O III] both visible at z = 1.23.
EXAMPLE_ID = 10307100339698


def _band_mean(wave, flam, lo, hi):
    m = (wave > lo) & (wave < hi)
    return float(np.mean(flam[m])) if m.sum() > 3 else np.nan


def build_example_cache(visit: int = 16188, sca: int = 10, cfg=None,
                        out: str = EXAMPLE_CACHE, n_keep: int = 8,
                        z_range: tuple[float, float] = (1.05, 1.85)) -> str:
    """Re-disperse one detector image and store cutouts of a few targets.

    The candidates are the targets of this image with H-alpha and [O III] both
    inside the grism band (``z_range``), ranked by the S/N of the extracted
    spectrum. Everything the example figure draws is stored, so the choice
    among them is a plotting decision.
    """
    from grizli import model as gmodel

    from ..extraction.batch import ExtractionConfig, _load_catalogues
    from ..extraction.catalog import ab_h158, detect_and_relabel, load_truth_index
    from ..extraction.download import sca_paths
    from ..extraction.extract import ExtractionFailure, extract_target, to_fixed_grids
    from ..extraction.frames import prepare_frames
    from ..extraction.seds import SEDLibrary
    from ..extraction.simulate import add_grism_noise, disperse_scene
    from ..grids import GRIZLI_BEAM_SIZE, GRIZLI_PAD

    cfg = cfg or ExtractionConfig()
    img, idx, _, _ = sca_paths(visit, sca, cfg.raw_dir)
    truth = load_truth_index(idx)
    redshifts, flux_row, flux_cols = _load_catalogues(cfg)
    seds = SEDLibrary(cfg.sed_hdf5)

    direct, grism = prepare_frames(img, cfg.prepared_dir, cfg.grism_exptime)
    flt = gmodel.GrismFLT(grism_file=grism, direct_file=direct,
                          pad=GRIZLI_PAD, verbose=False)
    compact_ids, object_ids, _, matched = detect_and_relabel(flt, truth)

    f_h = np.array([flux_cols["roman_flux_H158"][flux_row[o]] if o in flux_row
                    else 0.0 for o in object_ids])
    ab = ab_h158(f_h)
    zs = np.array([redshifts.get(int(o), -1.0) for o in object_ids])
    scene_sel = matched & (ab < cfg.ab_scene) & (zs >= 0)
    target_sel = scene_sel & (ab < cfg.ab_target) & (zs < cfg.z_max)
    scene_idx = [i for i in np.argsort(ab) if scene_sel[i]]

    kept = disperse_scene(flt, compact_ids, object_ids, ab, redshifts, seds,
                          scene_idx)
    scene = flt.model.astype(np.float64)
    # the seed of the batch extraction: the same noise the dataset row carries
    noisy, err2d = add_grism_noise(scene, exptime=cfg.grism_exptime,
                                   seed=int(visit) * 100 + int(sca))

    direct_img = np.asarray(flt.direct.data["SCI"], dtype=np.float32)
    seg = np.asarray(flt.seg)
    ny, nx = scene.shape

    cand = []
    for i in np.where(target_sel & (zs > z_range[0]) & (zs < z_range[1]))[0]:
        if i not in kept:
            continue
        spec, (sed_wave, sed_flux) = kept[i]
        try:
            wave, flam, flam_err = extract_target(
                flt, scene, noisy, err2d, int(compact_ids[i]), float(ab[i]), spec)
            lr, lr_err, hr = to_fixed_grids(wave, flam, flam_err, sed_wave, sed_flux)
        except ExtractionFailure:
            continue
        cand.append((float(np.nanmedian(flam / flam_err)), i, lr, lr_err, hr))
    cand.sort(key=lambda t: -t[0])

    store = {"visit": visit, "sca": sca, "wave_lr": WAVE_LR, "wave_hr": WAVE_HR,
             "phot_bands": np.array(PHOT_BANDS)}
    for n, (snr, i, lr, lr_err, hr) in enumerate(cand[:n_keep]):
        cid, oid = int(compact_ids[i]), int(object_ids[i])
        spec, (sed_wave, sed_flux) = kept[i]
        out2 = flt.compute_model_orders(id=cid, mag=float(ab[i]),
                                        size=GRIZLI_BEAM_SIZE, compute_size=False,
                                        spectrum_1d=spec, is_cgs=False,
                                        store=False, in_place=False)
        if isinstance(out2, (list, tuple)):
            out2 = out2[1]
        own2d = np.asarray(out2, dtype=np.float32).reshape(scene.shape)
        beam = flt.compute_model_orders(id=cid, mag=float(ab[i]),
                                        size=GRIZLI_BEAM_SIZE, compute_size=False,
                                        store=False, in_place=False,
                                        get_beams=["A"])["A"]
        slx, sly = beam.slx_parent, beam.sly_parent
        yy, xx = np.nonzero(seg == cid)
        xc, yc = float(xx.mean()), float(yy.mean())
        # a region that holds the source and its whole first-order trace
        x0 = int(max(0, min(xc - 160, slx.start - 40)))
        x1 = int(min(nx, max(xc + 160, slx.stop + 40)))
        y0, y1 = int(max(0, yc - 160)), int(min(ny, yc + 160))
        p = f"c{n}_"
        store.update({
            p + "id": oid, p + "z": float(zs[i]), p + "ab": float(ab[i]),
            p + "snr": snr, p + "lr": lr, p + "lr_err": lr_err, p + "hr": hr,
            p + "sed_wave": sed_wave.astype(np.float32),
            p + "sed_flux": sed_flux.astype(np.float64),
            p + "phot": np.array([flux_cols[b][flux_row[oid]] for b in PHOT_BANDS]),
            p + "beam_lam": np.asarray(beam.lam, dtype=np.float32),
            p + "beam_noisy": noisy[sly, slx].astype(np.float32),
            p + "beam_clean": (noisy[sly, slx]
                               - (scene - own2d)[sly, slx]).astype(np.float32),
            p + "beam_own": own2d[sly, slx],
            p + "region_direct": direct_img[y0:y1, x0:x1],
            p + "region_grism": noisy[y0:y1, x0:x1].astype(np.float32),
            p + "region_box": np.array([x0, x1, y0, y1]),
            p + "beam_box": np.array([slx.start, slx.stop, sly.start, sly.stop]),
            p + "xy": np.array([xc, yc]),
        })
        print(f"  candidate {n}: id {oid} z={zs[i]:.3f} AB={ab[i]:.2f} S/N={snr:.1f}")
    store["n"] = min(n_keep, len(cand))
    seds.close()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, **store)
    print(f"wrote {out}")
    return out


# ---------------------------------------------------------------------------
# The example figure
# ---------------------------------------------------------------------------
def _stretch(img, p_lo=30.0, p_hi=99.7):
    lo, hi = np.nanpercentile(img, [p_lo, p_hi])
    return dict(vmin=lo, vmax=hi, cmap="gray_r", origin="lower",
                interpolation="nearest")


def plot_example(cache: str = EXAMPLE_CACHE, which: int = EXAMPLE_ID,
                 name: str = "ou2024_example.png"):
    c = np.load(cache, allow_pickle=True)
    # `which` is an object id, or the rank of a cached candidate. The rank is
    # by extraction S/N and moves when the dataset is rebuilt; the id does not.
    if which >= int(c["n"]):
        ranks = [k for k in range(int(c["n"])) if int(c[f"c{k}_id"]) == which]
        if not ranks:
            raise ValueError(f"galaxy {which} is not among the cached candidates")
        which = ranks[0]
    p = f"c{which}_"
    z, ab = float(c[p + "z"]), float(c[p + "ab"])
    wl_lr, wl_hr = c["wave_lr"] * 1e-4, c["wave_hr"] * 1e-4

    # The pair as the dataset stores it. The extraction is already relative to
    # the mean over H158; the target is put on the same scale.
    lr, lr_err = c[p + "lr"], c[p + "lr_err"]
    hr = c[p + "hr"] / _band_mean(c["wave_hr"], c[p + "hr"], 13800.0, 17700.0)

    x0, x1, y0, y1 = c[p + "region_box"]
    bx0, bx1, by0, by1 = c[p + "beam_box"]
    xc, yc = c[p + "xy"]
    lam = c[p + "beam_lam"].astype(np.float64) * 1e-4
    in_band = np.where((lam > wl_hr[0]) & (lam < wl_hr[-1]))[0]
    # the trace is offset from the source row: find it in the noiseless model
    trace = int(np.argmax(c[p + "beam_own"][:, in_band].sum(axis=1)))
    half_h = 20
    # the cutout is not centred on the trace, which can sit near its edge
    r0 = max(trace - half_h, 0)
    r1 = min(trace + half_h, c[p + "beam_own"].shape[0])

    ylim = (max(y0 - yc, -110), min(y1 - yc, 110))
    ext = (x0 - xc, x1 - xc, y0 - yc, y1 - yc)
    # the source and the in-band part of its trace, without the frame padding
    xlim = (max(x0 - xc, min(bx0 + in_band.min() - xc, 0) - 70),
            min(x1 - xc, max(bx0 + in_band.max() - xc, 0) + 70))
    img_h = 11.6 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig = plt.figure(figsize=(13, 2 * img_h + 6.6))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[img_h, img_h, 1.05, 3.6],
                  hspace=0.5)

    # -- the image, and the grism exposure dispersed from it --
    a = fig.add_subplot(gs[0])
    a.imshow(c[p + "region_direct"], extent=ext, aspect="equal",
             **_stretch(c[p + "region_direct"], 20, 99.6))
    a.add_patch(plt.Circle((0, 0), 16, fill=False, color=COLOR_SR, lw=1.8))
    a.set_title("OpenUniverse2024 H158 image", fontsize=12, loc="left")
    b = fig.add_subplot(gs[1], sharex=a, sharey=a)
    b.imshow(c[p + "region_grism"], extent=ext, aspect="equal",
             **_stretch(c[p + "region_grism"], 20, 99.6))
    b.add_patch(plt.Rectangle((bx0 + in_band.min() - xc, by0 + r0 - yc),
                              np.ptp(in_band), r1 - r0, fill=False,
                              color=COLOR_SR, lw=1.8))
    b.set_title("Grism exposure dispersed from it, with single-exposure noise",
                fontsize=12, loc="left")
    b.set_xlabel("x (pixels from the target)")
    plt.setp(a.get_xticklabels(), visible=False)
    for ax in (a, b):
        ax.set_ylabel("y (pixels)")
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)

    # -- the two-dimensional spectrum of the target --
    order = np.argsort(lam)
    two_d = c[p + "beam_clean"][r0:r1][:, order]
    lam = lam[order]
    d = fig.add_subplot(gs[2])
    d.imshow(two_d, extent=(lam[0], lam[-1], r0 - trace, r1 - trace), aspect="auto",
             **_stretch(two_d, 5, 99.8))
    d.set_xlim(wl_hr[0], wl_hr[-1])
    d.set_ylabel("y (pixels)")
    d.set_title("Two-dimensional spectrum of the target, contamination "
                "subtracted", fontsize=12, loc="left")
    plt.setp(d.get_xticklabels(), visible=False)

    # -- the training pair --
    e = fig.add_subplot(gs[3], sharex=d)
    ok = np.isfinite(lr)
    e.fill_between(wl_lr[ok], (lr - lr_err)[ok], (lr + lr_err)[ok],
                   color=COLOR_LR, alpha=0.22, lw=0)
    e.plot(wl_lr[ok], lr[ok], color=COLOR_LR, lw=1.1,
           label="Extracted grism spectrum (input)")
    e.plot(wl_hr, hr, color=COLOR_HR, lw=1.0,
           label="OpenUniverse2024 spectrum (target)")
    top = max(np.nanpercentile(lr[ok], 99.9), 0.5 * np.nanmax(hr)) * 1.3
    e.set_ylim(min(np.nanpercentile((lr - lr_err)[ok], 1), 0.0) - 0.1, top)
    for lab, rest in MAIN_LABELS:
        lo = rest * (1 + z)
        if wl_hr[0] + 0.01 < lo < wl_hr[-1] - 0.01:
            e.axvline(lo, color="0.75", lw=0.8, ls="--", zorder=0)
            e.text(lo, 0.97, lab, rotation=90, ha="right", va="top", fontsize=9.5,
                   color="0.3", transform=e.get_xaxis_transform())
    e.set_xlabel(r"Observed wavelength ($\mu$m)")
    e.set_ylabel("Flux (normalized)")
    e.legend(frameon=False, fontsize=10, loc="upper right",
             bbox_to_anchor=(1.0, 0.99))
    e.text(0.992, 0.05, rf"$z={z:.3f}$,  $m_{{\rm H158}}={ab:.1f}$",
           transform=e.transAxes, ha="right", fontsize=10.5,
           bbox=dict(fc="w", ec="0.6", pad=3))
    return _save(fig, name)


# ---------------------------------------------------------------------------
# The parent-sample figure
# ---------------------------------------------------------------------------
def plot_parent_sample(dataset: str = "data/dataset/ou2024_h10307_dataset.npz",
                       data_dir: str = "data/ou2024", healpix: int = 10307,
                       ab_target: float = 22.5, n_parent: int = 60000,
                       name: str = "ou2024_parent.png", seed: int = 0):
    import pyarrow.parquet as pq

    gal = pq.read_table(f"{data_dir}/galaxy_{healpix}.parquet",
                        columns=["galaxy_id", "redshift",
                                 "um_source_galaxy_obs_sm"])
    flx = pq.read_table(f"{data_dir}/galaxy_flux_{healpix}.parquet",
                        columns=["galaxy_id", "roman_flux_H158"])
    gid = np.asarray(gal["galaxy_id"])
    if not np.array_equal(gid, np.asarray(flx["galaxy_id"])):
        raise ValueError("galaxy and flux catalogues are not row-aligned")
    z_p = np.asarray(gal["redshift"])
    m_p = AB_ANCHOR_H158 - 2.5 * np.log10(
        np.clip(np.asarray(flx["roman_flux_H158"]), 1e-12, None))
    sm_p = np.log10(np.clip(np.asarray(gal["um_source_galaxy_obs_sm"]), 1.0, None))

    d = np.load(dataset, allow_pickle=True)
    ids = np.unique(d["ids"])
    row = np.searchsorted(gid, ids) if np.all(np.diff(gid) > 0) else None
    if row is None:
        lut = {int(g): j for j, g in enumerate(gid)}
        row = np.array([lut[int(i)] for i in ids])
    z_s, m_s, sm_s = z_p[row], m_p[row], sm_p[row]

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(gid), size=min(n_parent, len(gid)), replace=False)
    bright = m_p < ab_target
    sub_b = rng.choice(np.where(bright)[0],
                       size=min(n_parent, int(bright.sum())), replace=False)

    col_all, col_cut = "0.72", "0.25"
    fig = plt.figure(figsize=(13, 5.4))
    outer = GridSpec(1, 2, figure=fig, wspace=0.22)

    # -- (a) redshift against magnitude, with marginals --
    ga = outer[0].subgridspec(2, 2, height_ratios=[1, 3.6], width_ratios=[3.6, 1],
                              hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(ga[1, 0])
    at = fig.add_subplot(ga[0, 0], sharex=ax)
    ar = fig.add_subplot(ga[1, 1], sharey=ax)
    ax.scatter(z_p[sub], m_p[sub], s=1.5, color=col_all, lw=0, rasterized=True)
    ax.scatter(z_s, m_s, s=2.5, color=COLOR_SR, lw=0, alpha=0.6, rasterized=True)
    ax.axhline(ab_target, color="k", lw=0.9, ls="--")
    ax.set_xlim(0, 3.1)
    ax.set_ylim(28.5, 15.5)
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"$m_{\rm H158}$ (AB)")
    zb, mb = np.linspace(0, 3.1, 48), np.linspace(15.5, 28.5, 53)
    step = dict(histtype="step", density=True, lw=1.3)
    at.hist(z_p, bins=zb, color=col_all, **step)
    at.hist(z_p[bright], bins=zb, color=col_cut, **step)
    at.hist(z_s, bins=zb, color=COLOR_SR, **step)
    ar.hist(m_p, bins=mb, color=col_all, orientation="horizontal", **step)
    ar.hist(m_s, bins=mb, color=COLOR_SR, orientation="horizontal", **step)
    for h in (at, ar):
        h.axis("off")
    at.set_title("(a) redshift and apparent magnitude", fontsize=12)
    handles = [
        plt.Line2D([], [], color=col_all, lw=2,
                   label=f"OpenUniverse2024 galaxies ({len(gid):,})"),
        plt.Line2D([], [], color=col_cut, lw=2,
                   label=rf"of which $m_{{\rm H158}}<{ab_target}$ "
                         f"({int(bright.sum()):,})"),
        plt.Line2D([], [], color=COLOR_SR, lw=2,
                   label=f"dataset used here ({len(ids):,})"),
    ]
    ax.legend(handles=handles, frameon=True, framealpha=0.9, edgecolor="none",
              fontsize=9, loc="lower right")

    # -- (b) stellar mass against redshift --
    bx = fig.add_subplot(outer[1])
    bx.scatter(z_p[sub_b], sm_p[sub_b], s=1.5, color=col_cut, lw=0, alpha=0.35,
               rasterized=True)
    bx.scatter(z_s, sm_s, s=2.5, color=COLOR_SR, lw=0, alpha=0.6, rasterized=True)
    bx.set_xlim(0, 3.1)
    bx.set_ylim(7.5, 12.2)
    bx.set_xlabel(r"Redshift $z$")
    bx.set_ylabel(r"$\log_{10}(M_\star/M_\odot)$")
    bx.set_title("(b) stellar mass", fontsize=12)
    path = _save(fig, name)
    stats = {
        "n_parent": int(len(gid)), "n_parent_bright": int(bright.sum()),
        "n_sample": int(len(ids)),
        "z_median_sample": float(np.median(z_s)),
        "z_median_bright": float(np.median(z_p[bright])),
        "mag_median_sample": float(np.median(m_s)),
        "mag_p05_p95_sample": [float(v) for v in np.percentile(m_s, [5, 95])],
        "logm_median_sample": float(np.median(sm_s)),
        "logm_p05_p95_sample": [float(v) for v in np.percentile(sm_s, [5, 95])],
        "logm_median_bright": float(np.median(sm_p[bright])),
    }
    # read by paper/make_paper_numbers.py for the macros of the data section
    import json
    with open(PARENT_STATS, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"wrote {PARENT_STATS}")
    return path, stats


def make_dataset_figures(outdir: str = "outputs/figures",
                         which: int = EXAMPLE_ID,
                         cache: str = EXAMPLE_CACHE, rebuild: bool = False,
                         dataset: str = "data/dataset/ou2024_h10307_dataset.npz"):
    """Render both dataset figures, building the example cache if needed."""
    from . import figures as _figs
    previous, _figs.OUTDIR = _figs.OUTDIR, outdir
    try:
        if rebuild or not os.path.exists(cache):
            build_example_cache(out=cache)
        with FigureStyle():
            plot_example(cache, which=which)
            plot_parent_sample(dataset)
    finally:
        _figs.OUTDIR = previous
