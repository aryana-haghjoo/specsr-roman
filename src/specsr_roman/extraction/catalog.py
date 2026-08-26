"""Truth catalogues, source detection, and the match between them.

Every extracted spectrum has to map back to the galaxy that produced it, or it
has no ground truth and is useless for training. That mapping is a position
match between grizli's segmentation and the simulation's per-image truth
index.
"""

from __future__ import annotations

import numpy as np

from ..grids import AB_ANCHOR_H158, OU2024_ZPTMAG

__all__ = ["ab_h158", "load_truth_index", "detect_and_relabel"]


def ab_h158(f_cat) -> np.ndarray:
    """OU2024 catalogue H158 flux -> AB magnitude.

    The images carry ``counts = f_cat * 10^(0.4 * ZPTMAG)``, which fixes the
    anchor at 14.96. Use this rather than the truth index's own ``mag``
    column, which is instrumental (``-2.5 log10(flux) + const``) and only good
    for ranking.
    """
    return AB_ANCHOR_H158 - 2.5 * np.log10(np.clip(f_cat, 1e-12, None))


def load_truth_index(path: str, zptmag: float = OU2024_ZPTMAG):
    """Per-image truth index -> ``(ids, ra, dec, x, y, mag)`` for galaxies only.

    Columns are ``object_id ra dec x y realized_flux flux mag obj_type``.
    Stars and transients are dropped: they have no SED entry in the galaxy
    catalogue, so they can shape the contamination scene but never be targets.
    """
    ids, ra, dec, x, y, mag = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if p[8] != "galaxy":
                continue
            ids.append(int(p[0]))
            ra.append(float(p[1]))
            dec.append(float(p[2]))
            x.append(float(p[3]))
            y.append(float(p[4]))
            mag.append(float(p[7]) + zptmag)
    return (np.array(ids), np.array(ra), np.array(dec),
            np.array(x), np.array(y), np.array(mag))


def detect_and_relabel(flt, truth, match_radius: float = 3.0,
                       threshold_sigma: float = 2.5, npixels: int = 8,
                       verbose: bool = True):
    """Detect sources and relabel the segmentation with compact ids.

    Returns ``(compact_ids, object_ids, mags, matched)``, all indexed by
    detection.

    Two things are load-bearing here.

    *Compact ids.* grizli keeps the segmentation map as float32, which cannot
    represent OU2024's 13-digit ``object_id`` exactly --- ids silently collide
    or shift. The segmentation therefore carries a small sequential id
    (detection index + 2) and ``object_ids`` maps back. ``-1`` marks a
    detection with no truth counterpart; those still shape the contamination
    scene, they just cannot be targets.

    *Detection runs on padded arrays.* grizli pads the frame so edge sources'
    beams stay in-array, so truth coordinates must be shifted by the pad
    before matching or every match is wrong by a few hundred pixels.
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources
    from scipy.spatial import cKDTree

    ids_t, _, _, x_t, y_t, mag_t = truth
    sci = flt.direct.data["SCI"]
    _, _, std = sigma_clipped_stats(sci[::4, ::4], sigma=3.0)
    segm = detect_sources(sci, threshold_sigma * std, npixels=npixels)
    segm = deblend_sources(sci, segm, npixels=npixels, progress_bar=False)
    scat = SourceCatalog(sci, segm)

    pady, padx = flt.pad
    tree = cKDTree(np.c_[x_t + padx, y_t + pady])
    dist, idx = tree.query(np.c_[scat.xcentroid, scat.ycentroid], k=1)
    matched = dist < match_radius

    relabel = np.zeros(segm.max_label + 1, dtype=np.int64)
    compact_ids = np.arange(len(scat), dtype=np.int64) + 2
    object_ids = np.where(matched, ids_t[idx], -1)
    mags = np.where(matched, mag_t[idx], 99.0)
    for i, lbl in enumerate(scat.labels):
        relabel[lbl] = compact_ids[i]
    flt.seg = relabel[segm.data].astype(np.float32)

    if verbose:
        print(f"{len(scat)} detections, {matched.sum()} matched "
              f"(median |dr| = {np.median(dist[matched]):.2f} px)")
    return compact_ids, object_ids, mags, matched
