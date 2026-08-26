"""Resumable batch driver: OU2024 products -> a training dataset.

Design constraints that shaped this file:

* **grizli's numba disperser corrupts the heap** when thousands of sources are
  dispersed in one process. Each (visit, SCA) therefore runs in its own
  subprocess, and a crash costs one SCA rather than the run.
* **Everything is resumable.** Each SCA writes a cached npz --- including a
  stub recording *why* it produced nothing --- so re-running skips completed
  work and never silently retries a pointing that is outside the healpix.
* **The same galaxy appears in many visits.** Rows carry ``ids``, and the
  merge step ends by reminding you to split on them.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

from ..grids import GRISM_EXPTIME, GRIZLI_PAD, PHOT_BANDS, WAVE_HR, WAVE_LR
from .catalog import ab_h158, detect_and_relabel, load_truth_index
from .download import candidate_scas, fetch, sca_paths
from .extract import ExtractionFailure, extract_target, to_fixed_grids
from .frames import prepare_frames
from .seds import SEDLibrary
from .simulate import add_grism_noise, disperse_scene

__all__ = ["ExtractionConfig", "run_worker", "run_batch", "merge"]


@dataclass
class ExtractionConfig:
    """Where the inputs are, and what counts as a target."""

    healpix: int = 10307
    data_dir: str = "data/ou2024"
    out_dataset: str = "data/dataset/ou2024_h10307_dataset.npz"

    max_scas: int = 60
    workers: int = 4
    ab_target: float = 22.5      # extract galaxies brighter than this
    ab_scene: float = 23.0       # disperse (as contamination) brighter than this
    min_hp_frac: float = 0.7     # skip pointings mostly outside the healpix
    grism_exptime: float = GRISM_EXPTIME
    cleanup: bool = False        # delete raw/prepared FITS after each SCA
    z_max: float = 3.2

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def prepared_dir(self) -> str:
        return os.path.join(self.data_dir, "prepared")

    @property
    def batch_dir(self) -> str:
        return os.path.join(self.data_dir, "batch")

    @property
    def galaxy_parquet(self) -> str:
        return os.path.join(self.data_dir, f"galaxy_{self.healpix}.parquet")

    @property
    def flux_parquet(self) -> str:
        return os.path.join(self.data_dir, f"galaxy_flux_{self.healpix}.parquet")

    @property
    def sed_hdf5(self) -> str:
        return os.path.join(self.data_dir, f"galaxy_sed_{self.healpix}.hdf5")

    @property
    def obseq_fits(self) -> str:
        return os.path.join(self.data_dir, "Roman_WAS_obseq_11_1_23.fits")


def _load_catalogues(cfg: ExtractionConfig):
    import pyarrow.parquet as pq
    gal = pq.read_table(cfg.galaxy_parquet, columns=["galaxy_id", "redshift"])
    redshifts = dict(zip(np.asarray(gal["galaxy_id"]).tolist(),
                         np.asarray(gal["redshift"]).tolist(), strict=True))
    flux_t = pq.read_table(cfg.flux_parquet)
    flux_row = {int(i): j for j, i in enumerate(np.asarray(flux_t["galaxy_id"]))}
    flux_cols = {b: np.asarray(flux_t[b]) for b in PHOT_BANDS}
    return redshifts, flux_row, flux_cols


def run_worker(visit: int, sca: int, cfg: ExtractionConfig) -> None:
    """Extract every target on one (visit, SCA) and cache the rows."""
    from grizli import model as gmodel

    os.makedirs(cfg.batch_dir, exist_ok=True)
    out_npz = f"{cfg.batch_dir}/v{visit}_s{sca}.npz"
    if os.path.exists(out_npz):
        print(f"[{visit}/{sca}] cached")
        return

    img, idx, img_url, idx_url = sca_paths(visit, sca, cfg.raw_dir)
    if not (fetch(idx_url, idx) and fetch(img_url, img)):
        np.savez(out_npz, empty=True, reason="download")
        print(f"[{visit}/{sca}] download failed")
        return

    truth = load_truth_index(idx)
    frac = float(np.mean(truth[0] // 10 ** 9 == cfg.healpix))
    if frac < cfg.min_hp_frac:
        np.savez(out_npz, empty=True, reason=f"hp_frac={frac:.2f}")
        print(f"[{visit}/{sca}] outside healpix ({frac:.2f})")
        return

    redshifts, flux_row, flux_cols = _load_catalogues(cfg)
    seds = SEDLibrary(cfg.sed_hdf5)

    direct, grism = prepare_frames(img, cfg.prepared_dir, cfg.grism_exptime)
    flt = gmodel.GrismFLT(grism_file=grism, direct_file=direct,
                          pad=GRIZLI_PAD, verbose=False)
    compact_ids, object_ids, _, matched = detect_and_relabel(flt, truth)

    # Select on the CATALOGUE H158 flux via the AB anchor, not on the truth
    # index's instrumental magnitudes.
    f_h = np.array([flux_cols["roman_flux_H158"][flux_row[o]] if o in flux_row
                    else 0.0 for o in object_ids])
    ab = ab_h158(f_h)
    zs = np.array([redshifts.get(int(o), -1.0) for o in object_ids])
    scene_sel = matched & (ab < cfg.ab_scene) & (zs >= 0)
    target_sel = scene_sel & (ab < cfg.ab_target) & (zs < cfg.z_max)
    order = np.argsort(ab)
    scene_idx = [i for i in order if scene_sel[i]]
    print(f"[{visit}/{sca}] scene {len(scene_idx)}, targets {int(target_sel.sum())}")

    kept = disperse_scene(flt, compact_ids, object_ids, ab, redshifts, seds,
                          scene_idx)
    scene = flt.model.astype(np.float64)
    noisy, err2d = add_grism_noise(scene, exptime=cfg.grism_exptime,
                                   seed=int(visit) * 100 + int(sca))

    rows = []
    for i in np.where(target_sel)[0]:
        if i not in kept:
            continue
        spec, (sed_wave, sed_flux) = kept[i]
        oid, z = int(object_ids[i]), float(zs[i])
        try:
            wave, flam, flam_err = extract_target(
                flt, scene, noisy, err2d, int(compact_ids[i]), float(ab[i]), spec)
            lr, lr_err, hr = to_fixed_grids(wave, flam, flam_err,
                                            sed_wave, sed_flux)
        except ExtractionFailure:
            continue
        snr = float(np.nanmedian(flam / flam_err))
        phot = np.array([flux_cols[b][flux_row[oid]] for b in PHOT_BANDS])
        rows.append((oid, z, lr, lr_err, hr, phot, float(ab[i]), snr))

    seds.close()
    if not rows:
        np.savez(out_npz, empty=True, reason="no_rows")
        print(f"[{visit}/{sca}] no rows")
        return

    oid_a, z_a, lr_a, le_a, hr_a, ph_a, ab_a, sn_a = zip(*rows, strict=True)
    np.savez_compressed(
        out_npz,
        ids=np.array(oid_a, dtype=np.int64), redshift=np.array(z_a),
        flux_low=np.array(lr_a, dtype=np.float32),
        flux_low_err=np.array(le_a, dtype=np.float32),
        flux_high=np.array(hr_a, dtype=np.float32),
        phot=np.array(ph_a, dtype=np.float64), ab_h158=np.array(ab_a),
        snr=np.array(sn_a), visit=np.full(len(rows), visit, dtype=np.int32),
        sca=np.full(len(rows), sca, dtype=np.int16),
        wavelength_low=WAVE_LR, wavelength_high=WAVE_HR,
        phot_bands=np.array(PHOT_BANDS))

    if cfg.cleanup:
        for p in (img, direct, grism):
            if os.path.exists(p):
                os.remove(p)
    print(f"[{visit}/{sca}] wrote {len(rows)} rows -> {out_npz}", flush=True)


def run_batch(cfg: ExtractionConfig) -> None:
    """Drive workers over candidate SCAs until ``max_scas`` produce rows."""
    os.makedirs(cfg.batch_dir, exist_ok=True)
    pairs, box = candidate_scas(cfg.galaxy_parquet, cfg.obseq_fits, cfg.max_scas)
    print(f"healpix box {box}; {len(pairs)} candidate (visit, sca) pairs")

    def productive(path: str) -> bool:
        return (os.path.exists(path)
                and "empty" not in np.load(path, allow_pickle=True))

    active, done = [], 0
    for visit, sca in pairs:
        if done >= cfg.max_scas:
            break
        out_npz = f"{cfg.batch_dir}/v{visit}_s{sca}.npz"
        if os.path.exists(out_npz):
            done += int(productive(out_npz))
            continue
        cmd = [sys.executable, "-m", "specsr_roman.cli", "extract", "worker",
               "--visit", str(visit), "--sca", str(sca),
               "--data-dir", cfg.data_dir, "--healpix", str(cfg.healpix),
               "--ab-target", str(cfg.ab_target), "--ab-scene", str(cfg.ab_scene),
               "--min-hp-frac", str(cfg.min_hp_frac),
               "--grism-exptime", str(cfg.grism_exptime)]
        if cfg.cleanup:
            cmd.append("--cleanup")
        active.append((visit, sca, subprocess.Popen(cmd)))
        while len(active) >= cfg.workers:
            time.sleep(5)
            still = []
            for v, s, proc in active:
                if proc.poll() is None:
                    still.append((v, s, proc))
                else:
                    done += int(productive(f"{cfg.batch_dir}/v{v}_s{s}.npz"))
            active = still
    for _, _, proc in active:
        proc.wait()
    print(f"driver finished: {done} productive SCAs")


def merge(cfg: ExtractionConfig) -> str:
    """Concatenate the per-SCA caches into one training dataset."""
    files = sorted(glob.glob(f"{cfg.batch_dir}/v*_s*.npz"))
    keys = ["ids", "redshift", "flux_low", "flux_low_err", "flux_high",
            "phot", "ab_h158", "snr", "visit", "sca"]
    acc: dict[str, list] = {k: [] for k in keys}
    bands = wave_lo = wave_hi = None
    n_files = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "empty" in d:
            continue
        for k in keys:
            acc[k].append(d[k])
        bands, wave_lo, wave_hi = (d["phot_bands"], d["wavelength_low"],
                                   d["wavelength_high"])
        n_files += 1
    if not n_files:
        raise RuntimeError(f"no non-empty caches in {cfg.batch_dir}")

    merged = {k: np.concatenate(v) for k, v in acc.items()}
    # Guard against the descending-wavelength interpolation bug: a spectrum
    # with no variation carries no information and must never reach training.
    lr_std = np.nanstd(merged["flux_low"], axis=1)
    const_frac = float(((lr_std < 1e-8) | ~np.isfinite(lr_std)).mean())
    print(f"flux_low constant-row fraction: {const_frac:.4f}")
    if const_frac > 0.01:
        raise RuntimeError(
            f"{const_frac:.1%} of flux_low rows are constant -- the extraction is "
            "broken (descending-wavelength interp?); refusing to merge")

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_dataset)), exist_ok=True)
    np.savez_compressed(cfg.out_dataset, **merged, phot_bands=bands,
                        wavelength_low=wave_lo, wavelength_high=wave_hi)
    n, uniq = len(merged["ids"]), len(np.unique(merged["ids"]))
    print(f"merged {n} rows ({uniq} unique galaxies) from {n_files} files "
          f"-> {cfg.out_dataset}")
    print("REMINDER: split by object_id, not by row -- the same galaxy appears "
          "in many visits.")
    return cfg.out_dataset
