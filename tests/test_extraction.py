"""Extraction primitives.

grizli itself is not exercised here (it needs real detector configs and
gigabytes of simulation products). What *is* exercised is everything around
it: the frame preparation, catalogue parsing, SED handling, noise model, grid
resampling and the merge step — which is where every bug in this pipeline's
history actually lived.
"""

from __future__ import annotations

import numpy as np
import pytest

from specsr_roman.extraction import (
    ExtractionConfig,
    ab_h158,
    add_grism_noise,
    load_truth_index,
    merge,
    prepare_frames,
)
from specsr_roman.extraction.extract import ExtractionFailure, to_fixed_grids
from specsr_roman.grids import (
    AB_ANCHOR_H158,
    GRISM_BKG,
    N_HR,
    N_LR,
    PHOT_BANDS,
    PHOTFLAM_F158,
    READ_NOISE,
    WAVE_HR,
    WAVE_LR,
)

fits = pytest.importorskip("astropy.io.fits", reason="needs astropy")


# --- catalogue -----------------------------------------------------------
def test_ab_h158_matches_the_documented_anchor():
    # counts = f_cat * 10^(0.4 * ZPTMAG)  ->  AB = 14.96 - 2.5 log10(f_cat)
    assert ab_h158(1.0) == pytest.approx(AB_ANCHOR_H158)
    assert ab_h158(10.0) == pytest.approx(AB_ANCHOR_H158 - 2.5)
    # Non-positive fluxes are clipped rather than producing -inf, so one bad
    # catalogue row cannot poison a whole SCA's selection.
    assert np.isfinite(ab_h158(0.0))
    assert np.isfinite(ab_h158(-1.0))


def test_load_truth_index_keeps_galaxies_only(tmp_path):
    """Stars have no SED entry: they may shape the scene but never be targets."""
    p = tmp_path / "index.txt"
    p.write_text(
        "# object_id ra dec x y realized_flux flux mag obj_type\n"
        "10307000000001 10.0 -5.0 100.0 200.0 1e6 1e6 -3.0 galaxy\n"
        "10307000000002 10.1 -5.1 300.0 400.0 1e5 1e5 -1.0 star\n"
        "10307000000003 10.2 -5.2 500.0 600.0 1e4 1e4  0.5 galaxy\n")
    ids, ra, dec, x, y, mag = load_truth_index(str(p), zptmag=16.8009)
    assert list(ids) == [10307000000001, 10307000000003]
    assert x[0] == 100.0 and y[0] == 200.0
    # The index magnitude is instrumental; the zeropoint is added on read.
    assert mag[0] == pytest.approx(-3.0 + 16.8009)


# --- frames --------------------------------------------------------------
def _fake_ou2024_image(path, exptime=139.8, sky=12.0, shape=(64, 64)):
    rng = np.random.default_rng(0)
    sci = (rng.normal(sky, 1.0, shape)).astype(np.float32)
    err = np.full(shape, 1.0, dtype=np.float32)
    hsci = fits.Header()
    hsci["EXPTIME"] = exptime
    hsci["SKY_MEAN"] = sky
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({"EXPTIME": exptime})),
        fits.ImageHDU(data=sci, header=hsci, name="SCI"),
        fits.ImageHDU(data=err, header=hsci, name="ERR"),
        fits.ImageHDU(data=np.zeros(shape, dtype=np.int16), name="DQ"),
    ])
    hdul.writeto(path, overwrite=True)
    return sci, err


def test_prepare_frames_converts_to_rate_and_tags_for_grizli(tmp_path):
    img = tmp_path / "Roman_WAS_simple_model_H158_1_1.fits"
    sci, err = _fake_ou2024_image(img, exptime=139.8, sky=12.0)
    direct, grism = prepare_frames(str(img), str(tmp_path / "prep"))

    with fits.open(direct) as h:
        # counts -> e-/s with the flat sky removed
        assert np.allclose(h["SCI"].data, (sci - 12.0) / 139.8, atol=1e-5)
        assert np.allclose(h["ERR"].data, err / 139.8, atol=1e-6)
        # INSTRUME drives grizli's choice of Roman.G150.conf
        assert h[0].header["INSTRUME"] == "WFI"
        assert h["SCI"].header["PHOTFLAM"] == PHOTFLAM_F158
        assert h["SCI"].header["BUNIT"] == "ELECTRONS/S"

    with fits.open(grism) as h:
        assert h[0].header["FILTER"] == "G150"
        # The grism frame is a shell: the scene is dispersed into it later.
        assert not np.any(h["SCI"].data)
        # Only the direct frame carries a photometric calibration; setting it
        # on the grism would make grizli rescale the dispersed scene.
        assert "PHOTFLAM" not in h["SCI"].header


def test_prepare_frames_is_idempotent(tmp_path):
    """The batch driver is resumable, so re-preparing must not redo work."""
    img = tmp_path / "Roman_WAS_simple_model_H158_1_1.fits"
    _fake_ou2024_image(img)
    a = prepare_frames(str(img), str(tmp_path / "prep"))
    mtime = [__import__("os").path.getmtime(p) for p in a]
    b = prepare_frames(str(img), str(tmp_path / "prep"))
    assert a == b
    assert [__import__("os").path.getmtime(p) for p in b] == mtime


# --- noise ---------------------------------------------------------------
def test_grism_noise_matches_the_stated_variance_model():
    scene = np.full((128, 128), 0.5)          # e-/s
    t = 301.0
    noisy, err = add_grism_noise(scene, exptime=t, seed=1)
    expected = np.sqrt((0.5 + GRISM_BKG) / t + (READ_NOISE / t) ** 2)
    assert err.mean() == pytest.approx(expected, rel=1e-4)
    # Unbiased: the noise adds nothing on average.
    assert (noisy - scene).mean() == pytest.approx(0.0, abs=5 * err.mean() / 128)


def test_grism_noise_is_reproducible_and_seed_dependent():
    scene = np.full((32, 32), 1.0)
    a, _ = add_grism_noise(scene, seed=7)
    b, _ = add_grism_noise(scene, seed=7)
    c, _ = add_grism_noise(scene, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_negative_scene_pixels_do_not_produce_nan_errors():
    """Contamination subtraction can drive pixels negative; sqrt must survive."""
    scene = np.full((16, 16), -5.0)
    noisy, err = add_grism_noise(scene)
    assert np.isfinite(err).all() and (err > 0).all()
    assert np.isfinite(noisy).all()


# --- gridding ------------------------------------------------------------
def _synthetic_extraction(descending=False):
    wave = np.linspace(10050, 19250, 900)
    flam = 1.0 + 5 * np.exp(-0.5 * ((wave - 15000) / 20.0) ** 2)
    err = np.full_like(flam, 0.1)
    if descending:
        wave, flam, err = wave[::-1], flam[::-1], err[::-1]
    return wave, flam, err


def test_to_fixed_grids_shapes_and_line_survival():
    wave, flam, err = _synthetic_extraction()
    sed_w = np.linspace(9000, 20000, 40000)
    sed_f = 1.0 + 50 * np.exp(-0.5 * ((sed_w - 15000) / 2.0) ** 2)
    lr, lr_err, hr = to_fixed_grids(wave, flam, err, sed_w, sed_f)
    assert lr.shape == lr_err.shape == (N_LR,)
    assert hr.shape == (N_HR,)
    assert hr.max() > 5.0            # the narrow line survived the rebin


def test_to_fixed_grids_handles_descending_wavelengths():
    """Roman's DLDP_A_1 is negative — this is the bug that cost a dataset.

    np.interp on a descending grid silently returns the edge value everywhere,
    producing a flat spectrum that trains a model into the prior mean.
    """
    sed_w = np.linspace(9000, 20000, 20000)
    sed_f = np.ones_like(sed_w)
    asc = to_fixed_grids(*_synthetic_extraction(False), sed_w, sed_f)
    desc = to_fixed_grids(*_synthetic_extraction(True), sed_w, sed_f)
    assert np.allclose(asc[0], desc[0], equal_nan=True)
    assert np.nanstd(desc[0]) > 0     # not collapsed to a constant


def test_to_fixed_grids_rejects_a_constant_spectrum():
    wave = np.linspace(10050, 19250, 900)
    flat = np.ones_like(wave)
    sed_w = np.linspace(9000, 20000, 20000)
    with pytest.raises(ExtractionFailure):
        to_fixed_grids(wave, flat, np.full_like(wave, 0.1), sed_w,
                       np.ones_like(sed_w))


# --- merge ---------------------------------------------------------------
def _write_sca_cache(path, n=5, seed=0, constant=False):
    rng = np.random.default_rng(seed)
    lo = (np.ones((n, N_LR)) if constant
          else rng.normal(1.0, 0.1, (n, N_LR))).astype(np.float32)
    np.savez_compressed(
        path,
        ids=np.arange(n, dtype=np.int64) + seed * 100,
        redshift=rng.uniform(0.5, 2.0, n),
        flux_low=lo,
        flux_low_err=np.full((n, N_LR), 0.1, dtype=np.float32),
        flux_high=rng.normal(1.0, 0.1, (n, N_HR)).astype(np.float32),
        phot=rng.uniform(0.1, 1.0, (n, 14)),
        ab_h158=rng.uniform(19, 22.5, n),
        snr=rng.uniform(1, 5, n),
        visit=np.full(n, seed, dtype=np.int32),
        sca=np.full(n, 1, dtype=np.int16),
        wavelength_low=WAVE_LR, wavelength_high=WAVE_HR,
        phot_bands=np.array(PHOT_BANDS))


def test_merge_concatenates_and_skips_empty_caches(tmp_path):
    cfg = ExtractionConfig(data_dir=str(tmp_path),
                           out_dataset=str(tmp_path / "merged.npz"))
    import os
    os.makedirs(cfg.batch_dir, exist_ok=True)
    _write_sca_cache(f"{cfg.batch_dir}/v1_s1.npz", n=5, seed=1)
    _write_sca_cache(f"{cfg.batch_dir}/v2_s1.npz", n=3, seed=2)
    # a pointing that produced nothing records why, and must be skipped
    np.savez(f"{cfg.batch_dir}/v3_s1.npz", empty=True, reason="hp_frac=0.10")

    merge(cfg)
    d = np.load(cfg.out_dataset, allow_pickle=True)
    assert len(d["ids"]) == 8
    assert d["flux_low"].shape == (8, N_LR)
    assert list(d["phot_bands"]) == list(PHOT_BANDS)
    assert np.allclose(d["wavelength_low"], WAVE_LR)


def test_merge_refuses_a_broken_extraction(tmp_path):
    """Constant flux_low means the descending-wavelength bug bit again."""
    cfg = ExtractionConfig(data_dir=str(tmp_path),
                           out_dataset=str(tmp_path / "merged.npz"))
    import os
    os.makedirs(cfg.batch_dir, exist_ok=True)
    _write_sca_cache(f"{cfg.batch_dir}/v1_s1.npz", n=5, seed=1, constant=True)
    with pytest.raises(RuntimeError, match="constant"):
        merge(cfg)


def test_merge_errors_clearly_when_there_is_nothing_to_merge(tmp_path):
    cfg = ExtractionConfig(data_dir=str(tmp_path),
                           out_dataset=str(tmp_path / "merged.npz"))
    import os
    os.makedirs(cfg.batch_dir, exist_ok=True)
    with pytest.raises(RuntimeError, match="no non-empty caches"):
        merge(cfg)


def test_extraction_config_paths_are_consistent():
    cfg = ExtractionConfig(healpix=10307, data_dir="somewhere")
    assert cfg.raw_dir.endswith("somewhere/raw")
    assert cfg.batch_dir.endswith("somewhere/batch")
    assert cfg.galaxy_parquet.endswith("galaxy_10307.parquet")
    assert cfg.flux_parquet.endswith("galaxy_flux_10307.parquet")
    assert cfg.sed_hdf5.endswith("galaxy_sed_10307.hdf5")


# --- SEDs ----------------------------------------------------------------
def _fake_sed_file(path, gid=10307000000001, n_comp=3):
    """A Diffsky-shaped HDF5 with an ADAPTIVE wavelength grid.

    Sub-Angstrom sampling at the line, coarse elsewhere — the property that
    makes flux-conserving rebinning mandatory.
    """
    h5py = pytest.importorskip("h5py")
    coarse = np.arange(1000.0, 12000.0, 25.0)
    fine = np.arange(4990.0, 5020.0, 0.05)          # around [OIII] 5007
    wave = np.unique(np.concatenate([coarse, fine]))
    line = 100.0 * np.exp(-0.5 * ((wave - 5007.0) / 1.0) ** 2)
    comps = np.stack([np.full_like(wave, 1e-24) + line * 1e-24 / n_comp
                      for _ in range(n_comp)])
    with h5py.File(path, "w") as f:
        f.create_dataset("meta/wave_list", data=wave)
        f.create_dataset(f"galaxy/{gid // 100000}/{gid}", data=comps)
    return wave, comps


def test_sed_library_sums_components_and_redshifts(tmp_path):
    from specsr_roman.extraction import SEDLibrary
    gid = 10307000000001
    wave, comps = _fake_sed_file(tmp_path / "sed.h5", gid)
    lib = SEDLibrary(str(tmp_path / "sed.h5"))
    z = 1.5
    w_obs, flam = lib.observed(gid, z)
    # Observed grid is uniform and starts where the loader says it does.
    assert np.allclose(np.diff(w_obs), 5.0)
    assert w_obs[0] == pytest.approx(2500.0)
    # The line moved to (1+z) * 5007.
    peak = w_obs[np.argmax(flam)]
    assert peak == pytest.approx(5007.0 * (1 + z), rel=2e-3)
    lib.close()


def test_sed_library_conserves_line_flux_off_the_adaptive_grid(tmp_path):
    """The reason `observed` rebins rather than interpolates."""
    from specsr_roman.extraction import SEDLibrary
    gid = 10307000000001
    wave, comps = _fake_sed_file(tmp_path / "sed.h5", gid)
    lib = SEDLibrary(str(tmp_path / "sed.h5"))
    z = 0.0
    w_obs, flam = lib.observed(gid, z)

    total = comps.sum(axis=0)
    cont = 1e-24 * comps.shape[0]
    m_src = (wave > 4990) & (wave < 5025)
    m_out = (w_obs > 4990) & (w_obs < 5025)
    src_line = np.trapezoid(total[m_src] - cont, wave[m_src])
    out_line = np.trapezoid(flam[m_out] - cont, w_obs[m_out])
    assert out_line == pytest.approx(src_line, rel=0.05)
    lib.close()


def test_grizli_spectrum_normalises_to_unity_in_the_direct_band(tmp_path):
    """grizli's is_cgs=False treats spectrum_1d as a multiplier on flat f_lambda.

    Without this normalisation the raw Diffsky values (~1e-24) disperse an
    effectively empty scene — a silent, total failure.
    """
    from specsr_roman.extraction import SEDLibrary
    from specsr_roman.extraction.seds import H158_NORM_HI, H158_NORM_LO
    gid = 10307000000001
    _fake_sed_file(tmp_path / "sed.h5", gid)
    lib = SEDLibrary(str(tmp_path / "sed.h5"))
    got = lib.grizli_spectrum(gid, 1.0)
    assert got is not None
    (w, f), (w_raw, f_raw) = got
    band = (w > H158_NORM_LO) & (w < H158_NORM_HI)
    assert f[band].mean() == pytest.approx(1.0, rel=1e-6)
    # The raw SED is returned untouched: it becomes the training target.
    assert f_raw.max() > 1e-23
    assert f_raw.max() != pytest.approx(f.max())
    lib.close()


def test_grizli_spectrum_returns_none_for_an_unusable_sed(tmp_path):
    """A galaxy with no flux in the normalising band must be skipped, not crash."""
    from specsr_roman.extraction import SEDLibrary
    h5py = pytest.importorskip("h5py")
    gid = 10307000000002
    wave = np.arange(1000.0, 6000.0, 10.0)          # nothing near H158 at z=1
    with h5py.File(tmp_path / "sed2.h5", "w") as f:
        f.create_dataset("meta/wave_list", data=wave)
        f.create_dataset(f"galaxy/{gid // 100000}/{gid}",
                         data=np.zeros((3, len(wave))))
    lib = SEDLibrary(str(tmp_path / "sed2.h5"))
    assert lib.grizli_spectrum(gid, 1.0) is None
    lib.close()


def test_sed_library_raises_keyerror_for_a_missing_galaxy(tmp_path):
    """The batch worker catches KeyError to skip catalogue gaps."""
    from specsr_roman.extraction import SEDLibrary
    _fake_sed_file(tmp_path / "sed.h5")
    lib = SEDLibrary(str(tmp_path / "sed.h5"))
    with pytest.raises(KeyError):
        lib.observed(99999999999999, 1.0)
    lib.close()
