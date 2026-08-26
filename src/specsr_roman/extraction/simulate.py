"""Dispersing a scene and adding Roman grism noise."""

from __future__ import annotations

import numpy as np

from ..grids import GRISM_BKG, GRISM_EXPTIME, GRIZLI_BEAM_SIZE, READ_NOISE

__all__ = ["disperse_scene", "add_grism_noise"]


def disperse_scene(flt, compact_ids, object_ids, mags, redshifts, seds,
                   scene_indices, verbose: bool = True):
    """Disperse every listed source with its true SED into ``flt.model``.

    Returns ``{index: grizli spectrum}`` for the sources that succeeded, so a
    later pass can re-disperse a target on its own without re-reading the SED
    file.

    ``size=85, compute_size=False`` is not tunable in practice. The Roman
    trace sits up to ~66 px from the source on detector 1 (and ~162 px on
    detector 4), which overflows grizli's default adaptive cutout and silently
    truncates the beam.
    """
    kept = {}
    n_fail = 0
    for j, i in enumerate(scene_indices):
        cid, oid = int(compact_ids[i]), int(object_ids[i])
        z = redshifts.get(oid)
        if z is None:
            n_fail += 1
            continue
        try:
            got = seds.grizli_spectrum(oid, float(z))
        except KeyError:
            got = None
        if got is None:
            n_fail += 1
            continue
        spec, raw = got
        kept[i] = (spec, raw)
        flt.compute_model_orders(id=cid, mag=float(mags[i]),
                                 size=GRIZLI_BEAM_SIZE, compute_size=False,
                                 spectrum_1d=spec, is_cgs=False,
                                 store=False, in_place=True)
        if verbose and (j + 1) % 100 == 0:
            print(f"  {j + 1}/{len(scene_indices)} dispersed", flush=True)
    if verbose and n_fail:
        print(f"  {n_fail} sources had no catalogue/SED entry (skipped)")
    return kept


def add_grism_noise(scene: np.ndarray, exptime: float = GRISM_EXPTIME,
                    background: float = GRISM_BKG,
                    read_noise: float = READ_NOISE, seed: int = 0):
    """Noiseless scene (e-/s) -> ``(noisy, error)``, both in e-/s.

    Poisson from source plus zodiacal background, plus read noise, all
    expressed as a rate variance so the arrays stay in the units grizli's
    optimal extraction expects. The Gaussian approximation to the Poisson term
    is safe here: even a faint HLSS source accumulates enough electrons over
    301 s that the distribution is near-normal, and the background alone
    contributes ~170 e-.
    """
    rng = np.random.default_rng(seed)
    var_rate = ((np.clip(scene, 0, None) + background) / exptime
                + (read_noise / exptime) ** 2)
    err = np.sqrt(var_rate).astype(np.float32)
    noisy = (scene + rng.normal(0, err)).astype(np.float32)
    return noisy, err
