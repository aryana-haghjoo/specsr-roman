#!/usr/bin/env python
"""AB zero points of the OpenUniverse2024 catalogue fluxes, from the simulation.

The ``galaxy_flux`` columns are photon rates through each bandpass, so each
band has its own zero point, ``AB = ZP - 2.5 log10(f)``. This measures them
instead of assuming a throughput: the SED of each galaxy gives AB colours
through the filter curves, the H158 anchor (fixed by the image calibration)
gives the scale, and the zero point of a band is what turns its catalogue
flux into that magnitude. The scatter over galaxies is the check.

    ./venv/bin/python scripts/derive_zeropoints.py

Prints the table that ``specsr_roman.grids.AB_ZEROPOINT`` holds.
"""

import argparse

import numpy as np

from specsr_roman.extraction.seds import SEDLibrary
from specsr_roman.grids import AB_ANCHOR_H158, AB_ZEROPOINT, PHOT_BANDS

SEDPY = {
    "lsst_flux_u": "lsst_baseline_u", "lsst_flux_g": "lsst_baseline_g",
    "lsst_flux_r": "lsst_baseline_r", "lsst_flux_i": "lsst_baseline_i",
    "lsst_flux_z": "lsst_baseline_z", "lsst_flux_y": "lsst_baseline_y",
    "roman_flux_R062": "roman_wfi_f062", "roman_flux_Z087": "roman_wfi_f087",
    "roman_flux_Y106": "roman_wfi_f106", "roman_flux_J129": "roman_wfi_f129",
    "roman_flux_W146": "roman_wfi_f146", "roman_flux_H158": "roman_wfi_f158",
    "roman_flux_F184": "roman_wfi_f184", "roman_flux_K213": "roman_wfi_f213",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/dataset/ou2024_h10307_dataset.npz")
    ap.add_argument("--seds", default="data/ou2024/galaxy_sed_10307.hdf5")
    ap.add_argument("-n", type=int, default=400)
    args = ap.parse_args()

    from sedpy.observate import getSED, load_filters
    filters = load_filters([SEDPY[b] for b in PHOT_BANDS])
    h = PHOT_BANDS.index("roman_flux_H158")

    d = np.load(args.data, allow_pickle=True)
    _, first = np.unique(d["ids"], return_index=True)
    rng = np.random.default_rng(0)
    rows = rng.choice(first, size=min(args.n, len(first)), replace=False)

    lib = SEDLibrary(args.seds)
    zps = []
    for i in rows:
        phot = d["phot"][i]
        if not np.all(phot > 0):
            continue
        # a finer grid than the default: the u band is narrow
        wave, flam = lib.observed(int(d["ids"][i]), float(d["redshift"][i]), dlam=2.0)
        mags = getSED(wave, flam, filters)
        ab = mags - mags[h] + AB_ANCHOR_H158 - 2.5 * np.log10(phot[h])
        zps.append(ab + 2.5 * np.log10(phot))
    lib.close()
    zps = np.array(zps)
    med = np.median(zps, axis=0)
    mad = 1.4826 * np.median(np.abs(zps - med), axis=0)
    print(f"{len(zps)} galaxies")
    for b, m, s in zip(PHOT_BANDS, med, mad, strict=True):
        now = AB_ZEROPOINT.get(b, float("nan"))
        print(f'    "{b}": {m:.3f},   # scatter {s:.3f}, in grids {now:.3f}')


if __name__ == "__main__":
    main()
