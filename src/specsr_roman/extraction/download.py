"""Fetching OpenUniverse2024 products.

The preview subset lives at IRSA; the full survey is on AWS S3 (public, no
credentials). Only H158 direct images and their truth indexes are pulled per
SCA --- the catalogues and the SED file are per-healpix and downloaded once.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

__all__ = ["S3_BASE", "IRSA_BASE", "fetch", "sca_paths", "candidate_scas"]

S3_BASE = "https://nasa-irsa-simulations.s3.amazonaws.com/openuniverse2024/roman/full"
IRSA_BASE = "https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/roman/preview"


def fetch(url: str, dest: str) -> bool:
    """Download to ``dest`` if absent. Returns success.

    Writes to a ``.tmp`` and renames, so an interrupted download can never be
    mistaken for a complete file by the resume logic --- which is the whole
    reason the batch driver can be killed and restarted freely.
    """
    if os.path.exists(dest):
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = subprocess.run(["curl", "-sf", "--retry", "5", "--retry-all-errors",
                        "-o", dest + ".tmp", url])
    if r.returncode != 0:
        if os.path.exists(dest + ".tmp"):
            os.remove(dest + ".tmp")
        return False
    os.rename(dest + ".tmp", dest)
    return True


def sca_paths(visit: int, sca: int, raw_dir: str, base: str = S3_BASE):
    """``(image_path, index_path, image_url, index_url)`` for one (visit, SCA)."""
    img = f"{raw_dir}/Roman_WAS_simple_model_H158_{visit}_{sca}.fits.gz"
    idx = f"{raw_dir}/Roman_WAS_index_H158_{visit}_{sca}.txt"
    img_url = (f"{base}/RomanWAS/images/simple_model/H158/{visit}/"
               f"Roman_WAS_simple_model_H158_{visit}_{sca}.fits.gz")
    idx_url = (f"{base}/RomanWAS/truth/H158/{visit}/"
               f"Roman_WAS_index_H158_{visit}_{sca}.txt")
    return img, idx, img_url, idx_url


def candidate_scas(galaxy_parquet: str, obseq_fits: str, max_scas: int):
    """(visit, SCA) pairs whose H158 pointing lands near the target healpix.

    The RA/Dec box overfills the healpix diamond, so visits are swept in order
    of distance from its centre: productive SCAs come first and edge pointings
    last, which matters because the driver stops after ``max_scas`` productive
    outputs rather than after a fixed number of attempts.
    """
    import pyarrow.parquet as pq
    from astropy.io import fits

    g = pq.read_table(galaxy_parquet, columns=["ra", "dec"])
    ra, dec = np.asarray(g["ra"]), np.asarray(g["dec"])
    box = (ra.min(), ra.max(), dec.min(), dec.max())

    with fits.open(obseq_fits) as h:
        o = h[1].data
        filt = np.char.strip(o["filter"].astype(str))
        m = ((o["ra"] > box[0]) & (o["ra"] < box[1])
             & (o["dec"] > box[2]) & (o["dec"] < box[3]) & (filt == "H158"))
        visits = np.where(m)[0]
        c_ra, c_dec = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
        d2 = ((np.cos(np.radians(c_dec)) * (o["ra"][visits] - c_ra)) ** 2
              + (o["dec"][visits] - c_dec) ** 2)
        visits = visits[np.argsort(d2)]

    pairs = [(int(v), s) for v in visits for s in range(1, 19)]
    return pairs[: max_scas * 8], box
