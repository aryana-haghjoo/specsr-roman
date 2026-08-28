# Data

## Getting the dataset

The built training set is on the Hub:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download("aryana-haghjoo/romansr-data",
                       "ou2024_h10307_dataset.npz", repo_type="dataset")
```

36,404 extracted spectra of 15,434 galaxies, healpix 10307, 0.06 < z < 3.1.

### The tutorial subset

A 3.8 MB, 512-row sample sits in the same repo under `tutorial/`, for the
[getting-started notebook](../tutorials/01_getting_started.ipynb) and for any
quick check that does not need 271 MB:

```python
path = hf_hub_download("aryana-haghjoo/romansr-data",
                       "tutorial/ou2024_h10307_tutorial.npz", repo_type="dataset")
```

Same schema, plus a `source_row` column recording where each row came from in
the full file. It is drawn from the **held-out side** of the canonical split
and sampled uniformly within it — so out-of-sample metrics computed on it are
honest, and the mix of recoverable and undetectable lines is the population's
own rather than a flattering selection. Rebuild it with
`python scripts/make_tutorial_dataset.py`.

## Schema

One npz, all rows on two shared grids:

| Key | Shape | Meaning |
|---|---|---|
| `flux_low`, `flux_low_err` | (N, 864) | extracted grism spectrum on `WAVE_LR` |
| `flux_high` | (N, 2500) | ground-truth SED on `WAVE_HR` |
| `redshift` | (N,) | true redshift |
| `ids` | (N,) | OU2024 `object_id` — **split on this** |
| `phot` | (N, 14) | catalogue fluxes, order fixed by `grids.PHOT_BANDS` |
| `ab_h158`, `snr` | (N,) | selection and quality columns |
| `visit`, `sca` | (N,) | provenance |
| `wavelength_low`, `wavelength_high` | (864,), (2500,) | the shared grids |

`flux_high` has no error column: the targets are noiseless simulated SEDs. The
dataset returns a zero error array anyway, so the loss signature matches the
JWST version where targets are real grating spectra.

## Splitting

**Split by object, never by row.** The same galaxy appears in many visits as
independent noise realisations; a row-wise split puts one realisation in train
and another in test and measures memorisation.

```python
from specsr_roman.data import get_or_make_group_split
train_idx, test_idx, _ = get_or_make_group_split(path, data["ids"])
```

Membership is a pure hash of the object id, which buys two things: every stage
derives the same split without passing a file around, and the split is stable
under dataset growth — adding SCAs never moves an existing galaxy across the
boundary, so a model trained on an older build stays evaluable on the newer
test set.

## Photometric tiers

Feed the redshift head only bands that **ship with the grism**:

```python
from specsr_roman.grids import ROMAN_MEDIUM_BANDS   # (8, 9, 11) — F106, F129, F158
from specsr_roman.data import RomanFixedGridDataset

ds = RomanFixedGridDataset(path, with_phot=True, phot_tier="medium")
```

Tiers are `medium`, `deep`, `all`, or an explicit `"8,9,11"`.

`all` includes LSST *ugrizy*. LSST coverage over Roman's grism footprint is
external, partial, and not guaranteed at first data release — and with all 14
noiseless bands the photometry is an effectively complete SED from which the
redshift can be read without the spectrum contributing anything. It is a
diagnostic configuration, never one to deploy.

:::{warning}
OU2024 uses the **old** Roman band names, which collide with the current WFI
scheme: OU2024 `R062` is 0.62 µm (current F062), and OU2024 `W146` is the
current wide filter R062. Always map by central wavelength, never by name.
`grids.PHOT_BANDS` fixes the index order.
:::

## Building it yourself

Requires the `extract` extra.

```bash
specsr-roman extract run --max-scas 60 --workers 4
specsr-roman extract merge
```

Resumable: each (visit, SCA) caches its result, including a stub recording why
a pointing produced nothing, so re-running skips completed work. Each runs in
its own subprocess because grizli's numba disperser corrupts the heap when
thousands of sources go through one process — a crash costs one SCA, not the
run.

Add `--cleanup` to delete raw and prepared FITS as it goes; the full set is
tens of gigabytes.

### What it does per SCA

1. Prepare a grizli direct image plus an empty grism shell (`INSTRUME='WFI'`,
   so grizli loads `Roman.G150.conf`).
2. Detect sources, KD-tree match to the truth index.
3. Disperse every source brighter than `--ab-scene` with its **true** SED.
4. Add Poisson (source + zodiacal) and read noise at 301 s.
5. Per target: re-disperse it alone, subtract from the scene for exact
   contamination, optimally extract, calibrate against a flat-f_λ pass.

Because the same configuration disperses and extracts, input and target are
self-consistent by construction — any residual is noise, blending or the
extraction, not a mismatched instrument model.

The merge step refuses to write if more than 1 % of rows have a constant
`flux_low`. A constant spectrum carries no information and would train a model
straight into the prior mean while every loss curve looked healthy.
