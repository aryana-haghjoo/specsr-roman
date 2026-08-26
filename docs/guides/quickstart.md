# Quickstart

## Super-resolve a spectrum

```python
import numpy as np
from specsr_roman import RomanPipeline
from specsr_roman.grids import WAVE_LR, ROMAN_MEDIUM_BANDS

pipe = RomanPipeline.from_pretrained()      # downloads ~11 MB on first use

out = pipe.predict(
    flux_low,          # (864,) on WAVE_LR, the native grism sampling
    flux_low_err,      # same shape; the model reads the flux/err ratio as S/N
    phot=phot[list(ROMAN_MEDIUM_BANDS)],    # F106, F129, F158 — in that order
)
```

The result carries more than a spectrum:

| Attribute | Meaning |
|---|---|
| `flux_sr`, `flux_sr_err` | super-resolved flux and per-pixel uncertainty, 2500 px |
| `flux_sr1` | the SR1 stage alone, for comparison |
| `z`, `z_err` | redshift mode and the spread of the full PDF |
| `pz`, `z_grid` | the whole P(z) |
| `presence` | per-line presence probability over 98 features |
| `wavelength` | observed wavelength grid, Å |

Values come back on the input's own flux scale, so they plot directly against
it.

## Keep the redshift PDF

`z` is the **mode** of `pz`, not its mean. Redshift from a grism is a line
*identification* problem: one observed line is consistent with Hα, [O III],
[O II] or Lyα. For an ambiguous source the second mode is real information, and
`z_err` widens precisely when the model is torn between identifications.

```python
top = np.argsort(out.pz)[::-1][:3]
for i in top:
    print(f"z = {out.z_grid[i]:.3f}  p = {out.pz[i]:.3f}")
```

If a source's second mode carries meaningful mass, treat the point estimate
with suspicion — that is what the PDF is telling you.

## Running without photometry

```python
out = pipe.predict(flux_low, flux_low_err, phot=None)
```

This works and is honest, but is much weaker: without a colour prior a single
in-band line is alias-degenerate, and roughly 46 % of redshifts are
catastrophic outliers against 5 % with Roman Medium-tier imaging.

## Batches

Pass 2D arrays and get a list back:

```python
outs = pipe.predict(flux_low_batch, err_batch, phot=phot_batch)
```

## From the command line

```bash
specsr-roman predict spectra.npz --out predictions.npz --phot-tier medium
```

where `spectra.npz` holds `flux_low` and optionally `flux_low_err`, `phot` and
`wavelength_low`.
