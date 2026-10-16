# Release 0.6.0

Version 0.6.0 uses the independently trained chain
`sr1_ou2024_v6_independent → zhead_ou2024_roman_med3_noisy_independent →
sr2_ou2024_v5_romanonly_independent`.

The dataset contains 23,845 spectra of 10,039 galaxies detected in H158,
selected by true Hα or [O III] λ5007 flux above 10⁻¹⁶ erg s⁻¹ cm⁻².
The galaxy-grouped split has 19,134 training spectra (8,042 galaxies),
2,327 validation spectra (992 galaxies), and **2,384 reserved test spectra
(1,005 galaxies)**. Checkpoints were selected on validation only; all nine
models were frozen before test predictions were generated. Evaluation uses
0.05 mag Roman photometric noise and the configured Rubin coadd noise.
The previous 4,711-spectrum results used the checkpoint-selection partition
and must be read as validation results.

**Redshift — every reserved test spectrum, no S/N cut**

| Configuration | NMAD | Catastrophic outliers |
|---|---:|---:|
| Grism + Roman F106/F129/F158 | 0.00133 | 0.67% (16/2,384) |
| + Rubin ugrizy, Year 1 | 0.00120 | 0.042% (1/2,384) |
| + Rubin ugrizy, Year 10 | 0.00123 | 0.042% (1/2,384) |
| Nine-band photometry-only redshift control | 0.01152 | None in 2,384 spectra |

Catastrophic means |Δz|/(1+z) > 0.15. The control removes the spectrum
from the redshift head; its SR2 still reads the spectrum. No observed outliers
in this finite sample does not imply a zero population outlier rate.

**Line amplitude recovery — Roman three-band chain**

Median recovered/true flux in detected line regions, binned by the best
integrated line S/N in the grism. Lines below S/N 2 are excluded.

| Recoverability | n | SR1 | SR2 |
|---|---:|---:|---:|
| Marginal (2–3) | 764 | 0.58 | 0.90 |
| Good (3–6) | 710 | 0.78 | 0.94 |
| Strong (>6) | 345 | 0.81 | 0.89 |

| SR2 configuration | Marginal / good / strong recovery |
|---|---|
| Roman three bands | 0.90 / 0.94 / 0.89 |
| + Rubin Year 1 | 0.90 / 1.01 / 0.98 |
| + Rubin Year 10 | 0.84 / 0.94 / 0.95 |
| Photometry-only redshift control | 0.82 / 0.94 / 0.93 |

Frozen test caches, exact metrics, the split record and weight provenance are
available with the [dataset](https://huggingface.co/datasets/aryana-haghjoo/romansr-data)
and [models](https://huggingface.co/aryana-haghjoo/roman-spectral-superresolution).
Response and uncertainty audits from earlier weights do not validate this
release. Line fluxes and ratios should not yet be treated as calibrated
measurements; independent response and uncertainty calibration remain open.

The pending audits described above were completed in the
[0.6.1 follow-up](audits-0.6.1.md), using the same weights and test caches.
