# specsr-roman

**Physics-informed super-resolution of Roman grism spectra.**

[![CI](https://github.com/aryana-haghjoo/specsr-roman/actions/workflows/ci.yml/badge.svg)](https://github.com/aryana-haghjoo/specsr-roman/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://aryana-haghjoo.github.io/specsr-roman/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22119689.svg)](https://doi.org/10.5281/zenodo.22119689)
[![PyPI](https://img.shields.io/pypi/v/specsr-roman)](https://pypi.org/project/specsr-roman/)
[![Python](https://img.shields.io/pypi/pyversions/specsr-roman)](https://pypi.org/project/specsr-roman/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model weights](https://img.shields.io/badge/%F0%9F%A4%97-model%20weights-yellow)](https://huggingface.co/aryana-haghjoo/roman-spectral-superresolution)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-training%20data-yellow)](https://huggingface.co/datasets/aryana-haghjoo/romansr-data)

Roman's High Latitude Spectroscopic Survey will deliver slitless grism spectra
(R ≈ 461 λ/µm over 1–1.93 µm) for millions of emission-line galaxies. At that
resolving power the diagnostic complexes stay blended — Hα+[N II],
[O III]+Hβ — across wide redshift ranges, biasing redshifts and everything
derived from them. Assembling survey-scale high-resolution training pairs to
fix that observationally is prohibitive.

`specsr-roman` super-resolves those spectra with a three-stage network trained on
simulated Roman products, and is built around one constraint:

> **A model that invents plausible emission lines is worse than useless in a
> survey.** Every design decision here follows from that.

The pipeline is calibrated on *recoverability* — whether the data could have
revealed a given line at all — and is explicitly penalised for drawing what it
could not have seen. It ships with the two audits that keep those claims
honest, and both are run in CI-adjacent form on every release.

This is the Roman half of a two-instrument project. The JWST/NIRSpec original
is [`specsr`](https://github.com/aryana-haghjoo/specsr) (Haghjoo et al. 2026).

---

## Install

```bash
pip install specsr-roman                 # inference + training
pip install "specsr-roman[hub]"          # + published checkpoints and data
pip install "specsr-roman[extract]"      # + dataset building (grizli, photutils)
pip install "specsr-roman[all]"          # everything
```

Python 3.10 or newer.

## Use it

```python
from specsr_roman import RomanPipeline

pipe = RomanPipeline.from_pretrained()      # downloads the published chain

out = pipe.predict(
    flux_low,                                # (864,) on the native grism grid
    flux_low_err,
    phot=roman_medium_fluxes,                # F106, F129, F158 — in that order
)

out.flux_sr        # super-resolved spectrum, 2500 px over 1.0–1.93 µm
out.flux_sr_err    # per-pixel uncertainty
out.z, out.z_err   # redshift point estimate and its spread
out.pz, out.z_grid # the full P(z) — keep it, see "Redshifts are multimodal"
out.presence       # per-line presence probability, over 98 features
```

`phot=None` supplies the redshift head's training-mean colours. It is a
missing-input mode of a photometry-trained head; a grism-only model would
need separate training. The older release's ablation numbers do not describe
these new weights.

**New here?** [`tutorials/01_getting_started.ipynb`](tutorials/01_getting_started.ipynb)
([rendered](https://aryana-haghjoo.github.io/specsr-roman/tutorials/01_getting_started.html))
runs the whole thing end to end on a 512-spectrum reserved-test sample: one super-resolved
spectrum, the redshift PDF, what photometry buys, and the recoverability-binned
line recovery below. Two minutes on a CPU, no dataset download.

From the command line:

```bash
specsr-roman info                                    # what's installed and canonical
specsr-roman predict spectra.npz --out predictions.npz
specsr-roman evaluate metrics --cache pred_cache.npz
specsr-roman evaluate figures --outdir figures/
```

---

## Results

Version 0.6.1 retains the independently trained v0.6.0 chain
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
Version 0.6.1 completes the audits of these weights. The Roman SR2 deployed
line-response exponent is **0.098** (galaxy-grouped 95% bootstrap
interval 0.089–0.106; 0.255 for strong lines), compared with 1 for
proportional recovery. The nominal central 68% redshift interval covers
**98.7%** of test truths. Spectral coverage is 75.5% within 1σ overall
and 92.1% within 1.96σ in detected-line regions.
These are diagnostics; no recalibration was fitted on the test data.
[Full audits, normalization controls and records](https://aryana-haghjoo.github.io/specsr-roman/guides/audits-0.6.1.html)
document the remaining limitations. Output line fluxes and ratios are not yet
calibrated measurements.

---

## How it works

Three stages, each doing one job it can be held to.

![The specsr-roman pipeline in three bands: simulating the training data (OpenUniverse2024 SEDs dispersed through the Wang+2022 grism model, extracted with grizli into 23,845 LR-HR pairs); spectral super-resolution (SR1, a conservative 1D residual CNN emitting mean and log-variance, refined by SR2, an attention network with one token per emission line); and the redshift branch (a ZHead taking the coarse spectrum plus three Roman bands and emitting P(z), whose top modes gate which lines SR2 may draw).](docs/_static/arch_pipeline_roman.png)

**SR1** is deliberately conservative. It sees `[flux, err]` scaled by a shared
factor, so the channel ratio *is* the per-pixel S/N and the network can
matched-filter rather than guess which bumps are noise.

**ZHead** predicts a distribution, not a number. Redshift from a grism is a
line-*identification* problem: one observed line is consistent with Hα,
[O III], [O II] or Lyα. A Gaussian head must average between aliases, which is
what produced the ~40 % catastrophic floor of two earlier designs. A softmax
over a grid carries the whole P(z) — the point estimate is the *mode*, and
ambiguity survives as PDF width instead of becoming a wrong answer.

**SR2** refines SR1 with one attention token per rest-frame feature, run once
per redshift hypothesis and combined by mode mass. That is what makes it
alias-robust: when the top mode is wrong the correct identification is usually
the second or third, and the right lines still get drawn.

`ARCHITECTURE.md` has the full design, including the losses and why each one
is shaped the way it is.

### Redshifts are multimodal — keep the PDF

`out.z` is the mode of `out.pz`, not its mean. For an alias-ambiguous source
the second mode is real information, and `out.z_err` widens precisely when the
model is torn between two line identifications. Discarding `pz` throws away
the part of the answer that tells you when not to trust the rest of it.

---

## Training

Every published checkpoint is reproducible from a config file:

```bash
specsr-roman train sr1   --config configs/sr1.yaml
specsr-roman train zhead --config configs/zhead.yaml     # needs SR1
specsr-roman train sr2   --config configs/sr2.yaml       # needs SR1 + ZHead
```

The configs *are* the published chain — `configs/sr1.yaml` produces
`sr1_ou2024_v6_independent` — and record the three-way split and upstream checkpoints, so
"what was this trained with" always has a readable answer. Any field can be
overridden on the command line (`--epochs 50`, `--no-augment`); an unknown key
is an error rather than a silent no-op.

Two settings deserve a warning if you change them:

- **`phot_tier`** must stay on bands a survey delivers with the grism.
  `medium` is the three Roman bands of the published chain; `medium_rubin`
  adds Rubin *ugrizy* at coadd depth (`rubin_depth_offset`).
  `grids.MAX_PHOT_BANDS` caps an explicit band list at those nine: hand a
  model more and it reads the redshift off an effectively complete SED
  instead of the grism.
- **`phot_eval_mag_err`** must stay above zero. A metric measured on noiseless
  truth photometry is not a metric.

### Checkpoint selection is not the validation loss

Both SR1 and SR2 select on custom goals, and this is load-bearing rather than
fussy. SR1's total validation loss is dominated by the NLL, which *rises* as
the model commits to line amplitudes — a confident half-amplitude line costs
more than a hedged flat one. SR2's plain NLL is continuum-dominated and
reliably selects the model that draws nothing.

So SR1 monitors line-flux recovery plus the hallucination penalty, and SR2
monitors `-recov_amp + lam_hallu * hallu_amp`. The Roman SR2 best checkpoint
is **epoch 8 of 150**, selected on validation. The final epoch is not the
published checkpoint.

---

## Building the dataset

OpenUniverse2024 provides direct images, truth catalogues and Diffsky SEDs but
**no grism images**, so `specsr-roman` disperses the scene itself with
[grizli](https://github.com/gbrammer/grizli) and extracts it back. That is not a
workaround: the same configuration file disperses and extracts, so any residual
is the fault of noise, blending and the extraction — not of a mismatched
instrument model.

```bash
specsr-roman extract run --max-scas 60 --workers 4    # resumable; safe to kill
specsr-roman extract merge                            # → every detected galaxy
specsr-roman extract select                           # → the training npz
```

`extract run` takes every galaxy the H158 image detects, with no magnitude
cut. A detection takes the identity of the brightest catalogue galaxy at its
position and keeps it only if the catalogue magnitude agrees with the detected
flux: the truth index lists objects far below the detection limit, and the
nearest one is often not the one detected. `extract select` then keeps the
spectra whose true Hα or [O III] flux exceeds the survey's line-flux limit
(`--line-flux`, default 10⁻¹⁶ erg s⁻¹ cm⁻²).

Or skip it: the built dataset is on the
[Hub](https://huggingface.co/datasets/aryana-haghjoo/romansr-data), together
with a 3.8 MB, 512-row `tutorial/` subset drawn from the held-out split for the
notebook above.

Each (visit, SCA) runs in its own subprocess — grizli's numba disperser
corrupts the heap when thousands of sources go through one process — and caches
its result, so a crash costs one SCA rather than the run.

**Split by object, never by row.** The same galaxy appears in many visits as
independent noise realisations. `specsr_roman.data.splits.get_or_make_three_way_split` keys
membership on a hash of the object id, which also makes it stable under dataset
growth: adding SCAs never moves an existing galaxy across the boundary.

---

## Limitations

Stated plainly, because they bound what the numbers mean.

- **Results are on the Diffsky manifold.** Targets use simulated line physics.
  Recovery can reflect the training prior. The new deployed-response audit
  gives Roman SR2 a median exponent of 0.098; proportional recovery would be 1.
- **Uncertainties require calibration.** The nominal 68% P(z) interval covers
  98.7% of test truths; spectral line-region tails under-cover at 1.96σ.
- **The test is reserved from new checkpoint selection.** These galaxies
  appeared in historical validation during hyperparameter development;
  a fully prospective sample and real-sky validation remain necessary.
- **Trained on simulations, not sky.** Real Roman spectra will differ. Domain
  adaptation via cross-instrument overlaps (DESI/PFS/JWST) is the intended
  route once real data exists.
- **The lines are not judged below S/N 2.** A quarter of the held-out spectra
  (about 24%) have no detected line in a single exposure. The redshift numbers
  include them; the line-recovery numbers do not, and say nothing about what
  the model draws there.
- **The sample is selected on the true line flux.** A survey has to estimate
  that flux from the data. In this simulation it also follows continuum
  brightness closely, so the sample lacks faint galaxies with strong lines.
- **Residual line-identification aliases remain.** Weak lines may require
  photometric disambiguation. These experiments do not establish an
  irreducible outlier rate.

---

## Citation

If you use this software, please cite the paper and the code. The method paper
for the JWST original is Haghjoo et al. (2026), *Learning to See Sharper: A
Physics-Informed Artificial Intelligence Framework for Super-Resolving Galaxy
Spectra*, ApJ.

The code is archived on Zenodo. Cite [`10.5281/zenodo.22119689`](https://doi.org/10.5281/zenodo.22119689)
for the software in general — it always resolves to the newest version — or the
version DOI on that record to pin the exact release you ran. `CITATION.cff`
carries both, and GitHub's "Cite this repository" button renders it as BibTeX
or APA.

Simulation products: OpenUniverse et al. (2025), OpenUniverse2024; Wang et al.
(2022) for the Roman GRS grism configuration.

## License

MIT — see [LICENSE](LICENSE).
