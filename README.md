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

`phot=None` drops the colour prior, and the redshift degrades sharply — 26 %
catastrophic against 5 %, see the table below — because a single in-band line
is alias-degenerate. It is not the same thing as a grism-only *model*: this
head was trained with photometry, so `None` hands it its training-mean
colours.

**New here?** [`tutorials/01_getting_started.ipynb`](tutorials/01_getting_started.ipynb)
([rendered](https://aryana-haghjoo.github.io/specsr-roman/tutorials/01_getting_started.html))
runs the whole thing end to end on a 512-spectrum sample: one super-resolved
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

Canonical chain `sr1_ou2024_v6 → zhead_ou2024_roman_med3_noisy →
sr2_ou2024_v5_romanonly`, evaluated on the held-out object-id split of the
OU2024 dataset (N = 4,711 spectra, 1,997 galaxies), with 0.05 mag photometric
noise applied at evaluation.

**The sample is the survey's.** Every galaxy detected in the H158 images is
extracted, with no cut on magnitude, and a galaxy is kept if the true flux of
its Hα or [O III] λ5007 line exceeds 10⁻¹⁶ erg s⁻¹ cm⁻², the line-flux limit of
the Roman spectroscopic survey (Wang et al. 2022). That is 23,845 spectra of
10,039 galaxies out of 68,806 detected, at 0.55 < z < 2.6.

**Redshift** — every held-out spectrum, no cut on signal-to-noise

| Configuration | NMAD | catastrophic |
|---|---|---|
| Grism + Roman Medium (F106/F129/F158), noisy — **deployable** | **0.0014** | **0.8 %** |
| The same chain called with `phot=None` | 0.0013 | 2.9 % |

The deployable row uses only the imaging that actually ships with the HLWAS
grism, with realistic noise at train *and* eval. The colours do not sharpen the
redshift; they remove outliers, by choosing between line identifications where
the line is too faint to settle it.

The second row is what this chain does when you call it with `phot=None`, and
it is **not** the grism-only information floor. The head was trained with
photometry, so passing `None` feeds it its training-mean colours: mean
imputation on an out-of-distribution input rather than a clean ablation of the
information. A head actually trained without colours is a separate experiment,
and has not been run.

Both rows come from `specsr-roman evaluate ablation`, which measures the
published head with and without its three colours and sweeps the photometric
noise. Every configuration it reports uses those three bands and no others.

**Line amplitude recovery** — median recovered flux fraction of the detected
lines, split by how well the best line is detected. The lines are judged only
where they are detected, at an integrated S/N of at least 2 in the
single-exposure grism data: 3,579 of the 4,711 spectra have such a line. Below
that the data carry no strong signal, and nothing is concluded about what the
model draws there.

| Recoverability (integrated line S/N) | n | SR1 | SR2 |
|---|---|---|---|
| marginal (2–3) | 1,497 | 0.52 | 0.87 |
| good (3–6) | 1,409 | 0.70 | 0.95 |
| strong (> 6) | 673 | 0.77 | 0.96 |

Wherever a line is detected the full pipeline recovers most of its flux, and
SR2 adds to SR1 in every bin, most in the marginal one. In 93 % of the spectra
with a detected line exactly one line is detected, and it is Hα (with [N II])
in essentially all of them.

**Prior-dominance audit** (`specsr-roman evaluate prior`): response exponent
r = 0.17 for the full pipeline and 0.22 for SR1 alone, on the 3,552 held-out
spectra with a detected line — 1 means the model reads line strengths from
the data, 0 means it recites the training manifold. For strongly detected
lines the full pipeline reaches 0.24. The model finds and sharpens the lines
the data contain, but their strengths still lean on the simulation, so line
fluxes and ratios from the output are not yet measurements. See
[Limitations](#limitations).

**SR2 by redshift configuration.** SR2 places its lines at the redshifts it is
given, so there is one SR2 for each redshift head, and every SR2 evaluation is
drawn once per configuration (`specsr-roman evaluate band-figures`). The line
fluxes are integrals of Gaussian fits, compared with the same fit on the
target, for lines detected in the grism.

| Redshift head | NMAD | recovery, marginal / good / strong | Hα flux / truth | [O III] flux / truth | response r |
|---|---|---|---|---|---|
| Roman Y106, J129, H158 | 0.0014 | 0.87 / 0.95 / 0.96 | 1.01 | 0.81 | 0.17 |
| + Rubin *ugrizy*, 1 yr | 0.0011 | 0.82 / 0.94 / 0.98 | 1.01 | 0.85 | 0.18 |
| + Rubin *ugrizy*, 10 yr | 0.0011 | 0.76 / 0.90 / 0.92 | 0.97 | 0.88 | 0.18 |
| 9 bands, no spectrum (control) | 0.0116 | 0.82 / 0.93 / 0.94 | 0.99 | 0.85 | 0.17 |

The redshift head changes SR2 little, and not in step with the quality of the
redshift: the spread in the marginal bin comes from SR2's early-epoch
checkpoint, not from the head. For comparison, the same fit on the grism input
gives an Hα ratio of 1.05 with 11 % of lines off by more than a factor of two
(6–9 % after SR2), and an [O III] ratio of 0.90 with 25 % (3–13 %).

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
`sr1_ou2024_v6` — and the dataclass defaults in `specsr_roman.config` match them, so
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
monitors `-recov_amp + lam_hallu * hallu_amp`. **SR2's best epoch is 2.** That
is the design working, not a truncated run: hallucination amplitude climbs from
0.47 to 0.60 by epoch 150 while recoverable amplitude barely moves.

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
independent noise realisations. `specsr_roman.data.get_or_make_group_split` keys
membership on a hash of the object id, which also makes it stable under dataset
growth: adding SCAs never moves an existing galaxy across the boundary.

---

## Limitations

Stated plainly, because they bound what the numbers mean.

- **Results are on the Diffsky manifold.** Targets are simulated SEDs with
  simulation line physics. A model can score well by learning that manifold
  rather than by measuring anything, and no reconstruction metric distinguishes
  the two. The prior-dominance audit puts the published chain at r ≈ 0.17: a line
  doubled in the input grows by about 12 % in the output.
- **Anti-prior augmentation is implemented but not used.** It raises r to ~0.51
  at fixed detectability while suppressing absolute line recovery, so the
  published SR1 is unaugmented. A version that jitters only *recoverable*
  information is open work.
- **Trained on simulations, not sky.** Real Roman spectra will differ. Domain
  adaptation via cross-instrument overlaps (DESI/PFS/JWST) is the intended
  route once real data exists.
- **The lines are not judged below S/N 2.** A quarter of the held-out spectra
  (24 %) have no detected line in a single exposure. The redshift numbers
  include them; the line-recovery numbers do not, and say nothing about what
  the model draws there.
- **The sample is selected on the true line flux.** A survey has to estimate
  that flux from the data. In this simulation it also follows continuum
  brightness closely, so the sample lacks faint galaxies with strong lines.
- **The remaining outliers are physics, not a bug to be tuned away.** Where
  the line is too faint to see, the identification rests on three colours;
  they settle most cases and cannot settle all.

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
