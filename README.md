# specsr-roman

**Physics-informed super-resolution of Roman grism spectra — recovering
emission lines without inventing them.**

[![CI](https://github.com/aryana-haghjoo/specsr-roman/actions/workflows/ci.yml/badge.svg)](https://github.com/aryana-haghjoo/specsr-roman/actions/workflows/ci.yml)
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

Not on PyPI yet — install from source:

```bash
GH=git+https://github.com/aryana-haghjoo/specsr-roman
pip install "specsr-roman @ $GH"               # inference + training
pip install "specsr-roman[hub] @ $GH"          # + published checkpoints and data
pip install "specsr-roman[extract] @ $GH"      # + dataset building (grizli, photutils)
pip install "specsr-roman[all] @ $GH"          # everything
```

Once released, `pip install specsr-roman` (with the same extras) will do the
same thing.

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

`phot=None` runs grism-only. That is honest but much weaker — see the table
below — because a single in-band line is alias-degenerate.

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
OU2024 dataset (N = 7,334 spectra, 3,098 galaxies), with 0.05 mag photometric
noise applied at evaluation.

**Redshift**

| Configuration | NMAD | median \|Δz\|/(1+z) | catastrophic |
|---|---|---|---|
| Grism + Roman Medium (F106/F129/F158), noisy — **deployable** | **0.0065** | 0.0047 | **5.1 %** |
| Grism only | 0.18 | — | 46 % |

The deployable row uses only the imaging that actually ships with the HLWAS
grism, with realistic noise at train *and* eval. The grism-only row is the
single-line information floor: with one line in band the redshift is genuinely
alias-degenerate, and ~37 % catastrophic is what the physics allows. Three
broadband colours break most of that degeneracy, which is the whole reason the
head takes photometry at all.

**Line amplitude recovery** — median recovered flux fraction, split by whether
the line was recoverable from the LR data at all:

| Recoverability (integrated line S/N) | n | SR1 | SR2 |
|---|---|---|---|
| unrecoverable (< 1) | 1,243 | −0.01 | **0.03** |
| marginal (1–3) | 1,984 | 0.20 | 0.46 |
| good (3–6) | 1,077 | 0.61 | 0.81 |
| strong (> 6) | 387 | 0.67 | 0.85 |

Read the first row first. SR2 sharpens recoverable lines toward truth while
leaving undetectable ones at ~0.03 of a line it does not draw. A model that
scored 0.85 in the bottom row and 0.85 in the top would be worthless, and no
single averaged metric would tell you.

**Prior-dominance audit** (`specsr-roman evaluate prior`): response exponent
r = 0.45 for the published SR1 on OU2024 — 1 means the model reads line
strengths from the data, 0 means it recites the training manifold. See
[Limitations](#limitations).

---

## How it works

Three stages, each doing one job it can be held to.

![The specsr-roman pipeline in three bands: simulating the training data (OpenUniverse2024 SEDs dispersed through the Wang+2022 grism model, extracted with grizli into 36,404 LR-HR pairs); spectral super-resolution (SR1, a conservative 1D residual CNN emitting mean and log-variance, refined by SR2, an attention network with one token per emission line); and the redshift branch (a ZHead taking the coarse spectrum plus three Roman bands and emitting P(z), whose top modes gate which lines SR2 may draw).](docs/_static/arch_pipeline_roman.png)

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

- **`phot_tier`** must stay on bands that ship with the grism. `all` adds LSST
  *ugrizy* plus every Roman band, an effectively complete SED from which the
  redshift can be read without the spectrum contributing anything.
- **`phot_eval_mag_err`** must stay above zero. A metric measured on noiseless
  truth photometry is not a metric.

### Checkpoint selection is not the validation loss

Both SR1 and SR2 select on custom goals, and this is load-bearing rather than
fussy. SR1's total validation loss is dominated by the NLL, which *rises* as
the model commits to line amplitudes — a confident half-amplitude line costs
more than a hedged flat one. SR2's plain NLL is continuum-dominated and
reliably selects the model that draws nothing.

So SR1 monitors line-flux recovery plus the hallucination penalty, and SR2
monitors `-recov_amp + lam_hallu * hallu_amp`. **SR2's best epoch is 4.** That
is the design working, not a truncated run: hallucination amplitude climbs from
0.26 to 0.62 by epoch 150 while recoverable amplitude barely moves.

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
specsr-roman extract merge                            # → one training npz
```

Or skip it: the built dataset is on the
[Hub](https://huggingface.co/datasets/aryana-haghjoo/romansr-data).

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
  the two. The prior-dominance audit puts the published SR1 at r ≈ 0.45 — it
  reads the data about half the time it could.
- **Anti-prior augmentation is implemented but not used.** It raises r to ~0.51
  at fixed detectability while suppressing absolute line recovery, so the
  published SR1 is unaugmented. A version that jitters only *recoverable*
  information is open work.
- **Trained on simulations, not sky.** Real Roman spectra will differ. Domain
  adaptation via cross-instrument overlaps (DESI/PFS/JWST) is the intended
  route once real data exists.
- **The 5.1 % catastrophic rate is physics, not a bug to be tuned away.** With
  a single line in band the identification is genuinely ambiguous; photometry
  breaks most of it and cannot break all of it.

---

## Citation

If you use this software, please cite the paper and the code (see
`CITATION.cff`). The method paper for the JWST original is Haghjoo et al.
(2026), *Learning to See Sharper: A Physics-Informed Artificial Intelligence
Framework for Super-Resolving Galaxy Spectra*, ApJ.

Simulation products: Troxel et al. (2025), OpenUniverse2024; Wang et al.
(2022) for the Roman GRS grism configuration.

## License

MIT — see [LICENSE](LICENSE).
