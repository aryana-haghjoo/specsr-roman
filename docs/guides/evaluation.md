# Evaluation

Every metric on this page is computed end to end, on real predictions, in the
[getting-started notebook](../tutorials/01_getting_started.ipynb) — including
the recoverability split below and what it looks like when a model behaves.

## The prediction cache

Every figure and every quoted number reads one frozen npz rather than a live
model, so a plotting tweak cannot quietly change a result.

```bash
specsr-roman evaluate cache --out outputs/pred_cache.npz
specsr-roman evaluate metrics --cache outputs/pred_cache.npz
specsr-roman evaluate figures --cache outputs/pred_cache.npz --outdir outputs/figures
```

Regenerate the cache deliberately when the chain changes — never by accident
while tuning a plot.

## Metrics

```python
from specsr_roman.evaluation import line_amplitude_recovery, redshift_summary

redshift_summary(cache["z_pred"], cache["z_true"])
line_amplitude_recovery(cache["sr2"], cache["hr"], cache["line_snr"],
                        z=cache["z_true"], wave_um=cache["wl_um"])
```

**Sample statistics use every spectrum; the lines are judged only where they
are detected.** Redshift accuracy is computed on the whole held-out split. A
line enters the amplitude metric only if its own integrated S/N reaches 2
(`MIN_BEST_LINE_SNR`): passing `z` and `wave_um` applies that
(`detected_line_mask`), and the bins start there. Below it the data carry no
strong signal, so the metric says nothing about what the model draws there.

`redshift_summary` reports NMAD, median |Δz|/(1+z), catastrophic fraction and
N. Report all of them: a model can shrink NMAD while pushing more objects past
the catastrophic threshold, and NMAD alone hides alias structure entirely.

## Read the recoverability split first

`line_amplitude_recovery` bins rows by their best line's integrated S/N:

| Bin | Integrated line S/N |
|---|---|
| `marginal` | 2–3 |
| `good` | 3–6 |
| `strong` | > 6 |

Report the bins separately. A single averaged amplitude ratio mixes marginal
detections with strong ones and hides how the recovery depends on the data.

`per_line_amplitude_recovery` is the diagnostic companion: same idea, scored
per transition rather than per row, so a failure can be attributed to a
specific line. The two use different definitions and should not be compared to
each other.

## Figures

| Key | What it shows |
|---|---|
| `spectra` | HR / LR / SR2 overlay with a zoom inset on the blended complex |
| `river` | residual maps sorted by z, with rest-frame line tracks |
| `sn` | per-line S/N, SR2 against the LR input |
| `redshift` | z_pred vs z_true — read the off-diagonal alias stripes |
| `psd` | signal and residual power spectra |

```python
from specsr_roman.evaluation.figures import make_figures
make_figures(cache, which=["spectra", "redshift"], outdir="figures/")
```

### One panel per redshift configuration

SR2 places its lines at the redshifts it is given, so its evaluation depends on
the redshift head behind it. `specsr_roman.evaluation.band_figures.BANDS` lists
four configurations (three Roman bands; the same with Rubin at one-year and at
ten-year depth; nine bands and no spectrum), each with its own SR2 checkpoint
and prediction cache (`outputs/pred_cache.npz` for the first,
`outputs/pred_cache_<key>.npz` for the others; all four are on the Hub).

```bash
specsr-roman evaluate band-figures      # every figure below, plus band_metrics.json
```

| File | What it shows |
|---|---|
| `spectra_examples_bands` | one galaxy, the output under each configuration |
| `residual_river_bands` | SR2 minus target, sorted by z, and its scatter |
| `psd_bands` | signal and residual power spectra |
| `line_recovery_bands` | recovered flux fraction per recoverability bin |
| `line_flux_bands` | Gaussian-fit line flux against the target, input and SR2 |
| `sn_comparison_bands` | per-line S/N, SR2 against the input |
| `response_bands` | the prior-dominance response exponent per bin |

The line-flux comparison is the one the others cannot replace. S/N never looks
at the target, so only flux against the target says whether a line is right.
The fit (`specsr_roman.evaluation.linefit`) is the one of the JWST companion
package. A configuration whose cache is missing is left out of the figures.

## Audits

Worth re-running after any retrain — this is the code that backs the
honesty claims.

### Photometry ablation

```bash
specsr-roman evaluate ablation
```

Answers: how much of the redshift accuracy is the *spectrum*? Because
photometry enters standardised with statistics baked into the checkpoint,
"drop a band" is exactly "feed it its training mean" — so this needs no
retraining.

The v0.6.0 reserved-test Roman chain has NMAD 0.00133 and 0.67% outliers.
The older release's colour-ablation numbers describe older weights. Re-run
ablations on the appropriate three-way partition before quoting new numbers.

:::{warning}
Read the no-photometry row as an upper bound, not as the information floor.
This head was *trained* with colours, so mean-imputing them measures the
deployed chain in a degraded mode — not a grism-only model, which is a
separate experiment that has not been run.
:::

### Prior dominance (inverse crime)

```bash
specsr-roman evaluate prior --max-sources 500
```

The training targets are simulated SEDs. A model can score well on every
reconstruction metric by learning that manifold rather than measuring
anything, and no reconstruction metric distinguishes the two.

This probes off-manifold sensitivity. Scale a recovered line in the *truth* by a factor `f` — off the
manifold — forward-model the difference onto the observed spectrum, re-run, and
measure

$$ r = \frac{\log(L_\mathrm{pred}' / L_\mathrm{pred})}{\log f} $$

`r = 1` means the model tracked the change; `r = 0` means it produced the same
line regardless.

**Read it carefully.** Where the injected change is genuinely below the noise,
low `r` is the *correct* behaviour — falling back on the prior is what a
calibrated model should do when the data says nothing. Bin by detectability and
judge `r` only where the information is present. Aggregate `r` mixes marginal and strong detections; inspect the injected-change
S/N breakdown as well as the original-line bins.

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

Use `--split-mode three_way --partition test` with the released checkpoint
names. Deployed `legacy_lr` restoration is the primary physical response.
The `target_fixed` convention is a normalized-output control, not a corrected
physical response or a fraction of information attributable to a prior.
