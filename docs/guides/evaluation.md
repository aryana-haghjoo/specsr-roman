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

Removing all three colours takes the published head from NMAD 0.0043 / 5.0 %
catastrophic to **0.0053 / 16.6 %**, so the colours carry most of the
alias-breaking. The same run sweeps the photometric noise: even with
*noiseless* colours the outlier rate is 4.9 % rather than zero, which is what a
head reading its spectrum should look like.

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

This one does. Scale a recovered line in the *truth* by a factor `f` — off the
manifold — forward-model the difference onto the observed spectrum, re-run, and
measure

$$ r = \frac{\log(L_\mathrm{pred}' / L_\mathrm{pred})}{\log f} $$

`r = 1` means the model tracked the change; `r = 0` means it produced the same
line regardless.

**Read it carefully.** Where the injected change is genuinely below the noise,
low `r` is the *correct* behaviour — falling back on the prior is what a
calibrated model should do when the data says nothing. Bin by detectability and
judge `r` only where the information is present. Aggregate `r` is dominated by
undetected lines and understates a good model.

The published SR1 and the full pipeline both score r ≈ 0.30 on OU2024 (936
held-out sources with a line above S/N 5), rising from 0.22 just above S/N 5 to
0.41 for the best-detected third. Pass `--zhead` and `--sr2` to audit the full
pipeline.
