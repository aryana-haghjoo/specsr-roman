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
line_amplitude_recovery(cache["sr2"], cache["hr"], cache["line_snr"])
```

`redshift_summary` reports NMAD, median |Δz|/(1+z), catastrophic fraction and
N. Report all of them: a model can shrink NMAD while pushing more objects past
the catastrophic threshold, and NMAD alone hides alias structure entirely.

## Read the recoverability split first

`line_amplitude_recovery` bins rows by their best line's integrated S/N:

| Bin | Integrated line S/N |
|---|---|
| `unrecoverable` | < 1 |
| `marginal` | 1–3 |
| `good` | 3–6 |
| `strong` | > 6 |

**The `unrecoverable` bin is the control.** A well-behaved model scores near
zero there — it declines to draw what the data cannot support. A model scoring
0.3 in that bin is inventing lines, however good its `strong` number looks, and
a single averaged amplitude ratio would not tell you.

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

Removing all three colours takes the published head from NMAD 0.0064 / 5.3 %
catastrophic to **0.0143 / 26.2 %**, so the colours carry most of the
alias-breaking. The same run sweeps the photometric noise: even with
*noiseless* colours the outlier rate is 3.9 % rather than zero, which is what a
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
unrecoverable cases and understates a good model.

The published SR1 scores r ≈ 0.45 on OU2024.
