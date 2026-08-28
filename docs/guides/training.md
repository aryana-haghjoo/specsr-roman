# Training

Every published checkpoint is reproducible from a config file.

```bash
specsr-roman train sr1   --config configs/sr1.yaml
specsr-roman train zhead --config configs/zhead.yaml     # needs SR1
specsr-roman train sr2   --config configs/sr2.yaml       # needs SR1 + ZHead
```

Or from Python:

```python
from specsr_roman.config import SR1Config, load_config
from specsr_roman.training import train_sr1

summary = train_sr1(load_config(SR1Config, "configs/sr1.yaml"))
```

The shipped configs *are* the published chain — `configs/sr1.yaml` produces
`sr1_ou2024_v6` — and the dataclass defaults match them. Any field can be
overridden (`--epochs 50`, `--no-augment`); an unknown key raises rather than
silently doing nothing.

## The three stages

**SR1** maps the LR spectrum to a mean and a heteroscedastic log-variance. It
sees `[flux, err]` on a shared scale, so the channel ratio is the per-pixel
S/N.

**ZHead** freezes SR1 and predicts P(z) over 310 bins from
`[LR flux, LR err, SR1 mean, SR1 log-σ]` plus photometry.

**SR2** freezes both and learns a delta on SR1, with one attention token per
rest-frame line, run once per redshift hypothesis.

## Two settings that must not drift

```yaml
phot_tier: medium         # bands that ship with the grism
phot_eval_mag_err: 0.05   # never score against noiseless truth photometry
```

`medium` is the only tier, and `grids.MAX_PHOT_BANDS` refuses an explicit band
list longer than three. Feeding a model more bands than the survey delivers
with the grism hands it an effectively complete, noiseless SED, from which the
redshift can be read without the spectrum contributing anything —
`specsr-roman evaluate ablation` exists to keep that visible.
Setting `phot_eval_mag_err: 0` reports an accuracy nobody will reproduce on
sky.

## Checkpoint selection is not the validation loss

This is the part most worth understanding before changing anything.

**SR1** monitors line-flux recovery plus the hallucination penalty, smoothed by
an EMA — not the total loss. The validation loss is dominated by the NLL, which
*rises* as the model commits to line amplitudes: a confident half-amplitude
line costs more than a hedged flat one. An early run had every line-recovery
metric improving to epoch 150 while the loss picked a mid-run "best".

**SR2** minimises `-recov_amp + lam_hallu * hallu_amp` — integrated
predicted-over-true flux on *recoverable* strong lines against the same on
*undetectable* ones. Plain NLL is continuum-dominated and reliably selects the
SR2 that draws nothing; a line-region MSE ratio is diluted across 98 windows
and picks the timidest epoch available.

:::{note}
**SR2's best epoch is 4.** That is the design working, not a truncated run:
hallucination amplitude climbs from 0.26 to 0.62 by epoch 150 while
recoverable amplitude barely moves. The goal deliberately selects the sharpest
epoch before hallucination runs away.
:::

## Logging

Runs sync to Weights & Biases when it is installed, logging example spectra and
residual histograms alongside scalars. That is not decoration: a scalar loss
curve cannot tell you a super-resolution model has converged to the prior mean.
The loss looks fine; only a picture of a predicted spectrum next to its target
shows the flat line.

Set `wandb_mode: disabled` to turn it off; training never fails because a
metrics service is unreachable.

## Publishing checkpoints

```yaml
push_to_hub: true
hub_repo: your-org/your-repo
```

Uploads the best checkpoint with a `card.json` recording metrics, dataset,
upstream checkpoints, resolved config and the W&B run URL. A `.pth` without
that record is an unreproducible artefact within a week.
