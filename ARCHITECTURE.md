# Architecture

How `specsr-roman` is put together, and why. The README says what it does; this
says how, at the level of detail you need to change it safely.

The organising constraint, repeated because every decision below follows from
it: **a model that invents plausible emission lines is worse than useless in a
survey.** A super-resolution network trained naively on simulated spectra will
learn to draw the lines its training manifold says *should* be there, and
every reconstruction metric will reward it for doing so. Most of the machinery
here exists to make that failure mode expensive.

---

## Package layout

```
src/specsr_roman/
  lines.py        rest-frame line lists; order is load-bearing
  grids.py        wavelength grids, instrument constants, photometric bands
  config.py       typed configs; defaults ARE the published chain
  checkpoints.py  local paths / Hub names / org:repo:name all resolve here
  paths.py        env-overridable data roots

  models/         blocks.py sr1.py zhead.py sr2.py
  data/           datasets.py splits.py augment.py transforms.py photometry.py
  training/       sr1.py zhead.py sr2.py losses.py common.py
  extraction/     frames.py catalog.py seds.py simulate.py extract.py batch.py
  inference/      pipeline.py
  evaluation/     cache.py metrics.py figures.py prior_dominance.py
  cli.py
```

Dependency direction is one-way: `models` → `data` → `inference` → `training`
→ `evaluation`. `training` imports `inference.build_sr2_input` rather than
duplicating it, so training, evaluation and user inference assemble SR2's
input identically. A divergence there would show up only as "the published
number does not reproduce", which is the worst kind of bug to hunt.

`extraction` is isolated behind the `[extract]` extra and imports grizli,
photutils, h5py and pyarrow lazily. Nothing else in the package touches them.

---

## The two shared grids

Everything lives on two fixed grids, defined once in `grids.py`:

- **LR**, 864 px: the Roman grism's native sampling (~10.764 Å/px) over
  1.0–1.93 µm. Extractions land here directly, so gridding neither creates nor
  destroys information.
- **HR**, 2500 px: the target grid, ~4.3× finer. Fine enough that a
  grism-resolution line profile is well sampled; coarse enough to keep the
  networks small.

The LR spectrum is interpolated onto the HR grid *at load time*. A fully
convolutional stack with no resampling layers cannot change the axis length,
and keeping input and output aligned makes the residual `pred - input`
meaningful pixel by pixel. This mirrors the JWST project's convention, which
is what made the cross-instrument warm-start experiments possible at all.

---

## Stage 1 — SR1

A 1D pre-activation ResNet: Conv1d stem → 16 residual blocks (120 channels,
GroupNorm + GELU) → two 1×1 heads for mean and log-variance. 1.4 M parameters.

**Two input channels, one scale.** SR1 sees `[flux, err]`, both divided by the
*same* per-row flux standard deviation, so the channel ratio is the per-pixel
S/N. The network can then matched-filter instead of guessing which bumps are
noise — the single largest quality jump in the Roman port. Normalising the
error channel independently would destroy exactly the quantity being read.

**Per-spectrum normalisation.** Not merely conditioning: the grizli extraction
is systematically ~1.7× brighter than the input SED (aperture losses), so an
absolute flux scale would teach the model a calibration error. Normalising per
row removes it and makes the task purely about shape.

**Residual scale α = 0.2 on the branch**, so a deep stack starts near the
identity and learning is spent on the delta. **Log-variance initialised to a
constant −2.0** with zero weights: the model must earn structure in its
uncertainty.

### Losses

The Roman losses differ from their JWST ancestors, and not cosmetically. The
JWST version got several scales for free from *target noise*: a robust MAD over
the target's derivative was a meaningful normaliser because the target had
noise. Roman's targets are noiseless simulated SEDs, so those denominators
collapse toward zero and terms balanced at O(1) arrive at 60× or 3500× the
NLL. Each is rescaled by a quantity measured **inside the line mask**, which is
O(1) in either regime.

| Term | What it does |
|---|---|
| Gaussian NLL | reconstruction, with predicted variance |
| gated sharpness (1st + 2nd derivative, line-masked) | teaches deblending — matching a complex's *shape*, not just its integral |
| `line_flux_loss_weighted` | integrated per-line L1, weighted by recoverability |
| `line_hallucination_loss` | penalises drawn amplitude the data cannot support |
| log-variance regularisation | stops the NLL being bought off by inflating σ |

The last two are the recoverability machinery, and they are asymmetric on
purpose.

**Why weighting was necessary.** About two thirds of Roman grism rows carry no
line the data could possibly reveal. Under a plain reconstruction loss the
optimal policy on those rows is to hedge — draw nothing, predict the prior mean
— and averaged over the majority of the training set, that is what SR1
converged to. Weighting each *present* line by `snr/(snr + snr0)` removes the
gradient from physically unrecoverable lines. *Absent* lines keep weight 1, so
drawing flux where there is none stays fully penalised: permissive about
failing to find the invisible, strict about inventing.

**Why weighting alone was not enough.** Given 300 epochs, SR1 still learned to
recite the prior, drawing ~23 % amplitude on lines that were present in the
target but had integrated LR S/N below 1 — indistinguishable, to a user, from
a real weak detection. `line_hallucination_loss` measures the model's *own*
drawn flux (high-pass of the prediction, integrated over each line window) and
penalises positive bumps by a non-recoverability weight.

**Why the knee is squared.** A first attempt used the complement of the
recovery weight, `snr0/(snr + snr0)`, which is still 0.25 at S/N 6 — a
quarter-strength drag on exactly the lines we want drawn — and it crushed
recoverable recovery from 0.75 to 0.20. The squared knee
`snr_h0² / (snr² + snr_h0²)` is 0.5 at `snr_h0`, 0.1 at 3× and 0.03 at 6×:
recoverable lines are essentially unpenalised while undetectable ones are
driven back to the continuum. Only positive bumps are penalised, so real
absorption is safe.

### Where recoverability labels come from

`RomanFixedGridDataset` precomputes, per row and per line in `SR1_LINES_AA`,
an integrated S/N on the native LR grid. Signal is the true line flux — the
continuum-subtracted *noiseless* target, converted into LR units via the
unit-H158-mean convention the extraction uses. Noise is the quadrature sum of
the extraction's own error array over a ±4-pixel window.

It answers, per line and per row: *could this line have been seen in this
spectrum at all?* Lines that could not are exactly the ones a model must not
draw, and the ones whose absence it must not be punished for.

`SR1_LINES_AA` order is load-bearing — the `line_snr` columns follow it and
the losses index it positionally. There is a test.

---

## Stage 2 — ZHead

Input: `[LR flux, LR err, SR1 mean, SR1 log-σ]` plus a wavelength ramp, and a
photometry branch. Output: logits over 310 redshift bins spanning 0 < z < 3.1.

**Why classification.** Redshift from a grism is line *identification*, and it
is intrinsically multimodal — one observed line is consistent with Hα, [O III],
[O II] or Lyα. A single Gaussian head must average between aliases, which is
what produced the ~40 % catastrophic-outlier floor of the two regression
designs that came first. A softmax over a grid carries the whole P(z): the
point estimate is the *mode* (never a mean between two lines), and ambiguity
lives in the PDF width. `pz_stats` refines within ±8 bins of the mode for
sub-bin accuracy without ever averaging across a distant alias, while reporting
σ over the *whole* grid — decisive estimate, honest error bar.

**Why the raw LR channels stay.** SR1 is the conservative stage and smooths
lines down to a few percent of their flux; a head reading only SR1's output was
starved. The LR spectrum still holds the line at its true observed wavelength,
noise and all.

**Why a wavelength ramp.** The convolutional stack is translation-equivariant
and global pooling discards position, so the first design had no absolute
positional signal and collapsed toward the prior mean (correlation 0.44,
predicted spread 0.27 against a true 0.60). Unlike the JWST prism — whose
varying resolution makes line *width* a wavelength cue — Roman's near-constant
R offers nothing to substitute. The ramp restores it, and attention pooling
reads the attended ramp value as a line centroid: the redshift, read out
explicitly.

**Why attention pooling.** Earlier heads pooled with weights derived from
SR1's smooth σ channel, which averaged a ~5-pixel emission line over 2500
pixels; the line could not steer the representation. Saliency now comes from
the flux features themselves.

### The photometry branch

Broadband colours over 0.35–2.1 µm break the single-line alias degeneracy —
exactly the information the grism band lacks. Fluxes enter raw; the branch
applies `log10` and standardises with train-split statistics carried in the
`phot_mu` / `phot_sig` buffers, so a checkpoint is self-describing.

The head takes Roman Medium-tier F106/F129/F158 — the three bands that ship
with the HLWAS grism — with 0.05 mag noise at train *and* eval. Band choice
here is a statement about what Roman will actually deliver, not a
hyperparameter to maximise: given a complete, noiseless SED a head can read
the redshift off the photometry alone, which measures the catalogue rather
than the instrument.

> **Naming trap.** OU2024 uses the *old* Roman band names, which collide with
> the current WFI scheme: OU2024 `R062` is 0.62 µm (current F062), and OU2024
> `W146` is the current wide filter R062. Always map by central wavelength,
> never by name. `grids.PHOT_BANDS` fixes the index order.

---

## Stage 3 — SR2

Two branches producing a delta on SR1's output.

**Line branch.** One token per rest-frame feature in `LINE_LIST_REST_AA` (98
of them). Each token reads a ±25 px window at that line's predicted observed
position, tokens cross-attend through a 4-layer transformer — so [O III]
4959/5007 and Hα/[N II] can agree on a consistent picture — and each decodes to
a gated Gaussian (amplitude × presence × in-band) scattered back onto the
wavelength axis.

**CNN branch.** A residual stack for the continuum and everything between
lines. Its output layer is zero-initialised, so SR2 begins as the identity on
SR1 and any delta it produces is learned rather than initialisation noise.

**Multi-hypothesis placement.** The line branch runs once per redshift
hypothesis (top 3 P(z) modes) and the deltas are combined weighted by mode
mass. This is what makes the stage alias-robust: when the top mode is wrong,
the correct identification is usually the second or third, and the right lines
still get drawn at reduced weight.

An over-complete line list is deliberate — a token whose line falls outside the
band is gated off by `in_range`, costing a little compute and buying robustness
across the redshift range.

### Three SR2 settings that are not defaults by accident

**`delta_cap = 40`.** Normalised SED lines peak above 30. An earlier cap of 3
made them literally unreachable and saturated the gradient into "predict flat".

**`lam_z = 0`** — the coupled z-loss is off. It is a JWST holdover from the
spectra-only regime, where it rewarded SR2 for drawing lines the ZHead could
read back. With a photometry-fed ZHead the redshift is already known, so the
term bought nothing and inflated hallucination amplitude to 2.6× truth during
warm-up. (When it *is* enabled, the LR channels are zeroed in the re-read —
left in, the head reads z straight off the raw LR spectrum and applies zero
line-drawing pressure.)

**Supervised presence.** We know which lines each noiseless target has, so the
presence head is trained with BCE against integrated-flux labels. Two earlier
generations left it to a sparsity prior; presence collapsed to zero both times,
taking every line the stage was meant to draw with it.

### Checkpoint goal

Plain validation NLL is continuum-dominated and reliably selects the SR2 that
draws nothing. A goal based on line-region MSE ratio was no better — diluted
across 98 windows it is blind to sharpening, and it picked the most timid epoch
available (strong-line amplitude 0.60, *worse* than the SR1 it was meant to
improve).

The goal used is `-recov_amp + lam_hallu * hallu_amp`: integrated
predicted-over-true flux on *recoverable* strong lines against the same
quantity on *undetectable* ones. It selects the sharpest epoch that is not yet
hallucinating — epoch 4 on the published run. That is the design working:
hallucination amplitude climbs from 0.26 to 0.62 by epoch 150 while
recoverable amplitude barely moves.

---

## Data pipeline

OpenUniverse2024 provides direct images, truth catalogues and Diffsky SEDs but
no grism images, so `specsr-roman` disperses the scene with grizli and extracts it
back (following Guo et al. 2025). Input and target are then self-consistent by
construction: the same `.conf` disperses and extracts, so any residual is
noise, blending or the extraction — not a mismatched instrument model.

Per (visit, SCA):

1. **`frames.prepare_frames`** — counts → e⁻/s with the flat sky removed;
   writes a grizli direct image plus an empty grism shell sharing its WCS,
   tagged `INSTRUME='WFI'` so grizli loads `Roman.G150.conf`.
2. **`catalog.detect_and_relabel`** — photutils detection, KD-tree matched to
   the truth index.
3. **`seds.SEDLibrary`** — Diffsky SEDs, summed over components, redshifted,
   flux-conservingly rebinned.
4. **`simulate.disperse_scene`** — every source brighter than `ab_scene`
   dispersed with its true SED.
5. **`simulate.add_grism_noise`** — Poisson (source + zodiacal) plus read
   noise, at 301 s.
6. **`extract.extract_target`** — the target re-dispersed alone, subtracted
   from the scene for exact contamination, then optimal extraction calibrated
   against a flat-f_λ pass through the same beam.

### Six things that will bite you

Each cost a debugging session; each is encoded in code with a comment.

- **`size=85, compute_size=False`.** The Roman trace sits up to ~66 px from
  the source on detector 1 (~162 px on detector 4), overflowing grizli's
  adaptive cutout and silently truncating the beam.
- **`pad=(120, 900)`.** Edge sources' beams fall out of the array otherwise.
  Detection coordinates are then in *padded* pixels — shift the truth
  coordinates or every match is wrong by hundreds of pixels.
- **Compact segmentation ids.** grizli keeps the segmentation as float32,
  which cannot represent OU2024's 13-digit `object_id` exactly. The map carries
  a small sequential id and a lookup restores the real one.
- **`is_cgs=False` means "multiplier against flat f_λ = 1".** The SED must be
  normalised to unity mean over the H158 band or the raw Diffsky values
  (~1e-24) disperse an effectively empty scene — a silent, total failure.
- **Adaptive SED wavelength grids.** Sub-Ångström bins at lines, coarse
  elsewhere. Point interpolation gets line flux wrong by a factor depending on
  where samples land. Always `fluxconserve_resample`.
- **Descending wavelengths.** Roman's `DLDP_A_1` is negative, so
  `optimal_extract` returns wavelengths in descending order. `np.interp` does
  not check and silently returns the edge value everywhere, producing a flat
  "spectrum" with zero information. Use `transforms.interp_ascending`. The
  dataset loader also refuses outright if more than 1 % of rows are constant.

Setting `PHOTFLAM` makes grizli rescale the SCI array into f_λ internally, so
anything reading those pixels directly must be unit-agnostic.

### Splits

**Split by object, never by row.** The same OU2024 galaxy appears in many
visits as independent noise realisations; a row-wise split puts realisation A
in train and B in test and measures memorisation.

`get_or_make_group_split` hashes the object id. Two consequences, both
deliberate: every stage derives the same split from the same ids without
passing a file around, and the split is stable under dataset growth — adding
SCAs never moves an existing galaxy across the boundary, so a model trained on
an older build can still be evaluated on the newer test set.

`filter_split_min_lines` is applied *after* the split, never before: filtering
first would change which galaxies land on which side.

---

## Evaluation

`evaluation/cache.py` freezes the full chain's test-split output to one npz.
Every figure and every quoted number reads that file rather than a live model,
so a plotting tweak cannot quietly change a result.

`metrics.line_amplitude_recovery` is the published metric: per row, summed
predicted flux over summed true flux across line pixels, binned by the row's
best line S/N. `per_line_amplitude_recovery` is the diagnostic companion,
attributing a failure to a specific transition.

The **`unrecoverable` bin is the control.** A well-behaved model scores near
zero there. A model scoring 0.3 is inventing lines, however good its `strong`
number looks — and no single averaged metric would tell you.

The honesty claims are backed by code you can re-run:

- **`prior_dominance.py`** — the inverse-crime test. Scale a recovered line in
  the *truth* off-manifold by a factor f, forward-model the difference through
  a grism-resolution LSF onto the observed spectrum, re-run, and measure
  `r = log(L_pred' / L_pred) / log(f)`. r = 1 means the model tracked the
  change; r = 0 means it produced the same line regardless.

  Reading it requires care: where the injected change is genuinely below the
  noise, low r is the *correct* behaviour. Bin by detectability and judge r
  only where the information is present.

  One implementation detail is load-bearing. The Diffsky SEDs carry an internal
  flux scale (~1e-20) unrelated to the extraction's units, and the dataset
  hides the mismatch by normalising per spectrum. An injected delta computed in
  SED units is numerically invisible once added to the LR spectrum, which pins
  r at exactly zero and makes every model look like it recites the prior. The
  bridge is the least-squares scale between the LSF-smoothed truth and the
  observation — the empirical flux-calibration ratio.

---

## Extending it

**A new photometric tier**: add it to `grids.PHOT_TIERS`. Everything —
dataset slicing, the ZHead's `n_phot`, the CLI flag — follows from that one
entry. Keep the "ships with the grism" rule.

**A new line**: append to `LINE_LIST_REST_AA` (SR2 tokens; safe, over-complete
by design). Adding to `SR1_LINES_AA` changes the `line_snr` column layout and
invalidates existing SR1/SR2 checkpoints — the losses index it positionally.

**A new simulation**: implement the `frames` / `catalog` / `seds` trio for the
new products. `simulate`, `extract` and `batch` are format-agnostic. The
schema contract is the npz keys in `extraction/batch.py::merge`.

**A new stage**: follow the SR2 pattern — take frozen upstream stages by
checkpoint spec, assemble inputs through a shared function in `inference`, and
define a checkpoint goal that measures the thing you actually want, not the
loss.
