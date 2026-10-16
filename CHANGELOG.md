# Changelog

Notable changes to `specsr-roman`. Versions follow [semantic
versioning](https://semver.org/); until 1.0 the public API may still move.

## [0.6.0] — 2026-10-16

All nine models were trained from scratch with galaxy-grouped training,
validation and reserved test partitions. The default pipeline uses new
`_independent` checkpoint names; previous Hub weights remain available.
On 2,384 reserved test spectra the Roman chain has NMAD 0.00133 and 0.67%
catastrophic outliers. Roman SR2 line recovery is 0.90 / 0.94 / 0.89 for
marginal / good / strong detections. Earlier 4,711-spectrum results are
checkpoint-selection validation results, not an independent test.

### Added

- Galaxy-grouped three-way training/validation/test partitions, upstream
  checkpoint provenance checks, and prediction-cache partition metadata.
- Redshift and spectral uncertainty-coverage diagnostics, galaxy-grouped
  bootstrap summaries, and truth-centered unsupported-emission diagnostics.
- Reproducible response-audit runners, normalization controls, release
  manifests, and an isolated fresh-training workflow with detached execution.

### Fixed

- Response-audit eligibility now compares target and predicted line integrals
  in consistent units. Deployed LR restoration and fixed-target restoration
  are explicitly separate conventions, with per-response records.

### Changed

- The manuscript identifies checkpoint-selection results as validation,
  reports repeated response audits and uncertainty limitations, and pins the
  existing v0.5.0 software archive. The public software and model release
  reports the new reserved-test results separately.

## [0.5.0] — 2026-10-12

**The sample changed, and every number with it.** The training set is no
longer cut at an H158 magnitude of 22.5. Every galaxy detected in the images
is extracted, and the sample is the galaxies whose H-alpha or [O III] 5007
flux exceeds 1e-16 erg/s/cm2, the line-flux limit of the Roman spectroscopic
survey (Wang et al. 2022). All nine checkpoints are retrained under the same
names; the previous ones are on the Hub under `superseded/h158_lt_22p5/`.
Numbers from 0.4.0 and earlier describe a different sample and are not
comparable: the galaxies without a strong line in the band, where most
redshift outliers were, are no longer in it.

### Added

- **`specsr-roman extract select`** and `extraction.select_line_flux`: the
  training dataset is selected from the deep extraction by true line flux
  (`SURVEY_LINE_FLUX_LIMIT`). `extract run` / `merge` now build the deep
  dataset, every detected galaxy, with no magnitude cut (`ab_target` and
  `ab_scene` default to no cut). Both files are on the Hub.
- `evaluation.band_figures.nii_halpha_ratio`: [N II] 6583 / H-alpha in the
  target and in the output, where H-alpha is detected. In OpenUniverse2024
  the true ratio is nearly constant (median 0.08, 5th-95th percentiles
  0.072-0.084), so the dataset cannot test the recovery of an individual
  galaxy's ratio.
- The parent-sample figure shows the catalogue, the galaxies the images
  detect, and the ones above the line-flux limit.

### Fixed

- **Detections are identified by the brightest counterpart.** The truth index
  lists objects far below the detection limit, and the nearest one to a
  detection is often not the object detected. The match now takes the
  brightest object in the radius and keeps it only if its catalogue magnitude
  agrees with the detected flux within a magnitude. In the previous dataset
  0.6% of the spectra carried the identity of a fainter neighbour.
- `band_figures` reads each cache array once. Indexing an `NpzFile` row by
  row made the S/N figure take hours.
- The redshift-breakdown figure leaves out a group with no spectra.

### Results on the new sample

Held-out split, 4,711 spectra, 76% with a line at S/N >= 2 in one exposure.
Redshift with the three Roman bands: NMAD 0.0014 with 0.8% outliers; with
Rubin 0.0011; nine bands and no spectrum 0.0116. Line recovery in the
marginal, good and strong bins: 0.87, 0.95, 0.96. Response exponent of the
full pipeline: 0.17. On the previous model's test spectra that pass the new
selection the redshift NMAD was 0.0021 with 0.5% outliers, so the fall in the
outlier rate from 5% is the sample, and the gain in precision is the model.

### Changed

- The `sample` figure is one panel, the redshift distribution by strong-line
  content; the best-line S/N histogram is gone.
- `residual_river_bands` has a row for the grism input minus the target above
  the row for the output minus the target, each with its own colour scale.

## [0.4.0] — 2026-10-11

### Added

- **One SR2 per redshift configuration, and the evaluation to match.**
  `evaluation.band_figures` draws every SR2 evaluation once per redshift head
  (three Roman bands; with Rubin at one-year and ten-year depth; nine bands
  and no spectrum): example spectrum, residual maps, power spectra, line
  recovery, line S/N, the response test and the line-flux comparison.
  `specsr-roman evaluate band-figures` renders them and writes
  `band_metrics.json`. New configs `sr2_rubin_y1.yaml` and
  `sr2_photonly_rubin_y10.yaml`; `scripts/run_chain.sh sr2_bands` trains them.
  All four SR2 checkpoints and their prediction caches are on the Hub.
- **Line-flux comparison.** `evaluation.linefit` is the Gaussian line fit of
  the JWST companion package, with windows for the grism band. Fitted fluxes
  of the output and of the grism input are compared with the same fit on the
  target. For detected H-alpha the median ratio is 1.00 with 7% of lines off
  by more than a factor of two, against 1.05 and 10% for the input; for
  [O III] 5007, 0.89 and 5% against 0.91 and 25%.
- `zhead_use_spectrum` (SR2 config, prediction cache, audit): a redshift head
  trained on photometry alone gets its spectral channels blanked, as in its
  own training, while SR2 still reads the spectrum.
- The prediction cache stores `hr_mean`, `hr_std`, `lr_mean`, `lr_std` and
  `ab_h158`, which turn its normalised arrays back into flux. A cache written
  before them is completed from the dataset on load.
- `specsr-roman evaluate prior` audits the full pipeline as well as SR1 when
  given `--zhead` and `--sr2`, reports the response exponent per
  recoverability bin, and writes `outputs/prior_dominance.json` and the
  `response` figure.

### Changed

- **The prior-dominance audit uses every detected line.** Sources enter at an
  integrated S/N of 2, the threshold used everywhere else, and no longer at 5.
  On the 3,242 held-out spectra with a detected line the response exponent is
  0.25 for SR1 and 0.16 for the full pipeline (0.33 for both in the strong
  bin). The 0.28 quoted with 0.3.0 was SR1 on the first 500 sources above
  S/N 5.
- The project is described as physics-informed super-resolution; the phrase
  "recoverability-calibrated" is dropped, since the response test shows that
  line strengths largely follow the simulation.

## [0.3.0] — 2026-10-09

**Every number produced with an earlier version is superseded.** The
OpenUniverse2024 SEDs were read in the wrong flux units, so the dataset, the
published checkpoints and the prediction cache of 0.1.0 and 0.2.0 describe
spectra that were too red by a factor of wavelength squared. The dataset has
been rebuilt and every checkpoint retrained under the same names. The
pre-fix weights remain on the Hub under `superseded/pre_fnu_fix/`.

### Fixed

- **The SEDs are f_nu.** `galaxy_sed_<healpix>.hdf5` stores flux per unit
  frequency (skyCatalogs reads it with `flux_type='fnu'`), and
  `extraction.SEDLibrary` handed it to grizli and to the targets as f_lambda.
  Inputs and targets shared the error, a continuum 3.7 times too red across
  1.0-1.93 um, while the catalogue photometry stayed correct. The loader now
  converts on the native grid. With the right counts in the blue half of the
  band, 45% of the held-out spectra have a detected line (it was 33%).
- **Catalogue zero points.** `grids.AB_ZEROPOINT` is now measured from the
  simulation for every band of the catalogue (`scripts/derive_zeropoints.py`). The Rubin
  values taken from the lsst/throughputs curves were 0.2-0.5 mag too high, so
  the Rubin heads of the development tree saw a survey shallower than
  labelled.
- **`normalize` computes in float64.** Spectra of order 1e-24 underflowed in
  float32 and came out "normalised" to values in the hundreds. This affected
  about 1% of the pre-fix dataset and was the population behind the median
  power spectrum.
- SR1 uploads its checkpoint to W&B as an artifact, as the other stages do.

### Results with the retrained chain

Held-out split, 7,334 spectra. Redshift with the three Roman bands: NMAD
0.0043, 5.0% outliers (0.0065, 5.1% before). Line recovery in the marginal,
good and strong bins: 0.87, 0.92, 0.89 (0.64, 0.82, 0.87 before); it no longer
rises across the detected bins. Prior-dominance exponent of SR1: 0.28 (0.45
before).

### Added

- **Dataset figures.** `specsr-roman evaluate dataset-figures` renders one
  galaxy followed from the H158 image to the training pair, and the sample
  against the parent catalogue (`evaluation.dataset_figures`).
- `scripts/run_chain.sh rubin` trains the Rubin runs, with checkpoints passed
  as local paths. `specsr-roman evaluate ablation` takes `--sr1` and
  `--zhead`.

- **A Roman + Rubin photometric tier.** `phot_tier: medium_rubin` feeds the
  redshift head Rubin *ugrizy* alongside Y106/J129/H158. The Rubin bands carry
  the sky-limited noise of a coadd of stated depth (`rubin_depth_offset`,
  `grids.phot_flux_sigma`), the head floors non-detections at one sigma, and
  the depth travels with the checkpoint (`phot_sigma`). `grids.MAX_PHOT_BANDS`
  is now nine. Configs: `zhead_rubin_y10`, `zhead_rubin_y1`, `sr2_rubin_y10`.
- **Band-set comparison** in `specsr-roman evaluate ablation`: the published
  three-band head against the Rubin heads and a photometry-only control, for
  the whole split and by redshift and best-line S/N
  (`phot_band_comparison.{csv,png}`).
- **Leak controls** in the same command: zero-point offsets and an aperture
  mismatch applied to the Rubin bands at evaluation
  (`phot_leak_controls.csv`).
- **Per-band AB zero points** (`grids.AB_ZEROPOINT`). OU2024 fluxes are photon
  rates through each bandpass; the H158 anchor applies to H158 only.
- **`specsr_roman.prior`**: the external-prior analysis (P(z) cache, prior
  combination, alias and outlier diagnostics), merged from the former
  `roman_rubin_super_resolution` repository. Scripts in `scripts/prior/`.
- Training runs upload their best checkpoint to W&B as an artifact, with the
  resolved config, split record, dataset and upstream-checkpoint hashes and
  git commit.
- **The HLSS cosmology selection.** `evaluation.true_line_flux_cgs` measures
  true H-alpha and [O III] fluxes in erg/s/cm2 from the noiseless targets,
  calibrated by the catalogue H158 magnitude, and
  `evaluation.hlss_cosmology_sample` applies the Wang et al. (2022) cut
  (`HLSS_LINE_FLUX_LIMIT`). The prediction cache stores `ha_flux` and
  `oiii_flux`; a cache written before them is upgraded on load without
  running a model. The `sample` figure outlines the selected spectra.
  `RomanFixedGridDataset` exposes `ab_h158`.

### Changed

- **The lines are judged only where they are detected.** A line is scored,
  plotted or labelled only if its own integrated S/N in the grism data reaches
  2 (`evaluation.MIN_BEST_LINE_SNR`, `evaluation.detected_line_mask`; [N II]
  counts with H-alpha). This applies to line recovery, the S/N comparison and
  the example spectra. Sample statistics are not cut: redshifts, residuals,
  power spectra and the band comparison use every held-out spectrum. Training
  and the prediction cache are unchanged.
- **`RECOVERABILITY_BINS` has three bins**: marginal 2-3, good 3-6,
  strong > 6. The `unrecoverable` bin is gone. This is a breaking change for
  code that indexes the result of `line_amplitude_recovery` by that name.
- `line_amplitude_recovery` takes `z` and `wave_um`; with them it scores the
  detected lines only.
- The S/N comparison figure shows H-alpha and [O III] 5007 on logarithmic
  axes, and the example-spectra figure labels only detected lines. The sample
  figure marks the region below S/N 2, and the redshift breakdown and the band
  comparison bin S/N at 2, 3 and 6.
- `specsr-roman evaluate ablation` writes `phot_band_zpred.npz` and
  `phot_band_zpred.png`: predicted against true redshift, one panel per
  photometric configuration.
- README, docs, architecture notes and the tutorial notebook (re-executed)
  quote the new sample, bins and numbers.

## [0.2.0] — 2026-10-02

### Added

- **Three figures that replace tables**, rendered by
  `specsr-roman evaluate figures` from the same frozen cache as the rest:
  `sample` (redshift distribution by strong-line content, and the best-line
  S/N distribution with the four recoverability bins), `recovery` (recovered
  line-flux fraction per bin for SR1 and the full chain, medians with 16–84 %
  ranges) and `zbreak` (redshift scatter and outlier rate by line content and
  by recoverability).

- **`tutorials/01_getting_started.ipynb`** — an executable walkthrough from
  install to the published numbers: one super-resolved spectrum, the redshift
  PDF and what its secondary modes mean, what photometry buys and why it must
  be noisy, and line recovery split by recoverability. Runs in about two
  minutes on a CPU. Committed with its outputs and rendered into the docs from
  those outputs, so a docs build never re-runs a model or republishes an
  unreviewed number.
- **A 512-row tutorial subset** of the dataset, on the Hub under `tutorial/`
  (3.8 MB). Drawn from the held-out side of the canonical object-id split and
  sampled uniformly within it, so metrics computed on it are honest
  out-of-sample numbers over the population's real mix of recoverable and
  undetectable lines. Built by `scripts/make_tutorial_dataset.py`.

### Changed

- **Roman Medium-tier photometry only, everywhere.** `grids.PHOT_TIERS` now
  offers `medium` alone, and `grids.MAX_PHOT_BANDS` refuses an explicit band
  list longer than the three bands that fly with the grism. Wider tiers let a
  model read the redshift off an effectively complete simulated SED, which
  scores well and means nothing on the sky.
- **The photometry ablation is Roman Medium-tier and ships.**
  `specsr-roman evaluate ablation` now measures the published three-band head
  with and without its colours and sweeps the photometric noise, and writes
  `phot_ablation.png` alongside the CSV.
- **Corrected the grism-only redshift number.** The README, the model card, the
  quickstart and the `RomanPipeline` docstring reported a figure taken from a
  masking ablation on a different, superseded head. Measured directly on the
  published chain, `phot=None` gives **NMAD 0.014 / 26 % catastrophic** over
  the same 7,334-row held-out split. All four now carry the measured number,
  and say plainly that `phot=None` is mean imputation on a head trained *with*
  photometry rather than a grism-only model — so it is not a measurement of
  the information floor either.
- Docs render notebooks via `myst-nb` (which replaces `myst-parser` in the
  `docs` extra and loads it itself).
- **The power-spectrum figure plots the median across spectra, not the mean.**
  Under 1 % of the held-out split (z < 0.25, targets whose normalised peaks
  reach the hundreds) carries more than 99 % of the summed power, so the mean
  described those few rows and sat outside its own 16–84 % band. The shaded
  bands are unchanged.
- **Figure labels.** The per-line S/N panels report the number of spectra and
  the S/N > 10 fractions in a box clear of the points; the redshift figure no
  longer carries a title; the photometry-ablation panels name the bands Y106,
  J129 and H158 and use "outlier rate".

### Fixed

- The README credited the OpenUniverse2024 simulation to "Troxel et al.
  (2025)". The paper's first author is the OpenUniverse collaboration; it is
  now cited as OpenUniverse et al. (2025).

## [0.1.0] — 2026-08-26

First packaged release. The science and the trained models predate it; this
version turns a working set of scripts into an installable, tested library.

### Added

- **`specsr-roman` package** with a one-way dependency structure —
  `models` → `data` → `inference` → `training` → `evaluation`. Training and
  evaluation share `inference.build_sr2_input`, so all three paths assemble
  SR2's input identically.
- **`RomanPipeline.from_pretrained()`** — the published three-stage chain in
  four lines, fetching weights from the Hugging Face Hub on first use. Accepts
  a single spectrum or a batch; returns super-resolved flux, per-pixel
  uncertainty, the full P(z), and per-line presence.
- **`specsr-roman` CLI** — `extract`, `train`, `predict`, `evaluate`, `info`.
- **Typed configs** (`specsr_roman.config`) with `configs/{sr1,zhead,sr2}.yaml`
  reproducing the published chain exactly. Unknown keys raise rather than
  silently doing nothing.
- **Checkpoint resolution** accepting a local path, a bare Hub run name, or
  `org/repo:name`, so nothing hard-codes a machine layout.
- **Test suite** (97 tests) covering loss *behaviour*, not just shapes: that
  the hallucination penalty leaves recoverable lines alone, that the flux-
  conserving rebin preserves a narrow line, that a descending wavelength grid
  is caught, that the group split cannot leak an object across train/test.
- `ARCHITECTURE.md` — the design and the reasoning behind each choice.

### Changed

- **Extraction, training and evaluation scripts became library modules.** The
  old `sys.path` cross-imports between sibling training directories are gone;
  everything now lives under `specsr_roman.*` with a one-way dependency structure.
- **Photometric band selection is now a named tier** (`medium`, `deep`, `all`)
  applied at dataset load, rather than an ad-hoc in-place array slice after
  construction. The dataset and the model it feeds can no longer disagree
  about band count.
- **Split records now live beside the dataset** rather than inside a training
  directory. Group-split *membership* is unaffected — it is a pure hash of the
  object id — and was verified bit-identical to the previously recorded split.
- Figure code moved to `specsr_roman.evaluation.figures` with a `make_figures`
  dispatcher. Output verified byte-identical to the figures in the manuscript.

### Fixed

These are fixes relative to the previous script-based workflow — bugs that
were live in this repository, not in unreleased code.

- **The prior-dominance audit now runs, and gives a real answer.** It had been
  unusable on OU2024: the Diffsky SEDs carry an internal flux scale (~1e-20)
  unrelated to the extraction's units, so the injected off-manifold
  perturbation was numerically invisible once added to the LR spectrum. That
  pinned the response exponent at exactly 0.000, which reads as "the model
  recites the prior" and is indistinguishable from a real, catastrophic
  result. Injection now goes through a fitted calibration bridge (the
  least-squares scale between the LSF-smoothed truth and the observation).
  The published SR1 scores **r ≈ 0.45**.
- **An unanchored `data/` rule in `.gitignore` matched `src/specsr_roman/data/`**,
  which would have excluded the data subpackage from git, from `ruff`, and
  from any built wheel — `import specsr_roman` succeeding while `specsr_roman.data` did
  not exist. All artefact rules are now anchored to the repository root.
- Photometry standardisation statistics are computed from the train split
  only, via `data.photometry.standardization_stats`.
- The photometry ablation no longer reports a "spectrum zeroed" row. Zeroing
  the spectral channels is out of distribution for SR1 — unlike masking a
  photometric band, which standardises to a value the network sees constantly
  — and it scores *worse* than masking every band, measuring the OOD input
  rather than the photometry's contribution.
- `np.trapz` / `np.trapezoid` handled across the NumPy 2.0 rename.

### Known limitations

- Results are on the Diffsky manifold; see README → Limitations.
- Anti-prior augmentation is implemented but not used by the published SR1 —
  it improves data-faithfulness at the cost of absolute line recovery.
- The ~5 % catastrophic redshift rate is a physical information floor for a
  single-line grism, not a tuning target.
