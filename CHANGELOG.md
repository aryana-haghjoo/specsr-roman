# Changelog

Notable changes to `specsr-roman`. Versions follow [semantic
versioning](https://semver.org/); until 1.0 the public API may still move.

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
- `np.trapz` / `np.trapezoid` handled across the NumPy 2.0 rename.

### Known limitations

- Results are on the Diffsky manifold; see README → Limitations.
- Anti-prior augmentation is implemented but not used by the published SR1 —
  it improves data-faithfulness at the cost of absolute line recovery.
- The ~5 % catastrophic redshift rate is a physical information floor for a
  single-line grism, not a tuning target.
