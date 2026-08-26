# Installation

`specsr-roman` is not on PyPI yet, so install it from the repository:

```bash
GH=git+https://github.com/aryana-haghjoo/specsr-roman
pip install "specsr-roman @ $GH"               # inference + training
pip install "specsr-roman[hub] @ $GH"          # + published checkpoints and data
pip install "specsr-roman[extract] @ $GH"      # + dataset building
pip install "specsr-roman[all] @ $GH"          # everything
```

Once it is released, `pip install specsr-roman` (with the same extras) will do
the same thing, and the lines above keep working either way.

Python 3.10 or newer.

## Extras

| Extra | Pulls in | Needed for |
|---|---|---|
| *(base)* | numpy, scipy, torch, astropy, matplotlib | inference, training |
| `hub` | huggingface_hub | fetching published checkpoints and data |
| `train` | wandb | metric syncing during training |
| `extract` | grizli, photutils, h5py, pyarrow | building datasets from raw sims |
| `docs` | sphinx, furo, myst-parser | building these pages |
| `dev` | pytest, ruff | the test suite |

The extraction stack is heavy and only needed if you are re-extracting rather
than downloading the released dataset. Everything else in the package works
without it.

## Checking an install

```bash
specsr-roman info
```

reports the version, whether CUDA is visible, which optional extras are
present, and the canonical checkpoint names.

## GPU

Training wants a GPU; inference on a handful of spectra is fine on CPU.

On Blackwell cards (RTX 50-series with CUDA 12.8) the fused attention kernels
return NaN in backward for SR2's line-token transformer, so `specsr-roman` forces
the math SDPA backend during SR2 training. The transformer is 98 tokens wide,
so the cost is negligible.

## Configuring paths

Data locations are environment-overridable and default to the working
directory:

| Variable | Default | What |
|---|---|---|
| `SPECSR_ROMAN_DATA` | `./data` | raw and prepared simulation products |
| `SPECSR_ROMAN_DATASETS` | `$SPECSR_ROMAN_DATA/dataset` | built training datasets |
| `SPECSR_ROMAN_RUNS` | `./runs` | checkpoints and predictions |
| `SPECSR_ROMAN_HUB_REPO` | `aryana-haghjoo/roman-spectral-superresolution` | checkpoint source |
| `GRIZLI` | `./grizli_conf` | grizli `CONF` tree (extraction only) |
