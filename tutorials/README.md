# Tutorials

Executable notebooks. Each one is committed **with its outputs**, so it can be
read straight through on GitHub or on the
[docs site](https://aryana-haghjoo.github.io/specsr-roman/) without running
anything — and the docs render exactly the run that was reviewed rather than
re-executing on every build.

| Notebook | What it covers |
|---|---|
| [`01_getting_started.ipynb`](01_getting_started.ipynb) | Install → published checkpoints → one super-resolved spectrum → the redshift PDF → what photometry buys → line recovery split by recoverability. ~2 min on a CPU. |

## Running them

```bash
pip install "specsr-roman[hub]" jupyter
jupyter lab tutorials/
```

Every input is downloaded from the Hub on first use — a 512-row
[tutorial subset](https://huggingface.co/datasets/aryana-haghjoo/romansr-data/blob/main/tutorial/ou2024_h10307_tutorial.npz)
of the training data (3.8 MB) and the three published checkpoints (13 MB).
Nothing here needs the full 271 MB dataset or a GPU.

## The tutorial dataset

The subset is drawn from the **held-out side** of the canonical object-id
split, as a uniform random sample within it — no cherry-picking on brightness
or line strength. Both properties are load-bearing: the first makes every
number in the notebook an honest out-of-sample number, and the second keeps the
population's real mix of recoverable and undetectable lines, which is what the
recoverability-binned metric needs in order to mean anything.

Rebuild it from the full dataset with:

```bash
python scripts/make_tutorial_dataset.py            # writes it locally
python scripts/make_tutorial_dataset.py --push     # and uploads it
```

The row selection is a seeded permutation, so it reproduces exactly.

## Re-running a notebook before committing it

Outputs are part of the file, so refresh them deliberately:

```bash
jupyter nbconvert --to notebook --execute --inplace tutorials/01_getting_started.ipynb
```

Check the diff before committing. Numbers that move are either a real change in
the chain or a change in what the notebook measures — both are worth a look,
and neither should arrive unnoticed inside a docs build.
