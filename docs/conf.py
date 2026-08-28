"""Sphinx configuration.

Warnings are errors in CI: a dangling cross-reference is a broken link in the
published docs, and the cheapest place to catch it is here.
"""

import importlib.metadata
import pathlib
import shutil

project = "specsr-roman"
author = "Aryana Haghjoo"
copyright = "2026, Aryana Haghjoo"

try:
    release = importlib.metadata.version("specsr-roman")
except importlib.metadata.PackageNotFoundError:  # building from a source tree
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # myst_nb supersedes myst_parser -- it loads it itself, and listing both
    # raises "extension already registered". It is what renders the notebooks.
    "myst_nb",
]

myst_enable_extensions = ["colon_fence", "deflist"]
# myst_nb registers .md and .ipynb itself; only .rst needs naming here.
source_suffix = {".md": "myst-nb", ".rst": "restructuredtext",
                 ".ipynb": "myst-nb"}

# Render the notebooks from the outputs they were committed with. Executing
# them here would make every docs build download checkpoints and a dataset,
# and would silently republish numbers nobody reviewed; the notebook in git is
# the one that was run and read.
nb_execution_mode = "off"
nb_merge_streams = True

# The tutorials live at the repository root, where someone browsing the repo
# finds them without digging through docs/. Sphinx only reads sources under
# this directory, so copy them in at build time rather than keeping a second
# copy in git that would drift from the first.
_DOCS = pathlib.Path(__file__).resolve().parent
_TUTORIALS = _DOCS.parent / "tutorials"
if _TUTORIALS.is_dir():
    dest = _DOCS / "tutorials"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir()
    for src in sorted(_TUTORIALS.glob("*.ipynb")):
        shutil.copy2(src, dest / src.name)

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"specsr-roman {version}"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# torch and the extraction stack are heavy and partly optional; documenting
# signatures does not require importing them.
autodoc_mock_imports = ["torch", "grizli", "photutils", "h5py", "pyarrow",
                        "wandb", "huggingface_hub"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

nitpicky = False
