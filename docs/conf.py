"""Sphinx configuration.

Warnings are errors in CI: a dangling cross-reference is a broken link in the
published docs, and the cheapest place to catch it is here.
"""

import importlib.metadata

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
    "myst_parser",
]

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

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
