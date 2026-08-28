"""The shipped notebooks.

Tutorials are committed **with their outputs**, and the docs render those
outputs rather than re-executing (``nb_execution_mode = "off"`` in
``docs/conf.py``). That makes the stored outputs published material: a
traceback left in a cell, or a notebook cleared of outputs before committing,
becomes a page on the docs site that nobody looked at.

These checks are cheap and offline -- they read the JSON, they do not run
anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TUTORIALS = Path(__file__).resolve().parent.parent / "tutorials"


def _notebooks():
    return sorted(TUTORIALS.glob("*.ipynb")) if TUTORIALS.is_dir() else []


if not _notebooks():  # installed wheel, or a tree without the tutorials
    pytest.skip("no tutorials directory in this tree", allow_module_level=True)


@pytest.fixture(scope="module", params=[p.name for p in _notebooks()])
def notebook(request):
    return json.loads((TUTORIALS / request.param).read_text())


def test_notebook_parses(notebook):
    assert notebook["cells"], "notebook has no cells"
    assert notebook["nbformat"] >= 4


def test_every_code_cell_ran(notebook):
    """An unexecuted cell means the committed outputs are stale."""
    unrun = [i for i, c in enumerate(notebook["cells"])
             if c["cell_type"] == "code" and c.get("execution_count") is None]
    assert not unrun, f"code cells never executed: {unrun}"


def test_no_cell_raised(notebook):
    """A traceback in a stored output is a traceback published in the docs."""
    failed = [i for i, c in enumerate(notebook["cells"])
              if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    assert not failed, f"cells with error outputs: {failed}"


def test_outputs_are_present(notebook):
    """At least the figures survived. A cleared notebook renders as bare code."""
    images = sum(1 for c in notebook["cells"]
                 for o in c.get("outputs", [])
                 if "image/png" in o.get("data", {}))
    assert images, "no figure outputs stored; was the notebook cleared?"


# Split so the literals do not appear in this file: the public-release script
# greps the whole assembled tree for them, and a test that names what it
# forbids would fail that scan itself.
_HOME_PREFIXES = ("/" + "home/", "/" + "Users/")


def test_no_absolute_home_paths(notebook):
    """Executed cells can bake in a machine layout -- e.g. a Hub cache path.

    That leaks the layout and makes the notebook unrunnable anywhere else, so
    it is also what ``scripts/make_public_release.sh`` fails the build on.
    Catching it here means finding out before assembly rather than during.
    """
    blob = json.dumps(notebook)
    found = [pre for pre in _HOME_PREFIXES if pre in blob]
    assert not found, f"absolute path prefix in stored outputs: {found}"
