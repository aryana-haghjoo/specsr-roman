"""Checkpoint resolution and loading.

A checkpoint may be named three ways, and every entry point accepts all three:

* a local path --- ``runs/sr1/sr1_ou2024_v6_best.pth``;
* a bare run name --- ``sr1_ou2024_v6``, fetched from the Hugging Face Hub;
* a fully qualified hub reference --- ``org/repo:run_name``.

Weights are downloaded once into the standard Hugging Face cache and reused,
so the published pipeline runs from a clean checkout with no manual downloads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

__all__ = ["DEFAULT_HUB_REPO", "CANONICAL_CHAIN", "resolve_checkpoint",
           "load_state_dict", "load_sr1", "load_zhead_ckpt", "load_sr2",
           "push_checkpoint"]

DEFAULT_HUB_REPO = os.environ.get(
    "SPECSR_ROMAN_HUB_REPO", "aryana-haghjoo/roman-spectral-superresolution")

#: The three checkpoints that make up the published pipeline. Anything else on
#: the Hub is a superseded generation kept for provenance.
CANONICAL_CHAIN = {
    "sr1": "sr1_ou2024_v6",
    "zhead": "zhead_ou2024_roman_med3_noisy",
    "sr2": "sr2_ou2024_v5_romanonly",
}


def resolve_checkpoint(spec: str, repo_id: str | None = None) -> Path:
    """Checkpoint spec -> a local file path, downloading from the Hub if needed."""
    p = Path(spec).expanduser()
    if p.exists():
        return p

    if ":" in spec:
        repo_id, run_name = spec.split(":", 1)
    else:
        run_name = spec
        repo_id = repo_id or DEFAULT_HUB_REPO

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"{spec!r} is not a local file and huggingface_hub is not installed; "
            "install it with `pip install specsr-roman[hub]` or pass a local path"
        ) from exc

    return Path(hf_hub_download(repo_id=repo_id,
                                filename=f"checkpoints/{run_name}/model.pth"))


def checkpoint_card(spec: str, repo_id: str | None = None) -> dict[str, Any] | None:
    """The ``card.json`` beside a hub checkpoint: metrics, dataset, W&B link."""
    if Path(spec).expanduser().exists():
        card = Path(spec).expanduser().with_name("card.json")
        return json.loads(card.read_text()) if card.exists() else None
    if ":" in spec:
        repo_id, run_name = spec.split(":", 1)
    else:
        run_name, repo_id = spec, repo_id or DEFAULT_HUB_REPO
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id,
                               filename=f"checkpoints/{run_name}/card.json")
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def load_state_dict(spec: str, map_location="cpu", repo_id: str | None = None) -> dict:
    return torch.load(resolve_checkpoint(spec, repo_id), map_location=map_location)


def load_sr1(spec: str | None = None, device="cpu", repo_id: str | None = None,
             hidden_dim: int = 120, num_res_blocks: int = 16):
    """Frozen SR1, ready for inference. Input width is read off the checkpoint."""
    from .models.sr1 import SuperRes1D
    spec = spec or CANONICAL_CHAIN["sr1"]
    state = load_state_dict(spec, map_location=device, repo_id=repo_id)
    model = SuperRes1D.from_state_dict(state, hidden_dim=hidden_dim,
                                       num_res_blocks=num_res_blocks).to(device)
    return model.eval()


def load_zhead_ckpt(spec: str | None = None, device="cpu", repo_id: str | None = None):
    """Whichever ZHead generation the checkpoint holds, ready for inference."""
    from .models.zhead import load_zhead
    spec = spec or CANONICAL_CHAIN["zhead"]
    state = load_state_dict(spec, map_location=device, repo_id=repo_id)
    return load_zhead(state).to(device).eval()


def load_sr2(spec: str | None = None, device="cpu", repo_id: str | None = None,
             in_channels: int = 6, line_rest_um=None, wave_hi_um=None):
    """Frozen SR2. Falls back to the packaged line list and HR grid."""
    import numpy as np

    from .grids import WAVE_HR
    from .lines import LINE_LIST_REST_AA, angstrom_to_micron
    from .models.sr2 import SR2Attention

    spec = spec or CANONICAL_CHAIN["sr2"]
    if line_rest_um is None:
        line_rest_um = angstrom_to_micron([w for _, w in LINE_LIST_REST_AA])
    if wave_hi_um is None:
        wave_hi_um = WAVE_HR.astype(np.float32) * 1e-4
    state = load_state_dict(spec, map_location=device, repo_id=repo_id)
    model = SR2Attention(in_channels, line_rest_um, wave_hi_um).to(device)
    model.load_state_dict(state)
    return model.eval()


def push_checkpoint(pth_path: str | Path, run_name: str,
                    meta: dict | None = None,
                    repo_id: str | None = None) -> str:
    """Upload a trained checkpoint plus a provenance card to the Hub.

    The card is not optional decoration. Local disk on a shared machine is not
    a backup, and a ``.pth`` with no record of the dataset, the upstream
    checkpoints, the config and the W&B run is an unreproducible artefact
    within a week.
    """
    import time

    from huggingface_hub import HfApi

    repo_id = repo_id or DEFAULT_HUB_REPO
    api = HfApi()
    card = {
        "run_name": run_name,
        "file": Path(pth_path).name,
        "uploaded": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(meta or {}),
    }
    api.upload_file(path_or_fileobj=str(pth_path),
                    path_in_repo=f"checkpoints/{run_name}/model.pth",
                    repo_id=repo_id)
    api.upload_file(
        path_or_fileobj=json.dumps(card, indent=2, default=str).encode(),
        path_in_repo=f"checkpoints/{run_name}/card.json",
        repo_id=repo_id)
    url = f"https://huggingface.co/{repo_id}/tree/main/checkpoints/{run_name}"
    print(f"pushed {pth_path} -> {url}")
    return url
