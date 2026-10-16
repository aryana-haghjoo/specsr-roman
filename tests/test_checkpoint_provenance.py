"""Published independent weights must retain their training provenance."""

from pathlib import Path

import pytest

from specsr_roman.checkpoints import resolve_checkpoint


def test_independent_hub_weights_download_provenance_from_same_revision(monkeypatch, tmp_path):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    calls = []
    snapshot = tmp_path / 'snapshots' / ('a' * 40)

    def download(*, repo_id, filename, revision=None):
        calls.append((repo_id, filename, revision))
        path = snapshot / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'weights' if filename.endswith('.pth') else b'{}')
        return str(path)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    path = resolve_checkpoint('example/repo:sr1_independent')
    assert path.with_suffix('.provenance.json').exists()
    assert calls == [
        ('example/repo', 'checkpoints/sr1_independent/model.pth', None),
        ('example/repo', 'checkpoints/sr1_independent/model.provenance.json', 'a' * 40),
    ]
    assert Path(resolve_checkpoint(str(path))) == path
