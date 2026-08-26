"""Config loading, CLI wiring, and the contract with the published checkpoints."""

from __future__ import annotations

import pytest

from specsr_roman.checkpoints import CANONICAL_CHAIN
from specsr_roman.cli import build_parser
from specsr_roman.config import SR1Config, SR2Config, ZHeadConfig, load_config, to_dict


@pytest.mark.parametrize("cls,path", [
    (SR1Config, "configs/sr1.yaml"),
    (ZHeadConfig, "configs/zhead.yaml"),
    (SR2Config, "configs/sr2.yaml"),
])
def test_shipped_configs_load(cls, path):
    cfg = load_config(cls, path)
    assert isinstance(cfg, cls)
    assert to_dict(cfg)


def test_shipped_configs_name_the_published_checkpoints():
    """The configs must reproduce the published chain, not something adjacent."""
    assert load_config(SR1Config, "configs/sr1.yaml").out_prefix == CANONICAL_CHAIN["sr1"]
    z = load_config(ZHeadConfig, "configs/zhead.yaml")
    assert z.out_prefix == CANONICAL_CHAIN["zhead"]
    assert z.sr1_ckpt == CANONICAL_CHAIN["sr1"]
    s = load_config(SR2Config, "configs/sr2.yaml")
    assert s.out_prefix == CANONICAL_CHAIN["sr2"]
    assert (s.sr1_ckpt, s.zhead_ckpt) == (CANONICAL_CHAIN["sr1"],
                                          CANONICAL_CHAIN["zhead"])


def test_zhead_default_uses_roman_bands_with_noise():
    """Two settings that must never silently revert.

    `phot_tier: medium` keeps the head on bands that ship with the grism, and
    a non-zero eval noise keeps the reported metric off noiseless truth
    photometry. Together they are the difference between NMAD 0.0065 and a
    leaked 0.003.
    """
    z = load_config(ZHeadConfig, "configs/zhead.yaml")
    assert z.phot_tier == "medium"
    assert z.phot_mag_err > 0
    assert z.phot_eval_mag_err > 0


def test_sr2_default_disables_the_coupled_z_loss():
    """lam_z inflated hallucination to 2.6x truth with a photometry-fed ZHead."""
    assert load_config(SR2Config, "configs/sr2.yaml").lam_z == 0.0


def test_unknown_config_key_is_an_error():
    # A typo'd hyperparameter that silently does nothing is the worst possible
    # failure mode for a run nobody looks at for six hours.
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(SR1Config, None, {"learning_rate": 1e-4})


def test_overrides_beat_the_file():
    cfg = load_config(SR1Config, "configs/sr1.yaml", {"epochs": 3})
    assert cfg.epochs == 3


@pytest.mark.parametrize("argv", [
    ["train", "sr1", "--config", "configs/sr1.yaml"],
    ["train", "zhead", "--epochs", "5"],
    ["train", "sr2", "--no-zhead-finetune"],
    ["extract", "run", "--max-scas", "2"],
    ["extract", "worker", "--visit", "1", "--sca", "3"],
    ["extract", "merge"],
    ["predict", "in.npz", "--out", "out.npz"],
    ["evaluate", "cache"],
    ["evaluate", "metrics"],
    ["evaluate", "figures", "--which", "spectra"],
    ["evaluate", "ablation"],
    ["evaluate", "prior"],
    ["info"],
])
def test_cli_parses_every_documented_invocation(argv):
    assert build_parser().parse_args(argv) is not None


def test_cli_boolean_flags_are_tri_state():
    """Unset must mean 'use the config value', not 'False'."""
    p = build_parser()
    assert p.parse_args(["train", "sr1"]).augment is None
    assert p.parse_args(["train", "sr1", "--augment"]).augment is True
    assert p.parse_args(["train", "sr1", "--no-augment"]).augment is False
