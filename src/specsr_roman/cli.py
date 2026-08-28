"""``specsr-roman`` command line interface.

Five verbs, matching the five things anyone does with this project:

    specsr-roman extract   build a training dataset from Roman simulation products
    specsr-roman train     train SR1, the ZHead, or SR2
    specsr-roman predict   run the published chain on a spectrum
    specsr-roman evaluate  metrics, figures, and the two audits
    specsr-roman info      what is installed, what is cached, what is canonical

Every training verb takes ``--config path.yaml`` plus overrides, so a run is
described by a file you can diff rather than a shell history line.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__

__all__ = ["main", "build_parser"]


def _add_overrides(p: argparse.ArgumentParser, fields: dict) -> None:
    """Expose dataclass fields as ``--kebab-case`` flags.

    Generated from the dataclass so a new hyperparameter is reachable from the
    command line the moment it exists, and cannot drift out of sync with the
    config.
    """
    for name, typ in fields.items():
        flag = "--" + name.replace("_", "-")
        if typ is bool:
            p.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction,
                           default=None)
        else:
            p.add_argument(flag, dest=name, type=typ, default=None)


def _dataclass_flags(cls) -> dict:
    import dataclasses
    out = {}
    for f in dataclasses.fields(cls):
        t = f.type
        if isinstance(t, str):                       # from __future__ annotations
            t = (bool if t.startswith("bool")
                 else int if t.startswith("int")
                 else float if t.startswith("float")
                 else str)
        out[f.name] = t if t in (bool, int, float, str) else str
    return out


def build_parser() -> argparse.ArgumentParser:
    from .config import SR1Config, SR2Config, ZHeadConfig

    p = argparse.ArgumentParser(prog="specsr-roman", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"specsr-roman {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- train ----------------------------------------------------------
    tr = sub.add_parser("train", help="train a stage")
    tr_sub = tr.add_subparsers(dest="stage", required=True)
    for stage, cls in (("sr1", SR1Config), ("zhead", ZHeadConfig),
                       ("sr2", SR2Config)):
        sp = tr_sub.add_parser(stage, help=f"train {stage}")
        sp.add_argument("--config", default=None,
                        help=f"YAML config (see configs/{stage}.yaml)")
        _add_overrides(sp, _dataclass_flags(cls))

    # ---- extract --------------------------------------------------------
    ex = sub.add_parser("extract", help="build a training dataset")
    ex_sub = ex.add_subparsers(dest="action", required=True)
    for action, helptext in (("run", "drive extraction over many SCAs"),
                             ("worker", "extract one (visit, SCA) [internal]"),
                             ("merge", "merge per-SCA caches into a dataset")):
        sp = ex_sub.add_parser(action, help=helptext)
        sp.add_argument("--data-dir", default="data/ou2024")
        sp.add_argument("--healpix", type=int, default=10307)
        sp.add_argument("--out-dataset",
                        default="data/dataset/ou2024_h10307_dataset.npz")
        sp.add_argument("--ab-target", type=float, default=22.5)
        sp.add_argument("--ab-scene", type=float, default=23.0)
        sp.add_argument("--min-hp-frac", type=float, default=0.7)
        sp.add_argument("--grism-exptime", type=float, default=301.0)
        sp.add_argument("--cleanup", action="store_true")
        if action == "run":
            sp.add_argument("--max-scas", type=int, default=60)
            sp.add_argument("--workers", type=int, default=4)
        if action == "worker":
            sp.add_argument("--visit", type=int, required=True)
            sp.add_argument("--sca", type=int, required=True)

    # ---- predict --------------------------------------------------------
    pr = sub.add_parser("predict", help="run the chain on spectra")
    pr.add_argument("input", help="npz with flux_low[, flux_low_err, phot]")
    pr.add_argument("--out", default="specsr_roman_predictions.npz")
    pr.add_argument("--sr1", default=None)
    pr.add_argument("--zhead", default=None)
    pr.add_argument("--sr2", default=None)
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--phot-tier", default="medium",
                    help="band subset to feed the ZHead, matching its training")
    pr.add_argument("--limit", type=int, default=None)

    # ---- evaluate -------------------------------------------------------
    ev = sub.add_parser("evaluate", help="metrics, figures, audits")
    ev_sub = ev.add_subparsers(dest="what", required=True)

    ca = ev_sub.add_parser("cache", help="build the frozen prediction cache")
    ca.add_argument("--data", default="data/dataset/ou2024_h10307_dataset.npz")
    ca.add_argument("--out", default="outputs/pred_cache.npz")
    ca.add_argument("--sr1", default=None)
    ca.add_argument("--zhead", default=None)
    ca.add_argument("--sr2", default=None)

    me = ev_sub.add_parser("metrics", help="redshift + line-recovery summary")
    me.add_argument("--cache", default="outputs/pred_cache.npz")
    me.add_argument("--json", default=None, help="also write metrics as JSON")

    fi = ev_sub.add_parser("figures", help="render the publication figures")
    fi.add_argument("--cache", default="outputs/pred_cache.npz")
    fi.add_argument("--outdir", default="outputs/figures")
    fi.add_argument("--which", default="all",
                    help="comma list: spectra,river,sn,redshift,psd")
    fi.add_argument("--rebuild", action="store_true")

    ab = ev_sub.add_parser("ablation",
                           help="photometry ablation: with and without colours")
    ab.add_argument("--data", default="data/dataset/ou2024_h10307_dataset.npz")
    ab.add_argument("--out-dir", default="outputs")

    pd = ev_sub.add_parser("prior", help="inverse-crime / prior-dominance audit")
    pd.add_argument("--data", default="data/dataset/ou2024_h10307_dataset.npz")
    pd.add_argument("--sr1", default=None)
    pd.add_argument("--max-sources", type=int, default=500)

    sub.add_parser("info", help="environment and canonical checkpoints")
    return p


def _cfg_from_args(cls, args, config_key: str = "config"):
    import dataclasses

    from .config import load_config
    known = {f.name for f in dataclasses.fields(cls)}
    overrides = {k: v for k, v in vars(args).items()
                 if k in known and v is not None}
    return load_config(cls, getattr(args, config_key, None), overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "info":
        return _cmd_info()
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "extract":
        return _cmd_extract(args)
    if args.command == "predict":
        return _cmd_predict(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    return 1


def _cmd_info() -> int:
    import torch

    from .checkpoints import CANONICAL_CHAIN, DEFAULT_HUB_REPO
    print(f"specsr-roman {__version__}")
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    print(f"hub repo: {DEFAULT_HUB_REPO}")
    print("canonical chain:")
    for stage, name in CANONICAL_CHAIN.items():
        print(f"  {stage:6s} {name}")
    for extra, mods in (("extract", ("grizli", "photutils", "h5py", "pyarrow")),
                        ("train", ("wandb",)),
                        ("hub", ("huggingface_hub",))):
        missing = []
        for m in mods:
            try:
                __import__(m)
            except ImportError:
                missing.append(m)
        state = "ok" if not missing else f"missing {', '.join(missing)}"
        print(f"  extra [{extra}]: {state}")
    return 0


def _cmd_train(args) -> int:
    from .config import SR1Config, SR2Config, ZHeadConfig
    from .training import train_sr1, train_sr2, train_zhead
    table = {"sr1": (SR1Config, train_sr1), "zhead": (ZHeadConfig, train_zhead),
             "sr2": (SR2Config, train_sr2)}
    cls, fn = table[args.stage]
    summary = fn(_cfg_from_args(cls, args))
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _cmd_extract(args) -> int:
    from .extraction import ExtractionConfig, merge, run_batch, run_worker
    cfg = ExtractionConfig(
        healpix=args.healpix, data_dir=args.data_dir,
        out_dataset=args.out_dataset, ab_target=args.ab_target,
        ab_scene=args.ab_scene, min_hp_frac=args.min_hp_frac,
        grism_exptime=args.grism_exptime, cleanup=args.cleanup,
        max_scas=getattr(args, "max_scas", 60),
        workers=getattr(args, "workers", 4))
    if args.action == "run":
        run_batch(cfg)
    elif args.action == "worker":
        run_worker(args.visit, args.sca, cfg)
    else:
        merge(cfg)
    return 0


def _cmd_predict(args) -> int:
    import numpy as np

    from .grids import resolve_phot_tier
    from .inference import RomanPipeline

    d = np.load(args.input, allow_pickle=True)
    flux = d["flux_low"]
    if args.limit:
        flux = flux[: args.limit]
    err = d["flux_low_err"][: len(flux)] if "flux_low_err" in d else None
    phot = None
    if "phot" in d:
        phot = d["phot"][: len(flux)]
        keep = resolve_phot_tier(args.phot_tier)
        if keep is not None:
            phot = phot[:, list(keep)]

    pipe = RomanPipeline.from_pretrained(sr1=args.sr1, zhead=args.zhead,
                                         sr2=args.sr2, device=args.device)
    results = pipe.predict(flux, err, phot=phot,
                           wave_low=d["wavelength_low"] if "wavelength_low" in d else None)
    if not isinstance(results, list):
        results = [results]
    np.savez_compressed(
        args.out,
        wavelength=results[0].wavelength,
        flux_sr=np.array([r.flux_sr for r in results]),
        flux_sr_err=np.array([r.flux_sr_err for r in results]),
        flux_sr1=np.array([r.flux_sr1 for r in results]),
        z=np.array([r.z for r in results]),
        z_err=np.array([r.z_err for r in results]),
        pz=np.array([r.pz for r in results]) if results[0].pz is not None else np.array([]),
        z_grid=results[0].z_grid if results[0].z_grid is not None else np.array([]))
    print(f"wrote {len(results)} predictions -> {args.out}")
    return 0


def _cmd_evaluate(args) -> int:
    import numpy as np

    if args.what == "cache":
        from .evaluation import CacheConfig, build_prediction_cache
        kwargs = {"data": args.data, "out": args.out}
        for key, val in (("sr1_ckpt", args.sr1), ("zhead_ckpt", args.zhead),
                         ("sr2_ckpt", args.sr2)):
            if val:
                kwargs[key] = val
        build_prediction_cache(CacheConfig(**kwargs))
        return 0

    if args.what == "metrics":
        from .evaluation import line_amplitude_recovery, redshift_summary
        c = np.load(args.cache, allow_pickle=True)
        out = {
            "redshift": redshift_summary(c["z_pred"], c["z_true"]),
            "line_amplitude": {
                "sr1": line_amplitude_recovery(c["sr1"], c["hr"], c["line_snr"]),
                "sr2": line_amplitude_recovery(c["sr2"], c["hr"], c["line_snr"]),
            },
        }
        print(json.dumps(out, indent=2))
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(out, fh, indent=2)
            print(f"wrote {args.json}")
        return 0

    if args.what == "figures":
        import matplotlib
        matplotlib.use("Agg")
        from .evaluation import load_prediction_cache
        from .evaluation.figures import make_figures
        c = load_prediction_cache(args.cache, rebuild=args.rebuild)
        which = None if args.which == "all" else args.which.split(",")
        make_figures(c, which=which, outdir=args.outdir)
        return 0

    if args.what == "ablation":
        from .evaluation.ablation import AblationConfig, run_ablation
        run_ablation(AblationConfig(data=args.data, out_dir=args.out_dir))
        return 0

    if args.what == "prior":
        from .evaluation.prior_dominance import PriorDominanceConfig, run_prior_dominance
        kwargs = {"data": args.data, "max_sources": args.max_sources}
        if args.sr1:
            kwargs["sr1_ckpt"] = args.sr1
        run_prior_dominance(PriorDominanceConfig(**kwargs))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
