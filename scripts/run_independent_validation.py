#!/usr/bin/env python3
"""Prepare or train a fresh 80/10/10 chain without reading reserved test metrics.

Default: write isolated configs and a command plan. --execute trains every
stage from scratch, then freezes all checkpoints before building test caches.
No uploads or publication occur. Existing manuscript weights are never reused.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from specsr_roman.data.splits import get_or_make_three_way_split

PAIRS = [
    ("roman3", "zhead", "sr2"),
    ("rubin_y1", "zhead_rubin_y1", "sr2_rubin_y1"),
    ("rubin_y10", "zhead_rubin_y10", "sr2_rubin_y10"),
    ("photonly", "zhead_rubin_y10_photonly", "sr2_photonly_rubin_y10"),
]


def prepare(data, outdir):
    outdir = Path(outdir)
    confdir = outdir / "configs"
    confdir.mkdir(parents=True, exist_ok=True)
    configs, commands = {}, []
    for name in ["sr1"] + [n for _, zh, s2 in PAIRS for n in (zh, s2)]:
        config = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
        stage = "sr1" if name == "sr1" else ("zhead" if name.startswith("zhead") else "sr2")
        config.update(
            data=str(Path(data).resolve()),
            split_mode="three_way",
            out_dir=str(outdir / "runs" / stage),
            out_prefix=config["out_prefix"].removesuffix("_independent") + "_independent",
            push_to_hub=False,
        )
        if stage == "sr1":
            config.update(export_predictions=False, init_checkpoint=None)
        else:
            config["sr1_ckpt"] = str(
                outdir / "runs/sr1" / (configs["sr1"]["out_prefix"] + "_best.pth")
            )
        if stage == "sr2":
            zh = next(zh for _, zh, s2 in PAIRS if s2 == name)
            config["zhead_ckpt"] = str(
                outdir / "runs/zhead" / (configs[zh]["out_prefix"] + "_best.pth")
            )
        configs[name] = config
        path = confdir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        commands.append(
            [sys.executable, "-m", "specsr_roman.cli", "train", stage, "--config", str(path)]
        )
    # Hold back all test evaluations until every training/configuration choice is frozen.
    for key, zh, s2 in PAIRS:
        cmd = [
            sys.executable,
            "-m",
            "specsr_roman.cli",
            "evaluate",
            "cache",
            "--data",
            str(data),
            "--sr1",
            configs[zh]["sr1_ckpt"],
            "--zhead",
            configs[s2]["zhead_ckpt"],
            "--sr2",
            str(outdir / "runs/sr2" / (configs[s2]["out_prefix"] + "_best.pth")),
            "--phot-tier",
            configs[zh]["phot_tier"],
            "--split-mode",
            "three_way",
            "--partition",
            "test",
            "--out",
            str(outdir / f"pred_cache_{key}.npz"),
        ]
        if not configs[zh].get("use_spectrum", True):
            cmd.append("--no-zhead-spectrum")
        commands.append(cmd)
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/dataset/ou2024_h10307_dataset.npz")
    p.add_argument("--outdir", default="outputs/independent_v1")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument(
        "--detach", action="store_true",
        help="execute in a separate session with a persistent runner log",
    )
    args = p.parse_args()
    if args.detach:
        if not args.execute:
            p.error("--detach requires --execute")
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "runner.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve())]
                + [arg for arg in sys.argv[1:] if arg != "--detach"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Launched PID {process.pid}; inspect {outdir / 'runner.log'}")
        return
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Keep an OS-level lock for the whole run; stale JSON is not a live process.
    run_lock = (outdir / "run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another runner holds this output directory") from None
    existing_plan = Path(args.outdir) / "plan.json"
    if existing_plan.exists():
        previous = json.loads(existing_plan.read_text())
        if previous.get("status") != "prepared":
            raise SystemExit("preserve the existing run; choose a fresh --outdir")
    with np.load(args.data) as d:
        keep = (
            np.isfinite(d["snr"])
            & (np.isfinite(d["flux_low"]).mean(1) > 0.7)
            & (np.isfinite(d["flux_high"]).mean(1) > 0.9)
        )
        ids = d["ids"][keep]
    tr, va, te, path = get_or_make_three_way_split(args.data, ids)
    commands = prepare(args.data, args.outdir)
    outdir = Path(args.outdir)
    plan = {
        "status": "prepared",
        "split_path": path,
        "rows": dict(train=len(tr), validation=len(va), test=len(te)),
        "galaxies": dict(
            train=len(np.unique(ids[tr])),
            validation=len(np.unique(ids[va])),
            test=len(np.unique(ids[te])),
        ),
        "commands": commands,
    }
    plan_path = outdir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan["rows"]), flush=True)
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
    if not args.execute:
        return
    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "GPU unavailable: configs prepared; use --allow-cpu only for an intentional CPU run"
        )
    logs = outdir / "logs"
    logs.mkdir(exist_ok=True)
    plan["status"] = "running"
    plan["pid"] = os.getpid()
    plan["started_utc"] = datetime.now(timezone.utc).isoformat()
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    for i, cmd in enumerate(commands):
        print(f"START {i + 1}/{len(commands)}: {shlex.join(cmd)}", flush=True)
        plan["current_command"] = i
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        with (logs / f"{i:02d}.log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            plan.update(status="failed", failed_command=i)
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
            raise SystemExit(f"Command {i} failed; inspect {logs / f'{i:02d}.log'}")
        plan["completed_commands"] = i + 1
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    plan["status"] = "complete"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
