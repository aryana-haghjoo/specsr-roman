#!/usr/bin/env python3
"""Summarize frozen reserved-test audits without fitting calibration on test data.

Run response audits under both output conventions and validate_frozen_results.py
for every configuration first. This command checks identities and partitions,
compares matched source/factor pairs, and writes an auditable report bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_response_scales import compare

from specsr_roman.data.splits import get_or_make_three_way_split
from specsr_roman.evaluation import redshift_summary
from specsr_roman.evaluation.validation import grouped_interval

KEYS = ("roman3", "rubin_y1", "rubin_y10", "photonly")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def grouped_median_interval(values, ids, draws=1000, seed=0):
    """Bootstrap galaxies, retaining every visit and perturbation in each draw."""
    values, ids = np.asarray(values), np.asarray(ids)
    groups = [values[ids == key] for key in np.unique(ids)]
    rng = np.random.default_rng(seed)
    bootstrap = [np.median(np.concatenate([groups[i] for i in indices]))
                 for indices in rng.integers(len(groups), size=(draws, len(groups)))]
    return {
        "median": float(np.median(values)),
        "n_galaxies": len(groups),
        "draws": draws,
        "seed": seed,
        "bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "note": "Spectrum/factor-pooled median; galaxies are resampling units.",
    }


def summarize(data, caches, audits, draws=1000):
    with np.load(data) as raw:
        keep = (np.isfinite(raw["snr"])
                & (np.isfinite(raw["flux_low"]).mean(1) > 0.7)
                & (np.isfinite(raw["flux_high"]).mean(1) > 0.9))
        ids = raw["ids"][keep]
        truth = raw["redshift"][keep]
    tr, va, te, split = get_or_make_three_way_split(data, ids, verbose=False)
    row_to_galaxy = dict(zip(te, ids[te], strict=True))
    result = {
        "weights_release": "0.6.0",
        "audit_release": "0.6.1",
        "dataset_sha256": sha256(data),
        "split": Path(split).name,
        "rows": {"train": len(tr), "validation": len(va), "test": len(te)},
        "galaxies": {"train": len(np.unique(ids[tr])),
                     "validation": len(np.unique(ids[va])),
                     "test": len(np.unique(ids[te]))},
        "test_protocol": "All nine models frozen before new reserved-test evaluation; "
                         "test galaxies appeared in historical validation during development.",
        "bootstrap": f"{draws} galaxy-grouped draws, seed 0; repeated visits are retained.",
        "calibration": "Coverage diagnostics only; no calibration fitted on reserved test data.",
        "response_method": "v2 Gaussian LSF injection with original errors held fixed; "
                           "not a repeat of two-dimensional morphological dispersion.",
        "configurations": {},
    }
    input_paths = [Path(data), Path(split)]
    for key in KEYS:
        cache_path = Path(caches) / f"pred_cache_{key}.npz"
        if key == "roman3" and not cache_path.exists():
            cache_path = Path(caches) / "pred_cache.npz"
        with np.load(cache_path) as cache:
            config = json.loads(str(cache["config"]))
            if (config["split_mode"], config["partition"]) != ("three_way", "test"):
                raise ValueError(f"{key}: not a reserved-test cache")
            np.testing.assert_array_equal(cache["dataset_row"], te)
            np.testing.assert_array_equal(cache["object_id"], ids[te])
            np.testing.assert_allclose(cache["z_true"], truth[te], atol=1e-6)
            z_pred, z_true, galaxy = cache["z_pred"], cache["z_true"], cache["object_id"]
            n_candidates = int((cache["line_snr"].max(axis=1) >= 2).sum())
        calibration_path = Path(audits) / "calibration" / f"{key}.json"
        calibration = json.loads(calibration_path.read_text())
        if calibration["evaluation_status"] != "three_way test":
            raise ValueError(f"{key}: calibration partition mismatch")
        if (calibration.get("cache_sha256") != sha256(cache_path)
                or calibration.get("cache_config") != config):
            raise ValueError(f"{key}: calibration cache/config identity mismatch")
        if calibration["redshift_uncertainty"]["n"] != len(te):
            raise ValueError(f"{key}: calibration sample mismatch")
        row = {
            "redshift": redshift_summary(z_pred, z_true),
            "galaxy_balanced_catastrophic": grouped_interval(
                np.abs(z_pred - z_true) / (1 + z_true) > 0.15, galaxy, draws=draws),
            "uncertainty": calibration,
            "response": {},
        }
        for directory, convention in [("lr_restored", "legacy_lr"),
                                      ("target_fixed", "target_fixed")]:
            path = Path(audits) / directory / f"response_{key}.json"
            record_path = path.with_suffix(".records.npz")
            response = json.loads(path.read_text())
            cfg = response["config"]
            if (response["method_version"], cfg["output_scale"], cfg["split_mode"],
                    cfg["partition"]) != (2, convention, "three_way", "test"):
                raise ValueError(f"{key}: mixed response methods or partitions")
            if response["n_candidates"] != n_candidates:
                raise ValueError(f"{key}: unexpected detected-line candidate population")
            for name in ("sr1_ckpt", "zhead_ckpt", "sr2_ckpt"):
                def run_name(spec):
                    return Path(spec.split(":")[-1]).stem.removesuffix("_best")

                if run_name(cfg[name]) != run_name(config[name]):
                    raise ValueError(f"{key}: response/cache checkpoint mismatch")
            for name in ("phot_tier", "eval_mag_err", "noise_seed", "delta_cap",
                         "sigma_base_um", "z_topk", "zhead_use_spectrum"):
                if cfg[name] != config[name]:
                    raise ValueError(f"{key}: response/cache setting {name} differs")
            intervals = {}
            with np.load(record_path) as records:
                for stage in ("sr1", "sr2"):
                    values = records[stage]
                    if (values.shape[1] != 5 or not np.isfinite(values[:, :4]).all()
                            or not np.isin(values[:, 0], [0.5, 2.0]).all()):
                        raise ValueError(f"{key}: invalid response records")
                    source = values[:, 3].astype(np.int64)
                    if not np.array_equal(source, values[:, 3]):
                        raise ValueError("response source IDs must be integers")
                    if not np.isin(source, te).all():
                        raise ValueError(f"{key}: response contains non-test sources")
                    gal = np.array([row_to_galaxy[i] for i in source])
                    intervals[stage] = grouped_median_interval(values[:, 2], gal, draws=draws)
                    np.testing.assert_allclose(
                        np.median(values[:, 2]), response[stage]["overall_median_r"])
            row["response"][directory] = {"summary": response, "grouped_intervals": intervals}
            input_paths += [path, record_path]
        with (np.load(Path(audits) / "lr_restored" / f"response_{key}.records.npz") as a,
              np.load(Path(audits) / "target_fixed" / f"response_{key}.records.npz") as b):
            row["normalization_control"] = compare(a, b)
        result["configurations"][key] = row
        input_paths += [cache_path, calibration_path]
    result["input_checksums"] = {str(p): sha256(p) for p in input_paths}
    return result


def render_report(result):
    labels = {"roman3": "Roman three bands", "rubin_y1": "+ Rubin Year 1",
              "rubin_y10": "+ Rubin Year 10", "photonly": "Photometry-only redshift"}
    configurations = result["configurations"]
    lines = [
        "# Release 0.6.1: audits of the independently trained weights", "",
        "The nine weights and four test prediction caches are unchanged from v0.6.0.",
        "These diagnostics use all 2,384 reserved test spectra of 1,005 galaxies.",
        "No calibration was fitted, and no checkpoint was selected, on these results.", "",
        "The galaxies were reserved from checkpoint selection for the new runs, but",
        "appeared in historical validation during development. This is not a fully",
        "prospective sample or a test outside the Diffsky simulation manifold.", "",
        "## Response to changing a line", "",
        "The strongest target line is halved or doubled, its difference is smoothed",
        "with the Gaussian grism LSF and added to the input in extraction units.",
        "The original errors and each source's noisy photometry are held fixed.",
        "The primary response includes the deployed input-dependent LR restoration.",
        "Eligibility requires best-line expected S/N ≥ 2 and positive baseline recovery",
        "of at least 20% in both stages. All 1,819 candidate spectra were attempted.",
        "This approximates spectral injection; it does not repeat 2D morphological dispersion.", "",
        "| Configuration | Eligible spectra | SR1 median r | SR2 median r (95% bootstrap) | "
        "SR2, strong lines |", "|---|---:|---:|---:|---:|",
    ]
    for key, row in configurations.items():
        primary = row["response"]["lr_restored"]
        s = primary["summary"]
        low, high = primary["grouped_intervals"]["sr2"]["bootstrap_95_interval"]
        strong = next(b["median"] for b in s["sr2"]["by_snr"] if b["bin"] == "strong")
        lines.append(f"| {labels[key]} | {s['n_sources']:,} | "
                     f"{s['sr1']['overall_median_r']:.3f} | "
                     f"{s['sr2']['overall_median_r']:.3f} ({low:.3f}–{high:.3f}) | {strong:.3f} |")
    lines += ["", "Intervals use 1,000 galaxy-grouped bootstrap draws (seed 0), retaining all",
              "visits and perturbations of each sampled galaxy. Medians pool spectrum/factor",
              "responses. Strong lines have original expected S/N > 6.", "",
              "| Configuration | SR2 r, halving | SR2 r, doubling |",
              "|---|---:|---:|"]
    for key, row in configurations.items():
        factors = row["response"]["lr_restored"]["summary"]["sr2"]["per_factor"]
        lines.append(f"| {labels[key]} | {factors['0.5']['median']:.3f} | "
                     f"{factors['2.0']['median']:.3f} |")
    lines += ["", "Response 1 means proportional output change in the selected restoration;",
              "response 0 means no output change. Low response to an injection below the noise",
              "can be appropriate. Original and injected-change S/N breakdowns are retained",
              "in the JSON and per-response records. A pooled exponent is not a fraction of",
              "flux attributable to a learned prior, nor a single doubling prediction.", "",
              "## Normalization control", "",
              "Fixed-target restoration holds the target mean and scale constant and isolates",
              "normalized network output. Input normalization can remove amplitude changes",
              "even for an identity estimator, so this is not a corrected physical response.",
              "The comparison below matches valid source IDs and perturbation factors.", "",
              "| Configuration | Matched SR2 responses | Deployed r | Fixed-target r |",
              "|---|---:|---:|---:|"]
    for key, row in configurations.items():
        s = row["normalization_control"]["sr2"]
        lines.append(f"| {labels[key]} | {s['n_responses']:,} | "
                     f"{s['primary_median']:.3f} | {s['fixed_median']:.3f} |")
    lines += ["", "## Uncertainty coverage", "",
              "These are coverage diagnostics, not a fitted correction. Gaussian reference",
              "coverage is 68.27% within 1σ and 95% within 1.96σ. Redshift coverage uses the",
              "central 68% P(z) interval with uniform density within each 0.01-wide bin.",
              "Bin-midpoint coverage and randomized PIT diagnostics are also retained.", "",
              "| Configuration | All pixels, 1σ | Detected-line pixels, 1σ | "
              "Detected-line pixels, 1.96σ | Redshift central 68% |",
              "|---|---:|---:|---:|---:|"]
    for key, row in configurations.items():
        s = row["uncertainty"]["spectral_uncertainty"]
        pz = row["uncertainty"]["redshift_uncertainty"]["central_uniform_bin_coverage"]["0.68"]
        lines.append(f"| {labels[key]} | {s['all']['coverage']['1.0']:.1%} | "
                     f"{s['detected_line']['coverage']['1.0']:.1%} | "
                     f"{s['detected_line']['coverage']['1.95996398454']:.1%} | {pz:.1%} |")
    lines += ["", "P(z) intervals over-cover this sample; their bin width also limits local",
              "precision. Spectral uncertainties are conservative in aggregate but miss",
              "nominal 95% coverage in detected-line tails. One global variance scale is",
              "not established by these results. Any fitted recalibration needs validation",
              "data and a separate evaluation; none was fitted on the test partition.", "",
              "## Unsupported-emission proxies", "",
              "The diagnostic uses truth-centered, overlapping Gaussian windows",
              "(σ = 0.005 µm), smoothed-continuum subtraction, and a threshold of 5 in",
              "standardized weighted-sum units. Off-grid windows are excluded. These",
              "proxies are not blind line-search false-positive rates.", "",
              "| Configuration | Weak-target windows above threshold | "
              "Present but expected S/N < 2, above threshold |",
              "|---|---:|---:|"]
    for key, row in configurations.items():
        u = row["uncertainty"]["unsupported_emission"]
        a, b = u["intrinsically_weak_proxy"], u["present_but_snr_below_2"]
        lines.append(f"| {labels[key]} | {a['n_drawn_above_threshold']:,}/{a['n_windows']:,} "
                     f"({a['fraction_above_threshold']:.2%}) | "
                     f"{b['n_drawn_above_threshold']:,}/{b['n_windows']:,} "
                     f"({b['fraction_above_threshold']:.1%}) |")
    lines += ["", "Line recovery against simulated targets remains distinct from measuring",
              "physical line fluxes. The weak perturbation response, uncertainty mismatch,",
              "historical development exposure, and simulation domain limit scientific use.", "",
              "## Reproduce", "",
              "Install `specsr-roman[hub]` and obtain the source scripts from the v0.6.1",
              "release. Download the full dataset and four v0.6.0 test caches from HF.",
              "Both local training outputs and `--runs hub` are supported by the audit runner.",
              "Use a fresh output directory; do not overwrite an earlier audit generation.", "",
              "```bash",
              "python scripts/run_response_audits.py --runs hub --output-scale legacy_lr "
              "--outdir outputs/audits/lr_restored",
              "python scripts/run_response_audits.py --runs hub --output-scale target_fixed "
              "--outdir outputs/audits/target_fixed",
              "# Repeat the coverage command for all four caches/configurations:",
              "python scripts/validate_frozen_results.py --cache outputs/pred_cache.npz "
              "--out outputs/audits/calibration/roman3.json",
              "python scripts/summarize_release_audits.py --caches outputs "
              "--audits outputs/audits", "```", "",
              "The bundle includes the four coverage reports, eight response summaries,",
              "eight per-response NPZ files, matched normalization comparisons, galaxy-grouped",
              "intervals, input checksums, exact model identities, and source scripts.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset/ou2024_h10307_dataset.npz")
    parser.add_argument("--caches", default="outputs/independent_v2")
    parser.add_argument("--audits", default="outputs/audits_v060")
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    out = summarize(args.data, args.caches, args.audits, args.draws)
    path = Path(args.audits) / "summary.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    path.with_suffix(".md").write_text(render_report(out))
    print(path)


if __name__ == "__main__":
    main()
