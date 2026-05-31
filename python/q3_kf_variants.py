#!/usr/bin/env python3
"""Compare Kalman filter variants with the rate-of-change scheduler.

Fixed scheduler: rate-of-change.
Four estimator variants tested:

  ema          -- exponential moving average (non-KF baseline)
  kf_naive     -- Kalman with expert hyperparams, no correlation matrix
  kf_analytical-- Kalman + data-driven Pearson correlation (correlation.json)
  kf_expert    -- Kalman + expert-augmented correlation (correlation_expert.json)

Outputs a timestamped CSV under --results-dir.
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from sim_utils import (
    FIXED_NUM_SLOTS,
    filter_traces_by_kind,
    is_significant,
    load_noise_floor,
    run_combo,
    REPO_ROOT,
)

FIXED_SCHEDULER = "rate-of-change"

KF_VARIANTS = [
    {"label": "ema",           "estimator": "ema",    "config": None},
    {"label": "kf_naive",      "estimator": "kalman", "config": "config/kf_naive.toml"},
    {"label": "kf_analytical", "estimator": "kalman", "config": "config/kf_analytical.toml"},
    {"label": "kf_expert",     "estimator": "kalman", "config": "config/kf_expert.toml"},
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare Kalman estimator variants using the rate-of-change scheduler."
        )
    )
    p.add_argument("--saccade", type=Path, default=Path("../target/release/saccade"))
    p.add_argument("--library", type=Path, default=Path("../event_lib.json"))
    p.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("./sweep_data_eval_traces"),
        help="Directory of ground-truth .perfetto traces",
    )
    p.add_argument("--results-dir", type=Path, default=Path("./results"))
    p.add_argument(
        "--workload-kind",
        choices=["spec", "npb", "all"],
        default="all",
        help="Limit traces to SPEC (spec_*), NPB (npb_*), or all (default: all)",
    )
    p.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Run only the trace whose stem matches this name",
    )
    p.add_argument("--q-schedule", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--noise-floor-json",
        type=Path,
        default=None,
        help="Optional JSON with 'nrmse_floor' or 'median_nrmse' for significance testing",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.saccade = args.saccade.resolve()
    args.library = args.library.resolve()
    args.traces_dir = args.traces_dir.resolve()
    args.results_dir = args.results_dir.resolve()

    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")
    if not args.library.exists():
        parser.error(f"Library not found: {args.library}")
    if not args.traces_dir.is_dir():
        parser.error(f"Traces directory not found: {args.traces_dir}")

    traces = filter_traces_by_kind(sorted(args.traces_dir.glob("*.perfetto")), args.workload_kind)
    if args.workload is not None:
        traces = [t for t in traces if t.stem == args.workload]
    if not traces:
        parser.error("No matching traces found.")

    config_dir = REPO_ROOT / "python" / "config"

    noise_floor = load_noise_floor(args.noise_floor_json)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "traces"
    tmp_dir.mkdir()
    out_csv = run_dir / "q3_kf_variants.csv"

    fieldnames = [
        "workload", "scheduler", "kf_variant", "estimator",
        "median_nrmse", "coverage", "nrmse_mean", "nrmse_stddev",
    ]
    if noise_floor is not None:
        fieldnames.append("significant")

    rows: list[dict] = []

    combos = [(t, v) for t in traces for v in KF_VARIANTS]
    bar = tqdm(combos, desc="kf_variants", unit="combo")
    for trace_path, variant in bar:
        workload = trace_path.stem
        bar.set_postfix_str(f"{workload}/{variant['label']}")

        base_config: Path | None = None
        if variant["config"] is not None:
            base_config = config_dir / Path(variant["config"]).name

        try:
            med_nrmse, cov, nrmse_mean, nrmse_std = run_combo(
                saccade=args.saccade,
                library=args.library,
                rates_trace=trace_path,
                scheduler=FIXED_SCHEDULER,
                estimator=variant["estimator"],
                num_slots=FIXED_NUM_SLOTS,
                q_schedule=args.q_schedule,
                seed=args.seed,
                llm_trials=1,
                tmp_dir=tmp_dir,
                workload=f"{workload}_{variant['label']}",
                base_config=base_config,
            )
        except subprocess.CalledProcessError as exc:
            tqdm.write(f"  ERROR {workload}/{variant['label']}: {exc}", file=sys.stderr)
            med_nrmse = cov = nrmse_mean = nrmse_std = None

        sig = is_significant(med_nrmse, noise_floor)
        row: dict = {
            "workload": workload,
            "scheduler": FIXED_SCHEDULER,
            "kf_variant": variant["label"],
            "estimator": variant["estimator"],
            "median_nrmse": med_nrmse if med_nrmse is not None else "",
            "coverage": cov if cov is not None else "",
            "nrmse_mean": nrmse_mean if nrmse_mean is not None else "",
            "nrmse_stddev": nrmse_std if nrmse_std is not None else "",
        }
        if noise_floor is not None:
            row["significant"] = "" if sig is None else str(sig).lower()
        rows.append(row)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to: {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
