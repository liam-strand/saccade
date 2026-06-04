#!/usr/bin/env python3
"""Compare Kalman filter variants across estimator-independent schedulers.

Schedulers tested: round-robin and rate-of-change. Both ignore the estimator
when picking counters, so all four estimator variants below see an identical
measurement schedule -- the only thing that changes is the correlation config.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from sim_utils import (
    FIXED_NUM_SLOTS,
    LATENCY_PROFILE,
    filter_traces_by_kind,
    is_significant,
    load_noise_floor,
    mean_coverage,
    median_nrmse,
    run_batch_simulate,
    run_evaluate,
    REPO_ROOT,
)

SCHEDULERS = ["round-robin", "rate-of-change"]

KF_VARIANTS = [
    {"label": "ema",           "estimator": "ema",    "config": None},
    {"label": "kf_naive",      "estimator": "kalman", "config": "config/kf_naive.toml"},
    {"label": "kf_analytical", "estimator": "kalman", "config": "config/kf_analytical.toml"},
    {"label": "kf_expert",     "estimator": "kalman", "config": "config/kf_expert.toml"},
]

# Full cross product run as a single batch per workload: every (scheduler,
# estimator-variant) combo shares the same rates trace, loaded once.
COMBOS = [{"scheduler": s, **v} for s in SCHEDULERS for v in KF_VARIANTS]


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
    p.add_argument(
        "--llm-latency-profile",
        type=Path,
        default=LATENCY_PROFILE,
        help="LLM latency profile JSON forwarded to every simulate call "
        "(e.g. a fresh q7 llm_latency_profile.json).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--noise-floor-json",
        type=Path,
        default=None,
        help="Optional JSON with 'nrmse_floor' or 'median_nrmse' for significance testing",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Rayon threads for the per-workload batch call (default: 8 = one per scheduler x KF variant combo)",
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

    # One batch per workload: all (scheduler x KF variant) combos run in a single
    # saccade process so the rates trace is loaded only once.  Both schedulers
    # are estimator-independent, so the four variants share an identical
    # measurement schedule within each scheduler.
    for trace_path in tqdm(traces, desc="kf_variants", unit="workload"):
        workload = trace_path.stem

        def _out_trace(scheduler: str, label: str) -> Path:
            return tmp_dir / f"est_{workload}_{scheduler}_{label}_slots{FIXED_NUM_SLOTS}_t0.perfetto"

        batch_spec = []
        for combo in COMBOS:
            entry: dict = {
                "scheduler": combo["scheduler"],
                "estimator": combo["estimator"],
                "trace": str(_out_trace(combo["scheduler"], combo["label"])),
            }
            if args.seed is not None:
                entry["seed"] = args.seed
            if combo["config"] is not None:
                entry["config"] = str(config_dir / Path(combo["config"]).name)
            batch_spec.append(entry)

        try:
            run_batch_simulate(
                args.saccade, args.library, trace_path, batch_spec,
                FIXED_NUM_SLOTS, args.q_schedule, args.jobs,
                tmp_dir, "google/gemma-4-26b-a4b-it",
                base_config=None,  # per-combo configs carry all hyperparams
                latency_profile=args.llm_latency_profile,
            )
        except subprocess.CalledProcessError as exc:
            tqdm.write(f"  ERROR batch-simulate {workload}: {exc}", file=sys.stderr)
            for combo in COMBOS:
                row: dict = {
                    "workload": workload,
                    "scheduler": combo["scheduler"],
                    "kf_variant": combo["label"],
                    "estimator": combo["estimator"],
                    "median_nrmse": "", "coverage": "",
                    "nrmse_mean": "", "nrmse_stddev": "",
                }
                if noise_floor is not None:
                    row["significant"] = ""
                rows.append(row)
            continue

        def _eval_combo(combo: dict) -> dict:
            scheduler = combo["scheduler"]
            label = combo["label"]
            out_trace = _out_trace(scheduler, label)
            try:
                ev = run_evaluate(args.saccade, trace_path, out_trace)
                mn = median_nrmse(ev)
                mc = mean_coverage(ev)
            except subprocess.CalledProcessError as exc:
                tqdm.write(f"  ERROR evaluate {workload}/{scheduler}/{label}: {exc}", file=sys.stderr)
                mn = mc = None
            sig = is_significant(mn, noise_floor)
            r: dict = {
                "workload": workload,
                "scheduler": scheduler,
                "kf_variant": label,
                "estimator": combo["estimator"],
                "median_nrmse": mn if mn is not None else "",
                "coverage": mc if mc is not None else "",
                "nrmse_mean": mn if mn is not None else "",
                "nrmse_stddev": "",
            }
            if noise_floor is not None:
                r["significant"] = "" if sig is None else str(sig).lower()
            return r

        with ThreadPoolExecutor(max_workers=len(COMBOS)) as ex:
            combo_rows = list(ex.map(_eval_combo, COMBOS))
        rows.extend(combo_rows)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to: {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
