#!/usr/bin/env python3
"""Compare LLM-driven schedulers with and without a guidance hint.

Runs all three LLM schedulers (static-llm, dynamic-llm, weighted-round-robin-llm)
under two conditions:
  guidance_condition=none       -- no guidance passed to the LLM
  guidance_condition=with_guidance -- the --guidance string is forwarded

Operates on SPEC workloads only.  Outputs a timestamped CSV.
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
    LLM_SCHEDULERS,
    filter_traces_by_kind,
    is_significant,
    load_noise_floor,
    run_combo_extended,
)

DEFAULT_GUIDANCE = (
    "Focus on memory hierarchy behavior: cache misses, TLB pressure, and memory "
    "bandwidth utilization. Deprioritize integer arithmetic counters."
)

LLM_SCHEDULER_LIST = sorted(LLM_SCHEDULERS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare LLM schedulers with vs. without a guidance hint (SPEC only)."
    )
    p.add_argument("--saccade", type=Path, default=Path("../target/release/saccade"))
    p.add_argument("--library", type=Path, default=Path("../event_lib.json"))
    p.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("./sweep_data_eval_traces"),
    )
    p.add_argument("--results-dir", type=Path, default=Path("./results"))
    p.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Run only the SPEC trace whose stem matches this name",
    )
    p.add_argument(
        "--estimator",
        type=str,
        default="kalman",
        help="Estimator to pair with each LLM scheduler (default: kalman)",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default="google/gemma-4-26b-a4b-it",
        help="Model name for LLM-driven schedulers; uses OpenRouter unless sim_utils is customized.",
    )
    p.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help=(
            "TOML config forwarded to every simulate call. "
            "Defaults to python/config/kf_expert.toml."
        ),
    )
    p.add_argument(
        "--guidance",
        type=str,
        default=DEFAULT_GUIDANCE,
        help="Guidance string for the 'with_guidance' condition",
    )
    p.add_argument("--llm-trials", type=int, default=3)
    p.add_argument("--q-schedule", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise-floor-json", type=Path, default=None)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.saccade = args.saccade.resolve()
    args.library = args.library.resolve()
    args.traces_dir = args.traces_dir.resolve()
    args.results_dir = args.results_dir.resolve()

    if args.base_config is None:
        candidate = Path(__file__).parent / "config" / "kf_expert.toml"
        if candidate.exists():
            args.base_config = candidate
    else:
        args.base_config = args.base_config.resolve()

    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")
    if not args.library.exists():
        parser.error(f"Library not found: {args.library}")
    if not args.traces_dir.is_dir():
        parser.error(f"Traces directory not found: {args.traces_dir}")

    # SPEC only
    traces = filter_traces_by_kind(sorted(args.traces_dir.glob("*.perfetto")), "spec")
    if args.workload is not None:
        traces = [t for t in traces if t.stem == args.workload]
    if not traces:
        parser.error("No matching SPEC traces found.")

    noise_floor = load_noise_floor(args.noise_floor_json)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "traces"
    tmp_dir.mkdir()
    out_csv = run_dir / "q4_llm_guidance.csv"

    fieldnames = [
        "workload", "scheduler", "estimator",
        "guidance_condition", "guidance_text",
        "median_nrmse",    # frozen primary metric
        "coverage",
        "nrmse_mean",      # trial-variability: mean of per-trial median_nrmse
        "nrmse_stddev",    # trial-variability: stddev of per-trial median_nrmse
        "nrmse_p90",       # per-eval distribution: 90th-percentile event nRMSE
        "nrmse_max",       # per-eval distribution: worst-case event nRMSE
        "nrmse_weighted",  # per-eval: importance_weighted_nrmse (secondary lens)
        "mean_calibration",
    ]
    if noise_floor is not None:
        fieldnames.append("significant")

    rows: list[dict] = []

    conditions = [
        ("none", None),
        ("with_guidance", args.guidance),
    ]

    combos = [
        (t, sched, cond_label, guidance_val)
        for t in traces
        for sched in LLM_SCHEDULER_LIST
        for cond_label, guidance_val in conditions
    ]

    bar = tqdm(combos, desc="llm_guidance", unit="combo")
    for trace_path, scheduler, cond_label, guidance_val in bar:
        workload = trace_path.stem
        bar.set_postfix_str(f"{workload}/{scheduler}/{cond_label}")

        try:
            metrics = run_combo_extended(
                saccade=args.saccade,
                library=args.library,
                rates_trace=trace_path,
                scheduler=scheduler,
                estimator=args.estimator,
                num_slots=FIXED_NUM_SLOTS,
                q_schedule=args.q_schedule,
                llm_model=args.llm_model,
                seed=args.seed,
                llm_trials=args.llm_trials,
                tmp_dir=tmp_dir,
                workload=f"{workload}_{scheduler}_{cond_label}",
                base_config=args.base_config,
                guidance=guidance_val,
            )
        except subprocess.CalledProcessError as exc:
            tqdm.write(
                f"  ERROR {workload}/{scheduler}/{cond_label}: {exc}", file=sys.stderr
            )
            metrics = {
                "median_nrmse": None, "coverage": None,
                "nrmse_mean": None, "nrmse_stddev": None,
                "nrmse_p90": None, "nrmse_max": None,
                "nrmse_weighted": None, "mean_calibration": None,
            }

        med_nrmse = metrics["median_nrmse"]
        sig = is_significant(med_nrmse, noise_floor)
        truncated_guidance = (
            "" if guidance_val is None
            else (guidance_val[:80] + "…" if len(guidance_val) > 80 else guidance_val)
        )

        def _fmt(v: float | None) -> str:
            return "" if v is None else str(v)

        row: dict = {
            "workload": workload,
            "scheduler": scheduler,
            "estimator": args.estimator,
            "guidance_condition": cond_label,
            "guidance_text": truncated_guidance,
            "median_nrmse": _fmt(metrics["median_nrmse"]),
            "coverage": _fmt(metrics["coverage"]),
            "nrmse_mean": _fmt(metrics["nrmse_mean"]),
            "nrmse_stddev": _fmt(metrics["nrmse_stddev"]),
            "nrmse_p90": _fmt(metrics["nrmse_p90"]),
            "nrmse_max": _fmt(metrics["nrmse_max"]),
            "nrmse_weighted": _fmt(metrics["nrmse_weighted"]),
            "mean_calibration": _fmt(metrics["mean_calibration"]),
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
