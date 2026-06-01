#!/usr/bin/env python3
"""Sweep scheduler × estimator combinations across workload traces to evaluate
accuracy of adaptive scheduling strategies.

Fixed: num_slots=4, full event library.
Iterates over every .perfetto trace in --traces-dir (or just --workload).
Outputs timestamped results under --results-dir/<timestamp>/.
"""

import csv
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import argparse

from tqdm import tqdm

import numpy as np

from sim_utils import (
    SCHEDULERS,
    ESTIMATORS,
    FIXED_NUM_SLOTS,
    LLM_SCHEDULERS,
    filter_traces_by_kind,
    importance_weighted_nrmse,
    is_significant,
    load_noise_floor,
    mean_calibration,
    mean_coverage,
    median_nrmse,
    nrmse_distribution,
    run_batch_simulate,
    run_evaluate,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep scheduler × estimator combinations across all workload traces "
            "to evaluate accuracy of adaptive scheduling strategies."
        )
    )
    parser.add_argument(
        "--saccade",
        type=Path,
        default=Path("../target/release/saccade"),
        help="Path to the saccade binary (default: ../target/release/saccade)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("../event_lib.json"),
        help="Event library JSON file (produced by saccade generate)",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("./sweep_data_eval_traces"),
        help="Directory of ground-truth .perfetto traces (default: ./sweep_data_eval_traces)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory to write CSV output files (default: ./results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible simulations (omit for OS-random)",
    )
    parser.add_argument(
        "--q-schedule",
        type=int,
        default=10_000_000,
        help="Scheduler quantum in nanoseconds (default: 10000000 = 10ms)",
    )
    parser.add_argument(
        "--noise-floor-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with key 'nrmse_floor' or 'median_nrmse' (float). "
            "When provided, a 'significant' column is added to the CSV."
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help=(
            "Optional saccade TOML config file forwarded as --config to every simulate "
            "call. Useful for setting subsection parameters (e.g. [kalman], [ema], [llm]). "
            "CLI args (--scheduler, --estimator, etc.) override values in this file."
        ),
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="google/gemma-4-26b-a4b-it",
        help="Model name for LLM-driven models; uses OpenRouter unless sim_utils is customized.",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default=None,
        help=(
            "Run only the trace whose stem matches this name (e.g. 'spec_531_deepsjeng_r'). "
            "By default all traces in --traces-dir are processed."
        ),
    )
    parser.add_argument(
        "--workload-kind",
        choices=["spec", "npb", "all"],
        default="all",
        help="Limit traces to SPEC (spec_*), NPB (npb_*), or all (default: all)",
    )
    parser.add_argument(
        "--exclude-scheduler",
        action="append",
        dest="exclude_schedulers",
        metavar="SCHEDULER",
        default=[],
        help=(
            "Exclude a scheduler from the grid (repeatable). "
            "Example: --exclude-scheduler dynamic-llm"
        ),
    )
    parser.add_argument(
        "--llm-trials",
        type=int,
        default=3,
        help="Number of trials for LLM schedulers; results are averaged (default: 3)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Rayon threads for each per-workload batch call (default: 1)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.saccade = args.saccade.resolve()
    args.library = args.library.resolve()
    args.traces_dir = args.traces_dir.resolve()
    args.results_dir = args.results_dir.resolve()

    if args.base_config is not None:
        args.base_config = args.base_config.resolve()
        if not args.base_config.exists():
            parser.error(f"Base config not found: {args.base_config}")

    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")
    if not args.library.exists():
        parser.error(f"Library file not found: {args.library}")
    if not args.traces_dir.is_dir():
        parser.error(f"Traces directory not found: {args.traces_dir}")

    all_traces = sorted(args.traces_dir.glob("*.perfetto"))
    if not all_traces:
        parser.error(f"No .perfetto files found in {args.traces_dir}")

    traces = filter_traces_by_kind(all_traces, args.workload_kind)
    if args.workload is not None:
        traces = [t for t in traces if t.stem == args.workload]
        if not traces:
            available = [t.stem for t in all_traces]
            parser.error(
                f"No trace named '{args.workload}' found in {args.traces_dir}. "
                f"Available: {available}"
            )

    if not traces:
        parser.error(
            f"No traces matched --workload-kind={args.workload_kind} in {args.traces_dir}"
        )

    schedulers = [s for s in SCHEDULERS if s not in args.exclude_schedulers]
    if not schedulers:
        parser.error("All schedulers have been excluded.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    traces_out_dir = run_dir / "traces"
    traces_out_dir.mkdir(exist_ok=True)

    noise_floor = load_noise_floor(args.noise_floor_json)

    grid_csv_path = run_dir / "q2_scheduler_estimator.csv"
    grid_combos = [
        (t.stem, sched, est)
        for t in traces
        for sched in schedulers
        for est in ESTIMATORS
    ]
    trace_map = {t.stem: t for t in traces}

    print(
        f"Found {len(traces)} workload trace(s). "
        f"Grid: {len(schedulers)} schedulers × {len(ESTIMATORS)} estimators "
        f"× {len(traces)} workloads = {len(grid_combos)} total.",
        file=sys.stderr,
    )
    if args.exclude_schedulers:
        print(f"Excluded schedulers: {args.exclude_schedulers}", file=sys.stderr)

    llm_schedulers = [s for s in schedulers if s in LLM_SCHEDULERS]
    non_llm_schedulers = [s for s in schedulers if s not in LLM_SCHEDULERS]

    def _out_trace(workload: str, sched: str, est: str, trial: int = 0) -> Path:
        return traces_out_dir / f"est_{workload}_{sched}_{est}_slots{FIXED_NUM_SLOTS}_t{trial}.perfetto"

    def _batch_spec(workload: str, sched_list: list[str], trial: int) -> list[dict]:
        entries = []
        for sched in sched_list:
            for est in ESTIMATORS:
                entry: dict = {
                    "scheduler": sched,
                    "estimator": est,
                    "trace": str(_out_trace(workload, sched, est, trial)),
                }
                if args.seed is not None:
                    entry["seed"] = args.seed
                entries.append(entry)
        return entries

    combo_to_row: dict[tuple, dict] = {}

    for trace in tqdm(traces, desc="simulate", unit="workload"):
        workload = trace.stem

        # Trial 0: all schedulers batched together in one process.
        # Trials 1..N-1: LLM schedulers only (non-LLM is deterministic with a
        # fixed seed so repeating adds no information).
        trial_specs = [_batch_spec(workload, schedulers, 0)]
        for trial in range(1, args.llm_trials):
            if llm_schedulers:
                trial_specs.append(_batch_spec(workload, llm_schedulers, trial))

        failed = False
        for trial, spec in enumerate(trial_specs):
            try:
                run_batch_simulate(
                    args.saccade, args.library, trace, spec,
                    FIXED_NUM_SLOTS, args.q_schedule, args.jobs,
                    traces_out_dir, args.llm_model, args.base_config,
                )
            except subprocess.CalledProcessError as exc:
                tqdm.write(
                    f"  ERROR batch-simulate {workload} trial {trial}: {exc}",
                    file=sys.stderr,
                )
                failed = True
                break

        if failed:
            for sched in schedulers:
                for est in ESTIMATORS:
                    row: dict = {
                        "workload": workload, "scheduler": sched, "estimator": est,
                        "median_nrmse": "", "coverage": "",
                        "nrmse_mean": "", "nrmse_stddev": "",
                        "nrmse_p90": "", "nrmse_max": "",
                        "nrmse_weighted": "", "mean_calibration": "",
                    }
                    if noise_floor is not None:
                        row["significant"] = ""
                    combo_to_row[(workload, sched, est)] = row
            continue

        # Evaluate all output traces in parallel, then average LLM results.
        # Each trial returns (median_nrmse, mean_coverage, p90, max, weighted, calibration).
        # nrmse_mean/nrmse_stddev are trial-variability stats (across LLM trials);
        # nrmse_p90/nrmse_max/nrmse_weighted are per-eval distribution stats aggregated
        # across trials with the same scheme (median/mean respectively).
        def _eval_trial(args_tuple: tuple) -> tuple:
            sched, est, trial = args_tuple
            out_trace = _out_trace(workload, sched, est, trial)
            try:
                ev = run_evaluate(args.saccade, trace, out_trace)
                dist = nrmse_distribution(ev)
                return (
                    sched, est, trial,
                    median_nrmse(ev), mean_coverage(ev),
                    dist["p90"], dist["max"],
                    importance_weighted_nrmse(ev),
                    mean_calibration(ev),
                )
            except subprocess.CalledProcessError as exc:
                tqdm.write(
                    f"  ERROR evaluate {workload}/{sched}/{est} t{trial}: {exc}",
                    file=sys.stderr,
                )
                return sched, est, trial, None, None, None, None, None, None

        n_trials = {sched: args.llm_trials if sched in LLM_SCHEDULERS else 1
                    for sched in schedulers}
        eval_work = [
            (sched, est, t)
            for sched in schedulers
            for est in ESTIMATORS
            for t in range(n_trials[sched])
        ]
        raw: dict[tuple, list] = defaultdict(list)
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            for fut in as_completed(ex.submit(_eval_trial, w) for w in eval_work):
                sched, est, _t, mn, mc, p90, mx, wt, cal = fut.result()
                raw[(sched, est)].append((mn, mc, p90, mx, wt, cal))

        for sched in schedulers:
            for est in ESTIMATORS:
                results = raw[(sched, est)]
                nrmse_vals = [r[0] for r in results if r[0] is not None]
                cov_vals = [r[1] for r in results if r[1] is not None]
                p90_vals = [r[2] for r in results if r[2] is not None]
                max_vals = [r[3] for r in results if r[3] is not None]
                wt_vals = [r[4] for r in results if r[4] is not None]
                cal_vals = [r[5] for r in results if r[5] is not None]
                final_nrmse = float(np.median(nrmse_vals)) if nrmse_vals else None
                final_cov = float(np.mean(cov_vals)) if cov_vals else None
                # nrmse_mean/stddev: variability across LLM trials (same axis as before)
                nrmse_mean = float(np.mean(nrmse_vals)) if nrmse_vals else None
                nrmse_std = (
                    float(np.std(nrmse_vals, ddof=1))
                    if len(nrmse_vals) > 1 else None
                )
                # per-eval distribution stats aggregated across trials
                final_p90 = float(np.median(p90_vals)) if p90_vals else None
                final_max = float(np.median(max_vals)) if max_vals else None
                final_wt = float(np.mean(wt_vals)) if wt_vals else None
                final_cal = float(np.mean(cal_vals)) if cal_vals else None
                sig = is_significant(final_nrmse, noise_floor)
                row = {
                    "workload": workload, "scheduler": sched, "estimator": est,
                    "median_nrmse": final_nrmse if final_nrmse is not None else "",
                    "coverage": final_cov if final_cov is not None else "",
                    "nrmse_mean": nrmse_mean if nrmse_mean is not None else "",
                    "nrmse_stddev": nrmse_std if nrmse_std is not None else "",
                    "nrmse_p90": final_p90 if final_p90 is not None else "",
                    "nrmse_max": final_max if final_max is not None else "",
                    "nrmse_weighted": final_wt if final_wt is not None else "",
                    "mean_calibration": final_cal if final_cal is not None else "",
                }
                if noise_floor is not None:
                    row["significant"] = "" if sig is None else str(sig).lower()
                combo_to_row[(workload, sched, est)] = row

    ordered_rows = [combo_to_row[c] for c in grid_combos if c in combo_to_row]

    grid_fieldnames = [
        "workload",
        "scheduler",
        "estimator",
        "median_nrmse",    # frozen primary metric
        "coverage",
        "nrmse_mean",      # trial-variability: mean of per-trial median_nrmse (LLM only)
        "nrmse_stddev",    # trial-variability: stddev of per-trial median_nrmse (LLM only)
        "nrmse_p90",       # per-eval distribution: 90th-percentile event nRMSE
        "nrmse_max",       # per-eval distribution: worst-case event nRMSE
        "nrmse_weighted",  # per-eval: importance_weighted_nrmse (secondary lens)
        "mean_calibration",
    ]
    if noise_floor is not None:
        grid_fieldnames.append("significant")

    with grid_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grid_fieldnames)
        writer.writeheader()
        writer.writerows(ordered_rows)

    print(f"\nResults written to {run_dir}/", file=sys.stderr)
    print(f"  {grid_csv_path}", file=sys.stderr)
    print(f"  {traces_out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
