#!/usr/bin/env python3
"""Sweep scheduler × estimator combinations across workload traces to evaluate
accuracy of adaptive scheduling strategies.

Fixed: num_slots=6, full event library.
Iterates over every .perfetto trace in --traces-dir (or just --workload).
Every combo runs --trials times; per-combo metrics are aggregated across
trials.  Simulation is parallelized through ``saccade simulate --batch`` (one
process per workload loads the rates trace once and fans the full
scheduler × estimator × trial grid across a Rayon pool of ``--jobs`` threads,
with up to ``--batch-jobs`` processes in flight); evaluation of the resulting
traces is parallelized with a thread pool.
Outputs timestamped results under --results-dir/<timestamp>/.
"""

import csv
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import shlex

from sim_utils import (
    SCHEDULERS,
    ESTIMATORS,
    FIXED_NUM_SLOTS,
    LATENCY_PROFILE,
    LLM_SCHEDULERS,
    build_batch_simulate_cmd,
    build_evaluate_cmd,
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
        "--llm-latency-profile",
        type=Path,
        default=LATENCY_PROFILE,
        help="LLM latency profile JSON forwarded to every simulate call "
        "(e.g. a fresh q7 llm_latency_profile.json).",
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
        "--scheduler",
        action="append",
        dest="schedulers",
        choices=SCHEDULERS,
        metavar="SCHEDULER",
        default=[],
        help=(
            "Run only the given scheduler(s) (repeatable). "
            f"Choices: {', '.join(SCHEDULERS)}. Default: all. "
            "Applied before --exclude-scheduler."
        ),
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
        "--trials",
        type=int,
        default=10,
        help=(
            "Number of trials per scheduler×estimator combo; results are "
            "aggregated across trials (default: 10). With --seed, trial t "
            "uses seed+t so trials differ but stay reproducible."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Rayon threads *within* each batch process, and workers for parallel evaluate.",
    )
    parser.add_argument(
        "--batch-jobs",
        type=int,
        default=0,
        help=(
            "Max number of `saccade simulate --batch` subprocesses to run "
            "concurrently. 0 (default) runs all of them at once (one per "
            "workload). Each batch loads its rates trace into memory once, "
            "shared by all trials, so peak RAM ≈ batch-jobs × trace size; "
            "total concurrent LLM requests ≈ batch-jobs × min(jobs, "
            "combos-per-batch)."
        ),
    )
    parser.add_argument(
        "--estimator",
        action="append",
        dest="estimators",
        choices=ESTIMATORS,
        metavar="ESTIMATOR",
        default=[],
        help=(
            "Run only the given estimator(s) (repeatable). "
            f"Choices: {', '.join(ESTIMATORS)}. Default: all."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned grid (workloads × schedulers × estimators) and the "
            "output directory, then exit without running any simulations."
        ),
    )
    return parser


def _print_dry_run(args: argparse.Namespace, traces: list[Path], schedulers: list[str]) -> None:
    """Print the planned grid and every saccade command that would run, then return.

    Reconstructs the same batch-simulate and evaluate invocations as the real
    run (one batch per workload-trial, one evaluate per output trace) using the
    shared command builders, so the printed commands stay in sync with reality.
    The LLM API key is shown as a literal ``$LLM_API_KEY`` placeholder rather
    than reading the real secret from the environment.
    """
    llm_scheds = [s for s in schedulers if s in LLM_SCHEDULERS]
    grid_combos = [
        (t.stem, sched, est)
        for t in traces
        for sched in schedulers
        for est in ESTIMATORS
    ]

    print(
        f"[dry-run] {len(traces)} workload(s) × {len(schedulers)} scheduler(s) "
        f"× {len(ESTIMATORS)} estimator(s) = {len(grid_combos)} combination(s), "
        f"× {args.trials} trial(s) each.",
        file=sys.stderr,
    )
    print(f"[dry-run] Workloads:  {[t.stem for t in traces]}", file=sys.stderr)
    print(f"[dry-run] Schedulers: {schedulers}", file=sys.stderr)
    print(f"[dry-run] Estimators: {ESTIMATORS}", file=sys.stderr)
    n_batches = len(traces)
    max_batches = n_batches if args.batch_jobs <= 0 else min(args.batch_jobs, n_batches)
    print(
        f"[dry-run] {n_batches} batch(es) (one per workload), "
        f"up to {max_batches} concurrently, {args.jobs} Rayon threads each.",
        file=sys.stderr,
    )
    if llm_scheds:
        print(
            f"[dry-run] LLM schedulers (model={args.llm_model}): {llm_scheds}",
            file=sys.stderr,
        )
    if args.exclude_schedulers:
        print(f"[dry-run] Excluded: {args.exclude_schedulers}", file=sys.stderr)

    # Placeholder output layout (no directories are created in a dry run).
    run_dir = args.results_dir / "<timestamp>"
    traces_out_dir = run_dir / "traces"
    print(
        f"[dry-run] Would write results under {run_dir}/.\n"
        f"[dry-run] Commands that would run (API key shown as $LLM_API_KEY):\n",
        file=sys.stderr,
    )

    def _out_trace(workload: str, sched: str, est: str, trial: int) -> Path:
        return traces_out_dir / (
            f"est_{workload}_{sched}_{est}_slots{FIXED_NUM_SLOTS}_t{trial}.perfetto"
        )

    for trace in traces:
        workload = trace.stem
        # One batch per workload covering the full scheduler × estimator
        # × trial grid (the trace is loaded once per batch process).
        spec_path = traces_out_dir / f"batch_spec_{workload}.json"
        cmd = build_batch_simulate_cmd(
            args.saccade, args.library, trace, spec_path,
            FIXED_NUM_SLOTS, args.q_schedule, args.jobs,
            args.llm_model, args.base_config, api_key="$LLM_API_KEY",
            latency_profile=args.llm_latency_profile,
        )
        combos = len(schedulers) * len(ESTIMATORS) * args.trials
        seed_note = (
            f", seeds {args.seed}..{args.seed + args.trials - 1}"
            if args.seed is not None else ""
        )
        print(
            f"# {workload}: batch of {combos} combo(s) "
            f"({len(schedulers)} sched × {len(ESTIMATORS)} est "
            f"× {args.trials} trial(s){seed_note})",
            file=sys.stderr,
        )
        print(shlex.join(cmd), file=sys.stderr)

        for sched in schedulers:
            for est in ESTIMATORS:
                for trial in range(args.trials):
                    cmd = build_evaluate_cmd(
                        args.saccade, trace, _out_trace(workload, sched, est, trial)
                    )
                    print(shlex.join(cmd), file=sys.stderr)

    print("\n[dry-run] No simulations run.", file=sys.stderr)


def main() -> None:
    global ESTIMATORS
    parser = build_parser()
    args = parser.parse_args()

    # Narrow the estimator grid if the user selected a subset (dedupe, keep order).
    if args.estimators:
        ESTIMATORS = list(dict.fromkeys(args.estimators))

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

    # Narrow to the requested scheduler(s) if given, then drop any excluded.
    # Iterate SCHEDULERS so grid order stays canonical regardless of CLI order.
    include = set(args.schedulers) if args.schedulers else set(SCHEDULERS)
    schedulers = [
        s for s in SCHEDULERS if s in include and s not in args.exclude_schedulers
    ]
    if not schedulers:
        parser.error("No schedulers selected (check --scheduler / --exclude-scheduler).")

    if args.dry_run:
        _print_dry_run(args, traces, schedulers)
        return

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

    print(
        f"Found {len(traces)} workload trace(s). "
        f"Grid: {len(schedulers)} schedulers × {len(ESTIMATORS)} estimators "
        f"× {len(traces)} workloads = {len(grid_combos)} total.",
        file=sys.stderr,
    )
    if args.exclude_schedulers:
        print(f"Excluded schedulers: {args.exclude_schedulers}", file=sys.stderr)

    def _out_trace(workload: str, sched: str, est: str, trial: int = 0) -> Path:
        return traces_out_dir / f"est_{workload}_{sched}_{est}_slots{FIXED_NUM_SLOTS}_t{trial}.perfetto"

    def _batch_spec(workload: str) -> list[dict]:
        entries = []
        for sched in schedulers:
            for est in ESTIMATORS:
                for trial in range(args.trials):
                    entry: dict = {
                        "scheduler": sched,
                        "estimator": est,
                        "trace": str(_out_trace(workload, sched, est, trial)),
                    }
                    if args.seed is not None:
                        # Offset per trial so repeated non-LLM runs actually
                        # differ while staying reproducible.
                        entry["seed"] = args.seed + trial
                    entries.append(entry)
        return entries

    # --- Phase 1: simulate.  One batch per workload, fired concurrently. ---
    # Each batch process loads its rates trace ONCE and fans the full
    # scheduler × estimator × trial grid across a Rayon pool of --jobs
    # threads, so trials share one copy of the trace data structures; the
    # batch *processes* themselves run in parallel up to --batch-jobs.
    max_batches = (
        len(traces) if args.batch_jobs <= 0 else min(args.batch_jobs, len(traces))
    )

    def _run_batch(trace: Path) -> None:
        run_batch_simulate(
            args.saccade, args.library, trace, _batch_spec(trace.stem),
            FIXED_NUM_SLOTS, args.q_schedule, args.jobs,
            traces_out_dir, args.llm_model, args.base_config,
            latency_profile=args.llm_latency_profile,
        )

    print(
        f"Simulating {len(traces)} batch(es) (one per workload, "
        f"{len(schedulers) * len(ESTIMATORS)} combos × {args.trials} trial(s) each), "
        f"up to {max_batches} concurrently, {args.jobs} threads each.",
        file=sys.stderr,
    )
    with ThreadPoolExecutor(max_workers=max(1, max_batches)) as ex:
        futs = {ex.submit(_run_batch, t): t for t in traces}
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="batch-simulate", unit="batch"
        ):
            trace = futs[fut]
            try:
                fut.result()
            except subprocess.CalledProcessError as exc:
                # The batch is internally failure-tolerant (it writes surviving
                # combos' traces); a non-zero exit means *all* its combos failed.
                # Missing traces are handled as None in the evaluate phase below.
                tqdm.write(
                    f"  ERROR batch-simulate {trace.stem}: {exc}",
                    file=sys.stderr,
                )
                if exc.stderr:
                    tqdm.write(f"    saccade stderr:\n{exc.stderr}", file=sys.stderr)

    # --- Phase 2: evaluate every produced trace in parallel, aggregate by combo. ---
    # Each trial returns (median_nrmse, mean_coverage, p90, max, weighted, calibration).
    # nrmse_mean/nrmse_stddev are trial-variability stats (across trials);
    # nrmse_p90/nrmse_max/nrmse_weighted are per-eval distribution stats aggregated
    # across trials with the same scheme (median/mean respectively).
    def _eval_trial(work_tuple: tuple) -> tuple:
        trace, sched, est, trial = work_tuple
        workload = trace.stem
        out_trace = _out_trace(workload, sched, est, trial)
        if not out_trace.exists():
            # A single combo can fail (transient LLM timeout) without aborting
            # the batch; its trace simply won't be written.
            return workload, sched, est, None, None, None, None, None, None
        try:
            ev = run_evaluate(args.saccade, trace, out_trace)
            dist = nrmse_distribution(ev)
            return (
                workload, sched, est,
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
            return workload, sched, est, None, None, None, None, None, None

    eval_work = [
        (trace, sched, est, t)
        for trace in traces
        for sched in schedulers
        for est in ESTIMATORS
        for t in range(args.trials)
    ]
    raw: dict[tuple, list] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for fut in tqdm(
            as_completed(ex.submit(_eval_trial, w) for w in eval_work),
            total=len(eval_work), desc="evaluate", unit="eval",
        ):
            wl, sched, est, mn, mc, p90, mx, wt, cal = fut.result()
            raw[(wl, sched, est)].append((mn, mc, p90, mx, wt, cal))

    combo_to_row: dict[tuple, dict] = {}

    for trace in traces:
        workload = trace.stem
        for sched in schedulers:
            for est in ESTIMATORS:
                results = raw[(workload, sched, est)]
                nrmse_vals = [r[0] for r in results if r[0] is not None]
                cov_vals = [r[1] for r in results if r[1] is not None]
                p90_vals = [r[2] for r in results if r[2] is not None]
                max_vals = [r[3] for r in results if r[3] is not None]
                wt_vals = [r[4] for r in results if r[4] is not None]
                cal_vals = [r[5] for r in results if r[5] is not None]
                final_nrmse = float(np.median(nrmse_vals)) if nrmse_vals else None
                final_cov = float(np.mean(cov_vals)) if cov_vals else None
                # nrmse_mean/stddev: variability across trials (same axis as before)
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
        "nrmse_mean",      # trial-variability: mean of per-trial median_nrmse
        "nrmse_stddev",    # trial-variability: stddev of per-trial median_nrmse
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
