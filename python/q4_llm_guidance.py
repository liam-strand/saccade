#!/usr/bin/env python3
"""Compare LLM-driven schedulers with and without a guidance hint.

Runs every LLM scheduler under two conditions:
  guidance_condition=none           -- no guidance passed to the LLM
  guidance_condition=with_guidance  -- the --guidance string is forwarded

Operates on SPEC workloads only.  Simulation is parallelized through
``saccade simulate --batch`` (one process per workload-trial loads the rates
trace once and fans the combos across a Rayon pool of ``--jobs`` threads);
evaluation of the resulting traces is parallelized with a thread pool.
Outputs a timestamped CSV.
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sim_utils import (
    FIXED_NUM_SLOTS,
    LATENCY_PROFILE,
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

DEFAULT_GUIDANCE = (
    "Focus on memory hierarchy behavior: cache misses, TLB pressure, and memory "
    "bandwidth utilization. Deprioritize integer arithmetic counters."
)

LLM_SCHEDULER_LIST = sorted(LLM_SCHEDULERS)

# (label, has-guidance) pairs.  The guidance string itself is resolved from
# args at runtime so --guidance can override it.
CONDITIONS = [("none", False), ("with_guidance", True)]


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
        default="ema",
        help="Estimator to pair with each LLM scheduler (default: ema)",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default="google/gemma-4-26b-a4b-it",
        help="Model name for LLM-driven schedulers; uses OpenRouter unless sim_utils is customized.",
    )
    p.add_argument(
        "--llm-latency-profile",
        type=Path,
        default=LATENCY_PROFILE,
        help="LLM latency profile JSON forwarded to every simulate call "
        "(e.g. a fresh q7 llm_latency_profile.json).",
    )
    p.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help=(
            "TOML config forwarded to every simulate call. "
            "Defaults to None; the EMA estimator needs no [kalman] config. "
            "(Was kf_expert.toml, which drove Kalman divergence in earlier runs.)"
        ),
    )
    p.add_argument(
        "--guidance",
        type=str,
        default=DEFAULT_GUIDANCE,
        help="Guidance string for the 'with_guidance' condition",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Rayon threads *within* each batch process, and workers for parallel evaluate.",
    )
    p.add_argument(
        "--batch-jobs",
        type=int,
        default=0,
        help=(
            "Max number of `saccade simulate --batch` subprocesses to run "
            "concurrently. 0 (default) runs all of them at once (one per "
            "workload×trial). Each batch loads the rates trace into memory, so "
            "peak RAM ≈ batch-jobs × trace size; total concurrent LLM requests "
            "≈ batch-jobs × min(jobs, combos-per-batch)."
        ),
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
    if args.base_config is not None:
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

    truncated_guidance = (
        args.guidance[:80] + "…" if len(args.guidance) > 80 else args.guidance
    )

    def _out_trace(workload: str, sched: str, cond: str, trial: int) -> Path:
        return tmp_dir / f"est_{workload}_{sched}_{cond}_t{trial}.perfetto"

    def _batch_spec(workload: str, trial: int) -> list[dict]:
        entries: list[dict] = []
        for sched in LLM_SCHEDULER_LIST:
            for cond_label, has_guidance in CONDITIONS:
                entry: dict = {
                    "scheduler": sched,
                    "estimator": args.estimator,
                    "trace": str(_out_trace(workload, sched, cond_label, trial)),
                }
                if has_guidance:
                    entry["guidance"] = args.guidance
                if args.seed is not None:
                    entry["seed"] = args.seed
                entries.append(entry)
        return entries

    combo_to_row: dict[tuple, dict] = {}

    # --- Phase 1: simulate.  Fire every (workload, trial) batch concurrently. ---
    # Every scheduler here is LLM (nondeterministic), so all conditions are
    # repeated across all trials.  Each batch process fans its (scheduler,
    # condition) combos across the Rayon pool; the batch *processes* themselves
    # run in parallel up to --batch-jobs.  spec_tag keeps the per-trial spec and
    # stderr-log filenames from colliding across concurrent batches of the same
    # trace.
    batch_work = [
        (trace, trial) for trace in traces for trial in range(args.llm_trials)
    ]
    max_batches = (
        len(batch_work) if args.batch_jobs <= 0 else min(args.batch_jobs, len(batch_work))
    )

    def _run_batch(work: tuple) -> None:
        trace, trial = work
        run_batch_simulate(
            args.saccade, args.library, trace, _batch_spec(trace.stem, trial),
            FIXED_NUM_SLOTS, args.q_schedule, args.jobs,
            tmp_dir, args.llm_model, args.base_config, spec_tag=f"t{trial}",
            latency_profile=args.llm_latency_profile,
        )

    print(
        f"Simulating {len(batch_work)} batch(es) "
        f"({len(traces)} workload × {args.llm_trials} trials), "
        f"up to {max_batches} concurrently, {args.jobs} threads each.",
        file=sys.stderr,
    )
    with ThreadPoolExecutor(max_workers=max(1, max_batches)) as ex:
        futs = {ex.submit(_run_batch, w): w for w in batch_work}
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="batch-simulate", unit="batch"
        ):
            trace, trial = futs[fut]
            try:
                fut.result()
            except subprocess.CalledProcessError as exc:
                # The batch is internally failure-tolerant (it writes surviving
                # combos' traces); a non-zero exit means *all* its combos failed.
                # Missing traces are handled as None in the evaluate phase below.
                tqdm.write(
                    f"  ERROR batch-simulate {trace.stem} trial {trial}: {exc}",
                    file=sys.stderr,
                )
                if exc.stderr:
                    tqdm.write(f"    saccade stderr:\n{exc.stderr}", file=sys.stderr)

    # --- Phase 2: evaluate every produced trace in parallel, aggregate by combo. ---
    def _eval_trial(work_tuple: tuple) -> tuple:
        trace, sched, cond, trial = work_tuple
        out_trace = _out_trace(trace.stem, sched, cond, trial)
        if not out_trace.exists():
            # A single combo can fail (transient LLM timeout) without aborting
            # the batch; its trace simply won't be written.
            return trace.stem, sched, cond, None, None, None, None, None, None
        try:
            ev = run_evaluate(args.saccade, trace, out_trace)
            dist = nrmse_distribution(ev)
            return (
                trace.stem, sched, cond,
                median_nrmse(ev), mean_coverage(ev),
                dist["p90"], dist["max"],
                importance_weighted_nrmse(ev), mean_calibration(ev),
            )
        except subprocess.CalledProcessError as exc:
            tqdm.write(
                f"  ERROR evaluate {trace.stem}/{sched}/{cond}: {exc}",
                file=sys.stderr,
            )
            return trace.stem, sched, cond, None, None, None, None, None, None

    eval_work = [
        (trace, sched, cond_label, t)
        for trace in traces
        for sched in LLM_SCHEDULER_LIST
        for cond_label, _ in CONDITIONS
        for t in range(args.llm_trials)
    ]
    raw: dict[tuple, list] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for fut in tqdm(
            as_completed(ex.submit(_eval_trial, w) for w in eval_work),
            total=len(eval_work), desc="evaluate", unit="eval",
        ):
            wl, sched, cond, mn, mc, p90, mx, wt, cal = fut.result()
            raw[(wl, sched, cond)].append((mn, mc, p90, mx, wt, cal))

    for trace in traces:
        workload = trace.stem
        for sched in LLM_SCHEDULER_LIST:
            for cond_label, _ in CONDITIONS:
                results = raw[(workload, sched, cond_label)]
                nrmse_vals = [r[0] for r in results if r[0] is not None]
                cov_vals = [r[1] for r in results if r[1] is not None]
                p90_vals = [r[2] for r in results if r[2] is not None]
                max_vals = [r[3] for r in results if r[3] is not None]
                wt_vals = [r[4] for r in results if r[4] is not None]
                cal_vals = [r[5] for r in results if r[5] is not None]
                final_nrmse = float(np.median(nrmse_vals)) if nrmse_vals else None
                final_cov = float(np.mean(cov_vals)) if cov_vals else None
                nrmse_mean = float(np.mean(nrmse_vals)) if nrmse_vals else None
                nrmse_std = (
                    float(np.std(nrmse_vals, ddof=1)) if len(nrmse_vals) > 1 else None
                )
                final_p90 = float(np.median(p90_vals)) if p90_vals else None
                final_max = float(np.median(max_vals)) if max_vals else None
                final_wt = float(np.mean(wt_vals)) if wt_vals else None
                final_cal = float(np.mean(cal_vals)) if cal_vals else None
                sig = is_significant(final_nrmse, noise_floor)

                def _fmt(v: float | None) -> str:
                    return "" if v is None else str(v)

                row = {
                    "workload": workload, "scheduler": sched,
                    "estimator": args.estimator,
                    "guidance_condition": cond_label,
                    "guidance_text": "" if cond_label == "none" else truncated_guidance,
                    "median_nrmse": _fmt(final_nrmse),
                    "coverage": _fmt(final_cov),
                    "nrmse_mean": _fmt(nrmse_mean),
                    "nrmse_stddev": _fmt(nrmse_std),
                    "nrmse_p90": _fmt(final_p90),
                    "nrmse_max": _fmt(final_max),
                    "nrmse_weighted": _fmt(final_wt),
                    "mean_calibration": _fmt(final_cal),
                }
                if noise_floor is not None:
                    row["significant"] = "" if sig is None else str(sig).lower()
                combo_to_row[(workload, sched, cond_label)] = row

    ordered_rows = [
        combo_to_row[(t.stem, sched, cond_label)]
        for t in traces
        for sched in LLM_SCHEDULER_LIST
        for cond_label, _ in CONDITIONS
        if (t.stem, sched, cond_label) in combo_to_row
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_rows)

    print(f"\nResults written to: {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
