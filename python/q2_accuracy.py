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

from sim_utils import (
    SCHEDULERS,
    ESTIMATORS,
    FIXED_NUM_SLOTS,
    LLM_SCHEDULERS,
    filter_traces_by_kind,
    is_significant,
    load_noise_floor,
    mean_coverage,
    median_nrmse,
    parallel_run_combos,
    run_batch_simulate,
    run_combo,
    run_evaluate,
)


# ---------------------------------------------------------------------------
# Per-combo worker (used by parallel_run_combos)
# ---------------------------------------------------------------------------


def _run_one_combo(
    combo: tuple,
    *,
    saccade: Path,
    library: Path,
    trace_map: dict,
    num_slots: int,
    q_schedule: int,
    seed: int | None,
    llm_trials: int,
    traces_out_dir: Path,
    base_config: Path | None,
    noise_floor: float | None,
) -> dict:
    """Execute run_combo for one (workload, scheduler, estimator) tuple.

    Returns a row dict ready for the CSV writer, with error values on failure.
    """
    workload, scheduler, estimator = combo
    try:
        med_nrmse, cov, nrmse_mean, nrmse_std = run_combo(
            saccade=saccade,
            library=library,
            rates_trace=trace_map[workload],
            scheduler=scheduler,
            estimator=estimator,
            num_slots=num_slots,
            q_schedule=q_schedule,
            seed=seed,
            llm_trials=llm_trials,
            tmp_dir=traces_out_dir,
            workload=workload,
            base_config=base_config,
        )
    except subprocess.CalledProcessError as exc:
        tqdm.write(f"  ERROR: {workload}/{scheduler}/{estimator}: {exc}", file=sys.stderr)
        med_nrmse = cov = nrmse_mean = nrmse_std = None

    sig = is_significant(med_nrmse, noise_floor)
    row: dict = {
        "workload": workload,
        "scheduler": scheduler,
        "estimator": estimator,
        "median_nrmse": med_nrmse if med_nrmse is not None else "",
        "coverage": cov if cov is not None else "",
        "nrmse_mean": nrmse_mean if nrmse_mean is not None else "",
        "nrmse_stddev": nrmse_std if nrmse_std is not None else "",
    }
    if noise_floor is not None:
        row["significant"] = "" if sig is None else str(sig).lower()
    return row


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
        "--llm-trials",
        type=int,
        default=3,
        help="Number of trials for non-deterministic LLM schedulers (default: 3)",
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
        "--jobs",
        type=int,
        default=1,
        help="Max parallel workers for non-LLM combos (default: 1)",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=1,
        help="Max parallel workers for LLM combos (default: 1)",
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

    # -----------------------------------------------------------------------
    # Non-LLM combos: batch simulate per workload (1 trace load shared across
    # all scheduler × estimator pairs), then evaluate in parallel.
    # -----------------------------------------------------------------------
    non_llm_schedulers = [s for s in schedulers if s not in LLM_SCHEDULERS]
    llm_schedulers = [s for s in schedulers if s in LLM_SCHEDULERS]

    non_llm_combos = [
        (t.stem, sched, est)
        for t in traces
        for sched in non_llm_schedulers
        for est in ESTIMATORS
    ]

    combo_to_row: dict[tuple, dict] = {}

    def _out_trace(workload: str, sched: str, est: str) -> Path:
        return traces_out_dir / f"est_{workload}_{sched}_{est}_slots{FIXED_NUM_SLOTS}_t0.perfetto"

    if non_llm_combos:
        # Group by workload so each batch call loads the trace only once.
        by_workload: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for workload, sched, est in non_llm_combos:
            by_workload[workload].append((sched, est))

        for trace in tqdm(traces, desc="batch-simulate", unit="workload"):
            workload = trace.stem
            pairs = by_workload.get(workload, [])
            if not pairs:
                continue
            batch_spec = [
                {
                    "scheduler": sched,
                    "estimator": est,
                    "trace": str(_out_trace(workload, sched, est)),
                    **({"seed": args.seed} if args.seed is not None else {}),
                }
                for sched, est in pairs
            ]
            try:
                run_batch_simulate(
                    args.saccade, args.library, trace, batch_spec,
                    FIXED_NUM_SLOTS, args.q_schedule, args.jobs,
                    traces_out_dir, args.base_config,
                )
            except subprocess.CalledProcessError as exc:
                tqdm.write(f"  ERROR batch-simulate {workload}: {exc}", file=sys.stderr)
                for sched, est in pairs:
                    sig = is_significant(None, noise_floor)
                    row: dict = {
                        "workload": workload, "scheduler": sched, "estimator": est,
                        "median_nrmse": "", "coverage": "",
                        "nrmse_mean": "", "nrmse_stddev": "",
                    }
                    if noise_floor is not None:
                        row["significant"] = ""
                    combo_to_row[(workload, sched, est)] = row
                continue

            def _eval(ws_es: tuple[str, str, str]) -> dict:
                wl, sc, es = ws_es
                out_trace = _out_trace(wl, sc, es)
                try:
                    ev = run_evaluate(args.saccade, trace_map[wl], out_trace)
                    mn = median_nrmse(ev)
                    mc = mean_coverage(ev)
                except subprocess.CalledProcessError as exc:
                    tqdm.write(f"  ERROR evaluate {wl}/{sc}/{es}: {exc}", file=sys.stderr)
                    mn = mc = None
                sig = is_significant(mn, noise_floor)
                r: dict = {
                    "workload": wl, "scheduler": sc, "estimator": es,
                    "median_nrmse": mn if mn is not None else "",
                    "coverage": mc if mc is not None else "",
                    "nrmse_mean": mn if mn is not None else "",
                    "nrmse_stddev": "",
                }
                if noise_floor is not None:
                    r["significant"] = "" if sig is None else str(sig).lower()
                return r

            eval_triples = [(workload, sched, est) for sched, est in pairs]
            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
                for fut in as_completed(ex.submit(_eval, t) for t in eval_triples):
                    row = fut.result()
                    combo_to_row[(row["workload"], row["scheduler"], row["estimator"])] = row

    # -----------------------------------------------------------------------
    # LLM combos: keep the existing throttled serial path.
    # -----------------------------------------------------------------------
    from functools import partial
    llm_grid_combos = [
        (t.stem, sched, est)
        for t in traces
        for sched in llm_schedulers
        for est in ESTIMATORS
    ]
    if llm_grid_combos:
        run_fn = partial(
            _run_one_combo,
            saccade=args.saccade,
            library=args.library,
            trace_map=trace_map,
            num_slots=FIXED_NUM_SLOTS,
            q_schedule=args.q_schedule,
            seed=args.seed,
            llm_trials=args.llm_trials,
            traces_out_dir=traces_out_dir,
            base_config=args.base_config,
            noise_floor=noise_floor,
        )
        llm_results = parallel_run_combos(
            llm_grid_combos,
            run_fn,
            jobs=1,
            llm_concurrency=args.llm_concurrency,
        )
        for combo, row in llm_results:
            combo_to_row[combo] = row

    ordered_rows = [combo_to_row[c] for c in grid_combos if c in combo_to_row]

    grid_fieldnames = [
        "workload",
        "scheduler",
        "estimator",
        "median_nrmse",
        "coverage",
        "nrmse_mean",
        "nrmse_stddev",
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
