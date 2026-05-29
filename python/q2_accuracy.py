#!/usr/bin/env python3
"""Sweep (scheduler × estimator) and (scheduler × slot-count) combinations
across all workload traces to evaluate accuracy of adaptive scheduling strategies.

Scheduler/estimator grid: fixed pool_size=16 and num_slots=4, sweeps all
scheduler × estimator combinations. Outputs q2_scheduler_estimator.csv.
Slot-count axis: controls --num-slots passed to saccade simulate.
Both sweeps iterate over every .perfetto trace in --traces-dir.
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCHEDULERS = [
    "round-robin",
    "random",
    "max-uncertainty",
    "rate-of-change",
    "static-llm",
    "dynamic-llm",
    "weighted-round-robin-llm",
]
ESTIMATORS = ["propagate", "ema", "kalman"]
LLM_SCHEDULERS = {"static-llm", "dynamic-llm", "weighted-round-robin-llm"}

# Slot counts: --num-slots values to sweep.
SLOT_COUNTS = [2, 4, 6, 8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_simulate(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    estimator: str,
    out_trace: Path,
    num_slots: int,
    q_schedule: int,
    seed: int | None,
) -> None:
    """Run saccade simulate, writing output to out_trace."""
    cmd = [
        str(saccade),
        "simulate",
        "--library",
        str(library),
        "--rates-trace",
        str(rates_trace),
        "--scheduler",
        scheduler,
        "--estimator",
        estimator,
        "--trace",
        str(out_trace),
        "--num-slots",
        str(num_slots),
        "--q-schedule",
        str(q_schedule),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_evaluate(
    saccade: Path,
    ground_truth: Path,
    estimated: Path,
) -> dict:
    """Run saccade evaluate --json and return the parsed JSON dict."""
    cmd = [
        str(saccade),
        "evaluate",
        "--ground-truth",
        str(ground_truth),
        "--estimated",
        str(estimated),
        "--json",
    ]
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def median_nrmse(eval_json: dict) -> float | None:
    """Compute median nRMSE across events, filtering out null (zero GT mean) entries."""
    vals = [e["nrmse"] for e in eval_json["per_event"] if e["nrmse"] is not None]
    if not vals:
        return None
    return float(np.median(vals))


def mean_coverage(eval_json: dict) -> float | None:
    """Return the mean coverage from the evaluate JSON (already computed by saccade)."""
    return eval_json.get("mean_coverage")


def write_filtered_library(
    library_data: dict,
    events_in_gt: list[str],
    pool_size: int,
    dest: Path,
) -> int:
    """Write a library JSON with only the first pool_size events that appear in the GT
    trace. Returns the actual number of events written (may be < pool_size if the GT
    trace has fewer events)."""
    all_events = library_data["events"]

    # Build a lookup from name to event dict for quick access.
    name_to_event = {e["name"]: e for e in all_events}

    # Take the first pool_size events that are both in the library and in the GT trace.
    filtered = []
    for name in events_in_gt:
        if name in name_to_event:
            filtered.append(name_to_event[name])
        if len(filtered) >= pool_size:
            break

    filtered_lib = {"events": filtered}
    dest.write_text(json.dumps(filtered_lib))
    return len(filtered)


def extract_gt_event_names(rates_trace: Path, saccade: Path) -> list[str]:
    """Extract the ordered list of event names present in the GT trace by running
    evaluate against itself."""
    result = subprocess.run(
        [
            str(saccade),
            "evaluate",
            "--ground-truth",
            str(rates_trace),
            "--estimated",
            str(rates_trace),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    # Deduplicate preserving order (multiple TIDs may share an event name).
    seen: set[str] = set()
    names: list[str] = []
    for entry in data["per_event"]:
        name = entry["event"]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def load_noise_floor(path: Path | None) -> float | None:
    """Load the noise floor threshold from a JSON file, if provided.

    The file is expected to contain a single numeric value under key "nrmse_floor",
    representing the minimum nRMSE delta considered statistically significant.
    """
    if path is None:
        return None
    data = json.loads(path.read_text())
    return float(data["nrmse_floor"])


def is_significant(nrmse: float | None, floor: float | None) -> bool | None:
    """Return True if the nRMSE value is above the noise floor, None if floor is not
    available or nrmse is None."""
    if floor is None or nrmse is None:
        return None
    return nrmse > floor


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def simulate_and_eval(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    estimator: str,
    num_slots: int,
    q_schedule: int,
    seed: int | None,
    tmp_dir: Path,
    workload: str,
    trial: int = 0,
) -> dict:
    """Run one simulate + evaluate pair. Returns the evaluate JSON dict."""
    trace_path = tmp_dir / f"est_{workload}_{scheduler}_{estimator}_slots{num_slots}_t{trial}.perfetto"
    run_simulate(
        saccade,
        library,
        rates_trace,
        scheduler,
        estimator,
        trace_path,
        num_slots,
        q_schedule,
        seed,
    )
    result = run_evaluate(saccade, rates_trace, trace_path)
    # Clean up trace immediately to save disk space.
    trace_path.unlink(missing_ok=True)
    return result


def run_combo(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    estimator: str,
    num_slots: int,
    q_schedule: int,
    seed: int | None,
    llm_trials: int,
    tmp_dir: Path,
    workload: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Run simulate+evaluate for a given combo, repeating for LLM schedulers.

    Returns (median_nrmse, coverage, nrmse_mean, nrmse_stddev).
    nrmse_mean and nrmse_stddev are only populated for LLM schedulers.
    """
    n_trials = llm_trials if scheduler in LLM_SCHEDULERS else 1
    nrmse_vals: list[float] = []
    coverage_vals: list[float] = []

    for trial in range(n_trials):
        eval_json = simulate_and_eval(
            saccade,
            library,
            rates_trace,
            scheduler,
            estimator,
            num_slots,
            q_schedule,
            seed,
            tmp_dir,
            workload,
            trial,
        )
        mn = median_nrmse(eval_json)
        mc = mean_coverage(eval_json)
        if mn is not None:
            nrmse_vals.append(mn)
        if mc is not None:
            coverage_vals.append(mc)

    final_nrmse = float(np.median(nrmse_vals)) if nrmse_vals else None
    final_cov = float(np.mean(coverage_vals)) if coverage_vals else None

    if n_trials > 1 and nrmse_vals:
        nrmse_mean = float(np.mean(nrmse_vals))
        nrmse_std = float(np.std(nrmse_vals, ddof=1)) if len(nrmse_vals) > 1 else 0.0
    else:
        nrmse_mean = final_nrmse
        nrmse_std = None

    return final_nrmse, final_cov, nrmse_mean, nrmse_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep (scheduler × estimator) and (scheduler × slot-count) combinations "
            "across all workload traces to evaluate accuracy of adaptive scheduling strategies."
        )
    )
    parser.add_argument(
        "--saccade",
        type=Path,
        default=Path("./target/release/saccade"),
        help="Path to the saccade binary (default: ./target/release/saccade)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
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
        default=5,
        help="Number of trials for non-deterministic LLM schedulers (default: 5)",
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
            "Optional JSON file with key 'nrmse_floor' (float). When provided, a "
            "'significant' column is added to CSVs indicating whether the nRMSE is "
            "above the noise floor."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve paths.
    args.saccade = args.saccade.resolve()
    args.library = args.library.resolve()
    args.traces_dir = args.traces_dir.resolve()
    args.results_dir = args.results_dir.resolve()

    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")
    if not args.library.exists():
        parser.error(f"Library file not found: {args.library}")
    if not args.traces_dir.is_dir():
        parser.error(f"Traces directory not found: {args.traces_dir}")

    traces = sorted(args.traces_dir.glob("*.perfetto"))
    if not traces:
        parser.error(f"No .perfetto files found in {args.traces_dir}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    noise_floor = load_noise_floor(args.noise_floor_json)

    # Load library JSON once.
    library_data = json.loads(args.library.read_text())

    FIXED_POOL_SIZE = 16
    FIXED_NUM_SLOTS = 4
    FIXED_ESTIMATOR = "propagate"

    grid_csv_path = args.results_dir / "q2_scheduler_estimator.csv"
    slot_csv_path = args.results_dir / "q2_slot_count.csv"

    grid_rows: list[dict] = []
    slot_rows: list[dict] = []

    grid_combos = [(sched, est) for sched in SCHEDULERS for est in ESTIMATORS]
    slot_combos = [(sched, ns) for sched in SCHEDULERS for ns in SLOT_COUNTS]

    print(
        f"Found {len(traces)} workload trace(s). "
        f"Grid: {len(grid_combos)} combos × {len(traces)} workloads = "
        f"{len(grid_combos) * len(traces)} total. "
        f"Slot-count: {len(slot_combos)} combos × {len(traces)} workloads = "
        f"{len(slot_combos) * len(traces)} total.",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for trace_path in traces:
            workload = trace_path.stem
            print(f"\n=== Workload: {workload} ===", file=sys.stderr)

            print("  Discovering events in GT trace...", file=sys.stderr)
            gt_event_names = extract_gt_event_names(trace_path, args.saccade)
            print(f"  Found {len(gt_event_names)} unique events.", file=sys.stderr)

            filtered_lib_path = tmp_dir / f"lib_pool{FIXED_POOL_SIZE}_{workload}.json"
            actual_size = write_filtered_library(
                library_data, gt_event_names, FIXED_POOL_SIZE, filtered_lib_path
            )
            if actual_size < FIXED_POOL_SIZE:
                print(
                    f"  Warning: pool_size={FIXED_POOL_SIZE} requested but only "
                    f"{actual_size} events available in GT trace + library.",
                    file=sys.stderr,
                )

            # -------------------------------------------------------------------
            # Scheduler/estimator grid
            # -------------------------------------------------------------------
            bar = tqdm(grid_combos, desc=f"  grid [{workload}]", unit="combo")
            for scheduler, estimator in bar:
                bar.set_postfix_str(f"{scheduler}/{estimator}")

                try:
                    med_nrmse, cov, nrmse_mean, nrmse_std = run_combo(
                        saccade=args.saccade,
                        library=filtered_lib_path,
                        rates_trace=trace_path,
                        scheduler=scheduler,
                        estimator=estimator,
                        num_slots=FIXED_NUM_SLOTS,
                        q_schedule=args.q_schedule,
                        seed=args.seed,
                        llm_trials=args.llm_trials,
                        tmp_dir=tmp_dir,
                        workload=workload,
                    )
                except subprocess.CalledProcessError as exc:
                    tqdm.write(
                        f"  ERROR: {workload}/{scheduler}/{estimator}: {exc}",
                        file=sys.stderr,
                    )
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
                grid_rows.append(row)

            # -------------------------------------------------------------------
            # Slot-count sweep
            # -------------------------------------------------------------------
            bar = tqdm(slot_combos, desc=f"  slots [{workload}]", unit="combo")
            for scheduler, num_slots in bar:
                bar.set_postfix_str(f"{scheduler}/slots={num_slots}")

                try:
                    med_nrmse, cov, nrmse_mean, nrmse_std = run_combo(
                        saccade=args.saccade,
                        library=filtered_lib_path,
                        rates_trace=trace_path,
                        scheduler=scheduler,
                        estimator=FIXED_ESTIMATOR,
                        num_slots=num_slots,
                        q_schedule=args.q_schedule,
                        seed=args.seed,
                        llm_trials=args.llm_trials,
                        tmp_dir=tmp_dir,
                        workload=workload,
                    )
                except subprocess.CalledProcessError as exc:
                    tqdm.write(
                        f"  ERROR: {workload}/{scheduler}/slots={num_slots}: {exc}",
                        file=sys.stderr,
                    )
                    med_nrmse = cov = nrmse_mean = nrmse_std = None

                sig = is_significant(med_nrmse, noise_floor)

                row = {
                    "workload": workload,
                    "scheduler": scheduler,
                    "num_slots": num_slots,
                    "median_nrmse": med_nrmse if med_nrmse is not None else "",
                    "coverage": cov if cov is not None else "",
                }
                if noise_floor is not None:
                    row["significant"] = "" if sig is None else str(sig).lower()
                slot_rows.append(row)

    # -----------------------------------------------------------------------
    # Write CSVs
    # -----------------------------------------------------------------------
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
        writer.writerows(grid_rows)

    slot_fieldnames = ["workload", "scheduler", "num_slots", "median_nrmse", "coverage"]
    if noise_floor is not None:
        slot_fieldnames.append("significant")

    with slot_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=slot_fieldnames)
        writer.writeheader()
        writer.writerows(slot_rows)

    print(f"\nResults written to:", file=sys.stderr)
    print(f"  {grid_csv_path}", file=sys.stderr)
    print(f"  {slot_csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
