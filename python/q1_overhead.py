#!/usr/bin/env python3
"""Measure wall-clock overhead imposed by saccade across parameter grid."""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Parameter grid
Q_SCHEDULE_NS = [100_000, 1_000_000, 10_000_000, 100_000_000]
Q_SAMPLE_NS = [10_000, 100_000, 1_000_000]
SINKS = ["none", "csv", "perfetto"]

CSV_TMP = Path("/tmp/saccade_q1.csv")


def run_timed(cmd: list[str], *, check: bool = True) -> float:
    """Run *cmd* and return wall-clock elapsed time in seconds."""
    t0 = time.perf_counter()
    subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - t0


def measure_baseline(target: list[str], warmup: int, reps: int) -> tuple[float, float]:
    """Run the target binary directly and return (median_s, iqr_s)."""
    for _ in range(warmup):
        run_timed(target)
    times = [run_timed(target) for _ in range(reps)]
    arr = np.array(times)
    return float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))


def build_saccade_cmd(
    saccade: Path,
    library: Path | None,
    q_schedule_ns: int,
    q_sample_ns: int,
    sink: str,
    results_dir: Path,
    target: list[str],
) -> list[str]:
    """Construct the full saccade run command for one grid cell."""
    cmd = [
        str(saccade),
        "run",
        "--q-schedule",
        str(q_schedule_ns),
        "--q-sample",
        str(q_sample_ns),
    ]
    if library is not None:
        cmd += ["--library", str(library)]

    if sink == "none":
        cmd += ["--trace", "/dev/null"]
    elif sink == "csv":
        cmd += ["--csv", str(CSV_TMP), "--trace", "/dev/null"]
    elif sink == "perfetto":
        perfetto_path = results_dir / "q1_tmp.perfetto"
        cmd += ["--trace", str(perfetto_path)]
    else:
        raise ValueError(f"Unknown sink: {sink!r}")

    cmd += ["--"] + target
    return cmd


def measure_saccade(
    cmd: list[str],
    warmup: int,
    reps: int,
) -> tuple[float, float]:
    """Run saccade with *cmd* and return (median_s, iqr_s)."""
    for _ in range(warmup):
        run_timed(cmd)
    times = [run_timed(cmd) for _ in range(reps)]
    arr = np.array(times)
    return float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))


def print_summary(rows: list[dict]) -> None:
    """Print a formatted summary table."""
    header = f"{'q_schedule_ns':>15}  {'q_sample_ns':>12}  {'sink':>9}  {'baseline_s':>10}  {'saccade_s':>10}  {'overhead%':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['q_schedule_ns']:>15}  "
            f"{r['q_sample_ns']:>12}  "
            f"{r['sink']:>9}  "
            f"{r['baseline_median_s']:>10.4f}  "
            f"{r['saccade_median_s']:>10.4f}  "
            f"{r['overhead_fraction'] * 100:>9.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure saccade runtime overhead across a parameter grid."
    )
    parser.add_argument(
        "--saccade",
        type=Path,
        default=Path("../target/release/saccade"),
        help="Path to saccade binary (default: ./target/release/saccade)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default="../event_lib.json",
        help="Event library JSON file (optional)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory for output files (default: ./results)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs per cell, discarded (default: 3)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Number of measured runs per cell (default: 10)",
    )
    parser.add_argument(
        "--target",
        nargs=argparse.REMAINDER,
        required=True,
        help="Workload binary and arguments (e.g. --target /usr/bin/true)",
    )
    args = parser.parse_args()

    # Strip a leading '--' that users may pass to separate target from script flags.
    target: list[str] = args.target
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        parser.error("--target requires at least one argument (the binary to profile)")

    args.saccade = args.saccade.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.library:
        args.library = args.library.resolve()

    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.results_dir / "q1_overhead.csv"

    # Build full grid list for progress bar.
    grid = [
        (sink, q_sched, q_samp)
        for sink in SINKS
        for q_sched in Q_SCHEDULE_NS
        for q_samp in Q_SAMPLE_NS
    ]

    print(f"Measuring baseline ({args.warmup} warmup + {args.reps} reps)...")
    baseline_median, baseline_iqr = measure_baseline(target, args.warmup, args.reps)
    print(f"  Baseline median: {baseline_median:.4f} s  IQR: {baseline_iqr:.4f} s")

    rows: list[dict] = []

    fieldnames = [
        "q_schedule_ns",
        "q_sample_ns",
        "sink",
        "baseline_median_s",
        "saccade_median_s",
        "overhead_fraction",
        "iqr_s",
        "reps",
    ]

    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()

        bar = tqdm(grid, desc="grid cells", unit="cell")
        for sink, q_sched, q_samp in bar:
            bar.set_postfix_str(f"sink={sink} q_sched={q_sched} q_samp={q_samp}")

            cmd = build_saccade_cmd(
                args.saccade,
                args.library,
                q_sched,
                q_samp,
                sink,
                args.results_dir,
                target,
            )

            saccade_median, saccade_iqr = measure_saccade(cmd, args.warmup, args.reps)
            overhead = (saccade_median - baseline_median) / baseline_median

            row = {
                "q_schedule_ns": q_sched,
                "q_sample_ns": q_samp,
                "sink": sink,
                "baseline_median_s": round(baseline_median, 6),
                "saccade_median_s": round(saccade_median, 6),
                "overhead_fraction": round(overhead, 6),
                "iqr_s": round(saccade_iqr, 6),
                "reps": args.reps,
            }
            rows.append(row)
            writer.writerow(row)
            csvfile.flush()

    print_summary(rows)
    print(f"\nResults written to: {out_csv}")


if __name__ == "__main__":
    main()
