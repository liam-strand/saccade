#!/usr/bin/env python3
"""Measure wall-clock overhead imposed by saccade across a parameter grid.

Two workloads are measured:
  * "target" -- the real workload passed via --target. Overhead is reported as a
    fraction of the unprofiled baseline (q1_overhead.csv).
  * "null"   -- /bin/true, which exits instantly. saccade-vs-bare here isolates
    saccade's *fixed* startup/teardown cost (eBPF attach, perf_event_open,
    spawn/reap) in absolute terms, independent of workload runtime
    (q1_startup.csv). A relative overhead is meaningless against a ~0 s baseline.

For each workload the unprofiled baseline is treated as just another treatment
and interleaved with the saccade configs: all (rounds x treatments) individual
runs are shuffled together with a seeded RNG, so thermal/load drift over the
session is decorrelated from both configuration and round rather than being
misattributed to whichever configs happen to run late.

Every individual run is persisted to a raw CSV (one row per run) so true
distributions and a drift regression (overhead vs execution order) are possible
downstream.
"""

import argparse
import csv
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Parameter grid
Q_SCHEDULE_NS = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
Q_SAMPLE_NS = [1_000, 10_000, 100_000]
SINKS = ["none", "csv", "perfetto"]

CSV_TMP = Path("/tmp/saccade_q1.csv")


def run_timed(cmd: list[str], *, check: bool = True) -> tuple[float, int | None]:
    """Run *cmd* and return (elapsed_s, samples_emitted | None).

    stderr is captured and searched for the last occurrence of
    ``samples_emitted=<N>`` (the INFO-level line emitted by ``saccade run``).
    Baseline runs (bare target, no saccade) emit nothing and return None for the
    sample count.  stdout is discarded.  subprocess.run drains the pipe so the
    single-line output cannot cause a deadlock.
    """
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    matches = re.findall(r"samples_emitted=(\d+)", result.stderr)
    samples_emitted = int(matches[-1]) if matches else None
    return elapsed, samples_emitted


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


def build_units(workloads: list[str]) -> list[dict]:
    """Per workload: one baseline unit plus the 36 grid cells."""
    units: list[dict] = []
    for wl in workloads:
        units.append(
            {"workload": wl, "kind": "baseline", "sink": "baseline", "q_sched": None, "q_samp": None}
        )
        for sink in SINKS:
            for q_sched in Q_SCHEDULE_NS:
                for q_samp in Q_SAMPLE_NS:
                    units.append(
                        {
                            "workload": wl,
                            "kind": "saccade",
                            "sink": sink,
                            "q_sched": q_sched,
                            "q_samp": q_samp,
                        }
                    )
    return units


def aggregate(
    raw_rows: list[dict], workload: str
) -> tuple[float, dict[tuple, np.ndarray], dict[tuple, float | None]]:
    """Return (baseline_median_s, {config: elapsed-time samples}, {config: samples_emitted median}).

    The third element maps each (sink, q_sched, q_samp) key to the median
    samples_emitted across reps (float), or None if no rows had a value.
    """
    base = [r["elapsed_s"] for r in raw_rows if r["workload"] == workload and r["is_baseline"]]
    baseline_median = float(np.median(base))
    by_config: dict[tuple, list[float]] = {}
    by_samples: dict[tuple, list[int]] = {}
    for r in raw_rows:
        if r["workload"] != workload or r["is_baseline"]:
            continue
        key = (r["sink"], r["q_schedule_ns"], r["q_sample_ns"])
        by_config.setdefault(key, []).append(r["elapsed_s"])
        se = r.get("samples_emitted")
        if se is not None:
            by_samples.setdefault(key, []).append(int(se))
    samples_median: dict[tuple, float | None] = {
        k: float(np.median(v)) if v else None for k, v in by_samples.items()
    }
    # Ensure every config key is present (may be None if no samples were recorded).
    for k in by_config:
        samples_median.setdefault(k, None)
    return baseline_median, {k: np.array(v) for k, v in by_config.items()}, samples_median


def write_overhead_csv(
    baseline_median: float,
    by_config: dict,
    reps: int,
    out_csv: Path,
    samples_median: dict | None = None,
) -> list[dict]:
    """Real-workload summary: per-config median overhead as a fraction of baseline."""
    fieldnames = [
        "q_schedule_ns",
        "q_sample_ns",
        "sink",
        "baseline_median_s",
        "saccade_median_s",
        "overhead_fraction",
        "iqr_s",
        "reps",
        "samples_median",
    ]
    rows: list[dict] = []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sink in SINKS:
            for q_sched in Q_SCHEDULE_NS:
                for q_samp in Q_SAMPLE_NS:
                    times = by_config[(sink, q_sched, q_samp)]
                    saccade_median = float(np.median(times))
                    iqr = float(np.percentile(times, 75) - np.percentile(times, 25))
                    overhead = (saccade_median - baseline_median) / baseline_median
                    sm = None
                    if samples_median is not None:
                        raw_sm = samples_median.get((sink, q_sched, q_samp))
                        sm = round(raw_sm) if raw_sm is not None else ""
                    row = {
                        "q_schedule_ns": q_sched,
                        "q_sample_ns": q_samp,
                        "sink": sink,
                        "baseline_median_s": round(baseline_median, 6),
                        "saccade_median_s": round(saccade_median, 6),
                        "overhead_fraction": round(overhead, 6),
                        "iqr_s": round(iqr, 6),
                        "reps": reps,
                        "samples_median": sm if sm is not None else "",
                    }
                    writer.writerow(row)
                    rows.append(row)
    return rows


def write_startup_csv(
    baseline_median: float, by_config: dict, reps: int, out_csv: Path
) -> list[dict]:
    """Null-workload (/bin/true) summary: saccade's fixed startup cost in absolute terms."""
    fieldnames = [
        "q_schedule_ns",
        "q_sample_ns",
        "sink",
        "bare_median_s",
        "saccade_median_s",
        "startup_overhead_s",
        "startup_overhead_ms",
        "iqr_s",
        "reps",
    ]
    rows: list[dict] = []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sink in SINKS:
            for q_sched in Q_SCHEDULE_NS:
                for q_samp in Q_SAMPLE_NS:
                    times = by_config[(sink, q_sched, q_samp)]
                    saccade_median = float(np.median(times))
                    iqr = float(np.percentile(times, 75) - np.percentile(times, 25))
                    startup = saccade_median - baseline_median
                    row = {
                        "q_schedule_ns": q_sched,
                        "q_sample_ns": q_samp,
                        "sink": sink,
                        "bare_median_s": round(baseline_median, 6),
                        "saccade_median_s": round(saccade_median, 6),
                        "startup_overhead_s": round(startup, 6),
                        "startup_overhead_ms": round(startup * 1000, 3),
                        "iqr_s": round(iqr, 6),
                        "reps": reps,
                    }
                    writer.writerow(row)
                    rows.append(row)
    return rows


def print_overhead(rows: list[dict]) -> None:
    header = f"{'q_schedule_ns':>15}  {'q_sample_ns':>12}  {'sink':>9}  {'baseline_s':>10}  {'saccade_s':>10}  {'overhead%':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['q_schedule_ns']:>15}  {r['q_sample_ns']:>12}  {r['sink']:>9}  "
            f"{r['baseline_median_s']:>10.4f}  {r['saccade_median_s']:>10.4f}  "
            f"{r['overhead_fraction'] * 100:>9.1f}%"
        )


def print_startup(rows: list[dict], baseline_median: float) -> None:
    print()
    print(f"saccade startup cost (/bin/true; bare baseline = {baseline_median * 1000:.1f} ms)")
    print(f"  {'sink':>9}  {'startup (ms)':>14}")
    print("  " + "-" * 26)
    # q_schedule/q_sample should barely matter for an instant workload, so aggregate by sink.
    for sink in SINKS:
        vals = [r["startup_overhead_ms"] for r in rows if r["sink"] == sink]
        print(f"  {sink:>9}  {float(np.median(vals)):>14.1f}")
    print(f"  {'overall':>9}  {float(np.median([r['startup_overhead_ms'] for r in rows])):>14.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure saccade runtime overhead across a parameter grid "
        "with shuffled, interleaved, repeated runs, plus a /bin/true startup-cost workload."
    )
    parser.add_argument(
        "--saccade",
        type=Path,
        default=Path("../target/release/saccade"),
        help="Path to saccade binary (default: ../target/release/saccade)",
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
        default=5,
        help="Number of global warmup runs at the start, discarded (default: 5)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Number of repeats of the full treatment set; = reps per config (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the run-order shuffle (default: 0)",
    )
    parser.add_argument(
        "--null-workload",
        type=str,
        default="/bin/true",
        help="Instant-exit workload for measuring saccade startup cost (default: /bin/true)",
    )
    parser.add_argument(
        "--no-null-workload",
        action="store_true",
        help="Skip the /bin/true startup-cost workload",
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

    # Map workload tag -> command. "target" is the real workload; "null" is /bin/true.
    targets: dict[str, list[str]] = {"target": target}
    if not args.no_null_workload:
        if not Path(args.null_workload).exists():
            parser.error(
                f"null workload not found: {args.null_workload} (use --null-workload to override)"
            )
        targets["null"] = [args.null_workload]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.results_dir / "q1_overhead_raw.csv"
    overhead_csv = args.results_dir / "q1_overhead.csv"
    startup_csv = args.results_dir / "q1_startup.csv"

    units = build_units(list(targets.keys()))

    # Build the full shuffled schedule: every (round, unit) pair as one run.
    schedule = [(rnd, unit) for rnd in range(args.rounds) for unit in units]
    random.Random(args.seed).shuffle(schedule)

    print(
        f"Plan: {len(units)} treatments x {args.rounds} rounds = {len(schedule)} measured runs "
        f"(+{args.warmup} warmup), shuffled with seed {args.seed}.\n"
        f"Workloads: {', '.join(targets)}."
    )

    # Global warmup, discarded.
    for _ in range(args.warmup):
        _elapsed, _ = run_timed(target, check=False)

    raw_fieldnames = [
        "order_index",
        "round",
        "workload",
        "sink",
        "q_schedule_ns",
        "q_sample_ns",
        "is_baseline",
        "elapsed_s",
        "samples_emitted",
    ]
    raw_rows: list[dict] = []

    with open(raw_csv, "w", newline="") as rawfile:
        raw_writer = csv.DictWriter(rawfile, fieldnames=raw_fieldnames)
        raw_writer.writeheader()
        rawfile.flush()

        bar = tqdm(schedule, desc="runs", unit="run")
        for order_index, (rnd, unit) in enumerate(bar):
            bar.set_postfix_str(
                f"round={rnd} wl={unit['workload']} sink={unit['sink']} "
                f"q_sched={unit['q_sched']} q_samp={unit['q_samp']}"
            )

            if unit["kind"] == "baseline":
                cmd = targets[unit["workload"]]
            else:
                cmd = build_saccade_cmd(
                    args.saccade,
                    args.library,
                    unit["q_sched"],
                    unit["q_samp"],
                    unit["sink"],
                    args.results_dir,
                    targets[unit["workload"]],
                )

            elapsed, samples_emitted = run_timed(cmd)

            raw_writer.writerow(
                {
                    "order_index": order_index,
                    "round": rnd,
                    "workload": unit["workload"],
                    "sink": unit["sink"],
                    "q_schedule_ns": unit["q_sched"] if unit["q_sched"] is not None else "",
                    "q_sample_ns": unit["q_samp"] if unit["q_samp"] is not None else "",
                    "is_baseline": int(unit["kind"] == "baseline"),
                    "elapsed_s": round(elapsed, 6),
                    "samples_emitted": samples_emitted if samples_emitted is not None else "",
                }
            )
            rawfile.flush()
            raw_rows.append(
                {
                    "workload": unit["workload"],
                    "is_baseline": unit["kind"] == "baseline",
                    "sink": unit["sink"],
                    "q_schedule_ns": unit["q_sched"],
                    "q_sample_ns": unit["q_samp"],
                    "elapsed_s": elapsed,
                    "samples_emitted": samples_emitted,
                }
            )

    # Real workload -> overhead summary (fraction of baseline).
    base_target, cfg_target, samp_target = aggregate(raw_rows, "target")
    overhead_rows = write_overhead_csv(
        base_target, cfg_target, args.rounds, overhead_csv, samples_median=samp_target
    )
    print_overhead(overhead_rows)
    print(f"\nRaw per-run results:  {raw_csv}")
    print(f"Overhead summary:     {overhead_csv}")

    # Null workload -> saccade startup cost (absolute).
    if "null" in targets:
        base_null, cfg_null, _samp_null = aggregate(raw_rows, "null")
        startup_rows = write_startup_csv(base_null, cfg_null, args.rounds, startup_csv)
        print_startup(startup_rows, base_null)
        print(f"Startup summary:      {startup_csv}")


if __name__ == "__main__":
    main()
