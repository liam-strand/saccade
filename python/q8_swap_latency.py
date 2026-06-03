#!/usr/bin/env python3
"""Measure the cost of saccade's counter-rotation ("slot swap") path across scheduler quanta.

Each `apply_schedule` call now logs a timing *breakdown* (profiler.rs), split into:
  * quiesce_ns  -- the stop-the-world spin-wait in `stop_counters`, summed across
                   changed slots. Dominated by waiting for every CPU to acknowledge
                   the quiesce, NOT by counter work.
  * reconfig_ns -- the actual perf_event_open + BPF-map update, summed across changed
                   slots. The true counter-reconfiguration cost.
  * slots_changed -- how many counter slots actually changed (0 => a no-op swap whose
                   cost is ~0 and which would otherwise pollute pooled medians).
The outer swap_ns (wall-clock total of the whole call) and quantum_ns (the *realized*
collection-window duration) are retained for backward compatibility.

Design (mirrors q1_overhead.py): every (round, quantum) pair is one measured saccade
run; the full schedule is shuffled with a seeded RNG so thermal/load drift over the
session is decorrelated from quantum rather than misattributed to whichever quanta run
late (the fixed-ascending-order confound of the previous version). Every individual swap
is persisted to a raw CSV (with execution order_index and a wall-clock stamp) so drift
can be tested directly downstream.

The summary fraction is reported against the *requested* quantum (q_schedule), not the
realized loop period -- the realized quantum already includes the swap, making any
swap/realized ratio semi-circular.
"""

import argparse
import csv
import os
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Scheduler quanta to sweep (nanoseconds).
Q_SCHEDULE_VALUES_NS = [
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
    100_000_000,
]

# Fields parsed out of each `slot_swap` debug line.
SWAP_FIELDS = ["swap_ns", "quiesce_ns", "reconfig_ns", "slots_changed", "quantum_ns"]
# Generic key=value scanner; robust to field ordering in the tracing output.
KV_PATTERN = re.compile(r"(\w+)=(\d+)")


def parse_swaps(stderr: str) -> list[dict]:
    """Extract one dict per `slot_swap` line from saccade's stderr.

    Fails loudly if a swap line parses a `swap_ns` value but is missing the
    split-timer fields -- that means the saccade binary predates this
    instrumentation, which would otherwise silently yield an empty result set.
    """
    swaps: list[dict] = []
    for line in stderr.splitlines():
        if "slot_swap" not in line:
            continue
        kv = {k: int(v) for k, v in KV_PATTERN.findall(line)}
        if "swap_ns" not in kv:
            # Not a parseable swap line (truncated/garbled); skip quietly.
            continue
        missing = [f for f in SWAP_FIELDS if f not in kv]
        if missing:
            raise RuntimeError(
                f"slot_swap line is missing fields {missing} -- the saccade binary "
                f"predates the split-timer instrumentation. Rebuild it "
                f"(cargo build --release) and point --saccade at the fresh binary.\n"
                f"  offending line: {line.strip()}"
            )
        swaps.append({f: kv[f] for f in SWAP_FIELDS})
    return swaps


def run_once(
    saccade: Path,
    q_schedule_ns: int,
    target: list[str],
    library: Path | None,
) -> list[dict]:
    """Run saccade once at a given quantum; return per-swap field dicts from stderr."""
    cmd = [
        str(saccade),
        "--verbose",
        "run",
        "--q-schedule",
        str(q_schedule_ns),
        "--trace",
        "/dev/null",
    ]
    if library is not None:
        cmd += ["--library", str(library)]
    cmd += ["--"] + target

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "NO_COLOR": "1"},
    )
    return parse_swaps(result.stderr)


def aggregate(raw_rows: list[dict], rounds: int) -> list[dict]:
    """Per-quantum summary. Stats over real swaps (slots_changed > 0); reframes the
    fraction against the requested quantum and tracks the no-op share separately."""
    rows: list[dict] = []
    for q in Q_SCHEDULE_VALUES_NS:
        q_rows = [r for r in raw_rows if r["q_schedule_ns"] == q]
        if not q_rows:
            continue
        real = [r for r in q_rows if r["slots_changed"] > 0]
        noop_fraction = 1.0 - (len(real) / len(q_rows)) if q_rows else float("nan")

        if real:
            swap = np.array([r["swap_ns"] for r in real], dtype=np.float64)
            quiesce = np.array([r["quiesce_ns"] for r in real], dtype=np.float64)
            reconfig = np.array([r["reconfig_ns"] for r in real], dtype=np.float64)
            median_swap = float(np.median(swap))
            iqr_swap = float(np.percentile(swap, 75) - np.percentile(swap, 25))
            median_quiesce = float(np.median(quiesce))
            median_reconfig = float(np.median(reconfig))
            requested_fraction = median_swap / q
        else:
            median_swap = iqr_swap = median_quiesce = median_reconfig = float("nan")
            requested_fraction = float("nan")

        rows.append(
            {
                "q_schedule_ns": q,
                "median_swap_ns": round(median_swap, 1),
                "iqr_swap_ns": round(iqr_swap, 1),
                "median_quiesce_ns": round(median_quiesce, 1),
                "median_reconfig_ns": round(median_reconfig, 1),
                "requested_fraction": round(requested_fraction, 6),
                "noop_fraction": round(noop_fraction, 4),
                "real_swap_count": len(real),
                "rounds": rounds,
            }
        )
    return rows


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'q_sched_ms':>11}  {'swap_ms':>9}  {'quiesce_ms':>11}  "
        f"{'reconfig_us':>12}  {'req_frac':>9}  {'noop%':>6}  {'n':>6}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        swap_ms = r["median_swap_ns"] / 1e6
        quiesce_ms = r["median_quiesce_ns"] / 1e6
        reconfig_us = r["median_reconfig_ns"] / 1e3
        print(
            f"{r['q_schedule_ns'] / 1e6:>11.1f}  {swap_ms:>9.2f}  {quiesce_ms:>11.2f}  "
            f"{reconfig_us:>12.1f}  {r['requested_fraction']:>9.3f}  "
            f"{r['noop_fraction'] * 100:>5.1f}%  {r['real_swap_count']:>6}"
        )
    print(
        "\nreconfig_us is the true counter-reconfiguration cost; quiesce_ms is the "
        "stop-the-world spin-wait.\nIf reconfig is small and flat while quiesce dominates, "
        "the swap path's cost is the world-stop, not counter work."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure saccade counter-rotation latency across scheduler quanta, "
        "split into quiesce vs reconfiguration, with shuffled repeated rounds."
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
        "--rounds",
        type=int,
        default=10,
        help="Number of repeats of the full quantum set; = reps per quantum (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the run-order shuffle (default: 0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of global warmup saccade runs at the start, discarded (default: 5)",
    )
    parser.add_argument(
        "--quanta",
        type=int,
        nargs="+",
        default=None,
        help="Override the default quantum values (nanoseconds, space-separated).",
    )
    parser.add_argument(
        "--target",
        nargs=argparse.REMAINDER,
        required=True,
        help="Workload binary and arguments (e.g. --target /path/to/bench)",
    )
    args = parser.parse_args()

    global Q_SCHEDULE_VALUES_NS
    if args.quanta:
        Q_SCHEDULE_VALUES_NS = list(args.quanta)

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
    raw_csv = args.results_dir / "q8_swap_latency_raw.csv"
    summary_csv = args.results_dir / "q8_swap_latency.csv"

    # Full shuffled schedule: every (round, quantum) pair as one run.
    schedule = [(rnd, q) for rnd in range(args.rounds) for q in Q_SCHEDULE_VALUES_NS]
    random.Random(args.seed).shuffle(schedule)

    print(
        f"Plan: {len(Q_SCHEDULE_VALUES_NS)} quanta x {args.rounds} rounds = "
        f"{len(schedule)} measured runs (+{args.warmup} warmup), shuffled with seed {args.seed}."
    )

    # Global warmup at a representative (median) quantum, discarded.
    warmup_q = Q_SCHEDULE_VALUES_NS[len(Q_SCHEDULE_VALUES_NS) // 2]
    for _ in range(args.warmup):
        run_once(args.saccade, warmup_q, target, args.library)

    raw_fieldnames = [
        "order_index",
        "round",
        "q_schedule_ns",
        "wall_clock_s",
        "swap_ns",
        "quiesce_ns",
        "reconfig_ns",
        "slots_changed",
        "quantum_ns",
    ]
    raw_rows: list[dict] = []

    with open(raw_csv, "w", newline="") as rawfile:
        raw_writer = csv.DictWriter(rawfile, fieldnames=raw_fieldnames)
        raw_writer.writeheader()
        rawfile.flush()

        bar = tqdm(schedule, desc="runs", unit="run")
        for order_index, (rnd, q) in enumerate(bar):
            bar.set_postfix_str(f"round={rnd} q={q / 1e6:.0f}ms")

            wall_clock_s = time.time()
            swaps = run_once(args.saccade, q, target, args.library)

            if not swaps:
                tqdm.write(
                    f"WARNING: q={q}ns round={rnd} produced no slot_swap events "
                    f"(workload may be shorter than one quantum)"
                )

            for s in swaps:
                row = {
                    "order_index": order_index,
                    "round": rnd,
                    "q_schedule_ns": q,
                    "wall_clock_s": round(wall_clock_s, 3),
                    **s,
                }
                raw_writer.writerow(row)
                raw_rows.append(row)
            rawfile.flush()

    summary_rows = aggregate(raw_rows, args.rounds)
    fieldnames = [
        "q_schedule_ns",
        "median_swap_ns",
        "iqr_swap_ns",
        "median_quiesce_ns",
        "median_reconfig_ns",
        "requested_fraction",
        "noop_fraction",
        "real_swap_count",
        "rounds",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print_summary(summary_rows)
    print(f"\nRaw per-swap results:  {raw_csv}")
    print(f"Summary:               {summary_csv}")


if __name__ == "__main__":
    main()
