#!/usr/bin/env python3
"""Measure the cost of saccade's counter-rotation path across different scheduler quantum sizes.

For each quantum size, runs the target workload under saccade with RUST_LOG=debug,
parses slot_swap log lines from stderr, and aggregates swap latency statistics.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

Q_SCHEDULE_VALUES_NS = [
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
    100_000_000,
]

SLOT_SWAP_PATTERN = re.compile(r"slot_swap swap_ns=(\d+) quantum_ns=(\d+)")


def run_once(
    saccade: Path,
    q_schedule_ns: int,
    target: list[str],
    library: Path | None,
) -> list[tuple[int, int]]:
    """Run saccade once and return list of (swap_ns, quantum_ns) pairs."""
    cmd = [
        str(saccade),
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
        env={**os.environ, "RUST_LOG": "debug"},
    )

    pairs: list[tuple[int, int]] = []
    for line in result.stderr.splitlines():
        m = SLOT_SWAP_PATTERN.search(line)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def aggregate(all_pairs: list[tuple[int, int]]) -> dict:
    """Aggregate (swap_ns, quantum_ns) pairs into summary statistics."""
    if not all_pairs:
        return {
            "median_swap_ns": float("nan"),
            "iqr_swap_ns": float("nan"),
            "median_swap_fraction": float("nan"),
            "swap_count": 0,
        }

    swap_ns_arr = np.array([p[0] for p in all_pairs], dtype=np.float64)
    quantum_ns_arr = np.array([p[1] for p in all_pairs], dtype=np.float64)
    fractions = swap_ns_arr / quantum_ns_arr

    return {
        "median_swap_ns": float(np.median(swap_ns_arr)),
        "iqr_swap_ns": float(np.percentile(swap_ns_arr, 75) - np.percentile(swap_ns_arr, 25)),
        "median_swap_fraction": float(np.median(fractions)),
        "swap_count": len(all_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure saccade counter-rotation latency across scheduler quantum sizes."
    )
    parser.add_argument(
        "--saccade",
        type=Path,
        default=Path("./target/release/saccade"),
        help="Path to saccade binary (default: ./target/release/saccade)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Event library JSON file (optional)",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        required=True,
        metavar="ARG",
        help="Workload binary and arguments",
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
        help="Number of warmup runs to discard per quantum value (default: 3)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Number of measurement repetitions per quantum value (default: 10)",
    )
    args = parser.parse_args()

    args.saccade = args.saccade.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.library is not None:
        args.library = args.library.resolve()

    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(Q_SCHEDULE_VALUES_NS) * (args.warmup + args.reps)
    rows: list[dict] = []

    with tqdm(total=total_runs, desc="runs", unit="run") as bar:
        for q_ns in Q_SCHEDULE_VALUES_NS:
            q_ms = q_ns / 1_000_000
            bar.set_postfix_str(f"q={q_ms:.0f}ms warmup")

            # Warmup runs — discard results
            for _ in range(args.warmup):
                run_once(args.saccade, q_ns, args.target, args.library)
                bar.update(1)

            bar.set_postfix_str(f"q={q_ms:.0f}ms measure")

            # Measurement runs — collect all (swap_ns, quantum_ns) pairs
            all_pairs: list[tuple[int, int]] = []
            for _ in range(args.reps):
                pairs = run_once(args.saccade, q_ns, args.target, args.library)
                all_pairs.extend(pairs)
                bar.update(1)

            stats = aggregate(all_pairs)
            rows.append(
                {
                    "q_schedule_ns": q_ns,
                    "median_swap_ns": stats["median_swap_ns"],
                    "iqr_swap_ns": stats["iqr_swap_ns"],
                    "median_swap_fraction": stats["median_swap_fraction"],
                    "swap_count": stats["swap_count"],
                    "reps": args.reps,
                }
            )

            if stats["swap_count"] == 0:
                tqdm.write(
                    f"WARNING: q={q_ns}ns — no slot_swap events observed "
                    f"(workload may be shorter than one quantum)"
                )

    # Write CSV
    csv_path = args.results_dir / "q4_swap_latency.csv"
    fieldnames = [
        "q_schedule_ns",
        "median_swap_ns",
        "iqr_swap_ns",
        "median_swap_fraction",
        "swap_count",
        "reps",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to: {csv_path}\n")

    # Summary table
    col_w = [14, 16, 14, 22, 12]
    header = (
        f"{'q_schedule_ns':>{col_w[0]}}"
        f"{'median_swap_ns':>{col_w[1]}}"
        f"{'iqr_swap_ns':>{col_w[2]}}"
        f"{'median_swap_frac':>{col_w[3]}}"
        f"{'swap_count':>{col_w[4]}}"
    )
    print(header)
    print("-" * sum(col_w))

    for row in rows:
        frac = row["median_swap_fraction"]
        frac_str = f"{frac:.4f}" if not (frac != frac) else "nan"  # isnan check
        warn = " *** >5%" if (frac == frac and frac > 0.05) else ""

        med_str = f"{row['median_swap_ns']:.1f}" if not (row['median_swap_ns'] != row['median_swap_ns']) else "nan"
        iqr_str = f"{row['iqr_swap_ns']:.1f}" if not (row['iqr_swap_ns'] != row['iqr_swap_ns']) else "nan"

        print(
            f"{row['q_schedule_ns']:>{col_w[0]}}"
            f"{med_str:>{col_w[1]}}"
            f"{iqr_str:>{col_w[2]}}"
            f"{frac_str:>{col_w[3]}}"
            f"{row['swap_count']:>{col_w[4]}}"
            f"{warn}"
        )

    warnings = [r for r in rows if r["median_swap_fraction"] == r["median_swap_fraction"] and r["median_swap_fraction"] > 0.05]
    if warnings:
        print(
            f"\nWARNING: {len(warnings)} quantum size(s) have median swap fraction > 5% "
            f"— counter-rotation overhead may be significant at those quanta."
        )

    print()


if __name__ == "__main__":
    main()
