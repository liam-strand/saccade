#!/usr/bin/env python3
"""Summarize best (scheduler, estimator) combinations across all workloads.

Loads SPEC and NPB grid CSVs produced by q2_accuracy.py, then for each
(scheduler, estimator) pair computes the mean and stddev of median_nrmse
across workloads, separately for SPEC, NPB, and combined.

Usage:
    python analysis/cross_workload.py \\
        --spec-grid results/<spec_ts>/q2_scheduler_estimator.csv \\
        --npb-grid  results/<npb_ts>/q2_scheduler_estimator.csv \\
        --out-dir   results/q6_summary/
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def read_grid(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("median_nrmse", "") != "":
                row["median_nrmse"] = float(row["median_nrmse"])
                rows.append(row)
    return rows


def combo_stats(rows: list[dict]) -> list[dict]:
    """Group by (scheduler, estimator), compute mean/stddev of median_nrmse."""
    from collections import defaultdict
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["scheduler"], r["estimator"])
        groups[key].append(float(r["median_nrmse"]))

    result = []
    for (scheduler, estimator), vals in sorted(groups.items()):
        arr = np.array(vals)
        result.append({
            "scheduler": scheduler,
            "estimator": estimator,
            "mean_median_nrmse": float(np.mean(arr)),
            "stddev_median_nrmse": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n_workloads": len(arr),
        })
    result.sort(key=lambda r: r["mean_median_nrmse"])
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-workload best-config summary from q2 grid CSVs."
    )
    p.add_argument(
        "--spec-grid",
        type=Path,
        required=True,
        help="q2_scheduler_estimator.csv from the SPEC run",
    )
    p.add_argument(
        "--npb-grid",
        type=Path,
        required=True,
        help="q2_scheduler_estimator.csv from the NPB run",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./results/q6_summary"),
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    for p, label in [(args.spec_grid, "--spec-grid"), (args.npb_grid, "--npb-grid")]:
        if not p.exists():
            parser.error(f"{label} not found: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    spec_rows = read_grid(args.spec_grid)
    npb_rows  = read_grid(args.npb_grid)
    all_rows  = spec_rows + npb_rows

    per_combo_csv = args.out_dir / "q6_per_combo_mean.csv"
    fieldnames = [
        "kind", "scheduler", "estimator",
        "mean_median_nrmse", "stddev_median_nrmse", "n_workloads",
    ]

    output_rows: list[dict] = []
    for kind, rows in [("spec", spec_rows), ("npb", npb_rows), ("combined", all_rows)]:
        if not rows:
            continue
        stats = combo_stats(rows)
        for s in stats:
            output_rows.append({"kind": kind, **s})

    with per_combo_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Results written to: {args.out_dir}", file=sys.stderr)
    print(f"  {per_combo_csv.name}", file=sys.stderr)

    # Print top-3 per kind
    for kind in ("spec", "npb", "combined"):
        kind_rows = [r for r in output_rows if r["kind"] == kind][:3]
        if not kind_rows:
            continue
        print(f"\nTop-3 combos ({kind}):", file=sys.stderr)
        for r in kind_rows:
            print(
                f"  {r['scheduler']:28s} + {r['estimator']:10s} "
                f"mean_nRMSE={r['mean_median_nrmse']:.4f} "
                f"(n={r['n_workloads']})",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
