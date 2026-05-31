#!/usr/bin/env python3
"""Extract 1D slices from a q2_accuracy.py grid CSV.

Produces three filtered CSVs from the full scheduler × estimator grid:

  slice_rr_all_estimators.csv      -- round-robin scheduler, all estimators
  slice_maxuncert_all_estimators.csv -- max-uncertainty scheduler, all estimators
  slice_kalman_all_schedulers.csv  -- kalman estimator, all schedulers

Usage:
    python analysis/slice_grid.py \\
        --grid-csv results/<ts>/q2_scheduler_estimator.csv \\
        --out-dir  results/<ts>/slices/
"""

import argparse
import csv
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path, fieldnames: list[str] | None = None) -> None:
    if not rows:
        print(f"  Warning: no rows for {path.name}", file=sys.stderr)
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract 1D slices from a q2 scheduler × estimator grid CSV."
    )
    p.add_argument(
        "--grid-csv",
        type=Path,
        required=True,
        help="Path to q2_scheduler_estimator.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as --grid-csv)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.grid_csv.exists():
        parser.error(f"Grid CSV not found: {args.grid_csv}")

    out_dir = args.out_dir if args.out_dir is not None else args.grid_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.grid_csv)
    all_fields = list(rows[0].keys()) if rows else []

    slices = [
        ("slice_rr_all_estimators.csv",      lambda r: r["scheduler"] == "round-robin"),
        ("slice_maxuncert_all_estimators.csv", lambda r: r["scheduler"] == "max-uncertainty"),
        ("slice_kalman_all_schedulers.csv",   lambda r: r["estimator"] == "kalman"),
    ]

    for filename, predicate in slices:
        filtered = [r for r in rows if predicate(r)]
        out_path = out_dir / filename
        write_csv(filtered, out_path, all_fields)
        print(f"  {out_path}  ({len(filtered)} rows)", file=sys.stderr)

    print(f"\nSlices written to: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
