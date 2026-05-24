#!/usr/bin/env python3
"""Measure run-to-run sweep variability to establish the noise floor for accuracy experiments.

Runs saccade sweep twice on the same target (after optional warmup runs) and evaluates
the two traces against each other to quantify intrinsic measurement noise.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def run_sweep(
    saccade: Path,
    target: list[str],
    trace: Path,
    library: Path | None,
) -> None:
    cmd = [str(saccade), "sweep", "--quiet", "--trace", str(trace)]
    if library:
        cmd += ["--library", str(library)]
    cmd += ["--"] + target
    subprocess.run(cmd, check=True)


def run_evaluate(
    saccade: Path,
    ground_truth: Path,
    estimated: Path,
) -> dict:
    cmd = [
        str(saccade),
        "evaluate",
        "--ground-truth",
        str(ground_truth),
        "--estimated",
        str(estimated),
        "--json",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure run-to-run sweep variability (noise floor) for saccade accuracy experiments."
    )
    parser.add_argument(
        "--saccade",
        type=Path,
        default=Path("./target/release/saccade"),
        help="saccade binary (default: ./target/release/saccade)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Event library JSON (optional)",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        required=True,
        metavar="ARG",
        help="Workload binary and arguments (e.g. ep.A.x)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory for output files (default: ./results)",
    )
    args = parser.parse_args()

    args.saccade = args.saccade.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.library:
        args.library = args.library.resolve()

    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Two measured sweeps.
    gt1 = args.results_dir / "gt1.perfetto"
    gt2 = args.results_dir / "gt2.perfetto"

    print("Running measured sweep 1/2...")
    run_sweep(args.saccade, args.target, gt1, args.library)

    print("Running measured sweep 2/2...")
    run_sweep(args.saccade, args.target, gt2, args.library)

    # Evaluate the two sweeps against each other.
    print("Evaluating sweep 1 vs sweep 2...")
    eval_result = run_evaluate(args.saccade, gt1, gt2)

    # Extract per-event nRMSE values (skip nulls where mean GT rate is zero).
    per_event: dict[str, float] = {}
    for entry in eval_result.get("per_event", []):
        if entry["nrmse"] is not None:
            key = f"{entry['event']} tid={entry['tid']}"
            per_event[key] = entry["nrmse"]

    nrmse_values = list(per_event.values())
    median_nrmse: float | None = float(np.median(nrmse_values)) if nrmse_values else None

    target_str = " ".join(args.target)
    output = {
        "median_nrmse": median_nrmse,
        "per_event": per_event,
        "target": target_str,
    }

    noise_floor_path = args.results_dir / "noise_floor.json"
    with noise_floor_path.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print()
    print(f"Target:        {target_str}")
    print(f"Events:        {len(per_event)}")
    if median_nrmse is not None:
        print(f"Median nRMSE:  {median_nrmse:.4f}")
    else:
        print("Median nRMSE:  N/A (no valid events)")
    print(f"Output:        {noise_floor_path}")

    if median_nrmse is not None and median_nrmse > 0.10:
        print(
            "WARNING: noise floor > 0.10, consider using workload.rs deterministic targets."
        )


if __name__ == "__main__":
    main()
