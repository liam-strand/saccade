#!/usr/bin/env python3
"""Compare saccade against kernel-native multiplexing (perf stat -I).

Q3: Does saccade produce better time-series fidelity than perf stat -I?

Runs the workload under:
  1. perf stat -I 100  (kernel-native multiplexing)
  2. saccade run --scheduler distribution --estimator propagate
  3. saccade run --scheduler round_robin --estimator propagate

Each approach is evaluated against a pre-existing sweep ground-truth trace
using ``saccade evaluate --json``, which reports mean_nrmse (lower is better).

Noise-floor JSON (from q6_noise_floor.py) expected schema:
  {"noise_floor": <float>}
where the float is the nRMSE measurement noise threshold.  Differences below
this threshold are labelled not significant.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _import_convert():
    """Lazily import perf_to_perfetto.convert so --help works before the module exists."""
    # perf_to_perfetto is written by a concurrent subagent and lives alongside this script.
    sys.path.insert(0, str(Path(__file__).parent))
    from perf_to_perfetto import convert  # noqa: PLC0415
    return convert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_events_from_library(library: Path) -> list[str]:
    """Return all event name strings from the saccade event library JSON."""
    with library.open() as f:
        data = json.load(f)
    return [e["name"] for e in data["events"]]


def perf_event_string(events: list[str]) -> str:
    """Join event names into a comma-separated string for perf -e."""
    return ",".join(events)


def run_evaluate(saccade: Path, gt_trace: Path, estimated_trace: Path) -> float | None:
    """Run ``saccade evaluate --json`` and return mean_nrmse (or None on failure)."""
    result = subprocess.run(
        [
            str(saccade),
            "evaluate",
            "--ground-truth",
            str(gt_trace),
            "--estimated",
            str(estimated_trace),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"  evaluate failed (rc={result.returncode}):\n{result.stderr.strip()}",
            file=sys.stderr,
        )
        return None
    try:
        data = json.loads(result.stdout)
        return data.get("mean_nrmse")
    except json.JSONDecodeError as exc:
        print(f"  evaluate output not valid JSON: {exc}", file=sys.stderr)
        return None


def run_perf_stat_rep(
    target: list[str],
    event_str: str,
    tmp_csv: Path,
    tmp_perfetto: Path,
    saccade: Path,
    gt_trace: Path,
) -> float | None:
    """Run one perf-stat rep and return mean_nrmse, or None on failure."""
    perf_cmd = [
        "perf",
        "stat",
        "-I", "100",
        "-x,",
        "-e", event_str,
        "--",
    ] + target

    result = subprocess.run(
        perf_cmd,
        capture_output=True,
        text=True,
    )
    # perf stat writes CSV to stderr (not stdout) when -x is used
    csv_data = result.stderr
    if result.returncode != 0 and not csv_data.strip():
        print(
            f"  perf stat failed (rc={result.returncode}):\n{result.stderr.strip()}",
            file=sys.stderr,
        )
        return None

    tmp_csv.write_text(csv_data)

    try:
        convert = _import_convert()
        convert(str(tmp_csv), str(tmp_perfetto))
    except Exception as exc:  # noqa: BLE001
        print(f"  perf_to_perfetto.convert failed: {exc}", file=sys.stderr)
        return None

    return run_evaluate(saccade, gt_trace, tmp_perfetto)


def run_saccade_rep(
    saccade: Path,
    library: Path,
    scheduler: str,
    estimator: str,
    target: list[str],
    tmp_perfetto: Path,
    gt_trace: Path,
) -> float | None:
    """Run one saccade-run rep and return mean_nrmse, or None on failure."""
    cmd = [
        str(saccade),
        "run",
        "--library", str(library),
        "--scheduler", scheduler,
        "--estimator", estimator,
        "--trace", str(tmp_perfetto),
        "--",
    ] + target

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"  saccade run failed (rc={result.returncode}):\n{result.stderr.strip()}",
            file=sys.stderr,
        )
        return None

    return run_evaluate(saccade, gt_trace, tmp_perfetto)


def collect_reps(
    label: str,
    run_fn,          # callable(rep_index: int) -> float | None
    warmup: int,
    reps: int,
) -> list[float]:
    """Run warmup+reps iterations; discard warmup results; return rep scores."""
    total = warmup + reps
    scores: list[float] = []
    for i in range(total):
        phase = "warmup" if i < warmup else f"rep {i - warmup + 1}/{reps}"
        print(f"  [{label}] {phase} ...", flush=True)
        val = run_fn(i)
        if i >= warmup:
            if val is None:
                print(f"  [{label}] rep {i - warmup + 1} returned None, skipping.", file=sys.stderr)
            else:
                scores.append(val)
    return scores


def median_iqr(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return float(np.median(arr)), q3 - q1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q3: compare saccade vs perf stat -I time-series fidelity."
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
        required=True,
        help="Event library JSON produced by `saccade generate`",
    )
    parser.add_argument(
        "--gt-trace",
        type=Path,
        required=True,
        help="Pre-existing sweep ground-truth Perfetto trace",
    )
    parser.add_argument(
        "--target",
        nargs=argparse.REMAINDER,
        required=True,
        help="Workload binary and arguments (everything after --target)",
    )
    parser.add_argument(
        "--noise-floor-json",
        type=Path,
        default=None,
        help='Optional JSON from q6_noise_floor.py: {"noise_floor": <float>}',
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory for output CSV (default: ./results)",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Comma-separated perf event names; if omitted, read from --library",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Number of timed repetitions per approach (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs discarded from statistics (default: 3)",
    )
    args = parser.parse_args()

    # Resolve paths early so relative paths work regardless of cwd.
    args.saccade = args.saccade.resolve()
    args.library = args.library.resolve()
    args.gt_trace = args.gt_trace.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.noise_floor_json:
        args.noise_floor_json = args.noise_floor_json.resolve()

    # Strip a leading "--" from --target if the user wrote: --target -- <binary>
    target = args.target
    if target and target[0] == "--":
        target = target[1:]

    if not target:
        parser.error("--target requires at least a binary path.")

    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")
    if not args.library.exists():
        parser.error(f"library not found: {args.library}")
    if not args.gt_trace.exists():
        parser.error(f"GT trace not found: {args.gt_trace}")

    # Determine event set.
    if args.events:
        events = [e.strip() for e in args.events.split(",") if e.strip()]
    else:
        events = load_events_from_library(args.library)
    event_str = perf_event_string(events)
    print(f"Event set: {len(events)} event(s)")

    # Load noise floor if provided.
    noise_floor: float | None = None
    if args.noise_floor_json:
        with args.noise_floor_json.open() as f:
            nf_data = json.load(f)
        noise_floor = float(nf_data["noise_floor"])
        print(f"Noise floor: {noise_floor:.4f} nRMSE")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="q3_") as tmp_dir:
        tmp = Path(tmp_dir)

        # ------------------------------------------------------------------
        # 1. perf stat
        # ------------------------------------------------------------------
        print("\n=== perf stat -I 100 ===")
        perf_csv = tmp / "perf_stat.csv"
        perf_perfetto = tmp / "perf_stat.perfetto"

        def run_perf(i: int) -> float | None:
            # Use per-rep filenames to avoid clobbering across reps.
            csv_path = tmp / f"perf_rep{i}.csv"
            pft_path = tmp / f"perf_rep{i}.perfetto"
            return run_perf_stat_rep(
                target, event_str, csv_path, pft_path, args.saccade, args.gt_trace
            )

        perf_scores = collect_reps("perf stat", run_perf, args.warmup, args.reps)

        if perf_scores:
            med, iqr = median_iqr(perf_scores)
            sig = (noise_floor is None) or (med > noise_floor)
            results.append({
                "approach": "perf_stat",
                "scheduler": "",
                "estimator": "",
                "median_nrmse": med,
                "iqr": iqr,
                "reps": len(perf_scores),
                "significant": sig,
            })
            print(f"  median nRMSE={med:.4f}  IQR={iqr:.4f}  n={len(perf_scores)}")
        else:
            print("  No valid reps collected for perf stat.", file=sys.stderr)

        # ------------------------------------------------------------------
        # 2. saccade — distribution + propagate
        # ------------------------------------------------------------------
        print("\n=== saccade --scheduler distribution --estimator propagate ===")

        def run_dist(i: int) -> float | None:
            pft_path = tmp / f"saccade_dist_rep{i}.perfetto"
            return run_saccade_rep(
                args.saccade, args.library,
                "distribution", "propagate",
                target, pft_path, args.gt_trace,
            )

        dist_scores = collect_reps("saccade/distribution", run_dist, args.warmup, args.reps)

        if dist_scores:
            med, iqr = median_iqr(dist_scores)
            sig = (noise_floor is None) or (med > noise_floor)
            results.append({
                "approach": "saccade",
                "scheduler": "distribution",
                "estimator": "propagate",
                "median_nrmse": med,
                "iqr": iqr,
                "reps": len(dist_scores),
                "significant": sig,
            })
            print(f"  median nRMSE={med:.4f}  IQR={iqr:.4f}  n={len(dist_scores)}")
        else:
            print("  No valid reps collected for saccade/distribution.", file=sys.stderr)

        # ------------------------------------------------------------------
        # 3. saccade — round_robin + propagate
        # ------------------------------------------------------------------
        print("\n=== saccade --scheduler round_robin --estimator propagate ===")

        def run_rr(i: int) -> float | None:
            pft_path = tmp / f"saccade_rr_rep{i}.perfetto"
            return run_saccade_rep(
                args.saccade, args.library,
                "round_robin", "propagate",
                target, pft_path, args.gt_trace,
            )

        rr_scores = collect_reps("saccade/round_robin", run_rr, args.warmup, args.reps)

        if rr_scores:
            med, iqr = median_iqr(rr_scores)
            sig = (noise_floor is None) or (med > noise_floor)
            results.append({
                "approach": "saccade",
                "scheduler": "round_robin",
                "estimator": "propagate",
                "median_nrmse": med,
                "iqr": iqr,
                "reps": len(rr_scores),
                "significant": sig,
            })
            print(f"  median nRMSE={med:.4f}  IQR={iqr:.4f}  n={len(rr_scores)}")
        else:
            print("  No valid reps collected for saccade/round_robin.", file=sys.stderr)

    # ----------------------------------------------------------------------
    # Write CSV
    # ----------------------------------------------------------------------
    out_csv = args.results_dir / "q3_comparison.csv"
    header = "approach,scheduler,estimator,median_nrmse,iqr,reps,significant"
    rows = [header]
    for r in results:
        rows.append(
            f"{r['approach']},{r['scheduler']},{r['estimator']},"
            f"{r['median_nrmse']:.6f},{r['iqr']:.6f},{r['reps']},{r['significant']}"
        )
    out_csv.write_text("\n".join(rows) + "\n")
    print(f"\nResults written to: {out_csv}")

    # ----------------------------------------------------------------------
    # Print comparison table
    # ----------------------------------------------------------------------
    print("\nComparison table (lower nRMSE is better):")
    print(f"{'Approach':<12}  {'Scheduler':<14}  {'Estimator':<12}  "
          f"{'Median nRMSE':>14}  {'IQR':>10}  {'n':>4}  {'Significant':>11}")
    print("-" * 88)
    for r in results:
        label = "yes" if r["significant"] else "no (< floor)"
        print(
            f"{r['approach']:<12}  {r['scheduler']:<14}  {r['estimator']:<12}  "
            f"{r['median_nrmse']:>14.4f}  {r['iqr']:>10.4f}  {r['reps']:>4}  {label:>11}"
        )

    if not results:
        print("No results collected; check stderr for errors.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
