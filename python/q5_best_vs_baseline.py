#!/usr/bin/env python3
"""Compare best saccade config, baseline saccade, and actual perf stat.

Reads a q2_accuracy.py grid CSV to identify the best (scheduler, estimator)
combination per workload (by median nRMSE), then evaluates three approaches:

  best      -- best (scheduler, estimator) from the grid CSV
  baseline  -- round-robin scheduler + propagate estimator
  perf_stat -- kernel-native multiplexing via `perf stat -I 100`

All three are evaluated against the ground-truth sweep traces using
`saccade evaluate --json`.  The perf stat output is converted to Perfetto
format via perf_to_perfetto.convert.

Operates on SPEC workloads only.

Benchmark info (binary, args, cwd) must be provided via --benchmarks-json,
a JSON file mapping workload slug to {"binary": ..., "args": [...], "cwd": ...}.
Example:
    {
      "spec_531_deepsjeng_r": {
        "binary": "/tank/.../531.deepsjeng_r/exe/deepsjeng_r_base.mytest-m64",
        "args": ["ref.txt"],
        "cwd": "/tank/.../531.deepsjeng_r/run/run_base_refrate_mytest-m64.0000"
      }
    }
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from sim_utils import (
    FIXED_NUM_SLOTS,
    filter_traces_by_kind,
    is_significant,
    load_noise_floor,
    median_nrmse,
    mean_coverage,
    run_evaluate,
    run_simulate,
    REPO_ROOT,
)

BASELINE_SCHEDULER = "round-robin"
BASELINE_ESTIMATOR = "propagate"


def _import_perf_convert():
    sys.path.insert(0, str(Path(__file__).parent))
    from perf_to_perfetto import convert  # noqa: PLC0415
    return convert


def load_grid_csv(grid_csv: Path) -> list[dict]:
    rows = []
    with grid_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("median_nrmse", ""):
                row["median_nrmse"] = float(row["median_nrmse"])
            rows.append(row)
    return rows


def best_combo_per_workload(rows: list[dict]) -> dict[str, dict]:
    """Return {workload: best_row} where best is argmin median_nrmse."""
    best: dict[str, dict] = {}
    for row in rows:
        if row.get("median_nrmse") == "":
            continue
        wl = row["workload"]
        if wl not in best or float(row["median_nrmse"]) < float(best[wl]["median_nrmse"]):
            best[wl] = row
    return best


def run_perf_stat(
    binary: str,
    binary_args: list[str],
    cwd: str,
    event_str: str,
    tmp_csv: Path,
) -> bool:
    """Run `perf stat -I 100 -x, -e <events>` and write CSV to tmp_csv.

    Returns True on success.  perf stat writes CSV to stderr when -x is used.
    """
    cmd = ["perf", "stat", "-I", "100", "-x,", "-e", event_str, "--", binary] + binary_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    csv_data = result.stderr
    if result.returncode != 0 and not csv_data.strip():
        print(f"  perf stat failed (rc={result.returncode}):\n{result.stderr.strip()}",
              file=sys.stderr)
        return False
    tmp_csv.write_text(csv_data)
    return True


def load_events_from_library(library: Path) -> list[str]:
    with library.open() as f:
        data = json.load(f)
    return [e["name"] for e in data["events"]]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare best saccade config vs. baseline vs. perf stat "
            "against the ground-truth sweep traces (SPEC only)."
        )
    )
    p.add_argument(
        "--grid-csv",
        type=Path,
        required=True,
        help="q2_accuracy.py output CSV (SPEC grid)",
    )
    p.add_argument(
        "--benchmarks-json",
        type=Path,
        required=True,
        help=(
            "JSON mapping workload slug to {binary, args, cwd}. "
            "Required for the perf stat comparison."
        ),
    )
    p.add_argument("--saccade", type=Path, default=Path("../target/release/saccade"))
    p.add_argument("--library", type=Path, default=Path("../event_lib.json"))
    p.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("./sweep_data_eval_traces"),
    )
    p.add_argument("--results-dir", type=Path, default=Path("./results"))
    p.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Process only this workload slug",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default="google/gemma-4-26b-a4b-it",
        help="Model name for LLM-driven schedulers; uses OpenRouter unless sim_utils is customized.",
    )
    p.add_argument("--q-schedule", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise-floor-json", type=Path, default=None)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.saccade = args.saccade.resolve()
    args.library = args.library.resolve()
    args.traces_dir = args.traces_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.grid_csv = args.grid_csv.resolve()
    args.benchmarks_json = args.benchmarks_json.resolve()

    for p, label in [
        (args.saccade, "saccade binary"),
        (args.library, "library"),
        (args.traces_dir, "traces-dir"),
        (args.grid_csv, "grid-csv"),
        (args.benchmarks_json, "benchmarks-json"),
    ]:
        if not p.exists():
            parser.error(f"{label} not found: {p}")

    grid_rows = load_grid_csv(args.grid_csv)
    best_per_workload = best_combo_per_workload(grid_rows)

    benchmarks: dict = json.loads(args.benchmarks_json.read_text())

    traces = filter_traces_by_kind(sorted(args.traces_dir.glob("*.perfetto")), "spec")
    if args.workload is not None:
        traces = [t for t in traces if t.stem == args.workload]
    if not traces:
        parser.error("No matching SPEC traces found.")

    events = load_events_from_library(args.library)
    event_str = ",".join(events)

    noise_floor = load_noise_floor(args.noise_floor_json)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "traces"
    tmp_dir.mkdir()
    out_csv = run_dir / "q5_comparison.csv"

    fieldnames = [
        "workload", "config_label", "scheduler", "estimator",
        "median_nrmse", "coverage",
    ]
    if noise_floor is not None:
        fieldnames.append("significant")

    rows: list[dict] = []
    convert = _import_perf_convert()

    for trace_path in traces:
        workload = trace_path.stem
        print(f"\n=== {workload} ===", file=sys.stderr)

        bench = benchmarks.get(workload)
        if bench is None:
            print(f"  No benchmark info for {workload}, skipping.", file=sys.stderr)
            continue

        best_row = best_per_workload.get(workload)

        # ------------------------------------------------------------------
        # 1. best config (re-simulate from grid result)
        # ------------------------------------------------------------------
        for config_label, scheduler, estimator in [
            ("best",     best_row["scheduler"] if best_row else None, best_row["estimator"] if best_row else None),
            ("baseline", BASELINE_SCHEDULER, BASELINE_ESTIMATOR),
        ]:
            if scheduler is None:
                print(f"  No grid result for {workload}, skipping {config_label}.", file=sys.stderr)
                continue
            print(f"  simulate [{config_label}] {scheduler}+{estimator} ...", file=sys.stderr)
            est_trace = tmp_dir / f"{workload}_{config_label}.perfetto"
            try:
                run_simulate(
                    args.saccade, args.library, trace_path,
                    scheduler, estimator, est_trace,
                    FIXED_NUM_SLOTS, args.q_schedule, args.llm_model, args.seed,
                )
                eval_json = run_evaluate(args.saccade, trace_path, est_trace)
                mn = median_nrmse(eval_json)
                mc = mean_coverage(eval_json)
            except subprocess.CalledProcessError as exc:
                print(f"  ERROR [{config_label}]: {exc}", file=sys.stderr)
                mn = mc = None

            sig = is_significant(mn, noise_floor)
            row: dict = {
                "workload": workload,
                "config_label": config_label,
                "scheduler": scheduler,
                "estimator": estimator,
                "median_nrmse": mn if mn is not None else "",
                "coverage": mc if mc is not None else "",
            }
            if noise_floor is not None:
                row["significant"] = "" if sig is None else str(sig).lower()
            rows.append(row)

        # ------------------------------------------------------------------
        # 2. perf stat
        # ------------------------------------------------------------------
        print(f"  perf stat ...", file=sys.stderr)
        with tempfile.TemporaryDirectory(prefix="q5_") as tmp:
            tmp_p = Path(tmp)
            perf_csv = tmp_p / "perf.csv"
            perf_perfetto = tmp_p / "perf.perfetto"

            if run_perf_stat(bench["binary"], bench["args"], bench["cwd"], event_str, perf_csv):
                try:
                    convert(str(perf_csv), str(perf_perfetto))
                    eval_json = run_evaluate(args.saccade, trace_path, perf_perfetto)
                    mn = median_nrmse(eval_json)
                    mc = mean_coverage(eval_json)
                except Exception as exc:  # noqa: BLE001
                    print(f"  perf stat eval failed: {exc}", file=sys.stderr)
                    mn = mc = None
            else:
                mn = mc = None

        sig = is_significant(mn, noise_floor)
        row = {
            "workload": workload,
            "config_label": "perf_stat",
            "scheduler": "",
            "estimator": "",
            "median_nrmse": mn if mn is not None else "",
            "coverage": mc if mc is not None else "",
        }
        if noise_floor is not None:
            row["significant"] = "" if sig is None else str(sig).lower()
        rows.append(row)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to: {out_csv}", file=sys.stderr)

    print("\nSummary (median nRMSE; lower is better):", file=sys.stderr)
    workloads = sorted({r["workload"] for r in rows})
    for wl in workloads:
        wl_rows = {r["config_label"]: r for r in rows if r["workload"] == wl}
        best_nrmse = wl_rows.get("best", {}).get("median_nrmse", "—")
        base_nrmse = wl_rows.get("baseline", {}).get("median_nrmse", "—")
        perf_nrmse = wl_rows.get("perf_stat", {}).get("median_nrmse", "—")
        print(f"  {wl}: best={best_nrmse}  baseline={base_nrmse}  perf_stat={perf_nrmse}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
