#!/usr/bin/env python3
"""Profile LLM call latency distributions for the three LLM scheduler call types.

Runs each LLM scheduler on a representative trace (repeated --reps times) and
parses ``llm_call`` lines from stderr to collect per-call latency samples.

The three call types:
  static_setup   -- static-llm and dynamic-llm initial scheduling call
  dynamic_update -- dynamic-llm periodic rescheduling call
  wrr_setup      -- weighted-round-robin-llm weight assignment call

Output:
  llm_latency_profile.json  -- distribution JSON suitable for --llm-latency-profile
  q7_llm_latency_raw.csv    -- per-call samples
  q7_llm_latency_summary.csv -- per-call-type statistics

The profile JSON can later be passed to `saccade simulate --llm-latency-profile`
to inject realistic LLM overhead without making live API calls.
"""

import argparse
import concurrent.futures
import csv
import json
import re
import subprocess
import sys
import tempfile
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from sim_utils import (
    FIXED_NUM_SLOTS,
    LLM_SCHEDULERS,
    filter_traces_by_kind,
    REPO_ROOT,
)

LLM_CALL_PATTERN = re.compile(
    r'llm_call latency_ms=(\d+) model="([^"]+)" call_type="([^"]+)"'
)

LLM_SCHEDULER_LIST = ["static-llm", "dynamic-llm", "weighted-round-robin-llm"]


def run_simulate_capture_stderr(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    out_trace: Path,
    num_slots: int,
    q_schedule: int,
    seed: int | None,
) -> str:
    """Run saccade simulate and return captured stderr (contains llm_call lines)."""
    cmd = [
        str(saccade),
        "simulate",
        "--library", str(library),
        "--rates-trace", str(rates_trace),
        "--scheduler", scheduler,
        "--estimator", "propagate",
        # "--llm-model", "gemma4",
        "--llm-model", "google/gemma-4-26b-a4b-it",
        "--llm-base-url", "https://openrouter.ai/api",
        "--llm-api-key", os.environ["LLM_API_KEY"],
        "--trace", str(out_trace),
        "--num-slots", str(num_slots),
        "--q-schedule", str(q_schedule),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"ERROR!!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result.stderr


def parse_llm_calls(stderr: str) -> list[dict]:
    """Extract llm_call log lines from stderr into a list of dicts."""
    calls = []
    for m in LLM_CALL_PATTERN.finditer(stderr):
        calls.append({
            "latency_ms": int(m.group(1)),
            "model": m.group(2),
            "call_type": m.group(3),
        })
    return calls


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Profile LLM call latency distributions for all three LLM schedulers."
    )
    p.add_argument("--saccade", type=Path, default=Path("../target/release/saccade"))
    p.add_argument("--library", type=Path, default=Path("../event_lib.json"))
    p.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("./sweep_data_eval_traces"),
    )
    p.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Stem of a single trace to use (default: first SPEC trace found)",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Repetitions per scheduler (default: 10)",
    )
    p.add_argument("--q-schedule", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("./results"))
    p.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel simulation runs (default: 1)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.saccade = args.saccade.resolve()
    args.library = args.library.resolve()
    args.traces_dir = args.traces_dir.resolve()
    args.results_dir = args.results_dir.resolve()

    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")
    if not args.library.exists():
        parser.error(f"Library not found: {args.library}")
    if not args.traces_dir.is_dir():
        parser.error(f"Traces directory not found: {args.traces_dir}")

    all_traces = filter_traces_by_kind(sorted(args.traces_dir.glob("*.perfetto")), "spec")
    if args.workload is not None:
        all_traces = [t for t in all_traces if t.stem == args.workload]
    if not all_traces:
        parser.error("No matching SPEC traces found.")
    trace_path = all_traces[0]
    print(f"Using trace: {trace_path.stem}", file=sys.stderr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (scheduler, rep)
        for scheduler in LLM_SCHEDULER_LIST
        for rep in range(args.reps)
    ]
    total = len(tasks)
    print(f"Running {total} simulations with -j {args.jobs} ...", file=sys.stderr)

    raw_rows: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="q7_") as tmp:
        tmp_dir = Path(tmp)

        def run_task(task: tuple[str, int]) -> list[dict]:
            scheduler, rep = task
            out_trace = tmp_dir / f"{scheduler}_rep{rep}.perfetto"
            try:
                stderr = run_simulate_capture_stderr(
                    args.saccade, args.library, trace_path,
                    scheduler, out_trace,
                    FIXED_NUM_SLOTS, args.q_schedule, args.seed,
                )
            except subprocess.CalledProcessError as exc:
                print(f"  {scheduler} rep {rep} failed: {exc}", file=sys.stderr)
                return []
            calls = parse_llm_calls(stderr)
            if not calls:
                print(
                    f"  {scheduler} rep {rep}: no llm_call lines found in stderr. "
                    "Is the binary built with the latest Rust changes?",
                    file=sys.stderr,
                )
            print(f"  {scheduler} rep {rep}: {len(calls)} call(s)", file=sys.stderr)
            return [
                {
                    "scheduler": scheduler,
                    "rep": rep,
                    "call_index": i,
                    "call_type": call["call_type"],
                    "latency_ms": call["latency_ms"],
                    "model": call["model"],
                }
                for i, call in enumerate(calls)
            ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for rows in executor.map(run_task, tasks):
                raw_rows.extend(rows)

    # -----------------------------------------------------------------------
    # Write raw CSV
    # -----------------------------------------------------------------------
    raw_csv = run_dir / "q7_llm_latency_raw.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scheduler", "rep", "call_index", "call_type", "latency_ms", "model"]
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    # -----------------------------------------------------------------------
    # Summary statistics per call_type
    # -----------------------------------------------------------------------
    by_call_type: dict[str, list[int]] = {}
    for row in raw_rows:
        ct = row["call_type"]
        by_call_type.setdefault(ct, []).append(row["latency_ms"])

    summary_rows: list[dict] = []
    for call_type, samples in sorted(by_call_type.items()):
        arr = np.array(samples, dtype=float)
        summary_rows.append({
            "call_type": call_type,
            "n_samples": len(samples),
            "median_ms": float(np.median(arr)),
            "p25_ms": float(np.percentile(arr, 25)),
            "p75_ms": float(np.percentile(arr, 75)),
            "p95_ms": float(np.percentile(arr, 95)),
        })

    summary_csv = run_dir / "q7_llm_latency_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["call_type", "n_samples", "median_ms", "p25_ms", "p75_ms", "p95_ms"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # -----------------------------------------------------------------------
    # Distribution JSON (input to --llm-latency-profile)
    # -----------------------------------------------------------------------
    profile: dict = {}
    for call_type, samples in by_call_type.items():
        arr = np.array(samples, dtype=float)
        profile[call_type] = {
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "samples": [int(s) for s in samples],
        }

    profile_json = run_dir / "llm_latency_profile.json"
    profile_json.write_text(json.dumps(profile, indent=2))

    print(f"\nResults written to {run_dir}/", file=sys.stderr)
    print(f"  {raw_csv.name}", file=sys.stderr)
    print(f"  {summary_csv.name}", file=sys.stderr)
    print(f"  {profile_json.name}", file=sys.stderr)

    print("\nLatency summary:", file=sys.stderr)
    for row in summary_rows:
        print(
            f"  {row['call_type']:20s}  n={row['n_samples']:4d}  "
            f"median={row['median_ms']:.0f}ms  p95={row['p95_ms']:.0f}ms",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
