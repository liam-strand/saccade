#!/usr/bin/env python3
"""Evaluate whether adaptive scheduling produces larger accuracy gains on phase-rich programs vs steady-state programs.

Runs saccade sweep to collect ground-truth traces (once per workload config), then
replays each trace through every scheduler using saccade simulate + evaluate, and
writes a CSV summarising median nRMSE, mean coverage, and delta vs random baseline.

Workload configs are written to JSON files that the workload binary reads:
  - Multi-phase: 4 phases x 10s, each a different workload kind.
  - Steady-state: 1 phase x 40s, cache_thrash (representative mid-range workload).
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Workload configurations
# ---------------------------------------------------------------------------

MULTIPHASE_CONFIG = {
    "phases": [
        {"duration_secs": 10, "threads": 1, "type": "cache_thrash", "array_size_kb": 4096},
        {"duration_secs": 10, "threads": 1, "type": "fp_heavy", "vector_size": 65536},
        {"duration_secs": 10, "threads": 1, "type": "branch_mispredict", "array_size": 1024},
        {"duration_secs": 10, "threads": 1, "type": "mem_stream", "buffer_size_mb": 256},
    ]
}

STEADYSTATE_CONFIG = {
    "phases": [
        {"duration_secs": 40, "threads": 1, "type": "cache_thrash", "array_size_kb": 4096},
    ]
}

# Schedulers that do not require an LLM API — always run these.
BASELINE_SCHEDULERS = ["random", "round-robin", "max-uncertainty", "rate-of-change"]

# LLM-backed schedulers — attempted but skipped gracefully on failure.
LLM_SCHEDULERS = ["static-llm", "dynamic-llm", "weighted-round-robin-llm"]

ALL_SCHEDULERS = BASELINE_SCHEDULERS + LLM_SCHEDULERS


# ---------------------------------------------------------------------------
# Helper: run saccade subcommands
# ---------------------------------------------------------------------------


def run_sweep(
    saccade: Path,
    workload: Path,
    config_json: Path,
    trace: Path,
    library: Path | None,
    q_schedule: int,
) -> None:
    """Run saccade sweep against the workload binary with a given JSON config."""
    cmd = [
        str(saccade),
        "sweep",
        "--quiet",
        "--trace",
        str(trace),
        "--q-schedule",
        str(q_schedule),
    ]
    if library:
        cmd += ["--library", str(library)]
    cmd += ["--", str(workload), str(config_json)]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_simulate(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    out_trace: Path,
    q_schedule: int,
    num_slots: int = 4,
) -> None:
    """Replay a sweep trace through the given scheduler."""
    cmd = [
        str(saccade),
        "simulate",
        "--library",
        str(library),
        "--rates-trace",
        str(rates_trace),
        "--scheduler",
        scheduler,
        "--estimator",
        "propagate",
        "--q-schedule",
        str(q_schedule),
        "--num-slots",
        str(num_slots),
        "--trace",
        str(out_trace),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def run_evaluate(
    saccade: Path,
    ground_truth: Path,
    estimated: Path,
) -> dict:
    """Compare estimated trace against ground-truth; returns parsed JSON."""
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


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def median_nrmse_from(eval_json: dict) -> float | None:
    """Extract median per-event nRMSE from evaluate JSON output."""
    values = [
        entry["nrmse"]
        for entry in eval_json.get("per_event", [])
        if entry["nrmse"] is not None
    ]
    return float(np.median(values)) if values else None


def mean_coverage_from(eval_json: dict) -> float | None:
    """Extract mean_coverage from evaluate JSON output (fraction 0-1)."""
    cov = eval_json.get("mean_coverage")
    return float(cov) if cov is not None else None


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def evaluate_workload(
    saccade: Path,
    library: Path,
    gt_trace: Path,
    q_schedule: int,
    schedulers: list[str],
    tmp_dir: Path,
) -> list[dict]:
    """Simulate every scheduler against gt_trace; return list of result dicts."""
    rows = []
    for sched in schedulers:
        est_trace = tmp_dir / f"est_{sched}.perfetto"
        try:
            run_simulate(saccade, library, gt_trace, sched, est_trace, q_schedule)
            eval_json = run_evaluate(saccade, gt_trace, est_trace)
            med = median_nrmse_from(eval_json)
            cov = mean_coverage_from(eval_json)
        except subprocess.CalledProcessError as exc:
            print(
                f"    WARN: scheduler={sched} failed (exit {exc.returncode}); skipping.",
                file=sys.stderr,
            )
            med = None
            cov = None
        rows.append({"scheduler": sched, "median_nrmse": med, "coverage": cov})
        status = f"{med:.4f}" if med is not None else "FAILED"
        print(f"    {sched:<30} median_nRMSE={status}")
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Q5: Evaluate whether adaptive scheduling produces larger accuracy "
            "gains on phase-rich programs vs steady-state programs."
        )
    )
    parser.add_argument(
        "--saccade",
        type=Path,
        default=Path("./target/release/saccade"),
        help="saccade binary (default: ./target/release/saccade)",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("./target/release/workload"),
        help="workload binary (default: ./target/release/workload)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Event library JSON file (required; passed to both sweep and simulate)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory for output files (default: ./results)",
    )
    parser.add_argument(
        "--gt-multiphase",
        type=Path,
        default=None,
        metavar="PATH",
        help="Pre-existing GT sweep trace for the multi-phase workload (skip sweep if provided)",
    )
    parser.add_argument(
        "--gt-steadystate",
        type=Path,
        default=None,
        metavar="PATH",
        help="Pre-existing GT sweep trace for the steady-state workload (skip sweep if provided)",
    )
    parser.add_argument(
        "--q-schedule",
        type=int,
        default=10_000_000,
        help="Scheduler quantum in nanoseconds (default: 10000000)",
    )
    args = parser.parse_args()

    # Resolve paths.
    args.saccade = args.saccade.resolve()
    args.workload = args.workload.resolve()
    args.library = args.library.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.gt_multiphase:
        args.gt_multiphase = args.gt_multiphase.resolve()
    if args.gt_steadystate:
        args.gt_steadystate = args.gt_steadystate.resolve()

    # Validate binaries.
    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")
    if not args.workload.exists():
        parser.error(f"workload binary not found: {args.workload}")
    if not args.library.exists():
        parser.error(f"library JSON not found: {args.library}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Write workload JSON configs to results dir so they are auditable.
    configs_dir = args.results_dir / "workload_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    multiphase_cfg_path = configs_dir / "multiphase.json"
    steadystate_cfg_path = configs_dir / "steadystate.json"

    multiphase_cfg_path.write_text(json.dumps(MULTIPHASE_CONFIG, indent=2) + "\n")
    steadystate_cfg_path.write_text(json.dumps(STEADYSTATE_CONFIG, indent=2) + "\n")
    print(f"Workload configs written to {configs_dir}/")

    # -----------------------------------------------------------------------
    # Ground-truth sweeps.
    # -----------------------------------------------------------------------

    gt_multiphase = args.gt_multiphase or (args.results_dir / "gt_multiphase.perfetto")
    gt_steadystate = args.gt_steadystate or (args.results_dir / "gt_steadystate.perfetto")

    if not args.gt_multiphase:
        print("\n[1/2] Collecting multi-phase ground-truth sweep...")
        run_sweep(
            args.saccade,
            args.workload,
            multiphase_cfg_path,
            gt_multiphase,
            args.library,
            args.q_schedule,
        )
        print(f"      -> {gt_multiphase}")
    else:
        print(f"\n[1/2] Using pre-existing multi-phase GT: {gt_multiphase}")

    if not args.gt_steadystate:
        print("\n[2/2] Collecting steady-state ground-truth sweep...")
        run_sweep(
            args.saccade,
            args.workload,
            steadystate_cfg_path,
            gt_steadystate,
            args.library,
            args.q_schedule,
        )
        print(f"      -> {gt_steadystate}")
    else:
        print(f"\n[2/2] Using pre-existing steady-state GT: {gt_steadystate}")

    # -----------------------------------------------------------------------
    # Simulate + evaluate all schedulers against both workload types.
    # -----------------------------------------------------------------------

    workloads = [
        ("multiphase", gt_multiphase),
        ("steadystate", gt_steadystate),
    ]

    all_rows: list[dict] = []  # {workload_type, scheduler, median_nrmse, coverage}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for wl_type, gt_trace in workloads:
            print(f"\nSimulating schedulers on workload_type={wl_type}...")
            rows = evaluate_workload(
                args.saccade,
                args.library,
                gt_trace,
                args.q_schedule,
                ALL_SCHEDULERS,
                tmp_dir,
            )
            for r in rows:
                all_rows.append({"workload_type": wl_type, **r})

    # -----------------------------------------------------------------------
    # Compute delta_vs_random.
    # -----------------------------------------------------------------------

    # Build lookup: workload_type -> random_median_nrmse
    random_nrmse: dict[str, float | None] = {}
    for row in all_rows:
        if row["scheduler"] == "random":
            random_nrmse[row["workload_type"]] = row["median_nrmse"]

    for row in all_rows:
        ref = random_nrmse.get(row["workload_type"])
        if ref is not None and row["median_nrmse"] is not None:
            row["delta_vs_random"] = row["median_nrmse"] - ref
        else:
            row["delta_vs_random"] = None

    # -----------------------------------------------------------------------
    # Write CSV.
    # -----------------------------------------------------------------------

    csv_path = args.results_dir / "q5_phase_sensitivity.csv"
    fieldnames = ["workload_type", "scheduler", "median_nrmse", "coverage", "delta_vs_random"]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    "workload_type": row["workload_type"],
                    "scheduler": row["scheduler"],
                    "median_nrmse": (
                        f"{row['median_nrmse']:.6f}" if row["median_nrmse"] is not None else ""
                    ),
                    "coverage": (
                        f"{row['coverage']:.4f}" if row["coverage"] is not None else ""
                    ),
                    "delta_vs_random": (
                        f"{row['delta_vs_random']:.6f}"
                        if row["delta_vs_random"] is not None
                        else ""
                    ),
                }
            )

    print(f"\nResults written to: {csv_path}")

    # -----------------------------------------------------------------------
    # Print summary.
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Q5 PHASE SENSITIVITY SUMMARY")
    print("=" * 70)
    print(
        f"{'Workload':<14} {'Scheduler':<30} {'Median nRMSE':>13} {'Coverage':>9} {'Delta/random':>13}"
    )
    print("-" * 85)

    for row in all_rows:
        nrmse_str = f"{row['median_nrmse']:.4f}" if row["median_nrmse"] is not None else "FAILED"
        cov_str = f"{row['coverage']:.3f}" if row["coverage"] is not None else "  N/A"
        delta_str = (
            f"{row['delta_vs_random']:+.4f}" if row["delta_vs_random"] is not None else "  N/A"
        )
        print(
            f"{row['workload_type']:<14} {row['scheduler']:<30} {nrmse_str:>13} {cov_str:>9} {delta_str:>13}"
        )

    # Interpret the results.
    print()
    print("Interpretation: delta_vs_random = scheduler_nRMSE - random_nRMSE")
    print("  Negative -> scheduler outperforms random")
    print("  Positive -> scheduler is worse than random")
    print()

    # Find schedulers that improve most on multi-phase vs steady-state.
    multiphase_rows = {r["scheduler"]: r for r in all_rows if r["workload_type"] == "multiphase"}
    steady_rows = {r["scheduler"]: r for r in all_rows if r["workload_type"] == "steadystate"}
    all_scheds = sorted(
        set(multiphase_rows) & set(steady_rows),
        key=lambda s: (
            (multiphase_rows[s]["delta_vs_random"] or 0.0)
            - (steady_rows[s]["delta_vs_random"] or 0.0)
        ),
    )

    print("Schedulers ranked by phase-sensitivity benefit")
    print("(most negative = largest extra gain on multi-phase vs steady-state):")
    print(f"  {'Scheduler':<30} {'Delta(multi) - Delta(steady)':>30}")
    for sched in all_scheds:
        mp_d = multiphase_rows[sched]["delta_vs_random"]
        ss_d = steady_rows[sched]["delta_vs_random"]
        if mp_d is not None and ss_d is not None:
            diff = mp_d - ss_d
            print(f"  {sched:<30} {diff:>+30.4f}")
        else:
            print(f"  {sched:<30} {'N/A':>30}")


if __name__ == "__main__":
    main()
