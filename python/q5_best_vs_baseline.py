#!/usr/bin/env python3
"""Compare best saccade config, baseline saccade, and actual perf stat.

Reads a q2_accuracy.py grid CSV to identify the best (scheduler, estimator)
combination per workload (by median nRMSE), then evaluates these approaches:

  best          -- best (scheduler, estimator) from the grid CSV (simulated)
  baseline      -- round-robin scheduler + propagate estimator (simulated)
  perf_stat     -- kernel-native multiplexing via `perf stat -I 100`
  best_real     -- the best config run live on real hardware via `saccade run`
                   (--real-reps reps, aggregated as median + IQR)
  baseline_real -- the baseline config run live on real hardware

All are evaluated against the ground-truth sweep traces using
`saccade evaluate --json`.  The perf stat output is converted to Perfetto
format via perf_to_perfetto.convert.

The *_real legs invoke `saccade run`, which executes the real benchmark binary
and profiles it live.  This requires the `saccade` binary to carry the needed
capabilities (granted via `setcap`) -- no sudo.  Note that `saccade run` uses
whatever physical hardware counter slots the CPU exposes; it has no --num-slots
flag.  The simulated legs use FIXED_NUM_SLOTS (4), so if the hardware slot count
differs from 4, part of any sim-vs-real gap is structural rather than estimator
error.  LLM-driven schedulers make live OpenRouter calls during the real run
(matching the endpoint/model used by the simulate path); LLM_API_KEY must be set.

CAVEAT -- sim-vs-real are NOT a clean apples-to-apples comparison of estimator
fidelity.  `saccade run` reprograms the hardware counters every quantum, and that
swap (HwCounters::update_slot -> one perf_event_open per CPU per changed slot;
e.g. 4 slots x 32 CPUs = 128 syscalls/quantum for random/round-robin) costs ~60ms
median (up to >1s), so a live quantum takes ~200ms+ even when --q-schedule asks
for 10ms.  `simulate` models this swap as free and steps exactly every q_schedule.
The live run therefore has ~20x coarser temporal resolution: its trace lands an
estimate in only ~20-25% of the ground-truth time bins (low `coverage`), and its
`median_nrmse` is scored over those sparse bins -- so real-vs-sim nRMSE differences
conflate estimator accuracy with this resolution gap.  Read the *_real `coverage`
and `median_nrmse` together, and against the simulated legs only as a rough sanity
check, not a controlled comparison.

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
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from sim_utils import (
    FIXED_NUM_SLOTS,
    LATENCY_PROFILE,
    LLM_SCHEDULERS,
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


def median_iqr(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return float(np.median(arr)), q3 - q1


def run_saccade_real_rep(
    saccade: Path,
    library: Path,
    scheduler: str,
    estimator: str,
    bench: dict,
    q_schedule: int,
    q_sample: int,
    llm_model: str,
    out_trace: Path,
    gt_trace: Path,
    gt_tid: int | None,
) -> tuple[float, float] | None:
    """Run one live `saccade run` and evaluate its trace against *gt_trace*.

    Executes the real benchmark binary (bench["binary"] + bench["args"]) in
    bench["cwd"], writing the live profiler trace to *out_trace*.  Returns
    (median_nrmse, mean_coverage), or None if the run failed.

    `saccade run` keys counter tracks under the target's real OS tid, whereas the
    sweep ground truth uses a synthetic tid; evaluate joins on (event, tid), so
    when *gt_tid* is given the run trace's tid is remapped to it before evaluation
    (see perf_to_perfetto.remap_trace_tid).  Without this the join misses entirely
    and coverage is 0.

    For LLM schedulers the LLM endpoint/model flags mirror sim_utils.run_simulate
    so the live run uses the same model and OpenRouter endpoint the simulator did.
    --llm-latency-profile is intentionally omitted: that injects synthetic latency
    into `simulate`, whereas `run` makes real calls in real time.
    """
    cmd = [
        str(saccade),
        "run",
        "--library", str(library),
        "--scheduler", scheduler,
        "--estimator", estimator,
        "--q-schedule", str(q_schedule),
        "--q-sample", str(q_sample),
        "--trace", str(out_trace),
    ]
    if scheduler in LLM_SCHEDULERS:
        cmd += [
            "--llm-model", llm_model,
            "--llm-base-url", "https://openrouter.ai/api",
            "--llm-api-key", os.environ["LLM_API_KEY"],
        ]
    cmd += ["--", bench["binary"]] + bench["args"]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=bench["cwd"])
    if result.returncode != 0:
        print(
            f"  saccade run failed (rc={result.returncode}):\n{result.stderr.strip()}",
            file=sys.stderr,
        )
        return None

    # Align the live run's real OS tid with the ground truth's synthetic tid so
    # `saccade evaluate` (which joins on (event, tid)) can match the series.
    eval_trace = out_trace
    if gt_tid is not None:
        sys.path.insert(0, str(Path(__file__).parent))
        from perf_to_perfetto import remap_trace_tid  # noqa: PLC0415
        aligned = out_trace.with_suffix(".tid.perfetto")
        orig_tids = remap_trace_tid(out_trace, aligned, gt_tid)
        if len(orig_tids) == 1:
            eval_trace = aligned
        else:
            print(f"  WARNING: run trace has {len(orig_tids)} thread tids {orig_tids}; "
                  f"cannot align to GT tid {gt_tid}, evaluating unaligned.", file=sys.stderr)

    eval_json = run_evaluate(saccade, gt_trace, eval_trace)
    return median_nrmse(eval_json), mean_coverage(eval_json)


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
    p.add_argument(
        "--llm-latency-profile",
        type=Path,
        default=LATENCY_PROFILE,
        help="LLM latency profile JSON forwarded to every simulate call "
        "(e.g. a fresh q7 llm_latency_profile.json).",
    )
    p.add_argument("--q-schedule", type=int, default=10_000_000)
    p.add_argument(
        "--q-sample",
        type=int,
        default=100_000,
        help="eBPF sample cadence (ns) forwarded to `saccade run` for the real legs.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise-floor-json", type=Path, default=None)
    p.add_argument(
        "--real-reps",
        type=int,
        default=5,
        help="Measured `saccade run` reps per config for the *_real legs (default: 5).",
    )
    p.add_argument(
        "--real-warmup",
        type=int,
        default=1,
        help="Discarded warm-up `saccade run` reps before the measured ones (default: 1).",
    )
    p.add_argument(
        "--no-real",
        action="store_true",
        help="Skip the live `saccade run` (*_real) legs entirely.",
    )
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
    raw_csv = run_dir / "q5_real_runs_raw.csv"

    fieldnames = [
        "workload", "config_label", "scheduler", "estimator",
        "median_nrmse", "coverage", "nrmse_iqr", "reps",
    ]
    if noise_floor is not None:
        fieldnames.append("significant")

    rows: list[dict] = []
    raw_rows: list[dict] = []  # per-rep samples for the *_real legs
    convert = _import_perf_convert()

    for trace_path in traces:
        workload = trace_path.stem
        print(f"\n=== {workload} ===", file=sys.stderr)

        bench = benchmarks.get(workload)
        if bench is None:
            print(f"  No benchmark info for {workload}, skipping.", file=sys.stderr)
            continue

        best_row = best_per_workload.get(workload)
        gt_tids: list[int] = []  # ground-truth thread tids, learned from a sim eval

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
                    latency_profile=args.llm_latency_profile,
                )
                eval_json = run_evaluate(args.saccade, trace_path, est_trace)
                mn = median_nrmse(eval_json)
                mc = mean_coverage(eval_json)
                if not gt_tids:
                    gt_tids = sorted({e["tid"] for e in eval_json.get("per_event", [])})
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
        # 2. real hardware runs (`saccade run`), --real-reps reps per config
        # ------------------------------------------------------------------
        if not args.no_real:
            # `saccade run` keys tracks under the target's real OS tid; align it
            # to the GT's synthetic tid (single-threaded SPEC rate workloads).
            real_gt_tid = gt_tids[0] if len(gt_tids) == 1 else None
            if real_gt_tid is None:
                print(f"  WARNING: {workload} GT has tids {gt_tids}; real-run traces "
                      f"cannot be tid-aligned, coverage will be ~0.", file=sys.stderr)
            for config_label, scheduler, estimator in [
                ("best_real",     best_row["scheduler"] if best_row else None, best_row["estimator"] if best_row else None),
                ("baseline_real", BASELINE_SCHEDULER, BASELINE_ESTIMATOR),
            ]:
                if scheduler is None:
                    print(f"  No grid result for {workload}, skipping {config_label}.", file=sys.stderr)
                    continue

                nrmses: list[float] = []
                covs: list[float] = []
                total = args.real_warmup + args.real_reps
                for i in range(total):
                    phase = "warmup" if i < args.real_warmup else f"rep {i - args.real_warmup + 1}/{args.real_reps}"
                    print(f"  run [{config_label}] {scheduler}+{estimator} {phase} ...", file=sys.stderr)
                    real_trace = tmp_dir / f"{workload}_{config_label}_rep{i}.perfetto"
                    res = run_saccade_real_rep(
                        args.saccade, args.library, scheduler, estimator,
                        bench, args.q_schedule, args.q_sample, args.llm_model,
                        real_trace, trace_path, real_gt_tid,
                    )
                    if i < args.real_warmup:
                        continue
                    rep_idx = i - args.real_warmup
                    if res is None:
                        print(f"  [{config_label}] rep {rep_idx + 1} failed, skipping.", file=sys.stderr)
                        rn = rc = ""
                    else:
                        rn, rc = res
                        nrmses.append(rn)
                        covs.append(rc)
                    raw_rows.append({
                        "workload": workload,
                        "config_label": config_label,
                        "rep": rep_idx,
                        "median_nrmse": rn,
                        "coverage": rc,
                    })

                if nrmses:
                    mn, iqr = median_iqr(nrmses)
                    mc = float(np.mean(covs))
                else:
                    mn = iqr = mc = None

                sig = is_significant(mn, noise_floor)
                row = {
                    "workload": workload,
                    "config_label": config_label,
                    "scheduler": scheduler,
                    "estimator": estimator,
                    "median_nrmse": mn if mn is not None else "",
                    "coverage": mc if mc is not None else "",
                    "nrmse_iqr": iqr if mn is not None else "",
                    "reps": len(nrmses),
                }
                if noise_floor is not None:
                    row["significant"] = "" if sig is None else str(sig).lower()
                rows.append(row)

        # ------------------------------------------------------------------
        # 3. perf stat
        # ------------------------------------------------------------------
        print(f"  perf stat ...", file=sys.stderr)
        # perf stat aggregates all threads; align its single series with the
        # ground truth's tid.  These SPEC rate workloads are single-threaded, so
        # GT has exactly one tid; if it ever has more, fall back to tid=0.
        if len(gt_tids) == 1:
            perf_tid = gt_tids[0]
        else:
            perf_tid = 0
            if len(gt_tids) > 1:
                print(f"  WARNING: {workload} ground truth has {len(gt_tids)} tids "
                      f"{gt_tids}; perf stat is thread-aggregate, coverage will be 0.",
                      file=sys.stderr)
        with tempfile.TemporaryDirectory(prefix="q5_") as tmp:
            tmp_p = Path(tmp)
            perf_csv = tmp_p / "perf.csv"
            perf_perfetto = tmp_p / "perf.perfetto"

            if run_perf_stat(bench["binary"], bench["args"], bench["cwd"], event_str, perf_csv):
                try:
                    convert(perf_csv, perf_perfetto, tid=perf_tid)
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

    if raw_rows:
        with raw_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["workload", "config_label", "rep", "median_nrmse", "coverage"]
            )
            writer.writeheader()
            writer.writerows(raw_rows)
        print(f"Per-rep real-run samples written to: {raw_csv}", file=sys.stderr)

    print("\nSummary (median nRMSE; lower is better):", file=sys.stderr)
    workloads = sorted({r["workload"] for r in rows})
    for wl in workloads:
        wl_rows = {r["config_label"]: r for r in rows if r["workload"] == wl}

        def fmt_real(label: str) -> str:
            r = wl_rows.get(label)
            if not r or r.get("median_nrmse", "") == "":
                return "—"
            iqr = r.get("nrmse_iqr", "")
            return f"{r['median_nrmse']:.4f}±{iqr:.4f}" if iqr != "" else f"{r['median_nrmse']}"

        best_nrmse = wl_rows.get("best", {}).get("median_nrmse", "—")
        base_nrmse = wl_rows.get("baseline", {}).get("median_nrmse", "—")
        perf_nrmse = wl_rows.get("perf_stat", {}).get("median_nrmse", "—")
        print(
            f"  {wl}: best={best_nrmse} (real {fmt_real('best_real')})  "
            f"baseline={base_nrmse} (real {fmt_real('baseline_real')})  "
            f"perf_stat={perf_nrmse}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
