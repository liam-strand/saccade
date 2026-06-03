#!/usr/bin/env python3
"""Master reproduction orchestrator for the saccade evaluation.

Runs every data-collection script (q1-q8; collect2.py's ground-truth sweep is
skipped, we reuse the existing ./sweep_data_eval_traces/) and then every analysis
script, with the correct arguments, to regenerate all figures in ./results/.

Run from the python/ directory:

    uv run python run_all.py                # full collection + analysis (LLM live)
    uv run python run_all.py --skip-collect # re-plot only, from existing data
    uv run python run_all.py --dry-run      # echo commands, run nothing

The two tricky bits this handles automatically:
  * Promotion - q2/q3 collection write their CSV into a fresh results/<timestamp>/
    dir, but the q2/q3 plot scripts read the *top-level* results/*.csv. We copy the
    timestamped CSV up after each of those collectors runs.
  * Dependency passing - q5 collection needs the q2 grid CSV via --grid-csv, and the
    q4/q5 plots take the captured run dir explicitly for determinism.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PY_DIR = Path(__file__).resolve().parent  # /home/yhe7443/saccade/python
RESULTS_DIR = PY_DIR / "results"

SACCADE = "../target/release/saccade"
LIBRARY = "../event_lib.json"
TRACES_DIR = "./sweep_data_eval_traces"
Q5_BENCH = "./config/q5_benchmarks.json"

NPB_BIN = "/tank/yhe7443/benchmarks/NPB3.3.1/NPB3.3-SER/bin"
Q1_TARGET = f"{NPB_BIN}/ua.A.x"
Q6_TARGET = f"{NPB_BIN}/cg.A.x"
Q8_TARGET = f"{NPB_BIN}/cg.B.x"

TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")

# Figures each analysis script is expected to (re)write, for the final summary.
EXPECTED_FIGURES = [
    "q1_overhead_bars.png",
    "q1_overhead_heatmap.png",
    "q1_overhead_by_sink.png",
    "q1_cost_model.png",
    "q1_insensitivity.png",
    "q1_sample_collapse.png",
    "q2_estimator_sweep_round-robin.png",
    "q2_estimator_sweep_max-uncertainty.png",
    "q2_estimator_sweep_dynamic-llm.png",
    "q2_heatmap.png",
    "q3_kf_variants.png",
    "q3_kf_sane.png",
    "q4_guidance.png",  # written inside the q4 run dir
    "q5_sim_vs_real.png",  # written inside the q5 run dir
    "q7_llm_latency_violin.png",
    "q8_swap_breakdown.png",
    "q8_swap_drift.png",
    "q8_swap_quantum.png",
]


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #
class Runner:
    """Tracks pass/fail/skip across stages and drives subprocesses."""

    def __init__(self, *, dry_run: bool, fail_fast: bool) -> None:
        self.dry_run = dry_run
        self.fail_fast = fail_fast
        self.results: list[tuple[str, str, str]] = []  # (label, status, detail)

    def record(self, label: str, status: str, detail: str = "") -> None:
        self.results.append((label, status, detail))

    def run(self, cmd: list[str], label: str) -> bool:
        """Run `uv run python <cmd>` in PY_DIR. Returns True on success."""
        printable = " ".join(str(c) for c in cmd)
        print(f"\n>>> [{label}] uv run python {printable}", flush=True)
        if self.dry_run:
            self.record(label, "DRY")
            return True
        try:
            proc = subprocess.run(
                ["uv", "run", "python", *[str(c) for c in cmd]],
                cwd=PY_DIR,
            )
        except Exception as exc:  # noqa: BLE001 - surface any spawn failure
            print(f"!!! [{label}] failed to launch: {exc}", file=sys.stderr)
            self.record(label, "FAIL", str(exc))
            if self.fail_fast:
                raise SystemExit(f"--fail-fast: aborting after {label}")
            return False
        if proc.returncode != 0:
            print(f"!!! [{label}] exited {proc.returncode}", file=sys.stderr)
            self.record(label, "FAIL", f"exit {proc.returncode}")
            if self.fail_fast:
                raise SystemExit(f"--fail-fast: aborting after {label}")
            return False
        self.record(label, "OK")
        return True

    def skip(self, label: str, reason: str) -> None:
        print(f"--- [{label}] SKIPPED: {reason}", flush=True)
        self.record(label, "SKIP", reason)


# --------------------------------------------------------------------------- #
# Helpers for the timestamped-run-dir dance
# --------------------------------------------------------------------------- #
def snapshot_run_dirs() -> set[str]:
    """Names of existing results/<timestamp>/ dirs."""
    if not RESULTS_DIR.is_dir():
        return set()
    return {
        p.name
        for p in RESULTS_DIR.iterdir()
        if p.is_dir() and TIMESTAMP_RE.match(p.name)
    }


def capture_new_run_dir(before: set[str]) -> Path:
    """Return the timestamp dir created since `before`. Raises if none appeared."""
    new = snapshot_run_dirs() - before
    if not new:
        raise RuntimeError("no new results/<timestamp>/ directory was created")
    return RESULTS_DIR / max(new)


def newest_run_dir_with(filename: str) -> Path | None:
    """Newest results/<ts>/ dir that contains `filename` (for --skip-collect)."""
    matches = sorted(
        p.parent for p in RESULTS_DIR.glob(f"*/{filename}") if p.is_file()
    )
    return matches[-1] if matches else None


def promote(run_dir: Path, filename: str) -> bool:
    """Copy run_dir/filename up to the top-level results/ dir."""
    src = run_dir / filename
    if not src.is_file():
        print(f"!!! promote: {src} missing", file=sys.stderr)
        return False
    shutil.copyfile(src, RESULTS_DIR / filename)
    print(f"    promoted {src.relative_to(RESULTS_DIR.parent)} -> results/{filename}")
    return True


# --------------------------------------------------------------------------- #
# Stage A - collection
# --------------------------------------------------------------------------- #
def collect(r: Runner) -> dict[str, Path | None]:
    """Run all collectors. Returns captured run dirs / grid path for later stages."""
    captured: dict[str, Path | None] = {
        "q2_grid": None,
        "q2_dir": None,
        "q3_dir": None,
        "q4_dir": None,
        "q5_dir": None,
        "q7_dir": None,
    }

    common = ["--saccade", SACCADE, "--library", LIBRARY]

    # 1. q1 overhead (top-level CSVs; live). --target must come last (REMAINDER).
    r.run(
        ["q1_overhead.py", *common, "--results-dir", "./results",
         "--warmup", "5", "--rounds", "20", "--target", Q1_TARGET],
        "q1_overhead",
    )

    # 2. q2 accuracy (timestamped; LLM live) -> promote + remember grid for q5.
    before = snapshot_run_dirs()
    if r.run(
        ["q2_accuracy.py", *common, "--traces-dir", TRACES_DIR,
         "--results-dir", "./results", "--jobs", "16"],
        "q2_accuracy",
    ):
        if not r.dry_run:
            try:
                q2_dir = capture_new_run_dir(before)
            except RuntimeError as exc:
                print(f"!!! q2_accuracy: {exc}", file=sys.stderr)
            else:
                captured["q2_dir"] = q2_dir
                if promote(q2_dir, "q2_scheduler_estimator.csv"):
                    captured["q2_grid"] = q2_dir / "q2_scheduler_estimator.csv"

    # 3. q3 kalman variants (timestamped) -> promote.
    before = snapshot_run_dirs()
    if r.run(
        ["q3_kf_variants.py", *common, "--traces-dir", TRACES_DIR,
         "--results-dir", "./results", "--jobs", "16"],
        "q3_kf_variants",
    ):
        if not r.dry_run:
            try:
                q3_dir = capture_new_run_dir(before)
            except RuntimeError as exc:
                print(f"!!! q3_kf_variants: {exc}", file=sys.stderr)
            else:
                captured["q3_dir"] = q3_dir
                promote(q3_dir, "q3_kf_variants.csv")

    # 4. q6 noise floor (top-level json; live). Override q6's wrong default --saccade.
    r.run(
        ["q6_noise_floor.py", *common, "--results-dir", "./results",
         "--target", Q6_TARGET],
        "q6_noise_floor",
    )

    # 5. q7 llm latency profile (timestamped; dummy model, no API) -> remember dir.
    before = snapshot_run_dirs()
    if r.run(
        ["q7_llm_latency.py", *common, "--traces-dir", TRACES_DIR,
         "--results-dir", "./results", "--reps", "10"],
        "q7_llm_latency",
    ):
        if not r.dry_run:
            try:
                captured["q7_dir"] = capture_new_run_dir(before)
            except RuntimeError as exc:
                print(f"!!! q7_llm_latency: {exc}", file=sys.stderr)

    # 6. q8 swap latency (top-level CSVs; live). --target last.
    r.run(
        ["q8_swap_latency.py", *common, "--results-dir", "./results",
         "--rounds", "10", "--target", Q8_TARGET],
        "q8_swap_latency",
    )

    # 7. q4 llm guidance (timestamped; always LLM) -> remember dir for plot.
    before = snapshot_run_dirs()
    if r.run(
        ["q4_llm_guidance.py", *common, "--traces-dir", TRACES_DIR,
         "--results-dir", "./results", "--estimator", "ema", "--llm-trials", "3",
         "--jobs", "16"],
        "q4_llm_guidance",
    ):
        if not r.dry_run:
            try:
                captured["q4_dir"] = capture_new_run_dir(before)
            except RuntimeError as exc:
                print(f"!!! q4_llm_guidance: {exc}", file=sys.stderr)

    # 8. q5 best-vs-baseline (timestamped; live real legs) -- needs the q2 grid.
    grid = captured["q2_grid"]
    if grid is None and not r.dry_run:
        r.skip("q5_best_vs_baseline", "no q2 grid CSV captured")
    else:
        grid_arg = str(grid) if grid is not None else "<q2_grid.csv>"
        before = snapshot_run_dirs()
        if r.run(
            ["q5_best_vs_baseline.py", *common, "--grid-csv", grid_arg,
             "--benchmarks-json", Q5_BENCH, "--results-dir", "./results",
             "--real-reps", "5"],
            "q5_best_vs_baseline",
        ):
            if not r.dry_run:
                try:
                    captured["q5_dir"] = capture_new_run_dir(before)
                except RuntimeError as exc:
                    print(f"!!! q5_best_vs_baseline: {exc}", file=sys.stderr)

    return captured


# --------------------------------------------------------------------------- #
# Stage B - analysis
# --------------------------------------------------------------------------- #
def analyze(r: Runner, captured: dict[str, Path | None], *, skip_collect: bool) -> None:
    # q1 family
    r.run(["analysis/q1_plot.py"], "q1_plot")
    r.run(["analysis/q1_cost_model.py"], "q1_cost_model")
    r.run(["analysis/q1_sample_collapse.py"], "q1_sample_collapse")

    # q2 family (read promoted top-level q2_scheduler_estimator.csv)
    r.run(["analysis/q2_estimator_sweep.py"], "q2_estimator_sweep")
    r.run(["analysis/q2_heatmap.py"], "q2_heatmap")

    # q3 family (reads promoted top-level q3_kf_variants.csv)
    r.run(["analysis/q3_kf_plot.py"], "q3_kf_plot")

    # q4 guidance: explicit CSV if captured, else let it glob newest.
    q4_dir = captured.get("q4_dir")
    if q4_dir is not None:
        r.run(["analysis/q4_guidance_plot.py", str(q4_dir / "q4_llm_guidance.csv")],
              "q4_guidance_plot")
    elif skip_collect:
        nd = newest_run_dir_with("q4_llm_guidance.csv")
        if nd is None:
            r.skip("q4_guidance_plot", "no results/*/q4_llm_guidance.csv found")
        else:
            r.run(["analysis/q4_guidance_plot.py", str(nd / "q4_llm_guidance.csv")],
                  "q4_guidance_plot")
    else:
        r.skip("q4_guidance_plot", "q4 collection did not produce a run dir")

    # q5 sim-vs-real: explicit run dir if captured, else glob newest.
    q5_dir = captured.get("q5_dir")
    if q5_dir is not None:
        r.run(["analysis/q5_sim_vs_real_plot.py", str(q5_dir)], "q5_sim_vs_real_plot")
    elif skip_collect:
        nd = newest_run_dir_with("q5_comparison.csv")
        if nd is None:
            r.skip("q5_sim_vs_real_plot", "no results/*/q5_comparison.csv found")
        else:
            r.run(["analysis/q5_sim_vs_real_plot.py", str(nd)], "q5_sim_vs_real_plot")
    else:
        r.skip("q5_sim_vs_real_plot", "q5 collection did not produce a run dir")

    # q7 violin: reads q7's llm_latency_profile.json (explicit if captured, else glob).
    q7_dir = captured.get("q7_dir")
    if q7_dir is not None:
        r.run(["analysis/q7_latency_violin.py", str(q7_dir / "llm_latency_profile.json")],
              "q7_latency_violin")
    elif skip_collect:
        nd = newest_run_dir_with("llm_latency_profile.json")
        if nd is None:
            r.skip("q7_latency_violin", "no results/*/llm_latency_profile.json found")
        else:
            r.run(["analysis/q7_latency_violin.py", str(nd / "llm_latency_profile.json")],
                  "q7_latency_violin")
    else:
        r.skip("q7_latency_violin", "q7 collection did not produce a run dir")

    r.run(["analysis/q8_swap_plot.py"], "q8_swap_plot")


# --------------------------------------------------------------------------- #
# Bundle - gather all data + figures (no traces) into one flat directory
# --------------------------------------------------------------------------- #
BUNDLE_EXT = {".csv", ".json", ".png"}

# Marker file -> captured-dir key, used to locate each experiment's run dir.
BUNDLE_MARKERS = {
    "q2_dir": "q2_scheduler_estimator.csv",
    "q3_dir": "q3_kf_variants.csv",
    "q4_dir": "q4_llm_guidance.csv",
    "q5_dir": "q5_comparison.csv",
    "q7_dir": "llm_latency_profile.json",
}


def bundle(captured: dict[str, Path | None], *, dry_run: bool) -> Path | None:
    """Copy every CSV/JSON/PNG (but no .perfetto traces) into one flat dir."""
    if dry_run:
        print("\n--- bundle: SKIPPED (--dry-run)")
        return None

    dest = PY_DIR / f"results_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Path] = {}

    def add(src: Path) -> None:
        if src.is_file() and src.suffix in BUNDLE_EXT and src.name not in copied:
            copied[src.name] = src

    # 1. Top-level results/ data + figures (loose .perfetto excluded by extension).
    for p in sorted(RESULTS_DIR.iterdir()):
        add(p)

    # 2. Each experiment's run dir (captured this run, else newest one with the
    #    marker). We iterate only the run dir's top level, so its traces/ subdir
    #    (perfetto + batch_spec manifests + stderr logs) is never copied.
    for key, marker in BUNDLE_MARKERS.items():
        run_dir = captured.get(key) or newest_run_dir_with(marker)
        if run_dir is not None:
            for p in sorted(run_dir.iterdir()):
                add(p)

    for name, src in sorted(copied.items()):
        shutil.copyfile(src, dest / name)

    print(f"\nBundled {len(copied)} files (CSV/JSON/PNG, no traces) -> "
          f"{dest.relative_to(PY_DIR)}/")
    return dest


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def print_summary(r: Runner, started_at: float, *, dry_run: bool) -> int:
    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    width = max((len(label) for label, _, _ in r.results), default=10)
    for label, status, detail in r.results:
        line = f"  {status:<4} {label:<{width}}"
        if detail:
            line += f"  ({detail})"
        print(line)

    if not dry_run:
        print("\nFigures in results/ (mtime relative to run start):")
        for fig in EXPECTED_FIGURES:
            # A figure may exist top-level and/or inside run dirs (q4/q5 write into
            # their run dir). Report whichever copy is freshest.
            candidates = [RESULTS_DIR / fig, *RESULTS_DIR.glob(f"*/{fig}")]
            existing = [p for p in candidates if p.is_file()]
            if not existing:
                print(f"  [ MISS] {fig}")
                continue
            path = max(existing, key=lambda p: p.stat().st_mtime)
            fresh = "fresh" if path.stat().st_mtime >= started_at else "STALE"
            rel = path.relative_to(RESULTS_DIR.parent)
            print(f"  [{fresh:>5}] {rel}")

    n_fail = sum(1 for _, s, _ in r.results if s == "FAIL")
    print(f"\n{n_fail} failure(s).")
    return 1 if n_fail else 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-collect", action="store_true",
                    help="Skip all collection; re-run analysis from existing data.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Echo commands without executing anything.")
    ap.add_argument("--fail-fast", action="store_true",
                    help="Abort on the first non-zero exit (default: continue).")
    args = ap.parse_args()

    started_at = time.time()
    r = Runner(dry_run=args.dry_run, fail_fast=args.fail_fast)

    if args.skip_collect:
        print("== STAGE A (collection) SKIPPED (--skip-collect) ==")
        captured: dict[str, Path | None] = {"q2_grid": None, "q4_dir": None,
                                            "q5_dir": None}
    else:
        print("== STAGE A: COLLECTION ==")
        captured = collect(r)

    print("\n== STAGE B: ANALYSIS ==")
    analyze(r, captured, skip_collect=args.skip_collect)

    print("\n== BUNDLE ==")
    bundle_dir = bundle(captured, dry_run=args.dry_run)

    rc = print_summary(r, started_at, dry_run=args.dry_run)
    if bundle_dir is not None:
        print(f"Bundle: {bundle_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
