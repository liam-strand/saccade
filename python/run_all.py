#!/usr/bin/env python3
"""Master reproduction orchestrator for the saccade evaluation.

Runs every data-collection script (q1-q8; collect2.py's ground-truth sweep is
skipped, we reuse the existing ./sweep_data_eval_traces/) and then every analysis
script, with the correct arguments, to regenerate all figures in ./results/.

Run from the python/ directory:

    uv run python run_all.py                # full collection + analysis (LLM live)
    uv run python run_all.py --skip-collect # re-plot only, from existing data
    uv run python run_all.py --from q2      # resume collection at q2 (skip q1/q7)
    uv run python run_all.py --steps q4,q5  # run only these collectors
    uv run python run_all.py --dry-run      # echo commands, run nothing

The tricky bits this handles automatically:
  * Promotion - q2/q3 collection write their CSV into a fresh results/<timestamp>/
    dir, but the q2/q3 plot scripts read the *top-level* results/*.csv. We copy the
    timestamped CSV up after each of those collectors runs.
  * Dependency passing - q5 collection needs the q2 grid CSV via --grid-csv, and the
    q4/q5 plots take the captured run dir explicitly for determinism.
  * Latency profile - q7 runs before the simulation collectors (q2/q3/q4/q5) and its
    fresh llm_latency_profile.json is forwarded to them via --llm-latency-profile.
    Pass --use-saved-latency-profile to skip q7 collection and let the simulations
    use the saved default profile from sim_utils instead.
  * Step selection - --steps/--from gate individual collectors. Deselected steps
    resolve any outputs needed downstream (q7 latency profile, q2 grid, q4/q5/q7
    run dirs) from the newest existing results/<timestamp>/ data, so later
    collectors and the plots still work after an interrupted run.
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
Q6_TARGET = f"{NPB_BIN}/ep.A.x"
Q8_TARGET = f"{NPB_BIN}/cg.B.x"

TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")

# Collection steps in execution order (the order collect() runs them), for
# --steps/--from. q7 runs early so the simulations can replay its profile.
COLLECT_STEPS = ["q1", "q7", "q2", "q3", "q6", "q8", "q4", "q5"]

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
    "q4_guidance.png",
    "q5_sim_vs_real.png",
    "q7_llm_latency_violin.png",
    "q8_swap_breakdown.png",
    "q8_swap_drift.png",
    "q8_swap_quantum.png",
]


# --------------------------------------------------------------------------- #
# Time formatting helpers
# --------------------------------------------------------------------------- #
def _clock() -> str:
    """Wall-clock time-of-day, e.g. 15:48:40."""
    return datetime.now().strftime("%H:%M:%S")


def _dur(seconds: float) -> str:
    """Human-friendly elapsed time, e.g. 4.2s or 3m12s or 1h05m."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #
class Runner:
    """Tracks pass/fail/skip across stages and drives subprocesses."""

    def __init__(self, *, dry_run: bool, fail_fast: bool) -> None:
        self.dry_run = dry_run
        self.fail_fast = fail_fast
        # (label, status, detail, duration_seconds | None)
        self.results: list[tuple[str, str, str, float | None]] = []

    def record(self, label: str, status: str, detail: str = "",
               duration: float | None = None) -> None:
        self.results.append((label, status, detail, duration))

    def run(self, cmd: list[str], label: str) -> bool:
        """Run `uv run python <cmd>` in PY_DIR. Returns True on success."""
        printable = " ".join(str(c) for c in cmd)
        print(f"\n>>> [{_clock()}] [{label}] uv run python {printable}", flush=True)
        if self.dry_run:
            self.record(label, "DRY")
            return True
        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["uv", "run", "python", *[str(c) for c in cmd]],
                cwd=PY_DIR,
            )
        except Exception as exc:  # noqa: BLE001 - surface any spawn failure
            elapsed = time.monotonic() - start
            print(f"!!! [{_clock()}] [{label}] failed to launch: {exc} "
                  f"({_dur(elapsed)})", file=sys.stderr)
            self.record(label, "FAIL", str(exc), elapsed)
            if self.fail_fast:
                raise SystemExit(f"--fail-fast: aborting after {label}")
            return False
        elapsed = time.monotonic() - start
        if proc.returncode != 0:
            print(f"!!! [{_clock()}] [{label}] exited {proc.returncode} "
                  f"({_dur(elapsed)})", file=sys.stderr)
            self.record(label, "FAIL", f"exit {proc.returncode}", elapsed)
            if self.fail_fast:
                raise SystemExit(f"--fail-fast: aborting after {label}")
            return False
        print(f"<<< [{_clock()}] [{label}] done in {_dur(elapsed)}", flush=True)
        self.record(label, "OK", "", elapsed)
        return True

    def skip(self, label: str, reason: str) -> None:
        print(f"--- [{_clock()}] [{label}] SKIPPED: {reason}", flush=True)
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


def newest_data_dir_with(filename: str) -> Path | None:
    """Like newest_run_dir_with, but falls back to the top-level results/ dir
    itself if `filename` only exists there (pre-timestamp layout / hand-promoted
    data). Used by --steps/--from to resolve deselected steps' outputs."""
    nd = newest_run_dir_with(filename)
    if nd is not None:
        return nd
    return RESULTS_DIR if (RESULTS_DIR / filename).is_file() else None


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
def collect(r: Runner, *, use_saved_latency: bool,
            steps: set[str]) -> dict[str, Path | None]:
    """Run the selected collectors. Returns captured run dirs / grid path.

    Collectors not in `steps` are skipped; outputs of theirs that later steps
    or plots need (q7 latency profile, q2 grid, q4/q5 run dirs) are resolved
    from the newest existing results/<timestamp>/ data instead.
    """
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
    if "q1" in steps:
        r.run(
            ["q1_overhead.py", *common, "--results-dir", "./results",
             "--warmup", "5", "--rounds", "20", "--target", Q1_TARGET],
            "q1_overhead",
        )
    else:
        r.skip("q1_overhead", "deselected by --steps/--from")

    # 2. q7 llm latency profile (timestamped; dummy model, no API). Runs before
    #    the simulation collectors so they can replay its fresh latency profile,
    #    forwarded below via --llm-latency-profile.
    profile_args: list[str] = []
    if use_saved_latency:
        r.skip("q7_llm_latency",
               "--use-saved-latency-profile: simulations use the saved default")
    elif "q7" not in steps:
        q7_dir = newest_data_dir_with("llm_latency_profile.json")
        if q7_dir is None:
            r.skip("q7_llm_latency", "deselected; no existing profile found, "
                   "simulations use the saved default")
        else:
            r.skip("q7_llm_latency", "deselected; reusing "
                   f"{q7_dir.name}/llm_latency_profile.json")
            captured["q7_dir"] = q7_dir
            profile_args = ["--llm-latency-profile",
                            str(q7_dir / "llm_latency_profile.json")]
    else:
        before = snapshot_run_dirs()
        if r.run(
            ["q7_llm_latency.py", *common, "--traces-dir", TRACES_DIR,
             "--results-dir", "./results", "--reps", "10"],
            "q7_llm_latency",
        ):
            if not r.dry_run:
                try:
                    q7_dir = capture_new_run_dir(before)
                except RuntimeError as exc:
                    print(f"!!! q7_llm_latency: {exc}", file=sys.stderr)
                else:
                    captured["q7_dir"] = q7_dir
                    profile = q7_dir / "llm_latency_profile.json"
                    if profile.is_file():
                        profile_args = ["--llm-latency-profile", str(profile)]
                        print(f"    simulations will use fresh latency profile: "
                              f"{profile.relative_to(PY_DIR)}")
                    else:
                        print(f"!!! q7_llm_latency: {profile} missing; simulations "
                              "fall back to the saved profile", file=sys.stderr)

    # 3. q2 accuracy (timestamped; LLM live) -> promote + remember grid for q5.
    if "q2" not in steps:
        q2_dir = newest_data_dir_with("q2_scheduler_estimator.csv")
        if q2_dir is None:
            r.skip("q2_accuracy", "deselected; no existing q2 grid found")
        else:
            r.skip("q2_accuracy", "deselected; reusing "
                   f"{q2_dir.name}/q2_scheduler_estimator.csv")
            captured["q2_dir"] = q2_dir
            captured["q2_grid"] = q2_dir / "q2_scheduler_estimator.csv"
    else:
        before = snapshot_run_dirs()
        if r.run(
            ["q2_accuracy.py", *common, "--traces-dir", TRACES_DIR,
             "--results-dir", "./results", "--jobs", "16", *profile_args],
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

    # 4. q3 kalman variants (timestamped) -> promote.
    if "q3" not in steps:
        r.skip("q3_kf_variants", "deselected by --steps/--from")
    else:
        before = snapshot_run_dirs()
        if r.run(
            ["q3_kf_variants.py", *common, "--traces-dir", TRACES_DIR,
             "--results-dir", "./results", "--jobs", "16", *profile_args],
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

    # 5. q6 noise floor (top-level json; live). Override q6's wrong default --saccade.
    if "q6" in steps:
        r.run(
            ["q6_noise_floor.py", *common, "--results-dir", "./results",
             "--target", Q6_TARGET],
            "q6_noise_floor",
        )
    else:
        r.skip("q6_noise_floor", "deselected by --steps/--from")

    # 6. q8 swap latency (top-level CSVs; live). --target last.
    if "q8" in steps:
        r.run(
            ["q8_swap_latency.py", *common, "--results-dir", "./results",
             "--rounds", "10", "--target", Q8_TARGET],
            "q8_swap_latency",
        )
    else:
        r.skip("q8_swap_latency", "deselected by --steps/--from")

    # 7. q4 llm guidance (timestamped; always LLM) -> remember dir for plot.
    if "q4" not in steps:
        q4_dir = newest_data_dir_with("q4_llm_guidance.csv")
        captured["q4_dir"] = q4_dir
        r.skip("q4_llm_guidance", "deselected; "
               + (f"plot reuses {q4_dir.name}/" if q4_dir is not None
                  else "no existing run dir for the plot"))
    else:
        before = snapshot_run_dirs()
        if r.run(
            ["q4_llm_guidance.py", *common, "--traces-dir", TRACES_DIR,
             "--results-dir", "./results", "--estimator", "ema", "--llm-trials", "3",
             "--jobs", "16", *profile_args],
            "q4_llm_guidance",
        ):
            if not r.dry_run:
                try:
                    captured["q4_dir"] = capture_new_run_dir(before)
                except RuntimeError as exc:
                    print(f"!!! q4_llm_guidance: {exc}", file=sys.stderr)

    # 8. q5 best-vs-baseline (timestamped; live real legs) -- needs the q2 grid.
    grid = captured["q2_grid"]
    if "q5" not in steps:
        q5_dir = newest_data_dir_with("q5_comparison.csv")
        captured["q5_dir"] = q5_dir
        r.skip("q5_best_vs_baseline", "deselected; "
               + (f"plot reuses {q5_dir.name}/" if q5_dir is not None
                  else "no existing run dir for the plot"))
    elif grid is None and not r.dry_run:
        r.skip("q5_best_vs_baseline", "no q2 grid CSV captured")
    else:
        grid_arg = str(grid) if grid is not None else "<q2_grid.csv>"
        before = snapshot_run_dirs()
        if r.run(
            ["q5_best_vs_baseline.py", *common, "--grid-csv", grid_arg,
             "--benchmarks-json", Q5_BENCH, "--results-dir", "./results",
             "--real-reps", "5", *profile_args],
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
def analyze(r: Runner, captured: dict[str, Path | None], *, skip_collect: bool,
            use_saved_latency: bool = False) -> None:
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
        nd = newest_data_dir_with("q4_llm_guidance.csv")
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
        nd = newest_data_dir_with("q5_comparison.csv")
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
    elif skip_collect or use_saved_latency:
        nd = newest_data_dir_with("llm_latency_profile.json")
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
    width = max((len(label) for label, _, _, _ in r.results), default=10)
    for label, status, detail, duration in r.results:
        dur = _dur(duration) if duration is not None else ""
        line = f"  {status:<4} {label:<{width}}  {dur:>7}"
        if detail:
            line += f"  ({detail})"
        print(line)

    if not dry_run:
        print("\nFigures in results/ (mtime relative to run start):")
        for fig in EXPECTED_FIGURES:
            # All analysis scripts write top-level now, but figures from older
            # runs may still sit inside run dirs. Report whichever is freshest.
            candidates = [RESULTS_DIR / fig, *RESULTS_DIR.glob(f"*/{fig}")]
            existing = [p for p in candidates if p.is_file()]
            if not existing:
                print(f"  [ MISS] {fig}")
                continue
            path = max(existing, key=lambda p: p.stat().st_mtime)
            fresh = "fresh" if path.stat().st_mtime >= started_at else "STALE"
            rel = path.relative_to(RESULTS_DIR.parent)
            print(f"  [{fresh:>5}] {rel}")

    n_fail = sum(1 for _, s, _, _ in r.results if s == "FAIL")
    total = sum(d for _, _, _, d in r.results if d is not None)
    print(f"\n{n_fail} failure(s). Total step time: {_dur(total)} "
          f"(wall clock: {_dur(time.time() - started_at)}).")
    return 1 if n_fail else 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Runs everything! (except for saccade sweep data trace collection)")
    ap.add_argument("--skip-collect", action="store_true",
                    help="Skip all collection; re-run analysis from existing data.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Echo commands without executing anything.")
    ap.add_argument("--fail-fast", action="store_true",
                    help="Abort on the first non-zero exit (default: continue).")
    ap.add_argument("--use-saved-latency-profile", action="store_true",
                    help="Skip q7 latency collection; simulations use the saved "
                         "default profile from sim_utils instead of a fresh one.")
    ap.add_argument("--steps", metavar="STEP[,STEP...]",
                    help="Only run these collection steps (comma-separated). "
                         f"Choices: {', '.join(COLLECT_STEPS)}. Deselected steps "
                         "reuse the newest existing results data.")
    ap.add_argument("--from", dest="from_step", metavar="STEP",
                    help="Resume collection from this step onward, in execution "
                         f"order ({' -> '.join(COLLECT_STEPS)}); earlier steps "
                         "reuse the newest existing results data.")
    args = ap.parse_args()

    steps = set(COLLECT_STEPS)
    if args.steps and args.from_step:
        ap.error("--steps and --from are mutually exclusive")
    if args.skip_collect and (args.steps or args.from_step):
        ap.error("--skip-collect cannot be combined with --steps/--from")
    if args.steps:
        steps = {s.strip() for s in args.steps.split(",") if s.strip()}
        unknown = steps - set(COLLECT_STEPS)
        if unknown:
            ap.error(f"--steps: unknown step(s) {', '.join(sorted(unknown))} "
                     f"(choices: {', '.join(COLLECT_STEPS)})")
        if not steps:
            ap.error("--steps: empty selection")
    elif args.from_step:
        if args.from_step not in COLLECT_STEPS:
            ap.error(f"--from: unknown step {args.from_step!r} "
                     f"(choices: {', '.join(COLLECT_STEPS)})")
        steps = set(COLLECT_STEPS[COLLECT_STEPS.index(args.from_step):])

    started_at = time.time()
    r = Runner(dry_run=args.dry_run, fail_fast=args.fail_fast)

    if args.skip_collect:
        print("== STAGE A (collection) SKIPPED (--skip-collect) ==")
        captured: dict[str, Path | None] = {"q2_grid": None, "q4_dir": None,
                                            "q5_dir": None}
    else:
        print("== STAGE A: COLLECTION ==")
        if steps != set(COLLECT_STEPS):
            selected = [s for s in COLLECT_STEPS if s in steps]
            print(f"   (steps: {', '.join(selected)}; "
                  "deselected steps reuse existing data)")
        captured = collect(r, use_saved_latency=args.use_saved_latency_profile,
                           steps=steps)

    print("\n== STAGE B: ANALYSIS ==")
    analyze(r, captured, skip_collect=args.skip_collect,
            use_saved_latency=args.use_saved_latency_profile)

    print("\n== BUNDLE ==")
    bundle_dir = bundle(captured, dry_run=args.dry_run)

    rc = print_summary(r, started_at, dry_run=args.dry_run)
    if bundle_dir is not None:
        print(f"Bundle: {bundle_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
