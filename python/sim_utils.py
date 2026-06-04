"""Shared utilities for saccade evaluation scripts.

All scripts that call `saccade simulate` or `saccade evaluate` should import
from here rather than re-implementing these helpers.
"""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
# Default LLM latency profile for `saccade simulate`. Scripts expose this as
# the --llm-latency-profile default; run_all.py overrides it with a freshly
# collected q7 profile.
LATENCY_PROFILE = Path(__file__).parent / "gemma4_8b_latency.json"

SCHEDULERS = [
    "round-robin",
    "random",
    "max-uncertainty",
    "rate-of-change",
    "static-llm",
    "dynamic-llm",
    "weighted-round-robin-llm",
    "reasoning-static-llm",
    "reasoning-dynamic-llm",
]
ESTIMATORS = ["propagate", "ema", "kalman"]
LLM_SCHEDULERS = frozenset({
    "static-llm",
    "dynamic-llm",
    "weighted-round-robin-llm",
    "reasoning-static-llm",
    "reasoning-dynamic-llm",
})
FIXED_NUM_SLOTS = 6


# ---------------------------------------------------------------------------
# Core subprocess wrappers
# ---------------------------------------------------------------------------


def _read_tail(path: Path, n_lines: int = 40) -> str:
    """Return the last *n_lines* of a text file, or an explanatory note on failure."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    tail = lines[-n_lines:]
    prefix = f"(last {len(tail)} of {len(lines)} lines; full log: {path})\n"
    return prefix + "\n".join(tail)


def run_simulate(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    estimator: str,
    out_trace: Path,
    num_slots: int,
    q_schedule: int,
    llm_model: str,
    seed: int | None,
    base_config: Path | None = None,
    guidance: str | None = None,
    latency_profile: Path = LATENCY_PROFILE,
) -> None:
    """Invoke `saccade simulate`, writing estimated output to *out_trace*.

    *base_config* is passed as a global ``--config`` flag so TOML values
    (e.g. ``[kalman]`` or ``[llm]``) take effect before CLI overrides.
    *guidance* appends ``--guidance <text>`` and overrides any guidance
    string in *base_config*.
    *latency_profile* is forwarded as ``--llm-latency-profile``; defaults to
    the saved profile (``LATENCY_PROFILE``).

    All paths should be absolute.  The subprocess cwd is set to the repo
    root so that relative ``correlation_path`` values in TOML configs
    (e.g. ``python/correlation.json``) resolve correctly.
    """
    cmd = [str(saccade)]
    if base_config is not None:
        cmd += ["--config", str(base_config)]
    cmd += [
        "simulate",
        "--library", str(library),
        "--rates-trace", str(rates_trace),
        "--scheduler", scheduler,
        "--estimator", estimator,
        "--trace", str(out_trace),
        "--num-slots", str(num_slots),
        "--q-schedule", str(q_schedule),
        "--llm-model", llm_model,
        "--llm-base-url", "https://openrouter.ai/api",
        "--llm-api-key", os.environ["LLM_API_KEY"],
        "--llm-latency-profile", str(latency_profile),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if guidance is not None:
        cmd += ["--guidance", guidance]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=REPO_ROOT,
    )


def build_batch_simulate_cmd(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    spec_path: Path,
    num_slots: int,
    q_schedule: int,
    jobs: int,
    llm_model: str,
    base_config: Path | None = None,
    api_key: str | None = None,
    latency_profile: Path = LATENCY_PROFILE,
) -> list[str]:
    """Build the ``saccade simulate --batch`` argv list.

    *api_key* defaults to the ``LLM_API_KEY`` environment variable; pass an
    explicit placeholder (e.g. ``"$LLM_API_KEY"``) to avoid reading/leaking the
    real secret when only printing the command (dry runs).
    """
    if api_key is None:
        api_key = os.environ["LLM_API_KEY"]
    cmd = [str(saccade)]
    if base_config is not None:
        cmd += ["--config", str(base_config)]
    cmd += [
        "simulate",
        "--library", str(library),
        "--rates-trace", str(rates_trace),
        "--num-slots", str(num_slots),
        "--q-schedule", str(q_schedule),
        "--batch", str(spec_path),
        "--jobs", str(max(1, jobs)),
        "--llm-model", llm_model,
        "--llm-base-url", "https://openrouter.ai/api",
        "--llm-api-key", api_key,
        "--llm-latency-profile", str(latency_profile),
    ]
    return cmd


def run_batch_simulate(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    combos: list[dict],
    num_slots: int,
    q_schedule: int,
    jobs: int,
    tmp_dir: Path,
    llm_model: str,
    base_config: Path | None = None,
    spec_tag: str = "",
    latency_profile: Path = LATENCY_PROFILE,
) -> None:
    """Run multiple simulate combos sharing one ``saccade simulate --batch`` process.

    Each entry in *combos* must have ``"scheduler"``, ``"estimator"``, and
    ``"trace"`` keys (absolute paths).  Optional keys: ``"seed"`` (int),
    ``"guidance"`` (str), ``"csv"`` (str path).

    Loads *rates_trace* once in saccade and runs all combos in a Rayon thread
    pool capped at *jobs* threads.  Use this instead of N calls to
    ``run_simulate`` to avoid N× memory usage when replaying the same trace.

    *base_config* is forwarded as the global ``--config`` flag; per-combo
    scheduler/estimator/seed/guidance override those values.

    *spec_tag* disambiguates the per-call spec/log filenames.  Pass a unique
    tag (e.g. a trial index) when launching several batches for the *same*
    rates_trace concurrently, so their ``batch_spec_*.json`` / ``*_stderr.log``
    files don't collide.  Defaults to "" (filename keyed by trace stem only).

    saccade's stderr is streamed to ``<tmp_dir>/batch_<workload>[_<tag>]_stderr.log``
    so failures are diagnosable.  On a non-zero exit a ``CalledProcessError`` is
    raised with the tail of that log attached as ``.stderr``.
    """
    suffix = f"_{spec_tag}" if spec_tag else ""
    spec_path = tmp_dir / f"batch_spec_{rates_trace.stem}{suffix}.json"
    spec_path.write_text(json.dumps(combos))
    cmd = build_batch_simulate_cmd(
        saccade, library, rates_trace, spec_path,
        num_slots, q_schedule, jobs, llm_model, base_config,
        latency_profile=latency_profile,
    )
    log_path = tmp_dir / f"batch_{rates_trace.stem}{suffix}_stderr.log"
    with log_path.open("w") as logf:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=logf,
            cwd=REPO_ROOT,
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=None, stderr=_read_tail(log_path),
        )


def build_evaluate_cmd(
    saccade: Path,
    ground_truth: Path,
    estimated: Path,
) -> list[str]:
    """Build the ``saccade evaluate --json`` argv list."""
    return [
        str(saccade),
        "evaluate",
        "--ground-truth", str(ground_truth),
        "--estimated", str(estimated),
        "--json",
    ]


def run_evaluate(
    saccade: Path,
    ground_truth: Path,
    estimated: Path,
) -> dict:
    """Invoke ``saccade evaluate --json`` and return the parsed result dict."""
    result = subprocess.run(
        build_evaluate_cmd(saccade, ground_truth, estimated),
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def median_nrmse(eval_json: dict) -> float | None:
    """Median per-event nRMSE, filtering null entries (zero GT mean)."""
    vals = [e["nrmse"] for e in eval_json.get("per_event", []) if e["nrmse"] is not None]
    if not vals:
        return None
    return float(np.median(vals))


def mean_coverage(eval_json: dict) -> float | None:
    """Return the pre-computed mean coverage from an evaluate JSON dict."""
    return eval_json.get("mean_coverage")


def nrmse_distribution(eval_json: dict) -> dict:
    """Distribution of per-event nRMSE values (mean, p50, p90, max).

    Surfaces the tail of hard-to-reconstruct events that the median alone hides.
    Returns ``{"mean": ..., "p50": ..., "p90": ..., "max": ...}``, with all
    values set to None if there are no non-null nRMSE entries.
    """
    vals = [e["nrmse"] for e in eval_json.get("per_event", []) if e["nrmse"] is not None]
    if not vals:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def importance_weighted_nrmse(eval_json: dict) -> float | None:
    """Per-event nRMSE weighted by each event's gt_cv (coefficient of variation).

    The weight is the *measured* GT signal variability (stddev/mean of the
    ground-truth rate series), not a hand-picked importance label.  Events
    with a high gt_cv are genuinely harder targets, so their reconstruction
    error matters more.

    Events with null nrmse or null/zero gt_cv are skipped entirely.

    This is a SECONDARY diagnostic lens — it is NOT the frozen primary metric.
    Use ``median_nrmse`` for head-to-head comparisons across runs.
    """
    weights = []
    weighted_nrmse = []
    for e in eval_json.get("per_event", []):
        nrmse = e.get("nrmse")
        gt_cv = e.get("gt_cv")
        if nrmse is None or gt_cv is None or gt_cv == 0.0:
            continue
        weights.append(gt_cv)
        weighted_nrmse.append(nrmse * gt_cv)
    if not weights:
        return None
    total_weight = sum(weights)
    return float(sum(weighted_nrmse) / total_weight)


def mean_calibration(eval_json: dict) -> float | None:
    """Return the pre-computed mean calibration from an evaluate JSON dict.

    Calibration is the GT-anchored fraction-in-band: fraction of time the
    true rate falls within the estimator's uncertainty interval.  Returns None
    for older traces that have no uncertainty track.
    """
    return eval_json.get("mean_calibration")


def load_noise_floor(path: Path | None) -> float | None:
    """Load noise-floor threshold from a JSON file.

    Accepts both schemas:
    - q2-style: ``{"nrmse_floor": <float>}``
    - q6-style: ``{"median_nrmse": <float>, ...}``
    """
    if path is None:
        return None
    data = json.loads(path.read_text())
    val = data.get("nrmse_floor") or data.get("median_nrmse")
    return float(val) if val is not None else None


def is_significant(nrmse: float | None, floor: float | None) -> bool | None:
    """True if *nrmse* exceeds *floor*; None if either is missing."""
    if floor is None or nrmse is None:
        return None
    return nrmse > floor


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def simulate_and_eval(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    estimator: str,
    num_slots: int,
    q_schedule: int,
    llm_model: str,
    seed: int | None,
    tmp_dir: Path,
    workload: str,
    trial: int = 0,
    base_config: Path | None = None,
    guidance: str | None = None,
    latency_profile: Path = LATENCY_PROFILE,
) -> dict:
    """Run one simulate + evaluate pair and return the evaluate JSON dict."""
    trace_path = (
        tmp_dir
        / f"est_{workload}_{scheduler}_{estimator}_slots{num_slots}_t{trial}.perfetto"
    )
    run_simulate(
        saccade, library, rates_trace, scheduler, estimator,
        trace_path, num_slots, q_schedule, llm_model, seed, base_config, guidance,
        latency_profile=latency_profile,
    )
    return run_evaluate(saccade, rates_trace, trace_path)


def run_combo(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    estimator: str,
    num_slots: int,
    q_schedule: int,
    llm_model: str,
    seed: int | None,
    llm_trials: int,
    tmp_dir: Path,
    workload: str,
    base_config: Path | None = None,
    guidance: str | None = None,
    latency_profile: Path = LATENCY_PROFILE,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Simulate + evaluate one (scheduler, estimator) combo, averaging LLM runs.

    LLM schedulers are run *llm_trials* times; all others run once.
    Returns ``(median_nrmse, mean_coverage, nrmse_mean, nrmse_stddev)``.
    ``nrmse_mean`` and ``nrmse_stddev`` are populated only for multi-trial runs.
    """
    n_trials = llm_trials if scheduler in LLM_SCHEDULERS else 1
    nrmse_vals: list[float] = []
    coverage_vals: list[float] = []

    for trial in range(n_trials):
        eval_json = simulate_and_eval(
            saccade, library, rates_trace, scheduler, estimator,
            num_slots, q_schedule, llm_model, seed, tmp_dir, workload, trial,
            base_config, guidance, latency_profile=latency_profile,
        )
        mn = median_nrmse(eval_json)
        mc = mean_coverage(eval_json)
        if mn is not None:
            nrmse_vals.append(mn)
        if mc is not None:
            coverage_vals.append(mc)

    final_nrmse = float(np.median(nrmse_vals)) if nrmse_vals else None
    final_cov = float(np.mean(coverage_vals)) if coverage_vals else None

    if n_trials > 1 and nrmse_vals:
        nrmse_mean = float(np.mean(nrmse_vals))
        nrmse_std = float(np.std(nrmse_vals, ddof=1)) if len(nrmse_vals) > 1 else 0.0
    else:
        nrmse_mean = final_nrmse
        nrmse_std = None

    return final_nrmse, final_cov, nrmse_mean, nrmse_std


def run_combo_extended(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    estimator: str,
    num_slots: int,
    q_schedule: int,
    llm_model: str,
    seed: int | None,
    llm_trials: int,
    tmp_dir: Path,
    workload: str,
    base_config: Path | None = None,
    guidance: str | None = None,
    latency_profile: Path = LATENCY_PROFILE,
) -> dict:
    """Like ``run_combo`` but returns a dict with all primary and secondary metrics.

    LLM schedulers are run *llm_trials* times; all others run once.  The
    returned dict includes:
      - ``median_nrmse``: frozen primary metric
      - ``coverage``: mean coverage across trials
      - ``nrmse_mean``, ``nrmse_stddev``: trial-variability stats (LLM only)
      - ``nrmse_p90``, ``nrmse_max``: tail of per-event distribution, median'd across trials
      - ``nrmse_weighted``: importance-weighted nRMSE (secondary lens), mean'd across trials
      - ``mean_calibration``: GT-anchored fraction-in-band, mean'd across trials (may be None)

    All values are None when the underlying call produced no usable data.
    """
    n_trials = llm_trials if scheduler in LLM_SCHEDULERS else 1
    nrmse_vals: list[float] = []
    coverage_vals: list[float] = []
    p90_vals: list[float] = []
    max_vals: list[float] = []
    wt_vals: list[float] = []
    cal_vals: list[float] = []

    for trial in range(n_trials):
        eval_json = simulate_and_eval(
            saccade, library, rates_trace, scheduler, estimator,
            num_slots, q_schedule, llm_model, seed, tmp_dir, workload, trial,
            base_config, guidance, latency_profile=latency_profile,
        )
        mn = median_nrmse(eval_json)
        mc = mean_coverage(eval_json)
        dist = nrmse_distribution(eval_json)
        wt = importance_weighted_nrmse(eval_json)
        cal = mean_calibration(eval_json)

        if mn is not None:
            nrmse_vals.append(mn)
        if mc is not None:
            coverage_vals.append(mc)
        if dist["p90"] is not None:
            p90_vals.append(dist["p90"])
        if dist["max"] is not None:
            max_vals.append(dist["max"])
        if wt is not None:
            wt_vals.append(wt)
        if cal is not None:
            cal_vals.append(cal)

    final_nrmse = float(np.median(nrmse_vals)) if nrmse_vals else None
    final_cov = float(np.mean(coverage_vals)) if coverage_vals else None

    if n_trials > 1 and nrmse_vals:
        nrmse_mean = float(np.mean(nrmse_vals))
        nrmse_std = float(np.std(nrmse_vals, ddof=1)) if len(nrmse_vals) > 1 else 0.0
    else:
        nrmse_mean = final_nrmse
        nrmse_std = None

    return {
        "median_nrmse": final_nrmse,
        "coverage": final_cov,
        "nrmse_mean": nrmse_mean,
        "nrmse_stddev": nrmse_std,
        "nrmse_p90": float(np.median(p90_vals)) if p90_vals else None,
        "nrmse_max": float(np.median(max_vals)) if max_vals else None,
        "nrmse_weighted": float(np.mean(wt_vals)) if wt_vals else None,
        "mean_calibration": float(np.mean(cal_vals)) if cal_vals else None,
    }


# ---------------------------------------------------------------------------
# Trace filtering
# ---------------------------------------------------------------------------


def filter_traces_by_kind(traces: list[Path], kind: str) -> list[Path]:
    """Filter a list of trace paths by workload kind.

    *kind* is ``"spec"`` (keep ``spec_*``), ``"npb"`` (keep ``npb_*``),
    or ``"all"`` (no filtering).
    """
    if kind == "all":
        return traces
    prefix = kind + "_"
    return [t for t in traces if t.stem.startswith(prefix)]


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def parallel_run_combos(
    combos: list,
    run_fn: Callable,
    jobs: int,
    llm_concurrency: int = 1,
    is_llm: Callable | None = None,
    batch_fn: Callable | None = None,
) -> list[tuple]:
    """Run *combos* in parallel with ``ThreadPoolExecutor``.

    *run_fn(combo)* is called for each combo and must return a result.
    Non-LLM combos run with up to *jobs* threads; LLM combos are throttled
    to *llm_concurrency* to avoid overloading the Ollama server.

    *is_llm* is a predicate ``(combo) -> bool`` that identifies LLM combos.
    When omitted, any combo containing an LLM scheduler name string qualifies.

    *batch_fn*, when provided, is called ONCE with the full list of non-LLM
    combos before individual ``run_fn`` calls.  Use this to run a single
    ``saccade simulate --batch`` invocation that loads the rates trace once and
    runs all combos in parallel threads, instead of N subprocesses each loading
    their own copy.  After *batch_fn* returns, ``run_fn`` is still called per
    combo to run evaluation (``saccade evaluate``) and return the result.

    Returns a list of ``(combo, result)`` pairs in completion order (not
    necessarily the same order as *combos*).
    """
    if is_llm is None:
        is_llm = lambda c: any(  # noqa: E731
            isinstance(x, str) and x in LLM_SCHEDULERS for x in c
        )

    llm_combos = [c for c in combos if is_llm(c)]
    non_llm = [c for c in combos if not is_llm(c)]

    results: list[tuple] = []

    if non_llm:
        if batch_fn is not None:
            # Run all simulate calls in one saccade process (1× trace memory),
            # then evaluate each output individually in parallel.
            batch_fn(non_llm)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = {ex.submit(run_fn, c): c for c in non_llm}
            for f in as_completed(futs):
                results.append((futs[f], f.result()))

    if llm_combos:
        with ThreadPoolExecutor(max_workers=max(1, llm_concurrency)) as ex:
            futs = {ex.submit(run_fn, c): c for c in llm_combos}
            for f in as_completed(futs):
                results.append((futs[f], f.result()))

    return results
