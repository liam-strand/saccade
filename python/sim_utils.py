"""Shared utilities for saccade evaluation scripts.

All scripts that call `saccade simulate` or `saccade evaluate` should import
from here rather than re-implementing these helpers.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).parent.parent

SCHEDULERS = [
    "round-robin",
    "random",
    "max-uncertainty",
    "rate-of-change",
    "static-llm",
    "dynamic-llm",
    "weighted-round-robin-llm",
]
ESTIMATORS = ["propagate", "ema", "kalman"]
LLM_SCHEDULERS = frozenset({"static-llm", "dynamic-llm", "weighted-round-robin-llm"})
FIXED_NUM_SLOTS = 4


# ---------------------------------------------------------------------------
# Core subprocess wrappers
# ---------------------------------------------------------------------------


def run_simulate(
    saccade: Path,
    library: Path,
    rates_trace: Path,
    scheduler: str,
    estimator: str,
    out_trace: Path,
    num_slots: int,
    q_schedule: int,
    seed: int | None,
    base_config: Path | None = None,
    guidance: str | None = None,
) -> None:
    """Invoke `saccade simulate`, writing estimated output to *out_trace*.

    *base_config* is passed as a global ``--config`` flag so TOML values
    (e.g. ``[kalman]`` or ``[llm]``) take effect before CLI overrides.
    *guidance* appends ``--guidance <text>`` and overrides any guidance
    string in *base_config*.

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


def run_evaluate(
    saccade: Path,
    ground_truth: Path,
    estimated: Path,
) -> dict:
    """Invoke ``saccade evaluate --json`` and return the parsed result dict."""
    result = subprocess.run(
        [
            str(saccade),
            "evaluate",
            "--ground-truth", str(ground_truth),
            "--estimated", str(estimated),
            "--json",
        ],
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
    vals = [e["nrmse"] for e in eval_json["per_event"] if e["nrmse"] is not None]
    if not vals:
        return None
    return float(np.median(vals))


def mean_coverage(eval_json: dict) -> float | None:
    """Return the pre-computed mean coverage from an evaluate JSON dict."""
    return eval_json.get("mean_coverage")


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
    seed: int | None,
    tmp_dir: Path,
    workload: str,
    trial: int = 0,
    base_config: Path | None = None,
    guidance: str | None = None,
) -> dict:
    """Run one simulate + evaluate pair and return the evaluate JSON dict."""
    trace_path = (
        tmp_dir
        / f"est_{workload}_{scheduler}_{estimator}_slots{num_slots}_t{trial}.perfetto"
    )
    run_simulate(
        saccade, library, rates_trace, scheduler, estimator,
        trace_path, num_slots, q_schedule, seed, base_config, guidance,
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
    seed: int | None,
    llm_trials: int,
    tmp_dir: Path,
    workload: str,
    base_config: Path | None = None,
    guidance: str | None = None,
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
            num_slots, q_schedule, seed, tmp_dir, workload, trial,
            base_config, guidance,
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
) -> list[tuple]:
    """Run *combos* in parallel with ``ThreadPoolExecutor``.

    *run_fn(combo)* is called for each combo and must return a result.
    Non-LLM combos run with up to *jobs* threads; LLM combos are throttled
    to *llm_concurrency* to avoid overloading the Ollama server.

    *is_llm* is a predicate ``(combo) -> bool`` that identifies LLM combos.
    When omitted, any combo containing an LLM scheduler name string qualifies.

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
