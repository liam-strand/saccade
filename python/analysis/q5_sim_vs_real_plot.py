"""Q5 simulation vs. real-hardware vs. perf-stat comparison figure.

The q5 story has two coupled axes that must be read together: estimator
accuracy (median nRMSE) and how much of the ground-truth timeline the trace
actually covers (coverage).  Reading nRMSE alone is misleading -- the live
`saccade run` legs collapse to ~7-16% coverage because counter reprogramming
costs ~60ms/quantum (see q8), so their nRMSE is scored over only ~1/5 of the
GT time bins and conflates estimator error with this resolution gap.

So the figure is two stacked panels sharing the workload x-axis:

  top     median nRMSE (log y; lower is better).  The noise floor is drawn as a
          reference line.  Real legs carry p25/p75 IQR whiskers from the per-rep
          raw CSV; sim and perf_stat legs are single deterministic points.
  bottom  coverage (fraction of GT time bins with an estimate).  This is the
          panel that explains the top one: ~1.0 for every simulated leg,
          ~0.1 for every live/perf_stat leg.

Five legs per workload, color = config, hatch = ran on real hardware:
  best (sim) / best_real     -- best (scheduler, estimator) from the q2 grid
  baseline (sim) / baseline_real -- round-robin + propagate
  perf_stat                  -- kernel-native counter multiplexing

Reads a q5_best_vs_baseline.py run directory:
  <run>/q5_comparison.csv     sim + perf_stat + aggregated real rows
  <run>/q5_real_runs_raw.csv  per-rep real samples (for IQR whiskers)

Usage:
  python analysis/q5_sim_vs_real_plot.py [run_dir]
If run_dir is omitted, the newest results/<timestamp>/ holding a non-empty
q5_comparison.csv is used.
"""

import csv
import sys
from pathlib import Path

import plot_style as ps

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ps.apply_style(14)

RESULTS_DIR = ps.RESULTS_DIR
NOISE_FLOOR_JSON = RESULTS_DIR / "noise_floor.json"
OUT = ps.out("q5_sim_vs_real.png")

# Leg draw order, with display label, bar color, and whether it ran on real hw.
# color groups the config family (best=blue, baseline=orange, perf_stat=gray);
# the hatch (applied to real legs) is what separates sim from live.
LEGS = [
    ("best",          "best\n(sim)",      ps.Q5_LEGS["best"],      False),
    ("best_real",     "best\n(real)",     ps.Q5_LEGS["best"],      True),
    ("baseline",      "baseline\n(sim)",  ps.Q5_LEGS["baseline"],  False),
    ("baseline_real", "baseline\n(real)", ps.Q5_LEGS["baseline"],  True),
    ("perf_stat",     "perf stat",        ps.Q5_LEGS["perf_stat"], True),
]
REAL_HATCH = ps.REAL_HATCH


def find_run_dir(argv: list[str]) -> Path:
    if len(argv) > 1:
        d = Path(argv[1])
        if not (d / "q5_comparison.csv").exists():
            raise SystemExit(f"{d} has no q5_comparison.csv")
        return d
    candidates = []
    for csv_path in RESULTS_DIR.glob("*/q5_comparison.csv"):
        with csv_path.open(newline="") as f:
            n = sum(1 for _ in csv.DictReader(f))
        if n:
            candidates.append(csv_path.parent)
    if not candidates:
        raise SystemExit(
            "no results/<timestamp>/q5_comparison.csv with data; "
            "run q5_best_vs_baseline.py first"
        )
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_comparison(run_dir: Path) -> tuple[list[str], dict]:
    """Return (workloads, {(workload, config_label): row})."""
    workloads: list[str] = []
    table: dict = {}
    with (run_dir / "q5_comparison.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            wl = r["workload"]
            if wl not in workloads:
                workloads.append(wl)
            table[(wl, r["config_label"])] = r
    workloads.sort()
    return workloads, table


def load_raw(run_dir: Path) -> dict:
    """Per-rep real samples: {(workload, config_label): {'nrmse': [...], 'cov': [...]}}."""
    raw: dict = {}
    raw_path = run_dir / "q5_real_runs_raw.csv"
    if not raw_path.exists():
        return raw
    with raw_path.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (r["workload"], r["config_label"])
            d = raw.setdefault(key, {"nrmse": [], "cov": []})
            for src, dst in (("median_nrmse", "nrmse"), ("coverage", "cov")):
                v = r.get(src, "")
                if v not in ("", "nan"):
                    try:
                        d[dst].append(float(v))
                    except ValueError:
                        pass
    return raw


def _float_or_nan(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def med_iqr(vals: list[float]) -> tuple[float, float, float]:
    """Median and p25/p75 distances from it (asymmetric whiskers)."""
    a = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return np.nan, 0.0, 0.0
    med = float(np.median(a))
    return med, med - float(np.percentile(a, 25)), float(np.percentile(a, 75)) - med


def load_noise_floor() -> float | None:
    if not NOISE_FLOOR_JSON.exists():
        return None
    import json  # noqa: PLC0415

    return json.loads(NOISE_FLOOR_JSON.read_text()).get("median_nrmse")


def leg_value(
    wl: str, label: str, is_real: bool, table: dict, raw: dict, field: str
) -> tuple[float, float, float]:
    """Resolve (median, lo_err, hi_err) for one leg+field ('nrmse' or 'cov').

    Real legs use the per-rep raw distribution (median + IQR whiskers); sim and
    perf_stat are single deterministic values from the comparison CSV.
    """
    if is_real and (wl, label) in raw:
        vals = raw[(wl, label)]["nrmse" if field == "nrmse" else "cov"]
        if vals:
            return med_iqr(vals)
    row = table.get((wl, label))
    if row is None:
        return np.nan, 0.0, 0.0
    col = "median_nrmse" if field == "nrmse" else "coverage"
    return _float_or_nan(row.get(col, "")), 0.0, 0.0


def draw_panel(ax, workloads, table, raw, field, log_y):
    n_legs = len(LEGS)
    width = 0.82 / n_legs
    x = np.arange(len(workloads))

    if log_y:
        ax.set_yscale("log")

    finite_vals: list[float] = []
    for li, (label, _disp, color, is_real) in enumerate(LEGS):
        offs = (li - (n_legs - 1) / 2) * width
        meds, los, his, missing = [], [], [], []
        for wi, wl in enumerate(workloads):
            m, lo, hi = leg_value(wl, label, is_real, table, raw, field)
            meds.append(m)
            los.append(lo)
            his.append(hi)
            missing.append(not np.isfinite(m))
            if np.isfinite(m):
                finite_vals.append(m)
        meds_arr = np.array(meds, dtype=float)
        # On a log axis a zero-height bar corrupts autoscaling, so draw missing
        # (and on log, also true-zero) bars as NaN -- matplotlib skips them.
        plot_vals = np.where(np.isfinite(meds_arr), meds_arr, np.nan)
        ax.bar(
            x + offs, plot_vals, width,
            color=color, edgecolor="black", linewidth=0.6,
            hatch=REAL_HATCH if is_real else None,
            yerr=[los, his] if any(h > 0 or l > 0 for l, h in zip(los, his)) else None,
            capsize=2.5,
            error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.7},
        )

    # Fix the y-range from finite data before annotating, so the log panel does
    # not autoscale toward zero and "n/a" labels sit at a sane baseline.
    if log_y and finite_vals:
        lo_lim = min(finite_vals) * 0.5
        ax.set_ylim(lo_lim, max(finite_vals) * 1.8)
    base = ax.get_ylim()[0]
    base_off = base * 1.15 if log_y else 0.01

    for li, (label, _disp, _color, is_real) in enumerate(LEGS):
        offs = (li - (n_legs - 1) / 2) * width
        for wi, wl in enumerate(workloads):
            m, _lo, _hi = leg_value(wl, label, is_real, table, raw, field)
            if not np.isfinite(m):
                ax.text(
                    x[wi] + offs, base_off, "n/a", rotation=90,
                    ha="center", va="bottom", fontsize=14, color="dimgray",
                )

    ax.set_xticks(x)
    ax.grid(True, axis="y", which="both", alpha=0.3)


def has_real_data(wl: str, table: dict, raw: dict) -> bool:
    """True if any live (_real) leg produced a usable nRMSE for this workload."""
    for label, _disp, _color, is_real in LEGS:
        if not is_real or label == "perf_stat":
            continue
        m, _lo, _hi = leg_value(wl, label, True, table, raw, "nrmse")
        if np.isfinite(m):
            return True
    return False


def main() -> None:
    run_dir = find_run_dir(sys.argv)
    workloads, table = load_comparison(run_dir)
    raw = load_raw(run_dir)
    noise = load_noise_floor()

    # Drop workloads whose live legs are entirely n/a (e.g. omnetpp: tid-unaligned
    # GT -> coverage 0 -> nan nRMSE); they would render as empty columns.
    dropped = [w for w in workloads if not has_real_data(w, table, raw)]
    workloads = [w for w in workloads if w not in dropped]
    if dropped:
        print(f"dropping workloads with no usable real legs: {', '.join(dropped)}")
    if not workloads:
        raise SystemExit("no workloads with usable real-hardware legs to plot")

    short = [w.replace("spec_", "").replace("_r", "") for w in workloads]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(2.6 * len(workloads) + 4, 10), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    draw_panel(ax_top, workloads, table, raw, "nrmse", log_y=True)
    ax_top.set_ylabel("median nRMSE\n(log; lower = better)")
    if noise is not None:
        ax_top.axhline(noise, ls="--", color="dimgray", lw=1.3, alpha=0.8)
        ax_top.text(
            ax_top.get_xlim()[1], noise, f" noise floor ({noise:.3f})",
            va="bottom", ha="right", fontsize=14, color="dimgray",
        )

    draw_panel(ax_bot, workloads, table, raw, "cov", log_y=False)
    ax_bot.set_ylabel("coverage\n(fraction of GT bins)")
    ax_bot.set_ylim(0, 1.05)
    ax_bot.set_xticklabels(short, rotation=10, ha="right")

    # Legend: color = config family, hatch = real hardware.
    handles = [
        Patch(facecolor=ps.Q5_LEGS["best"], edgecolor="black", label="best config"),
        Patch(facecolor=ps.Q5_LEGS["baseline"], edgecolor="black",
              label="baseline (round-robin + propagate)"),
        Patch(facecolor=ps.Q5_LEGS["perf_stat"], edgecolor="black", label="perf stat"),
        Patch(facecolor="white", edgecolor="black", hatch=REAL_HATCH, label="ran on real hardware"),
    ]
    ax_top.legend(handles=handles, fontsize=14, loc="upper left", framealpha=0.8)

    fig.suptitle("Simulation vs. real hardware vs. perf stat")
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}  ({len(workloads)} workloads from {run_dir.name})")


if __name__ == "__main__":
    main()
