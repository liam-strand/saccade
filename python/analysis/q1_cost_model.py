"""Q1 overhead *cost model*: where saccade's runtime overhead actually comes from.

The grid plots in q1_plot.py answer "how big is the overhead". These two answer
"what is it made of", using the same two summary CSVs plus the raw per-run CSV.

  q1_cost_model.png   Stacked decomposition (absolute ms), one panel per sink,
                      x = q_schedule. Each bar splits total target overhead into
                        1. fixed attach/teardown  -- constant intercept (~49 ms)
                        2. one-quantum teardown    -- the q_schedule-dependent part
                                                      of startup (≈ 1 x q_schedule)
                        3. runtime instrumentation floor -- everything left over.
                                                      Despite the name this does
                                                      NOT scale with sampling: the
                                                      q1_sample_collapse figure shows
                                                      overhead is independent of the
                                                      delivered-sample count (R²≈0).
                      The runtime layer is the median across q_sample; the cap
                      error bar spans its min..max across q_sample.

  q1_insensitivity.png  The headline finding: overhead is a near-constant floor
                      that the configuration knobs barely move. Two panels sweep
                      q_sample and q_schedule (each pooled over the other knob);
                      a shaded band is the 25-75th pct of the raw reps, coloured
                      lines are the per-sink medians, and a dashed line marks the
                      overall median floor. Only aggressive 10 µs sampling visibly
                      lifts the floor -- profiling hard is otherwise ~free.

The fixed/quantum split is read off a linear fit of the /bin/true startup cost
against q_schedule (startup ≈ intercept + slope x q_schedule); the intercept is
the constant attach/teardown floor, the remainder is the per-run extra quantum.
"""

import sys
sys.path.append("../python")
import csv

import plot_style as ps

import matplotlib.pyplot as plt
import numpy as np

from q1_overhead import Q_SAMPLE_NS, Q_SCHEDULE_NS, SINKS
ps.apply_style(14)

OVERHEAD_CSV = str(ps.RESULTS_DIR / "q1_overhead.csv")
STARTUP_CSV = str(ps.RESULTS_DIR / "q1_startup.csv")
RAW_CSV = str(ps.RESULTS_DIR / "q1_overhead_raw.csv")
OUT_COST = ps.out("q1_cost_model.png")
OUT_INSENS = ps.out("q1_insensitivity.png")

# Layer (color, hatch) pairs and per-sink colors from the shared style; the
# hatch is what keeps the stacked layers apart in grayscale.
C_FIXED, H_FIXED = ps.COST_LAYERS["fixed"]
C_QUANTUM, H_QUANTUM = ps.COST_LAYERS["quantum"]
C_RUNTIME, H_RUNTIME = ps.COST_LAYERS["runtime"]
SINK_COLORS = ps.SINK_COLORS


def fmt_ns(ns: int) -> str:
    """Human-readable duration label for a nanosecond value."""
    if ns >= 1_000_000:
        return f"{ns // 1_000_000} ms"
    if ns >= 1_000:
        return f"{ns // 1_000} µs"
    return f"{ns} ns"


def load() -> tuple[dict, dict, float]:
    """Return (overhead_s, startup_s, baseline_s) keyed by (sink, q_sched, q_samp).

    overhead_s  = target saccade median - target baseline   (absolute seconds)
    startup_s   = /bin/true saccade median - /bin/true bare  (absolute seconds)
    """
    baseline_s = 0.0
    overhead_s: dict = {}
    with open(OVERHEAD_CSV) as f:
        for r in csv.DictReader(f):
            baseline_s = float(r["baseline_median_s"])
            key = (r["sink"], int(r["q_schedule_ns"]), int(r["q_sample_ns"]))
            overhead_s[key] = float(r["overhead_fraction"]) * baseline_s
    startup_s: dict = {}
    with open(STARTUP_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["sink"], int(r["q_schedule_ns"]), int(r["q_sample_ns"]))
            startup_s[key] = float(r["startup_overhead_s"])
    return overhead_s, startup_s, baseline_s


def fit_startup_model(startup_s: dict) -> tuple[float, float]:
    """Fit startup_s ≈ intercept + slope * q_schedule_ns over the whole grid.

    Returns (intercept_s, slope_s_per_ns). The intercept is the constant
    attach/teardown floor; slope ≈ 1e-9 means one extra scheduling quantum.
    """
    xs = np.array([qs for (_, qs, _) in startup_s])
    ys = np.array(list(startup_s.values()))
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(intercept), float(slope)


def plot_cost_model(overhead_s: dict, startup_s: dict, intercept_s: float, slope: float) -> None:
    """Stacked fixed / quantum / runtime decomposition, one panel per sink."""
    fig, axes = plt.subplots(1, len(SINKS), figsize=(16, 6), sharey=True)
    x = np.arange(len(Q_SCHEDULE_NS))
    fixed_ms = intercept_s * 1000.0

    for ax, sink in zip(axes, SINKS):
        fixed = np.full(len(Q_SCHEDULE_NS), fixed_ms)
        quantum = np.empty(len(Q_SCHEDULE_NS))
        rt_med = np.empty(len(Q_SCHEDULE_NS))
        rt_lo = np.empty(len(Q_SCHEDULE_NS))
        rt_hi = np.empty(len(Q_SCHEDULE_NS))
        for i, qs in enumerate(Q_SCHEDULE_NS):
            # Startup is ~independent of q_sample; take the median across it.
            startup_here = np.median([startup_s[(sink, qs, qsa)] for qsa in Q_SAMPLE_NS]) * 1000.0
            quantum[i] = max(startup_here - fixed_ms, 0.0)
            # Runtime-scaling = total target overhead minus the fixed+quantum startup.
            rt = [
                overhead_s[(sink, qs, qsa)] * 1000.0 - startup_s[(sink, qs, qsa)] * 1000.0
                for qsa in Q_SAMPLE_NS
            ]
            rt_med[i], rt_lo[i], rt_hi[i] = np.median(rt), min(rt), max(rt)

        ax.bar(x, fixed, width=0.62, color=C_FIXED, hatch=H_FIXED, edgecolor="black",
               linewidth=0.4, label="fixed attach/teardown")
        ax.bar(x, quantum, width=0.62, bottom=fixed, color=C_QUANTUM, hatch=H_QUANTUM,
               edgecolor="black", linewidth=0.4, label="one-quantum teardown")
        top = fixed + quantum
        ax.bar(x, rt_med, width=0.62, bottom=top, color=C_RUNTIME, hatch=H_RUNTIME,
               edgecolor="black", linewidth=0.4, label="runtime instrumentation floor")
        # Cap error bar = runtime min..max across q_sample, anchored on the layer.
        ax.errorbar(x, top + rt_med, yerr=[rt_med - rt_lo, rt_hi - rt_med], fmt="none",
                    ecolor="black", elinewidth=1.0, capsize=4, alpha=0.7)

        ax.set_title(f"sink = {sink}")
        ax.set_xticks(x)
        ax.set_xticklabels([fmt_ns(q) for q in Q_SCHEDULE_NS], rotation=20, ha="right")
        ax.set_xlabel("q_schedule")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("overhead (ms, absolute)")
    axes[0].legend(fontsize=14, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_COST, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_COST}")


def load_raw_overhead_pct(baseline_s: float) -> list[tuple]:
    """Return (sink, q_sched, q_samp, overhead_pct) for every raw target rep."""
    runs: list[tuple] = []
    with open(RAW_CSV) as f:
        for r in csv.DictReader(f):
            if r["workload"] != "target" or r["is_baseline"] != "0":
                continue
            pct = (float(r["elapsed_s"]) - baseline_s) / baseline_s * 100.0
            runs.append((r["sink"], int(r["q_schedule_ns"]), int(r["q_sample_ns"]), pct))
    return runs


def _sweep_panel(ax, runs: list[tuple], axis_idx: int, knob_values: list[int],
                 xlabel: str, floor: float) -> None:
    """Draw one knob sweep: pooled IQR band + per-sink median lines on a log x-axis.

    axis_idx selects which tuple field is the swept knob (1 = q_sched, 2 = q_samp).
    Each knob value pools all reps sharing it (marginalising over the other knob).
    """
    xs = np.array(knob_values, dtype=float)

    # Pooled 25-75th percentile band across all sinks and the other knob.
    p25, p50, p75 = [], [], []
    for v in knob_values:
        pool = [r[3] for r in runs if r[axis_idx] == v]
        p25.append(np.percentile(pool, 25))
        p50.append(np.percentile(pool, 50))
        p75.append(np.percentile(pool, 75))
    ax.fill_between(xs, p25, p75, color=ps.NEUTRAL, alpha=0.5, lw=0,
                    label="pooled 25–75th pct")

    # Per-sink median lines: distinct linestyle + marker so the three sinks
    # stay apart in grayscale.
    for i, sink in enumerate(SINKS):
        med = [
            np.median([r[3] for r in runs if r[axis_idx] == v and r[0] == sink])
            for v in knob_values
        ]
        ax.plot(xs, med, color=SINK_COLORS[sink], ls=ps.LINESTYLES[i % len(ps.LINESTYLES)],
                marker=ps.MARKERS[i % len(ps.MARKERS)], lw=2.0, ms=6, label=f"{sink}")

    ax.axhline(floor, color="black", ls="--", lw=1.3)
    ax.text(xs[-1], floor, f" floor ≈ {floor:.1f}%", color="black", fontsize=14,
            va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt_ns(int(v)) for v in knob_values])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.3)


def plot_insensitivity(baseline_s: float) -> None:
    """Headline: overhead is a flat floor the q_sample / q_schedule knobs barely move."""
    runs = load_raw_overhead_pct(baseline_s)
    floor = float(np.median([r[3] for r in runs]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    _sweep_panel(ax1, runs, 2, Q_SAMPLE_NS, "q_sample (sampling period)", floor)
    _sweep_panel(ax2, runs, 1, Q_SCHEDULE_NS, "q_schedule (rotation period)", floor)

    ax1.set_ylabel("wall-clock overhead (% of baseline)")
    # Single shared legend (sinks + band) from the first panel.
    ax1.legend(fontsize=14, loc="upper right", title="sink", title_fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_INSENS, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_INSENS}")


def main() -> None:
    overhead_s, startup_s, baseline_s = load()
    intercept_s, slope = fit_startup_model(startup_s)
    plot_cost_model(overhead_s, startup_s, intercept_s, slope)
    plot_insensitivity(baseline_s)


if __name__ == "__main__":
    main()
