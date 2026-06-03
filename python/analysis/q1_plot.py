"""Plots for Q1 runtime-overhead sweep (q_schedule x q_sample x sink).

Reads results/q1_overhead.csv (and results/q1_startup.csv if present) and emits:

  q1_overhead_bars.png          faceted grouped bars (one panel per sink), the
                                primary figure. A dashed line per panel marks
                                that sink's fixed startup cost (from /bin/true).
  q1_overhead_bars_runtime.png  same bars with the fixed startup cost subtracted
                                -- the runtime-scaling overhead only.
  q1_overhead_heatmap.png       q_schedule x q_sample heatmap grid (one per
                                sink), the compact parameter-sweep view.
  q1_overhead_by_sink.png       distribution of overhead across configs per sink.

Overhead is plotted as a percentage of the (constant) unprofiled baseline.
The per-config raw reps are not re-read here -- only the median and the IQR
from the summary CSV -- so uncertainty is shown as +/- IQR/2 error bars (the
middle-50%% half-width), not true distributions.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/q1_overhead.csv"
STARTUP_CSV = "results/q1_startup.csv"
OUT_BARS = "results/q1_overhead_bars.png"
OUT_BARS_RUNTIME = "results/q1_overhead_bars_runtime.png"
OUT_HEATMAP = "results/q1_overhead_heatmap.png"
OUT_BY_SINK = "results/q1_overhead_by_sink.png"

SINKS = ["none", "csv", "perfetto"]
Q_SCHEDULE_NS = [100_000, 1_000_000, 10_000_000, 100_000_000]
Q_SAMPLE_NS = [10_000, 100_000, 1_000_000]


def fmt_ns(ns: int) -> str:
    """Human-readable duration label for a nanosecond value."""
    if ns >= 1_000_000:
        return f"{ns // 1_000_000} ms"
    if ns >= 1_000:
        return f"{ns // 1_000} µs"
    return f"{ns} ns"


def load() -> tuple[dict, dict]:
    """Return ({(sink, q_sched, q_samp): {overhead_pct, iqr_pct}}, meta) from the CSV.

    meta carries baseline_s and reps for dynamic captions.
    """
    cells: dict = {}
    meta = {"baseline_s": None, "reps": None}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            baseline = float(r["baseline_median_s"])
            meta["baseline_s"] = baseline
            meta["reps"] = int(r["reps"])
            key = (r["sink"], int(r["q_schedule_ns"]), int(r["q_sample_ns"]))
            cells[key] = {
                "overhead_pct": float(r["overhead_fraction"]) * 100.0,
                # IQR is in seconds; express as % of baseline to match the y-axis.
                "iqr_pct": float(r["iqr_s"]) / baseline * 100.0,
            }
    return cells, meta


def load_startup(target_baseline_s: float) -> dict | None:
    """Return {(sink, q_sched, q_samp): startup_pct} as a % of the target baseline.

    startup_pct expresses saccade's fixed startup cost (measured on /bin/true)
    on the same axis as the overhead bars. Returns None if the startup CSV is
    absent (e.g. the run used --no-null-workload).
    """
    if not Path(STARTUP_CSV).exists():
        return None
    startup: dict = {}
    with open(STARTUP_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["sink"], int(r["q_schedule_ns"]), int(r["q_sample_ns"]))
            startup[key] = float(r["startup_overhead_s"]) / target_baseline_s * 100.0
    return startup


def sink_startup_pct(startup: dict, sink: str) -> float:
    """Median fixed-startup cost (% of baseline) across the grid for one sink."""
    vals = [startup[(sink, qs, qsa)] for qs in Q_SCHEDULE_NS for qsa in Q_SAMPLE_NS]
    return float(np.median(vals))


def plot_bars(cells: dict, meta: dict, startup: dict | None) -> None:
    """Faceted grouped bars: one panel per sink, x = q_sample, bars = q_schedule.

    If startup data is available, a dashed line per panel marks that sink's fixed
    startup cost -- the floor of overhead that is independent of workload runtime.
    """
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(Q_SCHEDULE_NS) - 1)) for i in range(len(Q_SCHEDULE_NS))]

    fig, axes = plt.subplots(
        1, len(SINKS), figsize=(14, 5), sharey=True
    )
    x = np.arange(len(Q_SAMPLE_NS))
    width = 0.8 / len(Q_SCHEDULE_NS)

    for ax, sink in zip(axes, SINKS):
        for k, q_sched in enumerate(Q_SCHEDULE_NS):
            vals = [cells[(sink, q_sched, q_samp)]["overhead_pct"] for q_samp in Q_SAMPLE_NS]
            # Error bar = +/- IQR/2 (middle-50% half-width), clamped non-negative.
            err = [cells[(sink, q_sched, q_samp)]["iqr_pct"] / 2 for q_samp in Q_SAMPLE_NS]
            offset = (k - (len(Q_SCHEDULE_NS) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width,
                yerr=err,
                capsize=2,
                color=colors[k],
                edgecolor="black",
                linewidth=0.4,
                error_kw={"elinewidth": 0.8, "alpha": 0.6},
                label=fmt_ns(q_sched) if ax is axes[0] else None,
            )
        ax.axhline(0, color="black", lw=0.8)
        if startup is not None:
            floor = sink_startup_pct(startup, sink)
            ax.axhline(
                floor,
                color="crimson",
                ls="--",
                lw=1.3,
                label="fixed startup cost" if ax is axes[0] else None,
            )
            ax.text(
                len(Q_SAMPLE_NS) - 0.5,
                floor,
                f" startup ≈ {floor:.1f}%",
                color="crimson",
                fontsize=8,
                va="bottom",
                ha="right",
            )
        ax.set_title(f"sink = {sink}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([fmt_ns(q) for q in Q_SAMPLE_NS])
        ax.set_xlabel("q_sample")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("wall-clock overhead (% of baseline)")
    axes[0].legend(title="q_schedule", fontsize=8, title_fontsize=9)
    fig.suptitle(
        "Q1: saccade runtime overhead by sink, sampling, and scheduling period\n"
        f"error bars = ±IQR/2 (middle-50% spread of {meta['reps']} reps); "
        f"baseline = {meta['baseline_s']:.2f} s",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_BARS, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_BARS}")


def plot_bars_runtime(cells: dict, meta: dict, startup: dict) -> None:
    """Faceted grouped bars with the fixed startup cost subtracted per config.

    This isolates the runtime-scaling overhead -- the cost that grows with how
    long the workload runs (per-quantum sampling, counter swaps) -- by removing
    saccade's fixed startup/teardown cost measured on /bin/true.
    """
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(Q_SCHEDULE_NS) - 1)) for i in range(len(Q_SCHEDULE_NS))]

    fig, axes = plt.subplots(1, len(SINKS), figsize=(14, 5), sharey=True)
    x = np.arange(len(Q_SAMPLE_NS))
    width = 0.8 / len(Q_SCHEDULE_NS)

    for ax, sink in zip(axes, SINKS):
        for k, q_sched in enumerate(Q_SCHEDULE_NS):
            # Per-config subtraction: overhead minus that exact config's startup cost.
            vals = [
                cells[(sink, q_sched, q_samp)]["overhead_pct"]
                - startup[(sink, q_sched, q_samp)]
                for q_samp in Q_SAMPLE_NS
            ]
            err = [cells[(sink, q_sched, q_samp)]["iqr_pct"] / 2 for q_samp in Q_SAMPLE_NS]
            offset = (k - (len(Q_SCHEDULE_NS) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width,
                yerr=err,
                capsize=2,
                color=colors[k],
                edgecolor="black",
                linewidth=0.4,
                error_kw={"elinewidth": 0.8, "alpha": 0.6},
                label=fmt_ns(q_sched) if ax is axes[0] else None,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"sink = {sink}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([fmt_ns(q) for q in Q_SAMPLE_NS])
        ax.set_xlabel("q_sample")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("runtime-scaling overhead (% of baseline)")
    axes[0].legend(title="q_schedule", fontsize=8, title_fontsize=9)
    fig.suptitle(
        "Q1: runtime-scaling overhead (fixed startup cost removed)\n"
        f"per-config startup subtracted; error bars = ±IQR/2; "
        f"baseline = {meta['baseline_s']:.2f} s",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_BARS_RUNTIME, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_BARS_RUNTIME}")


def plot_heatmap(cells: dict) -> None:
    """q_schedule (rows) x q_sample (cols) heatmap, one panel per sink, shared scale."""
    grids = {
        sink: np.array(
            [
                [cells[(sink, qs, qsa)]["overhead_pct"] for qsa in Q_SAMPLE_NS]
                for qs in Q_SCHEDULE_NS
            ]
        )
        for sink in SINKS
    }
    vmax = max(g.max() for g in grids.values())

    fig, axes = plt.subplots(1, len(SINKS), figsize=(14, 5))
    fig.subplots_adjust(top=0.80)
    im = None
    for ax, sink in zip(axes, SINKS):
        g = grids[sink]
        im = ax.imshow(g, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(f"sink = {sink}", fontsize=11)
        ax.set_xticks(range(len(Q_SAMPLE_NS)))
        ax.set_xticklabels([fmt_ns(q) for q in Q_SAMPLE_NS], rotation=20, ha="right")
        ax.set_yticks(range(len(Q_SCHEDULE_NS)))
        ax.set_yticklabels([fmt_ns(q) for q in Q_SCHEDULE_NS])
        ax.set_xlabel("q_sample")
        if ax is axes[0]:
            ax.set_ylabel("q_schedule")
        for i, qs in enumerate(Q_SCHEDULE_NS):
            for j, qsa in enumerate(Q_SAMPLE_NS):
                c = cells[(sink, qs, qsa)]
                ax.text(
                    j,
                    i,
                    f"{c['overhead_pct']:.1f}%\n±{c['iqr_pct']:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if g[i, j] < vmax * 0.6 else "white",
                )

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("overhead (% of baseline)", fontsize=9)
    fig.suptitle(
        "Q1: overhead across the q_schedule × q_sample grid\n"
        "(cell = overhead %; ± = IQR as % of baseline)",
        fontsize=12,
        y=0.99,
    )
    fig.savefig(OUT_HEATMAP, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_HEATMAP}")


def plot_by_sink(cells: dict) -> None:
    """Distribution of per-config overhead for each sink (headline summary)."""
    data = [
        [cells[(sink, qs, qsa)]["overhead_pct"] for qs in Q_SCHEDULE_NS for qsa in Q_SAMPLE_NS]
        for sink in SINKS
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        data,
        tick_labels=SINKS,
        widths=0.5,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    for median in bp["medians"]:
        median.set_color("crimson")
    # Overlay the individual config points with a little horizontal jitter.
    jitter = np.linspace(-0.12, 0.12, len(data[0]))
    for i, vals in enumerate(data):
        ax.scatter(
            np.full(len(vals), i + 1) + jitter,
            vals,
            s=18,
            color="steelblue",
            alpha=0.7,
            zorder=3,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("wall-clock overhead (% of baseline)")
    ax.set_xlabel("output sink")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Q1: overhead distribution across all 12 configs per sink\n"
        "(box = quartiles, red = median, diamond = mean, dots = individual configs)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_BY_SINK, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_BY_SINK}")


def main() -> None:
    cells, meta = load()
    startup = load_startup(meta["baseline_s"])
    plot_bars(cells, meta, startup)
    if startup is not None:
        plot_bars_runtime(cells, meta, startup)
    else:
        print(f"(no {STARTUP_CSV}; skipping startup line and runtime-scaling plot)")
    plot_heatmap(cells)
    plot_by_sink(cells)


if __name__ == "__main__":
    main()
