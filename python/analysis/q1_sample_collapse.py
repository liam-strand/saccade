"""Q1 overhead vs. delivered-sample count: testing the per-sample cost model.

Hypothesis under test: wall-clock overhead is proportional to the number of
samples actually delivered (ringbuf submissions), not to the configuration
knobs themselves.  If true, every (q_schedule, q_sample, sink) combination
collapses onto a single OLS line as overhead_ms ~ samples_emitted.

Result (see the figure): it does NOT collapse — R² ≈ 0.  Overhead is a roughly
constant floor independent of how many samples are delivered, so the per-sample
marginal cost is negligible relative to a fixed always-on instrumentation cost.

Reads results/q1_overhead_raw.csv (produced by q1_overhead.py after a run with
an instrumented saccade binary that emits ``samples_emitted=N run_complete``
to stderr at the end of each profiling run).

Writes results/q1_sample_collapse.png.
"""

import sys
sys.path.append("../python")

import csv

import plot_style as ps

import matplotlib.pyplot as plt
import numpy as np
from q1_overhead import Q_SAMPLE_NS, Q_SCHEDULE_NS, SINKS

ps.apply_style(14)

RAW_CSV = str(ps.RESULTS_DIR / "q1_overhead_raw.csv")
OUT_PNG = ps.out("q1_sample_collapse.png")

SINK_COLORS = ps.SINK_COLORS

# One marker shape per q_schedule value (coarsest = diamond for easy spotting).
Q_SCHED_MARKERS = {
    1_000_000: "o",
    10_000_000: "s",
    100_000_000: "^",
    1_000_000_000: "D",
}


def fmt_ns(ns: int) -> str:
    """Human-readable duration label for a nanosecond value."""
    if ns >= 1_000_000:
        return f"{ns // 1_000_000} ms"
    if ns >= 1_000:
        return f"{ns // 1_000} µs"
    return f"{ns} ns"


def load(raw_csv: str) -> tuple[float, list[dict]]:
    """Return (baseline_median_s, rows) where rows are non-baseline target runs.

    Each row dict has keys: sink, q_schedule_ns (int), samples_emitted (int),
    elapsed_s (float).  Rows without a samples_emitted value are dropped.
    """
    baseline_elapsed: list[float] = []
    target_rows: list[dict] = []

    with open(raw_csv) as f:
        for r in csv.DictReader(f):
            if r["workload"] != "target":
                continue
            elapsed = float(r["elapsed_s"])
            # is_baseline is written as "0"/"1" (integer string).
            if r["is_baseline"] in ("1", 1):
                baseline_elapsed.append(elapsed)
                continue
            # Non-baseline: need samples_emitted.
            se_raw = r.get("samples_emitted", "").strip()
            if not se_raw:
                continue
            target_rows.append(
                {
                    "sink": r["sink"],
                    "q_schedule_ns": int(r["q_schedule_ns"]),
                    "samples_emitted": int(se_raw),
                    "elapsed_s": elapsed,
                }
            )

    baseline_median_s = float(np.median(baseline_elapsed))
    return baseline_median_s, target_rows


def main() -> None:
    baseline_median_s, rows = load(RAW_CSV)

    if not rows:
        raise RuntimeError(
            f"No non-baseline target rows with samples_emitted found in {RAW_CSV}.\n"
            "Re-run the eval with an instrumented saccade binary first."
        )

    xs = np.array([r["samples_emitted"] for r in rows], dtype=float)
    ys = np.array([(r["elapsed_s"] - baseline_median_s) * 1000.0 for r in rows])

    # OLS fit: y = slope * x + intercept.
    A = np.column_stack([xs, np.ones_like(xs)])
    (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
    y_pred = slope * xs + intercept
    ss_res = float(np.sum((ys - y_pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # slope is ms/sample; convert to ns/sample.
    slope_ns = slope * 1e6

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- scatter: color = sink, marker = q_schedule ---
    for sink in SINKS:
        for q_sched in Q_SCHEDULE_NS:
            pts = [r for r in rows if r["sink"] == sink and r["q_schedule_ns"] == q_sched]
            if not pts:
                continue
            px = [r["samples_emitted"] for r in pts]
            py = [(r["elapsed_s"] - baseline_median_s) * 1000.0 for r in pts]
            ax.scatter(
                px,
                py,
                color=SINK_COLORS[sink],
                marker=Q_SCHED_MARKERS[q_sched],
                s=55,
                alpha=0.75,
                linewidths=0.4,
                edgecolors="black",
                zorder=3,
            )

    # --- OLS fit line ---
    x_fit = np.linspace(xs.min(), xs.max(), 300)
    ax.plot(
        x_fit,
        slope * x_fit + intercept,
        color="black",
        lw=2.0,
        ls="--",
        label=f"OLS fit (R² = {r2:.3f})",
        zorder=4,
    )

    # --- annotation ---
    ax.annotate(
        f"slope = {slope_ns:.0f} ns/sample\n"
        f"intercept = {intercept:.0f} ms (fixed cost)\n"
        f"R² = {r2:.3f}",
        xy=(0.04, 0.96),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8, ec="#cccccc"),
    )

    # --- two-part legend: sink colors + q_schedule markers ---
    sink_handles = [
        plt.scatter([], [], color=SINK_COLORS[s], marker="o", s=60, label=f"sink={s}")
        for s in SINKS
    ]
    sched_handles = [
        plt.scatter(
            [],
            [],
            color=ps.NEUTRAL,
            marker=Q_SCHED_MARKERS[q],
            s=60,
            label=f"q_sched={fmt_ns(q)}",
        )
        for q in Q_SCHEDULE_NS
    ]
    fit_handle = plt.Line2D(
        [0], [0], color="black", lw=2, ls="--", label=f"OLS fit (R² = {r2:.3f})"
    )

    leg1 = ax.legend(
        handles=sink_handles,
        title="sink",
        title_fontsize=14,
        fontsize=14,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=sched_handles + [fit_handle],
        title="q_schedule",
        title_fontsize=14,
        fontsize=14,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.55),
        borderaxespad=0,
    )

    ax.set_xlabel("samples delivered (ringbuf submissions)")
    ax.set_ylabel("wall-clock overhead (ms)")
    ax.set_title("Overhead does not scale with delivered samples")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
