"""Plots for Q8 slot-swap latency (counter-rotation cost across scheduler quanta).

Reads results/q8_swap_latency_raw.csv (one row per swap, with execution order and a
wall-clock stamp) and emits:

  q8_swap_breakdown.png  per-quantum median reconfig_ns vs quiesce_ns (log y). The
                         core finding: if reconfig (true counter work) is small and
                         flat while quiesce (the stop-the-world spin) dominates, the
                         swap path's cost is the world-stop, not counter reconfiguration.
  q8_swap_drift.png      swap_ns vs execution order_index, colored by quantum, with a
                         per-quantum trend line. The drift discriminator: if swap
                         latency tracks execution order rather than quantum, the
                         original fixed-ascending-order result was a drift artifact.
  q8_swap_quantum.png    requested (q_schedule) vs realized: median swap_ns and median
                         realized quantum_ns against the requested quantum (log-log),
                         making the achievable-rotation-period floor visible while
                         keeping the realized/requested distinction explicit.

Only real swaps (slots_changed > 0) are used for the latency statistics; no-op swaps
have ~0 cost and would otherwise deflate the medians.
"""

import csv
from collections import defaultdict
from pathlib import Path

import plot_style as ps

import matplotlib.pyplot as plt
import numpy as np

ps.apply_style(14)

RAW_CSV = str(ps.RESULTS_DIR / "q8_swap_latency_raw.csv")
OUT_BREAKDOWN = ps.out("q8_swap_breakdown.png")
OUT_DRIFT = ps.out("q8_swap_drift.png")
OUT_QUANTUM = ps.out("q8_swap_quantum.png")


def fmt_ms(ns: float) -> str:
    """Human-readable label for a nanosecond quantum value."""
    ms = ns / 1e6
    return f"{ms:.0f} ms" if ms >= 1 else f"{ns / 1e3:.0f} us"


def load(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "order_index": int(r["order_index"]),
                    "q": int(r["q_schedule_ns"]),
                    "swap_ns": int(r["swap_ns"]),
                    "quiesce_ns": int(r["quiesce_ns"]),
                    "reconfig_ns": int(r["reconfig_ns"]),
                    "slots_changed": int(r["slots_changed"]),
                    "quantum_ns": int(r["quantum_ns"]),
                }
            )
    return rows


def quanta(rows: list[dict]) -> list[int]:
    return sorted({r["q"] for r in rows})


def _med_iqr(vals: list[int]) -> tuple[float, float, float]:
    """Median and the p25/p75 distances from it (for asymmetric error bars), in ms."""
    a = np.array(vals) / 1e6
    med = np.median(a)
    return med, med - np.percentile(a, 25), np.percentile(a, 75) - med


def plot_breakdown(rows: list[dict]) -> None:
    qs = quanta(rows)
    quiesce_med, quiesce_lo, quiesce_hi = [], [], []
    reconfig_med, reconfig_lo, reconfig_hi = [], [], []
    for q in qs:
        real = [r for r in rows if r["q"] == q and r["slots_changed"] > 0]
        qm, ql, qh = _med_iqr([r["quiesce_ns"] for r in real]) if real else (np.nan, 0, 0)
        rm, rl, rh = _med_iqr([r["reconfig_ns"] for r in real]) if real else (np.nan, 0, 0)
        quiesce_med.append(qm); quiesce_lo.append(ql); quiesce_hi.append(qh)
        reconfig_med.append(rm); reconfig_lo.append(rl); reconfig_hi.append(rh)

    x = np.arange(len(qs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - w / 2, quiesce_med, w, label="quiesce (stop-the-world spin)", color=ps.BAD,
        edgecolor="black", linewidth=0.4,
        yerr=[quiesce_lo, quiesce_hi], capsize=3, error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.6},
    )
    ax.bar(
        x + w / 2, reconfig_med, w, label="reconfig (perf_event_open + map)", color=ps.OKABE["blue"],
        hatch=ps.HATCHES[1], edgecolor="black", linewidth=0.4,
        yerr=[reconfig_lo, reconfig_hi], capsize=3, error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.6},
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_ms(q) for q in qs])
    ax.set_title("Swap cost breakdown: quiesce vs. reconfig")
    ax.set_xlabel("requested quantum (--q-schedule)")
    ax.set_ylabel("median per-swap time (ms, log scale)")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_BREAKDOWN, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_BREAKDOWN}")


def plot_drift(rows: list[dict]) -> None:
    qs = quanta(rows)
    cmap = plt.get_cmap(ps.CMAP)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, q in enumerate(qs):
        real = [r for r in rows if r["q"] == q and r["slots_changed"] > 0]
        if not real:
            continue
        order = np.array([r["order_index"] for r in real])
        swap_ms = np.array([r["swap_ns"] for r in real]) / 1e6
        color = cmap(i / max(1, len(qs) - 1))
        ax.scatter(order, swap_ms, s=8, alpha=0.35, color=color, label=fmt_ms(q))
        # Per-quantum linear trend vs execution order.
        if len(order) >= 2:
            slope, intercept = np.polyfit(order, swap_ms, 1)
            xs = np.array([order.min(), order.max()])
            ax.plot(xs, slope * xs + intercept, color=color, lw=2)

    ax.set_title("Swap latency vs. execution order")
    ax.set_xlabel("execution order_index (shuffled across the session)")
    ax.set_ylabel("swap_ns (ms)")
    ax.legend(title="quantum", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DRIFT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_DRIFT}")


def plot_quantum(rows: list[dict]) -> None:
    qs = quanta(rows)
    swap_med, realized_med = [], []
    for q in qs:
        real = [r for r in rows if r["q"] == q and r["slots_changed"] > 0]
        swap_med.append(np.median([r["swap_ns"] for r in real]) if real else np.nan)
        # Realized quantum is reported on every swap line, no-op or not.
        allq = [r for r in rows if r["q"] == q]
        realized_med.append(np.median([r["quantum_ns"] for r in allq]) if allq else np.nan)

    qs_ms = np.array(qs) / 1e6
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(qs_ms, np.array(realized_med) / 1e6, "o-", color=ps.BASELINE, label="realized quantum (elapsed)")
    ax.plot(qs_ms, np.array(swap_med) / 1e6, "s-", color=ps.ACCENT, label="median swap_ns")
    ax.plot(qs_ms, qs_ms, "--", color=ps.NEUTRAL, label="requested = realized (ideal)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Requested vs. realized rotation quantum")
    ax.set_xlabel("requested quantum --q-schedule (ms, log)")
    ax.set_ylabel("ms (log)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_QUANTUM, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_QUANTUM}")


def main() -> None:
    if not Path(RAW_CSV).exists():
        raise SystemExit(f"missing {RAW_CSV} -- run q8_swap_latency.py first")
    rows = load(RAW_CSV)
    if not rows:
        raise SystemExit(f"{RAW_CSV} has no swap rows")
    plot_breakdown(rows)
    plot_drift(rows)
    plot_quantum(rows)


if __name__ == "__main__":
    main()
