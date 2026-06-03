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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAW_CSV = "results/q8_swap_latency_raw.csv"
OUT_BREAKDOWN = "results/q8_swap_breakdown.png"
OUT_DRIFT = "results/q8_swap_drift.png"
OUT_QUANTUM = "results/q8_swap_quantum.png"


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


def plot_breakdown(rows: list[dict]) -> None:
    qs = quanta(rows)
    reconfig_med, quiesce_med = [], []
    for q in qs:
        real = [r for r in rows if r["q"] == q and r["slots_changed"] > 0]
        reconfig_med.append(np.median([r["reconfig_ns"] for r in real]) if real else np.nan)
        quiesce_med.append(np.median([r["quiesce_ns"] for r in real]) if real else np.nan)

    x = np.arange(len(qs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, np.array(quiesce_med) / 1e6, w, label="quiesce (stop-the-world spin)", color="crimson")
    ax.bar(x + w / 2, np.array(reconfig_med) / 1e6, w, label="reconfig (perf_event_open + map)", color="steelblue")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_ms(q) for q in qs])
    ax.set_xlabel("requested quantum (--q-schedule)")
    ax.set_ylabel("median per-swap time (ms, log scale)")
    ax.set_title(
        "Q8: where the slot-swap time goes\n"
        "quiesce spin-wait vs true counter reconfiguration (real swaps only)"
    )
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    # Both timers accumulate per changed slot, so a 4-slot cold-start swap pools
    # with 1-slot warm swaps; the raw CSV retains slots_changed for per-slot views.
    ax.text(
        0.99, -0.13, "medians pool across slots_changed (see raw CSV to normalize per slot)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray",
    )
    fig.tight_layout()
    fig.savefig(OUT_BREAKDOWN, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_BREAKDOWN}")


def plot_drift(rows: list[dict]) -> None:
    qs = quanta(rows)
    cmap = plt.get_cmap("viridis")
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

    ax.set_xlabel("execution order_index (shuffled across the session)")
    ax.set_ylabel("swap_ns (ms)")
    ax.set_title(
        "Q8 drift check: swap latency vs execution order\n"
        "trend should be flat per quantum if there is no thermal/load drift"
    )
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
    ax.plot(qs_ms, np.array(realized_med) / 1e6, "o-", color="darkorange", label="realized quantum (elapsed)")
    ax.plot(qs_ms, np.array(swap_med) / 1e6, "s-", color="crimson", label="median swap_ns")
    ax.plot(qs_ms, qs_ms, "--", color="gray", label="requested = realized (ideal)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("requested quantum --q-schedule (ms, log)")
    ax.set_ylabel("ms (log)")
    ax.set_title(
        "Q8 requested vs realized quantum\n"
        "the gap above the dashed line is the rotation-period floor set by the swap"
    )
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
