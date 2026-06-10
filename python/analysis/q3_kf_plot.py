"""Plots for Q3: Kalman-filter estimator variants under estimator-independent schedulers.

Reads results/q3_kf_variants.csv (4 variants x 2 schedulers x 7 workloads) and
emits two complementary figures, because the metric spans ~8 orders of magnitude:

  q3_kf_variants.png   primary. median nRMSE per workload, one panel per scheduler,
                       all four variants as markers on a LOG y-axis. Bars are avoided
                       here on purpose -- bar length is meaningless on a log axis. The
                       story: the correlation-aware variants (kf_analytical, kf_expert)
                       diverge by orders of magnitude, while ema and kf_naive stay in a
                       usable band.
  q3_kf_sane.png       zoom. Only the two stable variants (ema, kf_naive) on a LINEAR
                       y-axis, grouped bars per workload, so their competitive ordering
                       is legible -- ema wins the NPB workloads, kf_naive wins the
                       harder SPEC ones; neither dominates.

Caveat surfaced in both captions: the two schedulers have very different sample
coverage (round-robin ~0.99, rate-of-change ~0.70), so nRMSE is NOT comparable
*across* panels -- only the variant ordering *within* a panel is clean.
"""

import csv

import plot_style as ps

import matplotlib.pyplot as plt
import numpy as np

# Bumped font sizes: these figures are typeset at half text-width, so the
# default sizes are illegible once scaled down.
ps.apply_style(15)

CSV = str(ps.RESULTS_DIR / "q3_kf_variants.csv")
OUT_MAIN = ps.out("q3_kf_variants.png")
OUT_SANE = ps.out("q3_kf_sane.png")

SCHEDULERS = ["round-robin", "rate-of-change"]
# Variant -> (display color, marker); the marker doubles as the grayscale cue.
# Order is the legend/plot order.
VARIANTS = ps.KF_VARIANTS
SANE_VARIANTS = ["ema", "kf_naive"]
# nRMSE below this is the "usable" band; above it an estimator is effectively useless.
USABLE_NRMSE = 1.5


def load() -> tuple[dict, dict, list]:
    """Return (nrmse[(sched, variant, workload)], coverage[(sched, workload)], workloads)."""
    nrmse: dict = {}
    coverage: dict = {}
    workloads: list = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            w = r["workload"]
            if w not in workloads:
                workloads.append(w)
            key = (r["scheduler"], r["kf_variant"], w)
            nrmse[key] = float(r["median_nrmse"])
            coverage[(r["scheduler"], w)] = float(r["coverage"])
    return nrmse, coverage, workloads


def mean_coverage(coverage: dict, sched: str, workloads: list) -> float:
    """Average sample coverage for a scheduler across workloads (for captions)."""
    return float(np.mean([coverage[(sched, w)] for w in workloads]))


def short(workload: str) -> str:
    """Compact workload label for crowded x-axes."""
    return workload.replace("spec_", "").replace("npb_", "")


def plot_main(nrmse: dict, coverage: dict, workloads: list) -> None:
    """All four variants per workload as log-scale markers, one panel per scheduler."""
    x = np.arange(len(workloads))
    # Spread the four variants horizontally within each workload's slot.
    offsets = np.linspace(-0.24, 0.24, len(VARIANTS))

    fig, axes = plt.subplots(1, len(SCHEDULERS), figsize=(15, 6), sharey=True)

    for ax, sched in zip(axes, SCHEDULERS):
        for (variant, (color, marker)), dx in zip(VARIANTS.items(), offsets):
            ys = [nrmse[(sched, variant, w)] for w in workloads]
            ax.scatter(
                x + dx,
                ys,
                s=90,
                color=color,
                marker=marker,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
                label=variant if ax is axes[0] else None,
            )
        # Usable band: everything below USABLE_NRMSE is a workable estimate.
        ax.axhspan(
            ax.get_ylim()[0] if False else 1e-2,
            USABLE_NRMSE,
            color=ps.OKABE["bluish_green"],
            alpha=0.06,
            zorder=0,
        )
        ax.axhline(USABLE_NRMSE, color=ps.OKABE["bluish_green"], ls="--", lw=1.0, alpha=0.6)
        cov = mean_coverage(coverage, sched, workloads)
        ax.set_title(f"{sched}\n(mean coverage ≈ {cov:.2f})")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([short(w) for w in workloads], rotation=35, ha="right", fontsize=14)
        ax.grid(axis="y", which="both", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("median nRMSE (log scale)")
    axes[0].legend(title="estimator variant", fontsize=13, title_fontsize=14, loc="upper left")
    axes[-1].text(
        len(workloads) - 0.5,
        USABLE_NRMSE,
        " usable band",
        color=ps.OKABE["bluish_green"],
        fontsize=12,
        va="bottom",
        ha="right",
    )
    fig.suptitle("Correlation-aware KF variants diverge")
    fig.tight_layout()
    fig.savefig(OUT_MAIN, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_MAIN}")


def plot_sane(nrmse: dict, coverage: dict, workloads: list) -> None:
    """Zoom on the two stable variants (ema, kf_naive) -- linear bars, winner annotated."""
    colors = {v: VARIANTS[v][0] for v in SANE_VARIANTS}
    x = np.arange(len(workloads))
    width = 0.8 / len(SANE_VARIANTS)

    fig, axes = plt.subplots(1, len(SCHEDULERS), figsize=(15, 6), sharey=False)

    for ax, sched in zip(axes, SCHEDULERS):
        for k, variant in enumerate(SANE_VARIANTS):
            vals = [nrmse[(sched, variant, w)] for w in workloads]
            offset = (k - (len(SANE_VARIANTS) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width,
                color=colors[variant],
                hatch=ps.HATCHES[k % len(ps.HATCHES)],
                edgecolor="black",
                linewidth=0.4,
                label=variant if ax is axes[0] else None,
            )
        # Mark the per-workload winner among the two with a small star above its bar.
        for j, w in enumerate(workloads):
            pair = {v: nrmse[(sched, v, w)] for v in SANE_VARIANTS}
            best = min(pair, key=pair.get)
            k = SANE_VARIANTS.index(best)
            offset = (k - (len(SANE_VARIANTS) - 1) / 2) * width
            ax.scatter(j + offset, pair[best], marker="*", s=70, color=ps.OKABE["yellow"],
                       edgecolor="black", linewidth=0.4, zorder=4)
        cov = mean_coverage(coverage, sched, workloads)
        ax.set_title(f"{sched}\n(mean coverage ≈ {cov:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels([short(w) for w in workloads], rotation=35, ha="right", fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylabel("median nRMSE")

    axes[0].legend(title="estimator variant", fontsize=13, title_fontsize=14)
    fig.suptitle("The stable estimators: ema vs. kf_naive")
    fig.tight_layout()
    fig.savefig(OUT_SANE, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_SANE}")


def main() -> None:
    nrmse, coverage, workloads = load()
    plot_main(nrmse, coverage, workloads)
    plot_sane(nrmse, coverage, workloads)


if __name__ == "__main__":
    main()
