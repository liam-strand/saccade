"""Plot for Q4: impact of an LLM guidance hint on reconstruction accuracy.

Reads a q4_llm_guidance.csv (default: the most recent results/*/q4_llm_guidance.csv)
and emits a dumbbell chart: one row per LLM scheduler, two connected points for the
`none` and `with_guidance` conditions, faceted by workload (the workloads sit on very
different nRMSE scales, so each panel gets its own x-axis).

Primary metric is median_nrmse (lower = better). Horizontal whiskers show the
trial-to-trial stddev. Each scheduler's move is encoded by color AND marker shape
(so it survives colorblindness and grayscale), keyed on whether the change clears
trial noise (combined stddev = sqrt(s_none^2 + s_guid^2)):
  blue, up-triangle        -- guidance improved accuracy beyond noise
  vermillion, down-triangle -- guidance regressed accuracy beyond noise
  grey, circle             -- change is within trial noise (no real effect)

The takeaway the figure is built to show: guidance is not uniformly helpful. It moves
accuracy only where there is headroom (deepsjeng) and can go either way depending on
the scheduler; on the noise-floored workload (imagick) it does essentially nothing.
"""

import csv
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import plot_style as ps

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ps.apply_style(14)

IMPROVE = ps.GOOD
REGRESS = ps.BAD
FLAT = ps.NEUTRAL
# Marker shapes carry the same verdict without color: up = improved,
# down = regressed, circle = within noise.
IMPROVE_MARKER = "^"
REGRESS_MARKER = "v"
FLAT_MARKER = "o"


def find_csv() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    candidates = sorted(ps.RESULTS_DIR.glob("*/q4_llm_guidance.csv"))
    if not candidates:
        raise SystemExit("no results/*/q4_llm_guidance.csv found; pass a path as argv[1]")
    return candidates[-1]


def _f(s: str) -> float | None:
    return float(s) if s not in ("", None) else None


def load(path: Path) -> dict:
    """{(workload, scheduler): {cond: (median_nrmse, stddev)}}."""
    data: dict = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            med = _f(r["median_nrmse"])
            std = _f(r["nrmse_stddev"]) or 0.0
            if med is None:
                continue
            data[(r["workload"], r["scheduler"])][r["guidance_condition"]] = (med, std)
    return data


def verdict_for(none_med, guid_med, combined_std) -> tuple[str, str]:
    """(color, marker) for the guidance endpoint -- redundant encoding."""
    delta = guid_med - none_med
    if abs(delta) <= combined_std:
        return FLAT, FLAT_MARKER
    if delta < 0:
        return IMPROVE, IMPROVE_MARKER
    return REGRESS, REGRESS_MARKER


def main() -> None:
    csv_path = find_csv()
    data = load(csv_path)

    workloads = sorted({w for (w, _s) in data})
    schedulers = sorted({s for (_w, s) in data})

    fig, axes = plt.subplots(
        1, len(workloads), figsize=(7.5 * len(workloads), 0.7 * len(schedulers) + 3),
        squeeze=False,
    )
    axes = axes[0]

    for ax, workload in zip(axes, workloads):
        ys = list(range(len(schedulers)))
        for y, sched in zip(ys, schedulers):
            conds = data.get((workload, sched), {})
            if "none" not in conds or "with_guidance" not in conds:
                continue
            none_med, none_std = conds["none"]
            guid_med, guid_std = conds["with_guidance"]
            combined = sqrt(none_std**2 + guid_std**2)
            c, m = verdict_for(none_med, guid_med, combined)

            ax.plot([none_med, guid_med], [y, y], color=c, lw=2.5, zorder=1, alpha=0.8)
            ax.errorbar(
                none_med, y, xerr=none_std, fmt="o", ms=9, mfc="white",
                mec=ps.NEUTRAL, ecolor=ps.NEUTRAL, capsize=3, zorder=2,
            )
            ax.errorbar(
                guid_med, y, xerr=guid_std, fmt=m, ms=10, color=c,
                ecolor=c, capsize=3, zorder=3,
            )
            # annotate the percent change at the guidance endpoint
            pct = 100.0 * (guid_med - none_med) / none_med
            ax.annotate(
                f"{pct:+.0f}%", (guid_med, y),
                textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10, color=c,
            )

        ax.set_yticks(ys)
        ax.set_yticklabels(schedulers if ax is axes[0] else [""] * len(schedulers))
        ax.set_ylim(-0.6, len(schedulers) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel("median nRMSE  (lower = better)")
        ax.set_title(workload, fontsize=14)
        ax.margins(x=0.18)
        ax.grid(True, axis="x", alpha=0.3)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", mfc="white", mec=ps.NEUTRAL,
               ms=9, label="none"),
        Line2D([0], [0], marker="o", color="w", mfc=ps.OKABE["black"], ms=9,
               label="with_guidance"),
        Line2D([0], [0], marker=IMPROVE_MARKER, color=IMPROVE, mfc=IMPROVE, lw=3,
               ms=9, label="improved (beyond trial noise)"),
        Line2D([0], [0], marker=REGRESS_MARKER, color=REGRESS, mfc=REGRESS, lw=3,
               ms=9, label="regressed (beyond trial noise)"),
        Line2D([0], [0], marker=FLAT_MARKER, color=FLAT, mfc=FLAT, lw=3,
               ms=9, label="within trial noise"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=5,
        fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = ps.out("q4_guidance.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
