"""Heatmap of Q2 sweep median_nrmse: (scheduler x estimator) rows by workload columns.

Color encodes accuracy *relative to the best combo for that workload* (cell /
column-min), since absolute nRMSE scales differ by orders of magnitude across
workloads. 1.0 (white) = best for that workload; warmer = worse. Cells are
annotated with the raw median_nrmse.
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.rcParams.update({"font.size": 14})

CSV = "results/q2_scheduler_estimator.csv"
OUT = "results/q2_heatmap.png"

ESTIMATORS = ["propagate", "ema", "kalman"]
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

rows = {}
workloads = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        w = r["workload"]
        if w not in workloads:
            workloads.append(w)
        rows[(r["scheduler"], r["estimator"], w)] = float(r["median_nrmse"])

workloads.sort()
row_keys = [(s, e) for s in SCHEDULERS for e in ESTIMATORS]
row_labels = [f"{s}  ·  {e}" for s, e in row_keys]

raw = np.array(
    [[rows.get((s, e, w), np.nan) for w in workloads] for s, e in row_keys]
)

# Normalize each column by its minimum so the best combo per workload reads as 1.0.
col_min = np.nanmin(raw, axis=0, keepdims=True)
rel = raw / col_min

fig, ax = plt.subplots(figsize=(14, 18))
im = ax.imshow(rel, aspect="auto", cmap="YlOrRd", norm=LogNorm(vmin=1.0, vmax=np.nanmax(rel)))

ax.set_xticks(range(len(workloads)))
ax.set_xticklabels(workloads, rotation=35, ha="right")
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, family="monospace")

# Light separators between scheduler groups (every 3 estimator rows).
for i in range(len(ESTIMATORS), len(row_keys), len(ESTIMATORS)):
    ax.axhline(i - 0.5, color="white", lw=2)

# Annotate raw median_nrmse; bold + boxed the best combo in each workload column.
best_row_per_col = np.nanargmin(raw, axis=0)
for j in range(len(workloads)):
    for i in range(len(row_keys)):
        v = raw[i, j]
        if np.isnan(v):
            continue
        is_best = i == best_row_per_col[j]
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            color="black" if rel[i, j] < 3 else "white",
            fontweight="bold" if is_best else "normal",
        )
        if is_best:
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="lime", lw=2.2)
            )

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("median nRMSE relative to best combo for that workload\n(1.0 = best, log scale)")

fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}  ({raw.shape[0]} combos × {raw.shape[1]} workloads)")
