"""Heatmap of Q2 sweep nrmse_mean: (scheduler x estimator) rows by workload columns.

Color encodes accuracy *relative to the random·ema baseline for that workload*
(cell / random-ema), since absolute nRMSE scales differ by orders of magnitude
across workloads. The random·ema row is white; blue = better than the baseline,
vermillion = worse, with intensity scaling log-symmetrically with the ratio.
Cells are annotated with the raw nrmse_mean (mean across trials of per-trial
median nRMSE) ± nrmse_stddev (stddev across trials).
"""

import csv

import plot_style as ps

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

ps.apply_style(14)

CSV = str(ps.RESULTS_DIR / "q2_scheduler_estimator.csv")
OUT = ps.out("q2_heatmap.png")

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
stddevs = {}
workloads = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        w = r["workload"]
        if w not in workloads:
            workloads.append(w)
        rows[(r["scheduler"], r["estimator"], w)] = float(r["nrmse_mean"])
        stddevs[(r["scheduler"], r["estimator"], w)] = (
            float(r["nrmse_stddev"]) if r["nrmse_stddev"] else np.nan
        )

workloads.sort()
row_keys = [(s, e) for s in SCHEDULERS for e in ESTIMATORS]
row_labels = [f"{s}  ·  {e}" for s, e in row_keys]

raw = np.array(
    [[rows.get((s, e, w), np.nan) for w in workloads] for s, e in row_keys]
)
std = np.array(
    [[stddevs.get((s, e, w), np.nan) for w in workloads] for s, e in row_keys]
)

# Normalize each column by the random·ema baseline so that row reads as 1.0
# (white); cells better than the baseline go blue, worse go vermillion.
base = raw[row_keys.index(("random", "ema"))]
rel = raw / base
logr = np.log10(rel)

cmap = LinearSegmentedColormap.from_list("good_bad", [ps.GOOD, "white", ps.BAD])
cmap.set_bad("white")
vmax = np.nanmax(np.abs(logr))

fig, ax = plt.subplots(figsize=(14, 18))
im = ax.imshow(logr, aspect="auto", cmap=cmap, norm=Normalize(-vmax, vmax))

ax.set_xticks(range(len(workloads)))
ax.set_xticklabels(workloads, rotation=35, ha="right")
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, family="monospace")

# Light separators between scheduler groups (every 3 estimator rows).
for i in range(len(ESTIMATORS), len(row_keys), len(ESTIMATORS)):
    ax.axhline(i - 0.5, color="white", lw=2)

# Annotate raw nrmse_mean ± stddev; bold + boxed the best combo in each workload column.
best_row_per_col = np.nanargmin(raw, axis=0)
for j in range(len(workloads)):
    for i in range(len(row_keys)):
        v = raw[i, j]
        if np.isnan(v):
            continue
        is_best = i == best_row_per_col[j]
        sd = std[i, j]
        label = f"{v:.2f}" if np.isnan(sd) else f"{v:.2f}\n±{sd:.2f}"
        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=9,
            # Black text on pale/white cells, white only on the deep ends.
            color="white" if abs(logr[i, j]) > 0.6 * vmax else "black",
            fontweight="bold" if is_best else "normal",
        )
        if is_best:
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.2
                )
            )

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
ticks = [t for t in [0.01, 0.1, 0.33, 1, 3, 10, 100] if abs(np.log10(t)) <= vmax]
cbar.set_ticks(np.log10(ticks))
cbar.set_ticklabels([f"{t:g}×" for t in ticks])
cbar.set_label("mean nRMSE relative to random·ema for that workload\n(<1× = better, log scale)")

fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}  ({raw.shape[0]} combos × {raw.shape[1]} workloads)")
