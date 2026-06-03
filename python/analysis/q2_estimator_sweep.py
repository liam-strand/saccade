"""Estimator sweep within three chosen schedulers.

Compares the three estimators (propagate / ema / kalman) under each of three
schedulers chosen to tell the coupling story:

  * round-robin     -- naive, uncertainty-agnostic baseline
  * max-uncertainty -- adaptive policy that *consumes* the estimator's
                       uncertainty signal to decide what to sample next
  * dynamic-llm     -- best scheduler by mean-rank (each scheduler using its
                       best estimator per workload)

Headline: EMA is the default-best estimator under every uncertainty-agnostic
scheduler, but max-uncertainty needs Kalman -- it steers sampling on the
estimator's uncertainty, and only Kalman produces a calibrated one.

Reads results/q2_scheduler_estimator.csv; writes a focused CSV and a 3-panel
figure (raw median nRMSE per workload, log y) under results/.
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 14})

CSV = "results/q2_scheduler_estimator.csv"
# One figure per scheduler: results/q2_estimator_sweep_<scheduler>.png
OUT_FIG = "results/q2_estimator_sweep_{sched}.png"
OUT_CSV = "results/q2_estimator_sweep.csv"

ESTIMATORS = ["propagate", "ema", "kalman"]
# (scheduler, human-readable role) chosen for the three sweeps.
SCHEDULERS = [
    ("round-robin", "naive baseline"),
    ("max-uncertainty", "uncertainty-driven"),
    ("dynamic-llm", "best (by mean rank)"),
]
EST_COLOR = {"propagate": "#bdbdbd", "ema": "#4c72b0", "kalman": "#dd8452"}


def load():
    nrmse, cal, workloads = {}, {}, []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            w = r["workload"]
            if w not in workloads:
                workloads.append(w)
            key = (r["scheduler"], r["estimator"], w)
            try:
                nrmse[key] = float(r["median_nrmse"])
            except ValueError:
                pass
            try:
                cal[key] = float(r["mean_calibration"])
            except (ValueError, KeyError):
                pass
    workloads.sort()
    return nrmse, cal, workloads


def geomean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x) and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def main():
    nrmse, cal, workloads = load()

    def vec(s, e):
        return np.array([nrmse.get((s, e, w), np.nan) for w in workloads])

    # ---- focused CSV summary -------------------------------------------------
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["scheduler", "role", "estimator"]
            + [f"nrmse_{w}" for w in workloads]
            + ["geomean_nrmse", "wins_vs_other_estimators"]
        )
        for sched, role in SCHEDULERS:
            # per-workload winning estimator within this scheduler
            win = {e: 0 for e in ESTIMATORS}
            for i in range(len(workloads)):
                vals = {e: vec(sched, e)[i] for e in ESTIMATORS}
                win[min(vals, key=vals.get)] += 1
            for e in ESTIMATORS:
                row_v = vec(sched, e)
                wr.writerow(
                    [sched, role, e]
                    + [f"{v:.4f}" for v in row_v]
                    + [f"{geomean(row_v):.4f}", win[e]]
                )

    # ---- one figure per scheduler: grouped bars per workload, log y ----------
    x = np.arange(len(workloads))
    bw = 0.26
    handles = [plt.Rectangle((0, 0), 1, 1, color=EST_COLOR[e]) for e in ESTIMATORS]

    for sched, _role in SCHEDULERS:
        win = {e: 0 for e in ESTIMATORS}
        for i in range(len(workloads)):
            vals = {e: vec(sched, e)[i] for e in ESTIMATORS}
            win[min(vals, key=vals.get)] += 1

        fig, ax = plt.subplots(figsize=(7, 5.2))
        for j, e in enumerate(ESTIMATORS):
            ax.bar(x + (j - 1) * bw, vec(sched, e), bw, color=EST_COLOR[e])
        # mark the per-workload winner with a star above its bar
        for i in range(len(workloads)):
            vals = {e: vec(sched, e)[i] for e in ESTIMATORS}
            best_e = min(vals, key=vals.get)
            j = ESTIMATORS.index(best_e)
            ax.plot(x[i] + (j - 1) * bw, vals[best_e] * 1.06, marker="*",
                    color="black", ms=11, zorder=5)
        ax.set_yscale("log")
        ax.set_ylabel("median nRMSE (lower = better)")
        ax.set_xticks(x)
        ax.set_xticklabels([w.replace("spec_", "").replace("npb_", "")
                            for w in workloads], rotation=35, ha="right")
        ax.grid(axis="y", which="both", ls=":", alpha=0.4)
        ax.legend(handles, [f"{e}  ({win[e]}/{len(workloads)} wins)" for e in ESTIMATORS],
                  loc="upper left", framealpha=0.9)

        fig.tight_layout()
        out = OUT_FIG.format(sched=sched)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
    print(f"wrote {OUT_CSV}")

    # ---- console summary -----------------------------------------------------
    print("\nGeomean median nRMSE across workloads (lower = better):")
    print(f"{'scheduler':18s} {'propagate':>11s} {'ema':>11s} {'kalman':>11s}   winner")
    for sched, _role in SCHEDULERS:
        gms = {e: geomean(vec(sched, e)) for e in ESTIMATORS}
        best = min(gms, key=gms.get)
        print(f"{sched:18s} " + " ".join(f"{gms[e]:11.3f}" for e in ESTIMATORS)
              + f"   {best}")


if __name__ == "__main__":
    main()
