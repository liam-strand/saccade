"""Violin plot of LLM scheduler call latency, one violin per call type.

Reads q7_llm_latency.py's collection output (llm_latency_profile.json) and emits:

  results/q7_llm_latency_violin.png   one violin per LLM scheduler call type,
                                      with the raw samples overlaid as jittered
                                      dots and the median marked.

Input resolution (like the q4/q5 plot scripts):
  * an explicit path given as argv[1], or
  * the newest results/*/llm_latency_profile.json otherwise.

The call types (see q7_llm_latency.py for how they are produced):

  static_setup            initial scheduling call (static-llm / dynamic-llm)
  static_setup_reason     free-form reasoning call before the initial schedule
  dynamic_update          dynamic-llm periodic rescheduling call
  dynamic_update_reason   free-form reasoning call before each periodic update
  wrr_setup               weighted-round-robin-llm weight assignment call

Samples are stored in milliseconds; everything here is plotted in seconds.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import numpy as np

# Bumped font sizes: this figure is typeset at half text-width, so the default
# sizes are illegible once scaled down.
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)

OUT = "results/q7_llm_latency_violin.png"


def find_profile() -> Path:
    """Resolve the llm_latency_profile.json to plot: argv[1] or newest run dir."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_file():
            raise SystemExit(f"no such file: {path}")
        return path
    candidates = sorted(Path("results").glob("*/llm_latency_profile.json"))
    if not candidates:
        raise SystemExit(
            "no results/*/llm_latency_profile.json found; run q7_llm_latency.py first"
        )
    return candidates[-1]

# Display order: setup/update grouped, each "reason" variant next to its base call.
CALL_TYPES = [
    "static_setup",
    "static_setup_reason",
    "dynamic_update",
    "dynamic_update_reason",
    "wrr_setup",
]

LABELS = {
    "static_setup": "static\nsetup",
    "static_setup_reason": "static setup\n(reasoning)",
    "dynamic_update": "dynamic\nupdate",
    "dynamic_update_reason": "dynamic update\n(reasoning)",
    "wrr_setup": "WRR\nsetup",
}


def load(input_path: Path) -> dict[str, np.ndarray]:
    """Return {call_type: samples_in_seconds} for the call types present."""
    raw = json.loads(input_path.read_text())
    return {
        ct: np.asarray(raw[ct]["samples"], dtype=float) / 1000.0
        for ct in CALL_TYPES
        if ct in raw
    }


def plot(data: dict[str, np.ndarray]) -> None:
    """One violin per call type, raw samples jittered on top, median marked."""
    cts = [ct for ct in CALL_TYPES if ct in data]
    series = [data[ct] for ct in cts]
    positions = np.arange(1, len(cts) + 1)

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(cts) - 1)) for i in range(len(cts))]

    fig, ax = plt.subplots(figsize=(10, 6))

    parts = ax.violinplot(
        series,
        positions=positions,
        showextrema=False,
        widths=0.8,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.45)
        body.set_linewidth(0.6)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, series, colors):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            s=14,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.8,
            zorder=3,
        )
        med = float(np.median(vals))
        ax.hlines(med, pos - 0.42, pos + 0.42, color="crimson", lw=3.5, zorder=4)
        ax.text(
            pos + 0.46,
            med,
            f"{med:.0f} s",
            color="crimson",
            fontsize=13,
            va="center",
            ha="left",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{LABELS[ct]}\n(n={len(data[ct])})" for ct in cts], fontsize=14
    )
    ax.set_ylabel("call latency (s)")
    ax.set_xlabel("LLM scheduler call type")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


def main() -> None:
    input_path = find_profile()
    print(f"reading {input_path}")
    plot(load(input_path))


if __name__ == "__main__":
    main()
