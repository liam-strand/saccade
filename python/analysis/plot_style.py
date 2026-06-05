"""Shared plotting style: colorblind- and grayscale-safe colors + output paths.

Every analysis script imports this *before* pyplot:

    import plot_style as ps
    ps.apply_style()

Colors come from the Okabe-Ito palette (the canonical CVD-safe set). Color is
never the only channel: bars carry hatches, lines carry distinct linestyles or
markers, so every figure survives grayscale printing. Sequential data uses
viridis (perceptually uniform, monotone in luminance).

Outputs all land in the top-level results/ directory, resolved relative to
this file (not the cwd) via RESULTS_DIR / out().
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Canonical output dir: analysis/ -> python/results, independent of cwd.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Canonical sequential colormap (CVD-safe, monotone luminance for grayscale).
CMAP = "viridis"

# Okabe-Ito palette.
OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#999999",
}

# Semantic roles. GOOD/BAD differ in both hue and luminance; figures that use
# them must still add a non-color cue (marker shape, hatch) for grayscale.
GOOD = OKABE["blue"]
BAD = OKABE["vermillion"]
NEUTRAL = OKABE["gray"]
BASELINE = OKABE["orange"]
# Accent for medians / reference lines / highlight boxes (replaces "crimson").
ACCENT = OKABE["vermillion"]

# Output sinks (q1 figures). Ordered light -> dark so grayscale stays ordered.
SINK_COLORS = {
    "none": OKABE["sky_blue"],
    "csv": OKABE["orange"],
    "perfetto": OKABE["reddish_purple"],
}

# Estimators (q2 figures).
EST_COLOR = {
    "propagate": OKABE["gray"],
    "ema": OKABE["blue"],
    "kalman": OKABE["vermillion"],
}

# KF variants (q3 figures): color + marker, marker is the grayscale cue.
KF_VARIANTS = {
    "ema": (OKABE["blue"], "o"),
    "kf_naive": (OKABE["bluish_green"], "s"),
    "kf_analytical": (OKABE["orange"], "^"),
    "kf_expert": (OKABE["vermillion"], "D"),
}

# q1 cost-model stacked layers: color + hatch, hatch is the grayscale cue.
COST_LAYERS = {
    "fixed": (OKABE["gray"], ""),
    "quantum": (OKABE["orange"], "..."),
    "runtime": (OKABE["sky_blue"], "//"),
}

# q5 config families (the real/sim split is carried by REAL_HATCH).
Q5_LEGS = {
    "best": OKABE["blue"],
    "baseline": OKABE["orange"],
    "perf_stat": OKABE["gray"],
}

# Redundant-encoding cycles for grouped bars / multi-series lines.
HATCHES = ["", "//", "..", "xx", "\\\\", "++"]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
# q5's existing convention: hatched = ran on real hardware.
REAL_HATCH = "////"


def apply_style(font_size: int = 14) -> None:
    """Unified rcParams: font sizes plus an Okabe-Ito default color cycle."""
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size + 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "axes.prop_cycle": plt.cycler(
                color=[
                    OKABE["blue"],
                    OKABE["vermillion"],
                    OKABE["orange"],
                    OKABE["bluish_green"],
                    OKABE["sky_blue"],
                    OKABE["reddish_purple"],
                    OKABE["gray"],
                ]
            ),
        }
    )


def out(name: str) -> str:
    """Absolute path for an output file under the canonical results/ dir."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / name)
