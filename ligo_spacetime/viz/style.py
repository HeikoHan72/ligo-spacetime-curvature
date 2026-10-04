"""One shared dark theme for all figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # file output only, works without a display

import matplotlib.pyplot as plt  # noqa: E402

BG = "#0b1020"
PANEL = "#111833"
FG = "#e6e9f2"
MUTED = "#8f99b8"
GRID = "#2a3350"
H1_COLOR = "#ff7a59"
L1_COLOR = "#4cc9f0"
CMAP = "coolwarm"  # diverging: sign of the curvature matters


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": FG,
            "axes.titlecolor": FG,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": FG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.facecolor": PANEL,
            "legend.edgecolor": GRID,
            "legend.labelcolor": FG,
        }
    )
