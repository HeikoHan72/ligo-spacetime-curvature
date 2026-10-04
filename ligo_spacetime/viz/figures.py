"""Figure functions. Each one saves a PNG (or GIF) and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import signal

from ..config import EVENT_WINDOW_S, SPEED_OF_LIGHT_M_S
from ..waves import curvature, processing
from ..waves.pipeline import EventResult
from . import style

STRAIN_UNIT = 1e-21
CURVATURE_UNIT = 1e-33


def _save(fig, path: Path, dpi: int = 150) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def _symmetric_norm(values: np.ndarray) -> colors.Normalize:
    limit = float(np.max(np.abs(values))) or 1.0
    return colors.Normalize(vmin=-limit, vmax=limit)


# --------------------------------------------------------------------------- #
# 1. Spectrogram: proof that the chirp is really in the data
# --------------------------------------------------------------------------- #
def plot_spectrogram(result: EventResult, path: Path) -> Path:
    style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (ifo, det) in zip(axes, result.detectors.items(), strict=True):
        white = processing.whiten(det.raw)
        freqs, seg_times, power = signal.spectrogram(
            white.values, fs=white.fs, nperseg=256, noverlap=240, window="hann"
        )
        rel = det.raw.gps_start + seg_times - result.event_gps
        sel_t = (rel > -0.45) & (rel < 0.25)
        sel_f = (freqs >= 20) & (freqs <= 400)
        data = power[np.ix_(sel_f, sel_t)]
        mesh = ax.pcolormesh(
            rel[sel_t],
            freqs[sel_f],
            data,
            shading="gouraud",
            cmap="magma",
            vmin=np.percentile(data, 5),
            vmax=np.percentile(data, 99.8),
        )
        ax.set_title(f"{ifo}: whitened spectrogram")
        ax.set_xlabel("time relative to merger (s)")
        ax.grid(False)
    axes[0].set_ylabel("frequency (Hz)")
    fig.colorbar(mesh, ax=axes, label="power (arb. units)", pad=0.02)
    fig.suptitle("GW150914: the frequency sweep (chirp) in the raw data", y=1.0)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 2. Strain -> curvature time series
# --------------------------------------------------------------------------- #
def plot_strain_and_curvature(result: EventResult, path: Path) -> Path:
    style.apply_style()
    lo, hi = EVENT_WINDOW_S
    fig = plt.figure(figsize=(11, 9))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.45, wspace=0.18)
    palette = {"H1": style.H1_COLOR, "L1": style.L1_COLOR}

    for col, (ifo, det) in enumerate(result.detectors.items()):
        t = det.filtered.times - result.event_gps
        sel = (t >= lo) & (t <= hi)
        ax_h = fig.add_subplot(grid[0, col])
        ax_h.plot(t[sel], det.filtered.values[sel] / STRAIN_UNIT, color=palette[ifo], lw=1.2)
        ax_h.set_title(f"{ifo}: strain h(t), band {result.band_hz[0]:.0f}-{result.band_hz[1]:.0f} Hz")
        ax_h.set_ylabel(r"h ($10^{-21}$)")

        ax_r = fig.add_subplot(grid[1, col], sharex=ax_h)
        ax_r.plot(t[sel], det.curvature[sel] / CURVATURE_UNIT, color=palette[ifo], lw=1.2)
        ax_r.axhline(0, color=style.MUTED, lw=0.6)
        ax_r.set_title(r"curvature $R_{0x0x} = -\frac{1}{2c^2}\,\ddot{h}$")
        ax_r.set_ylabel(r"R ($10^{-33}\ \mathrm{m^{-2}}$)")
        ax_r.set_xlabel("time relative to merger (s)")

    ax_o = fig.add_subplot(grid[2, :])
    h1, l1 = result.detectors["H1"].filtered, result.detectors["L1"].filtered
    t_h1 = h1.times - result.event_gps - result.lag_s
    t_l1 = l1.times - result.event_gps
    sel_h1 = (t_h1 >= lo) & (t_h1 <= hi)
    sel_l1 = (t_l1 >= lo) & (t_l1 <= hi)
    ax_o.plot(t_l1[sel_l1], l1.values[sel_l1] / STRAIN_UNIT, color=style.L1_COLOR, lw=1.4, label="L1")
    ax_o.plot(
        t_h1[sel_h1],
        -h1.values[sel_h1] / STRAIN_UNIT,
        color=style.H1_COLOR,
        lw=1.4,
        label=f"H1, inverted and shifted back by {result.lag_s * 1000:.1f} ms",
    )
    ax_o.set_title(
        f"The same signal at both sites (correlation {result.correlation:+.2f})"
    )
    ax_o.set_xlabel("time relative to merger (s)")
    ax_o.set_ylabel(r"h ($10^{-21}$)")
    ax_o.legend(loc="upper left")
    fig.suptitle("GW150914: from measured strain to spacetime curvature", y=0.96)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 3. Test-mass grid in the plane transverse to the wave
# --------------------------------------------------------------------------- #
def plot_transverse_grid(
    result: EventResult, path: Path, ifo: str = "H1", n_snapshots: int = 6
) -> Path:
    style.apply_style()
    det = result.detectors[ifo]
    t = det.filtered.times
    h = det.filtered.values
    peak = det.summary["peak_time_gps"]
    step = 1.5e-3  # about a quarter period at ~170 Hz
    times = peak + step * (np.arange(n_snapshots) - n_snapshots // 2 - 0.5)
    h_at = np.interp(times, t, h)
    r_at = np.interp(times, t, det.curvature)

    max_strain = float(np.max(np.abs(h_at)))
    exaggeration = 0.20 / (0.5 * max_strain)  # largest stretch ~ 20 % of the spacing
    r_norm = _symmetric_norm(r_at)

    lines = np.linspace(-1, 1, 11)
    fine = np.linspace(-1, 1, 60)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(2.6 * n_snapshots, 3.5))
    for ax, t_i, h_i, r_i in zip(axes, times, h_at, r_at, strict=True):
        colour = plt.get_cmap(style.CMAP)(r_norm(r_i))
        segments = []
        for value in lines:  # lines of constant x and constant y
            for fixed_x in (True, False):
                a = np.full_like(fine, value)
                x, y = (a, fine) if fixed_x else (fine, a)
                xd, yd = curvature.transverse_grid_deformation(x, y, h_i, exaggeration)
                segments.append(np.column_stack([xd, yd]))
                ax.plot(x, y, color=style.GRID, lw=0.6, zorder=1)  # undistorted ghost
        ax.add_collection(LineCollection(segments, colors=[colour], linewidths=1.6, zorder=2))
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"t = {(t_i - peak) * 1000:+.1f} ms\n"
            rf"R = {r_i / CURVATURE_UNIT:+.1f}$\cdot10^{{-33}}$ m$^{{-2}}$",
            fontsize=9,
        )
    fig.suptitle(
        f"{ifo}: free test masses in the plane perpendicular to the wave "
        f"(distortion exaggerated by {exaggeration:.0e}; grey = undistorted)",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "line colour = sign and size of the curvature component "
        "(red: positive, blue: negative)",
        ha="center",
        color=style.MUTED,
        fontsize=9,
    )
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 4. 3D lattice of free test masses while the wave passes
# --------------------------------------------------------------------------- #
LATTICE_N = 5  # lines per side of the transverse lattice
Z_SPAN_S = (-0.05, 0.24)
# Colour scale saturates at this fraction of the peak curvature, so the weaker
# inspiral part stays visible next to the merger.
LATTICE_COLOUR_FRACTION = 0.6  # visible stretch of the propagation axis, in light-seconds


def _lattice_profile(det, t_now: float, n_z: int = 500):
    """Strain and curvature along z for a plane wave moving along +z."""
    z = np.linspace(*Z_SPAN_S, n_z) * SPEED_OF_LIGHT_M_S
    h = curvature.plane_wave_field(det.filtered.times, det.filtered.values, z, t_now)
    r = curvature.plane_wave_field(det.filtered.times, det.curvature, z, t_now)
    return z, h, r


def _draw_lattice(ax, det, t_now: float, exaggeration: float, r_limit: float):
    z, h, r = _lattice_profile(det, t_now)
    z_plot = z / 1e6  # thousands of km
    norm = colors.Normalize(vmin=-r_limit, vmax=r_limit)
    cmap = plt.get_cmap(style.CMAP)
    seg_colors = cmap(norm(0.5 * (r[:-1] + r[1:]) / CURVATURE_UNIT))

    axis = np.linspace(-1, 1, LATTICE_N)
    segments, colours = [], []
    for x0 in axis:  # rails along z, one per lattice point
        for y0 in axis:
            xd, yd = curvature.transverse_grid_deformation(
                np.full_like(h, x0), np.full_like(h, y0), h, exaggeration
            )
            points = np.column_stack([z_plot, xd, yd])
            segments.extend(np.stack([points[:-1], points[1:]], axis=1))
            colours.extend(seg_colors)
    ax.add_collection3d(Line3DCollection(segments, colors=colours, linewidths=1.1))

    # cross-section frames: the transverse lattice at regular z positions
    ring = np.concatenate(
        [
            np.column_stack([axis, np.full_like(axis, -1)]),
            np.column_stack([np.full_like(axis, 1), axis]),
            np.column_stack([axis[::-1], np.full_like(axis, 1)]),
            np.column_stack([np.full_like(axis, -1), axis[::-1]]),
        ]
    )
    for k in range(0, len(z), 40):
        xd, yd = curvature.transverse_grid_deformation(ring[:, 0], ring[:, 1], h[k], exaggeration)
        ax.plot(np.full_like(xd, z_plot[k]), xd, yd, color=cmap(norm(r[k] / CURVATURE_UNIT)),
                lw=1.8)


def _style_lattice_axes(ax, title: str):
    ax.set_facecolor(style.BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.09, 0.2, 1.0))
        axis._axinfo["grid"]["color"] = style.GRID
    ax.set_xlim(Z_SPAN_S[0] * SPEED_OF_LIGHT_M_S / 1e6, Z_SPAN_S[1] * SPEED_OF_LIGHT_M_S / 1e6)
    ax.set_ylim(-1.7, 1.7)
    ax.set_zlim(-1.7, 1.7)
    ax.set_box_aspect((3.4, 1.0, 1.0), zoom=1.55)
    ax.view_init(elev=14, azim=-66)
    ax.set_xlabel("propagation z (1000 km)", labelpad=10)
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, pad=0, fontsize=10)


def _lattice_exaggeration(det) -> float:
    return 0.25 / (0.5 * det.summary["peak_strain"])  # peak stretch of 25 %


def plot_spacetime_grid(result: EventResult, path: Path, ifo: str = "H1") -> Path:
    style.apply_style()
    det = result.detectors[ifo]
    exaggeration = _lattice_exaggeration(det)
    r_limit = LATTICE_COLOUR_FRACTION * det.summary["peak_curvature_per_m2"] / CURVATURE_UNIT

    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_axes([0.0, 0.04, 0.88, 0.86], projection="3d")
    t_now = det.summary["peak_time_gps"] + 0.10
    _draw_lattice(ax, det, t_now, exaggeration, r_limit)
    _style_lattice_axes(
        ax,
        f"{ifo}: a lattice of free test masses while the GW150914 wave passes "
        "(snapshot 100 ms after the merger)",
    )
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(-r_limit, r_limit), cmap=style.CMAP)
    fig.colorbar(sm, cax=fig.add_axes([0.9, 0.25, 0.014, 0.5]), label=r"R ($10^{-33}$ m$^{-2}$)")
    fig.text(
        0.5, 0.02,
        f"Wave travels along +z. Lattice distortion exaggerated by {exaggeration:.0e} "
        r"(real strain ~$10^{-21}$). Line colour = curvature $R_{0x0x}$ (colour scale clipped at 60 % of the peak).",
        ha="center", color=style.MUTED, fontsize=9,
    )
    return _save(fig, path)


def make_grid_animation(
    result: EventResult, path: Path, ifo: str = "H1", frames: int = 56
) -> Path:
    style.apply_style()
    det = result.detectors[ifo]
    exaggeration = _lattice_exaggeration(det)
    r_limit = LATTICE_COLOUR_FRACTION * det.summary["peak_curvature_per_m2"] / CURVATURE_UNIT
    peak = det.summary["peak_time_gps"]
    times = np.linspace(peak - 0.05, peak + 0.40, frames)

    fig = plt.figure(figsize=(8.5, 3.6))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.92], projection="3d")

    def draw(frame: int):
        ax.clear()
        _draw_lattice(ax, det, times[frame], exaggeration, r_limit)
        _style_lattice_axes(
            ax,
            f"GW150914 ({ifo}): wave passing a lattice of test masses, "
            f"t = {(times[frame] - peak) * 1000:+.0f} ms (distortion x{exaggeration:.0e})",
        )

    animation = FuncAnimation(fig, draw, frames=frames, interval=70)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=14), dpi=75,
                   savefig_kwargs={"facecolor": style.BG})
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 5. Context: real wave vs. the glitches of the Gravity Spy data set
# --------------------------------------------------------------------------- #
def plot_glitch_context(result: EventResult, glitches: pd.DataFrame, path: Path) -> Path:
    style.apply_style()
    values = glitches["equivalent_curvature_per_m2"]
    values = values[values > 0]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(np.log10(values), bins=60, color=style.L1_COLOR, alpha=0.85)
    ref = result.detectors["H1"].summary["peak_curvature_per_m2"]
    ax.axvline(np.log10(ref), color=style.H1_COLOR, lw=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.text(
        np.log10(ref) + 0.08,
        ax.get_ylim()[1] * 0.93,
        "GW150914 (H1, measured)",
        ha="left",
        va="top",
        color=style.H1_COLOR,
        bbox={"facecolor": style.PANEL, "edgecolor": "none", "pad": 3},
    )
    ax.set_xlabel(r"$\log_{10}$ of peak curvature (m$^{-2}$)")
    ax.set_ylabel("number of glitches")
    ax.set_title(
        f"Equivalent peak curvature of {len(values):,} Gravity Spy glitches (estimate) "
        "vs. the real event"
    )
    return _save(fig, path)
