"""End-to-end analysis of one event: strain -> filtered strain -> curvature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    BAND_HZ,
    BASELINE_END_BEFORE_EVENT_S,
    BASELINE_START_OFFSET_S,
    GW150914_GPS,
    PEAK_SEARCH_WINDOW_S,
)
from . import curvature, processing
from .strain import Strain


@dataclass(frozen=True)
class DetectorResult:
    ifo: str
    raw: Strain
    filtered: Strain
    curvature: np.ndarray  # R_0x0x(t) in 1/m^2
    summary: dict


@dataclass(frozen=True)
class EventResult:
    event_gps: float
    band_hz: tuple[float, float]
    detectors: dict[str, DetectorResult]
    lag_s: float  # H1 relative to L1 (positive: H1 arrives later)
    correlation: float  # signed; negative = inverted


def _summarise(
    filtered: Strain, curv: np.ndarray, event_gps: float
) -> dict:
    t = filtered.times
    h = filtered.values
    lo, hi = PEAK_SEARCH_WINDOW_S
    window = (t >= event_gps + lo) & (t <= event_gps + hi)
    baseline = (t >= t[0] + BASELINE_START_OFFSET_S) & (
        t <= event_gps - BASELINE_END_BEFORE_EVENT_S
    )
    i_h = np.flatnonzero(window)[np.argmax(np.abs(h[window]))]
    i_r = np.flatnonzero(window)[np.argmax(np.abs(curv[window]))]
    noise_h = float(np.std(h[baseline]))
    noise_r = float(np.std(curv[baseline]))
    peak_r = float(np.abs(curv[i_r]))
    return {
        "peak_time_gps": float(t[i_h]),
        "peak_strain": float(np.abs(h[i_h])),
        "noise_rms_strain": noise_h,
        "strain_peak_over_noise": float(np.abs(h[i_h]) / noise_h),
        "peak_curvature_per_m2": peak_r,
        "noise_rms_curvature_per_m2": noise_r,
        "curvature_peak_over_noise": peak_r / noise_r,
        "curvature_radius_m": curvature.curvature_radius_m(peak_r),
    }


def analyze_event(
    strains: dict[str, Strain],
    event_gps: float = GW150914_GPS,
    band_hz: tuple[float, float] = BAND_HZ,
) -> EventResult:
    """Filter both detectors, compute the curvature and the H1-L1 time lag."""
    detectors: dict[str, DetectorResult] = {}
    for ifo, raw in strains.items():
        filtered = processing.bandpass(raw, low=band_hz[0], high=band_hz[1])
        curv = curvature.riemann_arm_component(filtered.values, filtered.dt)
        detectors[ifo] = DetectorResult(
            ifo=ifo,
            raw=raw,
            filtered=filtered,
            curvature=curv,
            summary=_summarise(filtered, curv, event_gps),
        )

    h1, l1 = detectors["H1"].filtered, detectors["L1"].filtered
    lo, hi = PEAK_SEARCH_WINDOW_S
    window = (h1.times >= event_gps + lo) & (h1.times <= event_gps + hi)
    lag, corr = processing.time_lag(h1.values[window], l1.values[window], h1.dt)
    return EventResult(
        event_gps=event_gps,
        band_hz=band_hz,
        detectors=detectors,
        lag_s=lag,
        correlation=corr,
    )


def band_sensitivity(
    strains: dict[str, Strain],
    upper_edges_hz: tuple[float, ...] = (200.0, 250.0, 300.0),
    event_gps: float = GW150914_GPS,
) -> pd.DataFrame:
    """How the peak curvature depends on the upper band edge.

    The curvature is a second derivative, so it weights high frequencies
    strongly. Showing several band choices makes that dependence explicit.
    """
    rows = []
    for upper in upper_edges_hz:
        result = analyze_event(strains, event_gps, (BAND_HZ[0], upper))
        for ifo, det in result.detectors.items():
            rows.append(
                {
                    "band_hz": f"{BAND_HZ[0]:.0f}-{upper:.0f}",
                    "ifo": ifo,
                    "peak_strain": det.summary["peak_strain"],
                    "peak_curvature_per_m2": det.summary["peak_curvature_per_m2"],
                    "curvature_peak_over_noise": det.summary["curvature_peak_over_noise"],
                }
            )
    return pd.DataFrame(rows)
