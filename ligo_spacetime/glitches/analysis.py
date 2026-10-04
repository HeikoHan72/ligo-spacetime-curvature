"""Pure analysis functions for glitch metadata. They return DataFrames and never print."""

from __future__ import annotations

import pandas as pd

from ..config import HIGH_FREQUENCY_MIN_HZ, LOW_FREQUENCY_MAX_HZ


def top_events_by_amplitude(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The ``n`` events with the largest strain amplitude."""
    columns = [
        "label",
        "ifo",
        "amplitude",
        "snr",
        "displacement_proton_diameters",
        "equivalent_curvature_per_m2",
        "utc_time",
    ]
    return df.nlargest(n, "amplitude")[columns].reset_index(drop=True)


def detector_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Glitch count, share and most frequent class per detector."""
    rows = []
    for ifo, group in df.groupby("ifo"):
        label_counts = group["label"].value_counts()
        rows.append(
            {
                "ifo": ifo,
                "glitches": len(group),
                "share": len(group) / len(df),
                "top_class": label_counts.idxmax(),
                "top_class_share": label_counts.max() / len(group),
            }
        )
    return pd.DataFrame(rows)


def class_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and median SNR, duration and peak frequency.

    Medians are used because SNR and duration are strongly right-skewed
    (a few extreme events dominate the mean).
    """
    profile = (
        df.groupby("label")
        .agg(
            n=("label", "size"),
            median_snr=("snr", "median"),
            median_duration_s=("duration", "median"),
            median_peak_frequency_hz=("peak_frequency", "median"),
        )
        .sort_values("n", ascending=False)
    )
    profile["frequency_band"] = pd.cut(
        profile["median_peak_frequency_hz"],
        bins=[-float("inf"), LOW_FREQUENCY_MAX_HZ, HIGH_FREQUENCY_MIN_HZ, float("inf")],
        labels=["low (<50 Hz)", "mid (50-1000 Hz)", "high (>1000 Hz)"],
    )
    return profile.reset_index()
