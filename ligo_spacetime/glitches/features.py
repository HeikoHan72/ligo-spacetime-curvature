"""Feature engineering for glitch metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.time import Time

from ..config import LIGO_ARM_LENGTH_M, PROTON_DIAMETER_M, SPEED_OF_LIGHT_M_S


def gps_to_utc(gps_seconds: pd.Series) -> pd.Series:
    """Convert GPS seconds to UTC timestamps, including leap seconds.

    GPS time does not apply leap seconds. In 2015-2017 GPS is 16-17 s ahead of
    UTC, so a plain 'epoch + seconds' conversion would be off by that amount.
    """
    utc = Time(gps_seconds.to_numpy(), format="gps").utc.to_datetime()
    return pd.Series(pd.to_datetime(utc), index=gps_seconds.index, name="utc_time")


def strain_to_displacement_m(strain: pd.Series) -> pd.Series:
    """Equivalent arm-length change in metres: strain h = dL / L  ->  dL = h * L."""
    return strain * LIGO_ARM_LENGTH_M


def equivalent_curvature_per_m2(amplitude: pd.Series, peak_frequency_hz: pd.Series) -> pd.Series:
    """Order-of-magnitude curvature estimate for a sinusoid-like transient.

    For h(t) = A sin(2 pi f t) the tidal curvature component is
    R = -1/2 * d2h/dt2 / c^2, with peak value  A * (2 pi f)^2 / (2 c^2).

    Glitches are instrument noise, not gravitational waves. The number is the
    curvature a wave with this amplitude and frequency *would* have.
    """
    omega = 2 * np.pi * peak_frequency_hz
    return 0.5 * amplitude * omega**2 / SPEED_OF_LIGHT_M_S**2


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with derived columns.

    Added columns: ``utc_time``, ``displacement_m``,
    ``displacement_proton_diameters``, ``equivalent_curvature_per_m2``.
    """
    out = df.copy()
    out["utc_time"] = gps_to_utc(out["event_time"])
    out["displacement_m"] = strain_to_displacement_m(out["amplitude"])
    out["displacement_proton_diameters"] = out["displacement_m"] / PROTON_DIAMETER_M
    out["equivalent_curvature_per_m2"] = equivalent_curvature_per_m2(
        out["amplitude"], out["peak_frequency"]
    )
    return out
