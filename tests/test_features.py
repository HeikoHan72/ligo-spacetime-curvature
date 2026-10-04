"""Tests for the glitch feature engineering."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from ligo_spacetime.config import PROTON_DIAMETER_M
from ligo_spacetime.glitches import features


def test_gps_to_utc_gw150914():
    """GWOSC event page: GPS 1126259462 is 2015-09-14 09:50:45 UTC."""
    result = features.gps_to_utc(pd.Series([1126259462.0]))
    assert result.iloc[0] == pd.Timestamp("2015-09-14 09:50:45")


def test_gps_to_utc_applies_leap_seconds():
    """A naive 'epoch + seconds' conversion is 17 s off in September 2015."""
    gps = 1126259462.4
    naive = datetime(1980, 1, 6) + timedelta(seconds=gps)
    correct = features.gps_to_utc(pd.Series([gps])).iloc[0]
    assert (naive - correct.to_pydatetime()).total_seconds() == pytest.approx(17.0)


def test_proton_constant_is_diameter():
    assert PROTON_DIAMETER_M == pytest.approx(1.68e-15)


def test_equivalent_curvature_formula():
    """Peak of -1/2 h'' / c^2 for h = A sin(2 pi f t) is A (2 pi f)^2 / (2 c^2)."""
    value = features.equivalent_curvature_per_m2(
        pd.Series([1e-21]), pd.Series([100.0])
    ).iloc[0]
    assert value == pytest.approx(2.196e-33, rel=1e-3)


def test_add_features_columns_and_values():
    df = pd.DataFrame(
        {
            "event_time": [1126259462.4],
            "amplitude": [2.1e-19],
            "peak_frequency": [100.0],
        }
    )
    out = features.add_features(df)
    assert out["displacement_m"].iloc[0] == pytest.approx(8.4e-16)
    assert out["displacement_proton_diameters"].iloc[0] == pytest.approx(0.5)
    assert out["equivalent_curvature_per_m2"].iloc[0] > 0
    assert "utc_time" not in df.columns  # input is not modified
