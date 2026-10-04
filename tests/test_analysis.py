"""Tests for the glitch analysis functions."""

import pandas as pd
import pytest

from ligo_spacetime.glitches import analysis, data, features


@pytest.fixture
def df():
    frame = pd.DataFrame(
        {
            "event_time": [1126259462.4, 1126259500.0, 1126259600.0, 1126259700.0],
            "ifo": ["H1", "H1", "L1", "L1"],
            "label": ["Blip", "Blip", "Whistle", "Low_Frequency_Burst"],
            "amplitude": [1e-21, 5e-20, 2e-19, 1e-22],
            "snr": [10.0, 20.0, 30.0, 40.0],
            "duration": [0.2, 0.3, 0.5, 2.0],
            "peak_frequency": [200.0, 220.0, 1500.0, 10.0],
        }
    )
    return features.add_features(frame)


def test_top_events_sorted_by_amplitude(df):
    top = analysis.top_events_by_amplitude(df, n=2)
    assert list(top["amplitude"]) == [2e-19, 5e-20]


def test_detector_summary(df):
    summary = analysis.detector_summary(df).set_index("ifo")
    assert summary.loc["H1", "glitches"] == 2
    assert summary.loc["H1", "top_class"] == "Blip"
    assert summary["share"].sum() == pytest.approx(1.0)


def test_class_profile_uses_median_and_bands(df):
    profile = analysis.class_profile(df).set_index("label")
    assert profile.loc["Blip", "median_peak_frequency_hz"] == pytest.approx(210.0)
    assert profile.loc["Whistle", "frequency_band"] == "high (>1000 Hz)"
    assert profile.loc["Low_Frequency_Burst", "frequency_band"] == "low (<50 Hz)"


def test_quality_report(df):
    report = data.quality_report(df)
    assert report["rows"] == 4
    assert report["duplicate_rows"] == 0


def test_load_metadata_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_metadata(tmp_path / "nope.csv")


def test_load_metadata_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"label": ["Blip"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        data.load_metadata(path)
