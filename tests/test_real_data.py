"""Checks against the real data. Skipped when the data files are not present."""

import pytest

from ligo_spacetime.config import DEFAULT_GLITCH_CSV, DEFAULT_STRAIN_DIR, GW150914_GPS
from ligo_spacetime.glitches import analysis, data, features
from ligo_spacetime.waves import pipeline, strain

needs_strain = pytest.mark.skipif(
    not DEFAULT_STRAIN_DIR.is_dir() or not list(DEFAULT_STRAIN_DIR.glob("*.hdf5")),
    reason="GW150914 strain files not available in data/raw/gw150914/",
)
needs_glitches = pytest.mark.skipif(
    not DEFAULT_GLITCH_CSV.is_file(), reason="Gravity Spy CSV not available in data/raw/"
)


@pytest.fixture(scope="module")
def event():
    return pipeline.analyze_event(strain.load_event())


@needs_strain
def test_strain_files_are_complete_and_clean():
    for ifo, s in strain.load_event().items():
        assert s.ifo == ifo
        assert s.fs == 4096.0
        assert len(s.values) == 32 * 4096
        assert s.data_quality_ok


@needs_strain
def test_merger_time_matches_event_page(event):
    """The GWOSC event page lists GPS 1126259462; the peak must be within 50 ms of .4."""
    for det in event.detectors.values():
        assert abs(det.summary["peak_time_gps"] - GW150914_GPS) < 0.05


@needs_strain
def test_h1_l1_lag_is_physical_and_signal_is_inverted(event):
    assert 0.004 < event.lag_s < 0.0105  # L1 first, light travel time limit ~10 ms
    assert event.correlation < -0.4  # the two detectors see an inverted signal


@needs_strain
def test_signal_stands_out_of_the_noise(event):
    for det in event.detectors.values():
        s = det.summary
        assert 5e-22 < s["peak_strain"] < 3e-21  # about 1e-21, as published
        assert s["strain_peak_over_noise"] > 5
        assert s["curvature_peak_over_noise"] > 5
        assert 1e-33 < s["peak_curvature_per_m2"] < 1e-31


@needs_glitches
def test_glitch_data_shape_and_features():
    df = features.add_features(data.load_metadata())
    assert len(df) == 7966
    assert df["label"].nunique() == 22
    assert df["search"].nunique() == 1
    assert (df["equivalent_curvature_per_m2"] > 0).all()
    assert df["utc_time"].min().year == 2015
    assert len(analysis.class_profile(df)) == 22
