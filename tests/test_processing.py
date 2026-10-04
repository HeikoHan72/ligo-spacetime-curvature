"""Tests for filtering, differentiation and the time-lag estimate."""

import numpy as np
import pytest

from ligo_spacetime.waves import processing
from ligo_spacetime.waves.strain import Strain

FS = 4096.0
DT = 1 / FS
N = int(32 * FS)
T = np.arange(N) * DT


def _strain(values):
    return Strain(ifo="H1", gps_start=0.0, dt=DT, values=values)


def _amplitude_at(values, freq, lo=8.0, hi=24.0):
    """Amplitude of the component at ``freq`` in the central part (lock-in)."""
    sel = (T >= lo) & (T < hi)
    t = T[sel]
    x = values[sel]
    i = np.mean(x * np.sin(2 * np.pi * freq * t)) * 2
    q = np.mean(x * np.cos(2 * np.pi * freq * t)) * 2
    return float(np.hypot(i, q))


def test_bandpass_keeps_in_band_and_removes_out_of_band():
    amplitude = 1e-21
    signal_ = sum(amplitude * np.sin(2 * np.pi * f * T) for f in (10.0, 100.0, 400.0))
    out = processing.bandpass(_strain(signal_)).values
    assert _amplitude_at(out, 100.0) == pytest.approx(amplitude, rel=0.03)
    assert _amplitude_at(out, 10.0) < 0.01 * amplitude
    assert _amplitude_at(out, 400.0) < 0.01 * amplitude


def test_bandpass_removes_mains_line():
    values = 1e-21 * np.sin(2 * np.pi * 60.0 * T)
    out = processing.bandpass(_strain(values)).values
    assert _amplitude_at(out, 60.0) < 0.01e-21


def test_bandpass_rejects_invalid_band():
    with pytest.raises(ValueError):
        processing.bandpass(_strain(np.zeros(N)), low=300.0, high=100.0)


def test_spectral_derivative_of_sine():
    f = 100.0  # integer number of cycles in the window -> exact
    h = np.sin(2 * np.pi * f * T)
    d1 = processing.spectral_derivative(h, DT, order=1)
    d2 = processing.spectral_derivative(h, DT, order=2)
    np.testing.assert_allclose(d1, 2 * np.pi * f * np.cos(2 * np.pi * f * T), atol=1e-6 * 2 * np.pi * f)
    np.testing.assert_allclose(d2, -((2 * np.pi * f) ** 2) * h, atol=1e-6 * (2 * np.pi * f) ** 2)


def test_time_lag_recovers_known_shift_and_inversion():
    rng = np.random.default_rng(1)
    b = rng.standard_normal(2048)
    shift = 29  # samples, ~7.1 ms
    a = -np.roll(b, shift)  # a is b delayed by 'shift' samples and inverted
    lag, corr = processing.time_lag(a, b, DT)
    assert lag == pytest.approx(shift * DT)
    assert corr == pytest.approx(-1.0, abs=0.02)


def test_time_lag_respects_light_travel_limit():
    rng = np.random.default_rng(2)
    b = rng.standard_normal(2048)
    a = np.roll(b, 200)  # 49 ms: outside the physically allowed window
    lag, _ = processing.time_lag(a, b, DT)
    assert abs(lag) <= 0.0105


def test_time_lag_requires_equal_length():
    with pytest.raises(ValueError):
        processing.time_lag(np.zeros(10), np.zeros(11), DT)
