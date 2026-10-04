"""Signal processing: band-pass, spectral derivative, whitening, time lag."""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..config import (
    BAND_HZ,
    BAND_TAPER_HZ,
    MAX_LIGHT_TRAVEL_S,
    NOTCH_HALF_WIDTH_HZ,
    NOTCH_LINES_HZ,
    TUKEY_ALPHA,
)
from .strain import Strain


def _band_gain(
    freqs: np.ndarray,
    low: float,
    high: float,
    taper_hz: float,
    notch_lines: tuple[float, ...],
    notch_half_width: float,
) -> np.ndarray:
    """Frequency response: flat in [low, high], raised-cosine edges, notches."""
    gain = np.zeros_like(freqs)
    rising = (freqs >= low - taper_hz) & (freqs < low)
    gain[rising] = 0.5 * (1 - np.cos(np.pi * (freqs[rising] - (low - taper_hz)) / taper_hz))
    gain[(freqs >= low) & (freqs <= high)] = 1.0
    falling = (freqs > high) & (freqs <= high + taper_hz)
    gain[falling] = 0.5 * (1 + np.cos(np.pi * (freqs[falling] - high) / taper_hz))
    for line in notch_lines:
        gain[np.abs(freqs - line) <= notch_half_width] = 0.0
    return gain


def bandpass(
    strain: Strain,
    low: float = BAND_HZ[0],
    high: float = BAND_HZ[1],
    taper_hz: float = BAND_TAPER_HZ,
    notch_lines: tuple[float, ...] = NOTCH_LINES_HZ,
    notch_half_width: float = NOTCH_HALF_WIDTH_HZ,
    tukey_alpha: float = TUKEY_ALPHA,
) -> Strain:
    """Zero-phase band-pass with notches, applied in the frequency domain.

    The data are first tapered with a Tukey window to limit spectral leakage
    from the very strong low-frequency noise into the band of interest.
    """
    if not 0 < low < high < strain.fs / 2:
        raise ValueError(f"Invalid band ({low}, {high}) Hz for fs={strain.fs} Hz")
    windowed = strain.values * signal.windows.tukey(len(strain.values), alpha=tukey_alpha)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), strain.dt)
    gain = _band_gain(freqs, low, high, taper_hz, notch_lines, notch_half_width)
    filtered = np.fft.irfft(spectrum * gain, n=len(windowed))
    return strain.with_values(filtered)


def spectral_derivative(values: np.ndarray, dt: float, order: int = 1) -> np.ndarray:
    """Derivative of a (band-limited) series via the Fourier transform."""
    spectrum = np.fft.rfft(values)
    freqs = np.fft.rfftfreq(len(values), dt)
    return np.fft.irfft((2j * np.pi * freqs) ** order * spectrum, n=len(values))


def whiten(strain: Strain, low: float = 20.0, high: float = 500.0) -> Strain:
    """Divide the spectrum by the noise amplitude spectral density.

    Used for the spectrogram only. The result is dimensionless, not a strain.
    """
    nperseg = int(4 * strain.fs)
    freqs_psd, psd = signal.welch(
        strain.values, fs=strain.fs, nperseg=nperseg, noverlap=nperseg // 2
    )
    windowed = strain.values * signal.windows.tukey(len(strain.values), alpha=TUKEY_ALPHA)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), strain.dt)
    asd = np.sqrt(np.interp(freqs, freqs_psd, psd))
    gain = _band_gain(freqs, low, high, 3.0, (), 0.0)
    whitened = np.fft.irfft(spectrum / asd * gain, n=len(windowed))
    return strain.with_values(whitened)


def time_lag(
    a: np.ndarray, b: np.ndarray, dt: float, max_lag_s: float = MAX_LIGHT_TRAVEL_S
) -> tuple[float, float]:
    """Lag and normalised correlation between two series.

    A positive lag means ``a`` is delayed relative to ``b``. The search is
    limited to the light travel time between the two detectors. The
    correlation is signed: a negative value means the signals are inverted.
    """
    if len(a) != len(b):
        raise ValueError("Series must have equal length")
    correlation = signal.correlate(a, b, mode="full")
    lags = signal.correlation_lags(len(a), len(b)) * dt
    allowed = np.abs(lags) <= max_lag_s
    best = np.argmax(np.abs(correlation[allowed]))
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(lags[allowed][best]), float(correlation[allowed][best] / norm)
