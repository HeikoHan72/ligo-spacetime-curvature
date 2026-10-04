"""
Skript 02: Filtern - das Signal aus dem Rauschen holen
======================================================

Ziel: Den Bandpass-Filter des Projekts Schritt für Schritt nachbauen und zeigen, warum er
nötig ist. Am Ende vergleichen wir unseren Handbau mit der Projektfunktion
processing.bandpass(): Beide müssen dasselbe Ergebnis liefern.

Ablauf des Filters (alles im Frequenzraum):
  A. Tukey-Fenster       -> Ränder sanft ausblenden (kein Sprung zwischen Anfang und Ende)
  B. FFT                 -> Zeitreihe wird zu Spektrum (Anteil je Frequenz)
  C. Filterkurve         -> 1 im Band 43-250 Hz, Flanken 3 Hz, Kerben bei 60/120/180/240 Hz
  D. Spektrum x Kurve    -> unerwünschte Frequenzen werden auf 0 gesetzt
  E. Inverse FFT         -> zurück zur Zeitreihe

Starten in Spyder: F5 oder Zelle für Zelle mit Strg+Enter. Umgebung "py".
"""

# %% 0. Vorbereitung ----------------------------------------------------------
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ligo_spacetime.config import (  # noqa: E402
    BAND_HZ,
    BAND_TAPER_HZ,
    GW150914_GPS,
    NOTCH_HALF_WIDTH_HZ,
    NOTCH_LINES_HZ,
    TUKEY_ALPHA,
)
from ligo_spacetime.waves import processing, strain  # noqa: E402

strains = strain.load_event()
h1 = strains["H1"]
low, high = BAND_HZ
print(f"Band: {low:.0f}-{high:.0f} Hz, Flanken {BAND_TAPER_HZ:.0f} Hz, Kerben bei {NOTCH_LINES_HZ} Hz")

# %% 1. Warum filtern? Das Rauschspektrum --------------------------------------
# Welch-Methode: Das Signal wird in 4-Sekunden-Stücke geteilt, deren Leistungsspektren
# werden gemittelt. sqrt(PSD) = "ASD" (Amplitudendichte): wie laut ist das Rauschen je Frequenz?
fig, ax = plt.subplots(figsize=(10, 5))
for ifo, s in strains.items():
    nperseg = int(4 * s.fs)
    freqs_psd, psd = signal.welch(s.values, fs=s.fs, nperseg=nperseg, noverlap=nperseg // 2)
    ax.loglog(freqs_psd, np.sqrt(psd), lw=0.8, label=ifo)
ax.axvspan(low, high, color="green", alpha=0.12, label="gewähltes Band")
for linie in NOTCH_LINES_HZ:
    ax.axvline(linie, color="red", lw=0.5, alpha=0.6)
ax.set_xlim(10, 1000)
ax.set_xlabel("Frequenz (Hz)")
ax.set_ylabel("ASD (Strain / sqrt(Hz))")
ax.set_title("Rohdaten: unterhalb von ~40 Hz ist das Rauschen um Größenordnungen lauter")
ax.legend()
plt.tight_layout()
plt.show()

# %% 2. Schritt A: Tukey-Fenster ---------------------------------------------
# Die FFT nimmt an, dass sich die Daten periodisch wiederholen. Passen Anfang und Ende
# nicht zusammen, entsteht ein Sprung ("spektrales Lecken"). Das Tukey-Fenster ist in der
# Mitte 1 und blendet nur die Ränder mit einem Cosinus aus.
fenster = signal.windows.tukey(len(h1.values), alpha=TUKEY_ALPHA)
print(f"Fenster: {len(fenster)} Werte, Mitte = {fenster[len(fenster) // 2]}, Rand = {fenster[0]}")

plt.figure(figsize=(10, 3))
plt.plot(h1.times - h1.gps_start, fenster)
plt.xlabel("Zeit seit Dateistart (s)")
plt.ylabel("Fensterwert")
plt.title(f"Tukey-Fenster, alpha = {TUKEY_ALPHA} (je 5 % an beiden Enden ausgeblendet)")
plt.tight_layout()
plt.show()

gefenstert = h1.values * fenster  # elementweise Multiplikation

# %% 3. Schritt B: Fourier-Transformation -------------------------------------
# rfft = FFT für reelle Daten: liefert nur die positiven Frequenzen.
spektrum = np.fft.rfft(gefenstert)
frequenzen = np.fft.rfftfreq(len(gefenstert), h1.dt)
print(f"Spektrum: {len(spektrum)} komplexe Werte, Abstand {frequenzen[1]:.5f} Hz (= 1/32 s)")
print(f"Höchste Frequenz: {frequenzen[-1]:.0f} Hz (= halbe Abtastrate, Nyquist)")

# %% 4. Schritt C: Filterkurve -------------------------------------------------
# _band_gain ist die interne Funktion des Projekts (führender Unterstrich = intern).
# Wir nutzen sie hier nur zur Anschauung.
verstaerkung = processing._band_gain(
    frequenzen, low, high, BAND_TAPER_HZ, NOTCH_LINES_HZ, NOTCH_HALF_WIDTH_HZ
)

fig, achsen = plt.subplots(1, 2, figsize=(12, 3.5))
achsen[0].plot(frequenzen, verstaerkung)
achsen[0].set_xlim(0, 400)
achsen[0].set_title("Filterkurve (0 = sperren, 1 = durchlassen)")
achsen[1].plot(frequenzen, verstaerkung, marker=".", ms=3)
achsen[1].set_xlim(36, 66)
achsen[1].set_title("Zoom: weiche Flanke bei 43 Hz und Kerbe bei 60 Hz")
for achse in achsen:
    achse.set_xlabel("Frequenz (Hz)")
plt.tight_layout()
plt.show()

# %% 5. Schritte D und E: filtern und zurücktransformieren --------------------
gefiltert_hand = np.fft.irfft(spektrum * verstaerkung, n=len(gefenstert))

# %% 6. Kontrolle: stimmt unser Handbau mit dem Projekt überein? -----------------
gefiltert_projekt = processing.bandpass(h1).values
abweichung = np.max(np.abs(gefiltert_hand - gefiltert_projekt))
print(f"Größte Abweichung Handbau vs. Projektfunktion: {abweichung:.1e}")
assert np.allclose(gefiltert_hand, gefiltert_projekt, rtol=0, atol=1e-30)
print("OK: identisch")

# %% 7. Das gefilterte Signal -----------------------------------------------------
fig, achsen = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for achse, (ifo, roh) in zip(achsen, strains.items(), strict=True):
    gefiltert = processing.bandpass(roh)
    fenster_t = (gefiltert.times >= GW150914_GPS - 0.3) & (gefiltert.times <= GW150914_GPS + 0.1)
    achse.plot(gefiltert.times[fenster_t] - GW150914_GPS, gefiltert.values[fenster_t], lw=0.9)
    achse.set_ylabel(f"{ifo} Strain (gefiltert)")
    print(f"{ifo}: Spitzenwert im Fenster {np.max(np.abs(gefiltert.values[fenster_t])):.2e}")
achsen[-1].set_xlabel("Zeit relativ zu GW150914 (s)")
achsen[0].set_title("Nach dem Filtern ist der Chirp sichtbar (Strain ~1e-21)")
plt.tight_layout()
plt.show()
