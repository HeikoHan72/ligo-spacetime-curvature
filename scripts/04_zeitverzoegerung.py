"""
Skript 04: Zeitverzögerung und Inversion zwischen H1 und L1
===========================================================

Frage: Kommt das Signal an beiden Standorten zugleich an? Und mit gleichem Vorzeichen?

Methode: Kreuzkorrelation. Man verschiebt das L1-Signal Schritt für Schritt gegen das
H1-Signal und misst bei jeder Verschiebung, wie ähnlich beide sind. Die Verschiebung mit
der größten Ähnlichkeit ist die Zeitverzögerung. Ist die Ähnlichkeit dort negativ, sind
die Signale invertiert.

Ablauf:
  1. Kreuzkorrelation selbst berechnen (Schleife über Verschiebungen)
  2. Gegen die Projektfunktion processing.time_lag() prüfen
  3. Die Verschiebung grafisch kontrollieren (H1 invertiert und verschoben auf L1 legen)
  4. Physikalische Plausibilität: Lichtlaufzeit zwischen den Standorten

Starten in Spyder: F5 oder Zelle für Zelle mit Strg+Enter. Umgebung "py".
"""

# %% 0. Vorbereitung ----------------------------------------------------------
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ligo_spacetime.config import (  # noqa: E402
    GW150914_GPS,
    MAX_LIGHT_TRAVEL_S,
    PEAK_SEARCH_WINDOW_S,
    SPEED_OF_LIGHT_M_S,
)
from ligo_spacetime.waves import processing, strain  # noqa: E402

strains = strain.load_event()
h1 = processing.bandpass(strains["H1"])
l1 = processing.bandpass(strains["L1"])
dt = h1.dt

# Auswertefenster: +-0,2 s um das Ereignis (wie in der Projekt-Pipeline)
lo, hi = PEAK_SEARCH_WINDOW_S
fenster = (h1.times >= GW150914_GPS + lo) & (h1.times <= GW150914_GPS + hi)
a = h1.values[fenster]  # H1
b = l1.values[fenster]  # L1
print(f"Fenster: {len(a)} Messwerte = {len(a) * dt:.2f} s")

# %% 1. Kreuzkorrelation selbst berechnet ----------------------------------------
# Verschiebung k in Messpunkten. k > 0 bedeutet: a (H1) ist gegenüber b (L1) verspätet.
# Wir bilden die Summe der Produkte a[n + k] * b[n] und teilen durch sqrt(sum(a^2) * sum(b^2)),
# damit der Wert zwischen -1 (exakt invertiert) und +1 (exakt gleich) liegt.
norm = np.sqrt(np.sum(a**2) * np.sum(b**2))


def korrelation_bei(k):
    if k >= 0:
        x, y = a[k:], b[: len(b) - k]
    else:
        x, y = a[:k], b[-k:]
    return np.sum(x * y) / norm


max_k = int(0.015 / dt)  # bis +-15 ms suchen, um den Verlauf zu sehen
ks = np.arange(-max_k, max_k + 1)
werte = np.array([korrelation_bei(k) for k in ks])
lags_ms = ks * dt * 1000

# Erlaubt sind nur Verschiebungen innerhalb der Lichtlaufzeit (10,5 ms)
erlaubt = np.abs(lags_ms) <= MAX_LIGHT_TRAVEL_S * 1000
bester = np.argmax(np.abs(werte[erlaubt]))  # Betrag: auch negative Korrelation zählt
lag_hand = lags_ms[erlaubt][bester] / 1000
korr_hand = werte[erlaubt][bester]
print(f"Handbau: Verzögerung {lag_hand * 1000:.2f} ms, Korrelation {korr_hand:+.3f}")

plt.figure(figsize=(10, 4))
plt.plot(lags_ms, werte, lw=1)
plt.axvspan(-MAX_LIGHT_TRAVEL_S * 1000, MAX_LIGHT_TRAVEL_S * 1000, color="green", alpha=0.1,
            label="physikalisch erlaubt (Lichtlaufzeit)")
plt.plot(lag_hand * 1000, korr_hand, "ro", label=f"Maximum des Betrags: {lag_hand * 1000:.1f} ms")
plt.axhline(0, color="grey", lw=0.5)
plt.xlabel("Verschiebung von H1 gegenüber L1 (ms)")
plt.ylabel("normierte Korrelation")
plt.title("Kreuzkorrelation H1 / L1: negatives Maximum = invertiertes Signal")
plt.legend()
plt.tight_layout()
plt.show()

# %% 2. Kontrolle gegen die Projektfunktion -----------------------------------
lag_projekt, korr_projekt = processing.time_lag(a, b, dt)
print(f"Projekt: Verzögerung {lag_projekt * 1000:.2f} ms, Korrelation {korr_projekt:+.3f}")
assert abs(lag_hand - lag_projekt) < 1e-9 and abs(korr_hand - korr_projekt) < 1e-9
print("OK: Handbau und Projekt stimmen überein")

# %% 3. Grafische Kontrolle -------------------------------------------------------
# H1 mit umgekehrtem Vorzeichen und um die gefundene Verzögerung nach vorn geschoben
# muss auf L1 passen.
t_h1 = h1.times - GW150914_GPS - lag_projekt  # H1-Zeitachse um die Verzögerung nach vorn
sel_h = (t_h1 >= -0.1) & (t_h1 <= 0.05)
t_l1 = l1.times - GW150914_GPS
sel_l = (t_l1 >= -0.1) & (t_l1 <= 0.05)

plt.figure(figsize=(10, 4))
plt.plot(t_l1[sel_l], l1.values[sel_l], label="L1")
plt.plot(t_h1[sel_h], -h1.values[sel_h], label=f"H1 (invertiert, {lag_projekt * 1000:.1f} ms nach vorn)")
plt.xlabel("Zeit relativ zu GW150914 (s)")
plt.ylabel("Strain (gefiltert)")
plt.title("Nach Inversion und Verschiebung liegen die Signale weitgehend übereinander")
plt.legend()
plt.tight_layout()
plt.show()

# %% 4. Ist die Verzögerung physikalisch plausibel? --------------------------
# Hanford und Livingston liegen etwa 3000 km auseinander. Eine Welle, die mit Lichtgeschwindigkeit
# läuft, braucht dafür höchstens etwa 10 ms. Mehr Verzögerung ist nicht möglich.
abstand_m = 3.0e6
print(f"Maximale Laufzeit bei {abstand_m / 1000:.0f} km: {abstand_m / SPEED_OF_LIGHT_M_S * 1000:.1f} ms")
print(f"Gefundene Verzögerung: {lag_projekt * 1000:.1f} ms  (= {lag_projekt * h1.fs:.0f} Messpunkte)")
print(f"Zeitauflösung der Messung: {dt * 1000:.2f} ms")
print(f"Korrelation {korr_projekt:+.2f}: negatives Vorzeichen = H1 und L1 sehen das Signal invertiert")

# %% 5. Ändert die Lichtlaufzeit-Grenze das Ergebnis? --------------------------
# Die Begrenzung der Suche auf +-10,5 ms schließt unphysikalische Verschiebungen aus. Zum
# Vergleich: Suche über das ganze Fenster (+-200 ms). Liefern beide denselben Wert, hängt
# das Ergebnis nicht von der Grenze ab (bei stärkerem Rauschen könnte die freie Suche ein
# zufälliges Maximum außerhalb finden).
lag_frei, korr_frei = processing.time_lag(a, b, dt, max_lag_s=0.2)
print(f"Ohne Begrenzung: {lag_frei * 1000:.1f} ms, Korrelation {korr_frei:+.3f}")
print(f"Mit Begrenzung : {lag_projekt * 1000:.1f} ms, Korrelation {korr_projekt:+.3f}")
