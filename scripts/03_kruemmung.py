"""
Skript 03: Aus dem Strain die Raumzeit-Krümmung berechnen
=========================================================

Physik (kurz): Für eine ebene Gravitationswelle ist der Gezeitenanteil des Riemann-Tensors

    R_0x0x(t) = -1/2 * h''(t) / c^2        [Einheit: 1/m^2]

h''(t) ist die zweite Ableitung des Strains nach der Zeit. Ein Detektor misst nur eine
Komponente (entlang seiner Arme), nicht den ganzen Tensor.

Ablauf hier:
  1. Zweite Ableitung im Frequenzraum selbst berechnen (und gegen das Projekt prüfen)
  2. Krümmung daraus bilden
  3. Vergleich mit der Differenzen-Methode np.gradient
  4. Warum man vorher filtern muss
  5. Kennzahlen und Diagramm

Starten in Spyder: F5 oder Zelle für Zelle mit Strg+Enter. Umgebung "py".
"""

# %% 0. Vorbereitung ----------------------------------------------------------
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ligo_spacetime.config import GW150914_GPS, SPEED_OF_LIGHT_M_S  # noqa: E402
from ligo_spacetime.waves import curvature, pipeline, processing, strain  # noqa: E402

c = SPEED_OF_LIGHT_M_S
strains = strain.load_event()
raw = strains["H1"]
filtered = processing.bandpass(raw)  # wie in Skript 02

# %% 1. Zweite Ableitung im Frequenzraum ---------------------------------------
# Ableitungssatz der Fourier-Transformation:
#   Ableiten nach t  <->  Multiplizieren des Spektrums mit (i * 2 * pi * f)
#   zweimal Ableiten <->  Multiplizieren mit (i * 2 * pi * f)^2 = -(2 * pi * f)^2
h = filtered.values
dt = filtered.dt

spektrum = np.fft.rfft(h)
frequenzen = np.fft.rfftfreq(len(h), dt)
h_zweite_hand = np.fft.irfft((2j * np.pi * frequenzen) ** 2 * spektrum, n=len(h))

h_zweite_projekt = processing.spectral_derivative(h, dt, order=2)
print("Handbau == Projektfunktion:", np.allclose(h_zweite_hand, h_zweite_projekt, rtol=0, atol=1e-30))

# %% 2. Krümmung -----------------------------------------------------------------
# R = -1/2 * h'' / c^2.   c^2 ist ~9e16, deshalb werden die Zahlen winzig (1e-33).
kruemmung_hand = -0.5 * h_zweite_hand / c**2
kruemmung_projekt = curvature.riemann_arm_component(h, dt)
print("Krümmung Handbau == Projekt:", np.allclose(kruemmung_hand, kruemmung_projekt, rtol=0, atol=1e-45))

# Überschlag zur Kontrolle: Für h = A*sin(2*pi*f*t) ist die Krümmungsspitze A*(2*pi*f)^2 / (2*c^2).
A = np.max(np.abs(h[(filtered.times > GW150914_GPS - 0.2) & (filtered.times < GW150914_GPS + 0.2)]))
for f in (100.0, 150.0, 170.0):
    print(f"  Überschlag mit A = {A:.2e}, f = {f:.0f} Hz: R = {A * (2 * np.pi * f) ** 2 / (2 * c**2):.1e} 1/m^2")

# %% 3. Vergleich mit der Differenzen-Methode ----------------------------------
# np.gradient nähert die Ableitung durch Differenzen benachbarter Werte an.
# Zweimal angewandt ergibt das eine Näherung der zweiten Ableitung.
h_zweite_diff = np.gradient(np.gradient(h, dt), dt)

fenster = (filtered.times >= GW150914_GPS - 0.3) & (filtered.times <= GW150914_GPS + 0.1)
spitze_spektral = np.max(np.abs(h_zweite_hand[fenster]))
spitze_diff = np.max(np.abs(h_zweite_diff[fenster]))
print(f"Spitze von |h''| spektral   : {spitze_spektral:.3e}")
print(f"Spitze von |h''| np.gradient: {spitze_diff:.3e}  (Verhältnis {spitze_diff / spitze_spektral:.3f})")
# Differenzen sind ein Tiefpass: sie unterschätzen hohe Frequenzen etwas. Die spektrale
# Methode ist für bandbegrenzte Daten exakt.

# %% 4. Warum vorher filtern? ---------------------------------------------------
# Die zweite Ableitung gewichtet jede Frequenz mit (2*pi*f)^2: hohe Frequenzen werden
# massiv verstärkt. Ohne Filter dominiert das Rauschen.
kruemmung_roh = curvature.riemann_arm_component(raw.values, raw.dt)
fenster_roh = (raw.times >= GW150914_GPS - 0.3) & (raw.times <= GW150914_GPS + 0.1)
print(f"Spitze |R| ohne Filter: {np.max(np.abs(kruemmung_roh[fenster_roh])):.2e} 1/m^2")
print(f"Spitze |R| mit Filter : {np.max(np.abs(kruemmung_projekt[fenster])):.2e} 1/m^2")

# %% 5. Kennzahlen aus der Projekt-Pipeline -------------------------------------
ergebnis = pipeline.analyze_event(strains)
for ifo, det in ergebnis.detectors.items():
    s = det.summary
    print(f"\n{ifo}")
    print(f"  Spitzenzeit (GPS)   : {s['peak_time_gps']:.4f}")
    print(f"  Spitzen-Strain      : {s['peak_strain']:.2e}  ({s['strain_peak_over_noise']:.1f} x Rauschen)")
    print(f"  Spitzen-Krümmung    : {s['peak_curvature_per_m2']:.2e} 1/m^2  ({s['curvature_peak_over_noise']:.1f} x Rauschen)")
    print(f"  Krümmungsradius     : {s['curvature_radius_m']:.2e} m  (= {s['curvature_radius_m'] / 9.4607e15:.2f} Lichtjahre)")
print(f"\nH1 kommt {ergebnis.lag_s * 1000:.1f} ms nach L1, Korrelation {ergebnis.correlation:+.2f}")

# %% 6. Diagramm: Strain und Krümmung ------------------------------------------
fig, achsen = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
for zeile, (ifo, det) in enumerate(ergebnis.detectors.items()):
    t = det.filtered.times - GW150914_GPS
    sel = (t >= -0.3) & (t <= 0.1)
    achsen[zeile, 0].plot(t[sel], det.filtered.values[sel], lw=0.9)
    achsen[zeile, 0].set_ylabel(f"{ifo}  h(t)")
    achsen[zeile, 1].plot(t[sel], det.curvature[sel] * 1e33, lw=0.9, color="C3")
    achsen[zeile, 1].set_ylabel(f"{ifo}  R (1e-33 1/m^2)")
achsen[0, 0].set_title("Strain h(t) (gefiltert)")
achsen[0, 1].set_title("Krümmung R_0x0x(t) = -1/2 h'' / c^2")
for achse in achsen[-1]:
    achse.set_xlabel("Zeit relativ zu GW150914 (s)")
plt.tight_layout()
plt.show()
