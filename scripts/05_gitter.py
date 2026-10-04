"""
Skript 05: Das verzerrte Gitter
===============================

Ziel: Zeigen, was die Gravitationswelle mit freien Testmassen macht. Wir legen ein
quadratisches Gitter aus Testmassen senkrecht zur Ausbreitungsrichtung der Welle und
verformen es mit dem gemessenen Strain h(t).

Physik: Geodätische Abweichung für eine "+"-polarisierte Welle
    x' = x * (1 + h/2)      (Dehnung in x-Richtung, wenn h > 0)
    y' = y * (1 - h/2)      (gleichzeitig Stauchung in y-Richtung)

WICHTIG: Der echte Strain ist ~1e-21. Die Verformung wäre unsichtbar. Deshalb wird h für
die Darstellung mit einem Faktor multipliziert ("exaggeration"). Das Bild ist eine
Veranschaulichung, keine Messung der Verformung.

Ablauf:
  1. Verformungsformel an einem Punkt prüfen
  2. Verstärkungsfaktor wählen
  3. Gitter zu mehreren Zeitpunkten zeichnen (Farbe = Krümmung)
  4. Ausbreitung entlang der Wellenrichtung: R(z, t) = R_Detektor(t - z/c)

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
from ligo_spacetime.waves import curvature, pipeline, strain  # noqa: E402

ergebnis = pipeline.analyze_event(strain.load_event())
h1 = ergebnis.detectors["H1"]
zeit = h1.filtered.times
h = h1.filtered.values  # gefilterter Strain (Skript 02)
R = h1.curvature  # Krümmung in 1/m^2 (Skript 03)
t_spitze = h1.summary["peak_time_gps"]
print(f"H1: Spitzenzeit {t_spitze:.4f} (GPS), Spitzen-Strain {np.max(np.abs(h)):.2e}")

# %% 1. Die Verformungsformel an einem Punkt -----------------------------------
# Ein Testmassen-Paar 1 m vom Mittelpunkt entfernt. Echte Verschiebung bei h = 1e-21:
h_beispiel = 1e-21
print(f"Echte Verschiebung bei x = 1 m: {1.0 * h_beispiel / 2:.1e} m (Protonradius: 8,4e-16 m)")

# Die Projektfunktion macht dasselbe wie die Formel von Hand:
xs, ys = curvature.transverse_grid_deformation(np.array([1.0]), np.array([1.0]), h=0.1)
print(f"Projekt: x' = {xs[0]:.3f}, y' = {ys[0]:.3f}   (Formel: {1 * (1 + 0.1 / 2):.3f}, {1 * (1 - 0.1 / 2):.3f})")

# %% 2. Verstärkungsfaktor ------------------------------------------------------
# Wir wählen den Faktor so, dass die größte Verformung etwa 10 % beträgt:
# h_eff / 2 = 0.1  ->  Faktor = 0.2 / max|h|
verstaerkung = 0.2 / np.max(np.abs(h))
print(f"Verstärkungsfaktor für die Darstellung: {verstaerkung:.1e}")
print(f"(max. dargestellter Strain: {np.max(np.abs(h)) * verstaerkung:.2f} statt {np.max(np.abs(h)):.1e})")

# %% 3. Gitter zu mehreren Zeitpunkten ------------------------------------------
linien = np.linspace(-1, 1, 11)  # 11 Linien in x und y
fein = np.linspace(-1, 1, 60)  # feine Punktfolge, damit Linien glatt gezeichnet werden

offsets_ms = [-4.5, -3.0, -1.5, 0.0, 1.5, 3.0]  # relativ zur Spitzenzeit
farbskala = plt.get_cmap("coolwarm")
r_max = np.max(np.abs(R[(zeit > t_spitze - 0.01) & (zeit < t_spitze + 0.01)]))

fig, achsen = plt.subplots(1, len(offsets_ms), figsize=(15, 3.2))
for achse, offset in zip(achsen, offsets_ms, strict=True):
    t_i = t_spitze + offset / 1000
    h_i = np.interp(t_i, zeit, h)  # Strain zu diesem Zeitpunkt
    r_i = np.interp(t_i, zeit, R)  # Krümmung zu diesem Zeitpunkt
    farbe = farbskala(0.5 + 0.5 * r_i / r_max)  # rot = positiv, blau = negativ

    for wert in linien:
        # Unverzerrtes Gitter in Grau, verzerrtes Gitter farbig
        achse.plot(np.full_like(fein, wert), fein, color="0.8", lw=0.6)
        achse.plot(fein, np.full_like(fein, wert), color="0.8", lw=0.6)
        # Linie konstanten x (senkrecht) und konstanten y (waagrecht) verformen
        xv, yv = curvature.transverse_grid_deformation(np.full_like(fein, wert), fein, h_i, verstaerkung)
        achse.plot(xv, yv, color=farbe, lw=1.3)
        xh, yh = curvature.transverse_grid_deformation(fein, np.full_like(fein, wert), h_i, verstaerkung)
        achse.plot(xh, yh, color=farbe, lw=1.3)

    achse.set_aspect("equal")
    achse.set_xlim(-1.3, 1.3)
    achse.set_ylim(-1.3, 1.3)
    achse.axis("off")
    achse.set_title(f"t = {offset:+.1f} ms\nR = {r_i * 1e33:+.1f}e-33 1/m²", fontsize=9)
fig.suptitle(
    f"H1: Testmassen senkrecht zur Welle (Verformung um Faktor {verstaerkung:.0e} überhöht; grau = unverzerrt)"
)
plt.tight_layout()
plt.show()

# %% 4. Ausbreitung entlang der Wellenrichtung -----------------------------------
# Die Welle läuft mit c entlang +z. Der Detektor steht bei z = 0. An einem Ort z kommt
# das Signal um z/c später an:   R(z, t) = R_Detektor(t - z/c).
# Eine Momentaufnahme zum Zeitpunkt t_jetzt zeigt deshalb die Zeitreihe "räumlich gespiegelt".
t_jetzt = GW150914_GPS + 0.1  # 100 ms nach dem Merger
z = np.linspace(-1e7, 7e7, 3000)  # -10.000 km ... +70.000 km
feld = curvature.plane_wave_field(zeit, R, z, t_jetzt)

print(f"Länge von 1 ms Signal im Raum: {SPEED_OF_LIGHT_M_S * 1e-3 / 1000:.0f} km")
plt.figure(figsize=(11, 3.5))
plt.plot(z / 1e6, feld * 1e33, lw=0.9, color="C3")
plt.xlabel("Ort z entlang der Ausbreitung (1000 km)")
plt.ylabel("R (1e-33 1/m²)")
plt.title("Momentaufnahme: Krümmung entlang der Wellenrichtung, 100 ms nach dem Merger")
plt.axhline(0, color="grey", lw=0.5)
plt.tight_layout()
plt.show()
