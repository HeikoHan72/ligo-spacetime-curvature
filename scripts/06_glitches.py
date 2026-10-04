"""
Skript 06: Glitch-Analyse (Gravity Spy) und Einordnung von GW150914
===================================================================

Glitches sind kurze Störsignale ("Rauschtransienten") in den LIGO-Detektoren. Der Gravity-
Spy-Datensatz enthält 7.966 davon (H1 und L1, 2015-2017), jeweils mit einer Klasse (22
Klassen). Dieses Skript zeigt, wie man den Datensatz prüft, aufbereitet und auswertet, und
ordnet am Ende das echte Signal GW150914 in die Glitch-Verteilung ein.

Ablauf:
  1. Daten laden und ansehen
  2. Datenqualität prüfen
  3. Neue Spalten (Features) berechnen: UTC-Zeit, Verschiebung, äquivalente Krümmung
  4. Auswertungen: stärkste Glitches, H1 gegen L1, Klassenprofil
  5. Diagramme und die Einordnung von GW150914

Starten in Spyder: F5 oder Zelle für Zelle mit Strg+Enter. Umgebung "py".
"""

# %% 0. Vorbereitung ----------------------------------------------------------
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ligo_spacetime.glitches import analysis, data, features  # noqa: E402
from ligo_spacetime.waves import pipeline, strain  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# %% 1. Daten laden und ansehen -----------------------------------------------
# load_metadata() liest die CSV, prüft Pflichtspalten und wandelt Zahlenspalten um.
df = data.load_metadata()
print(f"{df.shape[0]} Zeilen, {df.shape[1]} Spalten")
wichtig = ["event_time", "ifo", "duration", "peak_frequency", "amplitude", "snr", "label", "sample_type"]
print(df[wichtig].head())

# %% 2. Klassen und Detektoren -------------------------------------------------
print(df["label"].value_counts())
print()
print(df["ifo"].value_counts())

# %% 3. Datenqualität -------------------------------------------------------------
for schluessel, wert in data.quality_report(df).items():
    print(f"{schluessel:>18}: {wert}")
# Hinweis: In der Spalte "search" standen "Omicron" und "OMICRON" gemischt. load_metadata()
# vereinheitlicht das (str.title()). Ohne das würden es zwei Gruppen.

# %% 4. Warum GPS-Zeit nicht "Startdatum + Sekunden" ist -----------------------
# GPS-Zeit zählt seit dem 6.1.1980 ohne Schaltsekunden. 2015 liegt GPS deshalb 17 s vor UTC.
gps = 1126259462.0  # GW150914
einfach = datetime(1980, 1, 6) + timedelta(seconds=gps)
richtig = features.gps_to_utc(pd.Series([gps])).iloc[0]
print(f"Einfache Rechnung : {einfach}")
print(f"Mit Schaltsekunden: {richtig}")
print(f"Unterschied       : {(einfach - richtig.to_pydatetime()).total_seconds():.0f} s")

# %% 5. Neue Spalten (Features) -----------------------------------------------
df = features.add_features(df)
neu = ["utc_time", "displacement_m", "displacement_proton_diameters", "equivalent_curvature_per_m2"]
print(df[neu].head())
# displacement_m                  = Strain * 4000 m (Armlänge)
# displacement_proton_diameters   = Verschiebung / Protondurchmesser (1,68e-15 m)
# equivalent_curvature_per_m2     = A * (2*pi*f)^2 / (2*c^2): Krümmung, die eine Sinuswelle mit
#                                   dieser Amplitude und Frequenz hätte (nur eine Abschätzung)

# %% 6. Die 10 stärksten Glitches ----------------------------------------------
top = analysis.top_events_by_amplitude(df)
print(
    top.to_string(
        formatters={
            "amplitude": "{:.2e}".format,
            "snr": "{:.1f}".format,
            "displacement_proton_diameters": "{:.1f}".format,
            "equivalent_curvature_per_m2": "{:.1e}".format,
            "utc_time": lambda t: t.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
)

# %% 7. H1 gegen L1 und Klassenprofil -------------------------------------------
print(
    analysis.detector_summary(df).to_string(
        index=False, formatters={"share": "{:.1%}".format, "top_class_share": "{:.1%}".format}
    )
)
print()
# Median statt Mittelwert: SNR und Dauer sind stark rechtsschief (wenige Extremwerte).
print(analysis.class_profile(df).to_string(index=False, float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"))

# %% 8. Aufteilung in Training / Validierung / Test ---------------------------
# Der Datensatz bringt schon eine Aufteilung für Machine Learning mit (Spalte sample_type).
print(df["sample_type"].value_counts())

# %% 9. Diagramm: Klassen je Detektor -------------------------------------------
tabelle = pd.crosstab(df["label"], df["ifo"])
tabelle = tabelle.loc[tabelle.sum(axis=1).sort_values().index]  # nach Gesamtzahl sortieren
tabelle.plot.barh(stacked=True, figsize=(9, 7))
plt.xlabel("Anzahl Glitches")
plt.title("Gravity Spy: Glitches je Klasse und Detektor")
plt.tight_layout()
plt.show()

# %% 10. GW150914 im Vergleich zu den Glitches -----------------------------------
# Wie liegt die gemessene Krümmung von GW150914 (H1) in der Verteilung der Glitches?
ergebnis = pipeline.analyze_event(strain.load_event())
r_gw = ergebnis.detectors["H1"].summary["peak_curvature_per_m2"]
werte = df["equivalent_curvature_per_m2"]
anteil = (werte < r_gw).mean()
print(f"GW150914 (H1): {r_gw:.2e} 1/m^2")
print(f"{anteil:.0%} der Glitches haben eine kleinere äquivalente Krümmung.")

plt.figure(figsize=(9, 4))
plt.hist(np.log10(werte[werte > 0]), bins=60, alpha=0.85)
plt.axvline(np.log10(r_gw), color="red", lw=2, label="GW150914 (H1, gemessen)")
plt.xlabel("log10 der Spitzenkrümmung (1/m^2)")
plt.ylabel("Anzahl Glitches")
plt.title("Äquivalente Krümmung der Glitches (Abschätzung) im Vergleich zum echten Ereignis")
plt.legend()
plt.tight_layout()
plt.show()
# Aussage: GW150914 liegt mitten in der Glitch-Verteilung. Amplitude oder Krümmung allein
# trennt Signal und Störung also nicht. Dafür braucht man die Form des Signals (Gravity Spy
# klassifiziert Spektrogramm-Bilder).
