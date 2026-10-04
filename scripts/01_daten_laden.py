"""
Skript 01: Daten laden und ansehen
==================================

Ziel: Verstehen, wie die LIGO-Rohdaten von GW150914 aufgebaut sind und wie das
Projekt sie einliest. Noch keine Filterung und keine Krümmung (kommt in Skript 02+).

Starten in Spyder: Datei öffnen und F5 drücken (ganzes Skript) oder mit
Strg+Enter Zelle für Zelle ausführen (Zellen beginnen mit  #%%).
Dafür muss die Umgebung "py" aktiv sein (requirements.txt installiert).
"""

# %% 0. Vorbereitung ----------------------------------------------------------
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Projektordner = ein Ordner über "scripts/". So läuft das Skript unabhängig davon,
# aus welchem Ordner Spyder gestartet wurde.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ligo_spacetime.config import DEFAULT_STRAIN_DIR, GW150914_GPS  # noqa: E402
from ligo_spacetime.waves import strain  # noqa: E402

# %% 1. Welche Dateien gibt es? -------------------------------------------------
# GWOSC liefert je Detektor eine HDF5-Datei mit 32 s Messdaten.
files = sorted(DEFAULT_STRAIN_DIR.glob("*.hdf5"))
for f in files:
    print(f"{f.name}  ({f.stat().st_size / 1e6:.2f} MB)")

# %% 2. Innenleben einer HDF5-Datei -------------------------------------------
# HDF5 ist wie ein Dateisystem in einer Datei: Gruppen (Ordner) und Datensätze.
# visititems() läuft durch alle Einträge und ruft für jeden die Funktion auf.
h1_file = next(f for f in files if "H1" in f.name)


def zeige(name, objekt):
    form = getattr(objekt, "shape", "")
    typ = getattr(objekt, "dtype", "")
    print(f"{name:35s} {str(form):10s} {typ}")


with h5py.File(h1_file, "r") as handle:
    handle.visititems(zeige)

    # Die eigentlichen Messwerte: 131072 Zahlen = 32 s x 4096 Hz
    daten = handle["strain/Strain"]
    print("\nAttribute von strain/Strain:")
    for schluessel, wert in daten.attrs.items():
        print(f"  {schluessel:10s} = {wert}")
    # Xspacing = Abstand zweier Messpunkte in s (1/4096), Xstart = GPS-Startzeit.

    # DQmask: pro Sekunde eine Zahl, deren Bits Qualitätsflags sind
    # (1 = Flag erfüllt, z. B. "Daten ok", "keine Hardware-Injektion").
    maske = handle["quality/simple/DQmask"]
    print("\nDQmask pro Sekunde:", maske[...])
    print("Anzahl Bits:", maske.attrs["Bits"])
    # 127 = 1111111 binär: alle 7 Flags sind in jeder Sekunde gesetzt = Daten in Ordnung.
    print("Beschreibung der Flags:", [b.decode() for b in handle["quality/simple/DQShortnames"][...]])

# %% 3. Dasselbe mit dem Projekt-Loader ---------------------------------------
# load_event() liest H1 und L1, prüft die Daten und gibt Strain-Objekte zurück.
strains = strain.load_event()

for ifo, s in strains.items():
    print(f"\n{ifo}")
    print(f"  Abtastrate        : {s.fs:.0f} Hz")
    print(f"  Messwerte         : {len(s.values)}")
    print(f"  Dauer             : {len(s.values) * s.dt:.1f} s")
    print(f"  Start (GPS)       : {s.gps_start:.0f}")
    print(f"  Standardabw.      : {np.std(s.values):.2e}")
    print(f"  Datenqualität ok  : {s.data_quality_ok}")

# Einordnung: Die Standardabweichung der Rohdaten liegt weit über 1e-21. Das Signal
# von GW150914 (~1e-21) ist in den Rohdaten also gar nicht zu sehen. Deshalb filtern wir
# in Skript 02.

# %% 4. Die Rohdaten ansehen --------------------------------------------------
fig, achsen = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for achse, (ifo, s) in zip(achsen, strains.items(), strict=True):
    achse.plot(s.times - GW150914_GPS, s.values, lw=0.4)
    achse.set_ylabel(f"{ifo} Strain")
    achse.axvline(0, color="red", lw=0.8, ls="--")  # Zeitpunkt des Ereignisses
achsen[-1].set_xlabel("Zeit relativ zu GW150914 (s)")
achsen[0].set_title("Rohdaten: das Signal ist im Rauschen nicht sichtbar")
plt.tight_layout()
plt.show()

# %% 5. Ausschnitt um das Ereignis ----------------------------------------------
fig, achsen = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for achse, (ifo, s) in zip(achsen, strains.items(), strict=True):
    fenster = (s.times >= GW150914_GPS - 0.5) & (s.times <= GW150914_GPS + 0.5)
    achse.plot(s.times[fenster] - GW150914_GPS, s.values[fenster], lw=0.6)
    achse.set_ylabel(f"{ifo} Strain")
achsen[-1].set_xlabel("Zeit relativ zu GW150914 (s)")
achsen[0].set_title("Ausschnitt +-0,5 s: überwiegend niederfrequentes Rauschen")
plt.tight_layout()
plt.show()
