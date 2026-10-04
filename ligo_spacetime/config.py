"""Physical constants and project-wide settings."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_GLITCH_CSV = DATA_DIR / "trainingset_v1d1_metadata.csv"
DEFAULT_STRAIN_DIR = DATA_DIR / "gw150914"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures"

# --- Physical constants ----------------------------------------------------
SPEED_OF_LIGHT_M_S = 299_792_458.0
LIGO_ARM_LENGTH_M = 4000.0

# Proton charge radius (~0.84 fm). The *diameter* is twice that value.
PROTON_RADIUS_M = 0.84e-15
PROTON_DIAMETER_M = 2 * PROTON_RADIUS_M

# --- GW150914 analysis settings --------------------------------------------
# Approximate merger time (GPS seconds). The GWOSC event page lists 1126259462;
# the fraction is only used to centre the analysis window.
GW150914_GPS = 1126259462.4

# Band-pass applied before differentiating. The signal sweeps from ~35 Hz to
# ~250 Hz. Below ~43 Hz the data are dominated by calibration lines and
# technical noise, so the lower edge is set above them.
BAND_HZ = (43.0, 250.0)
BAND_TAPER_HZ = 3.0
# Narrow spectral lines (mains power and harmonics) that are removed.
NOTCH_LINES_HZ = (60.0, 120.0, 180.0, 240.0)
NOTCH_HALF_WIDTH_HZ = 1.0
# Fraction of the data window that is tapered (Tukey window).
TUKEY_ALPHA = 0.1

# The two sites are ~3000 km apart: the signal can differ by at most ~10 ms.
MAX_LIGHT_TRAVEL_S = 0.0105

# Window around the merger used for peak search and plots (seconds).
EVENT_WINDOW_S = (-0.30, 0.10)
PEAK_SEARCH_WINDOW_S = (-0.20, 0.20)
# Pre-event interval used as noise baseline: starts this many seconds after
# the file start and ends this many seconds before the event.
BASELINE_START_OFFSET_S = 3.0
BASELINE_END_BEFORE_EVENT_S = 2.0

# --- Gravity Spy (glitch) settings -----------------------------------------
REQUIRED_COLUMNS = (
    "event_time",
    "ifo",
    "duration",
    "peak_frequency",
    "amplitude",
    "snr",
    "label",
)
LOW_FREQUENCY_MAX_HZ = 50.0
HIGH_FREQUENCY_MIN_HZ = 1000.0
