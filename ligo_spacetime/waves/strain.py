"""Loading GWOSC strain files (HDF5)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np

from ..config import DEFAULT_STRAIN_DIR


@dataclass(frozen=True)
class Strain:
    """A strain time series h(t) = dL / L of one detector."""

    ifo: str
    gps_start: float
    dt: float
    values: np.ndarray
    data_quality_ok: bool = True

    @property
    def fs(self) -> float:
        """Sampling rate in Hz."""
        return 1.0 / self.dt

    @property
    def times(self) -> np.ndarray:
        """GPS time of every sample."""
        return self.gps_start + np.arange(len(self.values)) * self.dt

    def with_values(self, values: np.ndarray) -> "Strain":
        return replace(self, values=values)


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def load_strain(path: str | Path) -> Strain:
    """Read one GWOSC HDF5 strain file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file has no strain dataset or contains non-finite values.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Strain file not found: {path}")

    with h5py.File(path, "r") as handle:
        if "strain/Strain" not in handle:
            raise ValueError(f"{path.name}: dataset 'strain/Strain' is missing")
        dataset = handle["strain/Strain"]
        values = np.asarray(dataset[...], dtype=float)
        dt = float(dataset.attrs["Xspacing"])
        gps_start = float(dataset.attrs["Xstart"])
        ifo = _decode(handle["meta/Detector"][()]) if "meta/Detector" in handle else "??"
        # All data-quality flags set in every second of the file?
        quality_ok = True
        if "quality/simple/DQmask" in handle:
            mask = np.asarray(handle["quality/simple/DQmask"][...], dtype=np.uint32)
            bits = int(handle["quality/simple/DQmask"].attrs.get("Bits", 0))
            full = (1 << bits) - 1 if bits else 0
            quality_ok = bool(np.all((mask & full) == full)) if full else True

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name}: strain contains NaN or inf")

    return Strain(
        ifo=ifo, gps_start=gps_start, dt=dt, values=values, data_quality_ok=quality_ok
    )


def load_event(directory: str | Path = DEFAULT_STRAIN_DIR) -> dict[str, Strain]:
    """Load the H1 and L1 strain files found in ``directory``."""
    directory = Path(directory)
    strains: dict[str, Strain] = {}
    for ifo in ("H1", "L1"):
        matches = sorted(directory.glob(f"*{ifo}*.hdf5")) if directory.is_dir() else []
        if not matches:
            raise FileNotFoundError(
                f"No {ifo} strain file (*{ifo}*.hdf5) in {directory}\n"
                "Download the 4096 Hz / 32 s HDF5 files for H1 and L1 from "
                "https://gwosc.org/events/GW150914/ (see data/README.md)."
            )
        strains[ifo] = load_strain(matches[0])
    return strains
