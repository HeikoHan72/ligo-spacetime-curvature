"""Loading, validation and quality checks for the Gravity Spy metadata CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DEFAULT_GLITCH_CSV, REQUIRED_COLUMNS

NUMERIC_COLUMNS = ("event_time", "duration", "peak_frequency", "amplitude", "snr")


def load_metadata(path: str | Path = DEFAULT_GLITCH_CSV) -> pd.DataFrame:
    """Read the metadata CSV and validate its structure.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (message explains where to put it).
    ValueError
        If required columns are missing or numeric columns cannot be parsed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Place 'trainingset_v1d1_metadata.csv' in data/raw/ "
            "or pass the path with --glitch-data."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")
        newly_missing = int(converted.isna().sum() - df[column].isna().sum())
        if newly_missing:
            raise ValueError(
                f"Column '{column}' has {newly_missing} values that are not numeric."
            )
        df[column] = converted

    # The 'search' column mixes 'Omicron' and 'OMICRON'.
    if "search" in df.columns:
        df["search"] = df["search"].str.title()

    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Return basic data-quality facts about the loaded table."""
    counts = df["label"].value_counts()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) == 1],
        "classes": int(df["label"].nunique()),
        "smallest_class": (counts.idxmin(), int(counts.min())),
        "largest_class": (counts.idxmax(), int(counts.max())),
    }
