"""Command line entry point.

    python -m ligo_spacetime [all|gw150914|glitches] [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import DEFAULT_FIGURE_DIR, DEFAULT_GLITCH_CSV, DEFAULT_STRAIN_DIR


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run_glitches(csv_path: Path) -> pd.DataFrame:
    from .glitches import analysis, data, features

    df = data.load_metadata(csv_path)

    _section("GRAVITY SPY: 1. DATA QUALITY")
    for key, value in data.quality_report(df).items():
        print(f"{key:>18}: {value}")

    df = features.add_features(df)

    _section("GRAVITY SPY: 2. TOP 10 GLITCHES BY STRAIN AMPLITUDE")
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

    _section("GRAVITY SPY: 3. DETECTOR COMPARISON (H1 vs. L1)")
    print(
        analysis.detector_summary(df).to_string(
            index=False,
            formatters={"share": "{:.1%}".format, "top_class_share": "{:.1%}".format},
        )
    )

    _section("GRAVITY SPY: 4. CLASS PROFILE (medians)")
    print(
        analysis.class_profile(df).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}",
        )
    )
    return df


def run_gw150914(strain_dir: Path, figure_dir: Path, glitches: pd.DataFrame | None, gif: bool):
    from .viz import figures
    from .waves import pipeline, strain

    strains = strain.load_event(strain_dir)
    result = pipeline.analyze_event(strains)

    _section("GW150914: STRAIN -> CURVATURE")
    print(f"band-pass: {result.band_hz[0]:.0f}-{result.band_hz[1]:.0f} Hz (+ notches at mains lines)")
    for ifo, det in result.detectors.items():
        s = det.summary
        print(f"\n{ifo}  (data-quality flags ok: {det.raw.data_quality_ok})")
        print(f"  peak time (GPS)        : {s['peak_time_gps']:.4f}")
        print(f"  peak strain            : {s['peak_strain']:.2e}  ({s['strain_peak_over_noise']:.1f} x noise rms)")
        print(f"  peak curvature R_0x0x  : {s['peak_curvature_per_m2']:.2e} 1/m^2  "
              f"({s['curvature_peak_over_noise']:.1f} x noise rms)")
        print(f"  radius of curvature    : {s['curvature_radius_m']:.2e} m")
    print(f"\nH1 arrives {result.lag_s * 1000:.1f} ms after L1 "
          f"(max. possible: ~10 ms), correlation {result.correlation:+.2f} (negative = inverted)")

    _section("GW150914: DEPENDENCE ON THE UPPER BAND EDGE")
    sensitivity = pipeline.band_sensitivity(strains)
    print(
        sensitivity.to_string(
            index=False,
            formatters={
                "peak_strain": "{:.2e}".format,
                "peak_curvature_per_m2": "{:.2e}".format,
                "curvature_peak_over_noise": "{:.1f}".format,
            },
        )
    )
    print("The curvature is a 2nd derivative: its peak value depends on the band.")

    _section("FIGURES")
    paths = [
        figures.plot_spectrogram(result, figure_dir / "01_spectrogram.png"),
        figures.plot_strain_and_curvature(result, figure_dir / "02_strain_to_curvature.png"),
        figures.plot_transverse_grid(result, figure_dir / "03_transverse_grid.png"),
        figures.plot_spacetime_grid(result, figure_dir / "04_spacetime_lattice.png"),
    ]
    if gif:
        paths.append(figures.make_grid_animation(result, figure_dir / "05_curvature_wave.gif"))
    if glitches is not None:
        paths.append(figures.plot_glitch_context(result, glitches, figure_dir / "06_glitch_context.png"))
    for path in paths:
        print(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=["all", "gw150914", "glitches"])
    parser.add_argument("--glitch-data", type=Path, default=DEFAULT_GLITCH_CSV,
                        help="Gravity Spy metadata CSV (default: data/raw/)")
    parser.add_argument("--strain-dir", type=Path, default=DEFAULT_STRAIN_DIR,
                        help="folder with the H1/L1 GW150914 HDF5 files (default: data/raw/gw150914/)")
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR,
                        help="output folder for figures (default: figures/)")
    parser.add_argument("--no-gif", action="store_true", help="skip the animation (faster)")
    args = parser.parse_args(argv)

    try:
        glitches = None
        if args.command in ("all", "glitches"):
            glitches = run_glitches(args.glitch_data)
        if args.command in ("all", "gw150914"):
            run_gw150914(args.strain_dir, args.figures, glitches, gif=not args.no_gif)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
