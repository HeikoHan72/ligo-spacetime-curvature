"""Spacetime curvature of a gravitational wave from the measured strain.

Physics
-------
For a plane gravitational wave in transverse-traceless (TT) gauge the tidal
part of the Riemann tensor is

    R_0i0j = -1/2 * d^2 h_ij / dt^2          (units: 1/s^2, divide by c^2 for 1/m^2)

A LIGO detector measures one combination h(t) = dL/L along its two arms. In the
frame of the arms this gives the components

    R_0x0x = -1/2 * h''(t) / c^2,    R_0y0y = +1/2 * h''(t) / c^2

A single detector cannot measure the full tensor, only this arm-projected
component. The second time derivative amplifies high frequencies by (2 pi f)^2,
so the result is only meaningful after band-pass filtering.
"""

from __future__ import annotations

import numpy as np

from ..config import SPEED_OF_LIGHT_M_S
from .processing import spectral_derivative


def riemann_arm_component(strain_values: np.ndarray, dt: float) -> np.ndarray:
    """R_0x0x(t) in 1/m^2 from a (band-limited) strain series."""
    second_derivative = spectral_derivative(strain_values, dt, order=2)
    return -0.5 * second_derivative / SPEED_OF_LIGHT_M_S**2


def curvature_radius_m(curvature_per_m2: float) -> float:
    """Radius of curvature 1 / sqrt(|R|) in metres."""
    return float(1.0 / np.sqrt(abs(curvature_per_m2)))


def transverse_grid_deformation(
    x: np.ndarray, y: np.ndarray, h: float, exaggeration: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Displace a grid of free test masses in the plane transverse to the wave.

    Geodesic deviation for a '+' polarised wave: x' = x (1 + h/2),
    y' = y (1 - h/2). Real strains (~1e-21) are invisible, so ``exaggeration``
    scales h for display only.
    """
    effective = h * exaggeration
    return x * (1 + effective / 2), y * (1 - effective / 2)


def plane_wave_field(
    times: np.ndarray, curvature: np.ndarray, z_m: np.ndarray, t_now: float
) -> np.ndarray:
    """Curvature along the propagation axis z at time ``t_now``.

    The wave travels at c along +z and the detector sits at z = 0, so
    R(z, t) = R_detector(t - z / c). Outside the recorded interval the
    curvature is set to zero.
    """
    return np.interp(
        t_now - z_m / SPEED_OF_LIGHT_M_S, times, curvature, left=0.0, right=0.0
    )
