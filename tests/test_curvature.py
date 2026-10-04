"""Tests for the curvature computation and the grid geometry."""

import numpy as np
import pytest

from ligo_spacetime.config import SPEED_OF_LIGHT_M_S as C
from ligo_spacetime.waves import curvature

FS = 4096.0
DT = 1 / FS
T = np.arange(int(32 * FS)) * DT


def test_riemann_component_of_sine():
    """h = A sin(w t)  ->  R = -1/2 h''/c^2 = +A w^2 sin(w t) / (2 c^2)."""
    amplitude, f = 1e-21, 100.0
    h = amplitude * np.sin(2 * np.pi * f * T)
    r = curvature.riemann_arm_component(h, DT)
    expected = 0.5 * amplitude * (2 * np.pi * f) ** 2 * np.sin(2 * np.pi * f * T) / C**2
    np.testing.assert_allclose(r, expected, rtol=1e-6, atol=1e-6 * expected.max())


def test_curvature_radius():
    assert curvature.curvature_radius_m(1e-32) == pytest.approx(1e16)
    assert curvature.curvature_radius_m(-1e-32) == pytest.approx(1e16)


def test_transverse_deformation_zero_strain_is_identity():
    x, y = np.array([-1.0, 0.5]), np.array([0.3, 1.0])
    xd, yd = curvature.transverse_grid_deformation(x, y, 0.0, exaggeration=1e20)
    np.testing.assert_array_equal(xd, x)
    np.testing.assert_array_equal(yd, y)


def test_transverse_deformation_stretches_x_and_compresses_y():
    xd, yd = curvature.transverse_grid_deformation(
        np.array([1.0]), np.array([1.0]), h=1e-21, exaggeration=1e20
    )
    assert xd[0] == pytest.approx(1.05)  # h_eff = 0.1 -> x * (1 + 0.05)
    assert yd[0] == pytest.approx(0.95)


def test_transverse_deformation_preserves_area_to_first_order():
    h_eff = 0.02
    xd, yd = curvature.transverse_grid_deformation(
        np.array([1.0]), np.array([1.0]), h=h_eff, exaggeration=1.0
    )
    assert xd[0] * yd[0] == pytest.approx(1 - h_eff**2 / 4)


def test_plane_wave_field_is_delayed_detector_signal():
    times = np.linspace(0.0, 1.0, 1001)
    series = np.sin(2 * np.pi * 5 * times)
    t_now = 0.6
    z = np.array([0.0, 0.1 * C, 0.3 * C])
    field = curvature.plane_wave_field(times, series, z, t_now)
    np.testing.assert_allclose(
        field,
        [np.sin(2 * np.pi * 5 * 0.6), np.sin(2 * np.pi * 5 * 0.5), np.sin(2 * np.pi * 5 * 0.3)],
        atol=1e-4,
    )


def test_plane_wave_field_zero_outside_recording():
    times = np.linspace(0.0, 1.0, 101)
    field = curvature.plane_wave_field(times, np.ones_like(times), np.array([5.0 * C]), 0.5)
    assert field[0] == 0.0
