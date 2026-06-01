import numpy as np
import pytest

from eht_inspection.utils import (
    get_baselines_from_station_list,
    get_subplot_grid,
    quadrangle_names_from_station_list,
    scan_ids_from_intervals,
    triangle_names_from_station_list,
    unwrap_phase,
    wrap_phase,
)


def test_baseline_naming_without_autocorr():
    assert get_baselines_from_station_list(["A", "B", "C"]) == [
        "A-B",
        "A-C",
        "B-C",
    ]


def test_baseline_naming_with_autocorr():
    assert get_baselines_from_station_list(["A", "B"], include_autocorr=True) == [
        "A-A",
        "A-B",
        "B-B",
    ]


def test_triangle_and_quadrangle_naming_with_exclusions():
    stations = ["AA", "B", "C", "D", "E"]
    assert triangle_names_from_station_list(stations, exclude_stations=("AA",)) == [
        ("B", "C", "D"),
        ("B", "C", "E"),
        ("B", "D", "E"),
        ("C", "D", "E"),
    ]
    assert quadrangle_names_from_station_list(stations, exclude_stations=("AA",)) == [
        ("B", "C", "D", "E")
    ]


def test_subplot_grid_helper():
    assert get_subplot_grid(1) == (1, 1)
    assert get_subplot_grid(4) == (1, 4)
    assert get_subplot_grid(5) == (2, 3)
    assert get_subplot_grid(10) == (3, 4)
    with pytest.raises(ValueError):
        get_subplot_grid(0)


def test_phase_wrap_degrees_and_radians():
    np.testing.assert_allclose(wrap_phase(np.array([190, -190, 360])), [-170, 170, 0])
    np.testing.assert_allclose(wrap_phase(np.pi + 0.1, rad=True), -np.pi + 0.1)
    np.testing.assert_allclose(
        wrap_phase(np.pi + 0.1, rad=True, return_degrees=True),
        np.rad2deg(-np.pi + 0.1),
    )


def test_phase_unwrap_degrees_and_radians():
    wrapped_deg = np.array([170, -170, -160])
    np.testing.assert_allclose(unwrap_phase(wrapped_deg), [170, 190, 200])

    wrapped_rad = np.deg2rad(wrapped_deg)
    np.testing.assert_allclose(unwrap_phase(wrapped_rad, rad=True), np.deg2rad([170, 190, 200]))


def test_scan_ids_from_intervals():
    scan_ids = scan_ids_from_intervals([0.1, 0.9, 1.0, 1.5, 2.5], [(0, 1), (1, 2)])
    assert scan_ids.tolist() == [0, 0, 1, 1, -1]

