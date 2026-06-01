import numpy as np

from eht_inspection.uvfits import build_closure_products_from_coherency


def _mock_scan_result():
    station_list = np.array(["A", "B", "C", "D"])
    allcoh = np.ones((2, 3, 4, 4, 2, 2), dtype=complex)

    # Make the four polarization products distinct but nonzero.
    allcoh[..., 0, 0] = 1.0 + 0.0j
    allcoh[..., 0, 1] = 2.0 + 0.0j
    allcoh[..., 1, 0] = 3.0 + 0.0j
    allcoh[..., 1, 1] = 4.0 + 0.0j

    return {
        "allcoh": allcoh,
        "station_list": station_list,
        "t_unique": np.array([0.0, 1.0]),
        "channel_list": np.array([0, 1, 2]),
        "scan_number": 0,
    }


def test_closure_quantity_shapes_for_mock_array():
    closure = build_closure_products_from_coherency(
        _mock_scan_result(),
        exclude_stations=(),
    )

    assert closure["closure_phase"].shape == (2, 3, 4, 2)
    assert closure["triangle_baseline_phase"].shape == (2, 3, 4, 3, 2)
    assert closure["closure_amp"].shape == (2, 3, 1, 2)
    assert closure["closure_logamp"].shape == (2, 3, 1, 2)
    assert closure["quadrangle_baseline_logamp"].shape == (2, 3, 1, 4, 2)
    assert closure["pol_labels"] == ("RR", "LL")


def test_closure_rejects_too_few_stations():
    scan_result = _mock_scan_result()
    closure_input = {
        **scan_result,
        "station_list": np.array(["A", "B"]),
        "allcoh": scan_result["allcoh"][:, :, :2, :2, :, :],
    }

    import pytest

    with pytest.raises(ValueError):
        build_closure_products_from_coherency(closure_input, exclude_stations=())

