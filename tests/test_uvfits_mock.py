import numpy as np

from eht_inspection.uvfits import (
    _load_obs_uvfits_result_dict,
    build_scan_coherency_matrix,
    build_scan_coherency_matrix_from_uvfits,
    visibility_arrays_to_dataframe,
)


def test_build_scan_coherency_matrix_shapes_and_reverse_baselines():
    times = np.zeros(6)
    scan_ids = np.zeros(6, dtype=int)
    t1 = np.array(["A", "A", "A", "B", "B", "C"])
    t2 = np.array(["B", "C", "D", "C", "D", "D"])
    u = np.arange(6.0)
    v = np.arange(6.0) + 10.0

    rr = np.ones((6, 2), dtype=complex)
    rl = np.ones((6, 2), dtype=complex) * 2
    lr = np.ones((6, 2), dtype=complex) * 3
    ll = np.ones((6, 2), dtype=complex) * 4

    result = build_scan_coherency_matrix(0, scan_ids, times, t1, t2, u, v, rr, rl, lr, ll)

    assert result["allcoh"].shape == (1, 2, 4, 4, 2, 2)
    assert result["station_list"].tolist() == ["A", "B", "C", "D"]
    np.testing.assert_allclose(result["allcoh"][0, :, 1, 0], np.swapaxes(result["allcoh"][0, :, 0, 1], -1, -2).conjugate())
    assert result["allu"][1, 0] == -result["allu"][0, 1]


def test_visibility_arrays_to_dataframe_long_format():
    rr = np.ones((2, 2), dtype=complex)
    rl = rr * 2
    lr = rr * 3
    ll = rr * 4

    df = visibility_arrays_to_dataframe(
        times=[1.0, 2.0],
        t1=["A", "A"],
        t2=["B", "C"],
        u=[10.0, 20.0],
        v=[0.0, 1.0],
        rr=rr,
        rl=rl,
        lr=lr,
        ll=ll,
        scan_ids=[0, 1],
    )

    assert df.shape == (16, 12)
    assert set(df["polarization"]) == {"RR", "RL", "LR", "LL"}
    assert df["channel"].tolist().count(0) == 8
    np.testing.assert_allclose(df[df["polarization"] == "LL"]["amp"].unique(), [4.0])


def test_load_obs_uvfits_result_dict_keys():
    arrays = [np.array([i]) for i in range(14)]
    result = _load_obs_uvfits_result_dict(*arrays)

    assert list(result) == [
        "times",
        "t1",
        "t2",
        "u",
        "v",
        "rr",
        "rl",
        "lr",
        "ll",
        "rrsigma",
        "rlsigma",
        "lrsigma",
        "llsigma",
        "scantable",
    ]
    assert result["times"] is arrays[0]
    assert result["scantable"] is arrays[-1]


def test_build_scan_coherency_matrix_from_uvfits_assigns_scan_ids(monkeypatch):
    times = np.array([0.1, 0.2, 1.1])
    loaded = {
        "times": times,
        "t1": np.array(["A", "A", "B"]),
        "t2": np.array(["B", "C", "C"]),
        "u": np.arange(3.0),
        "v": np.arange(3.0) + 10.0,
        "rr": np.ones((3, 1), dtype=complex),
        "rl": np.ones((3, 1), dtype=complex) * 2,
        "lr": np.ones((3, 1), dtype=complex) * 3,
        "ll": np.ones((3, 1), dtype=complex) * 4,
        "rrsigma": np.ones((3, 1)),
        "rlsigma": np.ones((3, 1)),
        "lrsigma": np.ones((3, 1)),
        "llsigma": np.ones((3, 1)),
        "scantable": np.array([[0.0, 1.0], [1.0, 2.0]]),
    }
    calls = {}

    def fake_load_obs_uvfits(filename, **kwargs):
        calls["load"] = (filename, kwargs)
        return loaded

    def fake_build_scan_coherency_matrix(scannum, scan_ids, *args, **kwargs):
        calls["build"] = (scannum, np.asarray(scan_ids), args, kwargs)
        return {"scan_ids": np.asarray(scan_ids), "scan_number": scannum}

    monkeypatch.setattr(
        "eht_inspection.uvfits.load_obs_uvfits",
        fake_load_obs_uvfits,
    )
    monkeypatch.setattr(
        "eht_inspection.uvfits.build_scan_coherency_matrix",
        fake_build_scan_coherency_matrix,
    )

    result = build_scan_coherency_matrix_from_uvfits(
        "example.uvfits",
        1,
        start_index=1,
        IF=[0],
        channel=[0],
        fill_missing=5.0 + 0.0j,
        conjugate_reverse=False,
    )

    assert calls["load"][0] == "example.uvfits"
    assert calls["load"][1]["return_dict"] is True
    assert calls["load"][1]["IF"] == [0]
    assert calls["load"][1]["channel"] == [0]
    np.testing.assert_array_equal(result["scan_ids"], [1, 1, 2])
    assert result["scan_number"] == 1
    assert calls["build"][3]["fill_missing"] == 5.0 + 0.0j
    assert calls["build"][3]["conjugate_reverse"] is False


def test_build_scan_coherency_matrix_from_uvfits_requires_scans(monkeypatch):
    loaded = {
        "times": np.array([0.1]),
        "t1": np.array(["A"]),
        "t2": np.array(["B"]),
        "u": np.array([0.0]),
        "v": np.array([0.0]),
        "rr": np.ones((1, 1), dtype=complex),
        "rl": np.ones((1, 1), dtype=complex),
        "lr": np.ones((1, 1), dtype=complex),
        "ll": np.ones((1, 1), dtype=complex),
        "rrsigma": np.ones((1, 1)),
        "rlsigma": np.ones((1, 1)),
        "lrsigma": np.ones((1, 1)),
        "llsigma": np.ones((1, 1)),
        "scantable": None,
    }

    monkeypatch.setattr(
        "eht_inspection.uvfits.load_obs_uvfits",
        lambda *args, **kwargs: loaded,
    )

    try:
        build_scan_coherency_matrix_from_uvfits("example.uvfits", 0)
    except ValueError as exc:
        assert "no scan intervals" in str(exc)
    else:
        raise AssertionError("Expected ValueError when scan intervals are unavailable")
