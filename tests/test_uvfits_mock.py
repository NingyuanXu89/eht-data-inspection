import numpy as np

from eht_inspection.uvfits import build_scan_coherency_matrix, visibility_arrays_to_dataframe


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

