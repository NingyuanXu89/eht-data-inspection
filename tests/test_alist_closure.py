"""Stage 3 alist fringe-fit consistency checks."""

import itertools

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from eht_inspection.alist import (
    compute_alist_closure_triangles,
    plot_alist_closure_vs_scan,
    summarize_alist_closure_outliers,
)

STATIONS = ("J", "L", "P", "S", "X")
DELAY = {"J": 0.01, "L": -0.02, "P": 0.03, "S": 0.015, "X": -0.005}
RATE = {"J": 1.0, "L": -2.0, "P": 0.5, "S": 1.5, "X": -0.7}


def _stage3_alist(nscan=3, reverse=()):
    rows = []
    for scan in range(nscan):
        for a, b in itertools.combinations(STATIONS, 2):
            baseline = a + b
            sign = 1.0
            if baseline in reverse:
                baseline, sign = b + a, -1.0
            for pol in ("RR", "LL"):
                rows.append({
                    "source": "M87",
                    "scan_no": scan,
                    "datetime": pd.Timestamp("2022-03-27") + pd.Timedelta(minutes=10 * scan),
                    "baseline": baseline,
                    "polarization": pol,
                    "snr": 100.0,
                    "mbdelay": sign * (DELAY[b] - DELAY[a]),
                    "delay_rate": sign * (RATE[b] - RATE[a]),
                    "ambiguity": 0.0325,
                })
    return pd.DataFrame(rows)


def test_station_based_values_close_and_reverse_baselines_are_handled():
    triangles = compute_alist_closure_triangles(
        _stage3_alist(reverse=("JL", "PX")), exclude_stations=()
    )
    assert len(triangles) == 3 * 2 * 10
    assert np.allclose(triangles["closure_mbdelay"], 0.0, atol=1e-9)
    assert np.allclose(triangles["closure_delay_rate"], 0.0, atol=1e-9)
    assert triangles.loc[triangles["triangle"] == "J-L-P", "baselines"].iloc[0] == "LJ,LP,JP"


def test_orientation_can_change_between_scans():
    data = _stage3_alist()
    changed = (data["scan_no"] == 1) & (data["baseline"] == "JL")
    data.loc[changed, "baseline"] = "LJ"
    data.loc[changed, ["mbdelay", "delay_rate"]] *= -1
    triangles = compute_alist_closure_triangles(data, exclude_stations=())
    assert np.allclose(triangles["closure_mbdelay"], 0.0, atol=1e-9)
    assert np.allclose(triangles["closure_delay_rate"], 0.0, atol=1e-9)
    assert triangles.loc[
        (triangles["scan_no"] == 1) & (triangles["triangle"] == "J-L-P"),
        "baselines",
    ].iloc[0] == "LJ,LP,JP"


def test_sources_are_never_mixed_when_scan_keys_collide():
    data = _stage3_alist(nscan=1)
    other = data.copy()
    other["source"] = "SGRA"
    other["snr"] = 200.0
    other["mbdelay"] += 0.1
    triangles = compute_alist_closure_triangles(
        pd.concat([data, other], ignore_index=True), exclude_stations=()
    )
    assert set(triangles["source"]) == {"M87", "SGRA"}
    assert len(triangles) == 2 * 2 * 10
    assert np.allclose(triangles.loc[triangles["source"] == "M87", "closure_mbdelay"], 0)


def test_bad_baseline_flags_triangles_and_ambiguity_ratio():
    data = _stage3_alist(nscan=6)
    bad = (data["baseline"] == "LP") & (data["scan_no"] == 2)
    data.loc[bad, "mbdelay"] += 0.0325
    triangles = compute_alist_closure_triangles(data, exclude_stations=())
    affected = triangles[(triangles["scan_no"] == 2) & triangles["triangle"].isin(
        ("J-L-P", "L-P-S", "L-P-X")
    )]
    assert np.allclose(affected["closure_mbdelay_over_ambiguity"].abs(), 1.0)
    triangles["closure_mbdelay"] += np.random.default_rng(0).normal(0, 1e-5, len(triangles))
    outliers = summarize_alist_closure_outliers(triangles)
    flagged = outliers["closure_mbdelay"]
    assert set(flagged["triangle"]) == {"J-L-P", "L-P-S", "L-P-X"}
    assert set(flagged["scan_no"]) == {2}
    assert set(outliers["stations"].iloc[:2]["station"]) == {"L", "P"}


def test_missing_baseline_snr_cut_and_supported_quantities():
    data = _stage3_alist()
    data = data[data["baseline"] != "JL"]
    triangles = compute_alist_closure_triangles(data, exclude_stations=("X",))
    assert not triangles["triangle"].str.contains("X").any()
    assert not triangles["triangle"].isin(("J-L-P", "J-L-S")).any()
    data.loc[data["baseline"] == "PS", "snr"] = 3.0
    cut = compute_alist_closure_triangles(data, exclude_stations=(), snr_min=5)
    assert not cut["baselines"].str.contains("PS").any()
    with pytest.raises(ValueError):
        compute_alist_closure_triangles(data, quantities=("resid_phas",))
    with pytest.raises(ValueError):
        compute_alist_closure_triangles(data, pols=("RL",))


def test_uvfits_style_plot_separates_rr_and_ll():
    triangles = compute_alist_closure_triangles(_stage3_alist(), exclude_stations=())
    fig, axs = plot_alist_closure_vs_scan(triangles, quantity="closure_mbdelay")
    titles = [ax.get_title() for ax in axs.flat if ax.get_title()]
    assert len(titles) == 20
    assert titles[:2] == ["J-L-P RR", "J-L-P LL"]
    assert "J-L-S RR (trivial)" in titles
    assert axs.flat[0].lines[0].get_color() == "blue"
    assert axs.flat[1].lines[0].get_color() == "red"
    assert axs.flat[0].lines[1].get_linestyle() == "--"
    fig, axs = plot_alist_closure_vs_scan(
        triangles, quantity="closure_delay_rate", groups=("J-L-P",),
        pols=("RR", "LL"), x_col="datetime",
    )
    assert [ax.get_title() for ax in axs.flat[:2]] == ["J-L-P RR", "J-L-P LL"]
