"""Small synthetic UVFITS fixtures and analytic bandpass SNR checks."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

from eht_inspection.uvfits import (
    build_scan_coherency_matrix,
    build_scan_coherency_matrix_from_uvfits,
    load_obs_uvfits,
    plot_scan_bandpass_all_baselines,
    summarize_scan_bandpass,
)


def _uvfits(weights, with_nx=True):
    """Three records, three IFs, one channel/IF; no external science data."""
    nr, ni, npol = weights.shape
    values = np.zeros((nr, 1, 1, ni, 1, npol, 3))
    values[:, 0, 0, :, 0, :, 0] = np.arange(1, nr + 1)[:, None, None]
    values[:, 0, 0, :, 0, :, 2] = weights
    primary = fits.GroupsHDU(fits.GroupData(
        values, bitpix=-64,
        parnames=["UU", "VV", "DATE", "_DATE", "BASELINE", "INTTIM"],
        pardata=[np.arange(nr), np.arange(nr) + 10, np.full(nr, 2461000.5),
                 np.array([0.1, 0.2, 0.6])[:nr] / 24,
                 np.array([258, 259, 258])[:nr], np.ones(nr)],
    ))
    primary.header.update({
        "CTYPE3": "STOKES", "CRVAL3": -1, "CDELT3": -1,
        "CTYPE4": "FREQ", "CRVAL4": 230e9, "CDELT4": 58e6,
        "CTYPE5": "IF", "OBSRA": 180.0, "OBSDEC": 12.0, "OBJECT": "M87",
    })
    antenna = fits.BinTableHDU.from_columns([
        fits.Column(name="ANNAME", format="8A", array=["AX", "GL", "LM"]),
        fits.Column(name="NOSTA", format="1J", array=[1, 2, 3]),
        fits.Column(name="STABXYZ", format="3D", array=np.zeros((3, 3))),
    ], name="AIPS AN")
    antenna.header["FREQ"] = 230e9
    hdus = [primary, antenna]
    if with_nx:
        hdus.append(fits.BinTableHDU.from_columns([
            fits.Column(name="TIME", format="1D", array=np.array([0.25, 0.75]) / 24),
            fits.Column(name="TIME INTERVAL", format="1D", array=np.array([0.5, 0.5]) / 24),
            fits.Column(name="START VIS", format="1J", array=[1, 3]),
            fits.Column(name="END VIS", format="1J", array=[2, 3]),
        ], name="AIPS NX"))
    return fits.HDUList(hdus)


def test_flags_and_dropped_records_stay_aligned(capsys):
    weights = np.full((3, 3, 4), 4.0)
    weights[0, 2, 2] = 0       # Partial IF cross-hand flag.
    weights[1, :, :2] = -1    # Both parallel hands: drop record 1.
    weights[2, 1, 1] = np.nan
    with _uvfits(weights) as hdus:
        loaded = load_obs_uvfits(hdus, return_dict=True)
    output = capsys.readouterr().out
    assert "Flagged samples: 8/36 (22.22%)" in output
    assert "Affected records: 3/3 (100.00%)" in output
    assert "Dropped records: 1/3 (33.33%)" in output
    assert "AX-GL RL scan 0: 1/3 samples (33.33%)" in output
    assert "IF 2; channel 0; time 0.100000-0.100000 h" in output
    assert "AX-LM RR scan 0: 3/3 samples (100.00%)" in output
    assert "AX-GL LL scan 1: 1/3 samples (33.33%)" in output
    assert "removing flagged data" not in output
    np.testing.assert_allclose(loaded["times"], [0.1, 0.6], atol=1e-8)
    assert loaded["t2"].tolist() == ["GL", "GL"]
    for pol in ("rr", "rl", "lr", "ll"):
        assert loaded[pol].shape == loaded[pol + "sigma"].shape == (2, 3)
    np.testing.assert_allclose(loaded["rr"], [[1, 1, 1], [3, 3, 3]])
    np.testing.assert_allclose(loaded["rrsigma"], 0.5)
    assert np.isnan(loaded["rl"][0, 2])
    assert np.isnan(loaded["llsigma"][1, 1])
    scan = build_scan_coherency_matrix(
        0, np.zeros(2), *[loaded[k] for k in
                         ("times", "t1", "t2", "u", "v", "rr", "rl", "lr", "ll")]
    )
    assert scan["allcoh"].shape[:2] == (2, 3)


def test_selected_flags_and_unknown_scan(capsys):
    weights = np.ones((3, 3, 4))
    weights[0, 0, 0] = 0      # Outside the selection.
    weights[2, 2, 3] = -1
    with _uvfits(weights, with_nx=False) as hdus:
        load_obs_uvfits(hdus, IF=[2])
    output = capsys.readouterr().out
    assert "Flagged samples: 1/12 (8.33%)" in output
    assert "Affected records: 1/3 (33.33%)" in output
    assert "Dropped records: 0/3 (0.00%)" in output
    assert "AX-GL LR scan unknown: 1/2 samples (50.00%)" in output
    assert "IF 2; channel 0; time 0.600000-0.600000 h" in output
    with _uvfits(weights) as hdus:
        load_obs_uvfits(hdus, IF=[1])
    assert "UVFITS flags" not in capsys.readouterr().out


@pytest.mark.parametrize("npol,force", [(1, None), (4, "R"), (4, "L"), (4, "LR")])
def test_absent_or_forced_products_are_not_file_flags(npol, force, capsys):
    with _uvfits(np.ones((3, 3, npol))) as hdus:
        result = load_obs_uvfits(hdus, force_singlepol=force)
    assert len(result) == 13
    assert "UVFITS flags" not in capsys.readouterr().out


def test_forcing_does_not_hide_original_flags(capsys):
    weights = np.ones((3, 3, 4))
    weights[0, 0, 1] = 0
    with _uvfits(weights) as hdus:
        load_obs_uvfits(hdus, force_singlepol="R")
    output = capsys.readouterr().out
    assert "Flagged samples: 1/36 (2.78%)" in output
    assert "AX-GL LL scan 0: 1/3 samples" in output
    assert "Dropped records: 0/3" in output


@pytest.mark.parametrize("print_flag_summary", [True, False])
def test_all_flagged_reports_before_error(capsys, print_flag_summary):
    with _uvfits(np.zeros((3, 3, 4))) as hdus:
        with pytest.raises(Exception, match="No unflagged RR or LL"):
            load_obs_uvfits(hdus, print_flag_summary=print_flag_summary)
    output = capsys.readouterr().out
    if print_flag_summary:
        assert "Flagged samples: 36/36 (100.00%)" in output
        assert "Dropped records: 3/3 (100.00%)" in output
    else:
        assert "UVFITS flags" not in output


@pytest.mark.parametrize("use_wrapper", [False, True])
def test_flag_summary_switch_preserves_data(use_wrapper, capsys):
    weights = np.ones((3, 3, 4))
    weights[0, 2, 2] = 0
    weights[1, :, :2] = 0
    results = []
    for enabled in (True, False):
        with _uvfits(weights) as hdus:
            if use_wrapper:
                result = build_scan_coherency_matrix_from_uvfits(
                    hdus, 0, unaveraged=True, print_flag_summary=enabled,
                )
            else:
                result = load_obs_uvfits(
                    hdus, return_dict=True, print_flag_summary=enabled,
                )
        output = capsys.readouterr().out
        assert ("UVFITS flags" in output) == enabled
        assert ("Flagged samples:" in output) == enabled
        assert ("AX-GL RL scan 0:" in output) == enabled
        results.append(result)
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        np.testing.assert_equal(results[0][key], results[1][key])


def _scan(vis=None, sigmas=None, **kwargs):
    if vis is None:
        vis = {pol: np.ones((2, 2), dtype=complex) for pol in ("rr", "rl", "lr", "ll")}
    return build_scan_coherency_matrix(
        0, [0, 0], [0.1, 0.2], ["AX", "AX"], ["GL", "GL"], [1, 2], [3, 4],
        vis["rr"], vis["rl"], vis["lr"], vis["ll"], sigmas=sigmas, **kwargs,
    )


def test_uncertainties_reverse_transpose_and_missing_entries():
    sigmas = {pol: np.full((2, 2), k + 1.0)
              for k, pol in enumerate(("rr", "rl", "lr", "ll"))}
    scan = _scan(sigmas=sigmas, unaveraged=True)
    expected = np.array([[1, 2], [3, 4]])
    np.testing.assert_allclose(scan["allsigma"][:, :, 0, 1], np.broadcast_to(expected, (2, 2, 2, 2)))
    np.testing.assert_allclose(scan["allsigma"][:, :, 1, 0], np.broadcast_to(expected.T, (2, 2, 2, 2)))
    assert np.isnan(scan["allsigma"][:, :, 0, 0]).all()
    assert scan["unaveraged"] is True
    assert "allsigma" not in _scan()
    assert np.isnan(_scan(sigmas=sigmas, conjugate_reverse=False)["allsigma"][:, :, 1, 0]).all()


@pytest.mark.parametrize("sigmas,message", [
    ({"rr": np.ones((2, 2))}, "must contain"),
    ({pol: np.ones((1, 2)) for pol in ("rr", "rl", "lr", "ll")}, "must match"),
])
def test_uncertainty_validation(sigmas, message):
    with pytest.raises(ValueError, match=message):
        _scan(sigmas=sigmas)


@pytest.mark.parametrize("var", ["amp", "phase"])
def test_snr_title_follows_legend_and_preserves_curves(var):
    expected = [11, 9.5, 1.5, 1.2]
    vis = {pol: np.full((2, 2), snr / np.sqrt(2), dtype=complex)
           for pol, snr in zip(("rr", "rl", "lr", "ll"), expected)}
    sigmas = {pol: np.ones((2, 2)) for pol in vis}
    scan = _scan(vis, sigmas, unaveraged=True)
    default_fig, default_ax = plot_scan_bandpass_all_baselines(scan, var=var)
    snr_fig, snr_ax = plot_scan_bandpass_all_baselines(scan, var=var, show_snr=True)
    try:
        assert default_ax[0, 0].get_title() == "AX-GL"
        assert snr_ax[0, 0].get_title() == "AX-GL: 11, 9.5, 1.5, 1.2"
        assert snr_ax[0, 0].get_legend_handles_labels()[1] == ["RR", "RL", "LR", "LL"]
        for old, new in zip(default_ax[0, 0].lines, snr_ax[0, 0].lines):
            np.testing.assert_equal(old.get_ydata(), new.get_ydata())
            np.testing.assert_equal(old.get_xdata(), new.get_xdata())
    finally:
        plt.close(default_fig)
        plt.close(snr_fig)


def test_snr_unequal_errors_missing_data_and_coherent_cancellation():
    vis = {pol: np.full((2, 2), 5 + 0j) for pol in ("rr", "rl", "lr", "ll")}
    sigmas = {pol: np.ones((2, 2)) for pol in vis}
    sigmas["rr"][:] = [[3, 3], [4, 4]]   # mean sigma=2.5, SNR=2.
    vis["rl"][:] = [[2, 2], [-2, -2]]    # Coherent cancellation: SNR=0.
    vis["lr"][:] = [[np.nan, 3], [3, 0]] # Each IF has one plotted sample.
    sigmas["lr"][:] = [[np.nan, 1], [1, np.inf]]
    sigmas["ll"][:] = [[np.nan, 0], [1, np.inf]]
    scan = _scan(vis, sigmas, unaveraged=True)
    fig, axs = plot_scan_bandpass_all_baselines(scan, var="amp", show_snr=True)
    try:
        assert axs[0, 0].get_title() == "AX-GL: 2, 0, 3, N/A"
        np.testing.assert_allclose(axs[0, 0].lines[0].get_ydata(), [5, 5])
        np.testing.assert_allclose(axs[0, 0].lines[1].get_ydata(), [0, 0])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mode,message", [
    ("averaged", "unaveraged=True"), ("no_time_average", "average_over_time=True"),
    ("no_sigma", "uncertainties"), ("single_bin", "multiple frequency bins"),
    ("bad_shape", "same shape"),
])
def test_invalid_snr_modes(mode, message):
    sigmas = {pol: np.ones((2, 2)) for pol in ("rr", "rl", "lr", "ll")}
    scan = _scan(sigmas=sigmas, unaveraged=True)
    if mode == "averaged":
        del scan["unaveraged"]
    elif mode == "no_sigma":
        del scan["allsigma"]
    elif mode == "single_bin":
        scan["allcoh"] = scan["allcoh"][:, :1]
    elif mode == "bad_shape":
        scan["allsigma"] = scan["allsigma"][:, :1]
    with pytest.raises(ValueError, match=message):
        plot_scan_bandpass_all_baselines(
            scan, show_snr=True, average_over_time=mode != "no_time_average",
        )


def test_wrapper_carries_loader_sigmas_only_when_requested():
    weights = np.full((3, 3, 4), 4.0)
    with _uvfits(weights) as hdus:
        scan = build_scan_coherency_matrix_from_uvfits(hdus, 1, unaveraged=True)
    assert scan["unaveraged"] is True
    np.testing.assert_allclose(scan["allsigma"][0, :, 0, 1], 0.5)
    with _uvfits(weights) as hdus:
        default = build_scan_coherency_matrix_from_uvfits(hdus, 1)
    assert "allsigma" not in default
    assert "unaveraged" not in default


def test_bandpass_summary_analytic_statistics_and_metadata(capsys):
    # Time means are [2+3j, 6+7j, 10+11j]; unequal mean errors are [2.5, 5, 2.5].
    vis = {pol: np.array([[1+2j, 5+6j, 9+10j], [3+4j, 7+8j, 11+12j]])
           for pol in ("rr", "rl", "lr", "ll")}
    sigmas = {pol: np.array([[3., 6., 3.], [4., 8., 4.]]) for pol in vis}
    scan = _scan(vis, sigmas, unaveraged=True)
    before = {key: value.copy() for key, value in scan.items() if isinstance(value, np.ndarray)}
    table = summarize_scan_bandpass(scan, obs_day="night.uvfits", scan_num=7)
    assert capsys.readouterr().out == ""
    assert table.columns.tolist() == [
        "obs_day", "scan_num", "baseline", "polarization",
        "real_std", "real_p16", "real_p50", "real_p84",
        "imag_std", "imag_p16", "imag_p50", "imag_p84",
        "snr_median", "thermal_rms", "n_if_usable", "n_if_total",
    ]
    assert table["polarization"].tolist() == ["RR", "RL", "LR", "LL"]
    assert table["baseline"].tolist() == ["AX-GL"] * 4
    assert table["obs_day"].tolist() == ["night.uvfits"] * 4
    assert table["scan_num"].tolist() == [7] * 4
    row = table.iloc[0]
    np.testing.assert_allclose(row[["real_std", "real_p16", "real_p50", "real_p84"]].astype(float),
                               [4, 3.28, 6, 8.72])
    np.testing.assert_allclose(row[["imag_std", "imag_p16", "imag_p50", "imag_p84"]].astype(float),
                               [4, 4.28, 7, 9.72])
    assert row["snr_median"] == pytest.approx(np.median(np.abs([2+3j, 6+7j, 10+11j]) / [2.5, 5, 2.5]))
    assert row["thermal_rms"] == pytest.approx(np.sqrt(12.5))
    assert row["n_if_usable"] == row["n_if_total"] == 3
    for key, value in before.items():
        np.testing.assert_equal(scan[key], value)
    default = summarize_scan_bandpass(scan)
    assert default["obs_day"].isna().all()
    assert default["scan_num"].tolist() == [0] * 4
    del scan["scan_number"]
    assert summarize_scan_bandpass(scan)["scan_num"].isna().all()


def test_bandpass_summary_joint_mask_cancellation_and_small_counts():
    vis = {pol: np.full((2, 3), 5+2j) for pol in ("rr", "rl", "lr", "ll")}
    sigmas = {pol: np.ones((2, 3)) for pol in vis}
    vis["rr"][:] = [[1+1j, 1+1j, 100+100j], [3+3j, 3+3j, 100+100j]]
    sigmas["rr"][0, 2] = np.nan  # Exclude this IF from every statistic, even though V is valid.
    vis["rl"][:] = [[2, 2, 2], [-2, -2, -2]]  # Zero mean from cancellation is valid.
    vis["lr"][:] = [[np.nan, 3+4j, 5], [3+4j, 0, 5]]
    sigmas["lr"][:] = [[np.nan, 1, np.inf], [2, np.inf, 1]]
    sigmas["ll"][:] = [[np.nan, 0, -1], [1, 1, 1]]
    table = summarize_scan_bandpass(_scan(vis, sigmas, unaveraged=True)).set_index("polarization")
    rr = table.loc["RR"]
    assert rr["n_if_usable"] == 2
    assert rr["real_p50"] == rr["imag_p50"] == 2
    assert rr["real_std"] == rr["imag_std"] == 0
    assert rr["thermal_rms"] == pytest.approx(1 / np.sqrt(2))
    rl = table.loc["RL"]
    assert rl["n_if_usable"] == 3
    assert rl["snr_median"] == rl["real_std"] == rl["imag_std"] == 0
    lr = table.loc["LR"]
    assert lr["n_if_usable"] == 2  # Sigma of missing/zero placeholders does not disqualify IFs.
    assert lr["snr_median"] == pytest.approx(3.75)
    assert lr["thermal_rms"] == pytest.approx(np.sqrt(2.5))
    assert table.loc["LL", "n_if_usable"] == 0
    assert table.loc["LL", table.columns[3:-2]].isna().all()
    sigmas["lr"][1, 0] = np.nan  # One usable IF remains.
    one = summarize_scan_bandpass(_scan(vis, sigmas, unaveraged=True)).iloc[2]
    assert one["n_if_usable"] == 1
    assert np.isnan(one["real_std"]) and np.isnan(one["imag_std"])
    assert one["real_p16"] == one["real_p50"] == one["real_p84"] == 3
    assert one["imag_p16"] == one["imag_p50"] == one["imag_p84"] == 4
    assert one["snr_median"] == 5
    assert one["thermal_rms"] == 1


def test_bandpass_summary_17_of_32_and_nonfinite_mean():
    vis = {pol: np.ones((2, 32), dtype=complex) for pol in ("rr", "rl", "lr", "ll")}
    sigmas = {pol: np.ones((2, 32)) for pol in vis}
    for values in vis.values():
        values[:, :15] = 0
    sigmas["rr"][:, :15] = np.nan
    table = summarize_scan_bandpass(_scan(vis, sigmas, unaveraged=True))
    assert table["n_if_usable"].tolist() == [17] * 4
    assert table["n_if_total"].tolist() == [32] * 4
    np.testing.assert_allclose(table["snr_median"], np.sqrt(2))
    np.testing.assert_allclose(table["thermal_rms"], 1 / np.sqrt(2))
    # A finite uncertainty cannot make an infinite complex mean usable.
    vis["rr"][:, 15] = complex(np.inf, np.inf)
    with np.errstate(invalid="ignore"):
        table = summarize_scan_bandpass(_scan(vis, sigmas, unaveraged=True))
    assert table.iloc[0]["n_if_usable"] == 16


def test_bandpass_summary_reversed_crosshands_missing_baseline_and_labels():
    vis = {pol: np.full((2, 2), value) for pol, value in
           zip(("rr", "rl", "lr", "ll"), (1+1j, 2+3j, 4+5j, 6+7j))}
    sigmas = {pol: np.full((2, 2), k + 1.) for k, pol in enumerate(vis)}
    # GL->AX input must be transposed and conjugated for the plotted AX-GL baseline.
    scan = build_scan_coherency_matrix(
        0, [0, 0], [.1, .2], ["GL", "GL"], ["AX", "AX"], [1, 2], [3, 4],
        *vis.values(), sigmas=sigmas, unaveraged=True,
    )
    table = summarize_scan_bandpass(scan)
    assert table["real_p50"].tolist() == [1, 4, 2, 6]
    assert table["imag_p50"].tolist() == [-1, -5, -3, -7]
    np.testing.assert_allclose(table["thermal_rms"], np.array([1, 3, 2, 4]) / np.sqrt(2))
    assert summarize_scan_bandpass(scan, alma_station="AX")["polarization"].tolist() == ["XR", "XL", "YR", "YL"]
    assert summarize_scan_bandpass(scan, alma_station="GL")["polarization"].tolist() == ["RX", "RY", "LX", "LY"]
    scan["allcoh"][:, :, 0, 1] = 0  # Entire baseline unavailable, rows must remain.
    empty = summarize_scan_bandpass(scan)
    assert len(empty) == 4
    assert empty["n_if_usable"].tolist() == [0] * 4
    assert empty.iloc[:, 4:14].isna().all().all()
    autocorr = summarize_scan_bandpass(scan, include_autocorr=True)
    assert autocorr["baseline"].tolist() == ["AX-AX"] * 4 + ["AX-GL"] * 4 + ["GL-GL"] * 4
    assert autocorr["n_if_usable"].eq(0).all()
    three_stations = build_scan_coherency_matrix(
        0, [0, 0], [.1, .2], ["AX", "AX"], ["GL", "LM"], [1, 2], [3, 4],
        *vis.values(), sigmas=sigmas, unaveraged=True,
    )
    three = summarize_scan_bandpass(three_stations)
    assert three["baseline"].tolist() == ["AX-GL"] * 4 + ["AX-LM"] * 4 + ["GL-LM"] * 4
    assert three["n_if_usable"].tolist() == [2] * 8 + [0] * 4
    assert three.iloc[8:, 4:14].isna().all().all()


@pytest.mark.parametrize("mode,message", [
    ("averaged", "unaveraged=True"), ("no_sigma", "uncertainties"),
    ("single_bin", "multiple frequency bins"), ("bad_sigma_shape", "same shape"),
    ("bad_coh_shape", "allcoh must have shape"),
])
def test_invalid_bandpass_summary_inputs(mode, message):
    sigmas = {pol: np.ones((2, 2)) for pol in ("rr", "rl", "lr", "ll")}
    scan = _scan(sigmas=sigmas, unaveraged=True)
    if mode == "averaged":
        del scan["unaveraged"]
    elif mode == "no_sigma":
        del scan["allsigma"]
    elif mode == "single_bin":
        scan["allcoh"] = scan["allcoh"][:, :1]
        scan["allsigma"] = scan["allsigma"][:, :1]
    elif mode == "bad_sigma_shape":
        scan["allsigma"] = scan["allsigma"][:, :1]
    elif mode == "bad_coh_shape":
        scan["allcoh"] = scan["allcoh"][..., 0]
    with pytest.raises(ValueError, match=message):
        summarize_scan_bandpass(scan)


@pytest.mark.parametrize("var", ["amp", "phase"])
def test_bandpass_summary_snr_matches_titles_and_does_not_change_plots(var):
    sigmas = {pol: np.ones((2, 2)) for pol in ("rr", "rl", "lr", "ll")}
    scan = _scan(sigmas=sigmas, unaveraged=True)
    fig_before, axes_before = plot_scan_bandpass_all_baselines(scan, var=var)
    table = summarize_scan_bandpass(scan)
    fig_after, axes_after = plot_scan_bandpass_all_baselines(scan, var=var, show_snr=True)
    try:
        expected = [f"{x:.1f}".removesuffix(".0") for x in table["snr_median"]]
        assert axes_after[0, 0].get_title() == "AX-GL: " + ", ".join(expected)
        assert axes_after[0, 0].get_legend_handles_labels()[1] == table["polarization"].tolist()
        assert axes_before[0, 0].get_title() == "AX-GL"
        for before, after in zip(axes_before[0, 0].lines, axes_after[0, 0].lines):
            np.testing.assert_equal(before.get_xdata(), after.get_xdata())
            np.testing.assert_equal(before.get_ydata(), after.get_ydata())
    finally:
        plt.close(fig_before)
        plt.close(fig_after)
