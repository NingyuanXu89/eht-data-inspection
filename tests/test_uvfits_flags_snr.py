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
