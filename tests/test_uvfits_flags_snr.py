"""Small synthetic UVFITS fixtures and analytic bandpass SNR checks."""

from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

from eht_inspection.plotting import plot_uvfits_flag_contributions
from eht_inspection.uvfits import (
    _scan_ids_from_uvfits_nx,
    build_scan_coherency_matrix,
    build_scan_coherency_matrix_from_uvfits,
    build_closure_products_from_coherency,
    load_obs_uvfits,
    plot_scan_bandpass_all_baselines,
    plot_result_vs_time_all_baselines,
    plot_results_vs_scan_all_baselines,
    plot_amp_uvdist_whole_dataset,
    plot_closure_phase_vs_time_all_triangles,
    plot_closure_amp_vs_time_all_quadrangles,
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
    assert "AX-GL RL time 00:00:00-00:30:00: 1/3 samples (33.33%)" in output
    assert "IF 2" in output
    assert "AX-LM RR time 00:00:00-00:30:00: 3/3 samples (100.00%)" in output
    assert "AX-GL LL time 00:30:00-01:00:00: 1/3 samples (33.33%)" in output
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
    assert "AX-GL LR time unknown-unknown: 1/2 samples (50.00%)" in output
    assert "IF 2" in output
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
    assert "AX-GL LL time 00:00:00-00:30:00: 1/3 samples" in output
    assert "Dropped records: 0/3" in output


@pytest.mark.parametrize("print_flag_summary", [True, False])
def test_all_flagged_reports_before_error(capsys, print_flag_summary):
    with _uvfits(np.zeros((3, 3, 4))) as hdus:
        with pytest.raises(Exception, match="No unflagged RR or LL"):
            load_obs_uvfits(
                hdus, print_flag_summary=print_flag_summary,
                return_dict=True, include_flag_summary=True,
            )
    output = capsys.readouterr().out
    if print_flag_summary:
        assert "Flagged samples: 36/36 (100.00%)" in output
        assert "Dropped records: 3/3 (100.00%)" in output
        assert "RR: 9 flagged samples; 9 non-NaN/nonzero samples in 3 records" in output
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
        assert ("AX-GL RL time 00:00:00-00:30:00:" in output) == enabled
        results.append(result)
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        np.testing.assert_equal(results[0][key], results[1][key])


def test_flag_table_keeps_original_counts_and_unflagged_denominators(capsys):
    weights = np.full((3, 3, 4), 4.0)
    weights[0, 2, 2] = 0
    weights[1, :, :2] = -1
    weights[2, 1, 1] = np.nan
    results = []
    for include in (False, True):
        with _uvfits(weights) as hdus:
            results.append(load_obs_uvfits(
                hdus, return_dict=True, print_flag_summary=False,
                include_flag_summary=include,
            ))
    assert "UVFITS flags" not in capsys.readouterr().out
    for key in results[0]:
        np.testing.assert_equal(results[0][key], results[1][key])
    table = results[1]["flag_summary"]
    assert len(table) == 12  # Three observed scan/baseline groups, four products.
    assert table["total_samples"].sum() == 36
    assert table["flagged_samples"].sum() == 8
    assert (table["flagged_samples"] == 0).any()
    dropped = table[(table["scan"] == 0) & (table["baseline"] == "AX-LM")]
    assert dropped["dropped_records"].tolist() == [1] * 4
    np.testing.assert_equal(dropped["flagged_fraction"].to_numpy(), [1, 1, 0, 0])
    per_if = results[1]["flag_summary_if"]
    grouped = per_if.groupby(["scan", "baseline", "polarization"])[
        ["flagged_samples", "total_samples"]
    ].sum()
    expected = table.set_index(["scan", "baseline", "polarization"])[grouped.columns]
    assert grouped.equals(expected.sort_index())
    assert per_if.query('scan == 0 and baseline == "AX-LM" and polarization == "RR"').flagged_samples.tolist() == [1, 1, 1]


def test_flag_table_selection_unknown_scans_and_original_forced_products():
    weights = np.ones((3, 3, 4))
    weights[0, 0, 0] = 0  # Excluded IF.
    weights[2, 2, 1] = -1  # Must remain LL even with forced R.
    with _uvfits(weights, with_nx=False) as hdus:
        result = load_obs_uvfits(
            hdus, IF=[2], force_singlepol="R", return_dict=True,
            include_flag_summary=True, print_flag_summary=False,
        )
    table = result["flag_summary"]
    assert table["scan"].tolist() == [-1] * 8
    assert table["flagged_samples"].sum() == 1
    assert table.loc[table["flagged_samples"] > 0, "polarization"].tolist() == ["LL"]
    assert table["total_samples"].sum() == 12
    per_if = result["flag_summary_if"]
    assert per_if["IF"].tolist() == [2] * 8
    assert per_if.scan_start.eq("unknown").all()
    assert per_if.flagged_samples.sum() == 1
    assert per_if.query('flagged_samples > 0').polarization.tolist() == ["LL"]
    with _uvfits(np.ones((3, 3, 1))) as hdus:
        single = load_obs_uvfits(hdus, return_dict=True, include_flag_summary=True)
    assert single["flag_summary"]["polarization"].unique().tolist() == ["RR"]
    assert single["flag_summary"]["flagged_samples"].sum() == 0
    with pytest.raises(ValueError, match="requires return_dict"):
        load_obs_uvfits("unused.uvfits", include_flag_summary=True)


def test_flag_contribution_bars_use_counts_not_average_group_fractions():
    weights = np.ones((3, 3, 4))
    weights[0, 0, 0] = 0
    weights[1, :, 0] = -1
    weights[2, :2, 0] = np.nan
    with _uvfits(weights) as hdus:
        table = load_obs_uvfits(
            hdus, return_dict=True, include_flag_summary=True,
            print_flag_summary=False,
        )["flag_summary"]
    before = table.copy(deep=True)
    plots = plot_uvfits_flag_contributions(table, max_contributors=1)
    try:
        for group, expected in (("scan", [100 * 4 / 6, 100 * 2 / 3]),
                                ("baseline", [100, 50])):
            fig, axes = plots[group]
            ax = axes.flat[0]  # RR.
            sums = np.zeros(len(expected))
            for patch in ax.patches:
                assert patch.get_width() > 0
                y = int(round(patch.get_y() + patch.get_height() / 2))
                sums[y] += patch.get_width()
            np.testing.assert_allclose(sums, expected)
            assert "Other" in ax.get_legend_handles_labels()[1]
            assert ax.get_xlim()[0] == 0
            assert len([a for a in axes.flat if a.get_visible()]) == 4
        assert table.equals(before)
    finally:
        for fig, _ in plots.values():
            plt.close(fig)


def test_flag_contribution_no_flags_and_unknown_scan():
    with _uvfits(np.ones((3, 3, 1)), with_nx=False) as hdus:
        table = load_obs_uvfits(
            hdus, return_dict=True, include_flag_summary=True,
        )["flag_summary"]
    plots = plot_uvfits_flag_contributions(table)
    try:
        for fig, axes in plots.values():
            ax = axes.flat[0]
            assert not ax.patches
            assert {text.get_text() for text in ax.texts} == {"No flagged samples"}
            assert not ax.get_yticklabels()
    finally:
        for fig, _ in plots.values():
            plt.close(fig)
    with pytest.raises(ValueError, match="must contain"):
        plot_uvfits_flag_contributions(table.iloc[:0])
    with pytest.raises(ValueError, match="positive integer"):
        plot_uvfits_flag_contributions(table, max_contributors=0)


def _flag_bar_totals(ax):
    totals = np.zeros(len(ax.get_yticklabels()))
    for patch in ax.patches:
        assert patch.get_width() > 0
        y = int(round(patch.get_y() + patch.get_height() / 2))
        totals[y] += patch.get_width()
    return totals


def test_flag_ten_contributors_have_distinct_styles_and_visible_legend():
    import pandas as pd

    table = pd.DataFrame([
        {"scan": 0, "scan_start": "25:01:02", "baseline": f"AA-S{i:02d}",
         "polarization": "RR", "flagged_samples": 11 - i, "total_samples": 20}
        for i in range(11)
    ])
    plots = plot_uvfits_flag_contributions(table, max_contributors=10)
    try:
        fig, axes = plots["scan"]
        patches = [bars.patches[0] for bars in axes.flat[0].containers]
        assert len(patches) == 11  # Ten baselines plus Other.
        assert len({patch.get_facecolor() for patch in patches[:10]}) == 10
        assert len({patch.get_hatch() for patch in patches[:10]}) == 10
        assert len(fig.legends[0].get_texts()) == 11
        assert fig.legends[0].get_texts()[-1].get_text() == "Other"
        np.testing.assert_allclose(_flag_bar_totals(axes.flat[0]), [100 * 66 / 220])
        fig.canvas.draw()
        legend_box = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
        assert legend_box.y0 >= 0 and legend_box.x0 >= 0
        assert legend_box.x1 <= fig.bbox.x1
        assert legend_box.y1 < axes.flat[0].get_window_extent().y0
    finally:
        for fig, _ in plots.values():
            plt.close(fig)


def test_flag_if_report_plot_fractions_times_colors_and_filtering(tmp_path, capsys):
    weights = np.ones((3, 3, 4))
    weights[0, 0, 0] = 0
    weights[1, :, 0] = -1
    weights[2, :2, 0] = np.nan
    weights[2, 2, 1] = 0
    with _uvfits(weights) as hdus:
        data = load_obs_uvfits(hdus, return_dict=True, include_flag_summary=True)
    printed = capsys.readouterr().out
    assert "Per-IF" not in printed
    path = tmp_path / "flags.txt"
    path.write_text(printed + "\nPer-IF flag counts (CSV):\n"
                    + data["flag_summary_if"].to_csv(index=False))
    plots = plot_uvfits_flag_contributions(data["flag_summary"], flag_report_path=path)
    try:
        assert set(plots) == {"scan", "baseline", "if"}
        scan_ax = plots["scan"][1].flat[0]
        assert [t.get_text() for t in scan_ax.get_yticklabels()] == [
            "00:00:00 (n=6)", "00:30:00 (n=3)",
        ]
        baseline_legend = plots["baseline"][0].legends[0]
        assert {t.get_text() for t in baseline_legend.get_texts()} == {"00:00:00", "00:30:00"}
        if_axes = plots["if"][1]
        np.testing.assert_allclose(_flag_bar_totals(if_axes.flat[0]), [100, 200/3, 100/3])
        assert [t.get_text() for t in if_axes.flat[1].get_yticklabels()] == ["2 (n=3)"]
        np.testing.assert_allclose(_flag_bar_totals(if_axes.flat[1]), [100/3])
        assert not if_axes.flat[2].patches and not if_axes.flat[2].get_yticklabels()
        for label in ("AX-GL", "AX-LM"):
            scan_color = next(b.patches[0].get_facecolor() for b in scan_ax.containers if b.get_label() == label)
            if_color = next(b.patches[0].get_facecolor() for b in if_axes.flat[0].containers if b.get_label() == label)
            assert scan_color == if_color
    finally:
        for fig, _ in plots.values():
            plt.close(fig)
    filtered = data["flag_summary"].query('scan == 0 and polarization == "RR"')
    plots = plot_uvfits_flag_contributions(filtered, flag_report_path=path)
    try:
        np.testing.assert_allclose(_flag_bar_totals(plots["if"][1].flat[0]), [100, 50, 50])
    finally:
        for fig, _ in plots.values():
            plt.close(fig)
    path.write_text(printed)
    with pytest.raises(ValueError, match="regenerate"):
        plot_uvfits_flag_contributions(filtered, flag_report_path=path)


def test_flag_empty_first_product_keeps_other_product_legend_and_unknown_label():
    weights = np.ones((3, 3, 4))
    weights[0, 1, 1] = 0  # LL only; RR panel has no flags.
    with _uvfits(weights, with_nx=False) as hdus:
        table = load_obs_uvfits(hdus, return_dict=True, include_flag_summary=True,
                                print_flag_summary=False)["flag_summary"]
    plots = plot_uvfits_flag_contributions(table)
    try:
        fig, axes = plots["scan"]
        assert not axes.flat[0].patches
        assert axes.flat[1].get_yticklabels()[0].get_text() == "unknown (n=9)"
        assert [t.get_text() for t in fig.legends[0].get_texts()] == ["AX-GL"]
    finally:
        for fig, _ in plots.values():
            plt.close(fig)


@pytest.mark.parametrize("all_flagged", [False, True])
def test_notebook_saves_if_counts_without_printing_them(tmp_path, capsys, all_flagged):
    import io
    import json
    from contextlib import redirect_stdout
    from pathlib import Path

    notebook = json.loads((Path(__file__).parents[1] / "notebooks/examples/inspect_uvfits.ipynb").read_text())
    cell = next("".join(c["source"]) for c in notebook["cells"]
                if "flag_report_path = " in "".join(c["source"]))
    saving = cell[cell.index("# Save the loader"):cell.index('scan_ids = uvdata')]
    # Redirect this cell's relative output directory to the test directory.
    scope = {"Path": lambda name: tmp_path / name, "io": io,
             "redirect_stdout": redirect_stdout, "obs_day": "3880",
             "load_obs_uvfits": load_obs_uvfits}
    weights = np.ones((3, 3, 4))
    weights[0, 0, 0] = 0
    if all_flagged:
        weights[:] = 0
    with _uvfits(weights) as hdus:
        scope["fname"] = hdus
        if all_flagged:
            with pytest.raises(Exception, match="No unflagged"):
                exec(saving, scope)
        else:
            exec(saving, scope)
    printed = capsys.readouterr().out
    saved = (tmp_path / "uvfit_flag_sum/3880_flag_summary.txt").read_text()
    assert "Per-IF flag counts" not in printed
    assert printed.startswith(scope["flag_report"].getvalue())
    assert saved.startswith(scope["flag_report"].getvalue())
    assert ("\nPer-IF flag counts (CSV):\n" in saved) == (not all_flagged)


def _overlapping_nx(hdus):
    """NX scan 1's window contains scan 0, but record ranges are disjoint."""
    hdus['AIPS NX'].data['TIME'][:] = np.array([0.25, 0.5]) / 24
    hdus['AIPS NX'].data['TIME INTERVAL'][:] = np.array([0.5, 1.0]) / 24
    return hdus


def test_nx_record_ranges_correct_overlap_and_align_after_dropped_records(capsys):
    weights = np.ones((3, 3, 4))
    weights[0, 0, 1] = 0
    weights[1, :, :2] = -1  # Dropped input record still belongs to scan 0.
    weights[2, 1, 2] = np.nan
    with _overlapping_nx(_uvfits(weights)) as hdus:
        result = load_obs_uvfits(
            hdus, return_dict=True, include_flag_summary=True, include_scan_ids=True,
        )
    assert result['scan_ids'].tolist() == [0, 1]
    output = capsys.readouterr().out
    assert 'AX-LM RR time 00:00:00-00:30:00: 3/3 samples' in output
    assert 'AX-GL LL time 00:00:00-00:30:00: 1/3 samples' in output
    assert 'AX-GL RL time 00:00:00-01:00:00: 1/3 samples' in output
    table = result['flag_summary']
    assert table.groupby('scan')['total_samples'].sum().to_dict() == {0: 24, 1: 12}
    assert table.groupby('scan')['flagged_samples'].sum().to_dict() == {0: 7, 1: 1}
    assert set(table[table.scan == 0].baseline) == {'AX-GL', 'AX-LM'}
    assert set(table[table.scan == 1].baseline) == {'AX-GL'}


def test_scan_wrapper_uses_nx_ids_and_respects_explicit_intervals(monkeypatch):
    monkeypatch.setattr(
        'eht_inspection.uvfits.build_scan_coherency_matrix',
        lambda scannum, scan_ids, *args, **kwargs: {"ids": np.asarray(scan_ids)},
    )
    for kwargs, expected in (
        ({}, [0, 0, 1]),
        ({'start_index': 1}, [1, 1, 2]),
        ({'scans': [(0, 0.15), (0.15, 1)]}, [0, 1, 1]),
        ({'scan_ids': [4, 5, 6]}, [4, 5, 6]),
    ):
        with _overlapping_nx(_uvfits(np.ones((3, 3, 4)))) as hdus:
            result = build_scan_coherency_matrix_from_uvfits(
                hdus, expected[0], print_flag_summary=False, **kwargs,
            )
        np.testing.assert_equal(result["ids"], expected)


@pytest.mark.parametrize('start,end', [([0, 3], [2, 3]), ([1, 3], [2, 4]),
                                      ([2, 3], [1, 3]), ([1, 2], [2, 3])])
def test_invalid_nx_record_ranges_warn_and_fall_back_to_times(start, end):
    with _uvfits(np.ones((3, 3, 4))) as hdus:
        nx = hdus['AIPS NX'].data
        nx['START VIS'][:] = start
        nx['END VIS'][:] = end
        with pytest.warns(RuntimeWarning, match='NX visibility record ranges'):
            ids = _scan_ids_from_uvfits_nx([0.1, 0.2, 0.6], [(0, 0.5), (0.5, 1)], nx)
    assert ids.tolist() == [0, 0, 1]


def test_scan_id_opt_in_preserves_default_keys_and_handles_missing_nx():
    results = []
    for include in (False, True):
        with _uvfits(np.ones((3, 3, 4)), with_nx=False) as hdus:
            results.append(load_obs_uvfits(hdus, return_dict=True, include_scan_ids=include))
    assert 'scan_ids' not in results[0]
    assert results[1]['scan_ids'].tolist() == [-1, -1, -1]
    for key in results[0]:
        np.testing.assert_equal(results[0][key], results[1][key])
    nx_without_ranges = fits.BinTableHDU.from_columns([
        fits.Column(name='TIME', format='1D', array=[0.25, 0.75]),
    ]).data
    assert _scan_ids_from_uvfits_nx([0.1, 0.2, 0.6], [(0, 0.5), (0.5, 1)], nx_without_ranges).tolist() == [0, 0, 1]
    with pytest.raises(ValueError, match='requires return_dict'):
        load_obs_uvfits('unused.uvfits', include_scan_ids=True)


def _scan(vis=None, sigmas=None, **kwargs):
    if vis is None:
        vis = {pol: np.ones((2, 2), dtype=complex) for pol in ("rr", "rl", "lr", "ll")}
    result = build_scan_coherency_matrix(
        0, [0, 0], [0.1, 0.2], ["AX", "AX"], ["GL", "GL"], [1, 2], [3, 4],
        vis["rr"], vis["rl"], vis["lr"], vis["ll"], sigmas=sigmas, **kwargs,
    )
    result.update(scan_start="01:32:00", scan_end="01:34:00")
    return result


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
        "obs_day", "scan_start", "scan_end", "baseline", "polarization",
        "real_std", "real_p16", "real_p50", "real_p84",
        "imag_std", "imag_p16", "imag_p50", "imag_p84",
        "snr_median", "thermal_rms", "n_if_usable", "n_if_total",
        "mean_amp", "total_std", "phase_std_rad", "scan_num",
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
    assert row["mean_amp"] == pytest.approx((np.sqrt(13) + np.sqrt(85) + np.sqrt(221)) / 3)
    assert row["total_std"] == pytest.approx(np.sqrt(32))
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
    assert rl["mean_amp"] == rl["total_std"] == 0
    assert np.isnan(rl["phase_std_rad"])
    lr = table.loc["LR"]
    assert lr["n_if_usable"] == 2  # Sigma of missing/zero placeholders does not disqualify IFs.
    assert lr["snr_median"] == pytest.approx(3.75)
    assert lr["thermal_rms"] == pytest.approx(np.sqrt(2.5))
    assert table.loc["LL", "n_if_usable"] == 0
    assert table.loc["LL", "real_std":"thermal_rms"].isna().all()
    assert table.loc["LL", ["mean_amp", "total_std", "phase_std_rad"]].isna().all()
    sigmas["lr"][1, 0] = np.nan  # One usable IF remains.
    one = summarize_scan_bandpass(_scan(vis, sigmas, unaveraged=True)).iloc[2]
    assert one["n_if_usable"] == 1
    assert np.isnan(one["real_std"]) and np.isnan(one["imag_std"])
    assert one["real_p16"] == one["real_p50"] == one["real_p84"] == 3
    assert one["imag_p16"] == one["imag_p50"] == one["imag_p84"] == 4
    assert one["snr_median"] == 5
    assert one["thermal_rms"] == 1
    assert one["mean_amp"] == 5
    assert np.isnan(one["total_std"]) and np.isnan(one["phase_std_rad"])


def test_bandpass_summary_nn_exclusions_after_averaging(monkeypatch):
    from eht_inspection import uvfits
    from pandas.testing import assert_frame_equal

    # NN is the second station in AX-NN and the first in NN-ZZ.
    values = np.arange(1, 33) + 1j * np.arange(33, 65)
    vis = {pol: np.tile(values + k, (6, 1)) for k, pol in enumerate(("rr", "rl", "lr", "ll"))}
    sigmas = {pol: np.ones((6, 32)) for pol in vis}
    for pol in vis:
        vis[pol][:, [11, 28]] = 10000 + 20000j
        sigmas[pol][:, [11, 28]] = 1000
    scan = build_scan_coherency_matrix(
        0, [0] * 6, [.1] * 3 + [.2] * 3,
        ["AX", "NN", "AX"] * 2, ["NN", "ZZ", "ZZ"] * 2,
        np.ones(6), np.ones(6), *vis.values(), sigmas=sigmas, unaveraged=True,
    )
    scan.update(scan_start="01:32:00", scan_end="01:34:00")
    before = {key: value.copy() for key, value in scan.items() if isinstance(value, np.ndarray)}
    default = summarize_scan_bandpass(scan)
    assert_frame_equal(default, summarize_scan_bandpass(scan, excluded_ifs_by_station={}))

    spectra = []
    original_uncertainty = uvfits._bandpass_uncertainty

    def capture_spectrum(V, V_chan, sigma):
        spectra.append(V_chan.copy())
        # The two original integrations remain intact, even when the IF mean is masked.
        assert V.shape == (2, 32)
        np.testing.assert_equal(V[:, [11, 28]], 10000 + 20000j)
        return original_uncertainty(V, V_chan, sigma)

    monkeypatch.setattr(uvfits, "_bandpass_uncertainty", capture_spectrum)
    actual = summarize_scan_bandpass(scan, excluded_ifs_by_station={"NN": [11, 28]})
    keep = np.ones(32, dtype=bool)
    keep[[11, 28]] = False
    expected_means = values[keep]
    for baseline in ("AX-NN", "NN-ZZ"):
        rows = actual.loc[actual.baseline.eq(baseline)]
        assert rows.polarization.tolist() == ["RR", "RL", "LR", "LL"]
        assert rows.n_if_total.tolist() == [32] * 4
        assert rows.n_if_usable.tolist() == [30] * 4
        for k, (_, row) in enumerate(rows.iterrows()):
            means = expected_means + k
            for part, component in (("real", means.real), ("imag", means.imag)):
                assert row[f"{part}_std"] == pytest.approx(np.std(component, ddof=1))
                np.testing.assert_allclose(
                    row[[f"{part}_p{p}" for p in (16, 50, 84)]].astype(float),
                    np.percentile(component, [16, 50, 84]),
                )
            assert row.thermal_rms == pytest.approx(1 / np.sqrt(2))
            assert row.snr_median == pytest.approx(np.median(np.abs(means)) * np.sqrt(2))
            assert row.mean_amp == pytest.approx(np.mean(np.abs(means)))
            assert row.total_std == pytest.approx(np.sqrt(np.var(means.real, ddof=1) + np.var(means.imag, ddof=1)))
            assert np.isfinite(row.phase_std_rad)
    assert sum(np.isnan(spectrum).sum() == 2 for spectrum in spectra) == 8
    assert all(spectrum.shape == (32,) for spectrum in spectra)
    assert_frame_equal(actual.loc[actual.baseline.eq("AX-ZZ")], default.loc[default.baseline.eq("AX-ZZ")])
    for key, value in before.items():
        np.testing.assert_equal(scan[key], value)


@pytest.mark.parametrize("indices", [[-1], [32], [1.5], [[11]], [True]])
def test_bandpass_summary_rejects_invalid_excluded_ifs(indices):
    vis = {pol: np.ones((2, 32), dtype=complex) for pol in ("rr", "rl", "lr", "ll")}
    sigmas = {pol: np.ones((2, 32)) for pol in vis}
    with pytest.raises(ValueError, match="Excluded IF positions"):
        summarize_scan_bandpass(_scan(vis, sigmas, unaveraged=True), excluded_ifs_by_station={"NN": indices})


def test_bandpass_summary_wrapped_phases_and_mean_amp():
    vis = {pol: np.tile(np.exp(1j * np.deg2rad([179, -179])), (2, 1))
           for pol in ("rr", "rl", "lr", "ll")}
    sigmas = {pol: np.ones((2, 2)) for pol in vis}
    row = summarize_scan_bandpass(_scan(vis, sigmas, unaveraged=True)).iloc[0]
    assert row.phase_std_rad == pytest.approx(np.sqrt(-2 * np.log(np.cos(np.deg2rad(1)))))
    assert row.mean_amp == pytest.approx(1)
    assert row.total_std == pytest.approx(np.sqrt(2) * np.sin(np.deg2rad(1)))
    # Overall phase rotation preserves the three new scalar statistics.
    rotated = {pol: values * np.exp(0.8j) for pol, values in vis.items()}
    rotated_row = summarize_scan_bandpass(_scan(rotated, sigmas, unaveraged=True)).iloc[0]
    np.testing.assert_allclose(
        row[["mean_amp", "total_std", "phase_std_rad"]].astype(float),
        rotated_row[["mean_amp", "total_std", "phase_std_rad"]].astype(float),
    )
    # A zero IF mean remains usable for complex statistics but has no phase.
    with_zero = {pol: np.column_stack(([1, -1], values)) for pol, values in vis.items()}
    zero_sigmas = {pol: np.ones((2, 3)) for pol in vis}
    zero_row = summarize_scan_bandpass(_scan(with_zero, zero_sigmas, unaveraged=True)).iloc[0]
    assert zero_row.n_if_usable == 3
    assert zero_row.phase_std_rad == pytest.approx(row.phase_std_rad)
    assert zero_row.mean_amp == pytest.approx(2 * row.mean_amp / 3)
    with_zero["rr"][:, 2] = [1, -1]
    one_phase = summarize_scan_bandpass(_scan(with_zero, zero_sigmas, unaveraged=True)).iloc[0]
    assert np.isnan(one_phase.phase_std_rad)


def test_bandpass_summary_mean_amp_averages_magnitudes_after_time_averaging():
    # Opposite IF phases cancel in a complex frequency mean, but not mean_amp.
    means = np.array([1, -1, 3j, -3j])
    vis = {pol: np.stack([means + 2.5j, means - 2.5j])
           for pol in ("rr", "rl", "lr", "ll")}
    sigmas = {pol: np.ones((2, 4)) for pol in vis}
    table = summarize_scan_bandpass(_scan(vis, sigmas, unaveraged=True))
    np.testing.assert_allclose(table.mean_amp, 2)
    assert abs(np.mean(means)) == 0
    assert np.mean(np.abs(vis["rr"])) > 2  # Taking magnitudes before time averaging differs.


def test_bandpass_notebook_merge_excludes_combined_csv(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import pandas as pd

    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "examples" / "inspect_uvfits.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(cell["source"]) for cell in cells
                  if "csv_files = sorted(folder.glob" in "".join(cell["source"]))
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "bandpass_summary"
    folder.mkdir()
    namespace = {"Path": Path, "pd": pd}
    with pytest.raises(ValueError, match="No per-day"):
        exec(source, namespace)
    first = pd.DataFrame([dict(obs_day="2026-03-10", scan_num=0, baseline="AX-NN", polarization="RR")])
    second = first.assign(obs_day="2026-04-10")
    first.to_csv(folder / "2026-03-10_bandpass_summary.csv", index=False)
    second.to_csv(folder / "2026-04-10_bandpass_summary.csv", index=False)
    combined = folder / "combined_M87.csv"
    first.to_csv(combined, index=False)  # Previous output must never be an input.
    exec(source, namespace)
    expected = pd.concat([first, second], ignore_index=True)
    pd.testing.assert_frame_equal(pd.read_csv(combined), expected)
    exec(source, namespace)
    pd.testing.assert_frame_equal(pd.read_csv(combined), expected)
    first.to_csv(folder / "duplicate_bandpass_summary.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate day/scan"):
        exec(source, namespace)
    pd.testing.assert_frame_equal(pd.read_csv(combined), expected)


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
    scan.update(scan_start="01:32:00", scan_end="01:34:00")
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
    assert empty.iloc[:, 5:15].isna().all().all()
    autocorr = summarize_scan_bandpass(scan, include_autocorr=True)
    assert autocorr["baseline"].tolist() == ["AX-GL"] * 4
    assert autocorr["n_if_usable"].eq(0).all()
    three_stations = build_scan_coherency_matrix(
        0, [0, 0], [.1, .2], ["AX", "AX"], ["GL", "LM"], [1, 2], [3, 4],
        *vis.values(), sigmas=sigmas, unaveraged=True,
    )
    three_stations.update(scan_start="01:32:00", scan_end="01:34:00")
    three = summarize_scan_bandpass(three_stations)
    assert three["baseline"].tolist() == ["AX-GL"] * 4 + ["AX-LM"] * 4
    assert three["n_if_usable"].tolist() == [2] * 8


def _sparse_scan(pairs=(("AX", "GL"), ("AX", "LM")), scan_number=0, value=1+2j):
    """Record-based presence, including pairs with unusable visibility values."""
    n = len(pairs)
    vis = np.full((n, 2), value, dtype=complex)
    result = build_scan_coherency_matrix(
        scan_number, [scan_number] * n, np.arange(n) / 60.,
        [a for a, _ in pairs], [b for _, b in pairs], np.ones(n), np.ones(n),
        vis, vis, vis, vis,
        sigmas={pol: np.ones(vis.shape) for pol in ("rr", "rl", "lr", "ll")},
        unaveraged=True,
    )
    result.update(scan_start="28:38:00", scan_end="28:42:00")
    return result


def test_observed_baselines_use_selected_records_and_normalize_orientation():
    vis = np.ones((4, 2), dtype=complex)
    scan = build_scan_coherency_matrix(
        0, [0, 0, 0, 1], [0., 1., 2., 3.],
        ["GL", "AX", "LM", "MM"], ["AX", "GL", "LM", "SW"],
        np.ones(4), np.ones(4), vis, vis, vis, vis,
        conjugate_reverse=False,
    )
    assert scan["station_list"].tolist() == ["AX", "GL", "LM"]
    assert scan["observed_baselines"] == ["AX-GL", "LM-LM"]
    assert scan["allcoh"].shape == (3, 2, 3, 3, 2, 2)
    assert (scan["allcoh"][:, :, 0, 2] == 0).all()
    np.testing.assert_equal(scan["allcoh"][0, :, 1, 0], np.ones((2, 2, 2)))


def test_sparse_summary_printing_legacy_fallback_and_unchanged_statistics(capsys):
    import pandas as pd

    scan = _sparse_scan()
    before = {key: value.copy() for key, value in scan.items() if isinstance(value, np.ndarray)}
    table = summarize_scan_bandpass(scan, obs_day="2026-05-07")
    assert table.baseline.tolist() == ["AX-GL"] * 4 + ["AX-LM"] * 4
    assert capsys.readouterr().out == (
        "Missing baselines from loaded scan records, 2026-05-07 time 28:38:00-28:42:00: GL-LM\n"
    )
    pd.testing.assert_frame_equal(
        table, summarize_scan_bandpass(scan, obs_day="2026-05-07", print_missing_baselines=False),
    )
    assert capsys.readouterr().out == ""
    summarize_scan_bandpass(scan)
    assert capsys.readouterr().out == (
        "Missing baselines from loaded scan records, time 28:38:00-28:42:00: GL-LM\n"
    )
    legacy = {key: value for key, value in scan.items() if key != "observed_baselines"}
    all_rows = summarize_scan_bandpass(legacy, obs_day="2026-05-07")
    assert len(all_rows) == 12
    assert all_rows.iloc[8:].n_if_usable.eq(0).all()
    pd.testing.assert_frame_equal(table, all_rows.iloc[:8])
    assert capsys.readouterr().out == ""
    for key, value in before.items():
        np.testing.assert_equal(scan[key], value)


@pytest.mark.parametrize("value", [0j, complex(np.nan, np.nan)])
def test_observed_baselines_with_unusable_products_remain(value, capsys):
    scan = _sparse_scan(value=value)
    assert scan["observed_baselines"] == ["AX-GL", "AX-LM"]
    table = summarize_scan_bandpass(scan, print_missing_baselines=False)
    assert len(table) == 8
    assert table.n_if_usable.eq(0).all()
    assert table.iloc[:, 5:15].isna().all().all()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("function", [plot_scan_bandpass_all_baselines, plot_result_vs_time_all_baselines])
def test_sparse_visibility_plots_filter_and_print_only_when_enabled(function, capsys):
    scan = _sparse_scan()
    for options in ({}, {"print_missing_baselines": False}, {"print_missing_baselines": True}):
        fig, axes = function(scan, obs_day="2026-05-07", **options)
        assert [ax.get_title() for ax in axes.flat if ax.axison] == ["AX-GL", "AX-LM"]
        text = capsys.readouterr().out
        assert text == ("Missing baselines from loaded scan records, 2026-05-07 time "
                        "28:38:00-28:42:00: GL-LM\n" if options.get("print_missing_baselines") else "")
        plt.close(fig)
    del scan["observed_baselines"]
    warning = (pytest.warns(RuntimeWarning, match="Mean of empty slice")
               if function is plot_scan_bandpass_all_baselines else nullcontext())
    with warning:
        fig, axes = function(scan, print_missing_baselines=True)
    assert [ax.get_title() for ax in axes.flat if ax.axison] == ["AX-GL", "AX-LM", "GL-LM"]
    assert capsys.readouterr().out == ""
    plt.close(fig)


def test_aggregate_observed_union_per_scan_reporting_and_legacy_curves(capsys):
    first = _sparse_scan()
    second = _sparse_scan((("AX", "GL"), ("GL", "LM")), scan_number=1)
    second.update(scan_start="28:55:00", scan_end="28:59:00")
    third = _sparse_scan((("MM", "SW"),), scan_number=2)
    scans = [first, second, third]
    fig, axes, values = plot_results_vs_scan_all_baselines(scans)
    assert capsys.readouterr().out == ""
    assert [ax.get_title() for ax in axes.flat if ax.axison] == ["AX-GL", "AX-LM", "GL-LM", "MM-SW"]
    assert values["AX-GL"]["RR"]["scan"].tolist() == [0, 1]
    assert values["AX-LM"]["RR"]["scan"].tolist() == [0]
    assert values["GL-LM"]["RR"]["scan"].tolist() == [1]
    plt.close(fig)
    fig, _, printed_values = plot_results_vs_scan_all_baselines(
        scans, print_missing_baselines=True, obs_day="2026-05-07",
    )
    assert capsys.readouterr().out.splitlines() == [
        "Missing baselines from loaded scan records, 2026-05-07 time 28:38:00-28:42:00: GL-LM",
        "Missing baselines from loaded scan records, 2026-05-07 time 28:55:00-28:59:00: AX-LM",
    ]
    plt.close(fig)
    legacy = [{key: value for key, value in scan.items() if key != "observed_baselines"} for scan in scans]
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        fig, _, old_values = plot_results_vs_scan_all_baselines(legacy, print_missing_baselines=True)
    assert capsys.readouterr().out == ""
    assert len(old_values) == 10  # Legacy panels retain all pairs of five combined stations.
    for baseline, products in values.items():
        for pol, quantities in products.items():
            for quantity, array in quantities.items():
                np.testing.assert_equal(array, old_values[baseline][pol][quantity])
                np.testing.assert_equal(array, printed_values[baseline][pol][quantity])
    plt.close(fig)


def test_autocorrelation_presence_and_empty_baseline_outputs(capsys):
    scan = _sparse_scan((("AX", "AX"), ("GL", "GL")))
    table = summarize_scan_bandpass(scan, print_missing_baselines=False)
    assert table.empty
    actual = summarize_scan_bandpass(scan, include_autocorr=True, print_missing_baselines=False)
    assert actual.baseline.tolist() == ["AX-AX"] * 4 + ["GL-GL"] * 4
    assert table.columns.tolist() == actual.columns.tolist()
    for function in (plot_scan_bandpass_all_baselines, plot_result_vs_time_all_baselines):
        with pytest.raises(ValueError, match="No baselines found"):
            function(scan)
        fig, axes = function(scan, include_autocorr=True)
        assert [ax.get_title() for ax in axes.flat if ax.axison] == ["AX-AX", "GL-GL"]
        plt.close(fig)
    with pytest.raises(ValueError, match="No baselines found"):
        plot_results_vs_scan_all_baselines([scan])
    legacy = {key: value for key, value in scan.items() if key != "observed_baselines"}
    assert len(summarize_scan_bandpass(legacy, include_autocorr=True)) == 12
    assert capsys.readouterr().out == ""


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


def test_nx_time_labels_match_summary_and_wrapper_after_drops():
    weights = np.ones((3, 3, 4))
    weights[1, :, :2] = 0
    with _overlapping_nx(_uvfits(weights)) as hdus:
        data = load_obs_uvfits(
            hdus, return_dict=True, include_scan_ids=True,
            include_flag_summary=True, print_flag_summary=False,
        )
        scan = build_scan_coherency_matrix_from_uvfits(hdus, 1, start_index=1)
    assert data['scan_ids'].tolist() == [0, 1]
    assert data['scan_start'].tolist() == ['00:00:00', '00:00:00']
    assert data['scan_end'].tolist() == ['00:30:00', '01:00:00']
    dropped = data['flag_summary'].query('baseline == "AX-LM"')
    assert dropped.scan_start.tolist() == ['00:00:00'] * 4
    assert dropped.scan_end.tolist() == ['00:30:00'] * 4
    assert (scan['scan_start'], scan['scan_end']) == ('00:00:00', '00:30:00')
    with _uvfits(weights, with_nx=False) as hdus:
        unknown = load_obs_uvfits(
            hdus, return_dict=True, include_scan_ids=True,
            include_flag_summary=True, print_flag_summary=False,
        )
    assert unknown['scan_start'].tolist() == ['unknown'] * 2
    assert unknown['scan_end'].tolist() == ['unknown'] * 2
    assert unknown['flag_summary'].scan_start.eq('unknown').all()


def test_custom_scan_ids_cannot_combine_distinct_nx_intervals():
    with _overlapping_nx(_uvfits(np.ones((3, 3, 4)))) as hdus:
        with pytest.raises(ValueError, match='conflicting NX'):
            build_scan_coherency_matrix_from_uvfits(hdus, 4, scan_ids=[4, 4, 4])


@pytest.mark.parametrize('force', [None, 'R'])
def test_flag_audit_uses_original_values_before_masking_and_forcing(force, capsys):
    weights = np.ones((3, 3, 4))
    weights[0, :, 1] = 0
    weights[1, :, :2] = -1  # Dropped record must still be audited.
    weights[2, 1, 2] = np.nan
    with _uvfits(weights) as hdus:
        raw = hdus[0].data['DATA'][:, 0, 0, :, 0]
        raw[0, :, 1, 0] = [0, np.nan, 0]
        raw[0, :, 1, 1] = [0, 0, 2]  # Purely imaginary nonzero value.
        raw[1, :, :2, 0] = 0
        raw[1, :, :2, 1] = 0
        raw[1, 0, 0, 0] = 3  # Purely real nonzero value in a dropped record.
        raw[2, 1, 2, 0] = 0
        data = load_obs_uvfits(
            hdus, return_dict=True, include_flag_summary=True, force_singlepol=force,
        )
    output = capsys.readouterr().out
    assert output.index('Dropped records: 1/3') < output.index('RR: 3 flagged samples')
    assert 'RR: 3 flagged samples; 1 non-NaN/nonzero samples in 1 records' in output
    assert 'LL: 6 flagged samples; 1 non-NaN/nonzero samples in 1 records' in output
    assert 'RL: 1 flagged samples; 0 non-NaN/nonzero samples in 0 records' in output
    assert 'LR: 0 flagged samples; 0 non-NaN/nonzero samples in 0 records' in output
    locations = output.split('Flagged samples with finite, nonzero original observations:')[1]
    assert 'AX-LM RR time 00:00:00-00:30:00: 1/3 samples' in locations
    assert 'AX-GL LL time 00:00:00-00:30:00: 1/3 samples' in locations
    assert 'IF 2' in locations and 'channel' not in locations
    table = data['flag_summary']
    assert table.finite_nonzero_samples.sum() == 2
    assert table.finite_nonzero_records.sum() == 2
    assert table.query('scan == 0 and baseline == "AX-LM"').finite_nonzero_samples.tolist() == [1, 0, 0, 0]


def test_flag_audit_if_selection_and_print_switch(capsys):
    weights = np.ones((3, 3, 4))
    weights[0, 0, 0] = 0  # Excluded sample is nonzero.
    weights[2, 2, 1] = 0
    with _uvfits(weights) as hdus:
        hdus[0].data['DATA'][2, 0, 0, 2, 0, 1, 0] = 0
        result = load_obs_uvfits(
            hdus, IF=[2], return_dict=True, include_flag_summary=True,
            print_flag_summary=False,
        )
    assert result['flag_summary'].flagged_samples.sum() == 1
    assert result['flag_summary'].finite_nonzero_samples.sum() == 0
    assert 'UVFITS flags' not in capsys.readouterr().out


@pytest.mark.parametrize('metadata', [
    {}, {'scan_start': 'unknown', 'scan_end': 'unknown'},
    {'scan_start': '02:00:00', 'scan_end': '01:00:00'},
    {'scan_start': ['01:00:00', '02:00:00'], 'scan_end': '03:00:00'},
])
def test_scan_outputs_require_known_consistent_metadata(metadata):
    scan = _scan(sigmas={pol: np.ones((2, 2)) for pol in ('rr', 'rl', 'lr', 'll')}, unaveraged=True)
    del scan['scan_start'], scan['scan_end']
    scan.update(metadata)
    before = plt.get_fignums()
    for function in (plot_scan_bandpass_all_baselines, plot_result_vs_time_all_baselines, summarize_scan_bandpass):
        with pytest.raises(ValueError, match='metadata'):
            function(scan)
    assert plt.get_fignums() == before


def test_visibility_time_titles_filenames_and_index_axes(tmp_path, monkeypatch):
    from pathlib import Path

    paths = []
    monkeypatch.setattr(plt.Figure, 'savefig', lambda self, path, **kwargs: paths.append(Path(path).name))
    scan = _scan()
    later = dict(scan, scan_number=7, scan_start='25:00:00', scan_end='25:02:00')
    options = dict(source='3C273', obs_day='3888', figdir=tmp_path, savefig=True)
    try:
        for function, suffix in (
            (plot_scan_bandpass_all_baselines, 'phase_vs_channel_all_baselines'),
            (plot_result_vs_time_all_baselines, 'phase_vs_time_all_baselines'),
        ):
            fig, _ = function(scan, **options)
            assert '3C273, 3888, time 01:32:00' in fig._suptitle.get_text()
            assert 'scan 0' not in fig._suptitle.get_text()
            prefix = 'avg_' if function is plot_result_vs_time_all_baselines else ''
            assert paths[-1] == f'{prefix}3C273_3888_013200_{suffix}.png'
            plt.close(fig)
        fig, axes, values = plot_results_vs_scan_all_baselines([later, scan], **options)
        assert '01:32:00-25:02:00' in fig._suptitle.get_text()
        assert axes[0, 0].get_xlabel() == 'Scan index'
        assert values['AX-GL']['RR']['scan'].tolist() == [0, 7]
        assert paths[-1] == 'avg_3C273_3888_013200_phase_vs_scan_all_baselines.png'
        plt.close(fig)
        fig, _ = plot_amp_uvdist_whole_dataset(
            np.array([1, 2]), np.array([3, 4]), *[np.ones((2, 2))] * 4,
            show=False, **options,
        )
        assert '3C273, 3888' in fig._suptitle.get_text()
        assert paths[-1] == '3C273_3888_amp_uvdist.png'
        plt.close(fig)
    finally:
        plt.close('all')


def test_closure_time_metadata_titles_and_filenames(tmp_path, monkeypatch):
    from pathlib import Path

    scan = dict(
        allcoh=np.ones((2, 2, 4, 4, 2, 2), dtype=complex),
        station_list=['AX', 'GL', 'KT', 'MG'], t_unique=np.array([1.54, 1.55]),
        channel_list=np.arange(2), scan_number=3,
        scan_start='01:32:00', scan_end='01:34:00',
    )
    closure = build_closure_products_from_coherency(scan)
    assert (closure['scan_start'], closure['scan_end']) == ('01:32:00', '01:34:00')
    paths = []
    monkeypatch.setattr(plt.Figure, 'savefig', lambda self, path, **kwargs: paths.append(Path(path).name))
    try:
        for function, suffix in (
            (plot_closure_phase_vs_time_all_triangles, 'closure_phase_vs_time_all_triangles'),
            (plot_closure_amp_vs_time_all_quadrangles, 'closure_amp_vs_time_all_quadrangles'),
        ):
            fig, _ = function(closure, source='3C273', obs_day='3888', figdir=tmp_path, savefig=True)
            assert '3C273, 3888, time 01:32:00' in fig._suptitle.get_text()
            assert paths[-1] == f'3C273_3888_013200_{suffix}.png'
            plt.close(fig)
            with pytest.raises(ValueError, match='metadata'):
                function(dict(closure, scan_start='unknown'))
    finally:
        plt.close('all')


def test_flag_audit_channel_selection_reads_only_original_selected_samples():
    with _uvfits(np.ones((3, 3, 4))) as hdus:
        original = hdus[0]
        values = np.repeat(original.data['DATA'], 2, axis=4)
        values[0, 0, 0, 2, :, 1, 2] = 0  # Both channels flagged in LL.
        values[0, 0, 0, 2, 1, 1, 0] = 0  # Selected channel has zero visibility.
        hdus[0] = fits.GroupsHDU(fits.GroupData(
            values, bitpix=-64, parnames=original.data.parnames,
            pardata=[original.data[name] for name in original.data.parnames],
        ), header=original.header)
        selected = load_obs_uvfits(
            hdus, IF=[2], channel=[1], return_dict=True,
            include_flag_summary=True, print_flag_summary=False,
        )['flag_summary']
    assert selected.flagged_samples.sum() == 1
    assert selected.total_samples.sum() == 12
    assert selected.finite_nonzero_samples.sum() == 0
