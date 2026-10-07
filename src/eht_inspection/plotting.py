"""Plotting entry points for EHT inspection workflows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import get_subplot_grid

_ALIST_PLOTS = {
    "plot_alist_closure_vs_scan",
    "plot_coherence_diagnostics",
    "plot_coherence_hist",
    "plot_quantity_vs_scan_by_station",
    "plot_snr_across_stages",
}

_UVFITS_PLOTS = {
    "plot_amp_uvdist_whole_dataset",
    "plot_closure_amp_vs_time_all_quadrangles",
    "plot_closure_phase_vs_time_all_triangles",
    "plot_result_vs_time_all_baselines",
    "plot_results_vs_scan_all_baselines",
    "plot_scan_bandpass_all_baselines",
}


def plot_uv_coverage(
    u,
    v,
    scale=1e9,
    unit="Gλ",
    include_conjugate=True,
    ax=None,
):
    """
    Plot uv coverage.

    Parameters
    ----------
    u, v
        UV coordinates, shape ``(Nrecord,)``. Values are divided by ``scale``
        before plotting.
    scale
        Coordinate scale factor. Default converts wavelengths to Gλ.
    unit
        Axis unit label.
    include_conjugate
        If True, also plot ``(-u, -v)``.
    ax
        Optional matplotlib axes.

    Returns
    -------
    tuple
        ``(fig, ax)`` matplotlib objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.scatter(u / scale, v / scale, s=3)
    if include_conjugate:
        ax.scatter(-np.asarray(u) / scale, -np.asarray(v) / scale, s=3)
    ax.set_xlabel(f"u [{unit}]")
    ax.set_ylabel(f"v [{unit}]")
    ax.set_title("uv coverage")
    ax.axis("equal")
    ax.grid(True)
    return fig, ax


def default_pol_marker_map():
    """
    Return the default polarization marker convention.

    Returns
    -------
    dict[str, str]
        Marker lookup for circular products ``RR, RL, LR, LL`` and ALMA
        mixed-pol products ``XR, XL, YR, YL``.
    """
    return {
        "LL": "o",
        "LR": "_",
        "RL": "|",
        "RR": "x",
        "XL": "o",
        "XR": "x",
        "YL": "|",
        "YR": "_",
    }


def save_figure(
    fig,
    path,
    dpi=300,
    bbox_inches="tight",
    **kwargs,
):
    """
    Create the output directory and save a matplotlib figure.

    Parameters
    ----------
    fig
        Matplotlib figure.
    path
        Output path.
    dpi, bbox_inches
        Passed to ``fig.savefig``.

    Returns
    -------
    pathlib.Path
        Path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    return path


def _plot_flag_contributions(table, group, contributor, max_contributors, figsize):
    """Plot additive sample fractions with one common denominator per bar."""
    polarizations = list(table["polarization"].drop_duplicates())
    counts = table.groupby(group)[["flagged_samples", "total_samples"]].sum()
    if group == "scan":
        groups = sorted(counts.index, key=lambda scan: (scan < 0, scan))
    else:
        fractions = counts["flagged_samples"] / counts["total_samples"]
        groups = list(fractions.sort_values(ascending=False, kind="stable").index)
    ranked = table.groupby(contributor)["flagged_samples"].sum()
    ranked = ranked[ranked > 0].sort_values(ascending=False, kind="stable")
    contributors = list(ranked.index[:max_contributors])
    has_other = len(ranked) > max_contributors
    labels = [
        ("unknown" if value < 0 else str(value)) if contributor == "scan" else str(value)
        for value in contributors
    ]
    if has_other:
        labels.append("Other")

    ncols = min(2, len(polarizations))
    nrows = (len(polarizations) + ncols - 1) // ncols
    if figsize is None:
        figsize = (6.5 * ncols, nrows * max(3, 0.35 * len(groups) + 1.5) + 1.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    colors = ("#3B6FB6", "#C49A28", "#D47732", "#818B45", "#B7688A")
    hatches = ("", "//", "..", "\\\\", "xx")
    positions = np.arange(len(groups))

    for ax, pol in zip(axes.flat, polarizations):
        subset = table[table["polarization"] == pol]
        totals = subset.groupby(group)[["flagged_samples", "total_samples"]].sum()
        totals = totals.reindex(groups, fill_value=0)
        denominator = totals["total_samples"].to_numpy()
        parts = subset.pivot_table(
            index=group, columns=contributor, values="flagged_samples",
            aggfunc="sum", fill_value=0,
        ).reindex(index=groups, columns=contributors, fill_value=0)
        values = [parts[value].to_numpy() for value in contributors]
        if has_other:
            values.append(totals["flagged_samples"].to_numpy() - parts.sum(axis=1).to_numpy())
        left = np.zeros(len(groups))
        for i, (label, count) in enumerate(zip(labels, values)):
            fraction = np.divide(
                100.0 * count, denominator, out=np.zeros(len(groups)), where=denominator > 0,
            )
            ax.barh(
                positions, fraction, left=left, label=label,
                color="#B8BCC2" if has_other and i == len(labels) - 1 else colors[i % len(colors)],
                hatch=hatches[i % len(hatches)], edgecolor="#444444", linewidth=0.4,
            )
            left += fraction
        group_labels = [
            ("unknown" if value < 0 else str(value)) if group == "scan" else str(value)
            for value in groups
        ]
        ax.set_yticks(positions, [
            f"{label} (n={int(total)})" for label, total in zip(group_labels, denominator)
        ])
        for y, percent, total in zip(positions, left, denominator):
            ax.text(percent + 1, y, f"{percent:.1f}%" if total else "no samples",
                    va="center", fontsize=9)
        ax.set_ylim(len(groups) - 0.5, -0.5)
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Flagged samples / all samples in this bar [%]")
        ax.set_ylabel("Scan" if group == "scan" else "Baseline")
        ax.set_title(pol)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(polarizations):]:
        ax.set_visible(False)

    fig.suptitle(
        f"UVFITS flagged fractions per {group}, split by {contributor}\n"
        "Original selected weights <= 0 or NaN; n = input samples per polarization, including dropped records",
        fontsize=11,
    )
    if labels:
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, title=contributor.capitalize(),
                   loc="lower center", ncol=min(3, len(labels)))
    footer = min(0.3, 0.85 / fig.get_figheight()) if labels else 0
    header = min(0.2, 0.25 / fig.get_figheight())
    fig.tight_layout(rect=(0, footer, 1, 1 - header))
    return fig, axes


def plot_uvfits_flag_contributions(flag_summary, *, max_contributors=5, figsize=None):
    """Plot original UVFITS flag fractions per scan and per baseline.

    Parameters
    ----------
    flag_summary : pandas.DataFrame
        ``uvdata["flag_summary"]`` from ``load_obs_uvfits`` with
        ``return_dict=True, include_flag_summary=True``. Keep zero-flag rows:
        they supply the unflagged part of each denominator. May be filtered
        to a subset of scans/baselines/products before plotting.
    max_contributors : int, default 5
        Largest contributors (by flagged-sample count across polarizations)
        to show individually per figure. Remaining contributors form "Other".
        All scan/baseline bars remain visible; nothing is removed from totals.
    figsize : tuple, optional
        Matplotlib figure size for each figure. Defaults scale with bar count.

    Returns
    -------
    dict
        ``{"scan": (fig, axes), "baseline": (fig, axes)}``. Axes are 2D arrays,
        with one panel per product present in the file. Scan bars are ordered
        by scan ID (unknown last), baseline bars by descending overall flagged
        fraction. No file data, baseline orientation, or polarization is changed.

    Notes
    -----
    Each segment is flagged samples for one contributing baseline/scan divided
    by ALL selected input samples in the bar's scan/baseline and polarization.
    Segments add to the total flagged fraction, rather than to 100%. They are
    contributions to the bar, not each subgroup's own flag rate; that rate is
    available in ``flag_summary.flagged_fraction``. The statistic is a fraction
    of record/IF/channel/product cells, not a fraction of affected records.
    Missing observations and absent products are not counted as flags.
    """
    required = {"scan", "baseline", "polarization", "flagged_samples", "total_samples"}
    missing = required.difference(flag_summary.columns)
    if missing:
        raise ValueError(f"flag_summary is missing columns: {sorted(missing)}")
    if flag_summary.empty:
        raise ValueError("flag_summary must contain input sample counts")
    if (isinstance(max_contributors, bool)
            or not isinstance(max_contributors, (int, np.integer)) or max_contributors < 1):
        raise ValueError("max_contributors must be a positive integer")
    flagged = flag_summary["flagged_samples"].to_numpy()
    total = flag_summary["total_samples"].to_numpy()
    if (not np.all(np.isfinite(flagged)) or not np.all(np.isfinite(total))
            or np.any(total <= 0) or np.any(flagged < 0) or np.any(flagged > total)):
        raise ValueError("counts must satisfy 0 <= flagged_samples <= total_samples, with total_samples > 0")
    return {
        "scan": _plot_flag_contributions(flag_summary, "scan", "baseline", max_contributors, figsize),
        "baseline": _plot_flag_contributions(flag_summary, "baseline", "scan", max_contributors, figsize),
    }


def __getattr__(name):
    if name in _ALIST_PLOTS:
        from . import alist
        return getattr(alist, name)
    if name in _UVFITS_PLOTS:
        from . import uvfits
        return getattr(uvfits, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "default_pol_marker_map",
    "plot_alist_closure_vs_scan",
    "get_subplot_grid",
    "plot_amp_uvdist_whole_dataset",
    "plot_closure_amp_vs_time_all_quadrangles",
    "plot_closure_phase_vs_time_all_triangles",
    "plot_coherence_diagnostics",
    "plot_coherence_hist",
    "plot_quantity_vs_scan_by_station",
    "plot_result_vs_time_all_baselines",
    "plot_results_vs_scan_all_baselines",
    "plot_scan_bandpass_all_baselines",
    "plot_snr_across_stages",
    "plot_uv_coverage",
    "plot_uvfits_flag_contributions",
    "save_figure",
]
