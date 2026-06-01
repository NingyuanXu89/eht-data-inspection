"""Plotting entry points for EHT inspection workflows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import get_subplot_grid

_ALIST_PLOTS = {
    "plot_coherence_diagnostics",
    "plot_coherence_hist",
    "plot_quantity_vs_scan_by_station",
    "plot_snr_across_stages",
}

_UVFITS_PLOTS = {
    "plot_amp_uvdist_whole_dataset",
    "plot_closure_amp_vs_time_all_quadrangles",
    "plot_closure_phase_vs_time_all_triangles",
    "plot_result_amp_vs_time_all_baselines",
    "plot_result_phase_vs_time_all_baselines",
    "plot_results_amp_vs_scan_all_baselines",
    "plot_results_phase_vs_scan_all_baselines",
    "plot_scan_amp_vs_channel_all_baselines",
    "plot_scan_phase_vs_channel_all_baselines",
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
    "get_subplot_grid",
    "plot_amp_uvdist_whole_dataset",
    "plot_closure_amp_vs_time_all_quadrangles",
    "plot_closure_phase_vs_time_all_triangles",
    "plot_coherence_diagnostics",
    "plot_coherence_hist",
    "plot_quantity_vs_scan_by_station",
    "plot_result_amp_vs_time_all_baselines",
    "plot_result_phase_vs_time_all_baselines",
    "plot_results_amp_vs_scan_all_baselines",
    "plot_results_phase_vs_scan_all_baselines",
    "plot_scan_amp_vs_channel_all_baselines",
    "plot_scan_phase_vs_channel_all_baselines",
    "plot_snr_across_stages",
    "plot_uv_coverage",
    "save_figure",
]
