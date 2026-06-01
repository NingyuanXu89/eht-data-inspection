"""Coherence diagnostics for alist data products."""

import numpy as np

from .alist import plot_coherence_diagnostics, plot_coherence_hist
from .filters import filter_df


def coherence_ratio(avg_scan_df, avg_short_df, amp_col="amp"):
    """
    Compute the alist coherence ratio ``C_short = A_scan / A_short``.

    Parameters
    ----------
    avg_scan_df, avg_short_df
        pandas DataFrames with matching row order and an amplitude column.
    amp_col
        Amplitude column name. Values are used as scalar amplitudes, not
        coherently averaged complex visibilities.

    Returns
    -------
    pandas.Series
        Coherence ratio for each row.
    """
    return avg_scan_df[amp_col] / avg_short_df[amp_col]


def select_low_coherence_high_snr(
    avg_scan_df,
    avg_short_df=None,
    c_col="C_2s",
    amp_col="amp",
    threshold=0.7,
    snr_min=7,
    snr_col="snr",
):
    """
    Select rows with low coherence and high SNR.

    If avg_short_df is supplied, coherence is computed as
    avg_scan_df[amp_col] / avg_short_df[amp_col]. Otherwise c_col must already
    exist in avg_scan_df.

    Returns
    -------
    pandas.DataFrame
        Rows satisfying ``C_short <= threshold`` and ``SNR >= snr_min``.
    """
    d = avg_scan_df.copy()
    if avg_short_df is not None:
        d[c_col] = coherence_ratio(d, avg_short_df, amp_col=amp_col)
    elif c_col not in d.columns:
        raise KeyError(
            f"Column '{c_col}' not found. Pass avg_short_df or precompute {c_col}."
        )
    d = d[np.isfinite(d[c_col])]
    return filter_df(d, {c_col: ("<=", threshold), snr_col: (">=", snr_min)})

__all__ = [
    "coherence_ratio",
    "plot_coherence_diagnostics",
    "plot_coherence_hist",
    "select_low_coherence_high_snr",
]
