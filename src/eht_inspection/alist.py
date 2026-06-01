from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import operator

from .utils import wrap_phase

# This script is for loading the alist files for given source, avg_time, and optionally freq and stage


def load_alist(
    stage=3,
    source="M87",
    avg_time=0,
    multi_freq=False,
    freq=230,
    multi_stage=False,
    data_dir=None,
):
    """
    Load a HOPS alist file into a pandas DataFrame.

    Parameters
    ----------
    stage
        Pipeline stage number used when ``multi_stage=True``.
    source
        Source name to keep after loading.
    avg_time
        Averaging time in seconds. ``0`` loads ``alist.v6``; nonzero values
        load ``alist.v6.<avg_time>s.avg``.
    multi_freq
        If True, look under a ``<freq>GHz`` subdirectory.
    freq
        Frequency label used with ``multi_freq``.
    multi_stage
        If True, prefix the filename with ``stage<stage>_``.
    data_dir
        Directory containing alist files. If omitted, use the current working
        directory.

    Returns
    -------
    pandas.DataFrame
        Alist table after HOPS/EAT loading, autocorrelation removal, metadata
        fixes, day/scan-number augmentation, and source filtering.
    """
    import eat.io.hops as hops
    import eat.io.util as io_util
    data_dir = Path("." if data_dir is None else data_dir)
    if multi_freq:
        freq_dir = str(freq) + "GHz/"
    else:
        freq_dir = ""
    if avg_time == 0:
        fname = "alist.v6"
    else:
        fname = "alist.v6."+str(avg_time)+"s.avg"
    if multi_stage:
        stage_dir = "stage"+str(stage)+"_"
    else:
        stage_dir = ""
    alist_file = data_dir / freq_dir / f"{stage_dir}{fname}"
    # remove auto-correlation, convert to pandas DataFrame
    df = io_util.noauto(hops.read_alist(str(alist_file)))
    io_util.fix(df)
    io_util.add_days(df)
    io_util.add_scanno(df)
    df = df[df["source"] == source]
    return df

OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "in": lambda s, v: s.isin(v),
    "not in": lambda s, v: ~s.isin(v),
}


def filter_df(df, filters):
    """
    Generic DataFrame filter.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : dict
        Example:
        {
            "baseline": ("==", "AA"),
            "polarization": ("==", "RR"),
            "snr": (">=", 10)
        }

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe.
    """
    mask = True
    for col, condition in filters.items():
        op, value = condition
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe")
        if op not in OPS:
            raise ValueError(f"Unsupported operation '{op}'")
        mask = mask & OPS[op](df[col], value)
    return df[mask]

# wrap phase to [-180, 180]


def wrap(phase, rad=False):
    """
    Wrap alist phase values.

    Parameters
    ----------
    phase
        Scalar or array-like phase values.
    rad
        If False, input and output are degrees wrapped to ``[-180, 180)``. If
        True, input and output are radians wrapped to ``[-pi, pi)``.

    Returns
    -------
    scalar or array-like
        Wrapped phase values in the same angular unit as the input.
    """
    return wrap_phase(phase, rad=rad, return_degrees=None)


def _get_quantity(df, quantity):
    """

    Return a Series for a requested quantity.

    Supported:

    1. Existing dataframe column, e.g. "snr", "resid_phas", "sbdelay"

    2. Simple expressions, e.g. "mbdelay - sbdelay"

    3. Callable function, e.g. lambda d: d["mbdelay"] - d["sbdelay"]

    """
    # Case 1: user passes a function
    if callable(quantity):
        return quantity(df)
    # Case 2: exact column name
    if quantity in df.columns:
        return df[quantity]
    # Case 3: common aliases
    q = quantity.lower().replace(" ", "")
    aliases = {
        "mbd": "mbdelay",
        "mbdelay": "mbdelay",
        "sbd": "sbdelay",
        "sbdelay": "sbdelay",
        "mbd-sbd": "mbdelay - sbdelay",
        "mbdelay-sbdelay": "mbdelay - sbdelay",
    }
    if q in aliases:
        quantity = aliases[q]
    # Case 4: simple column expression
    # Example: "mbdelay - sbdelay"
    allowed_names = {col: df[col] for col in df.columns}
    try:
        return eval(quantity, {"__builtins__": {}}, allowed_names)
    except Exception as e:
        raise ValueError(
            f"Could not understand quantity: {quantity}. "
            f"Use a dataframe column, simple expression, or callable."
        ) from e


def _select_pols_for_station(
    dsite,
    pols=None,
    pol_col="polarization",
    baseline_col="baseline",
    station=None,
):
    """
    Select polarizations for one station plot.

    pols can be:
        None
        list, e.g. ["RR", "LL"]
        dict, e.g.
            {
                "default": ["RR", "LL"],
                "A": ["XR", "XL", "YR", "YL"],
            }
    """
    if pols is None:
        return dsite
    if isinstance(pols, dict):
        use_pols = pols.get("default", None)
        if station is not None and station in pols:
            use_pols = pols[station]
        if use_pols is None:
            return dsite
        return dsite[dsite[pol_col].isin(use_pols)].copy()
    return dsite[dsite[pol_col].isin(pols)].copy()


def add_pol_difference(
    df,
    quantity="mbdelay",
    pol1="RR",
    pol2="LL",
    keys=("scan_no", "baseline", "datetime"),
    new_col=None,
):
    """
    Add quantity difference between two polarizations.

    Example:
        mbdelay_RR - mbdelay_LL

    Returns
    -------
    pandas.DataFrame
        Matched rows with ``new_col`` and ``polarization=f"{pol1}-{pol2}"``.
    """
    if new_col is None:
        new_col = f"{quantity}_{pol1}_minus_{pol2}"
    d1 = df[df["polarization"] == pol1][list(keys) + [quantity]].copy()
    d2 = df[df["polarization"] == pol2][list(keys) + [quantity]].copy()
    merged = d1.merge(
        d2,
        on=list(keys),
        suffixes=(f"_{pol1}", f"_{pol2}"),
        how="inner",
    )
    merged[new_col] = merged[f"{quantity}_{pol1}"] - merged[f"{quantity}_{pol2}"]
    # Put it in a plotting-compatible format
    merged["polarization"] = f"{pol1}-{pol2}"
    return merged


def add_rl_double_difference(
    df,
    quantity="mbdelay",
    keys=("scan_no", "baseline", "datetime"),
    pol_col="polarization",
    new_col=None,
):
    """
    Compute (RR - RL) - (LR - LL) for a given quantity.

    Usually quantity="mbdelay".

    Returns one row per key, with polarization="R-L".
    """
    if new_col is None:
        new_col = f"{quantity}_RR_RL_minus_LR_LL"
    needed_pols = ["RR", "RL", "LR", "LL"]
    d = df[df[pol_col].isin(needed_pols)].copy()
    wide = d.pivot_table(
        index=list(keys),
        columns=pol_col,
        values=quantity,
        aggfunc="mean",
    ).reset_index()
    # require all four pol products
    for pol in needed_pols:
        if pol not in wide.columns:
            raise ValueError(f"Missing polarization {pol}")
    wide[f"{quantity}_RR_minus_RL"] = wide["RR"] - wide["RL"]
    wide[f"{quantity}_LR_minus_LL"] = wide["LR"] - wide["LL"]
    wide[new_col] = (
        wide[f"{quantity}_RR_minus_RL"]
        - wide[f"{quantity}_LR_minus_LL"]
    )
    wide["polarization"] = "R-L"
    return wide

# ====== Plotting Functions ======


def plot_snr_across_stages(
    stage_dfs,
    filters=None,
    x_col="scan_no",
    snr_col="snr",
    labels=None,
    colors=None,
    figsize=(8, 4),
):
    """
    Plot SNR vs scan/time across different HOPS stages.

    Parameters
    ----------
    stage_dfs : dict
        Example:
            {
                "3.+adhoc": avg_scan_M87_stage3,
                "5.+close": avg_scan_M87_stage5,
            }

    filters : dict or None
        filter_df-compatible filters.
        Example:
            {
                "baseline": ("==", "LN"),
                "polarization": ("==", "LL"),
            }

    x_col : str
        Usually "scan_no" or "datetime".

    Returns
    -------
    matplotlib.figure.Figure
        SNR comparison figure.
    """
    if labels is None:
        labels = list(stage_dfs.keys())
    if colors is None:
        colors = [None] * len(stage_dfs)
    fig, ax = plt.subplots(figsize=figsize)
    for (stage_name, df), label, color in zip(stage_dfs.items(), labels, colors):
        d = df.copy()
        if filters is not None:
            d = filter_df(d, filters)
        d = d.sort_values(x_col)
        ax.plot(
            d[x_col],
            d[snr_col],
            ".",
            label=label,
            color=color,
        )
    ax.set_xlabel("Scan no" if x_col == "scan_no" else x_col)
    ax.set_ylabel("SNR")
    ax.legend()
    if x_col == "datetime":
        fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


def plot_coherence_hist(
    avg_scan_df,
    avg_short_df,
    amp_col="amp",
    threshold=0.7,
    bins=61,
    hist_range=(0, 1.5),
    label="M87",
    figsize=(7, 4),
):
    """
    Plot histogram of ``C_short = A_scan / A_short``.

    The ratio is computed from scalar alist amplitudes; no complex coherent
    averaging is performed in this function.

    Returns
    -------
    tuple
        ``(fig, C_short)`` where ``C_short`` is a NumPy array, shape ``(Nrow,)``.
    """
    C = avg_scan_df[amp_col] / avg_short_df[amp_col]
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(C, bins=bins, alpha=1, range=hist_range, label=label)
    ax.axvline(threshold, color="k", ls="--", lw=1.5, label=f"C = {threshold}")
    ax.set_xlabel(r"$C_{2s}=A_{\rm scan}/A_{2s}$")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, C.to_numpy()


def plot_coherence_diagnostics(
    avg_scan_df,
    C=None,
    c_col="C_2s",
    threshold=0.7,
    snr_min=7,
    source_label="M87",
    scan_col="scan_no",
    baseline_col="baseline",
    snr_col="snr",
    figsize=(15, 4),
):
    """
    Plot C_2s vs SNR, scan_no, and baseline.
    Outliers are C_2s <= threshold and SNR >= snr_min.

    Returns
    -------
    tuple
        ``(fig, outliers)`` where outliers is a pandas DataFrame.
    """
    d = avg_scan_df.copy()
    if C is not None:
        d[c_col] = C
    coherence_filter = {
        c_col: ("<=", threshold),
        snr_col: (">=", snr_min),
    }
    outliers = filter_df(d, coherence_filter)
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].scatter(
        d[snr_col],
        d[c_col],
        alpha=0.3,
        label=source_label,
        s=24,
    )
    axs[0].scatter(
        outliers[snr_col],
        outliers[c_col],
        edgecolors="k",
        facecolors="none",
        s=70,
        linewidths=1.4,
    )
    axs[0].set_xlabel("SNR", fontsize=8)
    axs[1].scatter(
        d[scan_col],
        d[c_col],
        alpha=0.3,
        label=source_label,
        s=24,
    )
    axs[1].scatter(
        outliers[scan_col],
        outliers[c_col],
        edgecolors="k",
        facecolors="none",
        s=70,
        linewidths=1.4,
    )
    sid = d[scan_col].dropna().unique()[::5]
    axs[1].set_xticks(sid)
    axs[1].set_xticklabels(sid, fontsize=8)
    axs[1].set_xlabel("Scan no", fontsize=8)
    axs[2].scatter(
        d[baseline_col],
        d[c_col],
        alpha=0.3,
        label=source_label,
        s=24,
    )
    axs[2].scatter(
        outliers[baseline_col],
        outliers[c_col],
        edgecolors="k",
        facecolors="none",
        s=70,
        linewidths=1.4,
    )
    bls = d[baseline_col].astype(str).unique()
    axs[2].set_xticks(range(len(bls)))
    axs[2].set_xticklabels(bls, rotation=90, fontsize=6)
    axs[2].set_xlabel("Baseline", fontsize=8)
    for ax in axs:
        ax.set_ylabel(r"$C_{2s}$", fontsize=10)
        ax.grid(alpha=0.25)
        ax.axhline(threshold, color="k", ls="--", lw=1)
    fig.suptitle(f"C2_s diagnostic for {source_label}", fontsize=15)
    fig.tight_layout()
    return fig, outliers


def plot_quantity_vs_scan_by_station(
    df,
    stations=None,
    quantities="resid_phas",
    pols=None,
    filters=None,
    scan_col="scan_no",
    baseline_col="baseline",
    pol_col="polarization",
    wrap_phase_cols=("resid_phas", "total_phas"),
    ylabel="custom quantity",
    title_prefix=None,
    figsize=(12, 5),
):
    """
    Plot selected quantities vs scan number for specified stations.

    One figure per station.
    Each figure contains:
        - all baselines connected to that station, colored by baseline
        - specified polarizations and/or quantities, shown by marker shape
        - no lines connecting points
        - two legends: baseline color and marker meaning

    Parameters
    ----------
    df : pandas.DataFrame
        Alist dataframe.

    stations : list[str] or None
        Stations to plot, e.g. ["A", "L", "N"].
        If None, use all stations appearing in baseline strings.

    quantities : str, callable, or list
        Examples:
            "delay_rate"
            "resid_phas"
            "sbdelay"
            "mbdelay - sbdelay"
            lambda d: d["mbdelay"] - d["sbdelay"]

        Phase-like alist quantities are plotted in degrees after wrapping to
        [-180, 180) when their names appear in ``wrap_phase_cols``.

    pols : dict or list[str] or None
        Polarizations to include, e.g. ["RR", "LL"].
        If None, use all available polarizations.

    filters : dict or None
        Optional filter_df-compatible dictionary.
        Example:
            {"snr": (">=", 7), "source": ("==", "M87")}

    scan_col : str
        Usually "scan_no".

    Returns
    -------
    figs : dict
        Dictionary mapping station name to figure.
    """
    d = df.copy()
    # Optional compatibility with your existing filter_df()
    if filters is not None:
        d = filter_df(d, filters).copy()
    if isinstance(quantities, (str, tuple)) or callable(quantities):
        quantities = [quantities]
    # if pols is not None:
    #     d = d[d[pol_col].isin(pols)].copy()
    if stations is None:
        stations = sorted(set("".join(d[baseline_col].astype(str))))
    pol_marker_map = {
        "LL": "o",
        "LR": "_",
        "RL": "|",
        "RR": "x",
        "XL": "o",
        "XR": "x",
        "YL": "|",
        "YR": "_",
    }
    fallback_markers = ["o", "s", "^", "v", "D", "x", "|", "_", "P", "*"]
    figs = {}
    for station in stations:
        dsite = d[d[baseline_col].astype(str).str.contains(station)].copy()
        dsite = _select_pols_for_station(
        dsite,
        pols=pols,
        pol_col=pol_col,
        baseline_col=baseline_col,
        station=station,
        )
        if len(dsite) == 0:
            print(f"No data found for station {station}")
            continue
        baselines = sorted(dsite[baseline_col].unique())
        used_pols = sorted(dsite[pol_col].unique())
        cmap = plt.get_cmap("tab10")
        baseline_color_map = {
            bl: cmap(i % 10) for i, bl in enumerate(baselines)
        }
        fig, ax = plt.subplots(figsize=figsize)
        marker_labels = []
        for iq, quantity in enumerate(quantities):
            qname = quantity if isinstance(quantity, str) else getattr(quantity, "__name__", f"quantity_{iq}")
            dsite[qname] = _get_quantity(dsite, quantity)
            # Optional phase wrapping
            if isinstance(quantity, str) and quantity in wrap_phase_cols:
                dsite[qname] = wrap(dsite[qname])
            for bl in baselines:
                db = dsite[dsite[baseline_col] == bl].copy()
                for ip, pol in enumerate(used_pols):
                    dp = db[db[pol_col] == pol].sort_values(scan_col)
                    if len(dp) == 0:
                        continue
                    # If only one quantity, marker means polarization.
                    # If multiple quantities, marker means quantity+pol.
                    if len(quantities) == 1:
                        marker = pol_marker_map.get(
                            pol,
                            fallback_markers[ip % len(fallback_markers)]
                        )
                        marker_label = pol
                    else:
                        marker = fallback_markers[(iq * len(used_pols) + ip) % len(fallback_markers)]
                        marker_label = f"{qname}, {pol}"
                    if marker_label not in marker_labels:
                        marker_labels.append(marker_label)
                    ax.plot(
                        dp[scan_col],
                        dp[qname],
                        linestyle="None",
                        marker=marker,
                        color=baseline_color_map[bl],
                        markersize=6,
                        alpha=0.9,
                    )
        ax.set_xlabel("Scan no")
        ax.set_ylabel(", ".join([q if isinstance(q, str) else ylabel for q in quantities]))
        if title_prefix is None:
            ax.set_title(f"Station {station}: quantity vs scan no")
        else:
            ax.set_title(f"{title_prefix}, station {station}")
        ax.grid(alpha=0.3)
        # Legend 1: baseline color
        baseline_handles = [
            Line2D(
                [0], [0],
                color=baseline_color_map[bl],
                lw=2,
                label=bl
            )
            for bl in baselines
        ]
        leg1 = ax.legend(
            handles=baseline_handles,
            title="Baseline",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=9,
        )
        ax.add_artist(leg1)
        # Legend 2: marker meaning
        marker_handles = [
            Line2D(
                [0], [0],
                color="k",
                marker=fallback_markers[i % len(fallback_markers)] if len(quantities) > 1
                else pol_marker_map.get(label, fallback_markers[i % len(fallback_markers)]),
                linestyle="None",
                markersize=8,
                label=label,
            )
            for i, label in enumerate(marker_labels)
        ]
        ax.legend(
            handles=marker_handles,
            title="Pol / Quantity",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.25),
            fontsize=9,
        )
        plt.tight_layout()
        plt.show()
        figs[station] = fig
    return figs

# ====== Outlier Table ======


def compute_rl_double_difference_table(
    df,
    delay_col="mbdelay",
    baseline_col="baseline",
    scan_col="scan_no",
    time_col="datetime",
    source_col="source",
    pol_col="polarization",
    snr_col="snr",
    scale=None,
    threshold=0.2,
):
    """
    Compute and flag the R-L double-difference diagnostic:

        (RR - RL) - (LR - LL)

    using delay_col, usually delay_col="mbdelay".

    This is intended to be called inside summarize_delay_outliers().
    It returns only flagged rows.

    Returns
    -------
    pandas.DataFrame
        Flagged rows with RR/RL/LR/LL delay columns and fractional double
        difference. Polarization order is ``RR, RL, LR, LL``.
    """
    needed_pols = ["RR", "RL", "LR", "LL"]
    d = df[df[pol_col].isin(needed_pols)].copy()
    if len(d) == 0:
        return d.iloc[0:0].copy()
    # Match four polarization products by scan and baseline.
    # Include source/time if available so output table remains informative.
    keys = [scan_col, baseline_col]
    info_cols = []
    for col in [time_col, source_col]:
        if col in d.columns:
            info_cols.append(col)
    # Pivot delay values into RR, RL, LR, LL columns
    wide = d.pivot_table(
        index=keys,
        columns=pol_col,
        values=delay_col,
        aggfunc="mean",
    ).reset_index()
    # Require all four polarization products
    for pol in needed_pols:
        if pol not in wide.columns:
            return wide.iloc[0:0].copy()
    # Add representative info columns, e.g. datetime/source
    if len(info_cols) > 0:
        info = (
            d[keys + info_cols]
            .drop_duplicates(subset=keys)
            .copy()
        )
        wide = wide.merge(info, on=keys, how="left")
    # Add SNR per polarization if available
    if snr_col in d.columns:
        snr_wide = d.pivot_table(
            index=keys,
            columns=pol_col,
            values=snr_col,
            aggfunc="mean",
        ).reset_index()
        snr_wide = snr_wide.rename(
            columns={pol: f"{snr_col}_{pol}" for pol in needed_pols}
        )
        wide = wide.merge(snr_wide, on=keys, how="left")
    # Main diagnostic
    wide[f"{delay_col}_RR_minus_RL"] = wide["RR"] - wide["RL"]
    wide[f"{delay_col}_LR_minus_LL"] = wide["LR"] - wide["LL"]
    dd_col = f"{delay_col}_double_difference"
    abs_dd_col = f"abs_{dd_col}"
    frac_col = f"{dd_col}_frac"
    flag_col = "flag_rl_double_difference"
    wide[dd_col] = (
        wide[f"{delay_col}_RR_minus_RL"]
        - wide[f"{delay_col}_LR_minus_LL"]
    )
    wide[abs_dd_col] = np.abs(wide[dd_col])
    if scale is None:
        scale = np.nanmean(np.abs(d[delay_col]))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    wide[frac_col] = wide[abs_dd_col] / scale
    wide[flag_col] = wide[frac_col] >= threshold
    # Make it compatible with plot_quantity_vs_scan_by_station()
    wide[pol_col] = "R-L"
    out = wide[wide[flag_col]].copy()
    out = out.sort_values(frac_col, ascending=False)
    return out


def summarize_delay_outliers(
    df,
    delay_col="mbdelay",
    sbd_col="sbdelay",
    rate_col="delay_rate",
    snr_col="snr",
    pol_col="polarization",
    baseline_col="baseline",
    scan_col="scan_no",
    time_col="datetime",
    source_col="source",
    sbd_mbd_frac_threshold=0.2,
    rate_threshold=2.0,
    rl_dd_frac_threshold=0.2,
    snr_min=None,
    include_rl_double_difference=True,
):
    """
    Summarize delay-related outliers.

    Flags three types of suspicious behavior:

    1. Large fractional SBD-MBD inconsistency:
           |SBD - MBD| / avg(|MBD|) >= threshold

    2. Large fractional delay-rate outlier:
           delay_rate - avg(delay_rate) >= threshold * sigma(delay_rate)

       Note: for delay rate, this is a relative outlier metric, not
       a physical "close to zero" metric.

    3. Non-close-to-zero R-L double-difference diagnostic:
           |(RR - RL) - (LR - LL)| / avg(|MBD|) >= threshold

       This should mainly be used after stage 4 or later, when
       R-L delay has been corrected by the pipeline.

    Returns
    -------
    out : dict of pandas.DataFrame
        out["sbd_mbd"]
        out["delay_rate"]
        out["rl_double_difference"]
        out["all_flags"]
    """
    d = df.copy()
    base_cols = [
        time_col,
        scan_col,
        source_col,
        baseline_col,
        pol_col,
        snr_col,
        delay_col,
        sbd_col,
        rate_col,
    ]
    base_cols = [c for c in base_cols if c in d.columns]
    # Optional SNR cut
    if snr_min is not None and snr_col in d.columns:
        d = d[d[snr_col] >= snr_min].copy()

    # ------------------------------------------------------------------
    # 1. SBD - MBD fractional inconsistency
    # ------------------------------------------------------------------
    d["sbd_minus_mbd"] = d[sbd_col] - d[delay_col]
    d["abs_sbd_minus_mbd"] = np.abs(d["sbd_minus_mbd"])
    mbd_scale = np.nanmean(np.abs(d[delay_col]))
    if not np.isfinite(mbd_scale) or mbd_scale == 0:
        mbd_scale = 1.0
    d["sbd_mbd_frac"] = d["abs_sbd_minus_mbd"] / mbd_scale
    d["flag_sbd_mbd"] = d["sbd_mbd_frac"] >= sbd_mbd_frac_threshold
    sbd_mbd_table = d[d["flag_sbd_mbd"]].copy()
    sbd_mbd_cols = base_cols + [
        "sbd_minus_mbd",
        "abs_sbd_minus_mbd",
        "sbd_mbd_frac",
        "flag_sbd_mbd",
    ]
    sbd_mbd_cols = [c for c in sbd_mbd_cols if c in sbd_mbd_table.columns]
    sbd_mbd_table = sbd_mbd_table[sbd_mbd_cols].sort_values(
        "sbd_mbd_frac",
        ascending=False,
    )

    # ------------------------------------------------------------------
    # 2. Delay-rate fractional outliers
    # ------------------------------------------------------------------
    if rate_col in d.columns:
        rate_scale = np.nanmean(d["delay_rate"])
        rate_sigma = np.nanstd(d["delay_rate"])
        if not np.isfinite(rate_scale) or rate_scale == 0:
            rate_scale = 1.0
        d["delay_rate_frac"] = np.abs(d["delay_rate"] - rate_scale) / rate_sigma
        d["flag_delay_rate"] = d["delay_rate_frac"] >= rate_threshold
        delay_rate_table = d[d["flag_delay_rate"]].copy()
        delay_rate_cols = base_cols + [
            "delay_rate_frac",
            "flag_delay_rate",
        ]
        delay_rate_cols = [c for c in delay_rate_cols if c in delay_rate_table.columns]
        delay_rate_table = delay_rate_table[delay_rate_cols].sort_values(
            "delay_rate_frac",
            ascending=False,
        )
    else:
        d["flag_delay_rate"] = False
        delay_rate_table = d.iloc[0:0].copy()

    # ------------------------------------------------------------------
    # 3. R-L double-difference diagnostic
    # ------------------------------------------------------------------
    rl_dd_table = None
    if include_rl_double_difference:
        rl_dd_table = compute_rl_double_difference_table(
            d,
            delay_col=delay_col,
            baseline_col=baseline_col,
            scan_col=scan_col,
            time_col=time_col,
            source_col=source_col,
            pol_col=pol_col,
            snr_col=snr_col,
            scale=mbd_scale,
            threshold=rl_dd_frac_threshold,
        )

    # ------------------------------------------------------------------
    # 4. Combined row-level flags
    # ------------------------------------------------------------------
    flag_cols = ["flag_sbd_mbd", "flag_delay_rate"]
    all_flags = d[
        d["flag_sbd_mbd"] | d["flag_delay_rate"]
    ].copy()
    all_flag_cols = base_cols + [
        "sbd_minus_mbd",
        "sbd_mbd_frac",
        "delay_rate_frac",
        "flag_sbd_mbd",
        "flag_delay_rate",
    ]
    all_flag_cols = [c for c in all_flag_cols if c in all_flags.columns]
    all_flags = all_flags[all_flag_cols].copy()
    out = {
        "sbd_mbd": sbd_mbd_table,
        "delay_rate": delay_rate_table,
        "rl_double_difference": rl_dd_table,
        "all_flags": all_flags,
    }
    return out
