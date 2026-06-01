"""Small shared utilities used across inspection modules."""

from importlib.util import find_spec
import itertools
import math


def optional_dependency_available(module_name):
    """
    Return whether a Python module can be imported.

    Parameters
    ----------
    module_name
        Import name, for example ``"eat"`` or ``"ehtim"``.

    Returns
    -------
    bool
        True if Python can find the module, False otherwise.
    """
    return find_spec(module_name) is not None


def wrap_phase(phase, rad=False, return_degrees=None):
    """
    Wrap phase values.

    Parameters
    ----------
    phase : scalar or array-like
        Phase values to wrap.
    rad : bool
        If True, input values are radians. If False, input values are degrees.
    return_degrees : bool or None
        If True, return degrees. If False, return radians for radian input.
        The default preserves the existing convention: degree input returns
        degrees, and radian input returns radians.
    """
    if return_degrees is None:
        return_degrees = not rad
    if rad:
        wrapped = (phase + math.pi) % (2 * math.pi) - math.pi
        if return_degrees:
            return wrapped * 180.0 / math.pi
        return wrapped
    return (phase + 180) % 360 - 180


def unwrap_phase(phase, rad=False):
    """
    Unwrap phase values along the last axis.

    Parameters
    ----------
    phase
        Scalar or array-like phase values.
    rad : bool
        If True, input and output are radians. If False, input and output are
        degrees.

    Returns
    -------
    scalar or array-like
        Unwrapped phase values in the same angular unit as the input.
    """
    import numpy as np

    if rad:
        return np.unwrap(phase)
    return np.rad2deg(np.unwrap(np.deg2rad(phase)))


def get_subplot_grid(nplots):
    """
    Choose a compact subplot grid.

    Parameters
    ----------
    nplots
        Number of panels.

    Returns
    -------
    tuple[int, int]
        ``(nrows, ncols)`` for a near-square layout, with one row for up to
        four panels.
    """
    if nplots < 1:
        raise ValueError("nplots must be >= 1")
    if nplots <= 4:
        return 1, nplots
    ncols = math.ceil(math.sqrt(nplots))
    nrows = math.ceil(nplots / ncols)
    return nrows, ncols


def get_baselines_from_station_list(
    station_list,
    include_autocorr=False,
    sep="-",
):
    """
    Generate baseline names from a station list.

    Parameters
    ----------
    station_list
        Ordered station names.
    include_autocorr
        If True, include ``station-station`` autocorrelation pairs.
    sep
        Separator between station names.

    Returns
    -------
    list[str]
        Baseline labels such as ``"AA-LM"``.
    """
    baselines = []
    nstation = len(station_list)
    for i in range(nstation):
        j_start = i if include_autocorr else i + 1
        for j in range(j_start, nstation):
            baselines.append(f"{station_list[i]}{sep}{station_list[j]}")
    return baselines


def triangle_names_from_station_list(
    station_list,
    exclude_stations=(),
):
    """
    Generate station-name triangles after excluding selected stations.

    Returns
    -------
    list[tuple[str, str, str]]
        Triangle names in station-list order, used for closure phase.
    """
    excluded = set(exclude_stations)
    stations = [str(st) for st in station_list if str(st) not in excluded]
    return list(itertools.combinations(stations, 3))


def quadrangle_names_from_station_list(
    station_list,
    exclude_stations=(),
):
    """
    Generate station-name quadrangles after excluding selected stations.

    Returns
    -------
    list[tuple[str, str, str, str]]
        Quadrangle names in station-list order, used for closure amplitude.
    """
    excluded = set(exclude_stations)
    stations = [str(st) for st in station_list if str(st) not in excluded]
    return list(itertools.combinations(stations, 4))


def scan_ids_from_intervals(times, scans):
    """
    Assign each time sample to a scan interval.

    Parameters
    ----------
    times
        Array-like time values, shape ``(Nrecord,)``.
    scans
        Sequence of ``(start, stop)`` intervals in the same time units.

    Returns
    -------
    numpy.ndarray
        Integer scan IDs, shape ``(Nrecord,)``. Values remain ``-1`` when no
        interval contains the record. Intervals are half-open except for the
        final interval, which includes its right edge.
    """
    import numpy as np

    times = np.asarray(times)
    scan_ids = np.full(len(times), -1, dtype=int)
    for iscan, (t_start, t_stop) in enumerate(scans):
        if iscan == len(scans) - 1:
            in_scan = (times >= t_start) & (times <= t_stop)
        else:
            in_scan = (times >= t_start) & (times < t_stop)
        scan_ids[in_scan] = iscan
    return scan_ids
