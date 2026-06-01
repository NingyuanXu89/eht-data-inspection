"""Closure phase and closure amplitude diagnostics."""

from .utils import (
    quadrangle_names_from_station_list,
    triangle_names_from_station_list,
)
from .uvfits import (
    build_closure_products_from_coherency,
    plot_closure_amp_vs_time_all_quadrangles,
    plot_closure_phase_vs_time_all_triangles,
)


def quadrangle_relation_label(quadrangle):
    """
    Return the closure-amplitude relation label for a station quadrangle.

    Parameters
    ----------
    quadrangle
        Four station names ``(i, j, k, l)``.

    Returns
    -------
    str
        Label for the convention ``|V_ij| |V_kl| / (|V_ik| |V_jl|)``.
    """
    i, j, k, l = quadrangle
    return f"{i}{j} * {k}{l} / ({i}{k} * {j}{l})"


def quadrangle_relation_labels(quadrangles):
    """
    Return closure-amplitude relation labels for station quadrangles.

    Parameters
    ----------
    quadrangles
        Iterable of station-name quadrangles.

    Returns
    -------
    list[str]
        One closure-amplitude relation label per quadrangle.
    """
    return [quadrangle_relation_label(quadrangle) for quadrangle in quadrangles]

__all__ = [
    "build_closure_products_from_coherency",
    "plot_closure_amp_vs_time_all_quadrangles",
    "plot_closure_phase_vs_time_all_triangles",
    "quadrangle_names_from_station_list",
    "quadrangle_relation_label",
    "quadrangle_relation_labels",
    "triangle_names_from_station_list",
]
