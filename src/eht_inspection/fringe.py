"""HOPS/fourfit fringe-file inspection helpers."""

from pathlib import Path


def _hops_util():
    try:
        import eat.hops.util as hu
    except Exception as exc:
        raise ImportError(
            "Fringe-file inspection requires EAT/HOPS. Install EAT and source "
            "the HOPS environment before using eht_inspection.fringe."
        ) from exc
    return hu


def check_hops_env():
    """
    Check whether the current Python environment supports HOPS/mk4 access.

    Returns
    -------
    bool
        True if ``eat.hops.util.mk4`` is available. False means alist/UVFITS
        inspection may still work, but mk4 fringe-file access will not.
    """
    try:
        hu = _hops_util()
        has_hops = hu.mk4 is not None
    except Exception:
        has_hops = False
    if has_hops:
        print("HOPS/EAT mk4 is available.")
    else:
        print("HOPS/EAT mk4 is not available. Can only process alist or uvfits files.")
    return has_hops


def get_fringe_file(filename):
    """
    Load a HOPS/fourfit fringe file.

    Parameters
    ----------
    filename
        Path to an individual fringe file.

    Returns
    -------
    Any
        EAT/HOPS fringe-file object returned by
        ``eat.hops.util.getfringefile``.
    """
    return _hops_util().getfringefile(filename)


def get_fringe_params(fringefile_or_filename):
    """
    Return HOPS parameter metadata for a fringe file.

    Parameters
    ----------
    fringefile_or_filename
        EAT/HOPS fringe-file object or path to one.

    Returns
    -------
    Any
        Parameter object with fields such as baseline, polarization, SNR, SBD,
        MBD, delay, delay rate, channel count, AP count, and reference
        frequency.
    """
    hu = _hops_util()
    ff = (
        hu.getfringefile(fringefile_or_filename)
        if isinstance(fringefile_or_filename, (str, Path))
        else fringefile_or_filename
    )
    return hu.params(ff)


def get_polarization(filename):
    """
    Return the polarization label reported by HOPS/EAT.

    Returns
    -------
    str
        Fringe polarization label, commonly one of ``RR, RL, LR, LL``.
    """
    return _hops_util().getpolarization(filename)


def _as_fringe_file(fringefile_or_filename, hu):
    if isinstance(fringefile_or_filename, (str, Path)):
        return hu.getfringefile(fringefile_or_filename)
    return fringefile_or_filename


def load_type212(fringefile_or_filename):
    """
    Load type-212 records for a fringe file.

    Type-212 records contain fringe-fit spectral/AP information used for
    bandpass, residual phase, and delay-spectrum inspection.

    Returns
    -------
    Any
        Object returned by ``eat.hops.util.pop212``.
    """
    hu = _hops_util()
    ff = _as_fringe_file(fringefile_or_filename, hu)
    return hu.pop212(ff)


def load_type120(fringefile_or_filename):
    """
    Load type-120 records for a fringe file, when available.

    Type-120 records may require the corresponding correlator/corel file to be
    present next to the fringe file.

    Returns
    -------
    Any
        Object returned by ``eat.hops.util.pop120``.
    """
    hu = _hops_util()
    ff = _as_fringe_file(fringefile_or_filename, hu)
    return hu.pop120(ff)


def find_fringe(fringefile_or_filename, kind=212):
    """
    Run ``eat.hops.util.findfringe`` for a fringe file.

    Parameters
    ----------
    kind
        HOPS record kind to inspect, usually 212 for fringe spectral/AP data.

    Returns
    -------
    Any
        EAT/HOPS fringe diagnostic output.
    """
    hu = _hops_util()
    ff = (
        hu.getfringefile(fringefile_or_filename)
        if isinstance(fringefile_or_filename, (str, Path))
        else fringefile_or_filename
    )
    return hu.findfringe(fringefile=ff, kind=kind)


def apply_adhoc(fringefile_or_filename):
    """
    Apply the HOPS/EAT adhoc correction used in the fringe notebook.

    Returns
    -------
    tuple
        ``(before, after)`` where ``before`` is the type-212 record before
        correction and ``after`` is the adhoc-corrected output. The correction
        is applied coherently to complex visibilities by EAT/HOPS.
    """
    hu = _hops_util()
    ff = (
        hu.getfringefile(fringefile_or_filename)
        if isinstance(fringefile_or_filename, (str, Path))
        else fringefile_or_filename
    )
    before = hu.pop212(ff)
    after = hu.adhoc(ff)
    return before, after


def corrected_visibility_phase_amp(adhoc_output):
    """
    Channel-average adhoc-corrected visibility and return phase/amplitude.

    Parameters
    ----------
    adhoc_output
        Object with complex ``vcorr`` array. Expected shape is approximately
        ``(Nap, Nchan)``.

    Returns
    -------
    tuple
        ``(phase_rad, amplitude)`` arrays, each shape ``(Nap,)``. The channel
        average is coherent because complex visibilities are averaged before
        taking ``angle`` and ``abs``. Phase is in radians.
    """
    import numpy as np
    vcorr_chavg = np.nanmean(adhoc_output.vcorr, axis=1)
    return np.angle(vcorr_chavg), np.abs(vcorr_chavg)


def spectrum(
    fringefile_or_filename,
    cf=None,
    precorrect=True,
):
    """
    Inspect the delay-spectrum/bandpass behavior for a fringe file.

    Parameters
    ----------
    cf
        Optional HOPS control file.
    precorrect
        If True, ask EAT/HOPS to apply its precorrection before plotting.

    Returns
    -------
    Any
        EAT/HOPS spectrum output.
    """
    hu = _hops_util()
    ff = (
        hu.getfringefile(fringefile_or_filename)
        if isinstance(fringefile_or_filename, (str, Path))
        else fringefile_or_filename
    )
    return hu.spectrum(ff, cf=cf, precorrect=precorrect)


def timeseries(fringefile_or_filename):
    """
    Inspect fringe-file visibility behavior versus time/AP.

    Returns
    -------
    Any
        EAT/HOPS timeseries output. Phase conventions are those of HOPS/EAT.
    """
    hu = _hops_util()
    ff = (
        hu.getfringefile(fringefile_or_filename)
        if isinstance(fringefile_or_filename, (str, Path))
        else fringefile_or_filename
    )
    return hu.timeseries(ff)


def fplot(filename):
    """
    Run HOPS ``fplot`` through EAT.

    Parameters
    ----------
    filename
        Path to an individual HOPS/fourfit fringe file.

    Returns
    -------
    Any
        EAT/HOPS PDF object with ``pdfdata`` bytes.
    """
    return _hops_util().fplot(filename)


def export_fplot_pdf(
    filename,
    outdir="pdf",
    suffix="_fplot.pdf",
):
    """
    Run ``fplot`` and write the returned PDF data to disk.

    Parameters
    ----------
    filename
        Path to an individual HOPS/fourfit fringe file.
    outdir
        Directory where the PDF should be written.
    suffix
        Suffix appended to the input fringe-file basename.

    Returns
    -------
    pathlib.Path
        Written PDF path.
    """
    out = fplot(filename)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{Path(filename).name}{suffix}"
    pdf_path.write_bytes(out.pdfdata)
    return pdf_path

__all__ = [
    "apply_adhoc",
    "check_hops_env",
    "corrected_visibility_phase_amp",
    "export_fplot_pdf",
    "find_fringe",
    "fplot",
    "get_fringe_file",
    "get_fringe_params",
    "get_polarization",
    "load_type120",
    "load_type212",
    "spectrum",
    "timeseries",
]
