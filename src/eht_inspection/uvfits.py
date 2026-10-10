import os

import matplotlib.pyplot as plt
import numpy as np
import copy
import itertools

from .utils import (
    get_baselines_from_station_list,
    get_subplot_grid,
    hours_to_hhmmss,
    scan_ids_from_intervals,
    wrap_phase,
)


_LOAD_OBS_UVFITS_KEYS = (
    "times",
    "t1",
    "t2",
    "u",
    "v",
    "rr",
    "rl",
    "lr",
    "ll",
    "rrsigma",
    "rlsigma",
    "lrsigma",
    "llsigma",
    "scantable",
)


def _load_obs_uvfits_result_dict(
    times,
    t1,
    t2,
    u,
    v,
    rr,
    rl,
    lr,
    ll,
    rrsigma,
    rlsigma,
    lrsigma,
    llsigma,
    scantable,
):
    return dict(zip(
        _LOAD_OBS_UVFITS_KEYS,
        (
            times,
            t1,
            t2,
            u,
            v,
            rr,
            rl,
            lr,
            ll,
            rrsigma,
            rlsigma,
            lrsigma,
            llsigma,
            scantable,
        ),
    ))


def _index_ranges(indices):
    """Format selected integer indices as compact contiguous ranges."""
    indices = np.unique(indices)
    groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    return ",".join(
        str(g[0]) if len(g) == 1 else f"{g[0]}-{g[-1]}" for g in groups if len(g)
    )


def _scan_ids_from_uvfits_nx(times, scans, nx):
    """Prefer NX's one-based inclusive input record ranges over time windows."""
    def fallback():
        return (scan_ids_from_intervals(times, scans) if scans is not None
                else np.full(len(times), -1, dtype=int))

    if nx is None or not {"START VIS", "END VIS"}.issubset(nx.names):
        return fallback()
    ranges = np.column_stack([nx["START VIS"], nx["END VIS"]])
    valid = (np.all(np.isfinite(ranges)) and np.all(ranges == np.floor(ranges))
             and np.all(ranges[:, 0] >= 1) and np.all(ranges[:, 1] <= len(times))
             and np.all(ranges[:, 0] <= ranges[:, 1]))
    scan_ids = np.full(len(times), -1, dtype=int)
    if valid:
        for iscan, (start, end) in enumerate(ranges.astype(int)):
            selected = slice(start - 1, end)
            if np.any(scan_ids[selected] >= 0):
                valid = False
                break
            scan_ids[selected] = iscan
    if not valid:
        import warnings
        warnings.warn(
            "Invalid or overlapping NX visibility record ranges; assigning scans "
            "from time intervals instead. Overlapping time intervals may be ambiguous.",
            RuntimeWarning, stacklevel=2,
        )
        return fallback()
    return scan_ids


def _scan_times_from_ids(scan_ids, scans):
    """Map indexing IDs to NX interval labels without inferring missing times."""
    result = np.full((len(scan_ids), 2), "unknown", dtype=object)
    if scans is not None and len(scans):
        known = (scan_ids >= 0) & (scan_ids < len(scans))
        intervals = np.asarray(scans)[scan_ids[known]]
        finite = np.all(np.isfinite(intervals), axis=1)
        result[np.flatnonzero(known)[finite]] = hours_to_hhmmss(intervals[finite])
    return result[:, 0], result[:, 1]


def _print_uvfits_flag_locations(flags, t1, t2, scan_ids, starts, ends, ifs, labels):
    """Print grouped locations using the full NX interval for each scan."""
    affected = np.any(flags, axis=(1, 2, 3))
    baselines = np.char.add(np.char.add(t1, "-"), t2)
    for baseline, scan in sorted(set(zip(baselines[affected], scan_ids[affected]))):
        rows = (baselines == baseline) & (scan_ids == scan)
        first = np.flatnonzero(rows)[0]
        for ipol, label in enumerate(labels):
            bad = flags[rows, :, :, ipol]
            if not np.any(bad):
                continue
            ir, fi, _ = np.nonzero(bad)
            count = len(ir)
            print(
                f"  {baseline} {label} time {starts[first]}-{ends[first]}: "
                f"{count}/{bad.size} samples ({100 * count / bad.size:.2f}%); "
                f"{len(np.unique(ir))}/{np.count_nonzero(rows)} affected records; "
                f"IF {_index_ranges(ifs[fi])}"
            )


def _print_uvfits_flags(flags, keep, t1, t2, scan_ids, starts, ends, ifs, labels,
                        finite_nonzero=None):
    """Report original selected file flags before polarization forcing."""
    if not np.any(flags):
        return
    affected = np.any(flags, axis=(1, 2, 3))
    nrecord = len(keep)
    print("UVFITS flags (selected input; weights <= 0 or NaN):")
    for name, count, total in (
        ("Flagged samples", np.count_nonzero(flags), flags.size),
        ("Affected records", np.count_nonzero(affected), nrecord),
        ("Dropped records", np.count_nonzero(~keep), nrecord),
    ):
        print(f"  {name}: {count}/{total} ({100 * count / total:.2f}%)")
    if finite_nonzero is not None:
        for ipol, label in enumerate(labels):
            present = finite_nonzero[:, :, :, ipol]
            print(
                f"  {label}: {np.count_nonzero(flags[:, :, :, ipol])} flagged samples; "
                f"{np.count_nonzero(present)} non-NaN/nonzero samples in "
                f"{np.count_nonzero(np.any(present, axis=(1, 2)))} records"
            )
    _print_uvfits_flag_locations(flags, t1, t2, scan_ids, starts, ends, ifs, labels)
    if finite_nonzero is not None and np.any(finite_nonzero):
        print("Flagged samples with finite, nonzero original observations:")
        _print_uvfits_flag_locations(
            finite_nonzero, t1, t2, scan_ids, starts, ends, ifs, labels,
        )


def _summarize_uvfits_flags(flags, keep, t1, t2, scan_ids, starts, ends, labels, ifs,
                            finite_nonzero):
    """Count original selected flags, retaining unflagged group denominators."""
    import pandas as pd

    baselines = np.char.add(np.char.add(t1, "-"), t2)
    rows = []
    if_rows = []
    for scan, baseline in sorted(set(zip(scan_ids, baselines))):
        selected = (scan_ids == scan) & (baselines == baseline)
        first = np.flatnonzero(selected)[0]
        for ipol, label in enumerate(labels):
            bad = flags[selected, :, :, ipol]
            present = finite_nonzero[selected, :, :, ipol]
            count = np.count_nonzero(bad)
            rows.append({
                "scan": int(scan),
                "scan_start": starts[first],
                "scan_end": ends[first],
                "baseline": baseline,
                "polarization": label,
                "flagged_samples": count,
                "total_samples": bad.size,
                "flagged_fraction": count / bad.size,
                "affected_records": np.count_nonzero(np.any(bad, axis=(1, 2))),
                "total_records": np.count_nonzero(selected),
                "dropped_records": np.count_nonzero(~keep[selected]),
                "finite_nonzero_samples": np.count_nonzero(present),
                "finite_nonzero_records": np.count_nonzero(np.any(present, axis=(1, 2))),
            })
            for i, if_index in enumerate(ifs):
                if_rows.append({
                    "scan": int(scan), "scan_start": starts[first],
                    "scan_end": ends[first], "baseline": baseline,
                    "polarization": label, "IF": int(if_index),
                    "flagged_samples": np.count_nonzero(bad[:, i, :]),
                    "total_samples": bad[:, i, :].size,
                })
    return pd.DataFrame(rows), pd.DataFrame(if_rows)


def load_obs_uvfits(
    filename,
    polrep='stokes',
    flipbl=False,
    allow_singlepol=True,
    force_singlepol=None,
    channel=all,
    IF=all,
    remove_nan=False,
    ignore_pzero_date=True,
    trial_speedups=False,
    return_dict=False,
    *,
    print_flag_summary=True,
    include_flag_summary=False,
    include_scan_ids=False,
):
    """Load observation data from a uvfits file.

       Args:
           filename: path to either an input text file or an HDUList object
           polrep: load data as either 'stokes' or 'circ'
           flipbl: flip baseline phases if True.
           allow_singlepol: If True and polrep='stokes',
                            treat single-polarization data as Stokes I
           force_singlepol: 'R' or 'L' to load only 1 polarization and treat as Stokes I
           channel: selected channels; the return retains only the first selected
                    channel within each IF (supported bandpass layout: one channel/IF).
           IF: selected IFs, retained as separate frequency bins in the return.
           remove_nan: whether or not to remove entries with nan data

           ignore_pzero_date: if True, ignore the offset parameters in DATE field
                              TODO: what is the correct behavior per AIPS memo 117?
           return_dict: If False, preserve the historical tuple return.
                        If True, return a dictionary keyed by array name
                        and include the UVFITS scan table as ``scantable``.
           print_flag_summary: If True (default), print the flag summary and
                               grouped locations when flags exist. False skips
                               this report without changing data or other messages.
           include_flag_summary: If True, requires ``return_dict=True`` and adds
                                 ``flag_summary``, a pandas DataFrame of original
                                 selected counts per scan/baseline/polarization.
                                 Includes NX time labels, zero-flag groups, dropped
                                 records, and counts of finite, nonzero original
                                 observations in flagged samples (before masking).
                                 Also adds ``flag_summary_if``, with original IF
                                 indices and flagged/total samples per
                                 scan/baseline/polarization/IF, including zero flags.
                                 This extra table does not change printed output.
           include_scan_ids: If True, requires ``return_dict=True`` and adds
                             ``scan_ids``, ``scan_start``, and ``scan_end``,
                             aligned with the returned records. Time labels are
                             rounded NX boundaries in HH:MM:SS, without wrapping
                             at 24 hours; unknown membership uses "unknown".
                             Uses valid NX record ranges, with time-interval
                             fallback if ranges are missing or invalid.
       Returns:
           ``(times, t1, t2, u, v, rr, rl, lr, ll, rrsigma, rlsigma,
           lrsigma, llsigma)`` by default. If ``return_dict=True``, return
           keys ``times``, ``t1``, ``t2``, ``u``, ``v``, ``rr``, ``rl``,
           ``lr``, ``ll``, ``rrsigma``, ``rlsigma``, ``lrsigma``,
           ``llsigma``, and ``scantable``. Visibility arrays have shape
           ``(Nrecord, Nchan)`` after IF/channel selection. Polarization
           order in the returned tuple is ``RR, RL, LR, LL``. Phases are
           not extracted here; returned visibilities are complex.

       Notes:
           Original selected weights <= 0 or NaN are reported by baseline,
           polarization, rounded NX time interval, and IF. Counts distinguish flagged
           samples, affected input records, and records dropped by the existing
           retention rule. Absent products and forced-polarization placeholders
           are not file flags. Observations absent from the file are not inferred.
           ``flag_summary`` contains ``scan`` (-1 when unknown), ``baseline``,
           ``polarization``, ``flagged_samples``, ``total_samples``,
           ``flagged_fraction`` (0 to 1), ``affected_records``, ``total_records``,
           ``dropped_records``, ``scan_start``, ``scan_end``,
           ``finite_nonzero_samples``, and ``finite_nonzero_records``. The audit
           requires both real and imaginary components finite, with at least one
           nonzero. With printing enabled, per-product audit totals follow the
           dropped-record total. A sample is one record/IF/channel/product
           cell. Dropped-record counts describe the loader's retention rule,
           shared across products, rather than a per-product drop decision.
           Scan IDs are zero-based NX row indices (-1 when unknown). File-based
           flags and IDs use the original one-based inclusive ``START VIS`` /
           ``END VIS`` ranges before filtering; NX time windows can overlap.
    """
    if include_flag_summary and not return_dict:
        raise ValueError("include_flag_summary requires return_dict=True")
    if include_scan_ids and not return_dict:
        raise ValueError("include_scan_ids requires return_dict=True")
    from astropy.io import fits
    import ehtim.const_def as ehc
    if not(polrep in ['stokes', 'circ']):
        raise Exception("polrep should be 'stokes' or 'circ' in load_uvfits")
    if not(force_singlepol is None or force_singlepol is False) and polrep != 'stokes':
        raise Exception(
            "force_singlepol is incompatible with polrep!='stokes' in load_uvfits")
    # Load the uvfits file
    if isinstance(filename, fits.HDUList):
        hdulist = filename.copy()
    else:
        print("Loading uvfits: ", filename)
        hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data
    # Load the array data
    tnames = hdulist['AIPS AN'].data['ANNAME']
    tnums = hdulist['AIPS AN'].data['NOSTA'] - 1
    xyz = np.real(hdulist['AIPS AN'].data['STABXYZ'])
    try:
        sefdr = np.real(hdulist['AIPS AN'].data['SEFD'])
        sefdl = np.real(hdulist['AIPS AN'].data['SEFD'])  # TODO add sefdl to uvfits?
    except KeyError:
        sefdr = np.zeros(len(tnames))
        sefdl = np.zeros(len(tnames))
    # TODO - get the *actual* values of these telescope parameters from the uvfits file?
    fr_par = np.zeros(len(tnames))
    fr_el = np.zeros(len(tnames))
    fr_off = np.zeros(len(tnames))
    dr = np.zeros(len(tnames)) + 1j * np.zeros(len(tnames))
    dl = np.zeros(len(tnames)) + 1j * np.zeros(len(tnames))
    tarr = [np.array((
            str(tnames[i]), xyz[i][0], xyz[i][1], xyz[i][2],
            sefdr[i], sefdl[i], dr[i], dl[i],
            fr_par[i], fr_el[i], fr_off[i]),
        dtype=ehc.DTARR) for i in range(len(tnames))]
    tarr = np.array(tarr)
    # Various header parameters
    try:
        ra = header['OBSRA'] * 12. / 180.
        dec = header['OBSDEC']
    except KeyError:
        if header['CTYPE6'] == 'RA':
            ra = header['CRVAL6'] * 12. / 180.
        else:
            raise Exception('Cannot find RA!')
        if header['CTYPE7'] == 'DEC':
            dec = header['CRVAL7']
        else:
            raise Exception('Cannot find DEC!')
    # catch bug if RA is in decimal degrees and > 24
    if ra>24 and ra>=0:
        ranew = ra*12/180.
        if ranew<24:
            print(' Warning! file RA>24, interpreting as decimal deg. : %.3f deg -> %.3f hr'%(ra,ranew))
            ra = ranew
        else:
            raise Exception('Cannot interpret fits file RA %.23!'%ra)
    elif ra<0:
        raise Exception('fits file RA %.3f<0!'%ra)
    src = header['OBJECT']
    rf = hdulist['AIPS AN'].header['FREQ']
    if header['CTYPE4'] == 'FREQ':
        ch1_freq = header['CRVAL4']
        ch_bw = header['CDELT4']
        nchan = header['NAXIS4']
    else:
        raise Exception('Cannot find observing frequencies!')
    nif = 1
    try:
        if header['CTYPE5'] == 'IF':
            nif = header['NAXIS5']
    except KeyError:
        print("no IF in uvfits header!")
    try:
        if header['CTYPE3'] == 'STOKES':
            if header['CRVAL3'] == 1:
                polrep_uvfits = 'stokes'
            elif header['CRVAL3'] == -1:
                polrep_uvfits = 'circ'
            else:
                raise Exception("header[CRVAL3] not a recognized polarization basis!")
    except BaseException:
        raise Exception("STOKES field not in expected header position 'CTYPE3'!")
    print('POLREP_UVFITS:', polrep_uvfits)
    if polrep_uvfits == 'stokes' and not(force_singlepol is None):
        raise Exception("force_singlepole not implemented on native Stokes uvfits files!")
    # determine the bandwidth
    bw = ch_bw * nchan * nif
    # Determine the number of correlation products in the data
    num_corr = data['DATA'].shape[5]
    print("Number of uvfits Correlation Products:", num_corr)
    if num_corr == 1 and force_singlepol is not None:
        print("Cannot force single polarization when file is not full polarization.")
        force_singlepol = None
    # If the user selects force_singlepol, then we must allow_singlepol for stokes conversion
    if force_singlepol is not None and polrep == 'stokes':
        allow_singlepol = True
    # Mask to screen bad data
    # Reducing to single frequency
    # prepare the arrays of if and channels that will be extracted from the data.
    nvis = data['DATA'].shape[0]
    full_nchannels = data['DATA'].shape[4]
    full_nifs = data['DATA'].shape[3]
    if channel == all:
        channel = np.arange(0, full_nchannels, 1)
        nchannels = full_nchannels
    else:
        try:
            nchannels = len(np.array(channel))
            channel = np.array(channel).reshape(-1)
        except TypeError:
            channel = np.array([channel]).reshape(-1)
            nchannels = len(np.array(channel))
    if IF == all:
        IF = np.arange(0, full_nifs, 1)
        nifs = full_nifs
    else:
        try:
            nifs = len(IF)
            IF = np.array(IF).reshape(-1)
        except TypeError:
            IF = np.array([IF]).reshape(-1)
            nifs = len(np.array(IF))
    if (np.max(channel) >= full_nchannels) or (np.min(channel) < 0):
        raise Exception('The specified channel does not exist')
    if (np.max(IF) >= full_nifs) or (np.min(IF) < 0):
        raise Exception('The specified IF does not exist')
    # NOTE: here we are assuming data is in RR, LL, RL, LR basis with the variable names
    # BUT: polrep_uvfits will correctly interpret these data as IQUV if necessary
    # TODO: change the variable names!
    rrweight = data['DATA'][:, 0, 0, IF, channel, 0, 2].reshape(nvis, nifs, nchannels)
    if num_corr >= 2:
        llweight = data['DATA'][:, 0, 0, IF, channel, 1, 2].reshape(nvis, nifs, nchannels)
    else:
        llweight = rrweight * 0.0
    if num_corr >= 3:
        rlweight = data['DATA'][:, 0, 0, IF, channel, 2, 2].reshape(nvis, nifs, nchannels)
    else:
        rlweight = rrweight * 0.0
    if num_corr >= 4:
        lrweight = data['DATA'][:, 0, 0, IF, channel, 3, 2].reshape(nvis, nifs, nchannels)
    else:
        lrweight = rrweight * 0.0
    if print_flag_summary or include_flag_summary:
        file_flags = np.stack([
            (weight <= 0) | np.isnan(weight)
            for weight in (rrweight, llweight, rlweight, lrweight)[:num_corr]
        ], axis=-1)
    finite_nonzero = None
    if include_flag_summary:
        finite_nonzero = np.zeros_like(file_flags)
        for ipol in range(num_corr):
            real = data['DATA'][:, 0, 0, IF, channel, ipol, 0].reshape(nvis, nifs, nchannels)
            imag = data['DATA'][:, 0, 0, IF, channel, ipol, 1].reshape(nvis, nifs, nchannels)
            finite_nonzero[:, :, :, ipol] = (
                file_flags[:, :, :, ipol] & np.isfinite(real) & np.isfinite(imag)
                & ((real != 0) | (imag != 0))
            )
    # If necessary, enforce single polarization
    if polrep_uvfits == 'circ':
        if force_singlepol in ['L', 'LL']:
            rrweight = rrweight * 0.0
            rlweight = rlweight * 0.0
            lrweight = lrweight * 0.0
        elif force_singlepol in ['R', 'RR']:
            llweight = llweight * 0.0
            rlweight = rlweight * 0.0
            lrweight = lrweight * 0.0
        elif force_singlepol == 'LR':
            print('WARNING: Putting LR data in Stokes I')
            rrweight = copy.deepcopy(lrweight)
            llweight = llweight * 0.0
            rlweight = rlweight * 0.0
            lrweight = lrweight * 0.0
        elif force_singlepol == 'RL':
            print('WARNING: Putting RL data in Stokes I')
            rrweight = copy.deepcopy(rlweight)
            llweight = llweight * 0.0
            rlweight = rlweight * 0.0
            lrweight = lrweight * 0.0
    # first, catch  nans
    rrnanmask_2d = (np.isnan(rrweight))
    llnanmask_2d = (np.isnan(llweight))
    rlnanmask_2d = (np.isnan(rlweight))
    lrnanmask_2d = (np.isnan(lrweight))
    rrweight[rrnanmask_2d] = 0.
    llweight[llnanmask_2d] = 0.
    rlweight[rlnanmask_2d] = 0.
    lrweight[lrnanmask_2d] = 0.
    # look for weights < 0
    rrmask_2d = (rrweight > 0.)
    llmask_2d = (llweight > 0.)
    rlmask_2d = (rlweight > 0.)
    lrmask_2d = (lrweight > 0.)
    # if there is any unmasked data in the frequency column, use it
    rrmask = np.any(np.any(rrmask_2d, axis=2), axis=1)
    llmask = np.any(np.any(llmask_2d, axis=2), axis=1)
    rlmask = np.any(np.any(rlmask_2d, axis=2), axis=1)
    lrmask = np.any(np.any(lrmask_2d, axis=2), axis=1)
    # Total intensity mask
    if polrep_uvfits == 'circ':
        mask = rrmask + llmask
    elif polrep_uvfits == 'stokes':
        mask = rrmask  # remember rr is really I when polrep_uvfits=='stokes'!
    # Obs Times
    paridx = data.parnames.index("DATE")+1
    if "PSCAL%d"%(paridx) in header.keys():
        jd1scal = header["PSCAL%d"%(paridx)]
    else:
        jd1scal = 1.0
    if "PZERO%d"%(paridx) in header.keys():
        jd1zero = header["PZERO%d"%(paridx)]
    else:
        jd1zero = 0.0
    if "PSCAL%d"%(paridx+1) in header.keys():
        jd2scal = header["PSCAL%d"%(paridx+1)]
    else:
        jd2scal = 1.0
    if "PZERO%d"%(paridx+1) in header.keys():
        jd2zero = header["PZERO%d"%(paridx+1)]
    else:
        jd2zero = 0.0
    if ignore_pzero_date:
        if jd1zero!=0. or jd2zero!=0.:
            print("Warning! ignoring nonzero header PZERO values for DATE. Check your observation mjd/times!")
        jd1zero = 0.
        jd2zero = 0.
    jds = jd1scal * data['DATE'].astype('d') + jd1zero
    jds += jd2scal * data['_DATE'].astype('d') + jd2zero
    mjd = int(np.min(jds[mask] if np.any(mask) else jds) - 2400000.5)
    input_times = (jds - 2400000.5 - mjd) * 24.0
    times = input_times[mask]
    nx_data = None
    try:
        scantable = []
        nxtable = hdulist['AIPS NX']
        for scan in nxtable.data:
            scan_start = scan['TIME']  # in days since reference date
            scan_dur = scan['TIME INTERVAL']
            scantable.append([scan_start - 0.5 * scan_dur,
                              scan_start + 0.5 * scan_dur])
        scantable = np.array(scantable) * 24
        nx_data = nxtable.data
    except BaseException:
        print("No NX table in uvfits!")
        scantable = None
    if print_flag_summary or include_flag_summary or include_scan_ids:
        input_scan_ids = _scan_ids_from_uvfits_nx(input_times, scantable, nx_data)
        scan_start, scan_end = _scan_times_from_ids(input_scan_ids, scantable)
    # Integration times
    try:
        tints = data['INTTIM'][mask]
    except KeyError:
        tints = np.zeros(len(mask))
    # Sites - add names
    t1c = data['BASELINE'].astype(int) // 256
    t2c = data['BASELINE'].astype(int) - t1c * 256
    t1c = t1c - 1
    t2c = t2c - 1
    # TODO make site identificantion faster
    if trial_speedups and (not np.any(tnums!=np.arange(len(tnums)))):
        sites = tarr['site']
        t1 = sites[t1c]
        t2 = sites[t2c]
    else: # original, slow code
        t1 = np.array([tarr[np.where(tnums==i)[0][0]]['site'] for i in t1c])
        t2 = np.array([tarr[np.where(tnums==i)[0][0]]['site'] for i in t2c])
    if print_flag_summary or include_flag_summary:
        labels = ("RR", "LL", "RL", "LR") if polrep_uvfits == 'circ' else ("I", "Q", "U", "V")
    if print_flag_summary:
        _print_uvfits_flags(
            file_flags, mask, t1, t2, input_scan_ids, scan_start, scan_end, IF,
            labels[:num_corr], finite_nonzero,
        )
    if include_flag_summary:
        flag_summary, flag_summary_if = _summarize_uvfits_flags(
            file_flags, mask, t1, t2, input_scan_ids, scan_start, scan_end,
            labels[:num_corr], IF, finite_nonzero,
        )
    if not np.any(mask):
        raise Exception("No unflagged RR or LL data in uvfits file!")
    t1, t2 = t1[mask], t2[mask]
    # Opacities (not in standard files)
    try:
        tau1 = data['TAU1'][mask]
        tau2 = data['TAU2'][mask]
    except KeyError:
        tau1 = tau2 = np.zeros(len(t1))
    # Convert uv in lightsec to lambda by multiplying by rf
    try:
        u = data['UU---SIN'][mask] * rf
        v = data['VV---SIN'][mask] * rf
    except KeyError:
        try:
            u = data['UU'][mask] * rf
            v = data['VV'][mask] * rf
        except KeyError:
            try:
                u = data['UU--'][mask] * rf
                v = data['VV--'][mask] * rf
            except KeyError:
                raise Exception("Cant figure out column label for UV coords")
    # Get and coherently average visibility data in frequency
    # replace masked vis with nans so they don't mess up the average
    rr_2d = data['DATA'][:, 0, 0, IF, channel, 0, 0] + \
        1j * data['DATA'][:, 0, 0, IF, channel, 0, 1]
    rr_2d = rr_2d.reshape(nvis, nifs, nchannels)
    if num_corr >= 2:
        ll_2d = data['DATA'][:, 0, 0, IF, channel, 1, 0] + \
            1j * data['DATA'][:, 0, 0, IF, channel, 1, 1]
        ll_2d = ll_2d.reshape(nvis, nifs, nchannels)
    else:
        ll_2d = rr_2d * 0.0
    if num_corr >= 3:
        rl_2d = data['DATA'][:, 0, 0, IF, channel, 2, 0] + \
            1j * data['DATA'][:, 0, 0, IF, channel, 2, 1]
        rl_2d = rl_2d.reshape(nvis, nifs, nchannels)
    else:
        rl_2d = rr_2d * 0.0
    if num_corr >= 4:
        lr_2d = data['DATA'][:, 0, 0, IF, channel, 3, 0] + \
            1j * data['DATA'][:, 0, 0, IF, channel, 3, 1]
        lr_2d = lr_2d.reshape(nvis, nifs, nchannels)
    else:
        lr_2d = rr_2d * 0.0
    if polrep_uvfits == 'circ':
        if force_singlepol == 'LR':
            rr_2d = copy.deepcopy(lr_2d)
        elif force_singlepol == 'RL':
            rr_2d = copy.deepcopy(rl_2d)
    rr_2d[~rrmask_2d] = np.nan
    ll_2d[~llmask_2d] = np.nan
    rl_2d[~rlmask_2d] = np.nan
    lr_2d[~lrmask_2d] = np.nan
    rr = np.nanmean(np.nanmean(rr_2d, axis=2), axis=1)[mask]
    ll = np.nanmean(np.nanmean(ll_2d, axis=2), axis=1)[mask]
    rl = np.nanmean(np.nanmean(rl_2d, axis=2), axis=1)[mask]
    lr = np.nanmean(np.nanmean(lr_2d, axis=2), axis=1)[mask]
    # average the weights
    # variances are mean / N , or sum / N^2
    # then replace masked weights with nans so they don't mess up the average
    rrweight[~rrmask_2d] = np.nan
    llweight[~llmask_2d] = np.nan
    rlweight[~rlmask_2d] = np.nan
    lrweight[~lrmask_2d] = np.nan
    nsig_rr = np.sum(np.sum(rrmask_2d, axis=2), axis=1).astype(float)
    nsig_rr[~rrmask] = np.nan
    rrsig = np.sqrt(np.nansum(np.nansum(1. / rrweight, axis=2), axis=1)) / nsig_rr
    rrsig = rrsig[mask]
    nsig_ll = np.sum(np.sum(llmask_2d, axis=2), axis=1).astype(float)
    nsig_ll[~llmask] = np.nan
    llsig = np.sqrt(np.nansum(np.nansum(1. / llweight, axis=2), axis=1)) / nsig_ll
    llsig = llsig[mask]
    nsig_rl = np.sum(np.sum(rlmask_2d, axis=2), axis=1).astype(float)
    nsig_rl[~rlmask] = np.nan
    rlsig = np.sqrt(np.nansum(np.nansum(1. / rlweight, axis=2), axis=1)) / nsig_rl
    rlsig = rlsig[mask]
    nsig_lr = np.sum(np.sum(lrmask_2d, axis=2), axis=1).astype(float)
    nsig_lr[~lrmask] = np.nan
    lrsig = np.sqrt(np.nansum(np.nansum(1. / lrweight, axis=2), axis=1)) / nsig_lr
    lrsig = lrsig[mask]
    # Reverse sign of baselines for correct imaging if asked
    if flipbl:
        u = -u
        v = -v
    # determine correct data type:
    # TODO add linear!
    if polrep_uvfits == 'circ':
        dtpol_out = ehc.DTPOL_CIRC
        poldict_out = ehc.POLDICT_CIRC
    elif polrep_uvfits == 'stokes':
        dtpol_out = ehc.DTPOL_STOKES
        poldict_out = ehc.POLDICT_STOKES
    result = (
        times,
        t1,
        t2,
        u,
        v,
        rr_2d[mask, :, 0],
        rl_2d[mask, :, 0],
        lr_2d[mask, :, 0],
        ll_2d[mask, :, 0],
        np.sqrt(1. / rrweight)[mask, :, 0],
        np.sqrt(1. / rlweight)[mask, :, 0],
        np.sqrt(1. / lrweight)[mask, :, 0],
        np.sqrt(1. / llweight)[mask, :, 0],
    )
    if return_dict:
        result_dict = _load_obs_uvfits_result_dict(*result, scantable)
        if include_flag_summary:
            result_dict["flag_summary"] = flag_summary
            result_dict["flag_summary_if"] = flag_summary_if
        if include_scan_ids:
            result_dict["scan_ids"] = input_scan_ids[mask]
            result_dict["scan_start"] = scan_start[mask]
            result_dict["scan_end"] = scan_end[mask]
        return result_dict
    return result

#     #TODO new, faster,
#     if trial_speedups:
#         datatable = np.empty((len(times)),dtype=dtpol_out)
#         datatable['time'] = times
#         datatable['tint'] = tints
#         datatable['t1'] = t1
#         datatable['t2'] = t2
#         datatable['tau1'] = tau1
#         datatable['tau2'] = tau2
#         datatable['u'] = u
#         datatable['v'] = v
#         datatable[poldict_out['vis1']] = rr
#         datatable[poldict_out['vis2']] = ll
#         datatable[poldict_out['vis3']] = rl
#         datatable[poldict_out['vis4']] = lr
#         datatable[poldict_out['sigma1']] = rrsig
#         datatable[poldict_out['sigma2']] = llsig
#         datatable[poldict_out['sigma3']] = rlsig
#         datatable[poldict_out['sigma4']] = lrsig
#     else: # original, slower code
#         datatable = []
#         for i in range(len(times)):
#             datatable.append(np.array
#                              ((
#                                  times[i], tints[i],
#                                  t1[i], t2[i], tau1[i], tau2[i],
#                                  u[i], v[i],
#                                  rr[i], ll[i], rl[i], lr[i],
#                                  rrsig[i], llsig[i], rlsig[i], lrsig[i]
#                              ), dtype=dtpol_out
#                              ))
#         datatable = np.array(datatable)

#     obs = ehtim.obsdata.Obsdata(ra, dec, rf, bw, datatable, tarr, polrep=polrep_uvfits,
#                                 source=src, mjd=mjd, scantable=scantable,
#                                 trial_speedups=trial_speedups)

#     # TODO -- this is bad and slow, use masks!
#     if remove_nan:
#         if polrep_uvfits == 'circ':
#             for j in range(len(obs.data)):
#                 if np.isnan(obs.data[j]['rrsigma']):
#                     obs.data[j]['rrsigma'] = obs.data[j]['llsigma']
#                 if np.isnan(obs.data[j]['llsigma']):
#                     obs.data[j]['llsigma'] = obs.data[j]['rrsigma']
#                 if np.isnan(obs.data[j]['rlsigma']):
#                     obs.data[j]['rlsigma'] = obs.data[j]['rrsigma']
#                 if np.isnan(obs.data[j]['lrsigma']):
#                     obs.data[j]['lrsigma'] = obs.data[j]['rrsigma']
#         else:
#             print("WARNING: remove_nan not implemented with stokes uvfits files!")

#     obs = obs.switch_polrep(polrep, allow_singlepol=allow_singlepol)

#     # TODO get calibration flags from uvfits?
#     return obs,datatable


def build_scan_coherency_matrix(
    scannum,
    scan_ids,
    times,
    t1,
    t2,
    u,
    v,
    rr,
    rl,
    lr,
    ll,
    fill_missing=0.0 + 0.0j,
    conjugate_reverse=True,
    flip_uv_reverse=True,
    *,
    sigmas=None,
    unaveraged=False,
):
    """
    Build a dense station-by-station coherency matrix for one scan.

    This array-based builder does not infer NX metadata. Attach scalar
    ``scan_start`` and ``scan_end`` HH:MM:SS labels to the result before making
    time-labelled plots or bandpass summaries, or use the UVFITS wrapper.

    This function takes UVFITS-style visibility arrays in row-based form,
    where each row corresponds to one time/baseline record, and reorganizes
    them into a dense array:

        allcoh[time_index, channel_index, station_i, station_j, pol_i, pol_j]

    where the 2x2 polarization block is:

        [[RR, RL],
         [LR, LL]]
    or if ALMA presents:
        [[XR, XL],
         [YR, YL]]

    Parameters
    ----------
    scannum : int
        Scan number to select.

    scan_ids : array-like, shape (Nrecords,)
        Integer scan ID for each visibility record.
        This should already have been created from obs.scans / times.

    times : array-like, shape (Nrecords,)
        Time value for each visibility record.

    t1, t2 : array-like, shape (Nrecords,)
        Station names for the first and second station of each baseline.

    u, v : array-like, shape (Nrecords,)
        UV coordinates for each visibility record.
        Usually in wavelengths for eht-imaging Obsdata.

    rr, rl, lr, ll : array-like
        Visibility arrays for the four circular polarization products.

        Expected shape is either:

            (Nrecords, Nchannels)

        or something that can be interpreted that way.

    fill_missing : complex, optional
        Value used for missing baseline/time/channel entries.
        Default is 0 + 0j.

    conjugate_reverse : bool, optional
        If True, fill the reverse baseline j-i using the Hermitian relation:

            V_ji = V_ij^dagger

        i.e. conjugate and transpose the 2x2 polarization matrix.

    flip_uv_reverse : bool, optional
        If True, reverse baselines get negative uv coordinates:

            u_ji = -u_ij
            v_ji = -v_ij

        This is usually the physically correct convention.

    sigmas : dict or None, optional
        Per-component thermal uncertainties keyed by ``rr``, ``rl``, ``lr``,
        and ``ll``, with arrays matching the corresponding visibilities.
        When supplied, add ``allsigma`` with the same axes as ``allcoh`` and
        NaN for missing entries. Reverse baselines transpose its pol block.

    unaveraged : bool, optional
        Caller declaration that input is unaveraged. Required for bandpass
        SNR titles; initially supported for multiple IFs, one channel per IF.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "allcoh" : complex ndarray, shape (Nt, Nc, Nstation, Nstation, 2, 2)
            Dense coherency matrix.

        - "allu" : float ndarray, shape (Nstation, Nstation)
            Mean u coordinate for each station pair.

        - "allv" : float ndarray, shape (Nstation, Nstation)
            Mean v coordinate for each station pair.

        - "station_list" : ndarray, shape (Nstation,)
            Sorted unique station names in this scan.

        - "t_unique" : ndarray, shape (Nt,)
            Sorted unique times in this scan.

        - "channel_list" : ndarray, shape (Nc,)
            Channel indices.

        - "scan_mask" : ndarray, shape (Nrecords,)
            Boolean mask selecting rows belonging to this scan.

        - "observed_baselines" : list[str]
            Station pairs present in the selected input records, in station-list
            order, including actual autocorrelations. Reversed pairs share one
            label. Presence is independent of visibility values and weights;
            records already dropped by the loader are not represented.

    Notes
    -----
    Unlike the original quick version, this function explicitly matches
    visibility records by time. Therefore it is safer when some baselines
    have missing records or uneven time sampling.
    """
    # Convert inputs to arrays
    scan_ids = np.asarray(scan_ids)
    times = np.asarray(times)
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    rr = np.asarray(rr)
    rl = np.asarray(rl)
    lr = np.asarray(lr)
    ll = np.asarray(ll)
    sigma_coh = None
    if sigmas is not None:
        keys = ("rr", "rl", "lr", "ll")
        if set(sigmas) != set(keys):
            raise ValueError("sigmas must contain rr, rl, lr, and ll")
        sigma_arrays = [np.asarray(sigmas[k], dtype=float) for k in keys]
        if any(s.shape != vis.shape for s, vis in zip(sigma_arrays, (rr, rl, lr, ll))):
            raise ValueError("sigmas arrays must match their visibility shapes")
        sigma_coh = np.stack(sigma_arrays, axis=-1).reshape(rr.shape + (2, 2))
    # Select records belonging to this scan
    scan_mask = scan_ids == scannum
    if not np.any(scan_mask):
        raise ValueError(f"No records found for scannum={scannum}")
    t = times[scan_mask]
    s1 = t1[scan_mask]
    s2 = t2[scan_mask]
    u2 = u[scan_mask]
    v2 = v[scan_mask]
    # Build coherency array for selected records:
    # shape: (Nrecords_in_scan, Nchannels, 2, 2)
    coh_all = np.transpose(
        np.array([[rr, rl], [lr, ll]]),
        axes=(2, 3, 0, 1)
    )
    coh = coh_all[scan_mask]
    # Unique times and stations in this scan
    t_unique = np.unique(t)
    station_list = np.unique(np.concatenate([s1, s2]))
    observed_pairs = {frozenset((a, b)) for a, b in zip(s1, s2)}
    observed_baselines = [
        baseline for baseline in get_baselines_from_station_list(station_list, include_autocorr=True)
        if frozenset(baseline.split("-")) in observed_pairs
    ]
    Nt = len(t_unique)
    Nc = coh.shape[1]
    Nstation = len(station_list)
    channel_list = np.arange(Nc)
    # Maps for quick indexing
    time_to_index = {time_val: i for i, time_val in enumerate(t_unique)}
    station_to_index = {st: i for i, st in enumerate(station_list)}
    # Output arrays
    allcoh = np.full(
        (Nt, Nc, Nstation, Nstation, 2, 2),
        fill_missing,
        dtype=complex,
    )
    allsigma = np.full(allcoh.shape, np.nan) if sigma_coh is not None else None
    sigma_scan = sigma_coh[scan_mask] if sigma_coh is not None else None
    allu = np.full((Nstation, Nstation), np.nan, dtype=float)
    allv = np.full((Nstation, Nstation), np.nan, dtype=float)
    # Fill direct baselines row by row
    for row in range(len(t)):
        it = time_to_index[t[row]]
        i = station_to_index[s1[row]]
        j = station_to_index[s2[row]]
        allcoh[it, :, i, j, :, :] = coh[row]
        if allsigma is not None:
            allsigma[it, :, i, j, :, :] = sigma_scan[row]
    # Fill mean u/v values for each baseline pair
    for i, st_i in enumerate(station_list):
        for j, st_j in enumerate(station_list):
            cond = (s1 == st_i) & (s2 == st_j)
            if np.any(cond):
                mean_u = np.mean(u2[cond])
                mean_v = np.mean(v2[cond])
                allu[i, j] = mean_u
                allv[i, j] = mean_v
                if conjugate_reverse:
                    allcoh[:, :, j, i, :, :] = np.transpose(
                        allcoh[:, :, i, j, :, :],
                        axes=(0, 1, 3, 2)
                    ).conjugate()
                    if allsigma is not None:
                        allsigma[:, :, j, i, :, :] = np.swapaxes(
                            allsigma[:, :, i, j, :, :], -1, -2
                        )
                if flip_uv_reverse:
                    allu[j, i] = -mean_u
                    allv[j, i] = -mean_v
                else:
                    allu[j, i] = mean_u
                    allv[j, i] = mean_v
    result = {
        "allcoh": allcoh,
        "allu": allu,
        "allv": allv,
        "station_list": station_list,
        "t_unique": t_unique,
        "channel_list": channel_list,
        "scan_mask": scan_mask,
        "scan_number": scannum,
        "observed_baselines": observed_baselines,
    }
    if allsigma is not None:
        result["allsigma"] = allsigma
    if unaveraged:
        result["unaveraged"] = True
    return result


def build_scan_coherency_matrix_from_uvfits(
    filename,
    scannum,
    scan_ids=None,
    scans=None,
    start_index=0,
    polrep='stokes',
    flipbl=False,
    allow_singlepol=True,
    force_singlepol=None,
    channel=all,
    IF=all,
    remove_nan=False,
    ignore_pzero_date=True,
    trial_speedups=False,
    fill_missing=0.0 + 0.0j,
    conjugate_reverse=True,
    flip_uv_reverse=True,
    *,
    unaveraged=False,
    print_flag_summary=True,
):
    """
    Load UVFITS data and build a scan coherency matrix.

    This is a convenience wrapper for the common workflow:

        obs = load_obs_uvfits(..., return_dict=True, include_scan_ids=True)
        scan_ids = obs["scan_ids"]
        build_scan_coherency_matrix(...)

    The lower-level ``build_scan_coherency_matrix`` remains focused on
    array-to-matrix conversion; UVFITS parsing stays in ``load_obs_uvfits``.

    Parameters
    ----------
    filename
        UVFITS filename or already-open FITS HDUList accepted by
        ``load_obs_uvfits``.

    scannum
        Scan number to pass to ``build_scan_coherency_matrix``.

    scan_ids
        Integer scan ID for each visibility record. If None, IDs are assigned
        from NX record ranges by default, or from ``scans`` if supplied.

    scans
        Scan intervals used when ``scan_ids`` is None. If omitted, the
        NX record ranges are preferred, with the NX time intervals as fallback.

    start_index
        First scan ID to assign when ``scan_ids`` is None. The default keeps
        zero-based scan IDs. Use ``start_index=1`` for one-based scan numbers.
        Records outside the NX record ranges (or supplied intervals) remain ``-1``.

    polrep, flipbl, allow_singlepol, force_singlepol, channel, IF, remove_nan,
    ignore_pzero_date, trial_speedups
        Passed through to ``load_obs_uvfits``.

    fill_missing, conjugate_reverse, flip_uv_reverse
        Passed through to ``build_scan_coherency_matrix``.

    unaveraged : bool, optional
        Declare that the file is unaveraged and carry loader uncertainties
        into ``allsigma`` for bandpass SNR. Averaging history is not inferred
        from filenames or headers. Initially supported for multiple IFs with
        one channel per IF; the caller must ensure this layout.

    print_flag_summary : bool, optional
        Passed to the loader. Default True; use False to avoid repeating the
        flag summary when building multiple scans from the same file.

    Returns
    -------
        Result from ``build_scan_coherency_matrix``.
        Includes scalar ``scan_start`` / ``scan_end`` labels from the selected
        records' shared NX interval. Selecting multiple distinct NX intervals
        raises ValueError. Unknown labels cannot be used for time-labelled plots.
    """
    obs = load_obs_uvfits(
        filename,
        polrep=polrep,
        flipbl=flipbl,
        allow_singlepol=allow_singlepol,
        force_singlepol=force_singlepol,
        channel=channel,
        IF=IF,
        remove_nan=remove_nan,
        ignore_pzero_date=ignore_pzero_date,
        trial_speedups=trial_speedups,
        return_dict=True,
        print_flag_summary=print_flag_summary,
        include_scan_ids=True,
    )
    if scan_ids is None:
        scan_intervals = obs["scantable"] if scans is None else scans
        if scan_intervals is None:
            raise ValueError(
                "scan_ids could not be inferred because no scan intervals "
                "were provided and the UVFITS file has no scantable"
            )
        if scans is None and "scan_ids" in obs:
            scan_ids = obs["scan_ids"]
        else:
            scan_ids = scan_ids_from_intervals(obs["times"], scan_intervals)
        if start_index != 0:
            scan_ids = np.asarray(scan_ids).copy()
            scan_ids[scan_ids >= 0] += start_index

    uncertainty_kwargs = {}
    if unaveraged:
        uncertainty_kwargs = {
            "sigmas": {pol: obs[pol + "sigma"] for pol in ("rr", "rl", "lr", "ll")},
            "unaveraged": True,
        }
    selected = np.asarray(scan_ids) == scannum
    intervals = set(zip(obs["scan_start"][selected], obs["scan_end"][selected]))
    if len(intervals) > 1:
        raise ValueError("Selected records have conflicting NX scan_start/scan_end intervals")
    result = build_scan_coherency_matrix(
        scannum,
        scan_ids,
        obs["times"],
        obs["t1"],
        obs["t2"],
        obs["u"],
        obs["v"],
        obs["rr"],
        obs["rl"],
        obs["lr"],
        obs["ll"],
        fill_missing=fill_missing,
        conjugate_reverse=conjugate_reverse,
        flip_uv_reverse=flip_uv_reverse,
        **uncertainty_kwargs,
    )
    if intervals:
        result["scan_start"], result["scan_end"] = intervals.pop()
    return result


def get_pol_labels_for_baseline(
    baseline,
    alma_station="AA",
):
    """
    Return polarization labels for a baseline.

    If the baseline contains ALMA, use mixed linear-circular labels.
    Otherwise use normal circular-circular labels.

    Notes
    -----
    We assume the coherency matrix is stored as:

        [[RR, RL],
         [LR, LL]]

    for ordinary circular-circular baselines.

    For ALMA mixed-pol baselines, the actual interpretation depends on
    station order. If the baseline is AA-X, the first station is ALMA,
    so the first polarization index is linear: X/Y.
    If the baseline is X-AA, the second station is ALMA,
    so the second polarization index is linear: X/Y.
    """
    s1, s2 = baseline.split("-")
    if s1 == alma_station and s2 != alma_station:
        # ALMA is first station: first pol index is linear, second is circular
        return {
            "XR": (0, 0),
            "XL": (0, 1),
            "YR": (1, 0),
            "YL": (1, 1),
        }
    elif s2 == alma_station and s1 != alma_station:
        # ALMA is second station: first pol index is circular, second is linear
        return {
            "RX": (0, 0),
            "RY": (0, 1),
            "LX": (1, 0),
            "LY": (1, 1),
        }
    else:
        # Normal circular-circular baseline
        return {
            "RR": (0, 0),
            "RL": (0, 1),
            "LR": (1, 0),
            "LL": (1, 1),
        }


# ====== Visibility Plotting Helpers ======


def _coherency_inputs_from_result(result):
    """
    Return allcoh/channel/station arrays from a result dict.
    """
    return result["allcoh"], result["channel_list"], result["station_list"]


def _scan_num_from_result(result, scan_num=None):
    if scan_num is not None:
        return scan_num
    if isinstance(result, dict):
        return result.get("scan_number", None)
    return None


def _scan_time_seconds(value):
    """Parse a rounded label, including hours beyond 24, for ordering."""
    try:
        parts = value.split(":") if isinstance(value, str) else []
        if (len(parts) != 3 or not all(part.isdigit() for part in parts)
                or len(parts[0]) < 2 or len(parts[1]) != 2 or len(parts[2]) != 2):
            raise ValueError
        hours, minutes, seconds = map(int, parts)
        if not (0 <= minutes < 60 and 0 <= seconds < 60):
            raise ValueError
        return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        raise ValueError("Known scan_start and scan_end metadata in HH:MM:SS format are required") from None


def _scan_times_from_result(result):
    """Require explicit interval metadata; never infer it from retained data."""
    start, end = result.get("scan_start"), result.get("scan_end")
    if _scan_time_seconds(start) > _scan_time_seconds(end):
        raise ValueError("Conflicting scan_start/scan_end metadata: start is after end")
    return start, end


def _baselines_from_result(result, include_autocorr=False, *,
                           print_missing_baselines=False, obs_day=None):
    """Filter station combinations by loaded record presence, when available."""
    candidates = get_baselines_from_station_list(
        result["station_list"], include_autocorr=include_autocorr,
    )
    if "observed_baselines" not in result:
        return candidates  # Legacy results do not establish baseline presence.
    observed = set(result["observed_baselines"])
    missing = [baseline for baseline in candidates if baseline not in observed]
    if print_missing_baselines and missing:
        start, end = _scan_times_from_result(result)
        day = "" if obs_day is None else f"{obs_day} "
        print(f"Missing baselines from loaded scan records, {day}time {start}-{end}: "
              + ", ".join(missing))
    return [baseline for baseline in candidates if baseline in observed]


def _observation_label(source, obs_day):
    return source if obs_day is None else f"{source}, {obs_day}"


def _scan_title(source, start, quantity, domain, detail=None, *, obs_day=None, end=None):
    time_text = f", time {start}" + (f"-{end}" if end is not None else "")
    detail_text = "" if detail is None else f" ({detail})"
    return f"{_observation_label(source, obs_day)}{time_text}: {quantity} vs {domain}{detail_text}"


def _scan_filename(source, start, suffix, prefix="", *, obs_day=None):
    day_text = "" if obs_day is None else f"_{obs_day}"
    time_text = "" if start is None else f"_{start.replace(':', '')}"
    return f"{prefix}{source}{day_text}{time_text}_{suffix}.png"


def _visibility_var_info(var):
    q = var.lower()
    if q in {"phase", "phas"}:
        return "phase", "phase", "Phase [deg]", "phase"
    if q in {"amp", "amplitude"}:
        return "amp", "amplitude", "Amplitude", "amp"
    raise ValueError("var must be either 'phase' or 'amp'")


def _visibility_quantity(values, var, unwrap_phase=False):
    var_key, _, _, _ = _visibility_var_info(var)
    if var_key == "phase":
        if unwrap_phase:
            return np.rad2deg(np.unwrap(np.angle(values)))
        return np.angle(values, deg=True)
    return np.abs(values)


def _channel_index(channel_list, channel, nchannel):
    channel_list = np.asarray(channel_list)
    if channel in channel_list:
        ichan = np.where(channel_list == channel)[0][0]
    else:
        ichan = int(channel)
    if ichan < 0 or ichan >= nchannel:
        raise IndexError(f"channel index {ichan} out of range for Nc={nchannel}")
    return ichan


def _set_amp_scale(ax, var, amp_scale):
    var_key, _, _, _ = _visibility_var_info(var)
    if var_key != "amp":
        return
    if amp_scale == "log":
        ax.set_yscale("log")
    elif amp_scale != "linear":
        raise ValueError("amp_scale must be 'linear' or 'log'")


# ====== Visibility Plotting ======


def _bandpass_uncertainty(V, V_chan, sigma):
    """Propagate component uncertainties for the plotted arithmetic mean."""
    present = ~np.isnan(V)
    counts = np.sum(present, axis=0)
    valid_sigma = np.isfinite(sigma) & (sigma > 0)
    usable = (counts > 0) & np.all(~present | valid_sigma, axis=0)
    variance_sum = np.sum(np.where(present & valid_sigma, sigma, 0.0) ** 2, axis=0)
    mean_sigma = np.full(counts.shape, np.nan)
    np.divide(np.sqrt(variance_sum), counts, out=mean_sigma, where=usable)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(V_chan) / mean_sigma
    return mean_sigma, snr


def _bandpass_snr(V, V_chan, sigma):
    """Median IF SNR of the plotted arithmetic complex mean."""
    _, snr = _bandpass_uncertainty(V, V_chan, sigma)
    snr = snr[np.isfinite(snr)]
    return f"{np.median(snr):.1f}".removesuffix(".0") if snr.size else "N/A"


def summarize_scan_bandpass(
    result,
    *,
    obs_day=None,
    scan_num=None,
    alma_station="AA",
    include_autocorr=False,
    print_missing_baselines=True,
    excluded_ifs_by_station=None,
):
    """Return frequency-scatter statistics of one coherently averaged scan.

    Parameters
    ----------
    result : dict
        Scan coherency result with ``unaveraged=True`` and ``allsigma``
        matching ``allcoh``. Supported input retains multiple IFs with one
        channel per IF. As in the bandpass plot, zero placeholders and NaNs
        do not contribute to the arithmetic complex mean over time.
    obs_day : str or None, optional
        Caller-supplied file/day label; no date is inferred. Default is missing.
    scan_num : int or None, optional
        Override ``result['scan_number']``; otherwise use that existing label.
    alma_station : str, optional
        Station using mixed polarization labels, following the bandpass plot.
    include_autocorr : bool, optional
        Include station autocorrelations. Default is False, as in the plot.
    print_missing_baselines : bool, optional
        Print absent station combinations once using the NX interval and optional
        day label. Default True. Filtering is independent of this switch.
    excluded_ifs_by_station : mapping or None, optional
        Station names mapped to zero-based IF positions in the supplied spectrum,
        e.g. {"NN": [11, 28]} when using IF=all. For every polarization of a
        baseline involving that station, replace those coherently time-averaged
        values with complex NaN before selecting usable IFs. No bins are removed
        and the input arrays are unchanged. Out-of-range positions are rejected.

    Returns
    -------
    pandas.DataFrame
        One row per baseline/polarization, in plot/legend order. Identification
        columns begin with ``obs_day``, ``scan_start``, ``scan_end``, ``baseline``,
        and ``polarization``; the indexing ``scan_num`` column is last.
        Known scalar HH:MM:SS start/end metadata is required in ``result``.
        Each of ``real`` and ``imag`` has ``_std``, ``_p16``, ``_p50``, ``_p84``
        columns. Diagnostics are ``snr_median``, ``thermal_rms``,
        ``n_if_usable``, ``n_if_total``, ``mean_amp``, ``total_std``, and
        ``phase_std_rad``. Values retain full numeric precision;
        this function does not save. Baselines absent from loaded scan records
        are excluded when ``observed_baselines`` is present. Observed baselines
        with unavailable products retain NaN statistics. Legacy results without
        presence metadata retain all combinations without missing-baseline
        diagnostics. Std requires at least two usable IFs (ddof=1);
        percentiles use linear interpolation and require at least one IF.
        ``mean_amp`` is mean_c(abs(V_chan)) over usable IFs: coherent time
        averaging happens before taking the magnitude of each IF mean.
        ``total_std`` is hypot(real_std, imag_std). ``phase_std_rad`` is the
        circular std sqrt(-2 * log(abs(mean(exp(1j * angle(V_chan)))))); zero
        means have undefined phase, and fewer than two defined phases give NaN.

    Notes
    -----
    For each IF c, sigma_c = sqrt(sum_t(sigma_tc**2)) / n_c and
    SNR_c = abs(mean_t(V_tc)) / sigma_c, using the plotted mean's integrations.
    Every statistic uses the same IF subset: finite mean, finite positive
    uncertainties for all contributors, finite positive propagated uncertainty,
    and finite SNR, excluding any requested IFs. Thermal RMS is
    sqrt(mean_c(sigma_c**2)), a real/imaginary component noise reference,
    not the uncertainty of a further frequency mean.
    Assumes calibrated inverse-variance weights and independent thermal noise.
    Bandpass structure and phase slopes can also contribute to observed scatter.
    No SNR cuts, detrending, weighted averaging, or debiasing are applied.
    The matching total scatter/noise ratio is total_std / (sqrt(2) * thermal_rms),
    since thermal_rms describes one complex component.
    """
    import pandas as pd

    if not result.get("unaveraged", False):
        raise ValueError("summarize_scan_bandpass requires unaveraged=True")
    allcoh, _, station_list = _coherency_inputs_from_result(result)
    station_list = np.asarray(station_list)
    if (
        allcoh.ndim != 6
        or allcoh.shape[2:] != (len(station_list), len(station_list), 2, 2)
    ):
        raise ValueError("allcoh must have shape (time, IF, station, station, 2, 2)")
    if allcoh.shape[1] < 2:
        raise ValueError("summarize_scan_bandpass requires multiple frequency bins (unaveraged IFs)")
    if "allsigma" not in result:
        raise ValueError("summarize_scan_bandpass requires uncertainties in result['allsigma']")
    if np.shape(result["allsigma"]) != allcoh.shape:
        raise ValueError("allsigma must have the same shape as allcoh")
    exclusions = {}
    for station, positions in (excluded_ifs_by_station or {}).items():
        indices = np.asarray(positions)
        if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError("Excluded IF positions must be one-dimensional integer sequences")
        if np.any(indices < 0) or np.any(indices >= allcoh.shape[1]):
            raise ValueError(f"Excluded IF positions for {station} are out of range")
        exclusions[station] = indices.astype(int)
    scan_start, scan_end = _scan_times_from_result(result)

    stat_columns = [
        f"{part}_{stat}" for part in ("real", "imag")
        for stat in ("std", "p16", "p50", "p84")
    ]
    new_columns = ["mean_amp", "total_std", "phase_std_rad"]
    columns = ["obs_day", "scan_start", "scan_end", "baseline", "polarization", *stat_columns,
               "snr_median", "thermal_rms", "n_if_usable", "n_if_total", *new_columns, "scan_num"]
    rows = []
    baselines = _baselines_from_result(
        result, include_autocorr, print_missing_baselines=print_missing_baselines, obs_day=obs_day,
    )
    for baseline in baselines:
        s1, s2 = baseline.split("-")
        i = np.where(station_list == s1)[0][0]
        j = np.where(station_list == s2)[0][0]
        for pol, (p, q) in get_pol_labels_for_baseline(baseline, alma_station).items():
            V = allcoh[:, :, i, j, p, q]
            V = np.where(V == 0, np.nan + 1j * np.nan, V)
            V_chan = np.full(V.shape[1], np.nan + 1j * np.nan)
            present = np.any(~np.isnan(V), axis=0)
            V_chan[present] = np.nanmean(V[:, present], axis=0)
            for station in (s1, s2):
                V_chan[exclusions.get(station, [])] = np.nan + 1j * np.nan
            sigma, snr = _bandpass_uncertainty(
                V, V_chan, result["allsigma"][:, :, i, j, p, q],
            )
            usable = np.isfinite(V_chan) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(snr)
            n = int(np.sum(usable))
            row = dict.fromkeys(stat_columns + ["snr_median", "thermal_rms", *new_columns], np.nan)
            row.update(
                obs_day=obs_day, scan_num=_scan_num_from_result(result, scan_num),
                scan_start=scan_start, scan_end=scan_end,
                baseline=baseline, polarization=pol, n_if_usable=n, n_if_total=V.shape[1],
            )
            if n:
                for part, values in (("real", V_chan[usable].real), ("imag", V_chan[usable].imag)):
                    row[f"{part}_std"] = np.std(values, ddof=1) if n > 1 else np.nan
                    row.update(zip(
                        [f"{part}_p{pct}" for pct in (16, 50, 84)],
                        np.percentile(values, [16, 50, 84]),
                    ))
                row["snr_median"] = np.median(snr[usable])
                row["thermal_rms"] = np.sqrt(np.mean(sigma[usable] ** 2))
                row["mean_amp"] = np.mean(np.abs(V_chan[usable]))
                row["total_std"] = np.hypot(row["real_std"], row["imag_std"])
                phase_values = V_chan[usable & (np.abs(V_chan) > 0)]
                if len(phase_values) > 1:
                    resultant = np.clip(np.abs(np.mean(np.exp(1j * np.angle(phase_values)))), 0, 1)
                    with np.errstate(divide="ignore"):
                        row["phase_std_rad"] = np.sqrt(-2 * np.log(resultant))
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def plot_scan_bandpass_all_baselines(
    result,
    var="phase",
    scan_num=None,
    average_over_time=True,
    unwrap_phase=False,
    alma_station="AA",
    include_autocorr=False,
    figsize_per_panel=(4.2, 3.0),
    amp_scale="linear",
    source="M87",
    figdir=None,
    savefig=False,
    *,
    show_snr=False,
    obs_day=None,
    print_missing_baselines=False,
):
    """
    Plot phase or amplitude vs channel for all baselines in one scan.

    This is intended for raw scan bandpass inspection. Each subplot is one
    baseline and contains all four polarization products.

    Baselines absent from loaded scan records are excluded when presence metadata
    is available; legacy results retain all station combinations. Keyword-only
    ``print_missing_baselines=True`` lists missing pairs once with the NX interval
    and optional day label. Default False; filtering always applies.

    Requires scalar ``scan_start`` / ``scan_end`` HH:MM:SS metadata in ``result``.
    ``obs_day`` is an optional caller-supplied day label for the title and filename.
    The starting time identifies the scan; ``scan_num`` is retained for call
    compatibility but is not displayed.

    ``show_snr=True`` requires declared unaveraged input, ``allsigma``, multiple
    frequency bins, and ``average_over_time=True``. Titles become e.g.
    ``AX-GL: 11, 9.5, 1.5, 1.2``: median IF thermal SNR for each polarization
    in plotted label/legend order. Unavailable values are ``N/A``. For the
    existing arithmetic complex mean, sigma_mean = sqrt(sum(sigma**2)) / N;
    SNR = abs(mean(V)) / sigma_mean. IFs with a contributing integration that
    lacks a finite positive uncertainty are excluded from the SNR summary.
    Assumes inverse-variance weights and independent integrations; no SNR
    filtering, debiasing, or change to plotted averaging is performed.
    """
    scan_start, _ = _scan_times_from_result(result)
    var_key, quantity_name, ylabel, file_token = _visibility_var_info(var)
    if figdir is None:
        figdir = f"{file_token}_channel"
    allcoh, channel_list, station_list = _coherency_inputs_from_result(result)
    if show_snr:
        if not average_over_time or not result.get("unaveraged", False):
            raise ValueError("show_snr requires unaveraged=True and average_over_time=True")
        if "allsigma" not in result:
            raise ValueError("show_snr requires uncertainties in result['allsigma']")
        if allcoh.shape[1] < 2:
            raise ValueError("show_snr requires multiple frequency bins (unaveraged IFs)")
        if np.shape(result["allsigma"]) != allcoh.shape:
            raise ValueError("allsigma must have the same shape as allcoh")
    station_list = np.asarray(station_list)
    baselines = _baselines_from_result(
        result, include_autocorr, print_missing_baselines=print_missing_baselines, obs_day=obs_day,
    )
    nbase = len(baselines)
    if nbase == 0:
        raise ValueError("No baselines found.")
    nrows, ncols = get_subplot_grid(nbase)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for ibl, baseline in enumerate(baselines):
        ax = axs.flat[ibl]
        s1, s2 = baseline.split("-")
        i = np.where(station_list == s1)[0][0]
        j = np.where(station_list == s2)[0][0]
        pol_map = get_pol_labels_for_baseline(
            baseline,
            alma_station=alma_station,
        )
        snr_values = []
        for pol_label, (p, q) in pol_map.items():
            V = allcoh[:, :, i, j, p, q]
            V = np.where(V == 0, np.nan + 1j * np.nan, V)
            if average_over_time:
                V_chan = np.nanmean(V, axis=0)
                if show_snr:
                    snr_values.append(_bandpass_snr(
                        V, V_chan, result["allsigma"][:, :, i, j, p, q]
                    ))
                y = _visibility_quantity(
                    V_chan,
                    var_key,
                    unwrap_phase=unwrap_phase,
                )
                ax.plot(
                    channel_list,
                    y,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=pol_label,
                )
            else:
                for it in range(V.shape[0]):
                    y = _visibility_quantity(
                        V[it, :],
                        var_key,
                        unwrap_phase=unwrap_phase,
                    )
                    label = pol_label if it == 0 else None
                    ax.plot(
                        channel_list,
                        y,
                        marker="o",
                        markersize=2,
                        linewidth=0.8,
                        alpha=0.25,
                        label=label,
                    )
        title = f"{baseline}: {', '.join(snr_values)}" if show_snr else baseline
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        _set_amp_scale(ax, var_key, amp_scale)
        if ibl == nbase - 1:
            handles, labels = ax.get_legend_handles_labels()
            ax.set_xlabel("Channel")
            ax.set_ylabel(ylabel)
            ax.legend(
                handles,
                labels,
                title="Pol",
                fontsize=8,
                title_fontsize=9,
                loc="best",
            )
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
    for k in range(nbase, nrows * ncols):
        axs.flat[k].axis("off")
    avg_text = "coherently averaged over time" if average_over_time else "no time averaging"
    unwrap_text = ", unwrapped phase" if var_key == "phase" and unwrap_phase else ""
    fig.suptitle(
        _scan_title(
            source,
            scan_start,
            quantity_name,
            "channel for all baselines",
            f"{avg_text}{unwrap_text}",
            obs_day=obs_day,
        ),
        fontsize=14,
    )
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = _scan_filename(
            source,
            scan_start,
            f"{file_token}_vs_channel_all_baselines",
            obs_day=obs_day,
        )
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs


def plot_result_vs_time_all_baselines(
    result,
    var="phase",
    channel=0,
    scan_num=None,
    unwrap_phase=False,
    alma_station="AA",
    include_autocorr=False,
    figsize_per_panel=(4.2, 3.0),
    amp_scale="linear",
    source="M87",
    figdir=None,
    savefig=False,
    *,
    obs_day=None,
    print_missing_baselines=False,
):
    """
    Plot phase or amplitude vs time for all baselines in one scan.

    The selected channel is read from a build_scan_coherency_matrix() result.
    Time is shown relative to scan start in minutes.
    Requires scalar ``scan_start`` / ``scan_end`` HH:MM:SS metadata in ``result``.
    ``obs_day`` is an optional caller-supplied label for the title and filename.
    ``scan_num`` is retained for call compatibility but is not displayed.
    Baselines absent from loaded scan records are excluded when presence metadata
    is available; legacy results retain all station combinations. Keyword-only
    ``print_missing_baselines=True`` lists missing pairs once with the NX interval
    and optional day label. Default False; filtering always applies.
    """
    scan_start, _ = _scan_times_from_result(result)
    var_key, quantity_name, ylabel, file_token = _visibility_var_info(var)
    if figdir is None:
        figdir = f"{file_token}_time"
    allcoh = result["allcoh"]
    t_unique = result["t_unique"]
    station_list = result["station_list"]
    channel_list = result["channel_list"]
    ichan = _channel_index(channel_list, channel, allcoh.shape[1])
    baselines = _baselines_from_result(
        result, include_autocorr, print_missing_baselines=print_missing_baselines, obs_day=obs_day,
    )
    nbase = len(baselines)
    if nbase == 0:
        raise ValueError("No baselines found.")
    nrows, ncols = get_subplot_grid(nbase)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    t_plot = (t_unique - np.nanmin(t_unique)) * 60.0
    for ibl, baseline in enumerate(baselines):
        ax = axs.flat[ibl]
        s1, s2 = baseline.split("-")
        i = np.where(station_list == s1)[0][0]
        j = np.where(station_list == s2)[0][0]
        pol_map = get_pol_labels_for_baseline(baseline, alma_station=alma_station)
        for pol_label, (p, q) in pol_map.items():
            V = allcoh[:, ichan, i, j, p, q]
            V = np.where(V == 0, np.nan + 1j * np.nan, V)
            y = _visibility_quantity(
                V,
                var_key,
                unwrap_phase=unwrap_phase,
            )
            ax.plot(
                t_plot,
                y,
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=pol_label,
            )
        ax.set_title(baseline, fontsize=10)
        ax.grid(alpha=0.3)
        _set_amp_scale(ax, var_key, amp_scale)
        if ibl == nbase - 1:
            ax.set_xlabel("Time in scan [min]")
            ax.set_ylabel(ylabel)
            ax.legend(title="Pol", fontsize=8, title_fontsize=9)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
    for k in range(nbase, nrows * ncols):
        axs.flat[k].axis("off")
    fig.suptitle(_scan_title(source, scan_start, quantity_name, "time", obs_day=obs_day), fontsize=14)
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = _scan_filename(
            source,
            scan_start,
            f"{file_token}_vs_time_all_baselines",
            obs_day=obs_day,
            prefix="avg_",
        )
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs


def plot_results_vs_scan_all_baselines(
    results,
    var="phase",
    channel=0,
    unwrap_phase=False,
    alma_station="AA",
    include_autocorr=False,
    figsize_per_panel=(4.2, 3.0),
    amp_scale="linear",
    source="M87",
    figdir="meta_plot",
    savefig=False,
    *,
    obs_day=None,
    print_missing_baselines=False,
):
    """
    Plot phase or amplitude vs scan using precomputed result dictionaries.

    For each scan, complex visibility is coherently averaged over time first.
    Results require ``scan_start`` / ``scan_end`` HH:MM:SS metadata. Titles show
    the full represented coverage; filenames use its earliest start and optional
    caller-supplied ``obs_day``. The x coordinates remain internal scan indices.
    Panels cover the union of observed baselines; absent pairs are skipped per
    scan. Legacy results without presence metadata retain all station combinations.
    Keyword-only ``print_missing_baselines=True`` lists missing pairs once per
    scan using that scan's station combinations, NX interval and optional day
    label. Default False; filtering always applies.
    """
    var_key, quantity_name, ylabel, file_token = _visibility_var_info(var)
    if len(results) == 0:
        raise ValueError("results is empty")
    intervals = [_scan_times_from_result(result) for result in results]
    scan_start = min((start for start, _ in intervals), key=_scan_time_seconds)
    scan_end = max((end for _, end in intervals), key=_scan_time_seconds)
    station_list_all = np.unique(
        np.concatenate([np.asarray(r["station_list"]) for r in results])
    )
    scan_baselines = [set(_baselines_from_result(
        result, include_autocorr, print_missing_baselines=print_missing_baselines, obs_day=obs_day,
    )) for result in results]
    observed = set().union(*scan_baselines)
    candidates = get_baselines_from_station_list(
        station_list_all, include_autocorr=include_autocorr,
    )
    legacy = all("observed_baselines" not in result for result in results)
    baselines = candidates if legacy else [baseline for baseline in candidates if baseline in observed]
    nbase = len(baselines)
    if nbase == 0:
        raise ValueError("No baselines found.")
    nrows, ncols = get_subplot_grid(nbase)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    out = {}
    for ibl, baseline in enumerate(baselines):
        ax = axs.flat[ibl]
        s1, s2 = baseline.split("-")
        pol_map = get_pol_labels_for_baseline(baseline, alma_station=alma_station)
        out[baseline] = {pol: {"scan": [], var_key: []} for pol in pol_map}
        for pol_label, (p, q) in pol_map.items():
            scan_nums = []
            values = []
            for r, present_baselines in zip(results, scan_baselines):
                if baseline not in present_baselines:
                    continue
                station_list = np.asarray(r["station_list"])
                if s1 not in station_list or s2 not in station_list:
                    continue
                i = np.where(station_list == s1)[0][0]
                j = np.where(station_list == s2)[0][0]
                ichan = _channel_index(r["channel_list"], channel, r["allcoh"].shape[1])
                V = r["allcoh"][:, ichan, i, j, p, q]
                V = np.where(V == 0, np.nan + 1j * np.nan, V)
                Vavg = np.nanmean(V)
                if not np.isfinite(Vavg):
                    continue
                scan_nums.append(r.get("scan_number", np.nan))
                values.append(_visibility_quantity(Vavg, var_key))
            scan_nums = np.asarray(scan_nums)
            values = np.asarray(values)
            order = np.argsort(scan_nums)
            scan_nums = scan_nums[order]
            values = values[order]
            if var_key == "phase" and unwrap_phase:
                values = np.rad2deg(np.unwrap(np.deg2rad(values)))
            ax.plot(
                scan_nums,
                values,
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=pol_label,
            )
            out[baseline][pol_label]["scan"] = scan_nums
            out[baseline][pol_label][var_key] = values
        ax.set_title(baseline, fontsize=10)
        ax.grid(alpha=0.3)
        _set_amp_scale(ax, var_key, amp_scale)
        if ibl == nbase - 1:
            ax.set_xlabel("Scan index")
            ax.set_ylabel(ylabel)
            ax.legend(title="Pol", fontsize=8, title_fontsize=9)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
    for k in range(nbase, nrows * ncols):
        axs.flat[k].axis("off")
    fig.suptitle(
        _scan_title(source, scan_start, quantity_name, "scan index", obs_day=obs_day, end=scan_end),
        fontsize=14,
    )
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = _scan_filename(
            source,
            scan_start,
            f"{file_token}_vs_scan_all_baselines",
            prefix="avg_",
            obs_day=obs_day,
        )
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs, out


def _uvdist_amp_points(uvdist, vis):
    """
    Return flattened ``(uvdist, abs(vis))`` points for a single plot call.

    ``vis`` may be ``(Nrecord,)`` or ``(Nrecord, Nchan)``. For channelized
    input, each record's uv distance is repeated for every channel so all
    channels are drawn as one artist (one legend entry) instead of one line
    per channel column. The plotted points are unchanged.
    """
    amp = np.abs(np.asarray(vis))
    uvdist = np.asarray(uvdist)
    if amp.ndim > 1:
        uvdist = np.broadcast_to(
            uvdist.reshape((-1,) + (1,) * (amp.ndim - 1)), amp.shape
        )
    return uvdist.ravel(), amp.ravel()


def plot_amp_uvdist_whole_dataset(
    u,
    v,
    rr,
    rl,
    lr,
    ll,
    source="M87",
    figdir="meta_plot",
    savefig=False,
    show=True,
    *,
    obs_day=None,
):
    """
    Plot visibility amplitude against uv distance for all four products.
    ``obs_day`` is an optional caller-supplied label for the title and filename.

    Parameters
    ----------
    u, v
        UV coordinates, shape ``(Nrecord,)``, in wavelengths.
    rr, rl, lr, ll
        Complex visibility arrays, shape ``(Nrecord,)`` or
        ``(Nrecord, Nchan)``. Polarization order is ``RR, RL, LR, LL``.

    Returns
    -------
    tuple
        ``(fig, axs)`` matplotlib objects. Amplitudes are ``abs(V)`` and are
        not averaged.
    """
    uvdist = np.sqrt(u**2 + v**2)/1e9
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{_observation_label(source, obs_day)}: visibility amplitude vs uv distance")
    axs[0, 0].plot(*_uvdist_amp_points(uvdist, rr), 'b.', label='RR')
    axs[0, 1].plot(*_uvdist_amp_points(uvdist, ll), 'r.', label='LL')
    axs[1, 0].plot(*_uvdist_amp_points(uvdist, lr), 'g.', label='LR')
    axs[1, 1].plot(*_uvdist_amp_points(uvdist, rl), 'm.', label='RL')
    for ax in axs.flat:
        ax.set_xlabel('uv distance [Gλ]')
        ax.set_ylabel('Visibility Amplitude')
        ax.legend()
        ax.grid(True)
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = _scan_filename(source, None, "amp_uvdist", obs_day=obs_day)
        fig.savefig(os.path.join(figdir, fname), dpi=300)
    if show:
        plt.show()
    return fig, axs


# ====== Visibility Table Helpers ======


def visibility_arrays_to_dataframe(
    times,
    t1,
    t2,
    u,
    v,
    rr,
    rl,
    lr,
    ll,
    scan_ids=None,
):
    """
    Convert visibility arrays into a long pandas DataFrame.

    Parameters
    ----------
    times, t1, t2, u, v
        Record-level arrays, each shape ``(Nrecord,)``.
    rr, rl, lr, ll
        Complex visibility arrays, shape ``(Nrecord,)`` or
        ``(Nrecord, Nchan)``. Polarization order is ``RR, RL, LR, LL``.
    scan_ids
        Optional scan IDs, shape ``(Nrecord,)``.

    Returns
    -------
    pandas.DataFrame
        Long table with one row per record/channel/polarization. ``phase`` is
        in radians and ``amp`` is ``abs(visibility)``. No averaging is applied.
    """
    import pandas as pd

    times = np.asarray(times)
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    u = np.asarray(u)
    v = np.asarray(v)
    pol_arrays = {
        "RR": np.asarray(rr),
        "RL": np.asarray(rl),
        "LR": np.asarray(lr),
        "LL": np.asarray(ll),
    }
    if scan_ids is not None:
        scan_ids = np.asarray(scan_ids)
    rows = []
    for pol, vis in pol_arrays.items():
        if vis.ndim == 1:
            vis = vis[:, None]
        for channel in range(vis.shape[1]):
            rows.append(
                pd.DataFrame(
                    {
                        "time": times,
                        "scan_id": scan_ids if scan_ids is not None else np.nan,
                        "t1": t1,
                        "t2": t2,
                        "baseline": [f"{a}-{b}" for a, b in zip(t1, t2)],
                        "u": u,
                        "v": v,
                        "channel": channel,
                        "polarization": pol,
                        "visibility": vis[:, channel],
                        "amp": np.abs(vis[:, channel]),
                        "phase": np.angle(vis[:, channel]),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def visibility_to_phase(vis, missing_zero_is_nan=True):
    """
    Convert complex visibility to phase in radians.

    Missing data are represented as NaN.

    Parameters
    ----------
    vis
        Complex visibility array of any shape.

    Returns
    -------
    numpy.ndarray
        Phase array in radians, same shape as ``vis``.
    """
    vis = np.asarray(vis)
    amp = np.abs(vis)
    phase = np.angle(vis).astype(float)
    bad = ~np.isfinite(phase)
    if missing_zero_is_nan:
        bad |= amp <= 0.0
    phase[bad] = np.nan
    return phase


def visibility_to_logamp(vis, missing_zero_is_nan=True):
    """
    Convert complex visibility to log amplitude.

    Missing data are represented as NaN.

    Returns
    -------
    numpy.ndarray
        ``log(abs(vis))`` with same shape as ``vis``.
    """
    vis = np.asarray(vis)
    amp = np.abs(vis)
    with np.errstate(divide="ignore", invalid="ignore"):
        logamp = np.log(amp).astype(float)
    bad = ~np.isfinite(logamp)
    if missing_zero_is_nan:
        bad |= amp <= 0.0
    logamp[bad] = np.nan
    return logamp


# ====== Closure Products ======


def build_closure_products_from_coherency(
    scan_result,
    exclude_stations=("AA", "ALMA"),
    parallel_pol_indices=((0, 0), (1, 1)),
    parallel_pol_labels=("RR", "LL"),
    missing_zero_is_nan=True,
):
    """
    Build closure phase and closure amplitude products from the output of
    build_scan_coherency_matrix().

    This function is designed for first-pass EHT / VLBI data inspection.

    It assumes the input coherency matrix has shape:

        allcoh[time, channel, station_i, station_j, pol_i, pol_j]

    where the polarization block is approximately:

        [[RR, RL],
         [LR, LL]]

    For mixed-pol ALMA data, ALMA should generally be excluded before computing
    ordinary RR/LL closure quantities. By default, this function excludes
    stations named "AA" and "ALMA".

    Parameters
    ----------
    scan_result : dict
        Output from build_scan_coherency_matrix().

        Required keys:

            "allcoh"
            "station_list"
            "t_unique"
            "channel_list"

    exclude_stations : tuple/list/set, optional
        Station names to exclude from closure quantities.

        Default:

            ("AA", "ALMA")

    parallel_pol_indices : tuple, optional
        Polarization products to use.

        Default:

            ((0, 0), (1, 1))

        corresponding to RR and LL.

    parallel_pol_labels : tuple, optional
        Labels for the selected polarization products.

        Default:

            ("RR", "LL")

        Therefore the output polarization axis is:

            idx = 0 -> RR
            idx = 1 -> LL

    missing_zero_is_nan : bool, optional
        If True, complex visibility values with amplitude == 0 are treated as
        missing data and converted to NaN in phase/log-amplitude outputs.

        This is useful because build_scan_coherency_matrix() often uses
        fill_missing = 0 + 0j.

    Returns
    -------
    result : dict

        Main triangle outputs
        ---------------------
        "triangle_baseline_phase" :
            ndarray, shape (Nt, Nc, Ntri, 3, Npol)

            For triangle (i, j, k), the three directed baselines are:

                0 -> V_ij
                1 -> V_jk
                2 -> V_ki

            Therefore:

                triangle_baseline_phase[it, ic, itri, 0, ipol]
                triangle_baseline_phase[it, ic, itri, 1, ipol]
                triangle_baseline_phase[it, ic, itri, 2, ipol]

        "closure_phase" :
            ndarray, shape (Nt, Nc, Ntri, Npol)

            Closure phase:

                arg(V_ij) + arg(V_jk) + arg(V_ki)

            wrapped to [-pi, pi].

        Main quadrangle outputs
        -----------------------
        "quadrangle_baseline_logamp" :
            ndarray, shape (Nt, Nc, Nquad, 4, Npol)

            For quadrangle (i, j, k, l), using convention:

                CA_ijkl = |V_ij| |V_kl| / (|V_ik| |V_jl|)

            the four baselines are:

                0 -> log |V_ij|
                1 -> log |V_kl|
                2 -> log |V_ik|
                3 -> log |V_jl|

        "closure_logamp" :
            ndarray, shape (Nt, Nc, Nquad, Npol)

            Closure log-amplitude:

                log CA_ijkl =
                    log |V_ij| + log |V_kl|
                    - log |V_ik| - log |V_jl|

        "closure_amp" :
            ndarray, shape (Nt, Nc, Nquad, Npol)

            Closure amplitude:

                CA_ijkl = exp(closure_logamp)

        Polarization convention
        -----------------------
        "pol_labels" :
            tuple, usually ("RR", "LL")

            pol index 0 -> RR
            pol index 1 -> LL
    """

    # ------------------------------------------------------------
    # Basic inputs
    # ------------------------------------------------------------
    allcoh = np.asarray(scan_result["allcoh"])
    station_list = [str(x) for x in scan_result["station_list"]]
    if allcoh.ndim != 6:
        raise ValueError(
            "scan_result['allcoh'] must have shape "
            "(Nt, Nc, Nstation, Nstation, 2, 2)."
        )
    Nt, Nc, Nstation_a, Nstation_b, Npol_a, Npol_b = allcoh.shape
    if Nstation_a != Nstation_b:
        raise ValueError("Station axes of allcoh must have the same length.")
    if len(station_list) != Nstation_a:
        raise ValueError(
            "len(scan_result['station_list']) does not match allcoh station axis."
        )
    if Npol_a < 2 or Npol_b < 2:
        raise ValueError(
            "Expected allcoh polarization block to be at least 2x2."
        )
    Nstation = len(station_list)
    parallel_pol_indices = tuple(parallel_pol_indices)
    parallel_pol_labels = tuple(parallel_pol_labels)
    if len(parallel_pol_indices) != len(parallel_pol_labels):
        raise ValueError(
            "parallel_pol_indices and parallel_pol_labels must have same length."
        )
    Npol = len(parallel_pol_indices)

    # ------------------------------------------------------------
    # Exclude ALMA / mixed-basis stations
    # ------------------------------------------------------------
    exclude_stations = set(exclude_stations)
    keep_station_mask = np.array(
        [st not in exclude_stations for st in station_list],
        dtype=bool,
    )
    keep_station_indices = np.where(keep_station_mask)[0]
    keep_station_names = [station_list[i] for i in keep_station_indices]
    if len(keep_station_indices) < 3:
        raise ValueError(
            "Need at least 3 non-excluded stations to form closure phases."
        )

    # ------------------------------------------------------------
    # Triangle closure phase
    #
    # For triangle (i, j, k), use directed baselines:
    #
    #     V_ij, V_jk, V_ki
    #
    # Closure phase:
    #
    #     CP_ijk = arg(V_ij) + arg(V_jk) + arg(V_ki)
    #
    # Output:
    #
    #     triangle_baseline_phase.shape = (Nt, Nc, Ntri, 3, Npol)
    #     closure_phase.shape           = (Nt, Nc, Ntri, Npol)
    # ------------------------------------------------------------
    triangles = list(itertools.combinations(keep_station_indices, 3))
    triangle_names = [
        tuple(station_list[i] for i in tri)
        for tri in triangles
    ]
    triangle_baselines = [
        ((i, j), (j, k), (k, i))
        for i, j, k in triangles
    ]
    triangle_baseline_names = [
        tuple(
            (station_list[a], station_list[b])
            for a, b in tri_bls
        )
        for tri_bls in triangle_baselines
    ]
    Ntri = len(triangles)
    triangle_baseline_phase = np.full(
        (Nt, Nc, Ntri, 3, Npol),
        np.nan,
        dtype=float,
    )
    closure_phase = np.full(
        (Nt, Nc, Ntri, Npol),
        np.nan,
        dtype=float,
    )
    for itri, tri_bls in enumerate(triangle_baselines):
        for ibl, (a, b) in enumerate(tri_bls):
            for ipol, (p0, p1) in enumerate(parallel_pol_indices):
                vis = allcoh[:, :, a, b, p0, p1]
                triangle_baseline_phase[:, :, itri, ibl, ipol] = (
                    visibility_to_phase(
                        vis,
                        missing_zero_is_nan=missing_zero_is_nan,
                    )
                )
        # Sum three directed phases.
        cp = np.sum(
            triangle_baseline_phase[:, :, itri, :, :],
            axis=2,
        )
        cp = wrap_phase(cp, rad=True, return_degrees=False)
        # If any of the three baselines is missing, closure phase is missing.
        missing = np.any(
            np.isnan(triangle_baseline_phase[:, :, itri, :, :]),
            axis=2,
        )
        cp[missing] = np.nan
        closure_phase[:, :, itri, :] = cp

    # ------------------------------------------------------------
    # Quadrangle closure amplitude
    #
    # For quadrangle (i, j, k, l), use one convention:
    #
    #     CA_ijkl = |V_ij| |V_kl| / (|V_ik| |V_jl|)
    #
    # log form:
    #
    #     log CA_ijkl =
    #         log |V_ij| + log |V_kl|
    #       - log |V_ik| - log |V_jl|
    #
    # Output:
    #
    #     quadrangle_baseline_logamp.shape = (Nt, Nc, Nquad, 4, Npol)
    #     closure_logamp.shape             = (Nt, Nc, Nquad, Npol)
    #     closure_amp.shape                = (Nt, Nc, Nquad, Npol)
    # ------------------------------------------------------------
    quadrangles = list(itertools.combinations(keep_station_indices, 4))
    quadrangle_names = [
        tuple(station_list[i] for i in q)
        for q in quadrangles
    ]
    quad_relations = []
    quad_relation_names = []
    quad_relation_labels = []
    for i, j, k, l in quadrangles:
        # Fixed convention:
        #
        #     CA_ijkl = |V_ij| |V_kl| / (|V_ik| |V_jl|)
        #
        rel = ((i, j), (k, l), (i, k), (j, l))
        quad_relations.append(rel)
        names = tuple(
            (station_list[a], station_list[b])
            for a, b in rel
        )
        quad_relation_names.append(names)
        num1, num2, den1, den2 = names
        quad_relation_labels.append(
            f"{num1[0]}{num1[1]} * {num2[0]}{num2[1]} / "
            f"({den1[0]}{den1[1]} * {den2[0]}{den2[1]})"
        )
    Nquad = len(quadrangles)
    quadrangle_baseline_logamp = np.full(
        (Nt, Nc, Nquad, 4, Npol),
        np.nan,
        dtype=float,
    )
    closure_logamp = np.full(
        (Nt, Nc, Nquad, Npol),
        np.nan,
        dtype=float,
    )
    closure_amp = np.full(
        (Nt, Nc, Nquad, Npol),
        np.nan,
        dtype=float,
    )
    for iq, rel in enumerate(quad_relations):
        for ibl, (a, b) in enumerate(rel):
            for ipol, (p0, p1) in enumerate(parallel_pol_indices):
                vis = allcoh[:, :, a, b, p0, p1]
                quadrangle_baseline_logamp[:, :, iq, ibl, ipol] = (
                    visibility_to_logamp(
                        vis,
                        missing_zero_is_nan=missing_zero_is_nan,
                    )
                )
        log_ca = (
            quadrangle_baseline_logamp[:, :, iq, 0, :]
            + quadrangle_baseline_logamp[:, :, iq, 1, :]
            - quadrangle_baseline_logamp[:, :, iq, 2, :]
            - quadrangle_baseline_logamp[:, :, iq, 3, :]
        )
        # If any of the four baselines is missing, closure amplitude is missing.
        missing = np.any(
            np.isnan(quadrangle_baseline_logamp[:, :, iq, :, :]),
            axis=2,
        )
        log_ca[missing] = np.nan
        closure_logamp[:, :, iq, :] = log_ca
        closure_amp[:, :, iq, :] = np.exp(log_ca)

    # ------------------------------------------------------------
    # Return everything
    # ------------------------------------------------------------
    return {
        # Metadata
        "scan_number": scan_result.get("scan_number", None),
        "scan_start": scan_result.get("scan_start", "unknown"),
        "scan_end": scan_result.get("scan_end", "unknown"),
        "station_list": station_list,
        "t_unique": scan_result["t_unique"],
        "channel_list": scan_result["channel_list"],
        # Station selection
        "excluded_stations": tuple(exclude_stations),
        "included_station_indices": keep_station_indices,
        "included_station_names": keep_station_names,
        # Polarization convention
        "pol_indices": parallel_pol_indices,
        "pol_labels": parallel_pol_labels,
        # Triangle geometry
        "triangles": triangles,
        "triangle_names": triangle_names,
        "triangle_baselines": triangle_baselines,
        "triangle_baseline_names": triangle_baseline_names,
        # Triangle data
        "triangle_baseline_phase": triangle_baseline_phase,
        "closure_phase": closure_phase,
        # Quadrangle geometry
        "quadrangles": quadrangles,
        "quadrangle_names": quadrangle_names,
        "quad_relations": quad_relations,
        "quad_relation_names": quad_relation_names,
        "quad_relation_labels": quad_relation_labels,
        # Quadrangle data
        "quadrangle_baseline_logamp": quadrangle_baseline_logamp,
        "closure_logamp": closure_logamp,
        "closure_amp": closure_amp,
    }


def plot_closure_phase_vs_time_all_triangles(
    closure,
    average_over_channel=True,
    channel=0,
    use_relative_time=True,
    figsize_per_panel=(4.2, 3.0),
    source="M87",
    figdir="closure_phase_time",
    savefig=False,
    *,
    obs_day=None,
):
    """
    Plot closure phase vs time for all triangles and both parallel-hand pols.

    Each subplot corresponds to one triangle and one polarization.
    Requires scalar ``scan_start`` / ``scan_end`` HH:MM:SS metadata, carried
    from the scan result. ``obs_day`` labels the title and saved filename.

    Expected input
    --------------
    closure : dict
        Output from build_closure_products_from_coherency().

        Required keys:

            closure["closure_phase"]
            closure["t_unique"]
            closure["channel_list"]
            closure["triangle_names"]
            closure["pol_labels"]

        closure["closure_phase"] has shape:

            (Nt, Nc, Ntri, Npol)

        where usually:

            pol index 0 -> RR
            pol index 1 -> LL

    Parameters
    ----------
    average_over_channel : bool
        If True, circularly average closure phase over channel first.

        If False, use one selected channel.

        Should be True for raw data; for averaged data this doesn't matter.

    channel : int
        Channel label or channel index used when average_over_channel=False.

    use_relative_time : bool
        If True, plot time relative to scan start in minutes.
        If False, plot raw closure["t_unique"].

    figsize_per_panel : tuple
        Size per subplot panel.

    source : str
        Source name used in title / filename.

    figdir : str
        Directory for saving figure.

    savefig : bool
        If True, save figure.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes.
    """
    scan_start, _ = _scan_times_from_result(closure)
    cp = np.asarray(closure["closure_phase"])  # (Nt, Nc, Ntri, Npol)
    t_unique = np.asarray(closure["t_unique"])
    channel_list = np.asarray(closure["channel_list"])
    triangle_names = closure["triangle_names"]
    pol_labels = closure["pol_labels"]
    if cp.ndim != 4:
        raise ValueError(
            "closure['closure_phase'] must have shape (Nt, Nc, Ntri, Npol)."
        )
    Nt, Nc, Ntri, Npol = cp.shape
    if len(pol_labels) != Npol:
        raise ValueError("len(closure['pol_labels']) does not match Npol.")

    # ------------------------------------------------------------
    # Choose data to plot: shape becomes (Nt, Ntri, Npol)
    # ------------------------------------------------------------
    if average_over_channel:
        # Circular average over channel.
        cp_plot = np.angle(
            np.nanmean(np.exp(1j * cp), axis=1)
        )
        channel_label = "channel-averaged"
    else:
        if channel in channel_list:
            ichan = np.where(channel_list == channel)[0][0]
            channel_label = channel
        else:
            ichan = int(channel)
            channel_label = channel
        if ichan < 0 or ichan >= Nc:
            raise IndexError(f"channel index {ichan} out of range for Nc={Nc}")
        cp_plot = cp[:, ichan, :, :]  # (Nt, Ntri, Npol)

    # ------------------------------------------------------------
    # Time axis
    # ------------------------------------------------------------
    if use_relative_time:
        t_plot = (t_unique - np.nanmin(t_unique)) * 60.0
        xlabel = "Time in scan [min]"
    else:
        t_plot = t_unique
        xlabel = "Time"

    # ------------------------------------------------------------
    # Subplots: one subplot per triangle-pol pair
    # ------------------------------------------------------------
    nplots = Ntri * Npol
    nrows, ncols = get_subplot_grid(nplots)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    pol_color = {
        "RR": "blue",
        "LL": "red",
    }
    iplot = 0
    for itri in range(Ntri):
        tri_name = "-".join(triangle_names[itri])
        for ipol in range(Npol):
            ax = axs.flat[iplot]
            pol_label = pol_labels[ipol]
            color = pol_color.get(pol_label, None)
            y = np.degrees(cp_plot[:, itri, ipol])
            ax.plot(
                t_plot,
                y,
                "o-",
                color=color,
                markersize=3,
                linewidth=1.1,
                label=pol_label,
            )
            ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{tri_name} {pol_label}", fontsize=9)
            ax.grid(alpha=0.3)
            if iplot == nplots - 1:
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Closure phase [deg]")
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
            iplot += 1
    for k in range(nplots, nrows * ncols):
        axs.flat[k].axis("off")
    fig.suptitle(
        _scan_title(source, scan_start, "closure phase", "time", channel_label, obs_day=obs_day),
        fontsize=14,
    )
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = _scan_filename(
            source,
            scan_start,
            "closure_phase_vs_time_all_triangles",
            obs_day=obs_day,
        )
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs


def plot_closure_amp_vs_time_all_quadrangles(
    closure,
    average_over_channel=True,
    channel=0,
    use_logamp=True,
    use_relative_time=True,
    figsize_per_panel=(4.2, 3.0),
    source="M87",
    figdir="closure_amp_time",
    savefig=False,
    *,
    obs_day=None,
):
    """
    Plot closure amplitude vs time for all quadrangles and both parallel-hand pols.

    Each subplot corresponds to one quadrangle and one polarization.
    Requires scalar ``scan_start`` / ``scan_end`` HH:MM:SS metadata, carried
    from the scan result. ``obs_day`` labels the title and saved filename.

    Expected input
    --------------
    closure : dict
        Output from build_closure_products_from_coherency().

        Required keys:

            closure["closure_amp"]
            closure["closure_logamp"]
            closure["t_unique"]
            closure["channel_list"]
            closure["quadrangle_names"]
            closure["quad_relation_labels"]
            closure["pol_labels"]

        closure["closure_amp"] has shape:

            (Nt, Nc, Nquad, Npol)

        where usually:

            pol index 0 -> RR
            pol index 1 -> LL

    Parameters
    ----------
    average_over_channel : bool
        If True, average closure amplitude over channel first.

        Should be True for raw data; for averaged data this doesn't matter.

        If use_logamp=True, averages log closure amplitude over channel.
        This is usually preferred.

    channel : int
        Channel label or channel index used when average_over_channel=False.

    use_logamp : bool
        If True, plot log closure amplitude.
        If False, plot raw closure amplitude.

        For diagnostics, log closure amplitude is usually better.

    use_relative_time : bool
        If True, plot time relative to scan start in minutes.
        If False, plot raw closure["t_unique"].

    figsize_per_panel : tuple
        Size per subplot panel.

    source : str
        Source name used in title / filename.

    figdir : str
        Directory for saving figure.

    savefig : bool
        If True, save figure.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes.
    """
    scan_start, _ = _scan_times_from_result(closure)
    ca = np.asarray(closure["closure_amp"])
    logca = np.asarray(closure["closure_logamp"])
    t_unique = np.asarray(closure["t_unique"])
    channel_list = np.asarray(closure["channel_list"])
    pol_labels = closure["pol_labels"]
    if ca.ndim != 4:
        raise ValueError(
            "closure['closure_amp'] must have shape (Nt, Nc, Nquad, Npol)."
        )
    if logca.shape != ca.shape:
        raise ValueError(
            "closure['closure_logamp'] must have the same shape as closure['closure_amp']."
        )
    Nt, Nc, Nquad, Npol = ca.shape
    if len(pol_labels) != Npol:
        raise ValueError("len(closure['pol_labels']) does not match Npol.")

    # ------------------------------------------------------------
    # Choose data to plot: shape becomes (Nt, Nquad, Npol)
    # ------------------------------------------------------------
    if use_logamp:
        data = logca
        ylabel = "log closure amplitude"
        reference_value = 0.0
    else:
        data = ca
        ylabel = "Closure amplitude"
        reference_value = 1.0
    if average_over_channel:
        if use_logamp:
            # Arithmetic average in log space = geometric mean in amp space.
            y_plot = np.nanmean(data, axis=1)
        else:
            # Raw arithmetic average of closure amplitude.
            y_plot = np.nanmean(data, axis=1)
        channel_label = "channel-averaged"
    else:
        if channel in channel_list:
            ichan = np.where(channel_list == channel)[0][0]
            channel_label = channel
        else:
            ichan = int(channel)
            channel_label = channel
        if ichan < 0 or ichan >= Nc:
            raise IndexError(f"channel index {ichan} out of range for Nc={Nc}")
        y_plot = data[:, ichan, :, :]  # (Nt, Nquad, Npol)

    # ------------------------------------------------------------
    # Time axis
    # ------------------------------------------------------------
    if use_relative_time:
        t_plot = (t_unique - np.nanmin(t_unique)) * 60.0
        xlabel = "Time in scan [min]"
    else:
        t_plot = t_unique
        xlabel = "Time"

    # ------------------------------------------------------------
    # Subplots: one subplot per quadrangle-pol pair
    # ------------------------------------------------------------
    nplots = Nquad * Npol
    nrows, ncols = get_subplot_grid(nplots)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    pol_color = {
        "RR": "blue",
        "LL": "red",
    }
    iplot = 0
    for iq in range(Nquad):
        if "quad_relation_labels" in closure:
            quad_label = closure["quad_relation_labels"][iq]
        else:
            quad_label = "-".join(closure["quadrangle_names"][iq])
        for ipol in range(Npol):
            ax = axs.flat[iplot]
            pol_label = pol_labels[ipol]
            color = pol_color.get(pol_label, None)
            y = y_plot[:, iq, ipol]
            ax.plot(
                t_plot,
                y,
                "o-",
                color=color,
                markersize=3,
                linewidth=1.1,
                label=pol_label,
            )
            ax.axhline(
                reference_value,
                color="k",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )
            ax.set_title(f"{quad_label} {pol_label}", fontsize=8)
            ax.grid(alpha=0.3)
            if iplot == nplots - 1:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
            iplot += 1
    for k in range(nplots, nrows * ncols):
        axs.flat[k].axis("off")
    quantity_name = "log closure amplitude" if use_logamp else "closure amplitude"
    fig.suptitle(
        _scan_title(source, scan_start, quantity_name, "time", channel_label, obs_day=obs_day),
        fontsize=14,
    )
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = _scan_filename(
            source,
            scan_start,
            "closure_amp_vs_time_all_quadrangles",
            obs_day=obs_day,
        )
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs
