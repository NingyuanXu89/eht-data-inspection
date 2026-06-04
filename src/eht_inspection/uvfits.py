import os

import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import itertools

from .utils import scan_ids_from_intervals, wrap_phase


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
):
    """Load observation data from a uvfits file.

       Args:
           filename: path to either an input text file or an HDUList object
           polrep: load data as either 'stokes' or 'circ'
           flipbl: flip baseline phases if True.
           allow_singlepol: If True and polrep='stokes',
                            treat single-polarization data as Stokes I
           force_singlepol: 'R' or 'L' to load only 1 polarization and treat as Stokes I
           channel: list of channels to average in the import. channel=all averages all
           IF: list of IFs to  average in  the import. IF=all averages all
           remove_nan: whether or not to remove entries with nan data

           ignore_pzero_date: if True, ignore the offset parameters in DATE field
                              TODO: what is the correct behavior per AIPS memo 117?
           return_dict: If False, preserve the historical tuple return.
                        If True, return a dictionary keyed by array name
                        and include the UVFITS scan table as ``scantable``.
       Returns:
           ``(times, t1, t2, u, v, rr, rl, lr, ll, rrsigma, rlsigma,
           lrsigma, llsigma)`` by default. If ``return_dict=True``, return
           keys ``times``, ``t1``, ``t2``, ``u``, ``v``, ``rr``, ``rl``,
           ``lr``, ``ll``, ``rrsigma``, ``rlsigma``, ``lrsigma``,
           ``llsigma``, and ``scantable``. Visibility arrays have shape
           ``(Nrecord, Nchan)`` after IF/channel selection. Polarization
           order in the returned tuple is ``RR, RL, LR, LL``. Phases are
           not extracted here; returned visibilities are complex.
    """
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
    if not np.any(mask):
        raise Exception("No unflagged RR or LL data in uvfits file!")
    if np.any(~(rrmask * llmask)):
        print("Warning: removing flagged data present!")
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
    jds = jd1scal * data['DATE'][mask].astype('d') + jd1zero
    jds += jd2scal * data['_DATE'][mask].astype('d') + jd2zero
    mjd = int(np.min(jds) - 2400000.5)
    times = (jds - 2400000.5 - mjd) * 24.0
    try:
        scantable = []
        nxtable = hdulist['AIPS NX']
        for scan in nxtable.data:
            scan_start = scan['TIME']  # in days since reference date
            scan_dur = scan['TIME INTERVAL']
            startvis = scan['START VIS'] - 1
            endvis = scan['END VIS'] - 1
            scantable.append([scan_start - 0.5 * scan_dur,
                              scan_start + 0.5 * scan_dur])
        scantable = np.array(scantable) * 24
    except BaseException:
        print("No NX table in uvfits!")
        scantable = None
    # Integration times
    try:
        tints = data['INTTIM'][mask]
    except KeyError:
        tints = np.zeros(len(mask))
    # Sites - add names
    t1c = data['BASELINE'][mask].astype(int) // 256
    t2c = data['BASELINE'][mask].astype(int) - t1c * 256
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
        rr_2d[:, :, 0],
        rl_2d[:, :, 0],
        lr_2d[:, :, 0],
        ll_2d[:, :, 0],
        np.sqrt(1. / rrweight)[:, :, 0],
        np.sqrt(1. / rlweight)[:, :, 0],
        np.sqrt(1. / lrweight)[:, :, 0],
        np.sqrt(1. / llweight)[:, :, 0],
    )
    if return_dict:
        return _load_obs_uvfits_result_dict(*result, scantable)
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
):
    """
    Build a dense station-by-station coherency matrix for one scan.

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
    allu = np.full((Nstation, Nstation), np.nan, dtype=float)
    allv = np.full((Nstation, Nstation), np.nan, dtype=float)
    # Fill direct baselines row by row
    for row in range(len(t)):
        it = time_to_index[t[row]]
        i = station_to_index[s1[row]]
        j = station_to_index[s2[row]]
        allcoh[it, :, i, j, :, :] = coh[row]
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
                if flip_uv_reverse:
                    allu[j, i] = -mean_u
                    allv[j, i] = -mean_v
                else:
                    allu[j, i] = mean_u
                    allv[j, i] = mean_v
    return {
        "allcoh": allcoh,
        "allu": allu,
        "allv": allv,
        "station_list": station_list,
        "t_unique": t_unique,
        "channel_list": channel_list,
        "scan_mask": scan_mask,
        "scan_number": scannum,
    }


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
):
    """
    Load UVFITS data and build a scan coherency matrix.

    This is a convenience wrapper for the common workflow:

        load_obs_uvfits(..., return_dict=True)
        scan_ids_from_intervals(...)
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
        with ``scan_ids_from_intervals(times, scans)``.

    scans
        Scan intervals used when ``scan_ids`` is None. If omitted, the
        ``scantable`` loaded from the UVFITS NX table is used.

    start_index
        First scan ID to assign when ``scan_ids`` is None. The default keeps
        zero-based scan IDs. Use ``start_index=1`` for one-based scan numbers.
        Records outside any scan interval remain ``-1``.

    polrep, flipbl, allow_singlepol, force_singlepol, channel, IF, remove_nan,
    ignore_pzero_date, trial_speedups
        Passed through to ``load_obs_uvfits``.

    fill_missing, conjugate_reverse, flip_uv_reverse
        Passed through to ``build_scan_coherency_matrix``.

    Returns
    -------
        Result from ``build_scan_coherency_matrix``.
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
    )
    if scan_ids is None:
        scan_intervals = obs["scantable"] if scans is None else scans
        if scan_intervals is None:
            raise ValueError(
                "scan_ids could not be inferred because no scan intervals "
                "were provided and the UVFITS file has no scantable"
            )
        scan_ids = scan_ids_from_intervals(obs["times"], scan_intervals)
        if start_index != 0:
            scan_ids = np.asarray(scan_ids).copy()
            scan_ids[scan_ids >= 0] += start_index

    return build_scan_coherency_matrix(
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
    )


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


def get_baselines_from_station_list(
    station_list,
    include_autocorr=False,
):
    """
    Generate baseline names from a station list.

    Parameters
    ----------
    station_list : array-like
        Station names.

    include_autocorr : bool
        If False, only return i < j baselines.
        If True, include i == j autocorrelations.

    Returns
    -------
    baselines : list of str
        Baseline names like ["AA-LM", "AA-SM", "LM-SM"].
    """
    baselines = []
    n = len(station_list)
    for i in range(n):
        j_start = i if include_autocorr else i + 1
        for j in range(j_start, n):
            baselines.append(f"{station_list[i]}-{station_list[j]}")
    return baselines

import math


def get_subplot_grid(nplots):
    """
    Choose a reasonable subplot grid for nplots.

    Rules
    -----
    1. If nplots <= 4:
       put all plots in one row.

           nrows = 1
           ncols = nplots

    2. If nplots > 4:
       use a near-square layout.

       The number of columns is chosen as ceil(sqrt(nplots)).
       The number of rows is then ceil(nplots / ncols).

    Examples
    --------
    nplots = 3  -> 1 x 3
    nplots = 4  -> 1 x 4
    nplots = 5  -> 2 x 3
    nplots = 6  -> 2 x 3
    nplots = 7  -> 3 x 3
    nplots = 8  -> 3 x 3
    nplots = 9  -> 3 x 3
    nplots = 10 -> 3 x 4
    nplots = 12 -> 3 x 4
    nplots = 16 -> 4 x 4

    Parameters
    ----------
    nplots : int
        Number of subplots needed.

    Returns
    -------
    nrows, ncols : tuple of int
        Number of rows and columns.
    """
    if nplots < 1:
        raise ValueError("nplots must be >= 1")
    if nplots <= 4:
        return 1, nplots
    ncols = math.ceil(math.sqrt(nplots))
    nrows = math.ceil(nplots / ncols)
    return nrows, ncols


def plot_scan_phase_vs_channel_all_baselines(
    allcoh,
    channel_list,
    station_list,
    scan_num=None,
    average_over_time=True,
    unwrap_phase=False,
    alma_station="AA",
    include_autocorr=False,
    max_cols=3,
    figsize_per_panel=(4.2, 3.0),
    source="M87",
    figdir="phase_channel",
    savefig=False,
):
    """
    Plot phase vs channel for all baselines in one scan.

    Each subplot corresponds to one baseline. Each subplot contains the
    four polarization products. The polarization labels are changed for
    baselines involving ALMA / AA because ALMA uses linear polarization.

    Parameters
    ----------
    allcoh : ndarray
        Coherency array with shape:

            Nt, Nc, Nstation, Nstation, 2, 2

        Usually returned by build_scan_coherency_matrix(...).

    channel_list : ndarray
        Channel indices.

    station_list : ndarray
        Station names for this scan.

    scan_num : int or None
        Scan number used in the figure suptitle.

    average_over_time : bool
        If True, coherently average over time first:

            V_avg[channel] = mean_t V[t, channel]

        Then plot angle(V_avg).

        If False, plot one faint line per time sample for each polarization.

    unwrap_phase : bool
        If True, unwrap phase along the channel axis.

    alma_station : str
        Station code for ALMA. Default is "AA".

    include_autocorr : bool
        Whether to include autocorrelation subplots.

    max_cols : int
        Maximum number of subplot columns.

    figsize_per_panel : tuple
        Size per subplot panel.
    source: str
        Source name for the figure suptitle.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes.
    """
    station_list = np.asarray(station_list)
    baselines = get_baselines_from_station_list(
        station_list,
        include_autocorr=include_autocorr,
    )
    nbase = len(baselines)
    if nbase == 0:
        raise ValueError("No baselines found.")
    nrows, ncols = get_subplot_grid(nbase)
    fig_width = figsize_per_panel[0] * ncols
    fig_height = figsize_per_panel[1] * nrows
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    pol_handles = None
    pol_labels = None
    for ibl, baseline in enumerate(baselines):
        ax = axs.flat[ibl]
        s1, s2 = baseline.split("-")
        i = np.where(station_list == s1)[0][0]
        j = np.where(station_list == s2)[0][0]
        pol_map = get_pol_labels_for_baseline(
            baseline,
            alma_station=alma_station,
        )
        for pol_label, (p, q) in pol_map.items():
            V = allcoh[:, :, i, j, p, q]  # Nt x Nchannel
            # Treat zero-filled missing entries as invalid.
            V = np.where(V == 0, np.nan + 1j * np.nan, V)
            if average_over_time:
                # Coherent average over time, keep channel axis.
                V_chan = np.nanmean(V, axis=0)
                if unwrap_phase:
                    phase = np.unwrap(np.angle(V_chan)) * 180.0 / np.pi
                else:
                    phase = np.angle(V_chan, deg=True)
                ax.plot(
                    channel_list,
                    phase,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=pol_label,
                )
            else:
                # No time averaging: one faint line per time sample.
                for it in range(V.shape[0]):
                    if unwrap_phase:
                        phase = np.unwrap(np.angle(V[it, :])) * 180.0 / np.pi
                    else:
                        phase = np.angle(V[it, :], deg=True)
                    label = pol_label if it == 0 else None
                    ax.plot(
                        channel_list,
                        phase,
                        marker="o",
                        markersize=2,
                        linewidth=0.8,
                        alpha=0.25,
                        label=label,
                    )
        ax.set_title(baseline, fontsize=10)
        ax.grid(alpha=0.3)
        # Save legend handles/labels from the last real subplot only.
        if ibl == nbase - 1:
            pol_handles, pol_labels = ax.get_legend_handles_labels()
            ax.set_xlabel("Channel")
            ax.set_ylabel("Phase [deg]")
            ax.legend(
                pol_handles,
                pol_labels,
                title="Pol",
                fontsize=8,
                title_fontsize=9,
                loc="best",
            )
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
    # Hide unused axes.
    for k in range(nbase, nrows * ncols):
        axs.flat[k].axis("off")
    avg_text = "coherently averaged over time" if average_over_time else "no time averaging"
    unwrap_text = ", unwrapped phase" if unwrap_phase else ""
    if scan_num is None:
        suptitle = f"Phase vs channel ({avg_text}{unwrap_text})"
    else:
        suptitle = f"Scan {scan_num}: phase vs channel for all baselines ({avg_text}{unwrap_text})"
    fig.suptitle(suptitle, fontsize=14)
    # plt.tight_layout()
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = f"{source}_scan{scan_num}_phase_vs_channel_all_baselines.png"
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs


def plot_scan_amp_vs_channel_all_baselines(
    allcoh,
    channel_list,
    station_list,
    scan_num=None,
    average_over_time=True,
    alma_station="AA",
    include_autocorr=False,
    figsize_per_panel=(4.2, 3.0),
    amp_scale="linear",
    source="M87",
    figdir="amp_channel",
    savefig=False,
):
    """
    Plot amplitude vs channel for all baselines in one scan.

    Each subplot corresponds to one baseline. Each subplot contains the
    four polarization products. The polarization labels are changed for
    baselines involving ALMA / AA because ALMA uses linear polarization.

    Parameters
    ----------
    allcoh : ndarray
        Coherency array with shape:

            Nt, Nc, Nstation, Nstation, 2, 2

        Usually returned by build_scan_coherency_matrix(...).

    channel_list : ndarray
        Channel indices.

    station_list : ndarray
        Station names for this scan.

    scan_num : int or None
        Scan number used in the figure suptitle.

    average_over_time : bool
        If True, coherently average over time first:

            V_avg[channel] = mean_t V[t, channel]

        Then plot abs(V_avg).

        If False, plot one faint line per time sample for each polarization.

    alma_station : str
        Station code for ALMA. Default is "AA".

    include_autocorr : bool
        Whether to include autocorrelation subplots.

    figsize_per_panel : tuple
        Size per subplot panel.

    amp_scale : {"linear", "log"}
        Y-axis amplitude scale.
    source : str
        Source name for the figure suptitle.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes.
    """
    station_list = np.asarray(station_list)
    baselines = get_baselines_from_station_list(
        station_list,
        include_autocorr=include_autocorr,
    )
    nbase = len(baselines)
    if nbase == 0:
        raise ValueError("No baselines found.")
    nrows, ncols = get_subplot_grid(nbase)
    fig_width = figsize_per_panel[0] * ncols
    fig_height = figsize_per_panel[1] * nrows
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    for ibl, baseline in enumerate(baselines):
        ax = axs.flat[ibl]
        s1, s2 = baseline.split("-")
        if s1 not in station_list or s2 not in station_list:
            continue
        i = np.where(station_list == s1)[0][0]
        j = np.where(station_list == s2)[0][0]
        pol_map = get_pol_labels_for_baseline(
            baseline,
            alma_station=alma_station,
        )
        for pol_label, (p, q) in pol_map.items():
            V = allcoh[:, :, i, j, p, q]  # Nt x Nchannel
            # Treat zero-filled missing entries as invalid.
            V = np.where(V == 0, np.nan + 1j * np.nan, V)
            if average_over_time:
                # Coherent average over time, keep channel axis.
                V_chan = np.nanmean(V, axis=0)
                amp = np.abs(V_chan)
                ax.plot(
                    channel_list,
                    amp,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=pol_label,
                )
            else:
                # No time averaging: one faint line per time sample.
                for it in range(V.shape[0]):
                    amp = np.abs(V[it, :])
                    label = pol_label if it == 0 else None
                    ax.plot(
                        channel_list,
                        amp,
                        marker="o",
                        markersize=2,
                        linewidth=0.8,
                        alpha=0.25,
                        label=label,
                    )
        ax.set_title(baseline, fontsize=10)
        ax.grid(alpha=0.3)
        if amp_scale == "log":
            ax.set_yscale("log")
        elif amp_scale != "linear":
            raise ValueError("amp_scale must be either 'linear' or 'log'")
        # Only the last real subplot shows labels and legend.
        if ibl == nbase - 1:
            handles, labels = ax.get_legend_handles_labels()
            ax.set_xlabel("Channel")
            ax.set_ylabel("Amplitude")
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
    # Hide unused axes.
    for k in range(nbase, nrows * ncols):
        axs.flat[k].axis("off")
    avg_text = "coherently averaged over time" if average_over_time else "no time averaging"
    if scan_num is None:
        suptitle = f"Amplitude vs channel for all baselines ({avg_text})"
    else:
        suptitle = f"Scan {scan_num}: amplitude vs channel for all baselines ({avg_text})"
    fig.suptitle(suptitle, fontsize=14)
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = f"{source}_scan{scan_num}_amp_vs_channel_all_baselines.png"
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs

### Similar functions, but for channel-averaged and 10s-averaged data


def plot_result_phase_vs_time_all_baselines(
    result,
    channel=0,
    scan_num=None,
    unwrap_phase=False,
    alma_station="AA",
    include_autocorr=False,
    figsize_per_panel=(4.2, 3.0),
    source="M87",
    figdir="phase_time",
    savefig=False,
):
    """
    Plot phase vs time for all baselines in one scan.

    Input is directly the dict returned by build_scan_coherency_matrix().
    Each subplot = one baseline.
    Each subplot contains all 4 pol products.
    Phase is plotted in degrees. No channel averaging is performed.
    """
    allcoh = result["allcoh"]
    t_unique = result["t_unique"]
    station_list = result["station_list"]
    channel_list = result["channel_list"]
    # Convert selected channel label to array index
    if channel in channel_list:
        ichan = np.where(channel_list == channel)[0][0]
        channel_label = channel
    else:
        ichan = int(channel)
        channel_label = channel
    if ichan < 0 or ichan >= allcoh.shape[1]:
        raise IndexError(f"channel index {ichan} out of range for Nc={allcoh.shape[1]}")
    baselines = get_baselines_from_station_list(
        station_list,
        include_autocorr=include_autocorr,
    )
    nbase = len(baselines)
    nrows, ncols = get_subplot_grid(nbase)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    # use time relative to scan start, in minutes
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
            if unwrap_phase:
                phase = np.rad2deg(np.unwrap(np.angle(V)))
            else:
                phase = np.angle(V, deg=True)
            ax.plot(
                t_plot,
                phase,
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=pol_label,
            )
        ax.set_title(baseline, fontsize=10)
        ax.grid(alpha=0.3)
        if ibl == nbase - 1:
            ax.set_xlabel("Time in scan [min]")
            ax.set_ylabel("Phase [deg]")
            ax.legend(title="Pol", fontsize=8, title_fontsize=9)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
    for k in range(nbase, nrows * ncols):
        axs.flat[k].axis("off")
    if scan_num is None:
        scan_num = result.get("scan_number", None)
    title_scan = "" if scan_num is None else f", scan {scan_num}"
    fig.suptitle(
        f"{source}{title_scan}: phase vs time",
        fontsize=14,
    )
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        if scan_num is None:
            fname = f"avg_{source}_phase_vs_time_all_baselines.png"
        else:
            fname = f"avg_{source}_scan{scan_num}_phase_vs_time_all_baselines.png"
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs


def plot_result_amp_vs_time_all_baselines(
    result,
    channel=0,
    scan_num=None,
    alma_station="AA",
    include_autocorr=False,
    figsize_per_panel=(4.2, 3.0),
    amp_scale="linear",
    source="M87",
    figdir="amp_time",
    savefig=False,
):
    """
    Plot amplitude vs time for all baselines in one scan.

    Input is directly the dict returned by build_scan_coherency_matrix().
    Each subplot = one baseline.
    Each subplot contains all 4 pol products.
    """
    allcoh = result["allcoh"]
    t_unique = result["t_unique"]
    station_list = result["station_list"]
    channel_list = result["channel_list"]
    if channel in channel_list:
        ichan = np.where(channel_list == channel)[0][0]
        channel_label = channel
    else:
        ichan = int(channel)
        channel_label = channel
    if ichan < 0 or ichan >= allcoh.shape[1]:
        raise IndexError(f"channel index {ichan} out of range for Nc={allcoh.shape[1]}")
    baselines = get_baselines_from_station_list(
        station_list,
        include_autocorr=include_autocorr,
    )
    nbase = len(baselines)
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
            amp = np.abs(V)
            ax.plot(
                t_plot,
                amp,
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=pol_label,
            )
        ax.set_title(baseline, fontsize=10)
        ax.grid(alpha=0.3)
        if amp_scale == "log":
            ax.set_yscale("log")
        elif amp_scale != "linear":
            raise ValueError("amp_scale must be 'linear' or 'log'")
        if ibl == nbase - 1:
            ax.set_xlabel("Time in scan [min]")
            ax.set_ylabel("Amplitude")
            ax.legend(title="Pol", fontsize=8, title_fontsize=9)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
    for k in range(nbase, nrows * ncols):
        axs.flat[k].axis("off")
    if scan_num is None:
        scan_num = result.get("scan_number", None)
    title_scan = "" if scan_num is None else f", scan {scan_num}"
    fig.suptitle(
        f"{source}{title_scan}: amplitude vs time",
        fontsize=14,
    )
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        if scan_num is None:
            fname = f"avg_{source}_amp_vs_time_all_baselines.png"
        else:
            fname = f"avg_{source}_scan{scan_num}_amp_vs_time_all_baselines.png"
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs


def plot_results_phase_vs_scan_all_baselines(
    results,
    channel=0,
    unwrap_phase=False,
    alma_station="AA",
    include_autocorr=False,
    figsize_per_panel=(4.2, 3.0),
    source="M87",
    figdir="meta_plot",
    savefig=False,
):
    """
    Plot phase vs scan using precomputed result dicts.

    Each result should be the output of build_scan_coherency_matrix().
    For each scan, coherently average complex visibility over time within that
    scan first, then plot phase in degrees.
    Each subplot = one baseline.
    Each subplot contains all 4 pols.
    """
    if len(results) == 0:
        raise ValueError("results is empty")
    # Use union of all stations appearing across provided results.
    station_list_all = np.unique(
        np.concatenate([np.asarray(r["station_list"]) for r in results])
    )
    baselines = get_baselines_from_station_list(
        station_list_all,
        include_autocorr=include_autocorr,
    )
    nbase = len(baselines)
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
        out[baseline] = {pol: {"scan": [], "phase": []} for pol in pol_map}
        for pol_label, (p, q) in pol_map.items():
            scan_nums = []
            phases = []
            for r in results:
                station_list = np.asarray(r["station_list"])
                if s1 not in station_list or s2 not in station_list:
                    continue
                i = np.where(station_list == s1)[0][0]
                j = np.where(station_list == s2)[0][0]
                channel_list = np.asarray(r["channel_list"])
                if channel in channel_list:
                    ichan = np.where(channel_list == channel)[0][0]
                else:
                    ichan = int(channel)
                V = r["allcoh"][:, ichan, i, j, p, q]
                V = np.where(V == 0, np.nan + 1j * np.nan, V)
                # coherent average over time within this scan
                Vavg = np.nanmean(V)
                if not np.isfinite(Vavg):
                    continue
                scan_num = r.get("scan_number", np.nan)
                scan_nums.append(scan_num)
                phases.append(np.angle(Vavg, deg=True))
            scan_nums = np.asarray(scan_nums)
            phases = np.asarray(phases)
            order = np.argsort(scan_nums)
            scan_nums = scan_nums[order]
            phases = phases[order]
            if unwrap_phase:
                phases = np.rad2deg(np.unwrap(np.deg2rad(phases)))
            ax.plot(
                scan_nums,
                phases,
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=pol_label,
            )
            out[baseline][pol_label]["scan"] = scan_nums
            out[baseline][pol_label]["phase"] = phases
        ax.set_title(baseline, fontsize=10)
        ax.grid(alpha=0.3)
        if ibl == nbase - 1:
            ax.set_xlabel("Scan number")
            ax.set_ylabel("Phase [deg]")
            ax.legend(title="Pol", fontsize=8, title_fontsize=9)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
    for k in range(nbase, nrows * ncols):
        axs.flat[k].axis("off")
    fig.suptitle(f"{source}: phase vs scan", fontsize=14)
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = f"avg_{source}_phase_vs_scan_all_baselines.png"
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs, out


def plot_results_amp_vs_scan_all_baselines(
    results,
    channel=0,
    alma_station="AA",
    include_autocorr=False,
    figsize_per_panel=(4.2, 3.0),
    amp_scale="linear",
    source="M87",
    figdir="meta_plot",
    savefig=False,
):
    """
    Plot amplitude vs scan using precomputed result dicts.

    Each result should be the output of build_scan_coherency_matrix().
    For each scan, coherently average complex visibility over time within that
    scan first, then plot amplitude.
    Each subplot = one baseline.
    Each subplot contains all 4 pols.
    """
    if len(results) == 0:
        raise ValueError("results is empty")
    station_list_all = np.unique(
        np.concatenate([np.asarray(r["station_list"]) for r in results])
    )
    baselines = get_baselines_from_station_list(
        station_list_all,
        include_autocorr=include_autocorr,
    )
    nbase = len(baselines)
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
        out[baseline] = {pol: {"scan": [], "amp": []} for pol in pol_map}
        for pol_label, (p, q) in pol_map.items():
            scan_nums = []
            amps = []
            for r in results:
                station_list = np.asarray(r["station_list"])
                if s1 not in station_list or s2 not in station_list:
                    continue
                i = np.where(station_list == s1)[0][0]
                j = np.where(station_list == s2)[0][0]
                channel_list = np.asarray(r["channel_list"])
                if channel in channel_list:
                    ichan = np.where(channel_list == channel)[0][0]
                else:
                    ichan = int(channel)
                V = r["allcoh"][:, ichan, i, j, p, q]
                V = np.where(V == 0, np.nan + 1j * np.nan, V)
                # coherent average over time within this scan
                Vavg = np.nanmean(V)
                if not np.isfinite(Vavg):
                    continue
                scan_num = r.get("scan_number", np.nan)
                scan_nums.append(scan_num)
                amps.append(np.abs(Vavg))
            scan_nums = np.asarray(scan_nums)
            amps = np.asarray(amps)
            order = np.argsort(scan_nums)
            scan_nums = scan_nums[order]
            amps = amps[order]
            ax.plot(
                scan_nums,
                amps,
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=pol_label,
            )
            out[baseline][pol_label]["scan"] = scan_nums
            out[baseline][pol_label]["amp"] = amps
        ax.set_title(baseline, fontsize=10)
        ax.grid(alpha=0.3)
        if amp_scale == "log":
            ax.set_yscale("log")
        elif amp_scale != "linear":
            raise ValueError("amp_scale must be 'linear' or 'log'")
        if ibl == nbase - 1:
            ax.set_xlabel("Scan number")
            ax.set_ylabel("Amplitude")
            ax.legend(title="Pol", fontsize=8, title_fontsize=9)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
    for k in range(nbase, nrows * ncols):
        axs.flat[k].axis("off")
    fig.suptitle(f"{source}: amplitude vs scan", fontsize=14)
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fname = f"avg_{source}_amp_vs_scan_all_baselines.png"
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs, out


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
):
    """
    Plot visibility amplitude against uv distance for all four products.

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
    axs[0, 0].plot(uvdist, np.abs(rr), 'b.', label='RR')
    axs[0, 1].plot(uvdist, np.abs(ll), 'r.', label='LL')
    axs[1, 0].plot(uvdist, np.abs(lr), 'g.', label='LR')
    axs[1, 1].plot(uvdist, np.abs(rl), 'm.', label='RL')
    for ax in axs.flat:
        ax.set_xlabel('uv distance [Gλ]')
        ax.set_ylabel('Visibility Amplitude')
        ax.legend()
        ax.grid(True)
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        fig.savefig(os.path.join(figdir, f"{source}_amp_uvdist.png"), dpi=300)
    if show:
        plt.show()
    return fig, axs


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
):
    """
    Plot closure phase vs time for all triangles and both parallel-hand pols.

    Each subplot corresponds to one triangle and one polarization.

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
    scan_num = closure.get("scan_number", None)
    title_scan = "" if scan_num is None else f", scan {scan_num}"
    fig.suptitle(
        f"{source}{title_scan}: closure phase vs time, {channel_label}",
        fontsize=14,
    )
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        if scan_num is None:
            fname = f"{source}_closure_phase_vs_time_all_triangles.png"
        else:
            fname = f"{source}_scan{scan_num}_closure_phase_vs_time_all_triangles.png"
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
):
    """
    Plot closure amplitude vs time for all quadrangles and both parallel-hand pols.

    Each subplot corresponds to one quadrangle and one polarization.

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
    scan_num = closure.get("scan_number", None)
    title_scan = "" if scan_num is None else f", scan {scan_num}"
    quantity_name = "log closure amplitude" if use_logamp else "closure amplitude"
    fig.suptitle(
        f"{source}{title_scan}: {quantity_name} vs time, {channel_label}",
        fontsize=14,
    )
    if savefig:
        os.makedirs(figdir, exist_ok=True)
        if scan_num is None:
            fname = f"{source}_closure_amp_vs_time_all_quadrangles.png"
        else:
            fname = f"{source}_scan{scan_num}_closure_amp_vs_time_all_quadrangles.png"
        fig.savefig(os.path.join(figdir, fname), dpi=300, bbox_inches="tight")
    return fig, axs
