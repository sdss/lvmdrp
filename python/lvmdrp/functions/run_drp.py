#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
import yaml
import pandas as pd
from typing import Union
from functools import lru_cache
import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from lvmdrp.functions.imageMethod import (preproc_raw_frame, create_master_frame,
                                          create_pixelmask, detrend_frame,
                                          find_peaks_auto, trace_peaks,
                                          extract_spectra)
from lvmdrp.functions.rssMethod import (determine_wavelength_solution, create_pixel_table,
                                        resample_wavelength, join_spec_channels)
from lvmdrp.utils.metadata import (get_frames_metadata, get_master_metadata, extract_metadata,
                                   get_analog_groups, match_master_metadata, create_master_path)
from lvmdrp import config, log, path, __version__ as drpver


def get_config_options(level: str, flavor: str = None) -> dict:
    """ Get the DRP config options

    Get the DRP custom configuration options for
    a given reduction step and flavor. ``level`` is
    a dotted string of the nested config levels from
    the top level down, i.e. "reduction_steps.preproc_raw_frame".
    ``flavor`` is the image type of the frame, e.g. "bias".

    Parameters
    ----------
    level : str
        A nested dotted path
    flavor : str
        The frame flavor or image type

    Returns
    -------
    dict
        any custom config options
    """
    # load custom config options

    cfg = config.copy()
    for lvl in level.split('.'):
        cfg = cfg.get(lvl, {})
    return cfg.get(flavor, cfg.get("default", {})) if flavor else cfg.get("default", cfg)


def create_masters(flavor: str, frames: pd.DataFrame):
    """ Create the master calibration frames

    Create the master calibration frames for a given flavor
    or imagetyp.  These files live in the "calib" subdirectory
    with the "lvm-m(flavor)-*" prefix.

    Parameters
    ----------
    flavor : str
        The image type of the exposure
    frames : pd.DataFrame
        The dataframe of raw frame metadata
    """

    # get flavor subset
    sub = frames[frames['imagetyp'] == flavor]

    # check for empty rows
    if len(sub) == 0:
        log.error(f'No exposures found for flavor {flavor}.  Cannot create master frame.')
        return

    mjd = sub['mjd'].iloc[0]
    tileid = sub['tileid'].iloc[0]

    # get the analog input files to create the master frame
    kwargs = get_config_options('reduction_steps.create_master_frame', flavor)
    frames, in_files, out_files = get_analog_groups(tileid, mjd, imagetyp=flavor, quality='excellent')
    for i, in_f in enumerate(in_files.values()):
        master = out_files[i]
        # create parent dir if needed
        if not pathlib.Path(master).parent.exists():
            pathlib.Path(master).parent.mkdir(parents=True, exist_ok=True)

        # create the master frame
        create_master_frame(in_images=in_f, out_image=master, **kwargs)


def find_masters(mjd: int, flavor: str, camera: str) -> dict:
    """ Find the matching master frames

    Find the matching master frames for a given flavor, camera
    and exposure time.  Returns a dict of each bias, dark, flat flavors
    and the appropriate master frame for that type.

    Parameters
    ----------
    flavor : str
        The image type
    camera : str
        The camera
    exptime : str
        The exposure time

    Returns
    -------
    dict
        The output master frame paths for each flavor
    """
    # try to match the master frames for a given flavor, camera
    matches = match_master_metadata(target_mjd=mjd, target_imagetyp=flavor, target_camera=camera)

    # construct the dict of filepaths
    files = dict.fromkeys(matches.keys())
    for key, val in matches.items():
        if val is None:
            continue
        files[key] = create_master_path(val)
    return files


def trace_fibers(in_file: str, camera: str, tileid: int, mjd: int):
    """ Perform flat fiber tracing

    Runs the fiber trace peak finder algorithm and traces the peaks to
    identify the fibers.

    Parameters
    ----------
    in_file : str
        the input preprocessed file path
    camera : str
        the name of the camera
    expnum : int
        the frame exposure number
    tileid : int
        the sky tile id
    mjd : int
        the MJD of observation
    """
    # out_peaks = path.full('lvm_master', mjd=mjd, camera=camera, kind='mpeaks', tileid=tileid,
    #                       drpver=drpver)
    out_trace = path.full('lvm_master', mjd=mjd, camera=camera, kind='mtrace', tileid=tileid,
                          drpver=drpver)

    # check for parent dir existence
    # if not pathlib.Path(out_peaks).parent.is_dir():
    #     pathlib.Path(out_peaks).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(out_trace):
        log.info('Trace file already exists.')
        return

    # log.info('--- Running auto peak finder ---')
    # kwargs = get_config_options('reduction_steps.find_peaks_auto')
    # log.info(f'custom configuration parameters for find_peaks_auto: {repr(kwargs)}')
    # find_peaks_auto(in_image=in_file, out_peaks=out_peaks, **kwargs)
    # log.info(f'Output peak finder file: {out_peaks}')

    log.info('--- Tracing fiber peaks ---')
    kwargs = get_config_options('reduction_steps.trace_peaks')
    log.info(f'custom configuration parameters for trace_peaks: {repr(kwargs)}')
    trace_peaks(in_image=in_file, out_trace=out_trace, in_peaks=None, **kwargs)
    log.info(f'Output trace fiber peaks file: {out_trace}')

    # TODO
    # add new function to trace the width
    # output of this file goes into extract_spectra
    # trace widths only for full reductions
    if not config.get('quick'):
        pass


def find_file(kind: str, camera: str = None, mjd: int = None, tileid: int = None) -> str:
    """ Find a master file

    Finds the master trace, wave, and lsf files.  These files are output
    from the fiber tracing and wavelength calibration routines run on the mflat and marc
    files.

    Parameters
    ----------
    kind : str
        The kind of file to find
    camera : str, optional
        The camera, by default None
    mjd : int, optional
        The MJD of the observation, by default None
    tileid : int, optional
        The tile id of the observartion, by default None

    Returns
    -------
    str
        the file path
    """
    files = sorted(path.expand('lvm_master', kind=kind, drpver=drpver, mjd=mjd, tileid=tileid,
                   camera=camera))

    if not files:
        log.warning(f"No {kind} files found for {tileid}, {mjd}, {camera}.  Discontinuing reduction.")
        return

    # pick the last one in the list, sorted by exposure number
    return files[-1]


def reduce_frame(filename: str, camera: str = None, mjd: int = None,
                 expnum: int = None, tileid: int = None,
                 flavor: str = None, master: bool = None, **fkwargs):
    """ Reduce a single raw frame exposure

    Reduces a single LVM raw frame sdR exposure

    Parameters
    ----------
    filename : str
        The sdR raw frame filepath
    camera : str, optional
        the camera and spectrograph name, e.g "b1", by default None
    mjd : int, optional
        the MJD of the file, by default None
    expnum : int, optional
        the exposure number of the frame, by default None
    tileid : int, optional
        the tile id of the frame, by default None
    flavor : str, optional
        the flavor or image type, by default None
    master : bool, optional
        flag if we are reducing master flats/arcs
    """
    # start logging for this mjd
    start_logging(mjd, tileid)

    log.info(f'--- Starting reduction of raw frame: {filename}')

    # set flavor
    flavor = flavor or fkwargs.get('imagetyp')
    flavor = 'fiberflat' if flavor == 'flat' else flavor

    # check master frames
    masters = find_masters(mjd, "object", camera)
    mbias = masters.get('bias')
    mdark = masters.get('dark')
    mpixflat = masters.get('pixelflat')
    mflat = masters.get('flat')
    marc = masters.get('arc')
    mpixmask = masters.get('pixmask')

    # log the master frames
    log.info(f'Using master bias: {mbias}')
    log.info(f'Using master dark: {mdark}')
    log.info(f'Using master pixel flat: {mpixflat}')
    log.info(f'Using master flat: {mflat}')
    log.info(f'Using master arc: {marc}')
    log.info(f'Using master pixel mask: {mpixmask}')

    # only run these steps for individual exposures
    if not master:
        # preprocess the frames
        log.info('--- Preprocessing raw frame ---')
        kwargs = get_config_options('reduction_steps.preproc_raw_frame', flavor)
        log.info(f'custom configuration parameters for preproc_raw_frame: {repr(kwargs)}')
        out_pre = path.full('lvm_anc', kind='p', imagetype=flavor, mjd=mjd, camera=camera,
                            drpver=drpver, expnum=expnum, tileid=tileid)
        # create the root dir if needed
        if not pathlib.Path(out_pre).parent.exists():
            pathlib.Path(out_pre).parent.mkdir(parents=True, exist_ok=True)

        preproc_raw_frame(in_image=filename, in_mask=mpixmask, out_image=out_pre, **kwargs)

        # process the flat/arc frames
        in_cal = path.full("lvm_anc", kind='p', imagetype=flavor, mjd=mjd, drpver=drpver,
                        camera=camera, tileid=tileid, expnum=expnum)
        out_cal = path.full("lvm_anc", kind='c', imagetype=flavor, mjd=mjd, drpver=drpver,
                            camera=camera, tileid=tileid, expnum=expnum)

        log.info(f'Output preproc file: {in_cal}')
        log.info('--- Running detrend frame ---')
        kwargs = get_config_options('reduction_steps.detrend_frame', flavor)
        log.info(f'custom configuration parameters for detrend_frame: {repr(kwargs)}')
        detrend_frame(in_image=in_cal, out_image=out_cal,
                      in_bias=mbias, in_dark=mdark, in_pixelflat=mpixflat,
                      in_slitmap=Table(fibermap.data),
                      **kwargs)
        log.info(f'Output calibrated file: {out_cal}')

    # end reduction for individual bias, darks, arcs and flats
    if flavor in {'bias', 'dark', 'arc', 'fiberflat', 'flat'} and not master:
        return

    # compute the input calibration file path
    if master:
        cal_file = marc if flavor == 'arc' else mflat
    else:
        cal_file = path.full("lvm_anc", kind='c', imagetype=flavor, mjd=mjd, drpver=drpver,
                             camera=camera, tileid=tileid, expnum=expnum)

    # fiber tracing for master flat
    if master and 'flat' in flavor:
        log.info('--- Running fiber trace ---')
        trace_fibers(mflat, camera, tileid, mjd)

    # extract fiber spectra

    # get the output file path
    xout_file = create_output_path(kind='x', flavor=flavor, mjd=mjd, tileid=tileid,
                                   camera=camera, expnum=expnum, master=master)

    # find the fiber trace file
    trace_file = find_file('mtrace', mjd=mjd, tileid=tileid, camera=camera)
    if not trace_file:
        return

    # perform the fiber extraction
    log.info('--- Extracting fiber spectra ---')
    kwargs = get_config_options('reduction_steps.extract_spectra')
    log.info(f'custom configuration parameters for extract_spectra: {repr(kwargs)}')
    extract_spectra(in_image=cal_file, out_rss=xout_file, in_trace=trace_file, **kwargs)
    log.info(f'Output extracted file: {xout_file}')

    # determine the wavelength solution
    if master and flavor == 'arc':
        wave_file = path.full('lvm_master', mjd=mjd, camera=camera, kind='mwave', tileid=tileid,
                              drpver=drpver)
        lsf_file = path.full('lvm_master', mjd=mjd, camera=camera, kind='mlsf', tileid=tileid,
                             drpver=drpver)
        kwargs = get_config_options('reduction_steps.determine_wavesol')
        log.info('--- Determining wavelength solution ---')
        log.info(f'custom configuration parameters for determine_wave_solution: {repr(kwargs)}')
        determine_wavelength_solution(in_arc=xout_file, out_wave=wave_file, out_lsf=lsf_file,
                                      **kwargs)
        log.info(f'Output wave peak traceset file: {wave_file}')
        log.info(f'Output lsf traceset file: {lsf_file}')

    # perform wavelength calibration
    wave_file = find_file('mwave', mjd=mjd, tileid=tileid, camera=camera)
    lsf_file = find_file('mlsf', mjd=mjd, tileid=tileid, camera=camera)
    if not (wave_file and lsf_file):
        return
    wout_file = create_output_path(kind='w', flavor=flavor, mjd=mjd, tileid=tileid,
                                   camera=camera, expnum=expnum, master=master)
    log.info('--- Creating pixel table ---')
    create_pixel_table(in_rss=xout_file, out_rss=wout_file, arc_wave=wave_file, arc_fwhm=lsf_file)
    log.info(f'Output calibrated wavelength file: {wout_file}')

    # set wavelength resample params
    CHANNEL_WL = {"b": (3600, 5930), "r": (5660, 7720), "z": (7470, 9800)}
    wave_range = CHANNEL_WL[camera[0]]

    # resample onto a common wavelength
    hout_file = create_output_path(kind='h', flavor=flavor, mjd=mjd, tileid=tileid,
                                   camera=camera, expnum=expnum, master=master)
    kwargs = get_config_options('reduction_steps.resample_wave', flavor)
    log.info('--- Resampling wavelength grid ---')
    log.info(f'custom configuration parameters for resample_wave: {repr(kwargs)}')
    resample_wavelength(in_rss=wout_file, out_rss=hout_file, start_wave=wave_range[0],
                        end_wave=wave_range[1], **kwargs)
    log.info(f'Output resampled wave file: {hout_file}')

    # write out RSS

    # perform quality checks


def create_output_path(kind: str, flavor: str, mjd: int, tileid: int, camera: str,
                       expnum: int = None, master: bool = None) -> str:
    """ Creates the output file path

    Creates the output file path for the science frames or the master arc/flats.
    For example, the extracted fiber spectra is
    "1111/60115/ancillary/lvm-xobject-b1-00060115.fits" for science frames
    or "1111/60115/calib/lvm-xmarc-b1.fits" for the master arc frame.

    Parameters
    ----------
    kind : str
        The kind of file to write
    flavor : str
        The flavor or imagetype of the observation
    mjd : int
        The MJD of the observation
    tileid : int
        The tile id of the observation
    camera : str
        The camera name, e.g. b1
    expnum : int
        The exposure number
    master : bool
        Flag to create the master output path

    Returns
    -------
    str
        the output file path
    """

    if master:
        return path.full('lvm_master', mjd=mjd, camera=camera, kind=f'{kind}m{flavor}',
                         tileid=tileid, drpver=drpver)
    else:
        return path.full("lvm_anc", kind=kind, imagetype=flavor, mjd=mjd, drpver=drpver,
                         camera=camera, tileid=tileid, expnum=expnum)


def parse_mjds(mjd: Union[int, str, list, tuple]) -> Union[int, list]:
    """ Parse the input MJD

    Parses the input MJD into a single integer MJD or a list
    of integer MJD.  Valid inputs are a single int 60010,
    a list of specific MJDs [60010, 60040], or a string
    range of MJDs 60010-60040.

    Parameters
    ----------
    mjd : Union[int, str, list, tuple]
        the input MJD range or value to parse

    Returns
    -------
    Union[int, list]
        Either a single integer MJD or list of MJDs
    """

    if isinstance(mjd, int):
        return mjd
    elif isinstance(mjd, (tuple, list)):
        return sorted(map(int, mjd))
    elif isinstance(mjd, str) and mjd.isdigit():
        return int(mjd)
    elif isinstance(mjd, str) and '-' in mjd:
        return split_mjds(mjd)


def split_mjds(mjd: str) -> list:
    """ Split a string range of MJDs

    Splits a string range of MJDs, e.g. "60010-60040", into
    a list of all (inclusive) MJDs within the range
    specified.  A range can also be specified as
    "-60040" or "60010-" to indicate that the range
    includes all mjds prior to or following the
    given MJD.

    Parameters
    ----------
    mjd : str
        An hyphen-separated MJD range

    Returns
    -------
    list
        A list of MJDs
    """
    start_mjd, end_mjd = mjd.split('-')
    start_mjd = int(start_mjd) if start_mjd else None
    end_mjd = int(end_mjd) if end_mjd else None

    p = pathlib.Path(os.getenv('LVM_DATA_S'))
    mjds = []
    for d in p.iterdir():
        if not d.stem.isdigit():
            continue
        mm = int(d.stem)
        if start_mjd and end_mjd and (mm >= start_mjd and mm <= end_mjd):
            mjds.append(mm)
        elif start_mjd and not end_mjd and (mm >= start_mjd):
            mjds.append(mm)
        elif not start_mjd and end_mjd and (mm <= end_mjd):
            mjds.append(mm)
    return sorted(mjds)


def filter_expnum(frame: pd.DataFrame, expnum: Union[int, str, list]) -> pd.DataFrame:
    """ Filter the dataframe by exposure number

    Filters the metadata dataframe by the input exposure numbers.  expnum
    can be a single integer value, a list of individual exposure numbers,
    or a string range, i.e. "190-200".  A range can also be specified as
    "-200" or "190-" to indicate that the range includes all exposures prior to
    or following the given exposure number.  Ranges are inclusive to input
    boundaries.

    Parameters
    ----------
    frame : pd.DataFrame
        the metadata of exposure information
    expnum : Union[int, str, list]
        the input exposure number range or value to parse

    Returns
    -------
    pd.DataFrame
        The subset of frames matching the condition
    """

    if isinstance(expnum, int):
        query = f"expnum == {expnum}"
    elif isinstance(expnum, (tuple, list)):
        query = f" expnum in {sorted(map(int, expnum))}"
    elif isinstance(expnum, str) and expnum.isdigit():
        query = f"expnum == {int(expnum)}"
    elif isinstance(expnum, str) and '-' in expnum:
        start_exp, end_exp = expnum.split('-')
        start_exp = int(start_exp) if start_exp else None
        end_exp = int(end_exp) if end_exp else None
        if start_exp and end_exp:
            query = f"{start_exp} <= expnum <= {end_exp}"
        elif start_exp:
            query = f"expnum >= {start_exp}"
        elif end_exp:
            query = f"expnum <= {end_exp}"
    return frame.query(query)


def reduce_file(filename: str):
    """ Reduce a single file

    Run a single raw frame through the LVM DRP

    Parameters
    ----------
    filename : str
        The full filepath name
    """
    meta = extract_metadata([filename])
    params = meta.iloc[0].to_dict()

    reduce_frame(filename, **params)


def reduce_set(frame: pd.DataFrame, settype: str = None, flavor: str = None,
               create_pixmask: bool = False):
    """ Reduce a set of precals, cals, or science """

    if settype not in {"precals", "cals", "science"}:
        raise ValueError('settype can only be "precals", "cals", or "science".')

    # if flavor is set, only reduce those
    if flavor:
        frame = frame[frame['imagetyp'] == flavor]

    # reduce frames
    rows = frame.to_dict('records')
    for row in rows:
        # get raw frame filepath
        filepath = path.full('lvm_raw', mjd=row['mjd'], expnum=row['expnum'],
                             hemi='s', camspec=row['camera'])

        # reduce the frame, pass in entire parameter set
        reduce_frame(filepath, **row)

    # don't create masters for science frames
    if settype == 'science':
        return

    # set the master flavors
    flavors = {'bias', 'dark', 'pixelflat'} if settype == 'precals' else {'arc', 'flat'}

    # if a flavor is set, only create those masters
    if flavor:
        flavors = {flavor}

    # create master frames
    for flavor in flavors:
        create_masters(flavor, frame)

    # build the master metadata cache ; always update it
    get_master_metadata(overwrite=True)

    # run pixel mask creation when requested
    if create_pixmask:
        # loop over set of cameras in frame
        for camera in set(frame['camera']):
            masters = find_masters(frame.mjd.iloc[0], "object", camera)
            mbias = masters.get('bias')
            mdark = masters.get('dark')
            mpixflat = masters.get('pixelflat')
            # pass master filenames into new function
            mpixmask = path.full('lvm_master', kind='mpixmask', drpver=drpver,
                                 mjd=frame.mjd.iloc[0], tileid=frame.tileid.iloc[0],
                                 camera=camera)
            create_pixelmask(in_bias=mbias, in_dark=mdark, in_pixelflat=mpixflat, out_mask=mpixmask)

        # update masters metadata to include new pixel masks
        get_master_metadata(overwrite=True)


def reduce_masters(mjd: int):
    """ Reduce master arcs and flats """
    masters = get_master_metadata()
    sub = masters[(masters['mjd'] == mjd) & (masters['imagetyp'].isin({'arc', 'flat'}))]
    path = create_master_path(sub.iloc[0])

    # sort the frames to flat, arc, flat
    sub = sort_cals(sub, master=True)

    # reduce frames
    rows = sub.to_dict('records')
    for row in rows:
        # construct master path
        path = create_master_path(pd.Series(row))

        # reduce the frame, pass in entire parameter set
        reduce_frame(path, master=True, **row)


def run_drp(mjd: Union[int, str, list], bias: bool = False, dark: bool = False,
            pixelflat: bool = False, skip_bd: bool = False, arc: bool = False, flat: bool = False,
            only_bd: bool = False, only_cal: bool = False, only_sci: bool = False, pixmask: bool = False,
            spec: int = None, camera: str = None, expnum: Union[int, str, list] = None,
            quick: bool = False):
    """ Run the LVM DRP

    Run the LVM data reduction pipeline on.  Optionally set flags
    to reduce only a subset of data.

    Parameters
    ----------
    mjd : Union[int, str, list], optional
        The MJD of the raw data to reduce, by default None
    arc : bool, optional
        Flag to only reduce arc frames, by default False
    flat : bool, optional
        Flag to only reduce flat frames, by default False
    skip_bd : bool, optional
        Flag to skip reduction of bias/darks
    """
    # update the quick redux flag if necessary
    if not config.get('quick') and quick:
        config['quick'] = quick

    # write the drp parameter configuration
    write_config_file()

    # parse the input MJD and loop over all reductions
    mjds = parse_mjds(mjd)
    if isinstance(mjds, list):
        for mjd in mjds:
            run_drp(mjd=mjd, bias=bias, dark=dark, pixelflat=pixelflat, skip_bd=skip_bd, arc=arc,
                    flat=flat, only_bd=only_bd, only_cal=only_cal, only_sci=only_sci, pixmask=pixmask,
                    spec=spec, camera=camera, expnum=expnum, quick=quick)
        return

    log.info(f'Processing MJD {mjd}')

    # check the MJD data directory path
    mjd_path = pathlib.Path(os.getenv('LVM_DATA_S')) / str(mjd)
    log.info(f'MJD processing path: {mjd_path}')
    if not mjd_path.is_dir():
        log.warning(f'{mjd = } is not valid raw data directory.')
        return

    # find files
    frames = get_frames_metadata(mjd=mjd)
    sub = frames.copy()

    # remove bad or test quality frames
    sub = sub[~(sub['quality'] != 'excellent')]

    # filter on files
    if bias:
        sub = sub[sub['imagetyp'] == 'bias']
    if dark:
        sub = sub[sub['imagetyp'] == 'dark']
    if pixelflat:
        sub = sub[sub['imagetyp'] == 'pixelflat']

    # filter on camera or spectrograph
    if spec:
        sub = sub[sub['spec'] == f'sp{spec}']
    if camera:
        sub = sub[sub['camera'].str.contains(camera)]

    # filter on exposure number
    if expnum:
        log.info(f'Filtering on exposure numbers {expnum}.')
        sub = filter_expnum(sub, expnum)

    # get biases and darks
    cond = sub['imagetyp'].isin(['bias', 'dark', 'pixelflat'])
    precals = sub[cond]
    if len(precals) == 0 and not skip_bd:
        log.error(f'No biases or darks found for mjd {mjd}. Discontinuing reduction.')
        return
    precals = precals.sort_values(['expnum', 'camera'])

    if not skip_bd:
        # reduce biases / darks / pixelflats
        reduce_set(precals, settype='precals', flavor='bias')
        if not pixmask:
            reduce_set(precals, settype='precals', flavor='dark')
            reduce_set(precals, settype='precals', flavor='pixelflat')
        else:
            on_pixflats = 'pixelflat' in set(precals['imagetyp'])
            reduce_set(precals, settype='precals', flavor='dark', create_pixmask=not on_pixflats)
            reduce_set(precals, settype='precals', flavor='pixelflat', create_pixmask=on_pixflats)

    # returning if only reducing bias/darks
    if only_bd:
        return

    # get all other image types
    sub = sub[~cond]
    if flat or arc:
        sub = sub[sub['imagetyp'] == ('arc' if arc else 'flat')]
    elif only_cal:
        sub = sub[~(sub['imagetyp'] == 'object')]
    elif only_sci:
        sub = sub[sub['imagetyp'] == 'object']

    # exit if not arcs, flats, or science frames in mjd
    if len(sub) == 0:
        log.error(f'No cals or science frames found for mjd {mjd}. Discontinuing reduction.')
        return

    # group the frames
    sub = sub.sort_values(['expnum', 'camera'])

    # split into cals, and science
    cals = sub[~(sub['imagetyp'] == 'object')]
    sci = sub[sub['imagetyp'] == 'object']

    # reduce the individual flats/arcs
    if not only_sci:
        reduce_set(cals, settype='cals')

    # reduce the master flat/arcs
    if not only_sci:
        reduce_masters(mjd=mjd)

    # return if only calibration set
    if only_cal or flat or arc or bias or dark or pixelflat:
        return

    # reduce science files
    reduce_set(sci, settype='science')

    # TODO - check for single elements
    mjd = list(set(sub['mjd']))[0]
    tileid = list(set(sub['tileid']))[0]

    # perform camera combination
    # produces ancillary/lvm-object-sp[id]-[expnum] files
    for tileid, mjd in sub.groupby(['tileid', 'mjd']).groups.keys():
        combine_cameras(tileid, mjd, spec=1)
        combine_cameras(tileid, mjd, spec=2)
        combine_cameras(tileid, mjd, spec=3)

    # perform spectrograph combination
    # produces lvm-CFrame file
    exposures = set(sci['expnum'].sort_values())
    for expnum in exposures:
        combine_spectrographs(tileid, mjd, expnum)

    # perform sky subtraction

    # perform flux calibration


def start_logging(mjd: int, tileid: int):
    """ Starts a file logger

    Starts a file logger for a given MJD and tile ID.

    Parameters
    ----------
    mjd : int
        The MJD of the observations
    tileid : int
        The tile ID of the observations
    """
    lpath = (os.path.join(os.getenv('LVM_SPECTRO_REDUX'),
             "{drpver}/{tileid}/{mjd}/lvm-drp-{tileid}-{mjd}.log"))
    logpath = lpath.format(drpver=drpver, mjd=mjd, tileid=tileid)
    logpath = pathlib.Path(logpath)

    # if logpath.exists():
    #     return

    if not logpath.parent.exists():
        logpath.parent.mkdir(parents=True, exist_ok=True)

    log.start_file_logger(logpath)


def write_config_file():
    """ Write out the DRP configuration file """
    cpath = pathlib.Path(os.getenv('LVM_SPECTRO_REDUX')) / drpver / f"lvm-config-{drpver}.yaml"

    # create dir if needed
    if not cpath.parent.is_dir():
        cpath.parent.mkdir(parents=True, exist_ok=True)

    # write the config file
    with open(cpath, 'w') as f:
        f.write(yaml.safe_dump(dict(config), sort_keys=False, indent=2))


def sort_cals(df: pd.DataFrame, master: bool = False) -> pd.DataFrame:
    """ Sort raw frames table by calibrations

    Sorts and orders the table of raw frames by calibration,
    then science frames.  Calibration frames are ordered by flats,
    then arcs, then flats again, to be reduced in that order.
    This is so flats can be properly wavelength calibration after
    arc reduction.

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe of raw frames to process
    master : bool
        Flag indicating the frame is for masters

    Returns
    -------
    pd.DataFrame
        a sorted dataframe of raw frames
    """

    # get unique flavors
    imtypes = set(df['imagetyp'])

    # return if no flat or arcs in dataset
    if {'flat', 'arc'} - imtypes:
        log.info("No flats or arcs found in dataset. No need to sort.")
        return df

    # check for image types and remove missing flavors from the index list
    flavors = ['flat', 'arc', 'object']
    missing = set(flavors) - imtypes
    __ = [flavors.remove(i) for i in missing]

    # sort and set index
    sort_fields = ['camera'] if master else ['camera', 'expnum']
    ss = df.sort_values(sort_fields)
    ee = ss.set_index('imagetyp', drop=False).loc[flavors]

    # check dimensions
    flats = ee.loc['flat']
    flats = flats.to_frame().transpose() if flats.ndim == 1 else flats
    if 'object' in imtypes:
        obj = ee.loc['object']
        obj = obj.to_frame().transpose() if obj.ndim == 1 else obj

    # append flats to end of calibration frames, and build new dataframe
    calibs = pd.concat([ee.loc[['flat', 'arc']], flats]).reset_index(drop=True)
    return (pd.concat([calibs, obj]).reset_index(drop=True)
            if 'object' in imtypes else calibs)


def find_best_mdark(tileid: int, mjd: int, camera: str) -> str:
    """ Find the best master dark frame

    Finds the master dark frame with the largest exposure time, for an
    input tileid, MJD, and camera.

    Parameters
    ----------
    tileid : int
        the sky tileid
    mjd : int
        the MJD of observation
    camera : str
        the camera name

    Returns
    -------
    str
        the filepath to the master dark
    """
    darks = path.expand("lvm_master", kind='mdark', mjd=mjd, drpver=drpver,
                        camera=camera, tileid=tileid)

    # return if no master dark found
    if not darks:
        log.warning(f'No master dark frame found for {tileid}, {mjd}, {camera}.')
        return

    # return first master dark in the list
    return darks[0]


def _parse_expnum_cam(name: str) -> tuple:
    """ Parse the filename

    Parse the camera and exposure number from the
    filename.

    Parameters
    ----------
    name : str
        the name of the file

    Returns
    -------
    tuple
        the camera and exposure number
    """
    pp = pathlib.Path(name).stem
    ss = pp.split('-')
    return int(ss[-1]), ss[-2]


def combine_cameras(tileid: int, mjd: int, expnum: int = None, spec: int = 1):
    """ Combine the cameras together

    Combines all available cameras (b, r, z) together on a given
    spectrograph. For all exposures, combines all "hobject" ancillary
    files into a single "bobject" ancillary file, per spectrograph.

    Parameters
    ----------
    tileid : int
        the sky tileid
    mjd : int
        the MJD of observation
    spec : int, optional
        the spectrograph id, by default 1
    """
    from itertools import groupby

    # pattern = f'*hobject-*-*{spec}-*'
    # hfiles = sorted(list(pathlib.Path(os.getenv('LVM_SPECTRO_REDUX')).rglob(pattern)),
    #                 key=_parse_expnum_cam)

    # find all the h object files
    hfiles = path.expand('lvm_anc', mjd=mjd, tileid=tileid, drpver=drpver,
                         imagetype='object', expnum=expnum or '****', kind='h', camera=f'*{spec}')
    hfiles = map(pathlib.Path, sorted(hfiles, key=_parse_expnum_cam))

    log.info(f'--- Combining cameras from spec {spec} ---')
    kwargs = get_config_options('reduction_steps.combine_cameras')
    log.info(f'custom configuration parameters for combine cameras: {repr(kwargs)}')

    m = [f'b{spec}', f'r{spec}', f'z{spec}']

    # loop over all files, grouped by exposure
    for key, exps in groupby(hfiles, lambda x: int(x.stem.split('-')[-1])):
        log.info(f'combining cameras for exposure {key}')

        # create the output b object file, 1 per spectrograph
        bout_file = path.full('lvm_anc', mjd=mjd, tileid=tileid, drpver=drpver,
                              imagetype='object', expnum=key, kind='', camera=f'sp{spec}')

        # pad the exposure list for missing cameras
        exps = list(exps)
        if len(exps) == 2:
            x = [any(e.match(f'*{n}*') for e in exps) for n in m]
            exps.insert(x.index(False), None)

        # combine the b, r, z channels together
        join_spec_channels(in_rss=list(exps), out_rss=bout_file, use_weights=False, **kwargs)
        log.info(f'Output combined camera file: {bout_file}')


def combine_spectrographs(tileid: int, mjd: int, expnum: int) -> fits.HDUList:
    """ Combine the spectrographs together for a given exposure

    For a given exposure, combines the three spectographs together
    into a single output lvm-CFrame file.  The input files are the
    ancillary camera-combined lvm-object-sp[id]-[expnum] files.

    Parameters
    ----------
    tileid : int
        The tileid of the observation
    mjd : int
        The MJD of the observation
    expnum : int
        The exposure number of the frames to combines

    Returns
    -------
    fits.HDUList
        the output FITS file
    """

    files = sorted(path.expand('lvm_anc', mjd=mjd, tileid=tileid, drpver=drpver,
                               kind='', camera='sp*', imagetype='object', expnum=expnum))

    if not files:
        log.error(f'No camera-combined files found for expnum: {expnum}')
        return

    if len(files) != 3:
        log.warning(f'Warning: Not all specids found for expnum: {expnum}')

    # construct output path
    cframe = path.full('lvm_frame', mjd=mjd, tileid=tileid, drpver=drpver,
                       kind='CFrame', expnum=expnum)

    # get the first header
    with fits.open(files[0]) as hdu:
        hdr = hdu[0].header.copy()

    # build the wavelength axis
    wcs = WCS(hdr)
    n_wave = hdr['NAXIS1']
    wl = wcs.spectral.all_pix2world(np.arange(n_wave), 0)[0]
    wave = fits.ImageHDU((wl * u.m).to(u.angstrom).value, name='WAVE')

    # get total number of fibers from the fibermap
    # do we use this to check the total output fiber number? should be the same?
    total_fibers = len(fibermap.data)

    # stack the data in the extensions
    flux_data = stack_ext(files, ext=0)
    err_data = stack_ext(files, ext='ERROR')
    mask_data = stack_ext(files, ext='BADPIX')
    fwhm_data = stack_ext(files, ext='INSTFWHM')

    # update the primary header
    hdr['SPEC'] = ', '.join([i.split('-')[2] for i in files])
    hdr['FILENAME'] = pathlib.Path(cframe).name
    hdr['DRPVER'] = drpver

    # remove the wcs from the primary header; add it to flux header
    [hdr.pop(i, None) for i in wcs.to_header().keys()]

    # create new hdr for flux extension
    newhdr = {'BUNIT': hdr.pop("BUNIT", None)}
    newhdr['BSCALE'] = hdr.pop("BSCALE", None)
    newhdr['BZERO'] = hdr.pop("BZERO", None)
    newhdr.update(wcs.to_header())

    # create the new FITS file
    prim = fits.PrimaryHDU(header=hdr)
    flux = fits.ImageHDU(flux_data, name='FLUX', header=fits.Header(newhdr))
    err = fits.ImageHDU(err_data, name='ERROR')
    mask = fits.ImageHDU(mask_data, name='MASK')
    fwhm = fits.ImageHDU(fwhm_data, name='FWHM')

    hdulist = fits.HDUList([prim, flux, err, mask, wave, fwhm, fibermap])

    # write out new file
    hdulist.writeto(cframe, overwrite=True)


def stack_ext(files: list, ext: Union[int, str] = 0) -> np.array:
    """ Stack the FITS data from a list of files

    Stack the FITS data for the given extension name or number,
    from the input list of files.  The output stack is in the order
    of the input list of files, i.e. for a list of sp1, sp2, sp3,
    the 0-index of the output array is the start of sp1.

    Parameters
    ----------
    files : list
        A list of files to stack
    ext : Union[int, str], optional
        The FITS extension name or number, by default 0

    Returns
    -------
    np.array
        The stacked data
    """
    new = []
    for i in files:
        with fits.open(i) as hdu:
            new.append(hdu[ext].data)
    return np.vstack(new)


@lru_cache
def read_fibermap(as_table: bool = None, as_hdu: bool = None,
                  filename: str = 'lvm_fiducial_fibermap.yaml') -> Union[pd.DataFrame, Table, fits.BinTableHDU]:
    """ Read the LVM fibermap

    Reads the LVM fibermap yaml file into a pandas
    DataFrame or Astropy Table or Astropy fits.BinTableHDU.

    Parameters
    ----------
    as_table : bool, optional
        If True, returns an Astropy Table, by default None
    as_hdu : bool, optional
        If True, returns an Astropy fits.BinTableHDU, by default None
    filename : str, optional
        Optional name of the fibermap file, by default "lvm_fiducial_fibermap.yaml"

    Returns
    -------
    Union[pd.DataFrame, Table, fits.BinTableHDU]
        the fibermap as a dataframe, table, or hdu
    """
    core_dir = os.getenv('LVMCORE_DIR')
    if not core_dir:
        raise ValueError("Environment variable LVMCORE_DIR not set. Set it or load lvmcore module file.")

    p = pathlib.Path(core_dir) / f'metrology/{filename}'
    if not p.is_file():
        log.warning("Cannot read fibermap from lvmcore.")
        return

    with open(p, 'r') as f:
        data = yaml.load(f, Loader=yaml.CSafeLoader)
        cols = [i['name'] for i in data['schema']]
        df = pd.DataFrame(data['fibers'], columns=cols)
        if as_table:
            return Table.from_pandas(df)
        if as_hdu:
            return fits.BinTableHDU(Table.from_pandas(df), name='SLITMAP')
        return df


fibermap = read_fibermap(as_hdu=True)


def select_fibers(specid: int = None, flag: str = 'SAIT') -> pd.DataFrame:
    """ Select fibers from the fibermap

    Select fibers from the fibermap dataframe. Use the flag keyword
    to set a predefined selection filter.  The default flag of SAIT
    selects on non-standard targets (targettype != "standard") and
    good fibers (fibstatus != 1), where good is both "good" (0) and "fibers
    with low throughput" (2).

    Parameters
    ----------
    specid : int, optional
        the spectrograph id, by default None
    flag : str, optional
        flag for setting a predefined seletion query, by default 'SAIT'

    Returns
    -------
    pd.DataFrame
        the fiber subset matching the query
    """
    df = read_fibermap()

    if flag == 'SAIT':
        query = 'targettype != "standard" & fibstatus != 1'
        if specid:
            query += f' & spectrographid == {specid}'

    return df.query(query)


def add_extension(hdu: Union[fits.ImageHDU, fits.BinTableHDU], filename: str):
    """ Add an astropy HDU to an existing FITS file

    _extended_summary_

    Parameters
    ----------
    hdu : Union[fits.ImageHDU, fits.BinTableHDU]
        the HDU to add
    filename : str
        the name of the file on disk

    Raises
    ------
    ValueError
        when the input hdu is not a valid image or table hdu
    ValueError
        when the input hdu does not a proper name
    """

    if not isinstance(hdu, (fits.ImageHDU, fits.BinTableHDU)):
        raise ValueError('Input hdu is not valid astropy FITS ImageHDU or BinTableHDU.')

    if not hdu.name:
        raise ValueError(f'Input hdu does not have a valid extension name: {hdu.name}. Cannot add.')

    with fits.open(filename, mode='update') as hdulist:
        if hdu.name not in hdulist:
            hdulist.append(hdu)
            hdulist.flush()
