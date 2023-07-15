#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
import yaml
import pandas as pd
from typing import Union
from functools import lru_cache

from astropy.io import fits
from astropy.table import Table
from lvmdrp.functions.imageMethod import (preproc_raw_frame, create_master_frame,
                                          detrend_frame, find_peaks_auto, trace_peaks,
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


def find_masters(flavor: str, camera: str) -> dict:
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
    matches = match_master_metadata(target_imagetyp=flavor, target_camera=camera)

    # construct the dict of filepaths
    files = dict.fromkeys(matches.keys())
    for key, val in matches.items():
        if val is None:
            continue
        files[key] = create_master_path(val)
    return files


def trace_fibers(in_file: str, camera: str, expnum: int, tileid: int, mjd: int):
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
    # TODO
    # check these output paths names; may need to change the tree paths to update the names
    # or maybe the "kind" keyword is changed to "mpeaks" or "mtrace"

    out_peaks = path.full("lvm_cal", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera,
                          expnum=expnum, kind='peaks', ext='fits')
    out_trace = path.full("lvm_cal", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera,
                          expnum=expnum, kind='trace', ext='fits')

    # check for parent dir existence
    if not pathlib.Path(out_peaks).parent.is_dir():
        pathlib.Path(out_peaks).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(out_trace):
        log.info('Trace file already exists.')
        return

    log.info('--- Running auto peak finder ---')
    kwargs = get_config_options('reduction_steps.find_peaks_auto')
    log.info(f'custom configuration parameters for find_peaks_auto: {repr(kwargs)}')
    find_peaks_auto(in_image=in_file, out_peaks=out_peaks, **kwargs)
    log.info(f'Output peak finder file: {out_peaks}')

    log.info('--- Tracing fiber peaks ---')
    kwargs = get_config_options('reduction_steps.trace_peaks')
    log.info(f'custom configuration parameters for trace_peaks: {repr(kwargs)}')
    trace_peaks(in_image=in_file, out_trace=out_trace, in_peaks=out_peaks, **kwargs)
    log.info(f'Output trace fiber peaks file: {out_trace}')

    # TODO
    # add new function to trace the width
    # output of this file goes into extract_spectra
    # trace widths only for full reductions
    if not config.get('quick'):
        pass


def find_file(kind, camera=None, mjd=None, tileid=None):
    files = sorted(path.expand('lvm_cal', kind=kind, drpver=drpver, mjd=mjd, tileid=tileid,
                   camera=camera, expnum='****', ext='fits'))
    if not files:
        log.warning(f"No {kind} files found for {tileid}, {mjd}, {camera}.  Discontinuing reduction.")
        return

    # pick the last one in the list, sorted by exposure number
    return files[-1]


def reduce_frame(filename: str, camera: str = None, mjd: int = None,
                 expnum: int = None, tileid: int = None,
                 flavor: str = None, **fkwargs):
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
    """
    # start logging for this mjd
    start_logging(mjd, tileid)

    log.info(f'--- Starting reduction of raw frame: {filename}')

    # set flavor
    flavor = flavor or fkwargs.get('imagetyp')
    flavor = 'fiberflat' if flavor == 'flat' else flavor

    # preprocess the frames
    log.info('--- Preprocessing raw frame ---')
    kwargs = get_config_options('reduction_steps.preproc_raw_frame', flavor)
    log.info(f'custom configuration parameters for preproc_raw_frame: {repr(kwargs)}')
    out_pre = path.full('lvm_anc', kind='p', imagetype=flavor, mjd=mjd, camera=camera,
                        drpver=drpver, expnum=expnum, tileid=tileid)
    # create the root dir if needed
    if not pathlib.Path(out_pre).parent.exists():
        pathlib.Path(out_pre).parent.mkdir(parents=True, exist_ok=True)

    preproc_raw_frame(filename, out_image=out_pre, **kwargs)

    # check master frames
    masters = find_masters(flavor, camera)
    mbias = masters.get('bias')
    mdark = masters.get('dark')
    mflat = masters.get('flat')
    mpixflat = masters.get('pixelflat')
    log.info(f'Using master bias: {mbias}')
    log.info(f'Using master dark: {mdark}')
    log.info(f'Using master flat: {mflat}')
    log.info(f'Using master pixel flat: {mpixflat}')

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
                  in_bias=mbias, in_dark=mdark, in_pixelflat=mflat, **kwargs)
    log.info(f'Output calibrated file: {out_cal}')

    # TODO
    # reduce individual frames for bias/darks/flats/arcs up to detrend
    # exit after indivi arcs/flats
    # reduce_set creates marc and mflat master frames

    # end reduction for bias and darks
    if flavor in {'bias', 'dark'}:
        return

    # TODO
    # add this extension also to the master flat file
    # add the fibermap to all flat and science files
    if flavor in {'fiberflat', 'flat', 'pixelflat', 'object', 'science'}:
        log.info('Adding slitmap extension')
        add_extension(fibermap, out_cal)

    # TODO
    # input to fiber tracing is the master flat file, change out_cal to the mflat file

    # TODO - mflat and marc redutions start here

    # fiber tracing
    if 'flat' in flavor:
        log.info('--- Running fiber trace ---')
        trace_fibers(out_cal, camera, expnum, tileid, mjd)

    # extract fiber spectra
    cal_file = path.full("lvm_anc", kind='c', imagetype=flavor, mjd=mjd, drpver=drpver,
                         camera=camera, tileid=tileid, expnum=expnum)
    xout_file = path.full("lvm_anc", kind='x', imagetype=flavor, mjd=mjd, drpver=drpver,
                          camera=camera, tileid=tileid, expnum=expnum)

    # find the fiber trace file
    trace_file = find_file('trace', mjd=mjd, tileid=tileid, camera=camera)
    if not trace_file:
        return

    # TODO
    # we want to extract the individual science frames, master flats and master arcs
    # input trace_file is the input master mtrace_file
    # input cal_file is either the indiv science or the master file name
    # output xout file is the same, lvm-xobject of indiv (has expnum), or lvm-xarc, or lvm-xfiberflat (no expnum)

    # reduce indiv flats, arcs - creates mflat, marc
    # reduce mflat, marc, mflat again (as like the original flat/arc)

    # perform the fiber extraction
    log.info('--- Extracting fiber spectra ---')
    kwargs = get_config_options('reduction_steps.extract_spectra')
    log.info(f'custom configuration parameters for extract_spectra: {repr(kwargs)}')
    extract_spectra(in_image=cal_file, out_rss=xout_file, in_trace=trace_file, **kwargs)
    log.info(f'Output extracted file: {xout_file}')

    # TODO
    # input to wavelength solution is the master arc file, change xout_file to the marc file
    # and change output paths wave/lsf to "mwave" and "mlsf"

    # determine the wavelength solution
    if flavor == 'arc':
        wave_file = path.full('lvm_cal', kind='wave', drpver=drpver, mjd=mjd, tileid=tileid,
                              camera=camera, expnum=expnum, ext='fits')
        lsf_file = path.full('lvm_cal', kind='lsf', drpver=drpver, mjd=mjd, tileid=tileid,
                             camera=camera, expnum=expnum, ext='fits')
        # line_ref = pathlib.Path(__file__).parent.parent / f"etc/lvm-neon_nist_{camera[0]}.txt"
        kwargs = get_config_options('reduction_steps.determine_wavesol')
        log.info('--- Determining wavelength solution ---')
        log.info(f'custom configuration parameters for determine_wave_solution: {repr(kwargs)}')
        determine_wavelength_solution(in_arc=xout_file, out_wave=wave_file, out_lsf=lsf_file,
                                      **kwargs)
        log.info(f'Output wave peak traceset file: {wave_file}')
        log.info(f'Output lsf traceset file: {lsf_file}')

    # TODO
    # same as the extract_spectra steps, indiv science and master flats/arcs
    # check the wave_file, and lsf_file names and wout_file names

    # perform wavelength calibration
    wave_file = find_file('wave', mjd=mjd, tileid=tileid, camera=camera)
    lsf_file = find_file('lsf', mjd=mjd, tileid=tileid, camera=camera)
    if not (wave_file and lsf_file):
        return
    wout_file = path.full("lvm_anc", kind='w', imagetype=flavor, mjd=mjd, drpver=drpver,
                          camera=camera, tileid=tileid, expnum=expnum)
    log.info('--- Creating pixel table ---')
    create_pixel_table(in_rss=xout_file, out_rss=wout_file, arc_wave=wave_file, arc_fwhm=lsf_file)
    log.info(f'Output calibrated wavelength file: {wout_file}')

    # set wavelength resample params
    CHANNEL_WL = {"b": (3600, 5930), "r": (5660, 7720), "z": (7470, 9800)}
    wave_range = CHANNEL_WL[camera[0]]

    # TODO
    # same as the extract_spectra steps, indiv science and master flats/arcs
    # check the hout_file lvm-hobject, lvm-harc, lvm-hflat (these are based on the masters)

    # resample onto a common wavelength
    hout_file = path.full("lvm_anc", kind='h', imagetype=flavor, mjd=mjd, drpver=drpver,
                          camera=camera, tileid=tileid, expnum=expnum)
    kwargs = get_config_options('reduction_steps.resample_wave', flavor)
    log.info('--- Resampling wavelength grid ---')
    log.info(f'custom configuration parameters for resample_wave: {repr(kwargs)}')
    resample_wavelength(in_rss=wout_file, out_rss=hout_file, start_wave=wave_range[0],
                        end_wave=wave_range[1], **kwargs)
    log.info(f'Output resampled wave file: {hout_file}')

    # write out RSS

    # perform quality checks


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


def reduce_set(frame: pd.DataFrame, settype: str = None, flavor: str = None):
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
    flavors = {'bias', 'dark'} if settype == 'precals' else {'arc', 'flat'}

    # if a flavor is set, only create those masters
    if flavor:
        flavors = {flavor}

    # create master frames
    for flavor in flavors:
        create_masters(flavor, frame)

    # build the master metadata cache ; always update it
    get_master_metadata(overwrite=True)

    # TODO
    # add step after master bias/darks creation to create
    # a new master pixel mask which is used in the preproc step of all indiv reductions
    # also input is master pixel flat when it is available
    # Alfredo to do this
    if settype == 'precals':
        # loop over set of cameras in frame
        # get names of the master files (find_masters)
        # pass master filenames into new function
        # add new function here
        pass


def run_drp(mjd: Union[int, str, list], bias: bool = False, dark: bool = False,
            skip_bd: bool = False, arc: bool = False, flat: bool = False,
            only_bd: bool = False, only_cal: bool = False, only_sci: bool = False,
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
            run_drp(mjd=mjd, bias=bias, dark=dark, skip_bd=skip_bd, arc=arc, flat=flat,
                    only_bd=only_bd, only_cal=only_cal, only_sci=only_sci, spec=spec, camera=camera,
                    expnum=expnum, quick=quick)
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
    cond = sub['imagetyp'].isin(['bias', 'dark'])
    precals = sub[cond]
    if len(precals) == 0 and not skip_bd:
        log.error(f'No biases or darks found for mjd {mjd}. Discontinuing reduction.')
        return
    precals = precals.sort_values(['expnum', 'camera'])

    if not skip_bd:
        # reduce biases / darks
        reduce_set(precals, settype="precals", flavor='bias')
        reduce_set(precals, settype="precals", flavor='dark')

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

    # sort the table by flat, arc, flat, science
    if not only_sci:
        sub = sort_cals(sub)

    # split into cals, and science
    cals = sub[~(sub['imagetyp'] == 'object')]
    sci = sub[sub['imagetyp'] == 'object']

    # reduce the flats/arcs
    if not only_sci:
        reduce_set(cals, settype='cals')

    # return if only calibration set
    if only_cal or flat or arc:
        return

    # reduce science files
    reduce_set(sci, settype='science')

    # TODO - check for single elements
    mjd = list(set(sub['mjd']))[0]
    tileid = list(set(sub['tileid']))[0]

    # perform camera combination
    for tileid, mjd in sub.groupby(['tileid', 'mjd']).groups.keys():
        combine_cameras(tileid, mjd, spec=1)
        combine_cameras(tileid, mjd, spec=2)
        combine_cameras(tileid, mjd, spec=3)

    # perform spectrograph combination
    # one file per exposure
    # needs to have fiber slit info in fits extension
    # output fed into sky

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


def sort_cals(df: pd.DataFrame) -> pd.DataFrame:
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
    ss = df.sort_values(['camera', 'expnum'])
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


def combine_cameras(tileid: int, mjd: int, spec: int = 1):
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
                         imagetype='object', expnum='****', kind='h', camera=f'*{spec}')
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
        join_spec_channels(in_rss=list(exps), out_rss=bout_file, **kwargs)
        log.info(f'Output combined camera file: {bout_file}')


@lru_cache
def read_fibermap(as_table: bool = None, as_hdu: bool = None) -> Union[pd.DataFrame, Table, fits.BinTableHDU]:
    """ Read the LVM fibermap

    Reads the LVM fibermap yaml file into a pandas
    DataFrame or Astropy Table or Astropy fits.BinTableHDU.

    Parameters
    ----------
    as_table : bool, optional
        If True, returns an Astropy Table, by default None
    as_hdu : bool, optional
        If True, returns an Astropy fits.BinTableHDU, by default None

    Returns
    -------
    Union[pd.DataFrame, Table, fits.BinTableHDU]
        the fibermap as a dataframe, table, or hdu
    """
    core_dir = os.getenv('LVMCORE_DIR')
    if not core_dir:
        raise ValueError("Environment variable LVMCORE_DIR not set. Set it or load lvmcore module file.")

    p = pathlib.Path(core_dir) / 'metrology/lvm_fiducial_fibermap.yaml'
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
