#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
import yaml
import shutil
import traceback
import pandas as pd
from typing import Union, List
from functools import lru_cache
from itertools import groupby
from pprint import pformat

from typing import Tuple


import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from lvmdrp.core.constants import CAMERAS, SPEC_CHANNELS, MASTERS_DIR
from lvmdrp.core.rss import RSS
from lvmdrp.functions.imageMethod import (preproc_raw_frame, create_master_frame,
                                          create_pixelmask, detrend_frame,
                                          add_astrometry, subtract_straylight,
                                          trace_peaks,
                                          extract_spectra)

from lvmdrp.functions.rssMethod import (determine_wavelength_solution, create_pixel_table, apply_fiberflat,
                                        resample_wavelength, shift_wave_skylines, join_spec_channels, stack_spectrographs)
from lvmdrp.functions.skyMethod import interpolate_sky, combine_skies, quick_sky_subtraction
from lvmdrp.core import fluxcal
from lvmdrp.functions.fluxCalMethod import fluxcal_standard_stars, fluxcal_sci_ifu_stars, apply_fluxcal
from lvmdrp.utils.metadata import (get_frames_metadata, get_master_metadata, extract_metadata,
                                   get_analog_groups, match_master_metadata, create_master_path,
                                   update_summary_file, convert_h5_to_fits)
from lvmdrp.utils.convert import tileid_grp
from lvmdrp.utils.paths import get_master_mjd, mjd_from_expnum, get_calib_paths, group_calib_paths
from lvmdrp.utils.timer import Timer

from lvmdrp import config, log, path, __version__ as drpver


def parse_expnums(expnum: Union[int, str, list, tuple]) -> Union[List, Tuple]:
    """Returns and exposure number list

    Parameters
    ----------
    expnum : int|str|list[int]|tuple[int]
        exposure numbers expression from which to parse an exposure number list

    Returns
    -------
    exps : list[int]
        a list of exposure numbers
    """
    if isinstance(expnum, str):
        start_exp, end_exp = expnum.split('-')
        start_exp = int(start_exp) if start_exp else None
        end_exp = int(end_exp) if end_exp else None

        if start_exp and end_exp:
            exps = list(range(start_exp, end_exp+1))
        else:
            exps = (start_exp, end_exp)
    elif isinstance(expnum, int):
        exps = [expnum]
    else:
        exps = list(expnum)

    return exps


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
    "11111/60115/ancillary/lvm-xobject-b1-00060115.fits" for science frames
    or "11111/60115/calib/lvm-xmarc-b1.fits" for the master arc frame.

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


def read_expfile(in_file: str) -> List[str]:
    """Reads a comma-separated or column of exposure numbers"""
    if not os.path.isfile(in_file):
        return []

    with open(in_file, "r") as expfile:
        expnums = expfile.readlines()

    if len(expnums) == 1 and "," in expnums[0]:
        expnums = expnums[0]
        expnums = [int(expnum) for expnum in expnums.split(",")]
    else:
        expnums = [int(expnum[:-1]) if "\n" in expnum else int(expnum) for expnum in expnums]

    return expnums

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
    tilegrp = tileid_grp(tileid)
    lpath = (os.path.join(os.getenv('LVM_SPECTRO_REDUX'),
             "{drpver}/{tilegrp}/{tileid}/{mjd}/lvm-drp-{tileid}-{mjd}.log"))
    logpath = lpath.format(drpver=drpver, mjd=mjd, tileid=tileid, tilegrp=tilegrp)
    logpath = pathlib.Path(logpath)

    # if logpath.exists():
    #     return

    if not logpath.parent.exists():
        logpath.parent.mkdir(parents=True, exist_ok=True)

    log.start_file_logger(logpath, rotating=False, with_json=True)
    return logpath


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


def build_supersky(tileid: int, mjd: int, expnum: int, imagetype: str) -> fits.BinTableHDU:
    """return super sky FITS table for a given exposure

    Parameters
    ----------
    tileid : int
        the sky tileid
    mjd : int
        the MJD of observation
    expnum : int
        the exposure number
    imagetype : str
        the image type

    Returns
    -------
    fits.BinTableHDU
        the super sky table
    """
    # select files for sky fibers
    fsci_paths = sorted(path.expand("lvm_anc", mjd=mjd, tileid=tileid, drpver=drpver,
                                kind="f", camera="*", imagetype=imagetype, expnum=expnum))

    fsci_paths_cam = groupby(fsci_paths, lambda x: x.split("-")[-2])
    sky_wave = []
    sky = []
    sky_error = []
    fiberidx = []
    spec = []
    telescope = []
    for cam, paths in fsci_paths_cam:
        specid = int(cam[-1])
        paths = sorted(list(paths))

        for sci_path in paths:

            # load flafielded camera frame
            fsci = RSS()
            fsci.loadFitsData(sci_path)

            # convert to density units if necessary
            if fsci._header["BUNIT"] == "electron":
                dlambda = np.diff(fsci._wave, axis=1, append=2*(fsci._wave[:, -1] - fsci._wave[:, -2])[:, None])
                fsci._data /= dlambda
                fsci._error /= dlambda
                fsci._header["BUNIT"] = "electron / Angstrom"

            # sky fiber selection
            slitmap = fsci._slitmap[fsci._slitmap["spectrographid"] == specid]
            select_skye = slitmap["telescope"] == "SkyE"
            select_skyw = slitmap["telescope"] == "SkyW"
            fiberidx_e = np.repeat(np.where(select_skye)[0][:, None], fsci._pixels.size, axis=1)
            fiberidx_w = np.repeat(np.where(select_skyw)[0][:, None], fsci._pixels.size, axis=1)

            # create super sky table
            nsam_e = np.sum(select_skye) * fsci._pixels.size
            nsam_w = np.sum(select_skyw) * fsci._pixels.size
            sky_wave.extend(fsci._wave[select_skye].ravel().tolist() + fsci._wave[select_skyw].ravel().tolist())
            sky.extend(fsci._data[select_skye].ravel().tolist() + fsci._data[select_skyw].ravel().tolist())
            sky_error.extend(fsci._error[select_skye].ravel().tolist() + fsci._error[select_skyw].ravel().tolist())
            fiberidx.extend(fiberidx_e.ravel().tolist() + fiberidx_w.ravel().tolist())
            spec.extend([specid] * (nsam_e + nsam_w))
            telescope.extend(["east"] * nsam_e + ["west"] * nsam_w)
    sort_idx = np.argsort(sky_wave)
    wave_c = fits.Column(name="wave", array=np.array(sky_wave)[sort_idx], unit="Angstrom", format="E")
    sky_c = fits.Column(name="sky", array=np.array(sky)[sort_idx], unit=fsci._header["BUNIT"], format="E")
    sky_error_c = fits.Column(name="sky_error", array=np.array(sky_error)[sort_idx], unit=fsci._header["BUNIT"], format="E")
    fiberidx_c = fits.Column(name="fiberidx", array=np.array(fiberidx)[sort_idx], format="J")
    spec_c = fits.Column(name="spectrographid", array=np.array(spec)[sort_idx], format="J")
    telescope_c = fits.Column(name="telescope", array=np.array(telescope)[sort_idx], format="4A")
    supersky = fits.BinTableHDU.from_columns([wave_c, sky_c, sky_error_c, fiberidx_c, spec_c, telescope_c], name="SUPERSKY")

    return supersky


def combine_channels(tileid: int, mjd: int, expnum: int, imagetype: str):
    """ Combine the spectrograph channels together

    For a given exposure, combines the three spectograph channels together
    into a single output lvm-CFrame file.  The input files are the
    ancillary spectrograph-combined lvm-object-[channel]-[expnum] files.

    Parameters
    ----------
    tileid : int
        the sky tileid
    mjd : int
        the MJD of observation
    expnum : int
        the exposure number
    imagetype : str
        the image type
    """

    # find all the h object files
    files = path.expand('lvm_anc', mjd=mjd, tileid=tileid, drpver=drpver,
                         imagetype=imagetype, expnum=expnum, kind='', camera='*')
    # filter out old lvm-object-sp?-*.fits files
    files = sorted([f for f in files if not os.path.basename(f).startswith(f"lvm-{imagetype}-sp")], key=_parse_expnum_cam)

    cframe_path = path.full("lvm_frame", mjd=mjd, drpver=drpver, tileid=tileid, expnum=expnum, kind='CFrame')

    log.info(f'combining channels for {expnum = }')
    kwargs = get_config_options('reduction_steps.combine_channels')
    log.info(f'custom configuration parameters for combine cameras: {repr(kwargs)}')

    # combine the b, r, z channels together
    rss_comb = join_spec_channels(in_rsss=files, out_rss=None, use_weights=True, **kwargs)

    # build the wavelength axis
    hdr = rss_comb._header
    wcs = WCS(hdr)
    n_wave = hdr['NAXIS1']
    wl = wcs.spectral.all_pix2world(np.arange(n_wave), 0)[0].astype("float32")
    wave = fits.ImageHDU((wl * u.m).to(u.angstrom).value, name='WAVE')

    # update the primary header
    hdr['SPEC'] = ', '.join([f"sp{specid+1}" for specid in range(3)])
    hdr['FILENAME'] = pathlib.Path(cframe_path).name
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
    flux = fits.ImageHDU(rss_comb._data, name='FLUX', header=fits.Header(newhdr))
    err = fits.ImageHDU(rss_comb._error, name='ERROR')
    mask = fits.ImageHDU(rss_comb._mask.astype("uint8"), name='MASK')
    fwhm = fits.ImageHDU(rss_comb._lsf, name='FWHM')
    sky = fits.ImageHDU(rss_comb._sky, name="SKY")
    sky_error = fits.ImageHDU(rss_comb._sky_error, name="SKY_ERROR")

    # build super sky
    supersky = build_supersky(tileid, mjd, expnum, imagetype)

    # write out new file
    log.info(f'writing output file in {os.path.basename(cframe_path)}')
    hdulist = fits.HDUList([prim, flux, err, mask, wave, fwhm, sky, sky_error, supersky, fibermap])
    hdulist.writeto(cframe_path, overwrite=True)


def combine_spectrographs(tileid: int, mjd: int, channel: str, expnum: int, imagetype: str) -> RSS:
    """ Combine the spectrographs together for a given exposure

    For a given exposure, combines the three spectographs together into a
    single output lvm-object-[channel]-[expnum] file. The input files are the
    ancillary rectified frames lvm-hobject-[channel]*-[expnum] files.

    Parameters
    ----------
    tileid : int
        The tileid of the observation
    mjd : int
        The MJD of the observation
    channel : str
        The channel of the spectrograph, e.g. b, r, z
    expnum : int
        The exposure number of the frames to combines
    imagetype : str
        The imagetype of the frames to combine

    Returns
    -------
    RSS
        The combined RSS object
    """

    hsci_paths = sorted(path.expand('lvm_anc', mjd=mjd, tileid=tileid, drpver=drpver,
                               kind='h', camera=f'{channel}[123]', imagetype=imagetype, expnum=expnum))

    if not hsci_paths:
        log.error(f'no rectified frames found for {expnum = }, {channel = }')
        return

    if len(hsci_paths) != 3:
        log.warning(f'not all spectrographs found for {expnum = }, {channel = }')

    # construct output path
    frame_path = path.full('lvm_anc', mjd=mjd, tileid=tileid, drpver=drpver,
                           kind='', camera=channel, imagetype=imagetype, expnum=expnum)

    # combine RSS files along fiber ID direction
    return stack_spectrographs(hsci_paths, frame_path)


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
        units = [u.Unit(i['unit']) if i['unit'] is not None else None for i in data['schema']]

        # define dtypes for Table and Numpy arrays because these two can't seem to talk to each other
        tb_dtypes = [i['dtype'] for i in data['schema']]
        np_dtypes = list(zip(cols, [d if d != 'str' else 'object' for d in tb_dtypes]))

        # create table with units and correct types
        table = Table(np.asarray([tuple(d) for d in data['fibers']], dtype=np_dtypes), units=units, dtype=tb_dtypes)
        if as_table:
            return table
        if as_hdu:
            return fits.BinTableHDU(table, name='SLITMAP')
        return table.to_pandas()


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


def _yield_dir(root: pathlib.Path, mjd: int) -> pathlib.Path:
    """ Iteratively yield a pathlib directory

    Parameters
    ----------
    root : pathlib.Path
        the top-level path
    mjd : int
        the MJD to look for

    Yields
    ------
    Iterator[pathlib.Path]
        the pathlib.Path
    """
    for item in root.iterdir():
        if item.stem == str(mjd):
            yield item
        if item.is_dir():
            yield from _yield_dir(item, mjd)


def should_run(mjd: int) -> bool:
    """ Check if the DRP should be run

    Checks to see if the DRP should be run for the
    given MJD.  Checks if the data transfer has completed
    and that no pipeline run has started yet.

    Parameters
    ----------
    mjd : int
        the MJD

    Returns
    -------
    bool
        Flag if the pipeline should be run
    """

    # not transferred yet
    done = pathlib.Path(os.getenv("LCO_STAGING_DATA")) / f'log/lvm/{mjd}/transfer-{mjd}.done'
    if not done.exists() or not done.is_file():
        # data not transferred yet, skip DRP running
        log.warning(f'Data transfer not yet complete for MJD {mjd}.')
        return False

    # check for MJD directory and any files in it
    # if no directory or no raw_metadata file in it, we run the DRP
    root = pathlib.Path(os.getenv("LVM_SPECTRO_REDUX")) / f'{drpver}'
    mjddir = list(_yield_dir(root, mjd))
    no_files = not any(mjddir[0].glob('raw_meta*')) if mjddir else True
    if not no_files:
        log.info(f"DRP for mjd {mjd} already running.")
    return no_files


def check_daily_mjd(test: bool = False, with_cals: bool = False):
    """ Check for daily MJD run

    Get the MJD for the current datetime and check if
    we should run the DRP or not.  If so, start the DRP
    for the given MJD.

    Parameters
    ----------
    test : bool, optional
        Flag to test the check without running the DRP, by default False
    with_cals: bool, optional
        Flag to turn on reduction of the individual calibration files
    """
    # get current MJD
    t = Time.now()
    mjd = int(t.mjd)
    log.info(f'It is {t.to_string()}.  The MJD is {int(t.mjd)}.')

    # check if we should run the DRP
    if should_run(mjd):
        log.info(f"Running DRP for mjd {mjd}")
        if not test:
            run_drp(mjd, with_cals=with_cals)


def create_status_file(tileid: int, mjd: int, status: str = 'started'):
    """ Create a DRP status file

    Create a DRP status file for the given tile_id, MJD.

    Parameters
    ----------
    tileid : int
        the tile iD
    mjd : int
        the MJD
    status : str, optional
        the DRP status, by default 'started'
    """
    tilegrp = tileid_grp(tileid)
    root = pathlib.Path(os.getenv("LVM_SPECTRO_REDUX")) / f'{drpver}/{tilegrp}/{tileid}/logs'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'lvm-drp-{tileid}-{mjd}.{status}'
    path.touch()


def remove_status_file(tileid: int, mjd: int, remove_all: bool = False):
    """ Remove a DRP status file

    Remove a DRP status file for the given tile_id, MJD, or
    optionally remove all status files.

    Parameters
    ----------
    tileid : int
        the tile iD
    mjd : int
        the MJD
    remove_all : bool, optional
        Flag to remove all status files, by default False
    """
    tilegrp = tileid_grp(tileid)
    root = pathlib.Path(os.getenv("LVM_SPECTRO_REDUX")) / f'{drpver}/{tilegrp}/{tileid}/logs'

    if remove_all:
        shutil.rmtree(root)
        return

    files = root.rglob(f'lvm-drp-{tileid}-{mjd}.*')
    for file in files:
        file.unlink()


def status_file_exists(tileid: int, mjd: int, status: str = 'started') -> bool:
    """ Check if a status file exists

    Parameters
    ----------
    tileid : int
        the tile iD
    mjd : int
        the MJD
    status : str, optional
        the DRP status, by default 'started'

    Returns
    -------
    bool
        Flag if the file exists
    """
    tilegrp = tileid_grp(tileid)
    root = pathlib.Path(os.getenv("LVM_SPECTRO_REDUX")) / f'{drpver}/{tilegrp}/{tileid}/logs'
    path = root / f'lvm-drp-{tileid}-{mjd}.{status}'
    return path.exists()


def update_error_file(tileid: int, mjd: int, expnum: int, error: str,
                      reset: bool = False):
    """ Update the DRP error file

    Appends to the "drp_errors.txt" file whenever
    there is an error during a reduction.

    Parameters
    ----------
    tileid : int
        the tile id
    mjd : int
        the MJD
    expnum : int
        the exposure number
    error : str
        the traceback
    reset : bool, optional
        Flag to reset the text file, by default False
    """

    path = pathlib.Path(os.getenv("LVM_SPECTRO_REDUX")) / f'{drpver}' / 'drp_errors.txt'
    path.parent.mkdir(parents=True, exist_ok=True)

    if reset:
        path.unlink()
        return

    with open(path, '+a') as f:
        f.write(f'ERROR on tileid, mjd, exposure: {tileid}, {mjd}, {expnum}\n')
        f.write(error)
        f.write('\n')


def reduce_2d(mjd, calibrations, expnums=None, exptime=None, cameras=CAMERAS,
              replace_with_nan=True, assume_imagetyp=None, reject_cr=True,
              add_astro=True, sub_straylight=True, parallel_run=1,
              skip_done=True, keep_ancillary=False):
    """Preprocess and detrend a list of 2D frames

    Given a set of MJDs and (optionally) exposure numbers, preprocess detrends
    and optionally fits and subtracts the stray light field from the 2D frames.
    This routine will store the preprocessed, detrended and
    straylight-subtracted frames in the corresponding calibration directory in
    the `masters_mjd` or by default in the smallest MJD in `mjds`.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    calibrations : dict[str, dict[str, str]]
        Paths to calibrations to use, including bias and pixel masks and flats
    expnums : list
        List of exposure numbers to reduce
    exptime : int
        Exposure time to filter by
    cameras : list
        List of cameras to filter by
    replace_with_nan : bool
        Replace rejected pixels with NaN
    assume_imagetyp : str
        Assume the given imagetyp for all frames
    reject_cr : bool
        Reject cosmic rays
    add_astro : bool, optional
        Add astrometric solution to detrended frames, by default True
    sub_straylight : bool, optional
        Subtract straylight from detrended frames, by default True
    parallel_run : int, optional
        Parallel run of fiber masking in straylight subtraction, by default 1 (single core)
    counts_threshold : int
        Minimum count level to consider when tracing centroids, defaults to 500
    poly_deg_cent : int
        Degree of the polynomial to fit to the centroids, by default 4
    skip_done : bool
        Skip pipeline steps that have already been done
    keep_ancillary : bool
        Keep ancillary files, by default False
    """

    frames = get_frames_metadata(mjd)
    if expnums is not None:
        frames.query("expnum in @expnums", inplace=True)
    if exptime is not None:
        frames.query("exptime == @exptime", inplace=True)
    if cameras:
        frames.query("camera in @cameras", inplace=True)
    frames.sort_values(["camera"], inplace=True)

    # preprocess and detrend frames
    for frame in frames.to_dict("records"):
        camera = frame["camera"]

        # assume given image type
        imagetyp = assume_imagetyp or frame["imagetyp"]

        # get master frames paths
        mpixmask_path = calibrations["pixmask"][camera]
        mpixflat_path = calibrations["pixflat"][camera]
        mbias_path = calibrations["bias"][camera]
        mtrace_path = calibrations["trace"][camera]

        # log the master frames
        log.info(f'Using pixel mask: {mpixmask_path}')
        log.info(f'Using bias: {mbias_path}')
        log.info(f'Using pixel flat: {mpixflat_path}')

        rframe_path = path.full("lvm_raw", camspec=frame["camera"], **frame)
        eframe_path = path.full("lvm_anc", drpver=drpver, kind="e", imagetype=imagetyp, **frame)
        frame_path = eframe_path if os.path.exists(eframe_path) else rframe_path
        pframe_path = path.full("lvm_anc", drpver=drpver, kind="p", imagetype=imagetyp, **frame)
        lframe_path = path.full("lvm_anc", drpver=drpver, kind="l", imagetype=imagetyp, **frame)
        lstr_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype="stray", **frame)

        # define agc coadd path
        agcsci_path = path.full('lvm_agcam_coadd', mjd=mjd, specframe=frame["expnum"], tel='sci')
        agcskye_path = path.full('lvm_agcam_coadd', mjd=mjd, specframe=frame["expnum"], tel='skye')
        agcskyw_path = path.full('lvm_agcam_coadd', mjd=mjd, specframe=frame["expnum"], tel='skyw')

        # bypass creation of detrended frame in case of imagetyp=bias
        if imagetyp != "bias":
            dframe_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=imagetyp, **frame)
        else:
            dframe_path = pframe_path

        os.makedirs(os.path.dirname(dframe_path), exist_ok=True)
        final_2d_dp = dframe_path if not sub_straylight else lframe_path
        if skip_done and os.path.isfile(final_2d_dp):
            log.info(f"skipping {final_2d_dp}, file already exist")
        else:
            with Timer(name='Preproc '+pframe_path, logger=log.info):
                preproc_raw_frame(in_image=frame_path, out_image=pframe_path,
                                  in_mask=mpixmask_path, replace_with_nan=replace_with_nan, assume_imagetyp=assume_imagetyp)
            with Timer(name='Detrend '+dframe_path, logger=log.info):
                detrend_frame(in_image=pframe_path, out_image=dframe_path,
                            in_bias=mbias_path,
                            in_pixelflat=mpixflat_path,
                            replace_with_nan=replace_with_nan,
                            reject_cr=reject_cr,
                            in_slitmap=fibermap if imagetyp in {"flat", "arc", "object"} else None)

            # add astrometry to frame
            if add_astro:
                with Timer(name='Astrometry '+dframe_path, logger=log.info):
                    add_astrometry(in_image=dframe_path, out_image=dframe_path, in_agcsci_image=agcsci_path, in_agcskye_image=agcskye_path, in_agcskyw_image=agcskyw_path)

            # subtract straylight
            if sub_straylight:
                with Timer(name='Straylight '+lframe_path, logger=log.info):
                    subtract_straylight(in_image=dframe_path, out_image=lframe_path, out_stray=lstr_path,
                                        in_cent_trace=mtrace_path, x_bins=40, select_nrows=(10,10), use_weights=True,
                                        aperture=11, nsigma=2.0, smoothing=None, median_box=101, parallel=parallel_run)


def reduce_1d(mjd, calibrations, expnums=None, replace_with_nan=True, sub_straylight=True, skip_done=True, keep_ancillary=False):

    frames = get_frames_metadata(mjd)
    if expnums is not None:
        frames.query("expnum in @expnums", inplace=True)
    frames.sort_values(["expnum", "camera"], inplace=True)

    for _, sci in frames.iterrows():
        if sub_straylight:
            dframe_path = path.full("lvm_anc", drpver=drpver, kind="l", imagetype=sci["imagetyp"], **sci)
        else:
            dframe_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=sci["imagetyp"], **sci)
        xframe_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=sci["imagetyp"], **sci)
        os.makedirs(os.path.dirname(xframe_path), exist_ok=True)

        # define calibration frames paths
        mtrace_path = calibrations["trace"][sci.camera]
        mwidth_path = calibrations["width"][sci.camera]
        mmodel_path = calibrations["model"][sci.camera]

        # extract 1d spectra
        if skip_done and os.path.isfile(xframe_path):
            continue
        else:
            extract_spectra(in_image=dframe_path, out_rss=xframe_path, in_trace=mtrace_path, in_fwhm=mwidth_path, in_model=mmodel_path, method="optimal", parallel=1)

    frames = frames.drop_duplicates(subset=["expnum"])
    for _, sci in frames.iterrows():
        mwave_groups = group_calib_paths(calibrations["wave"])
        mlsf_groups = group_calib_paths(calibrations["lsf"])
        for channel in "brz":
            sci["camera"] = f"{channel}[123]"
            xframe_paths = sorted(path.expand('lvm_anc', drpver=drpver, kind='x', imagetype=sci["imagetyp"], **sci))
            sci["camera"] = channel
            xframe_path = path.full('lvm_anc', drpver=drpver, kind='x', imagetype=sci["imagetyp"], **sci)
            wframe_path = path.full('lvm_anc', drpver=drpver, kind='w', imagetype=sci["imagetyp"], **sci)
            mwave_paths = mwave_groups[channel]
            mlsf_paths = mlsf_groups[channel]

            # stack spectrographs
            if skip_done and os.path.isfile(wframe_path):
                continue
            else:
                stack_spectrographs(in_rsss=xframe_paths, out_rss=xframe_path)
                if not os.path.exists(xframe_path):
                    log.error(f'No stacked file found: {xframe_path}. Skipping remaining pipeline.')
                    continue

                # wavelength calibrate
                create_pixel_table(in_rss=xframe_path, out_rss=wframe_path, in_waves=mwave_paths, in_lsfs=mlsf_paths)


def science_reduction(expnum: int,
                      use_longterm_cals: bool = True, from_sandbox: bool = True,
                      sky_weights: Tuple[float, float] = None,
                      fluxcal_method: str = 'STD',
                      ncpus: int = None,
                      aperture_extraction: bool = False,
                      clean_ancillary: bool = False,
                      skip_2d: bool = False,
                      skip_1d: bool = False,
                      skip_post_1d: bool = False,
                      skip_drpall: bool = False,
                      debug_mode: bool = False,
                      force_run: bool = False) -> None:
    """ Run the science reduction for a given exposure number.
    """

    if debug_mode:
        aperture_extraction = True
        clean_ancillary = False
        reject_cr = False
    else:
        reject_cr = True

    # validate parameters
    if sky_weights is not None:
        if len(sky_weights) != 2:
            log.error("sky weights must be a tuple of two floats")
            return
        elif any([w < 0.0 for w in sky_weights]):
            log.error("sky weights must be positive")
            return
        elif sum(sky_weights) == 0.0:
            log.error("sum of sky weights must be non-zero")
            return

    # define extraction method
    parallel_run = 1 if ncpus is None else ncpus
    extraction_method = "aperture" if aperture_extraction else "optimal"

    # get target frames metadata or extract if it doesn't exist
    sci_mjd = mjd_from_expnum(expnum)[0]
    sci_metadata = get_frames_metadata(mjd=sci_mjd)
    sci_metadata.query("expnum == @expnum", inplace=True)
    sci_metadata.sort_values("expnum", ascending=False, inplace=True)
    if not force_run:
        try:
            sci_metadata.query("qaqual == 'GOOD'", inplace=True)
        except KeyError:
            log.error("error while getting qaqual field in metadata.")
            log.error(f"Please try running `drp metadata regenerate -m {sci_mjd}` before trying reducing your exposure again.")
            return
        if sci_metadata.empty:
            log.error(f"exposure {expnum = } was flagged as 'BAD' by the raw data quality pipeline")
            return

    # define general metadata
    sci_tileid = sci_metadata["tileid"].unique()[0]
    sci_mjd = sci_metadata["mjd"].unique()[0]
    sci_expnum = sci_metadata["expnum"].unique()[0]
    sci_imagetyp = sci_metadata["imagetyp"].unique()[0]

    log.info(f"Reducing MJD {sci_mjd}, exposure {expnum}, tile_id {sci_tileid} ... ")

    # overwrite fiducial masters dir
    calibs, cals_mjd = get_calib_paths(
        mjd=sci_mjd,
        version=drpver,
        longterm_cals=use_longterm_cals,
        from_sanbox=from_sandbox,
        flavors=["pixmask", "pixflat", "bias", "trace", "width", "model", "wave", "lsf", "fiberflat_twilight"],
        return_mjd=True)

    log.info(f"calibrations parameters: {cals_mjd = }, {use_longterm_cals = }, {from_sandbox = }")
    for r in pformat(calibs).split("\n"):
        log.info(r)

    # make sure only one exposure number is being reduced
    # sci_metadata.query("expnum == @sci_expnum", inplace=True)
    sci_metadata.sort_values("camera", inplace=True)

    # detrend science exposure
    log.info(f"--- Starting science reduction for tile {sci_tileid} at MJD {sci_mjd} with exposure number {sci_expnum}")
    if skip_2d:
        log.info("skipping 2D reduction")
    else:
        with Timer(name='Reduce2d', logger=log.info):
            reduce_2d(mjd=sci_mjd, calibrations=calibs, expnums=[sci_expnum], reject_cr=reject_cr, skip_done=False)

    # run reduction loop for each science camera exposure
    if skip_1d:
        log.info("skipping 1D reduction")
    else:
        for sci in sci_metadata.to_dict("records"):
            # define science camera
            sci_camera = sci["camera"]

            dsci_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=sci["imagetyp"], **sci)
            xsci_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=sci["imagetyp"], **sci)
            wsci_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype=sci["imagetyp"], **sci)
            ssci_path = path.full("lvm_anc", drpver=drpver, kind="s", imagetype=sci["imagetyp"], **sci)
            hsci_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype=sci["imagetyp"], **sci)
            lsci_path = path.full("lvm_anc", drpver=drpver, kind="l", imagetype=sci["imagetyp"], **sci)
            os.makedirs(os.path.dirname(hsci_path), exist_ok=True)

            # define science product paths
            frame_path = path.full("lvm_frame", drpver=drpver, tileid=sci_tileid, mjd=sci_mjd, expnum=sci_expnum, kind=f"Frame-{sci_camera}")

            # define calibration frames paths
            mtrace_path = calibs["trace"][sci_camera]
            mwidth_path = calibs["width"][sci_camera]
            mmodel_path = calibs["model"][sci_camera]

            # extract 1d spectra
            with Timer(name='Extract '+xsci_path, logger=log.info):
                extract_spectra(in_image=lsci_path, out_rss=xsci_path, in_trace=mtrace_path, in_fwhm=mwidth_path,
                                in_model=mmodel_path, method=extraction_method, parallel=parallel_run)

    # per channel reduction
    cframe_path = path.full("lvm_frame", drpver=drpver, tileid=sci_tileid, mjd=sci_mjd, expnum=sci_expnum, kind='CFrame')
    sframe_path = path.full("lvm_frame", mjd=sci_mjd, drpver=drpver, tileid=sci_tileid, expnum=sci_expnum, kind='SFrame')
    if skip_post_1d:
        log.info("skipping post 1D reduction")
    else:
        mwave_groups = group_calib_paths(calibs["wave"])
        mlsf_groups = group_calib_paths(calibs["lsf"])
        for channel in "brz":
            xsci_paths = sorted(path.expand('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                                            kind='x', camera=f'{channel}[123]', imagetype=sci_imagetyp, expnum=expnum))
            xsci_path = path.full('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                                kind='x', camera=channel, imagetype=sci_imagetyp, expnum=expnum)
            wsci_path = path.full('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                                kind='w', camera=channel, imagetype=sci_imagetyp, expnum=expnum)
            mwave_paths = mwave_groups[channel]
            mlsf_paths = mlsf_groups[channel]
            mflat_path = calibs["fiberflat_twilight"][channel]

            frame_path = path.full('lvm_frame', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver, expnum=sci_expnum, kind=f'Frame-{channel}')
            ssci_path = path.full('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                                kind='s', camera=channel, imagetype=sci_imagetyp, expnum=expnum)
            hsci_path = path.full('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                                kind='h', camera=channel, imagetype=sci_imagetyp, expnum=expnum)

            # stack spectrographs
            with Timer(name='Stack Spectrographs '+xsci_path, logger=log.info):
                stack_spectrographs(in_rsss=xsci_paths, out_rss=xsci_path)
            if not os.path.exists(xsci_path):
                log.error(f'No stacked file found: {xsci_path}. Skipping remaining pipeline.')
                continue

            # wavelength calibrate
            with Timer(name='Wavelengths '+wsci_path, logger=log.info):
                create_pixel_table(in_rss=xsci_path, out_rss=wsci_path, in_waves=mwave_paths, in_lsfs=mlsf_paths)

            # apply fiberflat correction
            with Timer(name='Fiberflat '+frame_path, logger=log.info):
                apply_fiberflat(in_rss=wsci_path, out_frame=frame_path, in_flat=mflat_path)

            # correct thermal shift in wavelength direction
            with Timer(name='Thermal Shifts '+frame_path, logger=log.info):
                shift_wave_skylines(in_frame=frame_path, out_frame=frame_path)

            # interpolate sky fibers
            with Timer(name='Interpolate Sky '+ssci_path, logger=log.info):
                interpolate_sky(in_frame=frame_path, out_rss=ssci_path)

            # combine sky telescopes
            with Timer(name='Combine Sky '+ssci_path, logger=log.info):
                combine_skies(in_rss=ssci_path, out_rss=ssci_path, sky_weights=sky_weights)

            # resample wavelength into uniform grid along fiber IDs for science and sky fibers
            with Timer(name='Resample '+hsci_path, logger=log.info):
                resample_wavelength(in_rss=ssci_path,  out_rss=hsci_path, wave_range=SPEC_CHANNELS[channel], wave_disp=0.5, convert_to_density=True)

            # use resampled frames for flux calibration in each camera, using standard stars observed in the spec telescope
            #  and field stars found in the sci ifu
            with Timer(name='Fluxcal '+hsci_path, logger=log.info):
                fluxcal_standard_stars(hsci_path, GAIA_CACHE_DIR=MASTERS_DIR+'/gaia_cache')
                fluxcal_sci_ifu_stars(hsci_path, GAIA_CACHE_DIR=MASTERS_DIR+'/gaia_cache')

                # flux-calibrate each channel
                fframe_path = path.full("lvm_frame", mjd=sci_mjd, drpver=drpver, tileid=sci_tileid, expnum=sci_expnum, kind=f'FFrame-{channel}')
                apply_fluxcal(in_rss=hsci_path, out_fframe=fframe_path, method=fluxcal_method)

        # stitch channels
        fframe_paths = sorted(path.expand('lvm_frame', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver, kind='FFrame-?', expnum=sci_expnum))
        if len(fframe_paths) == 0:
            log.error('No fframe files found.  Cannot join spectrograph channels. Exiting pipeline.')
            return

        with Timer(name='Join Channels '+cframe_path, logger=log.info):
            join_spec_channels(in_fframes=fframe_paths, out_cframe=cframe_path, use_weights=True)

        # sky subtraction
        with Timer(name='QSky '+sframe_path, logger=log.info):
            quick_sky_subtraction(in_cframe=cframe_path, out_sframe=sframe_path)

    if skip_drpall:
        log.info("skipping create/update drpall summary file")
    else:
        # update the drpall summary file
        with Timer(name='DRPAll '+sframe_path, logger=log.info):
            log.info('Updating the drpall summary file')
            update_summary_file(sframe_path, tileid=sci_tileid, mjd=sci_mjd, expnum=sci_expnum, master_mjd=cals_mjd)

    # clean ancillary folder
    if clean_ancillary:
        ancillary_dir = os.path.dirname(dsci_path)
        qa_dir = os.path.join(ancillary_dir, "qa")
        log.info(f"removing ancillary files at {qa_dir}")
        if os.path.isdir(ancillary_dir):
            ancillary_paths = [os.path.join(ancillary_dir,p) for p in os.listdir(ancillary_dir) if str(sci_expnum) in p]
            qa_paths = [os.path.join(qa_dir,p) for p in os.listdir(qa_dir) if str(sci_expnum) in p]
            for ancillary_path in ancillary_paths:
                try:
                    os.remove(ancillary_path)
                except Exception as e:
                    log.warning(f"error while removing {ancillary_path}: {e}")
            for qa_path in qa_paths:
                try:
                    os.remove(qa_path)
                except Exception as e:
                    log.warning(f"error while removing {qa_path}: {e}")
            if len(os.listdir(qa_dir)) == 0:
                try:
                    shutil.rmtree(qa_dir)
                except Exception as e:
                    log.warning(f"error while removing {qa_dir}: {e}")
            if len(os.listdir(ancillary_dir)) == 0:
                try:
                    shutil.rmtree(ancillary_dir)
                except Exception as e:
                    log.warning(f"error while removing {ancillary_dir}: {e}")


def run_drp(mjd: Union[int, str, list], expnum: Union[int, str, list] = None,
            with_cals: bool = False, no_sci: bool = False,
            fluxcal_method: str = 'STD',
            skip_2d: bool = False, skip_1d: bool = False, skip_post_1d: bool = False, skip_drpall: bool = False,
            use_nightly_cals: bool = False, use_untagged_cals: bool = False,
            clean_ancillary: bool = False, debug_mode: bool = False, force_run: bool = False):
    """ Run the quick DRP

    Run the quick DRP for an MJD, or a range of MJDs. Reduces
    science frames with the function ``science_reduction``.
    Optionally can set flags to attempt reduction of the individual calibration
    frames in the MJD up through detrending, or to turn off science frame
    reduction.

    Parameters
    ----------
    mjd : Union[int, str, list]
        the MJD to reduce
    expnum : Union[int, str, list], optional
        the exposure numbers to reduce, by default None
    with_cals : bool, optional
        Flag to reduce individual calibration files, by default False
    no_sci : bool, optional
        Flag to turn off science frame reduction, by default False
    fluxcal_method : str, optional
        'NONE' or 'STD' for standard stars, 'SCI' for field stars in science IFU
    skip_2d : bool, optional
        Skip preprocessing and detrending, by default False
    skip_1d : bool, optional
        Skip astrometry, straylight subtraction and extraction, by default False
    skip_post_1d : bool, optional
        Skip wavelength calibration, flatfielding, sky processing and flux calibration
    skip_drpall : bool, optional
        Skip create/update drpall summary file
    use_nightly_cals : bool, optional
        Use nightly calibrations, by default False
    use_untagged_cals : bool, optional
        Use untagged (not from sandbox) calibrations, by default False
    clean_ancillary : bool, optional
        Flag to remove the ancillary paths, by default False
    debug_mode : bool, optional
        Flag to run in debug mode, by default False
    force_run : bool, optional
        Flag to force reductions even if the data was flagged as BAD by the QC pipeline, by default False
    """
    # # write the drp parameter configuration
    # write_config_file()

    if mjd is None:
        # parse expnums and get MJDs
        mjds = mjd_from_expnum(expnum)
    else:
        # parse the input MJD and loop over all reductions
        mjds = parse_mjds(mjd)

    if isinstance(mjds, list):
        for mjd in mjds:
            run_drp(mjd=mjd, expnum=expnum, with_cals=with_cals, no_sci=no_sci,
                    fluxcal_method=fluxcal_method,
                    skip_2d=skip_2d,
                    skip_1d=skip_1d,
                    skip_post_1d=skip_post_1d,
                    skip_drpall=skip_drpall,
                    clean_ancillary=clean_ancillary,
                    use_nightly_cals=use_nightly_cals,
                    use_untagged_cals=use_untagged_cals,
                    debug_mode=debug_mode,
                    force_run=force_run)
        return

    log.info(f'Processing MJD {mjd}')

    # check the MJD data directory path
    mjd_path = pathlib.Path(os.getenv('LVM_DATA_S')) / str(mjd)
    log.info(f'MJD processing path: {mjd_path}')
    if not mjd_path.is_dir():
        log.warning(f'{mjd = } is not valid raw data directory.')
        return

    # skip this reduction if the MJD is in a list of excluded (bad, engineering...) MJDs
    exclude_file = os.getenv('LVMCORE_DIR') + '/etc/exclude_mjds.txt'
    with open(exclude_file) as exclude_mjd_file:
        exclude = [tuple(map(int, line.split(','))) for line in exclude_mjd_file]
    if any([m[0] <= mjd <= m[1] for m in exclude]):
        log.info(f"MJD {mjd} falls within excluded period in {exclude_file}, skipping ...")
        return

    # generate the MJD metadata
    frames = get_frames_metadata(mjd=mjd)
    sub = frames.copy()

    # remove bad or test quality frames
    if force_run and (sub.qaqual == "BAD").all():
        tileid = sub.tileid.iloc[0]
        log.warning(f"You are about to reduce {expnum = } of {tileid = }, which was flagged as 'BAD' by the QC pipeline")
        log.warning("The DRP Team will not be responsible for failures during this reduction or the quality of its results")
        log.warning(f"We advice you to look for a good quality exposure of the same {tileid = }")
    else:
        try:
            sub = sub[(sub['qaqual'] == 'GOOD')]
        except KeyError:
            log.error("error while getting qaqual field in metadata.")
            log.error(f"Please try running `drp metadata regenerate -m {mjd}` before trying reducing your exposure again.")
            return
        if sub.empty:
            log.error(f"exposure {expnum = } was flagged as 'BAD' by the raw data quality pipeline")
            return

    # filter on exposure number
    if expnum:
        log.info(f'Filtering on exposure numbers {expnum}.')
        sub = filter_expnum(sub, expnum)

    # sort the frames
    sub = sub.sort_values(['expnum', 'camera'])

    # group by tileid, mjd
    groups = sub.groupby(['tileid', 'mjd'])

    # iterate over each group and reduce
    for key, group in groups:
        tileid, mjd = key

        # split into cals, and science
        cals = group[~(group['imagetyp'] == 'object')]
        sci = group[group['imagetyp'] == 'object']

        # avoid creating logs / status files for tileid+mjd groups with
        # no science files, unless explicitly reducing cals
        cal_cond = not cals.empty and with_cals
        sci_cond = not sci.empty and not no_sci

        # create start status
        create_status_file(tileid, mjd, status='started')

        if sci_cond or cal_cond:
            # start logging for this tileid, mjd
            logfile_path = start_logging(mjd, tileid)

        # attempt to reduce individual calibration files
        if cal_cond:
            for row in cals.to_dict("records"):
                try:
                    reduce_calib_frame(row)
                except Exception as e:
                    log.exception(f'Failed to reduce calib frame mjd {mjd} exposure {row["expnum"]}: {e}')
                    trace = traceback.format_exc()
                    update_error_file(tileid, mjd, row['expnum'], trace)

        # reduce the science data
        if sci_cond:
            kwargs = get_config_options('reduction_steps.science_reduction')
            for expnum in sci['expnum'].unique():
                with Timer(name=f'Reduction EXPNUM {expnum}', logger=log.info):
                    try:
                        science_reduction(expnum,
                                        fluxcal_method=fluxcal_method,
                                        skip_2d=skip_2d,
                                        skip_1d=skip_1d,
                                        skip_post_1d=skip_post_1d,
                                        skip_drpall=skip_drpall,
                                        clean_ancillary=clean_ancillary,
                                        use_longterm_cals=not use_nightly_cals,
                                        from_sandbox=not use_untagged_cals,
                                        debug_mode=debug_mode,
                                        force_run=force_run, **kwargs)
                    except Exception as e:
                        log.exception(f'Failed to reduce science frame mjd {mjd} exposure {expnum}: {e}')
                        create_status_file(tileid, mjd, status='error')
                        trace = traceback.format_exc()
                        update_error_file(tileid, mjd, expnum, trace)
                        log.info(f"the log file can be found here: {logfile_path}")
                        continue

        # create done status on successful run
        if not status_file_exists(tileid, mjd, status='error'):
            create_status_file(tileid, mjd, status='done')


def create_drpall(drp_version: str = None, overwrite: bool = False) -> None:
    """Create drpall summary file for a given DRP version

    Parameters
    ----------
    drp_version: str, optional
        Version of the DRP, by default None (current version)
    """
    drp_version = drp_version or drpver

    if overwrite:
        drpall = path.full('lvm_drpall', drpver=drp_version)
        drpall = drpall.replace('.fits', '.h5')
        if os.path.isfile(drpall):
            log.info(f"removing existing {drpall}")
            os.remove(drpall)
        else:
            log.info(f"no drpall file found for {drp_version = }")

    # define lvmSFrame paths
    sframe_paths = sorted(path.expand("lvm_frame", kind="SFrame", drpver=drp_version, tileid="*", mjd="*", expnum=8*"?"))
    nframes = len(sframe_paths)
    log.info(f"found {nframes} lvmSFrames under {drp_version = }")
    # iterate over each file and create/update the drpall file
    nfailed = 0
    failed = []
    for iframe, sframe_path in enumerate(sframe_paths):
        log.info(f"[{iframe+1}/{nframes}] {sframe_path = }")
        # extract Tile ID, MJD and exposure number from file
        # pars = path.extract("lvm_frame", sframe_path)
        pars = sframe_path.split(".fits")[0].split("/")
        tileid, mjd, expnum = int(pars[-3]), int(pars[-2]), int(pars[-1].split("-")[-1])
        cals_mjd = get_master_mjd(mjd)
        try:
            update_summary_file(sframe_path, tileid=tileid, mjd=mjd, expnum=expnum, master_mjd=cals_mjd, drpver=drp_version)
        except Exception as e:
            log.error(f"while updating drpall for {tileid = }, {mjd = }, {expnum = }: {e}")
            nfailed += 1
            failed.append(sframe_path)
            continue

    log.info(f"finished summarizing {nframes-nfailed} lvmSFrames in {drpall}")
    if nfailed != 0:
        log.warning(f"with {nfailed} failed frames:")
        log.warning(f"{failed = }")

    convert_h5_to_fits(drpall)
    log.info(f"finished converting HDF5 to FITS format in {drpall.replace('h5', '.fits')}")


def cache_gaia_spectra(mjds: Union[int, str, list], min_acquired=999, dry_run: bool = False) -> None:
    """Caches Gaia XP spectra for science field calibration

    Parameters
    ----------
    mjds : int|str|list[int]
        MJDs for which the caching should be run
    min_acquired : int, optional
        minimum number of acquired standard stars to skip caching, defaults to 999 (no skipping)
    dry_run : bool, optional
        lists exposures that will be targeted
    """
    log.info("start of Gaia XP spectra caching for science field flux calibration")
    gaia_cache_dir = os.path.join(os.getenv("LVM_MASTER_DIR"), "gaia_cache")
    os.makedirs(gaia_cache_dir, exist_ok=True)
    # parse MJDs
    mjds = parse_mjds(mjds)
    if isinstance(mjds, int):
        mjds = [mjds]
    log.info(f"selecting MJDs: {','.join(map(str, mjds))}")

    for mjd in mjds:
        # load metadata and filter good quality science frames
        frames = get_frames_metadata(mjd=mjd)
        frames.query("imagetyp == 'object' and qaqual == 'GOOD'", inplace=True)
        frames = frames.drop_duplicates(subset=["expnum"], keep="first")

        failed_expnums = []
        for exposure in frames.to_dict("records"):
            raw_path = path.full("lvm_raw", camspec=exposure["camera"], **exposure)
            # check for presence of standard stars metadata
            with fits.open(raw_path) as f:
                header = f[0].header
                expnum = exposure["expnum"]
                acquired_stds = list(header["STD*ACQ"].values())
                total_acquired = sum(acquired_stds)
                if total_acquired != 0:
                    if total_acquired >= min_acquired:
                        log.info(f"{expnum = } has standard stars metadata and {total_acquired} were acquired, skipping")
                        continue
                    log.info(f"{expnum = } has standard stars metadata and {total_acquired} were acquired")
                # get exposure parameters
                ra = header.get("POSCIRA", header.get("TESCIRA"))
                dec = header.get("POSCIDE", header.get("TESCIDE"))

            # cache corresponding gaia spectra
            log.info(f"going to download 15 field stars spectra with G<13.5 around {ra = }, {dec = } for {expnum = }")
            if not dry_run:
                try:
                    fluxcal.get_XP_spectra(expnum, ra, dec, plot=False, lim_mag=13.5, n_spec=15, GAIA_CACHE_DIR=gaia_cache_dir)
                except Exception as e:
                    log.error(f"failed caching of Gaia spectra for {expnum = }: {e}")
                    failed_expnums.append(expnum)
                    continue

    # summarize run
    log.info(f"cached metadata for {len(frames) - len(failed_expnums)} exposures, with {len(failed_expnums)} fails, {failed_expnums = }")


def reduce_calib_frame(row: dict):
    """ Reduce an individual calibration frame

    Tries to reduce an individual calibration exposure through
    the preprocessing and detrending stages.  Considered files
    are bias, darks, arcs, and flats.

    Parameters
    ----------
    row : dict
        info from the raw_metadata file
    """
    # get raw frame filepath
    filename = path.full('lvm_raw', **row, camspec=row['camera'])

    log.info(f'--- Starting calibration reduction of raw frame: {filename}')

    # set flavor
    flavor = row.get('imagetyp')
    flavor = 'fiberflat' if flavor == 'flat' else flavor

    # get master calibration paths
    mpixmask = path.full('lvm_calib', mjd=row['mjd'], camera=row['camera'], kind='pixmask')
    mbias = path.full('lvm_calib', mjd=row['mjd'], camera=row['camera'], kind='bias')
    mdark = path.full('lvm_calib', mjd=row['mjd'], camera=row['camera'], kind='dark')
    mpixflat = path.full('lvm_calib', mjd=row['mjd'], camera=row['camera'], kind='pixflat')

    # preprocess the frames
    log.info('--- Preprocessing raw frame ---')
    out_pre = path.full('lvm_anc', kind='p', drpver=drpver, imagetype=flavor, **row)
    preproc_raw_frame(in_image=filename, in_mask=mpixmask, out_image=out_pre)

    # detrend the frames
    log.info('--- Running detrend frame ---')

    in_cal = path.full('lvm_anc', kind='p', drpver=drpver, imagetype=flavor, **row)
    out_cal = path.full('lvm_anc', kind='d', drpver=drpver, imagetype=flavor, **row)

    detrend_frame(in_image=in_cal, out_image=out_cal,
                  in_bias=mbias, in_dark=mdark, in_pixelflat=mpixflat,
                  in_slitmap=Table(fibermap.data), reject_cr=False)
