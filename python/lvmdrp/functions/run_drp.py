#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
from lvmdrp.functions.imageMethod import (preproc_raw_frame, create_master_frame,
                                          basic_calibration, find_peaks_auto, trace_peaks,
                                          extractSpec_drp)
from lvmdrp.utils.examples import get_frames_metadata
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
    return cfg.get(flavor, {}) if flavor else cfg


def create_masters(flavor, frames):

    sub = frames[frames['imagetyp'] == flavor]
    mjd = sub['mjd'][0]
    tileid = sub['tileid'][0]

    if flavor == 'bias':
        grp = sub.group_by('camera')
    elif flavor == 'dark':
        grp = sub.group_by(['camera', 'exptime'])

    # get the ancillary preproc files
    rpath = (pathlib.Path(os.getenv("LVM_SPECTRO_REDUX")) / f'{drpver}/{tileid}/{mjd}')
    ff = [str(i) for i in rpath.rglob(f'*p{flavor}*fits')]

    for row in grp:
        camera = row['camera'] if 'camera' in row.columns else None
        exptime = int(row['exptime']) if 'exptime' in row.columns else None

        # get the master path
        if flavor == 'bias':
            master = path.full("lvm_cal_mbias", mjd=mjd, drpver=drpver, camera=camera,
                               tileid=tileid)
        elif flavor == 'dark':
            master = path.full("lvm_cal_time", kind='mdark', mjd=mjd, drpver=drpver, camera=camera,
                               tileid=tileid, exptime=exptime)

        # create parent directries if need be
        if not pathlib.Path(master).parent.exists():
            pathlib.Path(master).parent.mkdir(parents=True, exist_ok=True)

        # create the master frame
        kwargs = get_config_options('reduction_steps.create_master_frame', flavor)
        log.info(f'custom configuration parameters for create_master_frame: {repr(kwargs)}')
        create_master_frame(in_images=ff, out_image=master, **kwargs)


def trace_fibers(in_file, camera, expnum, tileid, mjd):

    out_peaks = path.full("lvm_cal", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera,
                          expnum=expnum, kind='peaks', ext='txt')
    out_trace = path.full("lvm_cal", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera,
                          expnum=expnum, kind='trace', ext='fits')

    kwargs = get_config_options('reduction_steps.find_peaks_auto')
    log.info(f'custom configuration parameters for find_peaks_auto: {repr(kwargs)}')
    find_peaks_auto(in_image=in_file, out_peaks=out_peaks, **kwargs)

    kwargs = get_config_options('reduction_steps.trace_peaks')
    log.info(f'custom configuration parameters for trace_peaks: {repr(kwargs)}')
    trace_peaks(in_image=in_file, out_trace=out_trace, in_peaks=out_peaks, **kwargs)


def reduce_frame(filename: str, camera: str = None, mjd: int = None,
                 expnum: int = None, tileid: int = None,
                 flavor: str = None, exptime: float = None):
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

    # preprocess the frames
    kwargs = get_config_options('reduction_steps.preproc_raw_frame', flavor)
    log.info(f'custom configuration parameters for preproc_raw_frame: {repr(kwargs)}')
    preproc_raw_frame(filename, flavor=flavor, kind='p', camera=camera,
                      mjd=mjd, expnum=expnum, tileid=tileid, **kwargs)

    # end reduction for bias and darks
    if flavor in {'bias', 'dark'}:
        return

    # check master frames
    mbias = path.full("lvm_cal_mbias", mjd=mjd, drpver=drpver, camera=camera, tileid=tileid)
    mdark = path.full("lvm_cal_time", kind='mdark', mjd=mjd, drpver=drpver, camera=camera,
                      tileid=tileid, exptime=300)
    print('bias', mbias)
    print('mdark', mdark)
    if not pathlib.Path(mbias).exists() or not pathlib.Path(mdark).exists():
        raise ValueError('master bias/dark does not exist yet')

    # process the flat/arc frames
    flavor = 'fiberflat' if flavor == 'flat' else flavor
    in_cal = path.full("lvm_anc", kind='p', imagetype=flavor, mjd=mjd, drpver=drpver,
                       camera=camera, tileid=tileid, expnum=expnum)
    out_cal = path.full("lvm_anc", kind='c', imagetype=flavor, mjd=mjd, drpver=drpver,
                        camera=camera, tileid=tileid, expnum=expnum)

    print('p-anc, input', in_cal)
    print('c-anc, output', out_cal)
    kwargs = get_config_options('reduction_steps.basic_calibration', flavor)
    log.info(f'custom configuration parameters for basic_calibration: {repr(kwargs)}')
    basic_calibration(in_image=in_cal, out_image=out_cal,
                      in_bias=mbias, in_dark=mdark, **kwargs)

    # fiber tracing
    if 'flat' in flavor and not camera.startswith('b') and camera.endswith('1'):
        trace_fibers(out_cal, camera, expnum, tileid, mjd)

    # extract fiber spectra
    # arc_file = path.full("lvm_anc", kind='c', imagetype='arc', mjd=mjd, drpver=drpver,
    #                      camera=camera, tileid=tileid, expnum=expnum)
    # trace_file = path.full("lvm_cal", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera,
    #                        expnum=expnum, kind='trace', ext='fits')
    # exarc_file = path.full("lvm_anc", kind='x', imagetype='arc', mjd=mjd, drpver=drpver,
    #                        camera=camera, tileid=tileid, expnum=expnum)
    # extractSpec_drp(in_image=arc_file, out_rss=exarc_file, in_trace=trace_file,
    #                 method="aperture", aperture=4, plot=1, parallel="auto")

    # process the science frame

    # write out RSS

    # perform quality checks


def run_drp(mjd: int = None, bias: bool = False, dark: bool = False,
            skip_bd: bool = False, arc: bool = False, flat: bool = False,
            only_cal: bool = False, only_sci: bool = False):
    """ Run the LVM DRP

    Run the LVM data reduction pipeline on.  Optionally set flags
    to reduce only a subset of data.

    Parameters
    ----------
    mjd : int, optional
        The MJD of the raw data to reduce, by default None
    arc : bool, optional
        Flag to only reduce arc frames, by default False
    flat : bool, optional
        Flag to only reduce flat frames, by default False
    skip_bd : bool, optional
        Flag to skip reduction of bias/darks
    """
    # find files
    frames = get_frames_metadata(mjd=mjd)
    sub = frames.copy()

    # filter on files
    if bias:
        sub = sub[sub['imagetyp'] == 'bias']
    if dark:
        sub = sub[sub['imagetyp'] == 'dark']

    # get biases and darks
    cond = (frames['imagetyp'] == 'bias') | (frames['imagetyp'] == 'dark')
    precals = frames[cond]
    precals = precals.group_by(['expnum', 'camera'])

    if not skip_bd:
        # reduce biases / darks
        for frame in precals:
            reduce_frame(frame['path'], camera=frame['camera'],
                         mjd=frame['mjd'],
                         expnum=frame['expnum'], tileid=frame['tileid'],
                         flavor=frame['imagetyp'], exptime=frame['exptime'])

        # create master biases and darks
        create_masters('bias', precals)
        create_masters('dark', precals)

    # get all other image types
    sub = frames[~cond]
    if flat or arc:
        cond = sub['imagetyp'] == ('arc' if arc else 'flat')
    elif only_cal:
        cond = ~(sub['imagetyp'] == 'object')
    elif only_sci:
        cond = sub['imagetyp'] == 'object'
    sub = sub[cond]

    # group the frames
    sub = sub.group_by(['expnum', 'camera'])

    # reduce remaining files
    for frame in sub:
        reduce_frame(frame['path'], camera=frame['camera'],
                     mjd=frame['mjd'],
                     expnum=frame['expnum'], tileid=frame['tileid'],
                     flavor=frame['imagetyp'], exptime=frame['exptime'])


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

    if logpath.exists():
        return

    if not logpath.parent.exists():
        logpath.parent.mkdir(parents=True, exist_ok=True)

    log.start_file_logger(logpath)
