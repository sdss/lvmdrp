#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
import yaml
from astropy.table import Table
from lvmdrp.functions.imageMethod import (preproc_raw_frame, create_master_frame,
                                          basic_calibration, find_peaks_auto, trace_peaks,
                                          extract_spectra)
from lvmdrp.functions.rssMethod import (determine_wavelength_solution, createPixTable_drp, resampleWave_drp)
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
    return cfg.get(flavor, {}) if flavor else cfg.get("default", cfg)


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
                 flavor: str = None):
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

    # preprocess the frames
    log.info('--- Preprocessing raw frame ---')
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
    log.info(f'Using master bias: {mbias}')
    log.info(f'Using master dark: {mdark}')
    if not pathlib.Path(mbias).exists() or not pathlib.Path(mdark).exists():
        log.error('No master bias or dark frames exist ---')
        raise ValueError('master bias/dark does not exist yet')

    # process the flat/arc frames
    flavor = 'fiberflat' if flavor == 'flat' else flavor
    in_cal = path.full("lvm_anc", kind='p', imagetype=flavor, mjd=mjd, drpver=drpver,
                       camera=camera, tileid=tileid, expnum=expnum)
    out_cal = path.full("lvm_anc", kind='c', imagetype=flavor, mjd=mjd, drpver=drpver,
                        camera=camera, tileid=tileid, expnum=expnum)

    log.info(f'Output preproc file: {in_cal}')
    log.info('--- Running basic calibration ---')
    kwargs = get_config_options('reduction_steps.basic_calibration', flavor)
    log.info(f'custom configuration parameters for basic_calibration: {repr(kwargs)}')
    basic_calibration(in_image=in_cal, out_image=out_cal,
                      in_bias=mbias, in_dark=mdark, **kwargs)
    log.info(f'Output calibrated file: {out_cal}')

    # fiber tracing
    log.info('--- Running fiber trace ---')
    if 'flat' in flavor and not camera.startswith('b') and camera.endswith('1'):
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
    log.info('--- Extracting fiber spectra ---')
    kwargs = get_config_options('reduction_steps.extract_spectra')
    log.info(f'custom configuration parameters for extract_spectra: {repr(kwargs)}')
    extract_spectra(in_image=cal_file, out_rss=xout_file, in_trace=trace_file, **kwargs)
    log.info(f'Output extracted file: {xout_file}')

    # determine the wavelength solution
    if flavor == 'arc':
        wave_file = path.full('lvm_cal', kind='wave', drpver=drpver, mjd=mjd, tileid=tileid,
                              camera=camera, expnum=expnum, ext='fits')
        lsf_file = path.full('lvm_cal', kind='lsf', drpver=drpver, mjd=mjd, tileid=tileid,
                             camera=camera, expnum=expnum, ext='fits')
        line_ref = pathlib.Path(__file__).parent.parent / f"etc/lvm-neon_nist_{camera}.txt"
        kwargs = get_config_options('reduction_steps.determine_wavesol')
        log.info('--- Determining wavelength solution ---')
        log.info(f'custom configuration parameters for determine_wave_solution: {repr(kwargs)}')
        determine_wavelength_solution(in_arc=xout_file, out_wave=wave_file, out_lsf=lsf_file,
                                      in_ref_lines=line_ref, **kwargs)
        log.info(f'Output wave peak traceset file: {wave_file}')
        log.info(f'Output lsf traceset file: {lsf_file}')

    # create pixel table
    wave_file = find_file('wave', mjd=mjd, tileid=tileid, camera=camera)
    lsf_file = find_file('lsf', mjd=mjd, tileid=tileid, camera=camera)
    wout_file = path.full("lvm_anc", kind='w', imagetype=flavor, mjd=mjd, drpver=drpver,
                          camera=camera, tileid=tileid, expnum=expnum)
    log.info('--- Creating pixel table ---')
    createPixTable_drp(in_rss=xout_file, out_rss=wout_file, arc_wave=wave_file, arc_fwhm=lsf_file)
    log.info(f'Output calibrated wavelength file: {wout_file}')

    # set wavelength resample params
    CHANNEL_WL = {"b1": (3600, 5930), "r1": (5660, 7720), "z1": (7470, 9800)}
    wave_range = CHANNEL_WL[camera]

    # resample wavelength
    hout_file = path.full("lvm_anc", kind='h', imagetype=flavor, mjd=mjd, drpver=drpver,
                          camera=camera, tileid=tileid, expnum=expnum)
    kwargs = get_config_options('reduction_steps.resample_wave', flavor)
    log.info('--- Resampling wavelength grid ---')
    log.info(f'custom configuration parameters for resample_wave: {repr(kwargs)}')
    resampleWave_drp(in_rss=wout_file, out_rss=hout_file, start_wave=wave_range[0],
                     end_wave=wave_range[1], disp_pix=1.0, method="linear", err_sim=10,
                     parallel="auto", extrapolate=True, **kwargs)
    log.info(f'Output resampled wave file: {hout_file}')


    # process the science frame

    # write out RSS

    # perform quality checks


def run_drp(mjd: int = None, bias: bool = False, dark: bool = False,
            skip_bd: bool = False, arc: bool = False, flat: bool = False,
            only_cal: bool = False, only_sci: bool = False, spec: int = None,
            camera: str = None):
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
    # write the drp parameter configuration
    write_config_file()

    # find files
    frames = get_frames_metadata(mjd=mjd)
    sub = frames.copy()

    # filter on files
    if bias:
        sub = sub[sub['imagetyp'] == 'bias']
    if dark:
        sub = sub[sub['imagetyp'] == 'dark']

    # filter on camera or spectrograph
    if spec:
        sub = sub[sub['spec'] == f'sp{spec}']
    if camera:
        sub = sub[[camera in i for i in sub['camera']]]

    # get biases and darks
    cond = (frames['imagetyp'] == 'bias') | (frames['imagetyp'] == 'dark')
    precals = frames[cond]
    precals = precals.group_by(['expnum', 'camera'])

    if not skip_bd:
        # reduce biases / darks
        for frame in precals:
            # skip bad or test quality
            if frame['quality'].lower() != 'excellent':
                log.info(f"Skipping frame {frame['name']} with quality: {frame['quality']}")
                continue

            reduce_frame(frame['path'], camera=frame['camera'],
                         mjd=frame['mjd'],
                         expnum=frame['expnum'], tileid=frame['tileid'],
                         flavor=frame['imagetyp'])

        # create master biases and darks
        create_masters('bias', precals)
        create_masters('dark', precals)

    # get all other image types
    sub = frames[~cond]
    if flat or arc:
        sub = sub[sub['imagetyp'] == ('arc' if arc else 'flat')]
    elif only_cal:
        sub = sub[~(sub['imagetyp'] == 'object')]
    elif only_sci:
        sub = sub[sub['imagetyp'] == 'object']

    # group the frames
    sub = sub.group_by(['expnum', 'camera'])

    # sort the table by flat, arc, science
    if not only_sci:
        sub = sort_cals(sub)

    # reduce remaining files
    for frame in sub:
        # skip bad or test quality
        if frame['quality'].lower() != 'excellent':
            log.info(f"Skipping frame {frame['name']} with quality: {frame['quality']}")
            continue

        reduce_frame(frame['path'], camera=frame['camera'],
                     mjd=frame['mjd'],
                     expnum=frame['expnum'], tileid=frame['tileid'],
                     flavor=frame['imagetyp'])


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
    with open(cpath, 'w') as f:
        f.write(yaml.safe_dump(dict(config), sort_keys=False, indent=2))


def sort_cals(table: Table) -> Table:
    """ sort the astropy table by flat, arcs, sci """
    df = table.to_pandas()
    ss = df.sort_values(['camera', 'expnum'])
    ee = ss.set_index('imagetyp', drop=False).loc[['flat', 'arc', 'object']].reset_index(drop=True)
    return table.from_pandas(ee)
