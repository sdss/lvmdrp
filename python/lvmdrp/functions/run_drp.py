#!/usr/bin/env python
# encoding: utf-8

from lvmdrp.functions.imageMethod import preproc_raw_frame
from lvmdrp.utils.examples import get_frames_metadata
from lvmdrp import config


def get_config_options(level: str, flavor: str) -> dict:
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
    return cfg.get(flavor, {})


def reduce_one_file(filename: str, camera: str = None, mjd: int = None,
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

    # preprocess a bias/dark frames
    if flavor in {'bias', 'dark'}:
        kwargs = get_config_options('reduction_steps.preproc_raw_frame', flavor)
        preproc_raw_frame(filename, flavor=flavor, kind='p', camera=camera,
                          mjd=mjd, expnum=expnum, tileid=tileid, **kwargs)

    # if needed, create master calibs

    # process the flat/arc frames

    # process the science frame

    # write out RSS

    # perform quality checks


def run_drp(mjd: int = None, bias: bool = False, dark: bool = False):
    """ Run the LVM DRP

    Run the LVM data reduction pipeline on.  Optionally set flags
    to reduce only a subset of data.

    Parameters
    ----------
    mjd : int, optional
        The MJD of the raw data to reduce, by default None
    bias : bool, optional
        Flag to only reduce bias frames, by default False
    dark : bool, optional
        Flag to only reduce dark frames, by default False
    """
    # find files
    frames = get_frames_metadata(mjd=mjd)
    sub = frames.copy()

    # filter on files
    if bias:
        sub = sub[sub['imagetyp'] == 'bias']
    if dark:
        sub = sub[sub['imagetyp'] == 'dark']

    # reduce files
    for frame in sub:
        reduce_one_file(frame['path'], camera=frame['camera'],
                        mjd=frame['mjd'],
                        expnum=frame['expnum'], tileid=frame['tileid'],
                        flavor=frame['imagetyp'])

