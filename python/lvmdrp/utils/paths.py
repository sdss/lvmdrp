import os
import fnmatch
from itertools import groupby
import pandas as pd

from typing import List, Union

from lvmdrp.core.constants import CALIBRATION_PRODUCTS, CAMERAS, MASTERS_DIR
from lvmdrp import path, __version__ as drpver
from lvmdrp.utils.convert import tileid_grp
from lvmdrp.utils import metadata as md


def get_master_mjd(sci_mjd: int) -> int:
    """ Get the correct master calibration MJD for a science frame

    Find the most relevant master calibration MJD given an
    input science frame MJD.

    Parameters
    ----------
    sci_mjd : int
        the MJD of the science exposure

    Returns
    -------
    int
        the master calibration MJD
    """
    masters_dir = sorted([f for f in os.listdir(MASTERS_DIR)
                          if os.path.isdir(os.path.join(MASTERS_DIR, f))])
    masters_dir = [f for f in masters_dir if f.isdigit()]
    target_master = list(filter(lambda f: sci_mjd >= int(f), masters_dir))
    return int(target_master[-1])


def mjd_from_expnum(expnum: Union[int, str, list, tuple]) -> List[int]:
    """Returns the MJD for the given exposure number

    Parameters
    ----------
    expnum : int|list[int]
        the exposure number(s)

    Returns
    -------
    list[int]
        the MJD of the exposure
    """
    if isinstance(expnum, int):
        pass
    elif isinstance(expnum, str) and "-" in expnum:
        expnum = [int(exp) for exp in expnum.split("-")]
        expnum = list(range(expnum[0], expnum[1]+1))

    if isinstance(expnum, (tuple, list)):
        mjds = [mjd_from_expnum(exp)[0] for exp in expnum]
        return mjds

    rpath = path.expand("lvm_raw", camspec="*", mjd="*", hemi="s", expnum=expnum)
    if len(rpath) == 0:
        raise ValueError(f"no raw frame found for exposure number {expnum}")
    mjd = path.extract("lvm_raw", rpath[0])["mjd"]
    return [int(mjd)]


def get_calib_paths(mjd, version=None, cameras="*", flavors=CALIBRATION_PRODUCTS, longterm_cals=True, from_sandbox=False, return_mjd=False):
    """Returns a dictionary containing paths for calibration frames

    Parameters
    ----------
    mjd : int
        MJD to reduce
    version : str, optional
        Version of the pipeline to pull calibrations from, by default None
    cameras : list[str]|str, optional
        List of cameras or wildcard to match, by default '*'
    flavors : list, tuple or set
        Only get paths for this calibrations, by default all available flavors
    longterm_cals : bool
        Whether to use long-term calibration frames or not, defaults to True
    from_sandbox : bool, optional
        Fall back option to pull calibrations from sandbox, by default False

    Returns
    -------
    calibs : dict[str, dict[str, str]]
        a dictionary containing calibrations for the given cameras
    """
    if version is None and not from_sandbox:
        raise ValueError(f"You must provide a version string to get calibration paths, {version = } given")

    # make long-term if taking calibrations from sandbox (nightly calibrations are not stored in sandbox)
    if from_sandbox:
        longterm_cals = True

    cams = fnmatch.filter(CAMERAS, cameras)
    channels = "".join(sorted(set(map(lambda c: c.strip("123"), cams))))

    tileid = 11111
    tilegrp = tileid_grp(tileid)

    # get long-term MJDs from sandbox using get_master_mjd, else use given MJD
    cals_mjd = get_master_mjd(mjd) if longterm_cals else mjd

    # define root path to pixel flats and masks
    # TODO: remove this once sdss-tree are updated with the corresponding species
    if from_sandbox:
        pixelmasks_path = os.path.join(MASTERS_DIR, "pixelmasks")
        path_species = "lvm_calib"
    else:
        pixelmasks_path = os.path.join(os.getenv('LVM_SPECTRO_REDUX'), f"{version}/{tilegrp}/{tileid}/pixelmasks")
        path_species = "lvm_master"

    # define paths to pixel flats and masks
    calibs = {}
    pixel_flavors = {"pixmask", "pixflat"}.intersection(flavors)
    for flavor in pixel_flavors:
        calibs[flavor] = {c: os.path.join(pixelmasks_path, f"lvm-m{flavor}-{c}.fits") for c in cams}

    # define paths to the rest of the calibrations
    flavors_ = set(flavors) - pixel_flavors
    for flavor in flavors_:
        # define camera for camera frames or spectrograph combined frames
        cam_or_chan = channels if flavor.startswith("fiberflat_") else cams

        # define calibration prefix
        # TODO: clean this after update in sdss-tree that will consistently handle prefixes for nightly and long-term cals
        if path_species == "lvm_calib":
            prefix = ""
        else:
            prefix = "m" if flavor in ["bias", "fiberflat_twilight"] or longterm_cals else "n"

        calibs[flavor] = {c: path.full(path_species, drpver=version, tileid=tileid, mjd=cals_mjd, kind=f"{prefix}{flavor}", camera=c) for c in cam_or_chan}

    if return_mjd:
        return calibs, cals_mjd
    return calibs


def group_calib_paths(calib_paths):
    """Returns a dictionary of calibration paths grouped by channel given a set of camera frame paths

    Parameters
    ----------
    calib_paths : dict[str, str]
        Dictionary containing camera frame calibrations

    Returns
    -------
    paths : dict[str, str]
        Calibration paths grouped by channel
    """
    paths = {}
    for channel, cameras in groupby(calib_paths, key=lambda p: os.path.basename(p).split(".")[0].split("-")[-1][0]):
        paths[channel] = sorted([calib_paths[camera] for camera in cameras])
    return paths


def get_frames_paths(mjds, kind, camera_or_channel, query=None, expnums=None, filetype="lvm_anc", drpver=drpver, filter_existing=True):
    """Generate file paths for a set of frames based on specified parameters.

    Parameters
    ----------
    mjds : int or list[int]
        MJD(s) to retrieve frame metadata for. Can be a single integer or a list of integers.
    kind : str
        The type of path to create (e.g., 'x', 'l', 'w').
    camera_or_channel : str
        The camera or channel identifier (e.g., 'r1', 'b').
    query : str, optional
        A query string to filter the frames DataFrame. Defaults to None.
    expnums : list[int], optional
        A list of exposure numbers to filter the frames. If None, no filtering
        is applied. Defaults to None.
    filetype : str, optional
        The type of file to generate paths for (e.g., 'lvm_anc', 'lvm_frame'). Defaults to "lvm_anc".
    drpver : str, optional
        The data reduction pipeline version to use. Defaults to the current version.
    filter_existing : bool, optional
        If True, only include paths that correspond to existing files.
        Defaults to True.

    Returns
    -------
    list[str]
        A list of file paths corresponding to the specified frames and parameters.
    """
    mjds = [mjds] if isinstance(mjds, int) else mjds
    frames = pd.concat([md.get_frames_metadata(mjd=mjd) for mjd in mjds], ignore_index=True)
    if query is not None:
        frames = frames.query(query)
    if expnums is not None:
        frames = frames.query("expnum in @expnums")

    f = frames.drop_duplicates(subset=["expnum"])
    paths = [path.full(filetype, mjd=s.mjd, tileid=s.tileid, drpver=drpver, kind=kind, camera=camera_or_channel, imagetype=s.imagetyp, expnum=s.expnum) for _, s in f.iterrows()]
    if filter_existing:
        paths = list(filter(lambda p: os.path.isfile(p), paths))
    return paths
