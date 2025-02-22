import os
import fnmatch
from itertools import groupby

from typing import List, Union

from lvmdrp.core.constants import CALIBRATION_NAMES, CAMERAS, MASTERS_DIR
from lvmdrp import path
from lvmdrp.utils.convert import tileid_grp


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


def get_calib_paths(mjd, version=None, cameras="*", flavors=CALIBRATION_NAMES, longterm_cals=True, from_sanbox=False):
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
    from_sanbox : bool, optional
        Fall back option to pull calibrations from sandbox, by default False

    Returns
    -------
    calibs : dict[str, dict[str, str]]
        a dictionary containing calibrations for the given cameras
    """
    if version is None and not from_sanbox:
        raise ValueError(f"You must provide a version string to get calibration paths, {version = } given")

    # make long-term if taking calibrations from sandbox (nightly calibrations are not stored in sandbox)
    if from_sanbox:
        longterm_cals = True

    cams = fnmatch.filter(CAMERAS, cameras)
    channels = "".join(sorted(set(map(lambda c: c.strip("123"), cams))))

    tileid = 11111
    tilegrp = tileid_grp(tileid)

    # get long-term MJDs from sandbox using get_master_mjd, else use given MJD
    cals_mjd = get_master_mjd(mjd) if from_sanbox else mjd

    # define root path to pixel flats and masks
    # TODO: remove this once sdss-tree are updated with the corresponding species
    if from_sanbox:
        pixelmasks_path = os.path.join(MASTERS_DIR, "pixelmasks")
        path_species = "lvm_calib"
    else:
        pixelmasks_path = os.path.join(os.getenv('LVM_SPECTRO_REDUX'), f"{version}/{tilegrp}/{tileid}/pixelmasks")
        path_species = "lvm_master"

    pixel_flavors = {"pixmask", "pixflat"}
    if not pixel_flavors.issubset(flavors):
        pixel_flavors = set()
    flavors_ = set(flavors) - pixel_flavors

    # define paths to pixel flats and masks
    calibs = {}
    for flavor in pixel_flavors:
        calibs[flavor] = {c: os.path.join(pixelmasks_path, f"lvm-m{flavor}-{c}.fits") for c in cams}

    # define paths to the rest of the calibrations
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
