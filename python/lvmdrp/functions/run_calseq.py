# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Feb 1, 2024
# @Filename: run_calseq.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

# * learn the current calibration sequence:
#   - bias/dark
#   - pixel flat
#   - fiber flat and twilights
#   - arcs
# * calibrate sequence:
#   - bias/dark/pixflat (2D)
#   - traces (centroids and widths)
#   - fiber flat (1D)
#   - illumination corrections
#   - wavelength fitting
# * produce QA plots
# * write calibration files
#   - master bias/dark/pixflat
#   - master pixelmask
#   - master traces
#   - master fiberflat (with illumination corrections)
#

import os
import numpy as np
from glob import glob
from copy import deepcopy as copy
from shutil import copy2, rmtree
from itertools import groupby
from astropy.stats import biweight_location, biweight_scale
from multiprocessing import Pool
from scipy import interpolate
from typing import List, Tuple, Dict

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.core.plot import create_subplots, save_fig
from lvmdrp.core import dataproducts as dp
from lvmdrp.core.constants import (
    CAMERAS,
    SPEC_CHANNELS,
    LVM_REFERENCE_COLUMN,
    LVM_NBLOCKS,
    MASTERS_DIR,
    ARC_LAMPS)
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.image import Image, loadImage
from lvmdrp.core.rss import RSS, lvmFrame

from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.main import get_config_options, read_fibermap, get_master_mjd, reduce_2d
from lvmdrp.functions.run_twilights import fit_fiberflat, remove_field_gradient, combine_twilight_sequence


SLITMAP = read_fibermap(as_table=True)
MASTER_CON_LAMPS = {"b": "ldls", "r": "ldls", "z": "quartz"}
MASK_BANDS = {
    "b": [(3910, 4000), (4260, 4330)],
    "r": [(6840,6960)],
    "z": [(7570, 7700)]
}

def get_calib_paths(mjd, flavors={"pixmask", "pixflat", "bias", "trace_guess", "trace", "width", "amp", "model", "wave", "lsf", "fiberflat_dome", "fiberflat_twilight"}, use_fiducial_cals=True):
    """Returns a dictionary containing paths for calibration frames

    Parameters
    ----------
    mjd : int
        MJD to reduce
    only_cals : list, tuple or set
        Only produce this calibrations, by default {"pixmask", "pixflat", "bias", "trace_gues", "trace", "width", "amp", "model", "wave", "lsf", "fiberflat_dome", "fiberflat_twilight"}
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True

    Returns
    -------
    calibs : dict[str, dict[str, str]]
        a dictionary containing calibrations for the given
    """

    master_mjd = get_master_mjd(mjd) if use_fiducial_cals else mjd
    path_species = "lvm_calib" if use_fiducial_cals else "lvm_master"
    calibs = {}
    for flavor in flavors:
        # define camera for camera frames or spectrograph combined frames
        cams = "brz" if flavor.startswith("fiberflat_") else CAMERAS
        # define calibration prefix
        prefix = "" if flavor == "bias" or use_fiducial_cals else "n"

        calibs[flavor] = {c: path.full(path_species, drpver=drpver, tileid=11111, mjd=master_mjd, kind=f"{prefix}{flavor}", camera=c) for c in cams}

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


def choose_sequence(frames, flavor, kind, truncate=True):
    """Returns exposure numbers splitted in different sequences

    Parameters:
    ----------
    frames : pd.DataFrame
        Pandas dataframe containing frames metadata
    flavor : str
        Flavor of calibration frame: 'twilight', 'bias', 'flat', 'arc'
    kind : str
        Kind of calibration frame: 'nightly', 'longterm'
    truncate : bool, optional
        Truncate sequences to match the expected number of exposures, by default True

    Return:
    ------
    list
        list containing arrays of exposure numbers for each sequence
    """
    EXPECTED_LENGTHS = {
        "bias": 7,
        "flat": 2 if kind=="nightly" else 24,
        "arc": 2 if kind=="nightly" else 24,
        "twilight": 24
    }

    if not isinstance(flavor, str) or flavor not in {"twilight", "bias", "flat", "arc"}:
        raise ValueError(f"invalid flavor '{flavor}', available values are 'twilight', 'bias', 'flat', 'arc'")
    if not isinstance(kind, str) or kind not in {"nightly", "longterm"}:
        raise ValueError(f"invalid kind '{kind}', available values are 'nightly' and 'longterm'")

    # TODO: filter out exposures with hartmann door wrong status

    if flavor == "twilight":
        query = "imagetyp == 'flat' and not (ldls|quartz) and not (neon|hgne|argon|xenon)"
    elif flavor == "bias":
        query = "imagetyp == 'bias'"
    elif flavor == "flat":
        query = "imagetyp == 'flat' and (ldls|quartz)"
    elif flavor == "arc":
        query = "imagetyp == 'arc' and not (ldls|quartz) and (neon|hgne|argon|xenon)"
    expnums = np.sort(frames.query(query).expnum.unique())
    diff = np.diff(expnums)
    div, = np.where(np.abs(diff) > 1)

    sequences = np.split(expnums, div+1)
    [seq.sort() for seq in sequences]
    log.info(f"found sequences: {sequences}")

    if len(sequences) == 0:
        raise ValueError(f"no calibration frames of flavor '{flavor}' found using the query: '{query}'")

    lengths = [len(seq) for seq in sequences]
    if flavor == "twilight":
        chosen_expnums = np.concatenate(sequences)
    else:
        if len(sequences) > 1:
            idx = lengths.index(min(lengths) if kind == "nightly" else max(lengths))
            chosen_expnums = sequences[idx]
        else:
            chosen_expnums = sequences[0]

    chosen_frames = frames.query("expnum in @chosen_expnums")
    expected_length = EXPECTED_LENGTHS[flavor]
    sequence_length = len(chosen_expnums)
    if sequence_length == expected_length:
        chosen_frames.sort_values(["expnum", "camera"], inplace=True)
        return chosen_frames, chosen_expnums

    log.warning(f"wrong sequence length for {flavor = }: {sequence_length}, expected {expected_length}")
    if truncate and sequence_length > expected_length:
        log.info(f"selecting first {expected_length} exposures")
        if flavor == "flat":
            qrtz_expnums = chosen_expnums[chosen_frames.quartz][:expected_length//2]
            ldls_expnums = chosen_expnums[chosen_frames.ldls][:expected_length//2]
            chosen_expnums = np.concatenate([qrtz_expnums, ldls_expnums])
        elif flavor == "arc":
            short_expnums = chosen_expnums[chosen_frames.exptime == 10][:expected_length//2]
            long_expnums = chosen_expnums[chosen_frames.exptime == 50][:expected_length//2]
            chosen_expnums = np.concatenate([short_expnums, long_expnums])
        else:
            chosen_expnums = chosen_expnums[:expected_length]
        chosen_frames = frames.query("expnum in @chosen_expnums")
        chosen_frames.sort_values(["expnum", "camera"], inplace=True)

    return chosen_frames, chosen_expnums


def get_fibers_signal(mjd, camera, expnum, imagetyp="flat"):
    img_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, expnum=expnum, camera=camera, imagetype=imagetyp, kind="d")
    img = loadImage(img_path)

    img._data = np.nan_to_num(img._data, posinf=0, neginf=0)
    img._error = np.nan_to_num(img._error, nan=np.inf, neginf=np.inf)

    fiberpos = img.match_reference_column()
    fiberpos = fiberpos.round().astype(int)
    data = img._data[fiberpos]
    error = img._error[fiberpos]

    snr = data / error

    log.info(f"average signal = {np.nanmean(data)}")
    log.info(f"average SNR = {np.nanmean(snr)}")
    log.info(f"standard deviation SNR = {np.nanstd(snr)}")

    return fiberpos, img


def get_exposed_std_fiber(mjd, expnums, camera, imagetyp="flat", ref_column=LVM_REFERENCE_COLUMN, snr_threshold=5, use_header=True, display_plots=False):
    """Returns the exposed standard fiber IDs for a given exposure sequence and camera

    Parameters
    ----------
    mjd : int
        MJD of the exposure sequence
    expnums : list
        List of exposure numbers in the sequence
    camera : str
        Camera name (e.g. "b1")
    ref_column : int
        Reference column for the fiber trace
    snr_threshold : float
        SNR threshold above which a fiber is considered to be exposed
    use_header : bool
        Use CALIBFIB header keyword if available, defaults to True
    display_plots : bool
        If True, display plots

    Returns
    -------
    dict
        Dictionary with the exposed standard fiber IDs for each exposure in the sequence
    """
    log.info(f"loading detrended frames for {camera = }, exposures = {expnums}")
    rframe_paths = sorted([path.expand("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, expnum=expnum, kind="d", imagetype=imagetyp)[0] for expnum in expnums])
    images = [image_tasks.loadImage(rframe_path) for rframe_path in rframe_paths]

    # define fibermap for camera & std fibers parameters
    fibermap = images[0]._slitmap[images[0]._slitmap["spectrographid"]==int(camera[1])]
    spec_select = fibermap["telescope"] == "Spec"
    ids_std = fibermap[spec_select]["orig_ifulabel"]
    log.info(f"standard fibers in {camera = }: {','.join(ids_std)}")

    # get exposed standard fibers from header if present
    exposed_stds = {image._header["EXPOSURE"]: image._header.get("CALIBFIB", None) for image in images}
    block_idxs = np.arange(LVM_NBLOCKS).tolist()
    if use_header and all([exposed_std is not None for exposed_std in exposed_stds.values()]):
        log.info(f"standard fibers in {camera = }: {','.join(exposed_stds.values())}")
        for expnum, exposed_std in exposed_stds.items():
            fiber_par = fibermap[fibermap["orig_ifulabel"] == exposed_std]
            block_idx = int(fiber_par["blockid"][0][1:])-1
            exposed_stds[expnum] = (exposed_std, [block_idx])
    else:
        if use_header:
            log.warning(f"exposed standard fibers not found in header for {camera = }, going to infer exposed fibers from SNR")
        else:
            log.info(f"inferring exposed standard fiber for {camera = } from SNR")

        # combine frames for given camera
        log.info(f"combining {len(images)} exposures")
        cimage = image_tasks.combineImages(images, normalize=False, background_subtract=False)
        cimage.setData(data=np.nan_to_num(cimage._data), error=np.nan_to_num(cimage._error, nan=np.inf))
        # calculate correction in reference Y positions along reference column
        fiber_pos = cimage.match_reference_column(ref_column)
        pos_std = fiber_pos[spec_select].round().astype("int")
        idx_std = np.arange(pos_std.size)

        # calculate SNR along colummn
        nrows = max(len(images)//3, 1)
        fig, axs = create_subplots(to_display=display_plots,
                                nrows=nrows, ncols=3,
                                figsize=(15,5*nrows),
                                sharex=True, sharey=True,
                                layout="constrained")
        fig.supxlabel("standard fiber ID")
        fig.supylabel("SNR at fiber centroid")
        fig.suptitle(f"exposed standard fibers in sequence {expnums[0]} - {expnums[-1]} in camera '{camera}'")
        exposed_stds, block_idxs = {}, np.arange(LVM_NBLOCKS).tolist()
        for image, ax in zip(images, axs):
            expnum = image._header["EXPOSURE"]
            column = image.getSlice(ref_column, axis="Y")
            snr = (column._data/column._error)
            snr_med = biweight_location(snr[fiber_pos.round().astype("int")], ignore_nan=True)
            snr_std = biweight_scale(snr[fiber_pos.round().astype("int")], ignore_nan=True)
            snr_std_med = biweight_location(snr[pos_std], ignore_nan=True)
            snr_std_std = biweight_scale(snr[pos_std], ignore_nan=True)
            log.debug(f"{expnum = } mean SNR = {snr_med:.2f} +/- {snr_std:.2f} (standard fibers: {snr_std_med:.2f} +/- {snr_std_std:.2f})")

            ax.set_title(f"{expnum = }", loc="left")
            ax.axhspan(snr_med-snr_std, snr_med+snr_std, lw=0, fc="0.7", alpha=0.5)
            ax.axhline(snr_med, lw=1, color="0.7")
            ax.axhspan(max(0, snr_std_med-snr_threshold*snr_std_std), snr_std_med+snr_threshold*snr_std_std, lw=0, fc="0.7", alpha=0.5)
            ax.axhline(snr_std_med, lw=1, color="0.7")
            ax.bar(idx_std, snr[pos_std], hatch="///////", lw=0, ec="tab:blue", fc="none", zorder=999)
            ax.set_xticks(idx_std)
            ax.set_xticklabels(ids_std)

            # select standard fiber exposed if any
            select_std = snr[pos_std] > snr_std_med + snr_threshold * snr_std_std
            exposed_std = ids_std[select_std]
            if select_std.sum() > 1:
                exposed_std_ = exposed_std[np.argmax(snr[pos_std[select_std]])]
                log.warning(f"more than one standard fiber selected in {expnum = }: {','.join(exposed_std)}, selecting highest SNR: '{exposed_std_}'")
                exposed_std = exposed_std_
            elif select_std.sum() > 0:
                exposed_std = exposed_std[0]
            else:
                exposed_std = None
                continue

            # highlight exposed fiber in plot
            select_exposed = ids_std == exposed_std
            ax.bar(idx_std[select_exposed], snr[pos_std][select_exposed], hatch="///////", lw=0, ec="tab:red", fc="none", zorder=999)

            # get block ID for exposed standard fiber
            fiber_par = image._slitmap[image._slitmap["orig_ifulabel"] == exposed_std]
            block_idx = int(fiber_par["blockid"][0][1:])-1
            if block_idx in block_idxs:
                block_idxs.remove(block_idx)
            log.info(f"exposed standard fiber in exposure {expnum}: '{exposed_std}' ({block_idx = })")

            exposed_stds[expnum] = (exposed_std, [block_idx])

        # handle case of no standard fiber exposed
        if len(exposed_stds) == 0:
            exposed_stds[expnums[0]] = (None, block_idxs)
            block_idxs = []

        # save figure
        save_fig(fig,
                product_path=path.full("lvm_anc", drpver=drpver, tileid=11111,
                                        mjd=mjd, camera=camera, expnum=f"{expnums[0]}_{expnums[-1]}",
                                        kind="d", imagetype=imagetyp),
                to_display=display_plots,
                figure_path="qa",
                label="exposed_std_fiber")

    # add missing blocks for first exposure
    if len(block_idxs) > 0:
        log.info(f"remaining blocks without exposed standard fibers: {block_idxs}, adding to first exposure")
        expnum = list(exposed_stds.keys())[0]
        exposed_stds[expnum] = (exposed_stds[expnum][0], sorted(exposed_stds[expnum][1]+block_idxs))

    # list unexposed standard fibers
    unexposed_stds = [fiber for fiber in ids_std if fiber not in list(zip(*exposed_stds.values()))[0]]

    return exposed_stds, unexposed_stds


def _load_shift_report(mjd):
    """Reads QC reports with the electronic pixel shifts"""

    with open(os.path.join(os.environ["LVM_SANDBOX"], "shift_monitor", f"shift_{mjd}.txt"), "r") as f:
        lines = f.readlines()[2:]

    shifts_report = {}
    for line in lines:
        cols = line[:-1].split()
        if not cols:
            continue
        _, exp, _, spec = cols[:4]
        exp = int(exp)
        spec = spec[-1]
        shifts = np.array([int(_) for _ in cols[4:]])
        shifts_report[(spec, exp)] = (shifts[::2], shifts[1::2])

    return shifts_report


def _get_reference_expnum(frame, ref_frames):
    """Get reference frame for a given frame

    Given a frame and a set of reference frames, get the reference frame for the
    given frame. This routine will return the reference frame with the closest
    exposure number to the given frame.

    Parameters:
    ----------
    frame : pd.Series
        Frame metadata
    ref_frames : pd.DataFrame
        Reference frames metadata

    Returns:
    -------
    pd.Series
        Reference frame metadata
    """
    if frame.imagetyp == "flat" and frame.ldls|frame.quartz:
        refs = ref_frames.query("imagetyp == 'flat' and (ldls|quartz)")
    elif frame.imagetyp == "flat":
        refs = ref_frames.query("imagetyp == 'flat' and not (ldls|quartz)")
    else:
        refs = ref_frames.query("imagetyp == @frame.imagetyp")

    ref_expnums = refs.expnum.unique()
    if len(ref_expnums) < 2:
        raise ValueError(f"no reference frame found for {frame.imagetyp}")
    idx = np.argmin(np.abs(ref_expnums-frame.expnum))
    if idx > 0:
        idx -= 1
    if idx == 0:
        idx += 1
    return ref_expnums[idx]


def _move_master_calibrations(mjd, kind=None):

    kinds = {"bias", "trace", "width", "fiberflat", "fiberflat_twilight", "wave", "lsf"}
    if isinstance(kind, (list, tuple, set, np.ndarray)):
        kinds = kind
    elif isinstance(kind, str) and kind in kinds:
        kinds = {kind}
    elif kind is None:
        pass
    else:
        raise ValueError(f"kind must be one of {kinds}")

    for kind in kinds:
        src_paths = path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind=f"m{kind}", camera="*")
        for src_path in src_paths:
            camera = os.path.basename(src_path).split(".")[0].split("-")[-1]
            dst_path = path.full("lvm_calib", mjd=mjd, kind=kind, camera=camera)
            if os.path.isfile(src_path):
                try:
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    copy2(src_path, dst_path)
                    log.info(f"copied {src_path} into {dst_path}")
                except PermissionError as e:
                    log.error(f"error while copying {src_path}: {e}")


def _clean_ancillary(mjd, expnums=None, flavors="all"):
    """Clean ancillary files

    Given a set of MJDs and (optionally) exposure numbers, clean the ancillary
    files for the given flavor of frames. This routine will remove the ancillary
    files for the given flavor of frames in the corresponding calibration
    directory in the `masters_mjd` or by default in the smallest MJD in `mjds`.

    Parameters:
    ----------
    mjd : int
        MJD to clean
    expnums : list
        List of exposure numbers to clean
    flavors : list, tuple, set or str
        type of, defaults to "all"
    """
    # filter by target image types
    all_flavors = {"bias", "dark", "flat", "arc", "cent", "amp", "width", "stray"}
    if not set(flavors).issubset(flavors):
        raise ValueError(f"Invalid flavor: '{flavors}'. Must be one of {all_flavors} or 'all'")

    ancillary_dir = os.path.join(os.getenv("LVM_SPECTRO_REDUX"), drpver, str(mjd), "ancillary")
    if flavors == "all":
        rmtree(ancillary_dir)
        return

    for flavor in flavors:
        # remove ancillary files
        ancillary_paths = path.expand("lvm_anc", drpver=drpver, mjd=mjd, tileid=11111, kind='*', imagetype=flavor, camera="*", expnum="*")
        [os.remove(ancillary_path) for ancillary_path in ancillary_paths]

    if not os.listdir(ancillary_dir):
        os.rmdir(ancillary_dir)


def _get_ring_expnums(expnums_ldls, expnums_qrtz, ring_size=12, sort_expnums=False):
    """Split expnums into primary and secondary ring expnums

    Given a set of MJDs and (optionally) exposure numbers, split the expnums
    into primary and secondary ring expnums. This routine will return the
    primary and secondary ring expnums for the given expnums.

    Parameters:
    ----------
    expnums_ldls : list
        List of LDLS expnums
    expnums_qrtz : list
        List of quartz expnums
    ring_size : int
        Size of the primary ring
    sort_expnums : bool
        Sort expnums

    Returns:
    -------
    expnum_params : dict
        Dictionary with the expnums parameters
    """

    # sort expnums
    if sort_expnums:
        expnums_ldls = sorted(expnums_ldls)
        expnums_qrtz = sorted(expnums_qrtz)

    # split expnums into primary and secondary ring expnums
    pri_ldls_expnums = expnums_ldls[:ring_size]
    pri_qrtz_expnums = expnums_qrtz[:ring_size]
    sec_ldls_expnums = expnums_ldls[ring_size:]
    sec_qrtz_expnums = expnums_qrtz[ring_size:]

    # define expnum parameters
    expnum_params = {camera: [] for camera in ["b1", "b2", "b3", "r1", "r2", "r3", "z1", "z2", "z3"]}
    for ring, ring_expnums in enumerate([(pri_ldls_expnums, pri_qrtz_expnums), (sec_ldls_expnums, sec_qrtz_expnums)]):
        for channel, expnums in [("b", ring_expnums[0]), ("r", ring_expnums[0]), ("z", ring_expnums[1])]:
            for fiber, expnum in enumerate(expnums):
                if expnum is None:
                    continue
                # define fiber ID
                # TODO: change this to use CALIBFIB header keyword
                fiber_str = f"P{ring+1}-{fiber+1}"
                # get spectrograph where current fiber is plugged
                fiber_par = SLITMAP[SLITMAP["orig_ifulabel"] == fiber_str]
                block_id = int(fiber_par["blockid"][0][1:])-1
                specid = fiber_par["spectrographid"][0]
                # define camera exposure
                camera = f"{channel}{specid}"

                # define exposure parameters
                expnum_params[camera].append((expnum, [block_id], fiber_str))

    # add missing blocks for first exposure
    for camera in expnum_params:
        if len(expnum_params[camera]) == 0:
            continue
        expnums, block_ids, fiber_strs = zip(*expnum_params[camera])
        block_ids = list(zip(*block_ids))[0]
        missing_block_ids = list(set(range(18)) - set(block_ids))
        filled_block_ids = list(block_ids)[0:1] + missing_block_ids
        expnum_params[camera][0] = (expnums[0], sorted(filled_block_ids), fiber_strs[0])

    return expnum_params


def _create_wavelengths_60177(use_fiducial_cals=True, skip_done=True):
    """Reduce arc sequence for MJD = 60177"""
    pixwav = {"z1": np.asarray([
    [88.57, 7488.8712, 1],
    [111.79, 7503.8690, 1],
    [128.59, 7514.6520, 1],
    [161.43, 7535.7741, 0],
    [237.58, 7584.6800, 1],
    [454.78, 7724.6233, 0],
    [840.33, 7967.3400, 1],
    [902.66, 8006.1570, 0],
    [916.05, 8014.7860, 0],
    [1109.70, 8136.4054, 1],
    [1222.22, 8206.3400, 0],
    [1262.80, 8231.6350, 0],
    [1372.72, 8300.3258, 0],
    [1478.12, 8365.7466, 1],
    [1497.42, 8377.6080, 0],
    [1573.44, 8424.6400, 1],
    [1688.33, 8495.3598, 0],
    [1768.50, 8544.6958, 1],
    [1844.78, 8591.2584, 0],
    [1915.81, 8634.6470, 0],
    [2029.76, 8704.1116, 0],
    [2087.89, 8739.3900, 0],
    [2220.99, 8819.4110, 0],
    [2369.50, 8908.7300, 1],
    [2387.51, 8919.5006, 1],
    [2406.58, 8930.8300, 1],
    [2600.14, 9045.4500, 0],
    [2775.63, 9148.6716, 1],
    [2866.74, 9201.7591, 0],
    [2994.24, 9275.5196, 1],
    [3164.63, 9373.3078, 0],
    [3256.41, 9425.3788, 1],
    [3364.76, 9486.6818, 0],
    [3473.13, 9547.4049, 1],
    [3671.55, 9657.7860, 1],
    [3721.46, 9685.3200, 1],
    [3781.02, 9718.1600, 1],
    [3902.11, 9784.5030, 1],
    [3930.05, 9799.7000, 0]]),
    "z2": np.asarray([
    [91.50, 7488.8712, 1],
    [114.69, 7503.8690, 1],
    [131.48, 7514.6520, 1],
    [164.29, 7535.7741, 0],
    [240.36, 7584.6800, 1],
    [457.35, 7724.6233, 0],
    [842.51, 7967.3400, 1],
    [904.78, 8006.1570, 0],
    [918.15, 8014.7860, 0],
    [1111.61, 8136.4054, 1],
    [1224.01, 8206.3400, 0],
    [1264.56, 8231.6350, 0],
    [1374.37, 8300.3258, 0],
    [1479.65, 8365.7466, 1],
    [1498.94, 8377.6080, 0],
    [1574.89, 8424.6400, 1],
    [1689.66, 8495.3598, 0],
    [1769.74, 8544.6958, 1],
    [1845.95, 8591.2584, 0],
    [1916.91, 8634.6470, 0],
    [2030.75, 8704.1116, 0],
    [2088.82, 8739.3900, 0],
    [2221.79, 8819.4110, 0],
    [2370.14, 8908.7300, 1],
    [2388.14, 8919.5006, 1],
    [2407.19, 8930.8300, 1],
    [2600.55, 9045.4500, 0],
    [2775.87, 9148.6716, 1],
    [2866.89, 9201.7591, 0],
    [2994.25, 9275.5196, 1],
    [3164.48, 9373.3078, 0],
    [3256.16, 9425.3788, 1],
    [3364.41, 9486.6818, 0],
    [3472.66, 9547.4049, 1],
    [3670.89, 9657.7860, 1],
    [3720.75, 9685.3200, 1],
    [3780.25, 9718.1600, 1],
    [3901.22, 9784.5030, 1],
    [3929.13, 9799.7000, 0]]),
    "z3": np.asarray([
    [68.74, 7488.8712, 1],
    [92.02, 7503.8690, 1],
    [108.87, 7514.6520, 1],
    [141.79, 7535.7741, 0],
    [218.13, 7584.6800, 1],
    [435.89, 7724.6233, 0],
    [822.42, 7967.3400, 1],
    [884.91, 8006.1570, 0],
    [898.33, 8014.7860, 0],
    [1092.48, 8136.4054, 1],
    [1205.28, 8206.3400, 0],
    [1245.97, 8231.6350, 0],
    [1356.17, 8300.3258, 0],
    [1461.83, 8365.7466, 1],
    [1481.18, 8377.6080, 0],
    [1557.40, 8424.6400, 1],
    [1672.58, 8495.3598, 0],
    [1752.95, 8544.6958, 1],
    [1829.43, 8591.2584, 0],
    [1900.64, 8634.6470, 0],
    [2014.88, 8704.1116, 0],
    [2073.16, 8739.3900, 0],
    [2206.60, 8819.4110, 0],
    [2355.48, 8908.7300, 1],
    [2373.54, 8919.5006, 1],
    [2392.66, 8930.8300, 1],
    [2586.71, 9045.4500, 0],
    [2762.65, 9148.6716, 1],
    [2853.99, 9201.7591, 0],
    [2981.81, 9275.5196, 1],
    [3152.64, 9373.3078, 0],
    [3244.65, 9425.3788, 1],
    [3353.28, 9486.6818, 0],
    [3461.92, 9547.4049, 1],
    [3660.85, 9657.7860, 1],
    [3710.89, 9685.3200, 1],
    [3770.60, 9718.1600, 1],
    [3892.00, 9784.5030, 1],
    [3920.01, 9799.7000, 0]])}

    mjd = 60177
    expnums = range(3453, 3466+1)

    reduce_2d(mjd, use_fiducial_cals=use_fiducial_cals, expnums=expnums, assume_imagetyp="arc", reject_cr=False, skip_done=skip_done)

    frames, _ = md.get_sequence_metadata(mjd=mjd, expnums=expnums, for_cals={"wave"})

    # define master paths for target frames
    calibs = get_calib_paths(mjd, use_fiducial_cals=use_fiducial_cals)

    lamps = [lamp.lower() for lamp in ARC_LAMPS]
    xarc_paths = {"b1": [], "b2": [], "b3": [], "r1": [], "r2": [], "r3": [], "z1": [], "z2": [], "z3": []}
    for lamp in lamps:
        arc_analogs = frames.loc[frames[lamp]].groupby(["camera",])
        for camera in arc_analogs.groups:
            arcs = arc_analogs.get_group((camera,))
            expnum_str = f"{arcs.expnum.min():>08}_{arcs.expnum.max():>08}"

            # define master frame path
            carc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype=f"arc_{lamp}", camera=camera, expnum=expnum_str)
            xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype=f"arc_{lamp}", camera=camera, expnum=expnum_str)
            os.makedirs(os.path.dirname(carc_path), exist_ok=True)
            darc_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="arc", **arc) for arc in arcs.to_dict("records")]
            xarc_paths[camera].append(xarc_path)

            # create master arc (2D image)
            if skip_done and os.path.exists(carc_path):
                log.info(f"skipping {carc_path}, file already exists")
            else:
                image_tasks.create_master_frame(in_images=darc_paths, out_image=carc_path)

            # extract combined (master) arc
            if skip_done and os.path.exists(xarc_path):
                log.info(f"skipping {xarc_path}, file already exists")
            else:
                image_tasks.extract_spectra(in_image=carc_path, out_rss=xarc_path, in_trace=calibs["trace"][camera], in_fwhm=calibs["width"][camera])

    expnum_str = f"{frames.expnum.min():>08}_{frames.expnum.max():>08}"
    for camera in np.sort(frames.camera.unique()):
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=camera, expnum=expnum_str)

        # coadd arcs
        if skip_done and os.path.isfile(xarc_path):
            log.info(f"skipping {xarc_path}, file already exists")
        else:
            rss_tasks.combine_rsss(in_rsss=xarc_paths[camera], out_rss=xarc_path, method="sum")

        # fit wavelength solution
        mwave_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mwave")
        mlsf_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mlsf")
        if skip_done and os.path.isfile(mwave_path) and os.path.isfile(mlsf_path):
            log.info(f"skipping wavelength solution {mwave_path} and {mlsf_path}, files already exists")
        else:
            pixels = pixwav[camera][:, 0] if camera in pixwav else []
            waves = pixwav[camera][:, 1] if camera in pixwav else []
            use_lines = pixwav[camera][:, 2].astype(bool) if camera in pixwav else []
            rss_tasks.determine_wavelength_solution(in_arcs=xarc_paths[camera], out_wave=calibs["wave"][camera], out_lsf=calibs["lsf"][camera],
                                                    pixel=pixels, ref_lines=waves, use_line=use_lines, aperture=12,
                                                    cc_correction=True, cc_max_shift=20, poly_disp=5, poly_fwhm=2, poly_cros=2,
                                                    flux_min=1e-12, fwhm_max=5, rel_flux_limits=[0.001, 1e12])

    mwave_paths = group_calib_paths(calibs["wave"])
    mlsf_paths = group_calib_paths(calibs["lsf"])
    for channel in "brz":
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=channel, expnum=expnum_str)
        harc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="h", imagetype="arc", camera=channel, expnum=expnum_str)

        # stack spectragraphs
        if skip_done and os.path.isfile(xarc_path):
            log.info(f"skipping stacked arc {xarc_path}, file already exists")
        else:
            xarc_paths = sorted(path.expand("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=f"{channel}?", expnum=expnum_str))
            rss_tasks.stack_spectrographs(in_rsss=xarc_paths, out_rss=xarc_path)
        # apply wavelength solution to arcs and rectify
        if skip_done and os.path.isfile(harc_path):
            log.info(f"skipping rectified arc {harc_path}, file already exists")
        else:
            rss_tasks.create_pixel_table(in_rss=xarc_path, out_rss=harc_path, in_waves=mwave_paths[channel], in_lsfs=mlsf_paths[channel])
            rss_tasks.resample_wavelength(in_rss=harc_path, out_rss=harc_path, method="linear", wave_range=SPEC_CHANNELS[channel], wave_disp=0.5)


def _create_fiberflats_60177(mjd):
    """Creates twilight fiberflats from given MJD to MJD = 60177

    Parameters
    ----------
    mjd : int
        MJD of calibration epoch from which the twilight fiberflats will be copied
    """
    mjd_ = 60177

     # define master paths for target frames
    calibs = get_calib_paths(mjd_, use_fiducial_cals=False)
    mwave_paths = group_calib_paths(calibs["wave"])
    mlsf_paths = group_calib_paths(calibs["lsf"])

    log.info(f"going to copy twilight fiberflats from {mjd = } to {mjd_}")
    for channel in "brz":
        log.info(f"preparing wavelength for new fiberflats: {mwave_paths[channel]}, {mlsf_paths[channel]}")
        mwaves = [TraceMask.from_file(mwave_path) for mwave_path in mwave_paths[channel]]
        mwave = TraceMask.from_spectrographs(*mwaves)
        mlsfs = [TraceMask.from_file(mlsf_path) for mlsf_path in mlsf_paths[channel]]
        mlsf = TraceMask.from_spectrographs(*mlsfs)

        fiberflat_path = path.full("lvm_calib", mjd=mjd, kind="fiberflat_twilight", camera=channel)
        log.info(f"loading reference fiberflat from {fiberflat_path}")
        fiberflat = RSS.from_file(fiberflat_path)

        # interpolate fiberflats to mjd_ wavelengths
        log.info("resampling fiberflat to new wavelengths")
        new_fiberflat = copy(fiberflat)
        new_fiberflat._header["MJD"] = mjd
        new_fiberflat._header["SMJD"] = mjd
        for ifiber in range(fiberflat._fibers):
            old_wave = fiberflat._wave[ifiber]
            new_wave = mwave._data[ifiber]
            old_flat = fiberflat._data[ifiber]

            new_fiberflat._data[ifiber] = interpolate.interp1d(old_wave, old_flat, bounds_error=False, fill_value="extrapolate")(new_wave)
            if new_fiberflat._error is not None:
                new_fiberflat._error[ifiber] = interpolate.interp1d(old_wave, fiberflat._error[ifiber], bounds_error=False, fill_value="extrapolate")(new_wave)
            if new_fiberflat._mask is not None:
                new_fiberflat._mask[ifiber] = interpolate.interp1d(old_wave, fiberflat._mask[ifiber].astype(int), bounds_error=False, kind="nearest", fill_value="extrapolate")(new_wave)
                new_fiberflat._mask[ifiber] = new_fiberflat._mask[ifiber].astype(bool)

        # update wavelength traces
        new_fiberflat.set_wave_trace(mwave)
        new_fiberflat.set_lsf_trace(mlsf)
        new_fiberflat.set_wave_array(mwave._data)
        new_fiberflat.set_lsf_array(mlsf._data)

        # store new fiberflat
        new_fiberflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_, camera=channel, kind="mfiberflat_twilight")
        log.info(f"writing new fiberflat to {new_fiberflat_path}")
        new_fiberflat.writeFitsData(new_fiberflat_path)


def messup_frame(mjd, expnum, spec="1", shifts=[1500, 2000, 3500], shift_size=-2, undo_messup=False):
    """ Mess up a frame by shifting the data by a given amount along given rows

    Parameters
    ----------
    mjd : int
        the MJD of the frame
    expnum : int
        the exposure number of the frame
    spec : str
        the spectrograph number
    shifts : list
        the rows to shift
    shift_size : int
        the amount to shift the data by
    undo_messup : bool
        whether to undo the mess up of the frame

    Returns
    -------
    list
        the messed up frames
    """
    specid = f"sp{spec}"
    frames = md.get_frames_metadata(mjd)
    frames.query("expnum == @expnum and spec == @specid", inplace=True)
    log.info(f"messing up frames for spectrograph = {specid}")

    rframe_paths = sorted(path.expand("lvm_raw", hemi="s", camspec=f"?{spec}", mjd=mjd, expnum=expnum))
    rframe_ori_paths = [rframe_path.replace(".fits.gz", "_good.fits.gz") for rframe_path in rframe_paths]
    original_exists = all([os.path.exists(rframe_ori_path) for rframe_ori_path in rframe_ori_paths])
    if undo_messup and original_exists:
        log.info(f"undoing mess up of frames {','.join(rframe_paths)}")
        [copy2(rframe_ori_path, rframe_path) for rframe_ori_path, rframe_path in zip(rframe_ori_paths, rframe_paths)]
        [os.remove(rframe_ori_path) for rframe_ori_path in rframe_ori_paths]
        return
    elif undo_messup:
        log.info(f"cannot undo mess up of frames {','.join(rframe_paths)} because original frames do not exist")
        return

    log.info(f"messing up frames {','.join(rframe_paths)} with {shifts = } and {shift_size = } pixels")
    if not original_exists:
        [copy2(rframe_path, rframe_ori_path) for rframe_path, rframe_ori_path in zip(rframe_paths, rframe_ori_paths)]

    messed_up_frames = []
    for rframe_path, rframe_ori_path in zip(rframe_paths, rframe_ori_paths):
        rframe = loadImage(rframe_ori_path)

        messup_frame = copy(rframe)
        for shift in shifts:
            messup_frame._data[shift:] = np.roll(messup_frame._data[shift:], shift_size, axis=1)

        log.info(f"saving messed up frame to {rframe_path}")
        messup_frame.writeFitsData(rframe_path)

        messed_up_frames.append(messup_frame)

    return messed_up_frames


def fix_raw_pixel_shifts(mjd, expnums=None, ref_expnums=None, use_fiducial_cals=True, specs="123",
                         y_widths=5, wave_list=None, wave_widths=0.6*5, max_shift=10, flat_spikes=11,
                         threshold_spikes=np.inf, shift_rows=None, interactive=False, skip_done=False,
                         display_plots=False):
    """Attempts to fix pixel shifts in a list of raw frames

    Given an MJD and (optionally) exposure numbers, fix the pixel shifts in a
    list of 2D frames. This routine will store the fixed frames in the
    corresponding calibration directory in the `use_fiducial_cals` or by default `mjd`.

    Parameters:
    ----------
    mjd : float
        MJD to reduce
    expnums : list
        List of exposure numbers to look for pixel shifts
    ref_expnums : list
        List of reference exposure numbers to use as reference for good frames, by default None
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    specs : str
        Spectrograph channels
    y_widths : int
        Width of the fibers along y-axis, by default 5
    wave_list : list
        List of lines to use for the wavelength calibration, by default None
    wave_widths : float
        Width of the wavelength axis for the lines, by default 0.6*5
    max_shift : int
        Maximum shift in pixels, by default 10
    flat_spikes : int
        Number of flat spikes, by default 11
    threshold_spikes : float
        Threshold for spikes, by default np.inf
    shift_rows : dict
        Rows to shift, by default None
    interactive : bool
        Interactive mode when report and measured shifts are different, by default False
    skip_done : bool
        Skip pipeline steps that have already been done
    display_plots : bool
        Display plots, by default False
    """

    if shift_rows is None:
        shift_rows = {}
    elif not isinstance(shift_rows, dict):
        raise ValueError("shift_rows must be a dictionary with keys (spec, expnum) and values a list of rows to shift")

    # get target frames & reference frames metadata
    frames = md.get_frames_metadata(mjd)
    if expnums is not None:
        frames.query("expnum in @expnums", inplace=True)
    ref_frames = md.get_frames_metadata(mjd)
    if ref_expnums is not None:
        frames.query("expnum in @ref_expnums", inplace=True)

    if use_fiducial_cals:
        masters_mjd = get_master_mjd(mjd)
        masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    ref_imagetyps = set(ref_frames.imagetyp)
    imagetyps = set(frames.imagetyp)
    if not imagetyps.issubset(ref_imagetyps):
        raise ValueError(f"the following image types are not present in the reference frames: {imagetyps - ref_imagetyps}")

    shifts_path = os.path.join(os.getenv('LVM_SANDBOX'), 'shift_monitor', f'shift_{mjd}.txt')
    shifts_report = {}
    if os.path.isfile(shifts_path):
        shifts_report = _load_shift_report(mjd)

    expnums_grp = frames.groupby("expnum")
    for spec in specs:
        for expnum in expnums_grp.groups:
            frame = expnums_grp.get_group(expnum).iloc[0]

            # find suitable reference frame for current frame
            ref_expnum = _get_reference_expnum(frame, ref_frames)

            rframe_paths = sorted(path.expand("lvm_raw", hemi="s", camspec=f"?{spec}", mjd=mjd, expnum=expnum))
            rframe_paths = [rframe_path for rframe_path in rframe_paths if ".gz" in rframe_path]

            # use fixed reference if exist, else use original raw frame
            cframe_paths = sorted([path.full("lvm_anc", drpver=drpver, tileid=frame.tileid, mjd=mjd, kind="e", imagetype=frame.imagetyp, expnum=ref_expnum, camera=f"{channel}{spec}") for channel in "brz"])
            if not all([os.path.exists(cframe_path) for cframe_path in cframe_paths]):
                cframe_paths = sorted(path.expand("lvm_raw", hemi="s", camspec=f"?{spec}", mjd=mjd, expnum=ref_expnum))
                cframe_paths = [cframe_path for cframe_path in cframe_paths if ".gz" in cframe_path]

            eframe_paths = [path.full("lvm_anc", drpver=drpver, tileid=frame.tileid, mjd=mjd, kind="e", imagetype=frame.imagetyp, expnum=expnum, camera=f"{channel}{spec}") for channel in "brz"]
            mask_2d_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, imagetype="mask2d",
                                     expnum=0, camera=f"sp{spec}", kind="")

            if len(rframe_paths) < 3:
                log.warning(f"skipping {rframe_paths = }, less than 3 files found")
                continue
            if len(cframe_paths) < 3:
                log.warning(f"skipping {cframe_paths = }, less than 3 files found")
                continue

            if use_fiducial_cals:
                mwave_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwave-?{spec}.fits")))
                mtrace_paths = sorted(glob(os.path.join(masters_path, f"lvm-mtrace-?{spec}.fits")))
            else:
                mwave_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mwave", camera=f"?{spec}"))
                mtrace_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mtrace", camera=f"?{spec}"))

            if skip_done and os.path.exists(mask_2d_path):
                log.info(f"skipping {mask_2d_path}, file already exists")
            else:
                image_tasks.select_lines_2d(in_images=cframe_paths, out_mask=mask_2d_path, lines_list=wave_list,
                                            in_cent_traces=mtrace_paths, in_waves=mwave_paths,
                                            y_widths=y_widths, wave_widths=wave_widths,
                                            display_plots=display_plots)

            image_tasks.fix_pixel_shifts(in_images=rframe_paths, out_images=eframe_paths,
                                         ref_images=cframe_paths, in_mask=mask_2d_path, report=shifts_report.get((spec, expnum), None),
                                         flat_spikes=flat_spikes, threshold_spikes=threshold_spikes,
                                         max_shift=max_shift, shift_rows=shift_rows.get((spec, expnum), None),
                                         interactive=interactive, display_plots=display_plots)


def create_detrending_frames(mjd, use_fiducial_cals=True, expnums=None, exptime=None, kind="all", assume_imagetyp=None, reject_cr=True, skip_done=True, keep_ancillary=False):
    """Reduce a sequence of bias/dark/pixelflat frames to produce master frames

    Given a set of MJDs and (optionally) exposure numbers, reduce the
    bias/dark/pixelflat frames. The kind argument specifies which type of frame
    to reduce:

        - bias
        - dark
        - pixflat
        - all (default)

    This routine will store the master of each kind of frame in the
    corresponding calibration directory in the `use_fiducial_cals` or by default in
    the smallest MJD in `mjds`.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    expnums : list
        List of exposure numbers to reduce
    exptime : int
        Exposure time to filter by
    kind : str
        Kind of frame to reduce
    assume_imagetyp : str
        Assume the given imagetyp for all frames
    reject_cr : bool
        Reject cosmic rays
    skip_done : bool
        Skip pipeline steps that have already been done
    keep_ancillary : bool
        Keep ancillary files, by default False
    """
    frames, _ = md.get_sequence_metadata(mjd, expnums=expnums, exptime=exptime, for_cals={"bias", "dark", "pixflat"})

    # filter by target image types
    if kind == "all":
        frames.query("imagetyp in ['bias', 'dark', 'pixflat']", inplace=True)
    elif kind in ["bias", "dark", "pixflat"]:
        frames.query("imagetyp == @kind", inplace=True)
    else:
        raise ValueError(f"Invalid kind: '{kind}'. Must be one of 'bias', 'dark', 'pixflat' or 'all'")

    # preprocess and detrend frames
    reduce_2d(mjd=mjd, use_fiducial_cals=use_fiducial_cals, expnums=set(frames.expnum), exptime=exptime, assume_imagetyp=assume_imagetyp, reject_cr=reject_cr, skip_done=skip_done)

    # define image types to reduce
    imagetypes = set(frames.imagetyp)

    # reduce each image type
    for imagetyp in imagetypes:
        frames_analog = frames.query("imagetyp == @imagetyp").groupby(["imagetyp", "camera"])

        # hack the imagetyp for cases in which the imagetyp is not set or is incorrect (e.g., pixelflats)
        if assume_imagetyp is not None:
            imagetyp = assume_imagetyp

        for keys in frames_analog.groups:
            analogs = frames_analog.get_group(keys)
            frame = analogs.iloc[0].to_dict()

            # combine into master frame
            kwargs = get_config_options('reduction_steps.create_master_frame', imagetyp)
            log.info(f'custom configuration parameters for create_master_frame: {repr(kwargs)}')
            mframe_path = path.full("lvm_master", drpver=drpver, tileid=frame["tileid"], mjd=mjd, kind=f'm{imagetyp}', camera=frame["camera"])
            if skip_done and os.path.isfile(mframe_path):
                log.info(f"skipping {mframe_path}, file already exist")
            else:
                os.makedirs(os.path.dirname(mframe_path), exist_ok=True)
                dframe_paths = [path.full("lvm_anc", drpver=drpver, kind="d" if imagetyp != "bias" else "p", imagetype=imagetyp, **frame) for frame in analogs.to_dict("records")]
                image_tasks.create_master_frame(in_images=dframe_paths, out_image=mframe_path, **kwargs)


    # ancillary paths clean up
    if not keep_ancillary:
        _clean_ancillary(mjd=mjd, expnums=expnums, kind=kind)


def create_pixelmasks(mjd, use_fiducial_cals=True, dark_expnums=None, pixflat_expnums=None,
                      short_exptime=900, long_exptime=3600, pixflat_exptime=5,
                      ignore_pixflats=True, keep_ancillary=False):
    """Create pixel mask from master pixelflat and/or dark frames

    Given a set of MJDs and (optionally) exposure numbers, create a pixel mask
    from the master pixelflat and/or dark frames. This routine will store the
    master pixelmask in the corresponding calibration directory in the
    `masters_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding detrended pixelflat and/or dark frames do not exist, they
    will be created first. Otherwise they will be read from disk. If
    `ignore_pixflats` is True, then the pixelflat frames will not be used to create the
    pixel mask.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    dark_expnums : list
        List of dark exposure numbers
    pixflat_expnums : list
        List of pixelflat exposure numbers
    short_exptime : int
        Short exposure time for dark frames
    long_exptime : int
        Long exposure time for dark frames
    pixflat_exptime : int
        Exposure time for pixelflat frames
    ignore_pixflats : bool
        Ignore pixelflat frames when creating pixel mask
    keep_ancillary : bool
        Keep ancillary files, by default False

    """
    if dark_expnums is not None and pixflat_expnums is not None:
        expnums = np.concatenate([dark_expnums, pixflat_expnums])
    else:
        expnums = None
    frames, _ = md.get_sequence_metadata(mjd, expnums=expnums, for_cals={"dark", "pixflat"})

    darks = frames.query("imagetyp == 'dark' and exptime == @short_exptime or exptime == @long_exptime", inplace=True)
    pixflats = frames.query("imagetyp == 'dark' or imagetyp == 'pixelflat' and exptime == @pixflat_exptime", inplace=True)

    # reduce darks
    reduce_2d(mjd=mjd, use_fiducial_cals=use_fiducial_cals, expnums=set(darks.expnum))

    ddark_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="dark", **dark) for dark in darks.to_dict("records")]
    darks["ddark_path"] = ddark_paths
    cam_groups = darks.groupby(["camera", "exptime"])
    for cam, exptime in cam_groups.groups:
        ddark_paths_cam = cam_groups.get_group((cam, exptime))["ddark_path"]

        mdark_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind=f"mdark-{int(exptime)}s", camera=cam)
        image_tasks.create_master_frame(in_images=ddark_paths_cam, out_image=mdark_path)

    # reduce pixflats
    if not ignore_pixflats:
        reduce_2d(mjd=mjd, use_fiducial_cals=use_fiducial_cals, expnums=set(pixflats.expnum),
                replace_with_nan=False, assume_imagetyp="pixelflat", reject_cr=False)
        dflat_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="pixelflat", **pixflat) for pixflat in pixflats.to_dict("records")]
        pixflats["dflat_path"] = dflat_paths

        cam_groups = pixflats.groupby("camera")
        for cam in cam_groups.groups:
            dflat_paths_cam = cam_groups.get_group(cam)["dflat_path"]

            # define output combined pixelflat path
            mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mpixflat", camera=cam)

            image_tasks.create_master_frame(in_images=dflat_paths_cam, out_image=mflat_path)

            # normalize pixelflat background
            pixflat_img = image_tasks.loadImage(mflat_path)
            pixflat_img = pixflat_img / pixflat_img.medianImg(size=20)
            pixflat_img.writeFitsData(mflat_path)

        dflat_groups = groupby(dflat_paths, key=lambda s: os.path.basename(s).split("-")[2])

        # create pixel mask
        for camera, group in dflat_groups:
            medians = []
            group = list(group)
            # compute medians of all detrended pixel flats
            for dflat_path in group:
                medians.append(np.nanmedian(image_tasks.loadImage(dflat_path)._data))

            # sort paths by median values
            idx = np.argsort(medians)
            dflat_group = np.asarray(group)[idx]

            # pick two most different pixelflats
            flat_a = dflat_group[0]
            flat_b = dflat_group[-1]

            mdark_short_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind=f"mdark-{short_exptime}s", camera=camera)
            mdark_long_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind=f"mdark-{long_exptime}s", camera=camera)
            mpixmask_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mpixmask", camera=camera)
            image_tasks.create_pixelmask(in_short_dark=mdark_short_path, in_long_dark=mdark_long_path,
                                        in_flat_a=flat_a, in_flat_b=flat_b,
                                        out_pixmask=mpixmask_path)
    else:
        log.info("Ignoring pixelflats when creating pixel mask")
        mdark_short_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind=f"mdark-{short_exptime}s", camera=camera)
        mdark_long_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind=f"mdark-{long_exptime}s", camera=camera)
        mpixmask_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mpixmask", camera=camera)
        image_tasks.create_pixelmask(in_short_dark=mdark_short_path, in_long_dark=mdark_long_path, out_pixmask=mpixmask_path)

    # ancillary paths clean up
    if not keep_ancillary:
        _clean_ancillary(mjd=mjd, expnums=expnums)


def create_nightly_traces(mjd, use_fiducial_cals=True, expnums_ldls=None, expnums_qrtz=None,
                        fit_poly=True, poly_deg_amp=5, poly_deg_cent=4, poly_deg_width=5,
                        skip_done=True):
    if expnums_ldls is not None and expnums_qrtz is not None:
        expnums = np.concatenate([expnums_ldls, expnums_qrtz])
    else:
        expnums = None
    frames, _ = md.get_sequence_metadata(mjd, expnums=expnums, for_cals={"flat"})

    # run 2D reduction on flats: preprocessing, detrending
    reduce_2d(mjd, use_fiducial_cals=use_fiducial_cals, expnums=expnums, reject_cr=False, skip_done=skip_done)

    for channel, lamp in MASTER_CON_LAMPS.items():
        counts_threshold = 5000 if lamp == "ldls" else 10000

        # select dome flats accoding to current channel-lamp combination
        flats_analogs = frames.loc[(frames[lamp])&(frames["camera"].str.startswith(channel))].groupby(["camera",])
        for camera in flats_analogs.groups:
            flats = flats_analogs.get_group((camera,))

            # combine dome flats
            expnum_str = f"{flats.expnum.min():>08}_{flats.expnum.max():>08}"
            dflat_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="flat", **flat) for flat in flats.to_dict("records")]
            cflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype="flat", camera=camera, expnum=expnum_str)
            os.makedirs(os.path.dirname(cflat_path), exist_ok=True)
            image_tasks.create_master_frame(in_images=dflat_paths, out_image=cflat_path)

            # define paths
            lflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="l", imagetype="flat", camera=camera, expnum=expnum_str)
            dstray_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="stray", camera=camera, expnum=expnum_str)
            dratio_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="ratio", camera=camera, expnum=expnum_str)

            cent_guess_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="ntrace_guess", camera=camera)
            flux_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="namp", camera=camera)
            cent_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="ntrace", camera=camera)
            fwhm_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="nwidth", camera=camera)
            model_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="nmodel", camera=camera)

            # first centroids trace
            if skip_done and os.path.isfile(cent_guess_path):
                log.info(f"skipping {cent_guess_path}, file already exist")
            else:
                log.info(f"going to trace centroids fibers in {camera}")
                centroids, img = image_tasks.trace_centroids(in_image=cflat_path, out_trace_cent=cent_guess_path,
                                                             correct_ref=True, median_box=(1,10), coadd=20, counts_threshold=counts_threshold,
                                                             max_diff=1.5, guess_fwhm=2.5, method="gauss", ncolumns=140,
                                                             fit_poly=fit_poly, poly_deg=poly_deg_cent,
                                                             interpolate_missing=True)

            # subtract stray light only if imagetyp is flat
            if skip_done and os.path.isfile(lflat_path):
                log.info(f"skipping {lflat_path}, file already exist")
            else:
                image_tasks.subtract_straylight(in_image=cflat_path, out_image=lflat_path, out_stray=dstray_path,
                                                in_cent_trace=cent_guess_path, select_nrows=(5,5), use_weights=True,
                                                aperture=15, smoothing=400, median_box=101, gaussian_sigma=20.0, parallel=0)

            if skip_done and os.path.isfile(flux_path) and os.path.isfile(cent_path) and os.path.isfile(fwhm_path):
                log.info(f"skipping {flux_path}, {cent_path} and {fwhm_path}, files already exist")
            else:
                log.info(f"going to trace fibers in {camera}")
                centroids, trace_cent_fit, trace_flux_fit, trace_fwhm_fit, img_stray, model, mratio = image_tasks.trace_fibers(
                    in_image=lflat_path,
                    out_trace_amp=flux_path, out_trace_cent=cent_path, out_trace_fwhm=fwhm_path, out_model=model_path,
                    in_trace_cent_guess=cent_guess_path,
                    median_box=(1,10), coadd=20,
                    counts_threshold=counts_threshold, max_diff=1.5, guess_fwhm=2.5,
                    ncolumns=40, fwhm_limits=(1.5, 4.5),
                    fit_poly=fit_poly, interpolate_missing=True,
                    poly_deg=(poly_deg_amp, poly_deg_cent, poly_deg_width)
                )

            # eval model continuum and ratio
            if skip_done and os.path.isfile(model_path) and os.path.isfile(dratio_path):
                log.info(f"skipping {model_path}, file already exist")
            else:
                log.info(f"going to create model image and mode/exposure ratio in {camera}")
                if "trace_cent_fit" not in locals():
                    trace_cent_fit = TraceMask.from_file(cent_path)
                    trace_flux_fit = TraceMask.from_file(flux_path)
                    trace_fwhm_fit = TraceMask.from_file(fwhm_path)
                    img_stray = loadImage(lflat_path)
                    img_stray.setData(data=np.nan_to_num(img_stray._data), error=np.nan_to_num(img_stray._error))
                    img_stray = img_stray.replaceMaskMedian(1, 10, replace_error=None)
                    img_stray._data = np.nan_to_num(img_stray._data)
                    img_stray = img_stray.medianImg((1,10), propagate_error=True)
                    img_stray = img_stray.convolveImg(np.ones((1, 20), dtype="uint8"))
                model, ratio = img_stray.eval_fiber_model(trace_cent_fit, trace_fwhm_fit, trace_flux_fit)
                model.writeFitsData(model_path)
                ratio.writeFitsData(dratio_path)


def create_traces(mjd, cameras=CAMERAS, use_fiducial_cals=True, expnums_ldls=None, expnums_qrtz=None,
                  fit_poly=True, poly_deg_amp=5, poly_deg_cent=4, poly_deg_width=5,
                  skip_done=True):
    """Create traces from master dome flats

    Given a set of MJDs and (optionally) exposure numbers, create traces from
    the master dome flats. This routine will store the master traces in the
    corresponding calibration directory in the `masters_mjd` or by default in
    the smallest MJD in `mjds`.

    If the corresponding master dome flats do not exist, they will be created
    first. Otherwise they will be read from disk.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    cameras : list or tuple, optional
        List of cameras (e.g., b2, z3) to create traces for
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    expnums_ldls : list
        List of exposure numbers for LDLS dome flats
    expnums_qrtz : list
        List of exposure numbers for quartz dome flats
    fit_poly : bool, optional
        Fit polynomials to traces, by default True
    poly_deg_amp : int, optional
        Degree of the polynomial to fit to the amplitude, by default 5
    poly_deg_cent : int, optional
        Degree of the polynomial to fit to the centroids, by default 4
    poly_deg_width : int, optional
        Degree of the polynomial to fit to the widths, by default 5
    skip_done : bool, optional
        Skip pipeline steps that have already been done, by default True
    """
    if expnums_ldls is not None and expnums_qrtz is not None:
        expnums = np.concatenate([expnums_ldls, expnums_qrtz])
    else:
        expnums = None
    frames, _ = md.get_sequence_metadata(mjd, expnums=expnums, cameras=cameras, for_cals={"trace"})
    tileid = frames.tileid.iloc[0]

    # run 2D reduction on flats: preprocessing, detrending
    reduce_2d(mjd, use_fiducial_cals=use_fiducial_cals, expnums=expnums, cameras=cameras, reject_cr=False, skip_done=skip_done)

    # iterate through exposures with std fibers exposed
    for camera in cameras:
        # initialize fiber traces
        mamps, mcents, mwidths = {}, {}, {}
        mamps[camera] = TraceMask()
        mamps[camera].createEmpty(data_dim=(648, 4086), poly_deg=poly_deg_amp)
        mcents[camera] = TraceMask()
        mcents[camera].createEmpty(data_dim=(648, 4086), poly_deg=poly_deg_cent)
        mwidths[camera] = TraceMask()
        mwidths[camera].createEmpty(data_dim=(648, 4086), poly_deg=poly_deg_width)

        expnums = expnums_qrtz if camera[0] == "z" else expnums_ldls
        counts_threshold = 10000 if camera[0] == "z" else 5000

        # select fibers in current spectrograph
        fibermap = SLITMAP[SLITMAP["spectrographid"] == int(camera[1])]

        exposed_stds, unexposed_stds = get_exposed_std_fiber(mjd=mjd, expnums=expnums, camera=camera)
        for expnum, (std_fiberid, block_idxs) in exposed_stds.items():
            # define paths
            dflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="flat", camera=camera, expnum=expnum)
            lflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="l", imagetype="flat", camera=camera, expnum=expnum)
            flux_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="flux", camera=camera, expnum=expnum)
            cent_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="cent", camera=camera, expnum=expnum)
            cent_guess_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="cent_guess", camera=camera, expnum=expnum)
            dstray_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="stray", camera=camera, expnum=expnum)
            fwhm_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="fwhm", camera=camera, expnum=expnum)
            dmodel_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="model", camera=camera, expnum=expnum)
            dratio_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="ratio", camera=camera, expnum=expnum)

            # first centroids trace
            if skip_done and os.path.isfile(cent_guess_path):
                log.info(f"skipping {cent_guess_path}, file already exist")
            else:
                log.info(f"going to trace all fibers in {camera}")
                centroids, img = image_tasks.trace_centroids(in_image=dflat_path, out_trace_cent=cent_guess_path,
                                                            correct_ref=True, median_box=(1,10), coadd=20, counts_threshold=counts_threshold,
                                                            max_diff=1.5, guess_fwhm=2.5, method="gauss", ncolumns=140,
                                                            fit_poly=fit_poly, poly_deg=poly_deg_cent,
                                                            interpolate_missing=True)

            # subtract stray light only if imagetyp is flat
            if skip_done and os.path.isfile(lflat_path):
                log.info(f"skipping {lflat_path}, file already exist")
            else:
                image_tasks.subtract_straylight(in_image=dflat_path, out_image=lflat_path, out_stray=dstray_path,
                                                in_cent_trace=cent_guess_path, select_nrows=(5,5), use_weights=True,
                                                aperture=15, smoothing=400, median_box=101, gaussian_sigma=20.0, parallel=0)

            if skip_done and os.path.isfile(cent_path) and os.path.isfile(flux_path) and os.path.isfile(fwhm_path):
                log.info(f"skipping {cent_path}, {flux_path} and {fwhm_path}, file already exist")
            else:
                log.info(f"going to trace std fiber {std_fiberid} in {camera} within {block_idxs = }")
                centroids, trace_cent_fit, trace_flux_fit, trace_fwhm_fit, img_stray, model, mratio = image_tasks.trace_fibers(
                    in_image=lflat_path,
                    out_trace_amp=flux_path, out_trace_cent=cent_path, out_trace_fwhm=fwhm_path,
                    in_trace_cent_guess=cent_guess_path,
                    median_box=(1,10), coadd=20,
                    counts_threshold=counts_threshold, max_diff=1.5, guess_fwhm=2.5,
                    ncolumns=40, iblocks=block_idxs, fwhm_limits=(1.5, 4.5),
                    fit_poly=fit_poly, interpolate_missing=False, poly_deg=(poly_deg_amp, poly_deg_cent, poly_deg_width)
                )

                # update master traces
                log.info(f"{camera = }, {expnum = }, {std_fiberid = }")
                select_block = np.isin(fibermap["blockid"], [f"B{id+1}" for id in block_idxs])
                if fit_poly:
                    mamps[camera]._coeffs[select_block] = trace_flux_fit._coeffs[select_block]
                    mcents[camera]._coeffs[select_block] = trace_cent_fit._coeffs[select_block]
                    mwidths[camera]._coeffs[select_block] = trace_fwhm_fit._coeffs[select_block]
                else:
                    mamps[camera]._coeffs = None
                    mcents[camera]._coeffs = None
                    mwidths[camera]._coeffs = None
                mamps[camera]._data[select_block] = trace_flux_fit._data[select_block]
                mcents[camera]._data[select_block] = trace_cent_fit._data[select_block]
                mwidths[camera]._data[select_block] = trace_fwhm_fit._data[select_block]
                mamps[camera]._mask[select_block] = False
                mcents[camera]._mask[select_block] = False
                mwidths[camera]._mask[select_block] = False

        mamp_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera, kind="mamp")
        mcent_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera, kind="mtrace")
        mwidth_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera, kind="mwidth")
        os.makedirs(os.path.dirname(mamp_path), exist_ok=True)
        if skip_done and os.path.isfile(mcent_path) and os.path.isfile(mamp_path) and os.path.isfile(mwidth_path):
            log.info(f"skipping {mcent_path}, {mamp_path} and {mwidth_path}, files already exist")
        else:
            # masking bad fibers
            bad_fibers = fibermap["fibstatus"] == 1
            mamps[camera]._mask[bad_fibers] = True
            mcents[camera]._mask[bad_fibers] = True
            mwidths[camera]._mask[bad_fibers] = True
            # masking untraced standard fibers (two cases: 1. not set for tracing and 2. flagged during tracing)
            untraced_fibers = np.isin(fibermap["orig_ifulabel"].value, unexposed_stds)
            mamps[camera]._mask[untraced_fibers] = True
            mcents[camera]._mask[untraced_fibers] = True
            mwidths[camera]._mask[untraced_fibers] = True
            failed_fibers = (np.nansum(mcents[camera]._data, axis=1) == 0)|(np.nansum(mwidths[camera]._data, axis=1) == 0)
            mamps[camera]._mask[failed_fibers] = True
            mcents[camera]._mask[failed_fibers] = True
            mwidths[camera]._mask[failed_fibers] = True

            # interpolate master traces in missing fibers
            if fit_poly:
                mamps[camera].interpolate_coeffs()
                mcents[camera].interpolate_coeffs()
                mwidths[camera].interpolate_coeffs()
            else:
                mamps[camera].interpolate_data(axis="Y", extrapolate=True)
                mcents[camera].interpolate_data(axis="Y", extrapolate=True)
                mwidths[camera].interpolate_data(axis="Y", extrapolate=True)

            # reset mask to propagate broken fibers
            mamps[camera]._mask[bad_fibers] = True
            mcents[camera]._mask[bad_fibers] = True
            mwidths[camera]._mask[bad_fibers] = True

            # save master traces
            mamps[camera].writeFitsData(mamp_path)
            mcents[camera].writeFitsData(mcent_path)
            mwidths[camera].writeFitsData(mwidth_path)

            # eval model continuum and ratio
            model, ratio = img_stray.eval_fiber_model(mcents[camera], mwidths[camera], mamps[camera])
            model.writeFitsData(dmodel_path)
            ratio.writeFitsData(dratio_path)


def create_dome_fiberflats(mjd, expnums_ldls, expnums_qrtz, use_fiducial_cals=True, kind="longterm", skip_done=True):
    """Create fiberflats from dome flats

    Parameters
    ----------
    mjd : int
        MJD to reduce
    expnums_ldls : list
        List of exposure numbers for LDLS dome flats
    expnums_qrtz : list
        List of exposure numbers for quartz dome flats
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    kind : str, optional
        Kind of calibration frames to produce, by default 'longterm'
    skip_done : bool
        Skip pipeline steps that have already been done
    """

    expnums = np.concatenate([expnums_ldls, expnums_qrtz])
    frames, _ = md.get_sequence_metadata(mjd, expnums=expnums, for_cals={"flat"})

    # define master paths for target frames
    calibs = get_calib_paths(mjd, use_fiducial_cals=use_fiducial_cals)
    for flavor in {"trace", "width", "wave", "lsf"}:
        calibs[flavor] = group_calib_paths(calibs[flavor])

    for channel, lamp in MASTER_CON_LAMPS.items():
        # read original combined dome flats and run extraction
        flats = frames.loc[(frames[lamp])&(frames["camera"].str.startswith(channel))]
        expnum_str = f"{flats.expnum.min():>08}_{flats.expnum.max():>08}"
        xflat_paths = []
        for i, specid in enumerate("123"):
            camera = f"{channel}{specid}"
            cflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype="flat", camera=camera, expnum=expnum_str)
            xflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="flat", camera=camera, expnum=expnum_str)
            if skip_done and os.path.isfile(xflat_path):
                log.info(f"skipping {xflat_path}, file already exists")
            else:
                image_tasks.extract_spectra(in_image=cflat_path, out_rss=xflat_path, in_trace=calibs["trace"][camera], in_fwhm=calibs["width"][camera], in_model=calibs["model"][camera])
            xflat_paths.append(xflat_path)
        xflat = RSS.from_spectrographs(*[RSS.from_file(xflat_path) for xflat_path in xflat_paths])

        # read mamp files
        if use_fiducial_cals:
            mamp_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mamp", camera=f"{channel}?"))
        else:
            mamp_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="namp", camera=f"{channel}?"))
        mamp = TraceMask.from_spectrographs(*[TraceMask.from_file(mamp_path) for mamp_path in mamp_paths])

        if kind == "longterm":
            mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mfiberflat_dome", camera=channel)
        else:
            mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="nfiberflat_dome", camera=channel)

        # read calibrations
        mcent = TraceMask.from_spectrographs(*[TraceMask.from_file(mtrace_path) for mtrace_path in calibs["trace"][channel]])
        mwidth = TraceMask.from_spectrographs(*[TraceMask.from_file(mwidth_path) for mwidth_path in calibs["width"][channel]])
        mwave = TraceMask.from_spectrographs(*[TraceMask.from_file(mwave_path) for mwave_path in calibs["wave"][channel]])
        mlsf = TraceMask.from_spectrographs(*[TraceMask.from_file(mlsf_path) for mlsf_path in calibs["lsf"][channel]])
        # normalize by median fiber
        fflat = RSS(data=mamp._data, error=np.sqrt(mamp._data), mask=xflat._mask, wave_trace=mwave, lsf_trace=mlsf, header=xflat._header)
        fflat = fflat.rectify_wave(method="linear", wave_range=SPEC_CHANNELS[channel], wave_disp=0.5)
        median_fiber = np.nanmedian(fflat._data, axis=0)
        fflat._data = fflat._data / median_fiber
        fflat.set_wave_trace(mwave)
        fflat.set_lsf_trace(mlsf)
        fflat = fflat.to_native_wave(method="linear", interp_density=False, return_density=False)
        fflat.writeFitsData(mflat_path)
        # create lvmFlat object
        lvmflat = lvmFlat(data=xflat._data / fflat._data, error=xflat._error, mask=xflat._mask, header=xflat._header,
                              cent_trace=mcent, width_trace=mwidth,
                              wave_trace=mwave, lsf_trace=mlsf,
                              superflat=fflat._data, slitmap=SLITMAP)
        lvmflat.writeFitsData(path.full("lvm_frame", mjd=mjd, tileid=11111, drpver=drpver, expnum=expnum_str, kind=f'Flat-{channel}'))


def create_twilight_fiberflats(mjd: int, use_fiducial_cals: bool = True, expnums: List[int] = None, median_box: int = 10, niter: bool = 1000,
                      threshold: Tuple[float,float]|float = (0.5,1.5), nknots: bool = 50,
                      b_mask: List[Tuple[float,float]] = MASK_BANDS["b"],
                      r_mask: List[Tuple[float,float]] = MASK_BANDS["r"],
                      z_mask: List[Tuple[float,float]] = MASK_BANDS["z"],
                      kind: str = "longterm",
                      skip_done: bool = False,
                      display_plots: bool = False) -> Dict[str, RSS]:
    """Reduce the twilight sequence and produces master twilight flats

    Given a sequence of twilight exposures, this function reduces them and
    produces master twilight flats for each camera.

    Parameters
    ----------
    mjd : int
        MJD to reduce
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    expnums : list
        List of twilight exposure numbers
    median_box : int, optional
        Size of the median filter box, by default 5
    niter : int, optional
        Number of iterations to fit the continuum, by default 1000
    threshold : float, optional
        Threshold to mask outliers, by default 0.5
    nknots : int, optional
        Number of knots for the spline fitting, by default 50
    b_mask : list, optional
        List of wavelength bands to mask in the blue channel, by default []
    r_mask : list, optional
        List of wavelength bands to mask in the red channel, by default []
    z_mask : list, optional
        List of wavelength bands to mask in the NIR channel, by default []
    use_master_centroids : bool, optional
        Use master centroids to trace the fibers, by default False
    kind : str, optional
        Kind of calibration frames to produce, by default 'longterm'
    skip_done : bool, optional
        Skip files that already exist, by default False
    display_plots : bool, optional
        Display plots, by default False

    Returns
    -------
    mfflats : dict
        Dictionary with the master twilight flats for each channel
    """
    # get metadata
    flats, _ = md.get_sequence_metadata(mjd, expnums=expnums, for_cals={"fiberflat"})

    # 2D reduction of twilight sequence
    reduce_2d(mjd=mjd, use_fiducial_cals=use_fiducial_cals, expnums=flats.expnum.unique(), reject_cr=False, skip_done=skip_done)

    # define master paths for target frames
    calibs = get_calib_paths(mjd, use_fiducial_cals=use_fiducial_cals)

    for flat in flats.to_dict("records"):
        camera = flat["camera"]

        # extract 1D spectra for each frame
        dflat_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype="flat", **flat)
        lflat_path = path.full("lvm_anc", drpver=drpver, kind="l", imagetype="flat", **flat)
        xflat_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype="flat", **flat)
        stray_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype="stray", **flat)

        # subtract stray light only if imagetyp is flat
        if skip_done and os.path.isfile(lflat_path):
            log.info(f"skipping {lflat_path}, file already exist")
        else:
            image_tasks.subtract_straylight(in_image=dflat_path, out_image=lflat_path, out_stray=stray_path,
                                            in_cent_trace=calibs["trace_guess"][camera], select_nrows=(5,5), use_weights=True,
                                            aperture=15, smoothing=400, median_box=101, gaussian_sigma=20.0, parallel=0)

        if skip_done and os.path.isfile(xflat_path):
            log.info(f"skipping {xflat_path}, file already exist")
        else:
            image_tasks.extract_spectra(in_image=lflat_path, out_rss=xflat_path,
                                        in_trace=calibs["trace"][camera], in_fwhm=calibs["width"][camera], in_model=calibs["model"][camera],
                                        method="optimal")

    # group calibs
    for flavor in ["trace", "width", "wave", "lsf"]:
        calibs[flavor] = group_calib_paths(calibs[flavor])

    # decompose twilight spectra into sun continuum and twilight components
    channels = "brz"
    mask_bands = dict(zip(channels, [b_mask, r_mask, z_mask]))
    mfflats = dict.fromkeys(channels)
    flat_channels = flats.groupby(flats.camera.str.__getitem__(0))
    tileid = flats.tileid.min()
    for channel in channels:
        flat_expnums = flat_channels.get_group(channel).groupby("expnum")
        fflats = []
        for expnum in flat_expnums.groups:
            flat = flat_expnums.get_group(expnum).iloc[0]

            xflat_paths = sorted(path.expand("lvm_anc", drpver=drpver, kind="x", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=f"{channel}?", expnum=expnum))
            fflat_flatfielded_path = path.full("lvm_anc", drpver=drpver, kind="flatfielded_",
                                   imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"],
                                   camera=channel, expnum=expnum)
            fflat_path = path.full("lvm_anc", drpver=drpver, kind="f",
                                   imagetype="flat", tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)

            # spectrograph stack xflats
            xflat_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            rss_tasks.stack_spectrographs(in_rsss=xflat_paths, out_rss=xflat_path)

            # calibrate in wavelength
            wflat_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            rss_tasks.create_pixel_table(in_rss=xflat_path, out_rss=wflat_path,
                                         in_waves=calibs["wave"][channel], in_lsfs=calibs["lsf"][channel])

            # rectify in wavelength
            hflat_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            rss_tasks.resample_wavelength(in_rss=wflat_path, out_rss=hflat_path, wave_disp=0.5, wave_range=SPEC_CHANNELS[channel])

            # fit fiber throughput
            fflat = fit_fiberflat(in_twilight=hflat_path, out_flat=fflat_path, out_rss=fflat_flatfielded_path, median_box=median_box, niter=niter,
                                   threshold=threshold, mask_bands=mask_bands.get(channel, []),
                                   display_plots=display_plots, nknots=nknots)

            # fit gradient and remove it
            gflat_path = path.full("lvm_anc", drpver=drpver, kind="g", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            remove_field_gradient(in_hflat=fflat_flatfielded_path, out_gflat=gflat_path, wrange=SPEC_CHANNELS[channel])

            # load fiber and wavelength traces
            mcent = TraceMask.from_spectrographs(*[TraceMask.from_file(master_cent) for master_cent in calibs["trace"][channel]])
            mwidth = TraceMask.from_spectrographs(*[TraceMask.from_file(master_width) for master_width in calibs["width"][channel]])
            mwave = TraceMask.from_spectrographs(*[TraceMask.from_file(master_wave) for master_wave in calibs["wave"][channel]])
            mlsf = TraceMask.from_spectrographs(*[TraceMask.from_file(master_lsf) for master_lsf in calibs["lsf"][channel]])

            fflat_flatfielded = RSS.from_file(fflat_flatfielded_path)
            gflat = RSS.from_file(gflat_path)
            gradient = copy(gflat)
            gradient._data = fflat_flatfielded._data / gflat._data
            fflat._data /= gradient._data
            fflat._mask |= np.isnan(fflat._data)
            fflat = fflat.interpolate_data(axis="X")
            # fflat = fflat.interpolate_data(axis="Y")
            fflat.writeFitsData(fflat_path)
            fflat.set_wave_trace(mwave)
            fflat.set_lsf_trace(mlsf)
            fflat = fflat.to_native_wave(method="linear", interp_density=False, return_density=False)
            fflats.append(fflat)

            # build lvmFlat
            twilight = RSS.from_file(fflat_flatfielded_path)
            twilight.set_wave_trace(mwave)
            twilight.set_lsf_trace(mlsf)
            twilight = twilight.to_native_wave(method="linear", interp_density=True, return_density=False)
            lvmflat = lvmFlat(data=twilight._data, error=twilight._error, mask=twilight._mask, header=twilight._header,
                              cent_trace=mcent, width_trace=mwidth,
                              wave_trace=mwave, lsf_trace=mlsf,
                              superflat=fflat._data, slitmap=twilight._slitmap)
            lvmflat.writeFitsData(path.full("lvm_frame", mjd=mjd, tileid=tileid, drpver=drpver, expnum=expnum, kind=f'Flat-{channel}'))

        mfflat = combine_twilight_sequence(fflats=fflats)
        mflat_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=mjd, kind="mfiberflat_twilight", camera=channel)
        mfflat.writeFitsData(mflat_path)
        mfflats[channel] = mfflat

    return mfflats


def create_illumination_corrections(mjd, use_fiducial_cals=True, expnums=None):
    """Create illumination corrections from master dome and twilight flats

    Given a set of MJDs and (optionally) exposure numbers, create illumination
    corrections from the master dome and twilight flats. This routine will
    store the master illumination corrections in the corresponding calibration
    directory in the `masters_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding master dome and twilight flats do not exist, they
    will be created first. Otherwise they will be read from disk.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    expnums : list
        List of exposure numbers to reduce
    """
    # read master fiber flats (dome and twilight)
    # calculate ratio twilight/dome
    raise NotImplementedError("create_illumination_corrections")


def create_wavelengths(mjd, use_fiducial_cals=True, expnums=None, kind="longterm", skip_done=True):
    """Reduces an arc sequence to create master wavelength solutions

    Given a set of MJDs and (optionally) exposure numbers, create wavelength
    solutions from the master arcs. This routine will store the master
    wavelength solutions in the corresponding calibration directory in the
    `masters_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding master arcs do not exist, they will be created first.
    Otherwise they will be read from disk.

    Parameters
    ----------
    mjd : int
        MJD to reduce
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    expnums : list
        List of exposure numbers to reduce
    kind : str, optional
        Kind of calibration frames to produce, by default 'longterm'
    skip_done : bool
        Skip pipeline steps that have already been done
    """
    frames, _ = md.get_sequence_metadata(mjd, expnums=expnums, for_cals={"wave"})

    reduce_2d(mjd, use_fiducial_cals=use_fiducial_cals, expnums=expnums, assume_imagetyp="arc", reject_cr=False, skip_done=skip_done)

    # define master paths for target frames
    calibs = get_calib_paths(mjd, use_fiducial_cals=use_fiducial_cals)

    expnum_str = f"{frames.expnum.min():>08}-{frames.expnum.max():>08}"
    arc_analogs = frames.groupby(["camera",])
    for camera in arc_analogs.groups:
        arcs = arc_analogs.get_group((camera,))

        # define product paths
        carc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype="arc", camera=camera, expnum=expnum_str)
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=camera, expnum=expnum_str)
        if kind == "longterm":
            mwave_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mwave")
            mlsf_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mlsf")
        else:
            mwave_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="nwave")
            mlsf_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="nlsf")
        os.makedirs(os.path.dirname(carc_path), exist_ok=True)

        # combine individual arcs into master arc
        if skip_done and os.path.isfile(carc_path):
            log.info(f"skipping combined arc {carc_path}, file already exists")
        else:
            darc_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="arc", **arc) for arc in arcs.to_dict("records")]
            image_tasks.create_master_frame(in_images=darc_paths, out_image=carc_path, batch_size=48)

        # extract arc
        if skip_done and os.path.isfile(xarc_path):
            log.info(f"skipping extracted arc {xarc_path}, file already exists")
        else:
            image_tasks.extract_spectra(in_image=carc_path, out_rss=xarc_path,
                                        in_trace=calibs["trace"][camera], in_fwhm=calibs["width"][camera], in_model=calibs["model"][camera],
                                        method="optimal")

        # fit wavelength solution
        if skip_done and os.path.isfile(mwave_path) and os.path.isfile(mlsf_path):
            log.info(f"skipping wavelength solution {mwave_path} and {mlsf_path}, files already exists")
        else:
            rss_tasks.determine_wavelength_solution(in_arcs=xarc_path, out_wave=mwave_path, out_lsf=mlsf_path, aperture=12,
                                                    cc_correction=True, cc_max_shift=20, poly_disp=5, poly_fwhm=2, poly_cros=2,
                                                    flux_min=1e-12, fwhm_max=5, rel_flux_limits=[0.001, 1e12])

    for channel in "brz":
        if kind == "longterm":
            mwave_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="mwave"))
            mlsf_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="mlsf"))
        else:
            mwave_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="nwave"))
            mlsf_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="nlsf"))

        xarc_paths = sorted(path.expand("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=f"{channel}?", expnum=expnum_str))
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=channel, expnum=expnum_str)
        harc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="h", imagetype="arc", camera=channel, expnum=expnum_str)

        # stack spectragraphs
        if skip_done and os.path.isfile(xarc_path):
            log.info(f"skipping stacked arc {xarc_path}, file already exists")
        else:
            rss_tasks.stack_spectrographs(in_rsss=xarc_paths, out_rss=xarc_path)
        # apply wavelength solution to arcs and rectify
        if skip_done and os.path.isfile(harc_path):
            log.info(f"skipping rectified arc {harc_path}, file already exists")
        else:
            rss_tasks.create_pixel_table(in_rss=xarc_path, out_rss=harc_path, in_waves=mwave_paths, in_lsfs=mlsf_paths)
            rss_tasks.resample_wavelength(in_rss=harc_path, out_rss=harc_path, method="linear", wave_range=SPEC_CHANNELS[channel], wave_disp=0.5)


def reduce_nightly_sequence(mjd, use_fiducial_cals=False, reject_cr=True, only_cals={"bias", "trace", "wave", "dome", "twilight"}, skip_done=True, keep_ancillary=False):
    """Reduces the nightly calibration sequence:

    The nightly calibration sequence consists of the following exposures:
        * 7 - 9 bias
        * 2 dome flat (LDLS and quartz)
        * 2 arc (10s and 50s exposures)
        * ~12 - 24 twilight (~half exposures at dawn and twilight)

    This routine will create *nightly* (not long-term) master calibrations
    at $LVM_SPECTRO_REDUX/{drpver}/0011XX/11111/{mjd}/calib

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to False
    reject_cr : bool
        Reject cosmic rays in 2D reduction, by default True
    only_cals : list, tuple or set
        Only produce this calibrations, by default {'bias', 'trace', 'wave', 'dome', 'twilight'}
    skip_done : bool
        Skip pipeline steps that have already been done
    keep_ancillary : bool
        Keep ancillary files, by default False
    """
    cal_types = {"bias", "trace", "wave", "dome", "twilight"}
    if not set(only_cals).issubset(cal_types):
        raise ValueError(f"some chosen image types in 'only_cals' are not valid: {only_cals.difference(cal_types)}")
    log.info(f"going to produce nightly calibrations: {only_cals}")

    frames, found_cals = md.get_sequence_metadata(mjd, for_cals=only_cals)

    if "bias" in only_cals and "bias" in found_cals:
        biases, bias_expnums = choose_sequence(frames, flavor="bias", kind="nightly")
        log.info(f"choosing {len(biases)} bias exposures: {bias_expnums}")
        create_detrending_frames(mjd=mjd, expnums=bias_expnums, kind="bias", use_fiducial_cals=use_fiducial_cals, skip_done=skip_done, keep_ancillary=keep_ancillary)
    else:
        log.log(20 if "bias" in found_cals else 40, "skipping production of bias frames")

    if "trace" in only_cals and "trace" in found_cals:
        dome_flats, dome_flat_expnums = choose_sequence(frames, flavor="flat", kind="nightly")
        log.info(f"choosing {len(dome_flats)} dome flat exposures: {dome_flat_expnums}")
        expnums_ldls = np.sort(dome_flats.query("ldls").expnum.unique())
        expnums_qrtz = np.sort(dome_flats.query("quartz").expnum.unique())
        create_nightly_traces(mjd=mjd, expnums_ldls=expnums_ldls, expnums_qrtz=expnums_qrtz, skip_done=skip_done)
    else:
        log.log(20 if "trace" in found_cals else 40, "skipping production of fiber traces")

    if mjd == 60177:
        if "wave" in only_cals and "wave" in found_cals:
            log.info(f"running dedicated script to create wavelength calibrations for MJD = {mjd}")
            _create_wavelengths_60177(use_fiducial_cals=False, skip_done=skip_done)
        else:
            log.log(20 if "wave" in found_cals else 40, "skipping production of wavelength calibrations")

        if "dome" in only_cals or "twilight" in only_cals and "dome" in found_cals:
            log.info(f"running dedicated script to create fiberflats for MJD = {mjd}")
            _create_fiberflats_60177(mjd=60255, use_fiducial_cals=False)
        else:
            log.log(20 if "dome" in found_cals or "twilight" in found_cals else 40, "skipping production of dome fiberflats")
    else:
        if "wave" in only_cals and "wave" in found_cals:
            arcs, arc_expnums = choose_sequence(frames, flavor="arc", kind="nightly")
            log.info(f"choosing {len(arcs)} arc exposures: {arc_expnums}")
            create_wavelengths(mjd=mjd, expnums=arc_expnums, use_fiducial_cals=False, kind="nightly", skip_done=skip_done)
        else:
            log.log(20 if "wave" in found_cals else 40, "skipping production of wavelength calibrations")

        if "dome" in only_cals and "dome" in found_cals:
            dome_flats, dome_flat_expnums = choose_sequence(frames, flavor="flat", kind="nightly")
            log.info(f"choosing {len(dome_flats)} dome flat exposures: {dome_flat_expnums}")
            expnums_ldls = np.sort(dome_flats.query("ldls").expnum.unique())
            expnums_qrtz = np.sort(dome_flats.query("quartz").expnum.unique())
            create_dome_fiberflats(mjd=mjd, expnums_ldls=expnums_ldls, expnums_qrtz=expnums_qrtz, use_fiducial_cals=False, kind="nightly", skip_done=skip_done)
        else:
            log.log(20 if "dome" in found_cals else 40, "skipping production of dome fiberflats")

        if "twilight" in only_cals and "twilight" in found_cals:
            twilight_flats, twilight_expnums = choose_sequence(frames, flavor="twilight", kind="nightly")
            log.info(f"choosing {len(twilight_flats)} twilight exposures: {twilight_expnums}")
            create_twilight_fiberflats(mjd=mjd, expnums=sorted(np.sort(twilight_flats.expnum.unique())), use_fiducial_cals=False, kind="nightly", skip_done=skip_done)
        else:
            log.log(20 if "twilight" in found_cals else 40, "skipping production of twilight fiberflats")

    if not keep_ancillary:
        _clean_ancillary(mjd)


def reduce_longterm_sequence(mjd, use_fiducial_cals=True, reject_cr=True, only_cals={"bias", "trace", "wave", "dome", "twilight"}, skip_done=True, keep_ancillary=False):
    """Reduces the long-term calibration sequence:

    The long-term calibration sequence consists of the following exposures:
        * 7 - 9 bias
        * 24 dome flat (12: LDLS and 12: quartz)
        * 24 arc (12: 10s and 12: 50s exposures)
        * ~12 - 24 twilight (~half exposures at dawn and twilight)

    This routine will create *long-term* master calibrations
    at $LVM_SANDBOX/calib/{mjd}

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    reject_cr : bool
        Reject cosmic rays in 2D reduction, by default True
    only_cals : list, tuple or set
        Only produce this calibrations, by default {'bias', 'trace', 'wave', 'dome', 'twilight'}
    skip_done : bool
        Skip pipeline steps that have already been done
    keep_ancillary : bool
        Keep ancillary files, by default False
    """
    cal_types = {"bias", "trace", "wave", "dome", "twilight"}
    if not set(only_cals).issubset(cal_types):
        raise ValueError(f"some chosen image types in 'only_cals' are not valid: {only_cals.difference(cal_types)}")
    log.info(f"going to produce long-term calibrations: {only_cals}")

    frames, found_cals = md.get_sequence_metadata(mjd, for_cals=only_cals)

    if "bias" in only_cals and "bias" in found_cals:
        biases, bias_expnums = choose_sequence(frames, flavor="bias", kind="longterm")
        log.info(f"choosing {len(biases)} bias exposures: {bias_expnums}")
        create_detrending_frames(mjd=mjd, expnums=bias_expnums, kind="bias", use_fiducial_cals=use_fiducial_cals, skip_done=skip_done, keep_ancillary=keep_ancillary)
        _move_master_calibrations(mjd=mjd, kind="bias")
    else:
        log.log(20 if "bias" in found_cals else 40, "skipping production of bias frames")

    if "trace" in only_cals and "trace" in found_cals:
        dome_flats, dome_flat_expnums = choose_sequence(frames, flavor="flat", kind="longterm")
        log.info(f"choosing {len(dome_flats)} dome flat exposures: {dome_flat_expnums}")
        expnums_ldls = np.sort(dome_flats.query("ldls").expnum.unique())
        expnums_qrtz = np.sort(dome_flats.query("quartz").expnum.unique())

        pool = Pool(9)
        threads = []
        for camera in CAMERAS:
            threads.append(pool.apply_async(create_traces,
                           kwds=dict(mjd=mjd, cameras=[camera],
                           use_fiducial_cals=use_fiducial_cals,
                           expnums_ldls=expnums_ldls, expnums_qrtz=expnums_qrtz,
                           skip_done=skip_done)))
        pool.close()
        pool.join()
        for ithr in range(len(threads)):
            threads[ithr].get()
        _move_master_calibrations(mjd=mjd, kind={"trace", "width"})
    else:
        log.log(20 if "trace" in found_cals else 40, "skipping production of fiber traces")

    if "wave" in only_cals and "wave" in found_cals:
        arcs, arc_expnums = choose_sequence(frames, flavor="arc", kind="longterm")
        log.info(f"choosing {len(arcs)} arc exposures: {arc_expnums}")
        create_wavelengths(mjd=mjd, expnums=np.sort(arcs.expnum.unique()), skip_done=skip_done)
        _move_master_calibrations(mjd=mjd, kind={"wave", "lsf"})
    else:
        log.log(20 if "wave" in found_cals else 40, "skipping production of wavelength calibrations")

    if "dome" in only_cals and "dome" in found_cals:
        create_dome_fiberflats(mjd=mjd, use_fiducial_cals=False)
    else:
        log.log(20 if "dome" in found_cals else 40, "skipping production of dome fiberflats")

    if "twilight" in only_cals and "twilight" in found_cals:
        twilight_flats, twilight_expnums = choose_sequence(frames, flavor="twilight", kind="longterm")
        log.info(f"choosing {len(twilight_flats)} twilight exposures: {twilight_expnums}")
        create_twilight_fiberflats(mjd=mjd, expnums=twilight_expnums, skip_done=skip_done)
        _move_master_calibrations(mjd=mjd, kind="fiberflat_twilight")
    else:
        log.log(20 if "twilight" in found_cals else 40, "skipping production of twilight fiberflats")

    if not keep_ancillary:
        _clean_ancillary(mjd)


def create_fiber_model(mjd, flux=10000):
    """Ancillary script to evaluate fiber models for a given calibration epoch"""
    masters_mjd = get_master_mjd(mjd)
    masters_path = os.path.join(MASTERS_DIR, f"{masters_mjd}")

    log.info(f"going to evaluate fiber model for cameras: {','.join(CAMERAS)}")
    for camera in CAMERAS:
        mcent_path = os.path.join(masters_path, f"lvm-mtrace-{camera}.fits")
        mwidth_path = os.path.join(masters_path, f"lvm-mwidth-{camera}.fits")

        if not (os.path.isfile(mcent_path) or os.path.isfile(mwidth_path)):
            log.error(f"skipping creation of fiber model for {mjd = }, {camera = }, incomplete fiber traces")
            continue

        trace_cent = TraceMask.from_file(mcent_path)
        trace_width = TraceMask.from_file(mwidth_path)

        model = Image(data=np.zeros((4080, 4086)))
        model, _ = model.eval_fiber_model(trace_cent, trace_width, trace_amp=flux)

        model.setHeader(trace_cent._header)
        model.setHdrValue("IMAGETYP", "fiber model")
        model.writeFitsData(os.path.join(masters_path, f"lvm-mmodel-{camera}.fits"))


class lvmFlat(lvmFrame):
    """lvmFlat class"""

    def __init__(self, data=None, error=None, mask=None,
                 cent_trace=None, width_trace=None, wave_trace=None, lsf_trace=None,
                 header=None, slitmap=None, superflat=None, **kwargs):
        lvmFrame.__init__(self, data=data, error=error, mask=mask,
                     cent_trace=cent_trace, width_trace=width_trace,
                     wave_trace=wave_trace, lsf_trace=lsf_trace,
                     header=header, slitmap=slitmap, superflat=superflat)

        self._blueprint = dp.load_blueprint(name="lvmFlat")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)


class lvmArc(lvmFrame):
    pass
