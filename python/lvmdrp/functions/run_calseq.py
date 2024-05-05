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
from shutil import copy2
from itertools import product, groupby
from typing import List, Tuple, Dict

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.core import dataproducts as dp
from lvmdrp.core.constants import SPEC_CHANNELS
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.image import loadImage
from lvmdrp.core.rss import RSS, lvmFrame

from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.functions.run_drp import get_config_options, read_fibermap
from lvmdrp.functions.run_quickdrp import get_master_mjd
from lvmdrp.functions.run_twilights import fit_fiberflat, remove_field_gradient, combine_twilight_sequence


SLITMAP = read_fibermap(as_table=True)
MASTER_CON_LAMPS = {"b": "ldls", "r": "ldls", "z": "quartz"}
MASTER_ARC_LAMPS = {"b": "hgne", "r": "neon", "z": "neon"}
MASTERS_DIR = os.getenv("LVM_MASTER_DIR")
MASK_BANDS = {
    "b": [(3910, 4000), (4260, 4330)],
    "r": [(6840,6960)],
    "z": [(7570, 7700)]
}


def get_sequence_metadata(mjd, expnums=None, exptime=None):
    """Get frames metadata for a given sequence

    Given a set of MJDs and (optionally) exposure numbers, get the frames
    metadata for the given sequence. This routine will return the frames
    metadata for the given MJDs and the MJD for the master frames.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    expnums : list
        List of exposure numbers to reduce
    exptime : int
        Filter frames metadata by exposure

    Returns:
    -------
    frames : pd.DataFrame
        Frames metadata
    masters_mjd : float
        MJD for master frames
    """
    # get frames metadata
    frames = md.get_frames_metadata(mjd=mjd)

    # filter by given expnums
    if expnums is not None:
        frames.query("expnum in @expnums", inplace=True)

    # filter by given exptime
    if exptime is not None:
        frames.query("exptime == @exptime", inplace=True)

    # simple fix of imagetyp, some images have the wrong type in the header
    twilight_selection = (frames.imagetyp == "flat") & ~(frames.ldls|frames.quartz)
    domeflat_selection = (frames.ldls|frames.quartz) & ~(frames.neon|frames.hgne|frames.argon|frames.xenon)
    arc_selection = (frames.neon|frames.hgne|frames.argon|frames.xenon) & ~(frames.ldls|frames.quartz)
    frames.loc[twilight_selection, "imagetyp"] = "flat"
    frames.loc[domeflat_selection, "imagetyp"] = "flat"
    frames.loc[arc_selection, "imagetyp"] = "arc"

    frames.sort_values(["expnum", "camera"], inplace=True)

    return frames


def choose_sequence(frames, flavor, kind):
    """Returns exposure numbers splitted in different sequences

    Parameters:
    ----------
    frames : pd.DataFrame
        Pandas dataframe containing frames metadata
    flavor : str
        Flavor of calibration frame: 'twilight', 'bias', 'flat', 'arc'
    kind : str
        Kind of calibration frame: 'nightly', 'longterm'

    Return:
    ------
    list
        list containing arrays of exposure numbers for each sequence
    """
    if not isinstance(flavor, str) or flavor not in {"twilight", "bias", "flat", "arc"}:
        raise ValueError(f"invalid flavor '{flavor}', available values are 'twilight', 'bias', 'flat', 'arc'")
    if not isinstance(kind, str) or kind not in {"nightly", "longterm"}:
        raise ValueError(f"invalid kind '{kind}', available values are 'nightly' and 'longterm'")

    if flavor == "twilight":
        query = "imagetyp == 'flat' and not ldls|quartz"
    elif flavor == "bias":
        query = "imagetyp == 'bias'"
    elif flavor == "flat":
        query = "imagetyp == 'flat' and ldls|quartz"
    elif flavor == "arc":
        query = "imagetyp == 'arc' and not ldls|quartz and neon|hgne|argon|xenon"
    expnums = frames.query(query).expnum.unique()
    diff = np.diff(expnums)
    div, = np.where(diff > 1)

    sequences = np.split(expnums, div+1)
    log.info(f"found sequences: {sequences}")

    if len(sequences) == 0:
        raise ValueError(f"no calibration frames of flavor '{flavor}' found using the query: {query}")

    lengths = [len(seq) for seq in sequences]
    idx = lengths.index(min(lengths) if kind == "nightly" else max(lengths))
    if len(sequences) > 1:
        chosen_expnums = sequences[idx]
    else:
        chosen_expnums = sequences[idx]

    if flavor == "twilight":
        expected_length = 24
    elif flavor == "bias":
        expected_length = 7
    elif flavor == "flat":
        expected_length = 2 if kind == "nightly" else 24
    elif flavor == "arc":
        expected_length = 2 if kind == "nightly" else 24

    if len(chosen_expnums) != expected_length:
        log.warning(f"wrong sequence length: {len(chosen_expnums)}")

    chosen_frames = frames.query("expnum in @chosen_expnums")
    chosen_frames.sort_values(["expnum", "camera"], inplace=True)
    return chosen_frames, chosen_expnums


def _load_shift_report(mjd):
    """Reads QC reports with the electronic pixel shifts"""

    with open(os.path.join(os.environ["LVM_SANDBOX"], "shift_monitor", f"shift_{mjd}.txt"), "r") as f:
        lines = f.readlines()[2:]

    shifts_report = {}
    for line in lines:
        cols = line[:-1].split()
        if not cols:
            continue
        print(cols)
        cols = [col for col in cols if col]
        _, exp, _, spec = cols[:4]
        exp = int(exp)
        spec = spec[-1]
        shifts = np.array([int(_) for _ in cols[4:]])
        shifts_report[(spec, exp)] = (shifts[::2]+1, shifts[1::2])

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
    refs = ref_frames.loc[(ref_frames.imagetyp == frame.imagetyp) & (ref_frames.ldls == frame.ldls) & (ref_frames.quartz == frame.quartz)]
    if len(refs) == 0:
                raise ValueError(f"no reference frame found for {frame.imagetyp}")
    idx = np.argmin(refs.expnum.sub(frame.expnum).abs())
    if idx > 0:
        idx -= 1
    if idx == 0:
        idx += 1
    return refs.iloc[idx]


def _clean_ancillary(mjd, expnums=None, kind="all"):
    """Clean ancillary files

    Given a set of MJDs and (optionally) exposure numbers, clean the ancillary
    files for the given kind of frames. This routine will remove the ancillary
    files for the given kind of frames in the corresponding calibration
    directory in the `masters_mjd` or by default in the smallest MJD in `mjds`.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    expnums : list
        List of exposure numbers to reduce
    kind : str
        Kind of frame to reduce, defaults to "all"
    """

    # get frames metadata
    frames = get_sequence_metadata(mjd, expnums=expnums)

    # filter by target image types
    if kind in ["bias", "dark", "flat", "arc", "object"]:
        frames.query("imagetyp == @kind", inplace=True)
    elif kind != "all":
        raise ValueError(f"Invalid kind: '{kind}'. Must be one of 'bias', 'dark', 'flat', 'arc', 'object' or 'all'")

    ancillary_dirs = []
    for frame in frames.to_dict("records"):
        # remove ancillary files
        ancillary_paths = path.expand("lvm_anc", drpver=drpver, kind='*', imagetype=frame["imagetyp"], **frame)
        [os.remove(ancillary_path) for ancillary_path in ancillary_paths]
        # get ancillary directories
        ancillary_dirs.extend([os.path.dirname(ancillary_path) for ancillary_path in ancillary_paths])

    ancillary_dirs = set(ancillary_dirs)
    for ancillary_dir in ancillary_dirs:
        # check if there are any files left in the ancillary directory
        # if not, remove the directory
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


def fix_raw_pixel_shifts(mjd, ref_expnums, use_fiducial_cals=True, expnums=None, specs="123",
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
    ref_expnums : list
        List of reference exposure numbers to use as reference for good frames
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    expnums : list
        List of exposure numbers to look for pixel shifts
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
    frames = get_sequence_metadata(mjd, expnums=expnums)
    ref_frames = get_sequence_metadata(mjd, expnums=ref_expnums)

    if use_fiducial_cals:
        masters_mjd = get_master_mjd(mjd)
        masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    ref_imagetyps = set(ref_frames.imagetyp)
    imagetyps = set(frames.imagetyp)
    if not imagetyps.issubset(ref_imagetyps):
        raise ValueError(f"the following image types are not present in the reference frames: {imagetyps - ref_imagetyps}")

    shifts_path = os.path.join(os.getenv('LVM_SANDBOX'), 'shift_monitor', f'shift_{mjd}.txt')
    if os.path.isfile(shifts_path):
        shifts_report = _load_shift_report(mjd)

    expnums_grp = frames.groupby("expnum")
    for spec in specs:
        for expnum in expnums_grp.groups:
            frame = expnums_grp.get_group(expnum).iloc[0]

            # find suitable reference frame for current frame
            ref_frame = _get_reference_expnum(frame, ref_frames)
            ref_expnum = ref_frame.expnum

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
                log.warning(f"skipping {rframe_paths}, less than 3 files found")
                continue

            if use_fiducial_cals:
                mwave_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwave_neon_hgne_argon_xenon-?{spec}.fits")))
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


def reduce_2d(mjd, use_fiducial_cals=True, expnums=None, exptime=None,
              replace_with_nan=True, assume_imagetyp=None, reject_cr=True,
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
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    expnums : list
        List of exposure numbers to reduce
    exptime : int
        Exposure time to filter by
    replace_with_nan : bool
        Replace rejected pixels with NaN
    assume_imagetyp : str
        Assume the given imagetyp for all frames
    reject_cr : bool
        Reject cosmic rays
    counts_threshold : int
        Minimum count level to consider when tracing centroids, defaults to 500
    poly_deg_cent : int
        Degree of the polynomial to fit to the centroids, by default 4
    skip_done : bool
        Skip pipeline steps that have already been done
    keep_ancillary : bool
        Keep ancillary files, by default False
    """

    frames = get_sequence_metadata(mjd, expnums=expnums, exptime=exptime)
    masters_mjd = get_master_mjd(mjd)
    masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    # preprocess and detrend frames
    for frame in frames.to_dict("records"):
        camera = frame["camera"]

        # assume given image type
        imagetyp = assume_imagetyp or frame["imagetyp"]

        # get master frames paths
        mpixmask_path = os.path.join(masters_path, f"lvm-mpixmask-{camera}.fits")
        mpixflat_path = os.path.join(masters_path, f"lvm-mpixelflat-{camera}.fits")
        if use_fiducial_cals:
            mbias_path = os.path.join(masters_path, f"lvm-mbias-{camera}.fits")
        else:
            mbias_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mbias", camera=camera)

        # log the master frames
        log.info(f'Using master pixel mask: {mpixmask_path}')
        log.info(f'Using master bias: {mbias_path}')
        log.info(f'Using master pixel flat: {mpixflat_path}')

        rframe_path = path.full("lvm_raw", camspec=frame["camera"], **frame)
        eframe_path = path.full("lvm_anc", drpver=drpver, kind="e", imagetype=imagetyp, **frame)
        frame_path = eframe_path if os.path.exists(eframe_path) else rframe_path
        pframe_path = path.full("lvm_anc", drpver=drpver, kind="p", imagetype=imagetyp, **frame)

        # bypass creation of detrended frame in case of imagetyp=bias
        if imagetyp != "bias":
            dframe_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=imagetyp, **frame)
        else:
            dframe_path = pframe_path

        os.makedirs(os.path.dirname(dframe_path), exist_ok=True)
        if skip_done and os.path.isfile(dframe_path):
            log.info(f"skipping {dframe_path}, file already exist")
        else:
            image_tasks.preproc_raw_frame(in_image=frame_path, out_image=pframe_path,
                                          in_mask=mpixmask_path, replace_with_nan=replace_with_nan, assume_imagetyp=assume_imagetyp)
            image_tasks.detrend_frame(in_image=pframe_path, out_image=dframe_path,
                                      in_bias=mbias_path,
                                      in_pixelflat=mpixflat_path,
                                      replace_with_nan=replace_with_nan,
                                      reject_cr=reject_cr,
                                      in_slitmap=SLITMAP if imagetyp in {"flat", "arc", "object"} else None)


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
    frames = get_sequence_metadata(mjd, expnums=expnums, exptime=exptime)

    # filter by target image types
    if kind == "all":
        frames.query("imagetyp in ['bias', 'dark', 'pixelflat']", inplace=True)
    elif kind in ["bias", "dark", "pixelflat"]:
        frames.query("imagetyp == @kind", inplace=True)
    else:
        raise ValueError(f"Invalid kind: '{kind}'. Must be one of 'bias', 'dark', 'pixelflat' or 'all'")

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
    frames = get_sequence_metadata(mjd, expnums=expnums)

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
            mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mpixelflat", camera=cam)

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


def create_nighly_traces(mjd, use_fiducial_cals=True, expnums_ldls=None, expnums_qrtz=None,
                        fit_poly=True, poly_deg_amp=5, poly_deg_cent=4, poly_deg_width=5,
                        skip_done=True):
    if expnums_ldls is not None and expnums_qrtz is not None:
        expnums = np.concatenate([expnums_ldls, expnums_qrtz])
    else:
        expnums = None
    frames = get_sequence_metadata(mjd, expnums=expnums)

    # run 2D reduction on flats: preprocessing, detrending
    reduce_2d(mjd, use_fiducial_cals=use_fiducial_cals, expnums=expnums, reject_cr=False, skip_done=skip_done)

    for channel, lamp in MASTER_CON_LAMPS.items():
        if lamp == "ldls":
            counts_threshold = 5000
        elif lamp == "quartz":
            counts_threshold = 10000

        cameras = [f"{channel}{spec}" for spec in range(1, 4)]
        flats = frames.loc[(frames.ldls)&(frames.camera.isin(cameras))]
        for _, flat in flats.iterrows():
            camera = flat.camera
            # define paths
            dflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="flat", camera=camera, expnum=flat["expnum"])
            lflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="l", imagetype="flat", camera=camera, expnum=flat["expnum"])
            dstray_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="stray", camera=camera, expnum=flat["expnum"])
            dmodel_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="model", camera=camera, expnum=flat["expnum"])
            dratio_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="ratio", camera=camera, expnum=flat["expnum"])

            cent_guess_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mcent_guess", camera=camera)
            flux_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mamps", camera=camera)
            cent_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mcent", camera=camera)
            fwhm_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mwidth", camera=camera)

            # first centroids trace
            if skip_done and os.path.isfile(cent_guess_path):
                log.info(f"skipping {cent_guess_path}, file already exist")
            else:
                log.info(f"going to trace centroids fibers in {camera}")
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
                                                in_cent_trace=cent_guess_path, select_nrows=5,
                                                aperture=13, smoothing=400, median_box=21,
                                                gaussian_sigma=0.0)

            if skip_done and os.path.isfile(flux_path) and os.path.isfile(cent_path) and os.path.isfile(fwhm_path):
                log.info(f"skipping {flux_path}, {cent_path} and {fwhm_path}, files already exist")
            else:
                log.info(f"going to trace fibers in {camera}")
                centroids, trace_cent_fit, trace_flux_fit, trace_fwhm_fit, img_stray, model, mratio = image_tasks.trace_fibers(
                    in_image=lflat_path,
                    out_trace_amp=flux_path, out_trace_cent=cent_path, out_trace_fwhm=fwhm_path,
                    in_trace_cent_guess=cent_guess_path,
                    median_box=(1,10), coadd=20,
                    counts_threshold=counts_threshold, max_diff=1.5, guess_fwhm=2.5,
                    ncolumns=40, fwhm_limits=(1.5, 4.5),
                    fit_poly=fit_poly, interpolate_missing=True,
                    poly_deg=(poly_deg_amp, poly_deg_cent, poly_deg_width)
                )

            # eval model continuum and ratio
            if skip_done and os.path.isfile(dmodel_path) and os.path.isfile(dratio_path):
                log.info(f"skipping {dmodel_path}, file already exist")
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
                model, ratio = img_stray.eval_fiber_model(trace_flux_fit, trace_cent_fit, trace_fwhm_fit)
                model.writeFitsData(dmodel_path)
                ratio.writeFitsData(dratio_path)


def create_traces(mjd, use_fiducial_cals=True, expnums_ldls=None, expnums_qrtz=None,
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
    frames = get_sequence_metadata(mjd, expnums=expnums)

    # run 2D reduction on flats: preprocessing, detrending
    reduce_2d(mjd, use_fiducial_cals=use_fiducial_cals, expnums=expnums, reject_cr=False, skip_done=skip_done)

    # load current traces
    mamps, mcents, mwidths = {}, {}, {}
    for _ in product("brz", "123"):
        camera = "".join(_)
        mamps[camera] = TraceMask()
        mamps[camera].createEmpty(data_dim=(648, 4086), poly_deg=poly_deg_amp)
        mcents[camera] = TraceMask()
        mcents[camera].createEmpty(data_dim=(648, 4086), poly_deg=poly_deg_cent)
        mwidths[camera] = TraceMask()
        mwidths[camera].createEmpty(data_dim=(648, 4086), poly_deg=poly_deg_width)

    tileid = frames.tileid.iloc[0]
    # iterate through exposures with std fibers exposed
    expnum_params = _get_ring_expnums(expnums_ldls, expnums_qrtz, ring_size=12)
    for camera, expnums in expnum_params.items():
        for expnum, block_idxs, fiber_str in expnums:
            con_lamp = MASTER_CON_LAMPS[camera[0]]
            if con_lamp == "ldls":
                counts_threshold = 5000
            elif con_lamp == "quartz":
                counts_threshold = 10000

            # select fibers in current spectrograph
            fibermap = SLITMAP[SLITMAP["spectrographid"] == int(camera[1])]
            # select illuminated std fiber
            select = fibermap["orig_ifulabel"] == fiber_str
            # select fiber index
            fiber_idx = np.where(select)[0][0]

            # define paths
            dflat_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=mjd, kind="d", imagetype="flat", camera=camera, expnum=expnum)
            lflat_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=mjd, kind="l", imagetype="flat", camera=camera, expnum=expnum)
            flux_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=mjd, kind="d", imagetype="flux", camera=camera, expnum=expnum)
            cent_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=mjd, kind="d", imagetype="cent", camera=camera, expnum=expnum)
            cent_guess_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=mjd, kind="d", imagetype="cent_guess", camera=camera, expnum=expnum)
            dstray_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=mjd, kind="d", imagetype="stray", camera=camera, expnum=expnum)
            fwhm_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=mjd, kind="d", imagetype="fwhm", camera=camera, expnum=expnum)
            dmodel_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=mjd, kind="d", imagetype="model", camera=camera, expnum=expnum)
            dratio_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=mjd, kind="d", imagetype="ratio", camera=camera, expnum=expnum)

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
                                                in_cent_trace=cent_guess_path, select_nrows=5,
                                                aperture=13, smoothing=400, median_box=21,
                                                gaussian_sigma=0.0)

            if skip_done and os.path.isfile(flux_path):
                log.info(f"skipping {flux_path}, file already exist")
                trace_cent_fit = TraceMask.from_file(cent_path)
                trace_flux_fit = TraceMask.from_file(flux_path)
                trace_fwhm_fit = TraceMask.from_file(fwhm_path)
                img_stray = loadImage(lflat_path)
                img_stray.setData(data=np.nan_to_num(img_stray._data), error=np.nan_to_num(img_stray._error))
                img_stray = img_stray.replaceMaskMedian(1, 10, replace_error=None)
                img_stray._data = np.nan_to_num(img_stray._data)
                img_stray = img_stray.medianImg((1,10), propagate_error=True)
                img_stray = img_stray.convolveImg(np.ones((1, 20), dtype="uint8"))
            else:
                log.info(f"going to trace std fiber {fiber_str} in {camera} within {block_idxs = }")
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
            log.info(f"{camera = }, {expnum = }, {fiber_str = :>6s}, fiber_idx = {fiber_idx:>3d}, FWHM = {np.nanmean(trace_fwhm_fit._data[fiber_idx]):.2f}")
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

        # masking bad fibers
        bad_fibers = fibermap["fibstatus"] == 1
        mamps[camera]._mask[bad_fibers] = True
        mcents[camera]._mask[bad_fibers] = True
        mwidths[camera]._mask[bad_fibers] = True
        # masking untraced standard fibers
        try:
            fiber_strs = list(zip(*expnum_params[camera]))[2]
        except IndexError:
            fiber_strs = []
        untraced_fibers = np.isin(fibermap["orig_ifulabel"].value, list(set(fibermap[fibermap["telescope"] == "Spec"]["orig_ifulabel"])-set(fiber_strs)))
        mamps[camera]._mask[untraced_fibers] = True
        mcents[camera]._mask[untraced_fibers] = True
        mwidths[camera]._mask[untraced_fibers] = True

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
        mamp_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera, kind="mamps")
        mcent_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera, kind="mtrace")
        mwidth_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=mjd, camera=camera, kind="mwidth")
        os.makedirs(os.path.dirname(mamp_path), exist_ok=True)
        mamps[camera].writeFitsData(mamp_path)
        mcents[camera].writeFitsData(mcent_path)
        mwidths[camera].writeFitsData(mwidth_path)

        # eval model continuum and ratio
        model, ratio = img_stray.eval_fiber_model(mamps[camera], mcents[camera], mwidths[camera])
        model.writeFitsData(dmodel_path)
        ratio.writeFitsData(dratio_path)


def create_fiberflats(mjd: int, use_fiducial_cals: bool = True, expnums: List[int] = None, median_box: int = 10, niter: bool = 1000,
                      threshold: Tuple[float,float]|float = (0.5,1.5), nknots: bool = 50,
                      b_mask: List[Tuple[float,float]] = MASK_BANDS["b"],
                      r_mask: List[Tuple[float,float]] = MASK_BANDS["r"],
                      z_mask: List[Tuple[float,float]] = MASK_BANDS["z"],
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
    flats = get_sequence_metadata(mjd, expnums=expnums)
    if expnums is None:
        flats.query("imagetyp == 'flat' and not ldls|quartz", inplace=True)

    # 2D reduction of twilight sequence
    reduce_2d(mjd=mjd, use_fiducial_cals=use_fiducial_cals, expnums=flats.expnum.unique(), reject_cr=False, skip_done=skip_done)

    for flat in flats.to_dict("records"):

        # master calibration paths
        camera = flat["camera"]
        mjd = flat["mjd"]
        masters_mjd = get_master_mjd(mjd)
        masters_path = os.path.join(MASTERS_DIR, f"{masters_mjd}")
        if use_fiducial_cals:
            master_cals = {
                "cent" : os.path.join(masters_path, f"lvm-mtrace-{camera}.fits"),
                "width" : os.path.join(masters_path, f"lvm-mwidth-{camera}.fits")
            }
        else:
            master_cals = {
                "cent" : path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mtrace", camera=camera),
                "width" : path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mwidth", camera=camera)
            }

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
                                            in_cent_trace=master_cals.get("cent"), select_nrows=5,
                                            aperture=13, smoothing=400, median_box=21,
                                            gaussian_sigma=0.0)

        if skip_done and os.path.isfile(xflat_path):
            log.info(f"skipping {xflat_path}, file already exist")
        else:
            image_tasks.extract_spectra(in_image=lflat_path, out_rss=xflat_path,
                                        in_trace=master_cals.get("cent"), in_fwhm=master_cals.get("width"),
                                        method="optimal", parallel=10)

    # decompose twilight spectra into sun continuum and twilight components
    channels = "brz"
    mask_bands = dict(zip(channels, [b_mask, r_mask, z_mask]))
    mfflats = dict.fromkeys(channels)
    flat_channels = flats.groupby(flats.camera.str.__getitem__(0))
    tileid = flats.tileid.min()
    for channel in channels:
        mwave_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwave-{channel}?.fits")))
        mwaves = [TraceMask.from_file(master_wave) for master_wave in mwave_paths]
        mwave = TraceMask.from_spectrographs(*mwaves)

        mlsf_paths = sorted(glob(os.path.join(masters_path, f"lvm-mlsf-{channel}?.fits")))
        mlsfs = [TraceMask.from_file(master_lsf) for master_lsf in mlsf_paths]
        mlsf = TraceMask.from_spectrographs(*mlsfs)

        mcent_paths = sorted(glob(os.path.join(masters_path, f"lvm-mtrace-{channel}?.fits")))
        mcents = [TraceMask.from_file(master_cent) for master_cent in mcent_paths]
        mcent = TraceMask.from_spectrographs(*mcents)

        mwidth_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwidth-{channel}?.fits")))
        mwidths = [TraceMask.from_file(master_width) for master_width in mwidth_paths]
        mwidth = TraceMask.from_spectrographs(*mwidths)

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

            mwave_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwave-{channel}?.fits")))
            mlsf_paths = sorted(glob(os.path.join(masters_path, f"lvm-mlsf-{channel}?.fits")))

            # spectrograph stack xflats
            xflat_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            rss_tasks.stack_spectrographs(in_rsss=xflat_paths, out_rss=xflat_path)

            # calibrate in wavelength
            wflat_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            rss_tasks.create_pixel_table(in_rss=xflat_path, out_rss=wflat_path,
                                         in_waves=mwave_paths, in_lsfs=mlsf_paths)

            # rectify in wavelength
            hflat_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            rss_tasks.resample_wavelength(in_rss=wflat_path, out_rss=hflat_path, wave_disp=0.5, wave_range=SPEC_CHANNELS[channel])

            # fit gradient and remove it
            gflat_path = path.full("lvm_anc", drpver=drpver, kind="g", imagetype=flat["imagetyp"], tileid=flat["tileid"], mjd=flat["mjd"], camera=channel, expnum=expnum)
            remove_field_gradient(in_hflat=hflat_path, out_gflat=gflat_path, wrange=SPEC_CHANNELS[channel])

            # fit fiber throughput
            fflat = fit_fiberflat(in_twilight=gflat_path, out_flat=fflat_path, out_rss=fflat_flatfielded_path, median_box=median_box, niter=niter,
                                   threshold=threshold, mask_bands=mask_bands.get(channel, []),
                                   display_plots=display_plots, nknots=nknots)
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


def create_wavelengths(mjd, use_fiducial_cals=True, expnums=None, skip_done=True):
    """Reduces an arc sequence to create master wavelength solutions

    Given a set of MJDs and (optionally) exposure numbers, create wavelength
    solutions from the master arcs. This routine will store the master
    wavelength solutions in the corresponding calibration directory in the
    `masters_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding master arcs do not exist, they will be created first.
    Otherwise they will be read from disk.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    use_fiducial_cals : bool
        Whether to use fiducial calibration frames or not, defaults to True
    expnums : list
        List of exposure numbers to reduce
    skip_done : bool
        Skip pipeline steps that have already been done
    """
    frames = get_sequence_metadata(mjd, expnums=expnums)
    frames = frames.query("imagetyp=='arc' or (not ldls|quartz and neon|hgne|argon|xenon)")
    frames["imagetyp"] = "arc"

    if use_fiducial_cals:
        masters_mjd = get_master_mjd(mjd)
        masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    reduce_2d(mjd, use_fiducial_cals=use_fiducial_cals, expnums=expnums, assume_imagetyp="arc", reject_cr=False, skip_done=skip_done)

    expnum_str = f"{frames.expnum.min():>08}-{frames.expnum.max():>08}"
    arc_analogs = frames.groupby(["camera",])
    for camera in arc_analogs.groups:
        arcs = arc_analogs.get_group((camera,))

        # define master paths for target frames
        if use_fiducial_cals:
            mtrace_path = os.path.join(masters_path, f"lvm-mtrace-{camera}.fits")
            mwidth_path = os.path.join(masters_path, f"lvm-mwidth-{camera}.fits")
        else:
            mtrace_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mtrace")
            mwidth_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mwidth")

        # define product paths
        carc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype="arc", camera=camera, expnum=expnum_str)
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=camera, expnum=expnum_str)
        mwave_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mwave")
        mlsf_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mlsf")
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
            image_tasks.extract_spectra(in_image=carc_path, out_rss=xarc_path, in_trace=mtrace_path, in_fwhm=mwidth_path, method="optimal")

        # fit wavelength solution
        if skip_done and os.path.isfile(mwave_path) and os.path.isfile(mlsf_path):
            log.info(f"skipping wavelength solution {mwave_path} and {mlsf_path}, files already exists")
        else:
            rss_tasks.determine_wavelength_solution(in_arcs=xarc_path, out_wave=mwave_path, out_lsf=mlsf_path, aperture=12,
                                                    cc_correction=True, cc_max_shift=20, poly_disp=5, poly_fwhm=2, poly_cros=2,
                                                    flux_min=1e-12, fwhm_max=5, rel_flux_limits=[0.001, 1e12])

    for channel in "brz":
        mwave_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="mwave"))
        mlsf_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="mlsf"))

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


def reduce_nightly_sequence(mjd, use_fiducial_cals=True, reject_cr=True, skip_done=True, keep_ancillary=False):
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
        Whether to use fiducial calibration frames or not, defaults to True
    reject_cr : bool
        Reject cosmic rays in 2D reduction, by default True
    skip_done : bool
        Skip pipeline steps that have already been done
    keep_ancillary : bool
        Keep ancillary files, by default False
    """

    cal_imagetyps = {"bias", "flat", "arc"}
    log.info(f"going to reduce nightly calibration frames: {cal_imagetyps}")

    frames = md.get_frames_metadata(mjd)
    frames.query("imagetyp in @cal_imagetyps", inplace=True)
    if len(frames) == 0:
        raise ValueError(f"no frames found for MJD = {mjd}")

    biases = frames.query("imagetyp == 'bias'")
    if len(biases) != 0:
        log.info(f"found {len(biases)} bias exposures: {set(biases.expnum)}")
        create_detrending_frames(mjd=mjd, expnums=set(biases.expnum), kind="bias", use_fiducial_cals=use_fiducial_cals, skip_done=skip_done, keep_ancillary=keep_ancillary)
    else:
        log.warning("no bias exposures found")

    dome_flats = frames.query("imagetyp == 'flat' and ldls|quartz")
    if len(dome_flats) != 0:
        expnums_ldls = dome_flats.query("ldls").expnum.unique()
        expnums_qrtz = dome_flats.query("quartz").expnum.unique()
        log.info(f"found {len(dome_flats)} dome flat exposures: {set(dome_flats.expnum)}")
        create_nighly_traces(mjd=mjd, expnums_ldls=expnums_ldls, expnums_qrtz=expnums_qrtz, skip_done=skip_done)
        # create_nightly_fiberflats(mjd=mjd, expnums_ldls=expnums_ldls, expnums_qrtz=expnums_qrtz, skip_done=skip_done)
    else:
        log.warning("no dome flat exposures found")

    arcs = frames.query("imagetyp == 'arc' and not ldls|quartz and neon|hgne|argon|xenon")
    if len(arcs) != 0:
        log.info(f"found {len(arcs)} arc exposures: {set(arcs.expnum)}")
        create_wavelengths(mjd=mjd, expnums=arcs.expnum.unique(), skip_done=skip_done)
    else:
        log.warning("no arc exposures found")

    twilight_flats = frames.query("imagetyp == 'flat' and not ldls|quartz")
    if len(twilight_flats) != 0:
        log.info(f"found {len(twilight_flats)} twilight exposures: {set(twilight_flats.expnum)}")
        create_fiberflats(mjd=mjd, expnums=sorted(twilight_flats.expnum.unique()), skip_done=skip_done)
    else:
        log.warning("no twilight exposures found")


def reduce_longterm_sequence(mjd, use_fiducial_cals=True, reject_cr=True, skip_done=True, keep_ancillary=False):
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
    skip_done : bool
        Skip pipeline steps that have already been done
    keep_ancillary : bool
        Keep ancillary files, by default False
    """
    cal_imagetyps = {"bias", "flat", "arc"}
    log.info(f"going to reduce nightly calibration frames: {cal_imagetyps}")

    frames = md.get_frames_metadata(mjd)
    frames.query("imagetyp in @cal_imagetyps", inplace=True)
    if len(frames) == 0:
        raise ValueError(f"no frames found for MJD = {mjd}")

    biases, bias_expnums = choose_sequence(frames, flavor="bias", kind="longterm")
    if len(biases) != 0:
        log.info(f"found {len(biases)} bias exposures: {bias_expnums}")
        create_detrending_frames(mjd=mjd, expnums=bias_expnums, kind="bias", use_fiducial_cals=use_fiducial_cals, skip_done=skip_done, keep_ancillary=keep_ancillary)
    else:
        log.warning("no bias exposures found")

    dome_flats, dome_flat_expnums = choose_sequence(frames, flavor="flat", kind="longterm")
    if len(dome_flats) != 0:
        expnums_ldls = dome_flats.query("ldls").expnum.unique()
        expnums_qrtz = dome_flats.query("quartz").expnum.unique()
        log.info(f"found {len(dome_flats)} dome flat exposures: {dome_flat_expnums}")
        create_traces(mjd=mjd, expnums_ldls=expnums_ldls, expnums_qrtz=expnums_qrtz, skip_done=skip_done)
        # create_nightly_fiberflats(mjd=mjd, expnums_ldls=expnums_ldls, expnums_qrtz=expnums_qrtz, skip_done=skip_done)
    else:
        log.warning("no dome flat exposures found")

    arcs, arc_expnums = choose_sequence(frames, flavor="arc", kind="longterm")
    if len(arcs) != 0:
        log.info(f"found {len(arcs)} arc exposures: {arc_expnums}")
        create_wavelengths(mjd=mjd, expnums=arcs.expnum.unique(), skip_done=skip_done)
    else:
        log.warning("no arc exposures found")

    twilight_flats, twilight_expnums = choose_sequence(frames, flavor="twilight", kind="longterm")
    if len(twilight_flats) != 0:
        log.info(f"found {len(twilight_flats)} twilight exposures: {twilight_expnums}")
        create_fiberflats(mjd=mjd, expnums=twilight_expnums, skip_done=skip_done)
    else:
        log.warning("no twilight exposures found")


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


if __name__ == '__main__':
    import tracemalloc

    tracemalloc.start()

    # MJD = 60255
    # ldls_expnums = np.arange(7264, 7269+1)
    # ldls_expnums = [None] * (12-ldls_expnums.size) + ldls_expnums.tolist()
    # qrtz_expnums = np.arange(7252, 7263+1)

    # MJD = 60185
    # ldls_expnums = np.arange(3936, 3937+1).tolist() + [None] * 10
    # qrtz_expnums = np.arange(3938, 3939+1).tolist() + [None] * 10

    # MJD = 60255
    # # ldls_expnums = np.arange(7230, 7240+1)
    # # qrtz_expnums = np.arange(7341, 7352+1)
    # ldls_expnums = qrtz_expnums = [7230] * 6
    # ldls_expnums += [None] * 6
    # qrtz_expnums += [None] * 6

    try:
        # create_detrending_frames(mjd=60171, masters_mjd=60142, kind="bias", skip_done=False)
        # create_detrending_frames(mjd=60146, masters_mjd=60142, kind="dark", exptime=900, reject_cr=False, skip_done=False)
        # create_detrending_frames(mjd=60171, masters_mjd=60142, expnums=np.arange(3098, 3117+1), kind="dark", assume_imagetyp="pixelflat", reject_cr=False, skip_done=False)

        # create_detrending_frames(mjd=60255, kind="bias", skip_done=False)
        # create_traces(mjd=MJD, expnums_ldls=ldls_expnums, expnums_qrtz=qrtz_expnums, subtract_straylight=True)
        # create_wavelengths(mjd=60255, masters_mjd=60255, expnums=np.arange(7276,7323+1), skip_done=True)

        # expnums = [7231]
        # expnums = np.arange(7341, 7352+1)
        # expnums = [7352]
        # create_fiberflats(mjd=60255, expnums=expnums, median_box=10, niter=1000, threshold=(0.5,2.5), nknots=60, skip_done=True, display_plots=False)

        # reduce_nightly_sequence(mjd=60265, reject_cr=False, use_fiducial_cals=False, skip_done=True, keep_ancillary=True)
        # reduce_longterm_sequence(mjd=60177, reject_cr=False, use_fiducial_cals=False, skip_done=True, keep_ancillary=True)

        # frames = md.get_frames_metadata(60264)
        # frames.sort_values(by="expnum", inplace=True)
        # frames, sequence = split_sequences(frames, flavor="arc", kind="nightly")
        # print(sequence)
        # print(frames.to_string())

        fix_raw_pixel_shifts(mjd=60412, expnums=[16148], ref_expnums=None, wave_widths=5000, y_widths=20, flat_spikes=21, threshold_spikes=0.6, skip_done=True, interactive=True, display_plots=True)

    except Exception as e:
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)

        raise e