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
import click
import cloup
import numpy as np
import pandas as pd
from glob import glob
from copy import deepcopy as copy
from shutil import copy2
from itertools import product, groupby

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.core.constants import SPEC_CHANNELS
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.image import loadImage

from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.functions.run_drp import get_config_options, read_fibermap
from lvmdrp.functions.run_quickdrp import get_master_mjd
from lvmdrp.functions.run_twilights import reduce_twilight_sequence


SLITMAP = read_fibermap(as_table=True)
MASTER_CON_LAMPS = {"b": "ldls", "r": "ldls", "z": "quartz"}
MASTER_ARC_LAMPS = {"b": "hgne", "r": "neon", "z": "neon"}
MASTERS_DIR = os.getenv("LVM_MASTER_DIR")


def get_sequence_metadata(mjds, target_mjd=None, expnums=None, exptime=None):
    """Get frames metadata for a given sequence

    Given a set of MJDs and (optionally) exposure numbers, get the frames
    metadata for the given sequence. This routine will return the frames
    metadata for the given MJDs and the MJD for the master frames.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
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
    # change to list if single MJD is given
    if not isinstance(mjds, (list, tuple)):
        mjds = [mjds]

    # define MJD for master frames
    masters_mjd = target_mjd or min(mjds)

    # get frames metadata
    frames = [md.get_frames_metadata(mjd=mjd) for mjd in mjds]
    frames = pd.concat(frames, ignore_index=True)

    # filter by given expnums
    if expnums is not None:
        frames.query("expnum in @expnums", inplace=True)

    # filter by given exptime
    if exptime is not None:
        frames.query("exptime == @exptime", inplace=True)

    frames.sort_values(["expnum", "camera"], inplace=True)

    return frames, masters_mjd


def _clean_ancillary(mjds, expnums=None, kind=None):
    """Clean ancillary files

    Given a set of MJDs and (optionally) exposure numbers, clean the ancillary
    files for the given kind of frames. This routine will remove the ancillary
    files for the given kind of frames in the corresponding calibration
    directory in the `target_mjd` or by default in the smallest MJD in `mjds`.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    expnums : list
        List of exposure numbers to reduce
    kind : str
        Kind of frame to reduce
    """

    # change to list if single MJD is given
    if not isinstance(mjds, (list, tuple)):
        mjds = [mjds]

    # get frames metadata
    frames = [md.get_metadata(tileid="*", mjd=mjd) for mjd in mjds]
    frames = pd.concat(frames, ignore_index=True)

    # filter by given expnums
    if expnums is not None:
        frames.query("expnum in @expnums", inplace=True)

    # filter by target image types
    if kind == "all":
        frames.query("imagetyp in ['bias', 'dark', 'pixflat']", inplace=True)
    elif kind in ["bias", "dark", "pixelflat"]:
        frames.query("imagetyp == @kind", inplace=True)
    else:
        raise ValueError(f"Invalid kind: '{kind}'. Must be one of 'bias', 'dark', 'pixelflat' or 'all'")

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


def fix_raw_pixel_shifts(mjd, expnums=None, ref_expnums=None, specs="123",
                         y_widths=5, wave_list=None, wave_widths=0.6*5, max_shift=10, flat_spikes=11,
                         threshold_spikes=np.inf, shift_rows=None, create_mask_always=False, dry_run=False,
                         undo_corrections=False, display_plots=False):
    """Attempts to fix pixel shifts in a list of raw frames

    Given an MJD and (optionally) exposure numbers, fix the pixel shifts in a
    list of 2D frames. This routine will store the fixed frames in the
    corresponding calibration directory in the `target_mjd` or by default `mjd`.

    Parameters:
    ----------
    mjd : float
        MJD to reduce
    expnums : list
        List of exposure numbers to look for pixel shifts
    ref_expnums : list
        List of reference exposure numbers to use as reference for good frames
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
    create_mask_always : bool
        Create mask always, by default False
    dry_run : bool
        Dry run, by default False
    undo_corrections : bool
        Only undo corrections for previous runs, by default False
    display_plots : bool
        Display plots, by default False
    """

    if isinstance(ref_expnums, (list, tuple, np.ndarray)):
        ref_expnum = ref_expnums[0]
    elif isinstance(ref_expnums, (int, np.int64)):
        ref_expnum = ref_expnums
    else:
        raise ValueError("no valid reference exposure number given")

    if shift_rows is None:
        shift_rows = {}
    elif not isinstance(shift_rows, dict):
        raise ValueError("shift_rows must be a dictionary with keys (spec, expnum) and values a list of rows to shift")

    frames, _ = get_sequence_metadata(mjd, target_mjd=None, expnums=expnums)
    masters_mjd = get_master_mjd(sci_mjd=mjd)
    masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    expnums_grp = frames.groupby("expnum")
    for spec in specs:
        for expnum in expnums_grp.groups:
            rframe_paths = sorted(path.expand("lvm_raw", hemi="s", camspec=f"?{spec}", mjd=mjd, expnum=expnum))
            cframe_paths = sorted(path.expand("lvm_raw", hemi="s", camspec=f"?{spec}", mjd=mjd, expnum=ref_expnum))
            rframe_paths = [rframe_path for rframe_path in rframe_paths if ".gz" in rframe_path]
            cframe_paths = [cframe_path for cframe_path in cframe_paths if ".gz" in cframe_path]

            if len(rframe_paths) < 3:
                log.warning(f"skipping {rframe_paths}, less than 3 files found")
                continue

            mwave_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwave_neon_hgne_argon_xenon-?{spec}.fits")))
            mtrace_paths = sorted(glob(os.path.join(masters_path, f"lvm-mtrace-?{spec}.fits")))
            mask_2d_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, imagetype="mask2d",
                                     expnum=0, camera=f"sp{spec}", kind="")
            pixshift_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, imagetype="pixshift",
                                     expnum=expnum, camera=f"sp{spec}", kind="")

            if not undo_corrections and (create_mask_always or expnum == list(expnums_grp.groups)[0]):
                os.makedirs(os.path.dirname(mask_2d_path), exist_ok=True)
                image_tasks.select_lines_2d(in_images=cframe_paths, out_mask=mask_2d_path, lines_list=wave_list,
                                            in_cent_traces=mtrace_paths, in_waves=mwave_paths,
                                            y_widths=y_widths, wave_widths=wave_widths,
                                            display_plots=display_plots)

            image_tasks.fix_pixel_shifts(in_images=rframe_paths, out_pixshift=pixshift_path,
                                         ref_images=cframe_paths, in_mask=mask_2d_path, flat_spikes=flat_spikes,
                                         threshold_spikes=threshold_spikes, max_shift=max_shift, shift_rows=shift_rows.get((spec, expnum), None),
                                         dry_run=dry_run, undo_correction=undo_corrections, display_plots=display_plots)


def reduce_2d(mjds, target_mjd=None, expnums=None, exptime=None,
              replace_with_nan=True, assume_imagetyp=None, reject_cr=True,
              counts_threshold=5000, poly_deg_cent=4, use_master_centroids=False,
              skip_done=True):
    """Preprocess and detrend a list of 2D frames

    Given a set of MJDs and (optionally) exposure numbers, preprocess detrends
    and optionally fits and subtracts the stray light field from the 2D frames.
    This routine will store the preprocessed, detrended and
    straylight-subtracted frames in the corresponding calibration directory in
    the `target_mjd` or by default in the smallest MJD in `mjds`.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
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
        Minimum count level to consider when tracing centroids, defaults to 5000
    poly_deg_cent : int
        Degree of the polynomial to fit to the centroids, by default 4
    skip_done : bool
        Skip pipeline steps that have already been done
    """

    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums, exptime=exptime)
    masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    # preprocess and detrend frames
    for frame in frames.to_dict("records"):
        camera = frame["camera"]

        # assume given image type
        imagetyp = assume_imagetyp or frame["imagetyp"]
        # get master frames paths
        # masters = find_masters(masters_mjd, imagetyp, camera)
        mpixmask_path = os.path.join(masters_path, f"lvm-mpixmask-{camera}.fits")
        mbias_path = os.path.join(masters_path, f"lvm-mbias-{camera}.fits")
        mdark_path = os.path.join(masters_path, f"lvm-mdark-{camera}.fits")
        mpixflat_path = os.path.join(masters_path, f"lvm-mpixelflat-{camera}.fits")

        # log the master frames
        log.info(f'Using master pixel mask: {mpixmask_path}')
        log.info(f'Using master bias: {mbias_path}')
        log.info(f'Using master dark: {mdark_path}')
        log.info(f'Using master pixel flat: {mpixflat_path}')

        frame_path = path.full("lvm_raw", camspec=frame["camera"], **frame)
        pframe_path = path.full("lvm_anc", drpver=drpver, kind="p", imagetype=imagetyp, **frame)
        dcent_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype="cent", **frame)
        lframe_path = path.full("lvm_anc", drpver=drpver, kind="l", imagetype=imagetyp, **frame)
        dstray_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype="stray", **frame)

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
                                      in_bias=mbias_path, in_dark=mdark_path,
                                      in_pixelflat=mpixflat_path,
                                      replace_with_nan=replace_with_nan,
                                      reject_cr=reject_cr,
                                      in_slitmap=SLITMAP if imagetyp in {"flat", "arc", "object"} else None)

        # subtract stray light only if imagetyp is flat
        if imagetyp == "flat" and skip_done and os.path.isfile(lframe_path):
            log.info(f"skipping {lframe_path}, file already exist")
        elif imagetyp == "flat":
            # quick and dirty trace of centroids to subtract stray light
            if not use_master_centroids:
                image_tasks.trace_fibers(in_image=dframe_path,
                                        out_trace_cent=None,
                                        out_trace_cent_guess=dcent_path,
                                        correct_ref=True, median_box=(1,10), coadd=20,
                                        counts_threshold=counts_threshold, max_diff=1.5,
                                        guess_fwhm=2.5, method="gauss", ncolumns=140,
                                        fit_poly=True, poly_deg=poly_deg_cent,
                                        interpolate_missing=True, only_centroids=True)
            else:
                dcent_path = os.path.join(masters_path, f"lvm-mtrace-{camera}.fits")
            image_tasks.subtract_straylight(in_image=dframe_path, out_image=lframe_path, out_stray=dstray_path,
                                            in_cent_trace=dcent_path, select_nrows=5,
                                            aperture=13, smoothing=400, median_box=21,
                                            gaussian_sigma=0.0)


def create_detrending_frames(mjds, target_mjd=None, expnums=None, exptime=None, kind="all", assume_imagetyp=None, reject_cr=True, skip_done=True):
    """Reduce a sequence of bias/dark/pixelflat frames to produce master frames

    Given a set of MJDs and (optionally) exposure numbers, reduce the
    bias/dark/pixelflat frames. The kind argument specifies which type of frame
    to reduce:

        - bias
        - dark
        - pixflat
        - all (default)

    This routine will store the master of each kind of frame in the
    corresponding calibration directory in the `target_mjd` or by default in
    the smallest MJD in `mjds`.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
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
    """
    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums, exptime=exptime)

    # filter by target image types
    if kind == "all":
        frames.query("imagetyp in ['bias', 'dark', 'pixelflat']", inplace=True)
    elif kind in ["bias", "dark", "pixelflat"]:
        frames.query("imagetyp == @kind", inplace=True)
    else:
        raise ValueError(f"Invalid kind: '{kind}'. Must be one of 'bias', 'dark', 'pixelflat' or 'all'")

    # preprocess and detrend frames
    reduce_2d(mjds=mjds, target_mjd=masters_mjd, expnums=set(frames.expnum), exptime=exptime, assume_imagetyp=assume_imagetyp, reject_cr=reject_cr, skip_done=skip_done)

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
            mframe_path = path.full("lvm_master", drpver=drpver, tileid=frame["tileid"], mjd=masters_mjd, kind=f'm{imagetyp}', camera=frame["camera"])
            os.makedirs(os.path.dirname(mframe_path), exist_ok=True)

            dframe_paths = [path.full("lvm_anc", drpver=drpver, kind="d" if imagetyp != "bias" else "p", imagetype=imagetyp, **frame) for frame in analogs.to_dict("records")]
            image_tasks.create_master_frame(in_images=dframe_paths, out_image=mframe_path, **kwargs)

    # ancillary paths clean up
    _clean_ancillary(mjds=mjds, expnums=expnums, kind=kind)


def create_pixelmasks(mjds, target_mjd=None, dark_expnums=None, pixflat_expnums=None,
                      short_exptime=900, long_exptime=3600, pixflat_exptime=5,
                      ignore_pixflats=True):
    """Create pixel mask from master pixelflat and/or dark frames

    Given a set of MJDs and (optionally) exposure numbers, create a pixel mask
    from the master pixelflat and/or dark frames. This routine will store the
    master pixelmask in the corresponding calibration directory in the
    `target_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding detrended pixelflat and/or dark frames do not exist, they
    will be created first. Otherwise they will be read from disk. If
    `ignore_pixflats` is True, then the pixelflat frames will not be used to create the
    pixel mask.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
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

    """
    if dark_expnums is not None and pixflat_expnums is not None:
        expnums = np.concatenate([dark_expnums, pixflat_expnums])
    else:
        expnums = None
    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums)
    masters_path = os.path.join(MASTERS_DIR, f"{masters_mjd}")

    darks = frames.query("imagetyp == 'dark' and exptime == @short_exptime or exptime == @long_exptime", inplace=True)
    pixflats = frames.query("imagetyp == 'dark' or imagetyp == 'pixelflat' and exptime == @pixflat_exptime", inplace=True)

    # reduce darks
    reduce_2d(mjds=mjds, target_mjd=target_mjd, expnums=set(darks.expnum))

    ddark_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="dark", **dark) for dark in darks.to_dict("records")]
    darks["ddark_path"] = ddark_paths
    cam_groups = darks.groupby(["camera", "exptime"])
    for cam, exptime in cam_groups.groups:
        ddark_paths_cam = cam_groups.get_group((cam, exptime))["ddark_path"]

        mdark_path = os.path.join(masters_path, f"lvm-mdark-{cam}-{int(exptime)}s.fits")
        image_tasks.create_master_frame(in_images=ddark_paths_cam, out_image=mdark_path)

    # reduce pixflats
    if not ignore_pixflats:
        reduce_2d(mjds=mjds, target_mjd=target_mjd, expnums=set(pixflats.expnum),
                replace_with_nan=False, assume_imagetyp="pixelflat", reject_cr=False)
        dflat_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="pixelflat", **pixflat) for pixflat in pixflats.to_dict("records")]
        pixflats["dflat_path"] = dflat_paths

        cam_groups = pixflats.groupby("camera")
        for cam in cam_groups.groups:
            dflat_paths_cam = cam_groups.get_group(cam)["dflat_path"]

            # define output combined pixelflat path
            mflat_path = os.path.join(masters_path, f"lvm-mpixelflat-{cam}.fits")

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

            mdark_short_path = os.path.join(masters_path, f"lvm-mdark-{camera}-{short_exptime}s.fits")
            mdark_long_path = os.path.join(masters_path, f"lvm-mdark-{camera}-{long_exptime}s.fits")
            mpixmask_path = os.path.join(masters_path, f"lvm-mpixmask-{camera}.fits")
            image_tasks.create_pixelmask(in_short_dark=mdark_short_path, in_long_dark=mdark_long_path,
                                        in_flat_a=flat_a, in_flat_b=flat_b,
                                        out_pixmask=mpixmask_path)
    else:
        log.info("Ignoring pixelflats when creating pixel mask")
        mdark_short_path = os.path.join(masters_path, f"lvm-mdark-{camera}-{short_exptime}s.fits")
        mdark_long_path = os.path.join(masters_path, f"lvm-mdark-{camera}-{long_exptime}s.fits")
        mpixmask_path = os.path.join(masters_path, f"lvm-mpixmask-{camera}.fits")
        image_tasks.create_pixelmask(in_short_dark=mdark_short_path, in_long_dark=mdark_long_path, out_pixmask=mpixmask_path)

    # ancillary paths clean up
    _clean_ancillary(mjds=mjds, expnums=expnums)


def create_traces(mjds, target_mjd=None, expnums_ldls=None, expnums_qrtz=None,
                  subtract_straylight=False, fit_poly=True, poly_deg_amp=5,
                  poly_deg_cent=4, poly_deg_width=5, skip_done=True):
    """Create traces from master dome flats

    Given a set of MJDs and (optionally) exposure numbers, create traces from
    the master dome flats. This routine will store the master traces in the
    corresponding calibration directory in the `target_mjd` or by default in
    the smallest MJD in `mjds`.

    If the corresponding master dome flats do not exist, they will be created
    first. Otherwise they will be read from disk.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
    expnums_ldls : list
        List of exposure numbers for LDLS dome flats
    expnums_qrtz : list
        List of exposure numbers for quartz dome flats
    subtract_straylight : bool, optional
        Subtract stray light from dome flats, by default False
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
    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums)

    # run 2D reduction on flats: preprocessing, detrending and stray light subtraction
    reduce_2d(mjds, target_mjd=masters_mjd, expnums=expnums, reject_cr=False, skip_done=skip_done)

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
            lflat_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="l", imagetype="flat", camera=camera, expnum=expnum)
            flux_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="flux", camera=camera, expnum=expnum)
            cent_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="cent", camera=camera, expnum=expnum)
            cent_guess_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="cent_guess", camera=camera, expnum=expnum)
            fwhm_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="fwhm", camera=camera, expnum=expnum)
            model_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="model", camera=camera, expnum=expnum)
            mratio_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="mratio", camera=camera, expnum=expnum)

            log.info(f"going to trace std fiber {fiber_str} in {camera} within {block_idxs = }")
            centroids, trace_cent_fit, trace_flux_fit, trace_fwhm_fit, img_stray, model, mratio = image_tasks.trace_fibers(
                in_image=lflat_path,
                out_trace_amp=flux_path, out_trace_cent=cent_path, out_trace_fwhm=fwhm_path,
                out_trace_cent_guess=cent_guess_path,
                correct_ref=True, median_box=(1,10), coadd=20,
                counts_threshold=counts_threshold, max_diff=1.5, guess_fwhm=2.5, method="gauss",
                ncolumns=(140, 40), iblocks=block_idxs, fwhm_limits=(1.5, 4.5),
                fit_poly=fit_poly, interpolate_missing=False, poly_deg=(poly_deg_amp, poly_deg_cent, poly_deg_width), use_given_centroids=True
            )

            # update master traces
            log.info(f"{camera = }, {expnum = }, {fiber_str = :>6s}, fiber_idx = {fiber_idx:>3d}, FWHM = {trace_fwhm_fit._data[fiber_idx].mean():.2f}")
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
        mamp_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=masters_mjd, camera=camera, kind="mamps")
        mcent_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=masters_mjd, camera=camera, kind="mtrace")
        mwidth_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=masters_mjd, camera=camera, kind="mwidth")
        os.makedirs(os.path.dirname(mamp_path), exist_ok=True)
        mamps[camera].writeFitsData(mamp_path)
        mcents[camera].writeFitsData(mcent_path)
        mwidths[camera].writeFitsData(mwidth_path)

        # eval model continuum and ratio
        model, ratio = img_stray.eval_fiber_model(mamps[camera], mcents[camera], mwidths[camera])
        model.writeFitsData(model_path)
        ratio.writeFitsData(mratio_path)


def create_fiberflats(mjds, target_mjd=None, expnums=None):
    """Create fiber flats from master twilight flats

    Given a set of MJDs and (optionally) exposure numbers, create fiber flats
    from the master twilight flats. This routine will store the master fiber
    flats in the corresponding calibration directory in the `target_mjd` or by
    default in the smallest MJD in `mjds`.

    If the corresponding master twilight flats do not exist, they will be
    created first. Otherwise they will be read from disk.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
    expnums : list
        List of exposure numbers to reduce
    """
    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums)

    reduce_2d(mjds, target_mjd=masters_mjd, expnums=expnums)

    if expnums is None:
        expnums = set(frames.query("imagetyp == 'flat' and not ldls and not quartz").expnum)

    reduce_twilight_sequence(expnums=expnums)


def create_illumination_corrections(mjds, target_mjd=None, expnums=None):
    """Create illumination corrections from master dome and twilight flats

    Given a set of MJDs and (optionally) exposure numbers, create illumination
    corrections from the master dome and twilight flats. This routine will
    store the master illumination corrections in the corresponding calibration
    directory in the `target_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding master dome and twilight flats do not exist, they
    will be created first. Otherwise they will be read from disk.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
    expnums : list
        List of exposure numbers to reduce
    """
    # read master fiber flats (dome and twilight)
    # calculate ratio twilight/dome
    raise NotImplementedError("create_illumination_corrections")


def create_wavelengths(mjds, target_mjd=None, expnums=None, skip_done=True):
    """Reduces an arc sequence to create master wavelength solutions

    Given a set of MJDs and (optionally) exposure numbers, create wavelength
    solutions from the master arcs. This routine will store the master
    wavelength solutions in the corresponding calibration directory in the
    `target_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding master arcs do not exist, they will be created first.
    Otherwise they will be read from disk.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
    expnums : list
        List of exposure numbers to reduce
    skip_done : bool
        Skip pipeline steps that have already been done
    """
    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums)
    frames = frames.query("imagetyp=='arc' or (not ldls|quartz and neon|hgne|argon|xenon)")
    frames["imagetyp"] = "arc"
    masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    reduce_2d(mjds, target_mjd=masters_mjd, expnums=expnums, assume_imagetyp="arc", reject_cr=False, skip_done=skip_done)

    expnum_str = f"{frames.expnum.min():>08}-{frames.expnum.max():>08}"
    arc_analogs = frames.groupby(["camera",])
    for camera in arc_analogs.groups:
        arcs = arc_analogs.get_group((camera,))
        arc = arcs.iloc[0]

        # define master paths for target frames
        mtrace_path = os.path.join(masters_path, f"lvm-mtrace-{camera}.fits")
        mwidth_path = os.path.join(masters_path, f"lvm-mwidth-{camera}.fits")

        # define product paths
        carc_path = path.full("lvm_anc", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], kind="c", imagetype=arc["imagetyp"], camera=arc["camera"], expnum=expnum_str)
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], kind="x", imagetype=arc["imagetyp"], camera=arc["camera"], expnum=expnum_str)
        mwave_path = path.full("lvm_master", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], camera=arc["camera"], kind="mwave")
        mlsf_path = path.full("lvm_master", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], camera=arc["camera"], kind="mlsf")
        os.makedirs(os.path.dirname(carc_path), exist_ok=True)

        # combine individual arcs into master arc
        if skip_done and os.path.isfile(carc_path):
            log.info(f"skipping combined arc {carc_path}, file already exists")
        else:
            darc_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype=arc["imagetyp"], **arc) for arc in arcs.to_dict("records")]
            image_tasks.create_master_frame(in_images=darc_paths, out_image=carc_path, batch_size=48)

        # extract arc
        if skip_done and os.path.isfile(xarc_path):
            log.info(f"skipping extracted arc {xarc_path}, file already exists")
        else:
            image_tasks.extract_spectra(in_image=carc_path, out_rss=xarc_path, in_trace=mtrace_path, in_fwhm=mwidth_path, method="optimal")

        # fit wavelength solution
        rss_tasks.determine_wavelength_solution(in_arcs=xarc_path, out_wave=mwave_path, out_lsf=mlsf_path, aperture=12,
                                                cc_correction=True, cc_max_shift=20, poly_disp=5, poly_fwhm=2, poly_cros=2,
                                                flux_min=1e-12, fwhm_max=5, rel_flux_limits=[0.001, 1e12])

    arc = frames.iloc[0]
    for channel in "brz":
        mwave_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], camera=f"{channel}?", kind="mwave"))
        mlsf_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], camera=f"{channel}?", kind="mlsf"))

        xarc_paths = sorted(path.expand("lvm_anc", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], kind="x", imagetype=arc["imagetyp"], camera=f"{channel}?", expnum=expnum_str))
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], kind="x", imagetype=arc["imagetyp"], camera=channel, expnum=expnum_str)
        harc_path = path.full("lvm_anc", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], kind="h", imagetype=arc["imagetyp"], camera=channel, expnum=expnum_str)

        # stack spectragraphs
        rss_tasks.stack_spectrographs(in_rsss=xarc_paths, out_rss=xarc_path)
        # apply wavelength solution to arcs
        rss_tasks.create_pixel_table(in_rss=xarc_path, out_rss=harc_path, in_waves=mwave_paths, in_lsfs=mlsf_paths)
        # rectify arcs
        rss_tasks.resample_wavelength(in_rss=harc_path, out_rss=harc_path, method="linear", wave_range=SPEC_CHANNELS[channel], wave_disp=0.5)


@cloup.command(short_help='Run the calibration sequence reduction', show_constraints=True)
@click.option('-m', '--mjds', type=int, multiple=True, help='list of MJDs with calibration sequence taken')
@click.option('--target-mjd', type=int, help='MJD to store the resulting master frames in')
@click.option('-e', '--expnums', type=int, multiple=True, help='list of exposure numbers to target for reduction')
@click.option('--pixelflats', is_flag=True, default=False, help='flag to create pixel flats')
@click.option('--pixelmasks', is_flag=True, default=False, help='flag to create pixel masks')
@click.option('-i', '--illumination-corrections', is_flag=True, default=False, help='flag to create illumination corrections')
def run_calibration_sequence(mjds, target_mjd=None, expnums=None,
                             pixelflats: bool = False, pixelmasks: bool = False,
                             illumination_corrections: bool = False):
    """Run the calibration sequence reduction

    Given a set of MJDs and (optionally) exposure numbers, run the calibration
    sequence reduction. This routine will store the master calibration frames
    in the corresponding calibration directory in the `target_mjd` or by
    default in the smallest MJD in `mjds`.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
    expnums : list
        List of exposure numbers to reduce
    pixelflats : bool
        Flag to create pixel flats
    pixelmasks : bool
        Flag to create pixel masks
    illumination_corrections : bool
        Flag to create illumination corrections
    """

    # split exposures into the exposure sequence of each type of master frame
    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums)
    bias_frames = frames.query("imagetyp == 'bias'")
    dark_frames = frames.query("imagetyp == 'dark'")
    ldls_frames = frames.query("imagetyp == 'flat & ldls")
    qrtz_frames = frames.query("imagetyp == 'flat & quartz'")
    twilight_frames = frames.query("imagetyp == 'flat'")
    arc_frames = frames.query("imagetyp == 'arc'")

    # TODO: verify sequences completeness

    # reduce bias/dark
    create_detrending_frames(mjds, target_mjd=target_mjd, expnums=set(bias_frames.expnum), kind="bias")
    create_detrending_frames(mjds, target_mjd=target_mjd, expnums=set(dark_frames.expnum), kind="dark")

    # create traces
    create_traces(mjds, target_mjd=target_mjd, expnums_ldls=set(ldls_frames.expnum), expnums_qrtz=set(qrtz_frames.expnum), subtract_straylight=True)

    # create fiber flats
    create_fiberflats(mjds, target_mjd=target_mjd, expnums=set(twilight_frames.expnum))

    # create wavelength solutions
    create_wavelengths(mjds, target_mjd=target_mjd, expnums=set(arc_frames.expnum))

    # create pixel flats
    if pixelflats:
        pixflat_frames = frames.query("imagetyp == 'pixelflat'")
        create_detrending_frames(mjds, target_mjd=target_mjd, expnums=set(pixflat_frames.expnum), kind="pixflat")

    # create pixel mask
    if pixelmasks:
        create_pixelmasks(mjds, target_mjd=target_mjd,
                          dark_expnums=set(dark_frames.expnum),
                          pixflat_expnums=set(pixflat_frames.expnum),
                          ignore_pixflats=False)

    # create illumination corrections
    if illumination_corrections:
        create_illumination_corrections(mjds, target_mjd=target_mjd, expnums=expnums)


if __name__ == '__main__':
    import tracemalloc

    tracemalloc.start()

    MJD = 60255
    ldls_expnums = np.arange(7264, 7269+1)
    ldls_expnums = [None] * (12-ldls_expnums.size) + ldls_expnums.tolist()
    qrtz_expnums = np.arange(7252, 7263+1)

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
        # create_detrending_frames(mjds=60255, target_mjd=60255, kind="bias")
        create_traces(mjds=MJD, expnums_ldls=ldls_expnums, expnums_qrtz=qrtz_expnums, subtract_straylight=True)
        # create_wavelengths(mjds=60264, target_mjd=60255, expnums=[7750,7751])
    except Exception as e:
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)

        raise e