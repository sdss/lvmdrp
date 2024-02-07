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
from itertools import product
from astropy.io import fits

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.core.constants import SPEC_CHANNELS
from lvmdrp.core.tracemask import TraceMask

from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.functions.run_drp import get_config_options, read_fibermap


SLITMAP = read_fibermap(as_table=True)
MASTER_CON_LAMPS = {"b": "ldls", "r": "ldls", "z": "quartz"}
MASTER_ARC_LAMPS = {"b": "hgne", "r": "neon", "z": "neon"}
MASTERS_DIR = os.getenv("LVM_MASTER_DIR")


def get_sequence_metadata(mjds, target_mjd=None, expnums=None):
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
    frames = [md.get_metadata(tileid="*", mjd=mjd) for mjd in mjds]
    frames = pd.concat(frames, ignore_index=True)

    # filter by given expnums
    if expnums is not None:
        frames.query("expnum in @expnums", inplace=True)

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


def reduce_2d(mjds, target_mjd=None, expnums=None):
    """Preprocess and detrend a list of 2D frames

    Given a set of MJDs and (optionally) exposure numbers, preprocess and
    detrend the 2D frames. This routine will store the preprocessed and
    detrended frames in the corresponding calibration directory in the
    `target_mjd` or by default in the smallest MJD in `mjds`.

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
    masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    # preprocess and detrend frames
    for frame in frames.to_dict("records"):
        camera = frame["camera"]
        imagetyp = frame["imagetyp"]
        # get master frames paths
        # masters = find_masters(masters_mjd, imagetyp, camera)
        mpixmask_path = os.path.join(masters_path, f"lvm-mpixmask-{camera}.fits")
        mbias_path = os.path.join(masters_path, f"lvm-mbias-{camera}.fits")
        mdark_path = os.path.join(masters_path, f"lvm-mdark-{camera}.fits")
        mpixflat_path = os.path.join(masters_path, f"lvm-mpixflat-{camera}.fits")

        # log the master frames
        log.info(f'Using master pixel mask: {mpixmask_path}')
        log.info(f'Using master bias: {mbias_path}')
        log.info(f'Using master dark: {mdark_path}')
        log.info(f'Using master pixel flat: {mpixflat_path}')

        frame_path = path.full("lvm_raw", camspec=frame["camera"], **frame)
        pframe_path = path.full("lvm_anc", drpver=drpver, kind="p", imagetype=imagetyp, **frame)

        # bypass creation of detrended frame in case of imagetyp=bias
        if imagetyp != "bias":
            dframe_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=imagetyp, **frame)
        else:
            dframe_path = pframe_path

        os.makedirs(os.path.dirname(dframe_path), exist_ok=True)
        if os.path.isfile(dframe_path):
            log.info(f"skipping {dframe_path}, file already exist")
        else:
            image_tasks.preproc_raw_frame(in_image=frame_path, out_image=pframe_path, in_mask=mpixmask_path)
            image_tasks.detrend_frame(in_image=pframe_path, out_image=dframe_path,
                                        in_bias=mbias_path, in_dark=mdark_path,
                                        in_pixelflat=mpixflat_path,
                                        in_slitmap=SLITMAP)


def create_detrending_frames(mjds, target_mjd=None, expnums=None, kind="all", assume_imagetyp=None):
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
    kind : str
        Kind of frame to reduce
    assume_imagetyp : str
        Assume the given imagetyp for all frames
    """
    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums)

    # filter by target image types
    if kind == "all":
        frames.query("imagetyp in ['bias', 'dark', 'pixflat']", inplace=True)
    elif kind in ["bias", "dark", "pixflat"]:
        frames.query("imagetyp == @kind", inplace=True)
    else:
        raise ValueError(f"Invalid kind: '{kind}'. Must be one of 'bias', 'dark', 'pixflat' or 'all'")

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

            # preprocess and detrend frames
            reduce_2d(mjds=set(analogs.mjd), target_mjd=masters_mjd, expnums=analogs.expnum.values)

            # combine into master frame
            kwargs = get_config_options('reduction_steps.create_master_frame', imagetyp)
            log.info(f'custom configuration parameters for create_master_frame: {repr(kwargs)}')
            mframe_path = path.full("lvm_master", drpver=drpver, tileid=frame["tileid"], mjd=masters_mjd, kind=f'm{imagetyp}', camera=frame["camera"])
            os.makedirs(os.path.dirname(mframe_path), exist_ok=True)

            dframe_paths = [path.full("lvm_anc", drpver=drpver, kind="d" if imagetyp != "bias" else "p", imagetype=imagetyp, **frame) for frame in analogs.to_dict("records")]
            image_tasks.create_master_frame(in_images=dframe_paths, out_image=mframe_path, **kwargs)

    # ancillary paths clean up
    _clean_ancillary(mjds=mjds, expnums=expnums, kind=kind)


def create_pixelmasks(mjds, target_mjd=None, expnums=None, ignore_darks=True):
    """Create pixel mask from master pixelflat and/or dark frames

    Given a set of MJDs and (optionally) exposure numbers, create a pixel mask
    from the master pixelflat and/or dark frames. This routine will store the
    master pixelmask in the corresponding calibration directory in the
    `target_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding master pixelflat and/or dark frames do not exist, they
    will be created first. Otherwise they will be read from disk. If
    `ignore_darks` is True, then the dark frames will not be used to create the
    pixel mask.

    Parameters:
    ----------
    mjds : list
        List of MJDs to reduce
    target_mjd : float
        MJD to store the master frames in
    expnums : list
        List of exposure numbers to reduce
    ignore_darks : bool
        Ignore dark frames when creating pixel mask

    """
    pass


def create_traces(mjds, target_mjd=None, expnums_ldls=None, expnums_qrtz=None,
                  subtract_straylight=False, fit_poly=True, poly_deg_amp=5,
                  poly_deg_cent=4, poly_deg_width=5):
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
    """
    expnums = np.concatenate([expnums_ldls, expnums_qrtz])
    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums)

    reduce_2d(mjds, target_mjd=masters_mjd, expnums=expnums)

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
        # if camera != "r1":
        #     continue
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
            dflat_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="flat", camera=camera, expnum=expnum)
            sflat_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="s", imagetype="flat", camera=camera, expnum=expnum)
            flux_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="flux", camera=camera, expnum=expnum)
            cent_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="cent", camera=camera, expnum=expnum)
            fwhm_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="fwhm", camera=camera, expnum=expnum)
            model_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="model", camera=camera, expnum=expnum)
            mratio_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="mratio", camera=camera, expnum=expnum)
            mstray_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="stray", camera=camera, expnum=expnum)
            stray_path = path.full("lvm_anc", drpver=drpver, tileid=tileid, mjd=masters_mjd, kind="d", imagetype="stray_model", camera=camera, expnum=expnum)

            if subtract_straylight:
                # trace only centroids
                centroids, img = image_tasks.trace_fibers(
                    in_image=dflat_path,
                    out_trace_cent=cent_path,
                    correct_ref=True, median_box=(1,10), coadd=20,
                    counts_threshold=counts_threshold, max_diff=1.5,
                    guess_fwhm=2.5, method="gauss", ncolumns=140,
                    fit_poly=True, poly_deg=poly_deg_cent,
                    interpolate_missing=True, only_centroids=True
                )

                # subtract stray light
                image_tasks.subtract_straylight(
                    in_image=dflat_path,
                    out_image=sflat_path,
                    in_cent_trace=cent_path,
                    out_stray=mstray_path,
                    smooth_disp=11, aperture=5, poly_cross=9, smooth_gauss=10
                )
            else:
                sflat_path = dflat_path

            log.info(f"going to trace std fiber {fiber_str} in {camera} within {block_idxs = }")
            centroids, trace_cent_fit, trace_flux_fit, trace_fwhm_fit, img_stray, model, mratio = image_tasks.trace_fibers(
                in_image=sflat_path,
                out_trace_amp=flux_path, out_trace_cent=cent_path, out_trace_fwhm=fwhm_path,
                out_trace_cent_guess=None,
                correct_ref=True, median_box=(1,10), coadd=20,
                counts_threshold=counts_threshold, max_diff=1.5, guess_fwhm=2.5, method="gauss",
                ncolumns=(140, 40), iblocks=block_idxs, fwhm_limits=(1.5, 4.5),
                fit_poly=fit_poly, interpolate_missing=False, poly_deg=(poly_deg_amp, poly_deg_cent, poly_deg_width)
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
        fiber_strs = list(zip(*expnum_params[camera]))[2]
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

        # save master traces
        mamp_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=masters_mjd, camera=camera, kind="mamps")
        mcent_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=masters_mjd, camera=camera, kind="mtrace")
        mwidth_path = path.full("lvm_master", drpver=drpver, tileid=tileid, mjd=masters_mjd, camera=camera, kind="mwidth")
        os.makedirs(os.path.dirname(mamp_path), exist_ok=True)
        mamps[camera].writeFitsData(mamp_path)
        mcents[camera].writeFitsData(mcent_path)
        mwidths[camera].writeFitsData(mwidth_path)

        # eval model continuum and ratio
        model, ratio = image_tasks._eval_continuum_model(img_stray, mamps[camera], mcents[camera], mwidths[camera])
        model.writeFitsData(model_path)
        ratio.writeFitsData(mratio_path)

        if subtract_straylight:
            stray_model = fits.HDUList()
            stray_model.append(fits.PrimaryHDU(data=img._data, header=img._header))
            stray_model.append(fits.ImageHDU(data=img_stray._data, name="STRAY_CORR"))
            stray_model.append(fits.ImageHDU(data=img._data-img_stray._data, name="STRAYLIGHT"))
            stray_model.append(fits.ImageHDU(data=model._data, name="CONT_MODEL"))
            stray_model.append(fits.ImageHDU(data=img_stray._data-model._data, name="STRAY_MODEL"))
            stray_model.append(fits.ImageHDU(data=img._data-model._data, name="NOSTRAY_MODEL"))
            stray_model.writeto(stray_path, overwrite=True)

    return mamps, mcents, mwidths, img, img_stray, model, ratio


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
    # read master twilight flats
    # read master traces
    # extract twilight fibers
    # fit continuum avoiding absorption/emission lines
    # calculate median continuum
    # normalize fibers by median continuum
    # write master fiber flats
    pass


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
    pass


def create_wavelengths(mjds, target_mjd=None, expnums=None):
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
    """
    frames, masters_mjd = get_sequence_metadata(mjds, target_mjd=target_mjd, expnums=expnums)
    masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    reduce_2d(mjds, target_mjd=masters_mjd, expnums=expnums)

    arc_analogs = frames.query("imagetyp=='arc'").groupby(["camera",])
    for camera in arc_analogs.groups:
        arcs = arc_analogs.get_group((camera,))
        arc = arcs.iloc[0]

        # define master paths for target frames
        mtrace_path = os.path.join(masters_path, f"lvm-mtrace-{camera}.fits")
        mwidth_path = os.path.join(masters_path, f"lvm-mwidth-{camera}.fits")

        # define product paths
        marc_path = path.full("lvm_master", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], kind="marc", camera=arc["camera"])
        xarc_path = path.full("lvm_master", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], kind="xarc", camera=arc["camera"])
        os.makedirs(os.path.dirname(marc_path), exist_ok=True)

        # combine individual arcs into master arc
        if os.path.isfile(marc_path):
            log.info(f"skipping master arc {marc_path}, file already exists")
        else:
            darc_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype=arc["imagetyp"], **arc) for arc in arcs.to_dict("records")]
            image_tasks.create_master_frame(in_images=darc_paths, out_image=marc_path)

        # extract arc
        if os.path.isfile(xarc_path):
            log.info(f"skipping extracted arc {xarc_path}, file already exists")
        else:
            image_tasks.extract_spectra(in_image=marc_path, out_rss=xarc_path, in_trace=mtrace_path, in_fwhm=mwidth_path, method="optimal")

        # define products paths
        xarc_path = path.full("lvm_master", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], kind="xarc", camera=arc["camera"])
        harc_path = path.full("lvm_master", drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], kind="harc", camera=arc["camera"])

        # fit wavelength solution
        out_wave, out_lsf = os.path.join("data_wave", f"lvm-mwave-{camera}.fits"), os.path.join("data_wave", f"lvm-mlsf-{camera}.fits")
        rss_tasks.determine_wavelength_solution(in_arcs=xarc_path, out_wave=out_wave, out_lsf=out_lsf, aperture=12,
                                                cc_correction=True, cc_max_shift=20, poly_disp=5, poly_fwhm=2, poly_cros=2,
                                                flux_min=1e-12, fwhm_max=5, rel_flux_limits=[0.001, 1e12])

        # apply wavelength solution to arcs
        rss_tasks.create_pixel_table(in_rss=xarc_path, out_rss=harc_path, arc_wave=out_wave, arc_fwhm=out_lsf)

        # rectify arcs
        iwave, fwave = SPEC_CHANNELS[camera[0]]
        rss_tasks.resample_wavelength(in_rss=harc_path, out_rss=harc_path,
                                      method="linear", disp_pix=0.5,
                                      start_wave=iwave, end_wave=fwave,
                                      err_sim=10, parallel=0, extrapolate=False)

@cloup.command(short_help='Run the calibration sequence reduction', show_constraints=True)
@click.option('-m', '--mjds', type=int, multiple=True, help='list of MJDs with calibration sequence taken')
@click.option('--target-mjd', type=int, help='MJD to store the resulting master frames in')
@click.option('-e', '--expnums', type=int, multiple=True, help='list of exposure numbers to target for reduction')
@click.option('-i', '--illumination-corrections', is_flag=True, default=False, help='flag to create illumination corrections')
def run_calibration_sequence(mjds, target_mjd=None, expnums=None, illumination_corrections: bool = False):
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
    """

    # reduce bias/dark/pixflat
    create_detrending_frames(mjds, target_mjd=target_mjd, expnums=expnums)

    # create pixel mask
    create_pixelmasks(mjds, target_mjd=target_mjd, expnums=expnums)

    # create traces
    create_traces(mjds, target_mjd=target_mjd, expnums=expnums)

    # create fiber flats
    create_fiberflats(mjds, target_mjd=target_mjd, expnums=expnums)

    # create illumination corrections
    if illumination_corrections:
        create_illumination_corrections(mjds, target_mjd=target_mjd, expnums=expnums)

    # create wavelength solutions
    create_wavelengths(mjds, target_mjd=target_mjd, expnums=expnums)


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

    try:
        # create_detrending_frames(mjds=60255, target_mjd=60255, kind="bias")
        # create_traces(mjds=MJD, expnums_ldls=ldls_expnums, expnums_qrtz=qrtz_expnums, subtract_straylight=True)
        create_wavelengths(mjds=60264, target_mjd=60255, expnums=[7750,7751])
    except Exception as e:
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)

        raise e