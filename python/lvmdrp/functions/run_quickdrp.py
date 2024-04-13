#!/usr/bin/env python
# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Aug 9, 2023
# @Filename: quickdrp
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
from glob import glob
from typing import Tuple

from astropy.table import Table

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.functions import run_drp as drp
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.functions import skyMethod as sky_tasks
from lvmdrp.functions import fluxCalMethod as flux_tasks
from lvmdrp.core.constants import SPEC_CHANNELS


ORIG_MASTER_DIR = os.getenv("LVM_MASTER_DIR")


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
    masters_dir = sorted([f for f in os.listdir(ORIG_MASTER_DIR)
                          if os.path.isdir(os.path.join(ORIG_MASTER_DIR, f))])
    masters_dir = [f for f in masters_dir if f.isdigit()]
    target_master = list(filter(lambda f: sci_mjd >= int(f), masters_dir))
    return int(target_master[-1])


def quick_science_reduction(expnum: int, use_fiducial_master: bool = False,
                            skip_sky_subtraction: bool = False,
                            sky_weights: Tuple[float, float] = None,
                            skip_flux_calibration: bool = False,
                            ncpus: int = None,
                            aperture_extraction: bool = False) -> None:
    """ Run the Quick DRP for a given exposure number.
    """
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
    extraction_parallel = "auto" if ncpus is None else ncpus
    extraction_method = "aperture" if aperture_extraction else "optimal"

    # get target frames metadata
    sci_metadata = md.get_metadata(tileid="*", mjd="*", expnum=expnum)
    sci_metadata.sort_values("expnum", ascending=False, inplace=True)

    # define general metadata
    sci_tileid = sci_metadata["tileid"].unique()[0]
    sci_mjd = sci_metadata["mjd"].unique()[0]
    sci_expnum = sci_metadata["expnum"].unique()[0]
    sci_imagetyp = sci_metadata["imagetyp"].unique()[0]
    log.info(f"running Quick DRP for tile {sci_tileid} at MJD {sci_mjd} with exposure number {sci_expnum}")

    master_mjd = get_master_mjd(sci_mjd)
    log.info(f"target master MJD: {master_mjd}")

    # overwrite fiducial masters dir
    os.environ["LVM_MASTER_DIR"] = os.path.join(ORIG_MASTER_DIR, f"{master_mjd}")
    log.info(f"target master path: {os.getenv('LVM_MASTER_DIR')}")

    # make sure only one exposure number is being reduced
    sci_metadata.query("expnum == @sci_expnum", inplace=True)
    sci_metadata.sort_values("camera", inplace=True)

    # define arc lamps configuration per spectrograph channel
    if master_mjd == 60142:
        arc_lamps = {"b": "hgne", "r": "neon", "z": "neon"}
    else:
        arc_lamps = {"b": "neon_hgne_argon_xenon", "r": "neon_hgne_argon_xenon", "z": "neon_hgne_argon_xenon"}

    # run reduction loop for each science camera exposure
    for sci in sci_metadata.to_dict("records"):
        # define science camera
        sci_camera = sci["camera"]

        # define ancillary product paths
        rsci_path = path.full("lvm_raw", camspec=sci_camera, **sci)
        psci_path = path.full("lvm_anc", drpver=drpver, kind="p", imagetype=sci["imagetyp"], **sci)
        dsci_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=sci["imagetyp"], **sci)
        lsci_path = path.full("lvm_anc", drpver=drpver, kind="l", imagetype=sci["imagetyp"], **sci)
        xsci_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=sci["imagetyp"], **sci)
        wsci_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype=sci["imagetyp"], **sci)
        fsci_path = path.full("lvm_anc", drpver=drpver, kind="f", imagetype=sci["imagetyp"], **sci)
        ssci_path = path.full("lvm_anc", drpver=drpver, kind="s", imagetype=sci["imagetyp"], **sci)
        hsci_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype=sci["imagetyp"], **sci)
        fskye_path = path.full("lvm_anc", drpver=drpver, kind="f", imagetype="sky_e", **sci)
        fskyw_path = path.full("lvm_anc", drpver=drpver, kind="f", imagetype="sky_w", **sci)
        hskye_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype="sky_e", **sci)
        hskyw_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype="sky_w", **sci)
        os.makedirs(os.path.dirname(hsci_path), exist_ok=True)

        # define science product paths
        frame_path = path.full("lvm_frame", drpver=drpver, tileid=sci_tileid, mjd=sci_mjd, expnum=sci_expnum, kind=f"Frame-{sci_camera}")
        # define current arc lamps to use for wavelength calibration
        lamps = arc_lamps[sci_camera[0]]

        # define calibration frames paths
        if use_fiducial_master:
            masters_path = os.getenv("LVM_MASTER_DIR")
            log.info(f"using fiducial master calibration frames for {sci_camera} at $LVM_MASTER_DIR = {masters_path}")
            if masters_path is None:
                raise ValueError("LVM_MASTER_DIR environment variable is not defined")
            mpixmask_path = os.path.join(masters_path, f"lvm-mpixmask-{sci_camera}.fits")
            mbias_path = os.path.join(masters_path, f"lvm-mbias-{sci_camera}.fits")
            mdark_path = os.path.join(masters_path, f"lvm-mdark-{sci_camera}.fits")
            mpixflat_path = os.path.join(masters_path, f"lvm-mpixflat-{sci_camera}.fits")
            mtrace_path = os.path.join(masters_path, f"lvm-mtrace-{sci_camera}.fits")
            mwidth_path = os.path.join(masters_path, f"lvm-mwidth-{sci_camera}.fits")
        else:
            log.info(f"using master calibration frames from DRP version {drpver}, mjd = {sci_mjd}, camera = {sci_camera}")
            masters = md.match_master_metadata(target_mjd=sci_mjd,
                                               target_camera=sci_camera,
                                               target_imagetyp=sci["imagetyp"])
            mpixmask_path = path.full("lvm_master", drpver=drpver, kind="mpixmask", **masters["pixmask"].to_dict())
            mbias_path = path.full("lvm_master", drpver=drpver, kind="mbias", **masters["bias"].to_dict())
            mdark_path = path.full("lvm_master", drpver=drpver, kind="mdark", **masters["dark"].to_dict())
            mpixflat_path = None
            mtrace_path = path.full("lvm_master", drpver=drpver, kind="mtrace", **masters["trace"].to_dict())
            mwidth_path = None

        log.info(f'--- Starting science reduction of raw frame: {rsci_path}')

        # preprocess frame
        image_tasks.preproc_raw_frame(in_image=rsci_path, out_image=psci_path, in_mask=mpixmask_path)

        # detrend frame
        image_tasks.detrend_frame(in_image=psci_path, out_image=dsci_path,
                                  in_bias=mbias_path, in_dark=mdark_path, in_pixelflat=mpixflat_path,
                                  in_slitmap=Table(drp.fibermap.data), reject_cr=False)

        # subtract straylight
        if sci_imagetyp == "flat":
            image_tasks.subtract_straylight(in_image=dsci_path, out_image=lsci_path,
                                                in_cent_trace=mtrace_path, select_nrows=5,
                                                aperture=13, smoothing=400, median_box=21, gaussian_sigma=0.0)
        else:
            lsci_path = dsci_path

        # extract 1d spectra
        image_tasks.extract_spectra(in_image=dsci_path, out_rss=xsci_path, in_trace=mtrace_path, in_fwhm=mwidth_path,
                                    method=extraction_method, parallel=extraction_parallel)

    # per channel reduction
    for channel in "brz":
        xsci_paths = sorted(path.expand('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                                        kind='x', camera=f'{channel}[123]', imagetype=sci_imagetyp, expnum=expnum))
        xsci_path = path.full('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                              kind='x', camera=channel, imagetype=sci_imagetyp, expnum=expnum)
        wsci_path = path.full('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                              kind='w', camera=channel, imagetype=sci_imagetyp, expnum=expnum)
        mwave_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwave_{lamps}-{channel}?.fits")))
        mlsf_paths = sorted(glob(os.path.join(masters_path, f"lvm-mlsf_{lamps}-{channel}?.fits")))
        frame_path = path.full('lvm_frame', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver, expnum=sci_expnum, kind=f'Frame-{channel}')
        mflat_paths = sorted(glob(os.path.join(masters_path, f"lvm-mfiberflat_twilight-{channel}?.fits")))
        if not mflat_paths:
            mflat_paths = sorted(glob(os.path.join(masters_path, f"lvm-mfiberflat-{channel}?.fits")))
        ssci_path = path.full('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                              kind='s', camera=channel, imagetype=sci_imagetyp, expnum=expnum)
        hsci_path = path.full('lvm_anc', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver,
                              kind='h', camera=channel, imagetype=sci_imagetyp, expnum=expnum)

        # stack spectrographs
        rss_tasks.stack_spectrographs(in_rsss=xsci_paths, out_rss=xsci_path)

        # wavelength calibrate
        rss_tasks.create_pixel_table(in_rss=xsci_path, out_rss=wsci_path, in_waves=mwave_paths, in_lsfs=mlsf_paths)

        # apply fiberflat correction
        rss_tasks.apply_fiberflat(in_rss=wsci_path, out_frame=frame_path, in_flats=mflat_paths)

        # interpolate sky fibers
        sky_tasks.interpolate_sky(in_frame=frame_path, out_rss=ssci_path)

        # combine sky telescopes
        sky_tasks.combine_skies(in_rss=ssci_path, out_rss=ssci_path, sky_weights=sky_weights)

        # resample wavelength into uniform grid along fiber IDs for science and sky fibers
        rss_tasks.resample_wavelength(in_rss=ssci_path,  out_rss=hsci_path, wave_range=SPEC_CHANNELS[channel], wave_disp=0.5, convert_to_density=True)

        # use sky subtracted resampled frames for flux calibration in each camera
        flux_tasks.fluxcal_Gaia(channel, hsci_path, GAIA_CACHE_DIR=ORIG_MASTER_DIR+'/gaia_cache')

        # flux-calibrate each channel
        fframe_path = path.full("lvm_frame", mjd=sci_mjd, drpver=drpver, tileid=sci_tileid, expnum=sci_expnum, kind=f'FFrame-{channel}')
        flux_tasks.apply_fluxcal(in_rss=hsci_path, out_rss=fframe_path, skip_fluxcal=skip_flux_calibration)

    # stitch channels
    fframe_paths = sorted(path.expand('lvm_frame', mjd=sci_mjd, tileid=sci_tileid, drpver=drpver, kind='FFrame-?', expnum=expnum))
    cframe_path = path.full("lvm_frame", drpver=drpver, tileid=sci_tileid, mjd=sci_mjd, expnum=sci_expnum, kind='CFrame')
    rss_tasks.join_spec_channels(in_rsss=fframe_paths, out_rss=cframe_path, use_weights=True)

    # sky subtraction
    sframe_path = path.full("lvm_frame", mjd=sci_mjd, drpver=drpver, tileid=sci_tileid, expnum=sci_expnum, kind='SFrame')
    sky_tasks.quick_sky_subtraction(in_cframe=cframe_path, out_sframe=sframe_path, skip_subtraction=skip_sky_subtraction)

    # TODO: add quick report routine

    # TODO: by default remove the extra files for the given expnum
