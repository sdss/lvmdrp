#!/usr/bin/env python
# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Aug 9, 2023
# @Filename: quickdrp
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import numpy as np
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


def illumination_correction(in_fiberflat: bool = None) -> dict:
    if in_fiberflat is None:
        # NOTE: this was done using exposure 3900
        factors = {('SkyW', 'b1'): 0.8854398,
                   ('SkyE', 'b1'): 1.0811657,
                   ('Sci', 'b1'): 1.0333946,
                   ('Spec', 'b1'): 1.3,
                   ('SkyW', 'r1'): 0.9435192,
                   ('SkyE', 'r1'): 1.0190876,
                   ('Sci', 'r1'): 1.0373933,
                   ('Spec', 'r1'): 1.3,
                   ('SkyW', 'z1'): 0.9378496,
                   ('SkyE', 'z1'): 1.0072966,
                   ('Sci', 'z1'): 1.0548539,
                   ('Spec', 'z1'): 1.3,
                   ('SkyW', 'b2'): 0.9148656,
                   ('SkyE', 'b2'): 1.0618604,
                   ('Sci', 'b2'): 1.023274,
                   ('Spec', 'b2'): 1.3,
                   ('SkyW', 'r2'): 0.9465178,
                   ('SkyE', 'r2'): 1.0136424,
                   ('Sci', 'r2'): 1.0398397,
                   ('Spec', 'r2'): 1.3,
                   ('SkyW', 'z2'): 0.93933785,
                   ('SkyE', 'z2'): 1.0083715,
                   ('Sci', 'z2'): 1.0522904,
                   ('Spec', 'z2'): 1.3,
                   ('SkyW', 'b3'): 0.88334155,
                   ('SkyE', 'b3'): 1.0627162,
                   ('Sci', 'b3'): 1.0539422,
                   ('Spec', 'b3'): 1.3,
                   ('SkyW', 'r3'): 0.9305622,
                   ('SkyE', 'r3'): 1.013976,
                   ('Sci', 'r3'): 1.0554619,
                   ('Spec', 'r3'): 1.3,
                   ('SkyW', 'z3'): 0.9152639,
                   ('SkyE', 'z3'): 1.0191175,
                   ('Sci', 'z3'): 1.0656188,
                   ('Spec', 'z3'): 1.3}

        return factors

    fibers = [2, 106, 37]
    fibermap = Table(drp.fibermap.data)
    select = np.isin(fibermap["fiberid"]-1, fibers)
    skw1_fibers = fibermap[~select][fibermap[~select]["ifulabel"] == "SkyW1"]["fiberid"] - 1
    ske1_fibers = fibermap[~select][fibermap[~select]["ifulabel"] == "SkyE1"]["fiberid"] - 1
    sci1_fibers = fibermap[~select][fibermap[~select]["ifulabel"] == "Sci1"]["fiberid"] - 1

    factors = {}
    for spec in "123":
        for cam in ["b", "r", "z"]:
            cam = cam + spec
            fflat = rss_tasks.RSS()
            fflat.loadFitsData(f"/home/mejia/Research/lvm/lvmdata/calib/60177/lvm-mfiberflat-{cam}.fits")
            fflat._data[(fflat._mask) | (fflat._data <= 0)] = np.nan
            sci_factor = np.nanmedian(fflat._data[sci1_fibers][:, 1000:3000])
            skw_factor = np.nanmedian(fflat._data[skw1_fibers][:, 1000:3000])
            ske_factor = np.nanmedian(fflat._data[ske1_fibers][:, 1000:3000])
            norm = np.mean([sci_factor, skw_factor, ske_factor])
            factors[("SkyW", cam)] = skw_factor/norm
            factors[("SkyE", cam)] = ske_factor/norm
            factors[("Sci", cam)] = sci_factor/norm

        return factors


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
    sci_metadata = md.get_metadata(tileid="*", mjd="*", expnum=expnum, imagetyp="object")
    sci_metadata.sort_values("expnum", ascending=False, inplace=True)

    # define general metadata
    sci_tileid = sci_metadata["tileid"].unique()[0]
    sci_mjd = sci_metadata["mjd"].unique()[0]
    sci_expnum = sci_metadata["expnum"].unique()[0]
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

        # define sci paths
        rsci_path = path.full("lvm_raw", camspec=sci_camera, **sci)
        psci_path = path.full("lvm_anc", drpver=drpver, kind="p", imagetype=sci["imagetyp"], **sci)
        dsci_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=sci["imagetyp"], **sci)
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
            # macorr_path = os.path.join(masters_path, f"lvm-apercorr-{sci_camera}.fits")
            mwave_path = os.path.join(masters_path, f"lvm-mwave_{lamps}-{sci_camera}.fits")
            mlsf_path = os.path.join(masters_path, f"lvm-mlsf_{lamps}-{sci_camera}.fits")
            mflat_path = os.path.join(masters_path, f"lvm-mfiberflat_twilight-{sci_camera}.fits")
            if not os.path.isfile(mflat_path):
                mflat_path = os.path.join(masters_path, f"lvm-mfiberflat-{sci_camera}.fits")
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
            # macorr_path = None
            mwave_path = path.full("lvm_master", drpver=drpver, kind=f"mwave_{lamps}", **masters["wave"].to_dict())
            mlsf_path = path.full("lvm_master", drpver=drpver, kind=f"mlsf_{lamps}", **masters["lsf"].to_dict())
            mflat_path = path.full("lvm_master", drpver=drpver, kind="mfiberflat", **masters["fiberflat"].to_dict())

        log.info(f'--- Starting science reduction of raw frame: {rsci_path}')

        # preprocess frame
        image_tasks.preproc_raw_frame(in_image=rsci_path, out_image=psci_path, in_mask=mpixmask_path)

        # detrend frame
        image_tasks.detrend_frame(in_image=psci_path, out_image=dsci_path,
                                  in_bias=mbias_path, in_dark=mdark_path, in_pixelflat=mpixflat_path,
                                  in_slitmap=Table(drp.fibermap.data), reject_cr=False)

        # # extract 1d spectra
        image_tasks.extract_spectra(in_image=dsci_path, out_rss=xsci_path, in_trace=mtrace_path, in_fwhm=mwidth_path, method=extraction_method, parallel=extraction_parallel)

        # wavelength calibrate
        rss_tasks.create_pixel_table(in_rss=xsci_path, out_rss=wsci_path, arc_wave=mwave_path, arc_fwhm=mlsf_path)

        # apply fiberflat correction
        rss_tasks.apply_fiberflat(in_rss=wsci_path, out_rss=fsci_path, in_flat=mflat_path)

        # NOTE: this is a temporary fix for the illumination bias across telescopes whe using dome flats
        rss = rss_tasks.RSS()
        factors = illumination_correction(in_fiberflat=None)
        rss.loadFitsData(fsci_path)
        fibermap = rss._slitmap
        fibermap = fibermap[fibermap["spectrographid"] == int(sci_camera[1])]
        for ifiber in range(rss._fibers):
            f = factors.get((fibermap[ifiber]["telescope"], sci_camera), 1)
            # print(f"applying factor {(fibermap[ifiber]['telescope'], sci_camera)} : {f}")
            rss._data[ifiber] *= f
            rss._error[ifiber] *= f
        rss.writeFitsData(fsci_path)

        # interpolate sky fibers
        sky_tasks.interpolate_sky(in_rss=fsci_path, out_sky=fskye_path, which="e")
        sky_tasks.interpolate_sky(in_rss=fsci_path, out_sky=fskyw_path, which="w")

        # compute master sky and subtract if requested
        sky_tasks.quick_sky_subtraction(in_rss=fsci_path, out_rss=ssci_path, in_skye=fskye_path, in_skyw=fskyw_path, sky_weights=sky_weights, skip_subtraction=skip_sky_subtraction)

        # resample wavelength into uniform grid along fiber IDs for science and sky fibers
        iwave, fwave = SPEC_CHANNELS[sci_camera[0]]
        rss_tasks.resample_wavelength(in_rss=ssci_path,  out_rss=hsci_path, method="linear", compute_densities=True, disp_pix=0.5, start_wave=iwave, end_wave=fwave, err_sim=10, parallel=0, extrapolate=False)
        rss_tasks.resample_wavelength(in_rss=fskye_path, out_rss=hskye_path, method="linear", compute_densities=True, disp_pix=0.5, start_wave=iwave, end_wave=fwave, err_sim=10, parallel=0, extrapolate=False)
        rss_tasks.resample_wavelength(in_rss=fskyw_path, out_rss=hskyw_path, method="linear", compute_densities=True, disp_pix=0.5, start_wave=iwave, end_wave=fwave, err_sim=10, parallel=0, extrapolate=False)

        # use sky subtracted resampled frames for flux calibration in each camera
        flux_tasks.fluxcal_Gaia(sci_camera, hsci_path, GAIA_CACHE_DIR=ORIG_MASTER_DIR+'/gaia_cache')

    # # combine spectrographs
    drp.combine_spectrographs(tileid=sci_tileid, mjd=sci_mjd, channel="b", expnum=sci_expnum)
    drp.combine_spectrographs(tileid=sci_tileid, mjd=sci_mjd, channel="r", expnum=sci_expnum)
    drp.combine_spectrographs(tileid=sci_tileid, mjd=sci_mjd, channel="z", expnum=sci_expnum)

    # flux-calibrate each channel
    sci_paths = sorted(drp.path.expand("lvm_anc", drpver=drpver, tileid=sci_tileid, mjd=sci_mjd, kind="", imagetype="object", camera="*", expnum=sci_expnum))
    sci_paths = [sci_path for sci_path in sci_paths if "lvm-object-sp" not in sci_path]
    for sci_path in sci_paths:
        flux_tasks.apply_fluxcal(in_rss=sci_path, out_rss=sci_path, skip_fluxcal=skip_flux_calibration)

    # combine channels
    drp.combine_channels(tileid=sci_tileid, mjd=sci_mjd, expnum=sci_expnum)

    # refine sky subtraction
    sky_tasks.quick_sky_refinement(in_cframe=path.full("lvm_frame", mjd=sci_mjd, drpver=drpver, tileid=sci_tileid, expnum=sci_expnum, kind='CFrame'))

    # TODO: add quick report routine

    # TODO: combine exposures
    # TODO: by default remove the extra files for the given expnum

    # TODO: flux calibration
