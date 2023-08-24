#!/usr/bin/env python
# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Aug 9, 2023
# @Filename: quickdrp
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import click
import cloup

from astropy.table import Table

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.functions import run_drp as drp
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.functions import skyMethod as sky_tasks
from lvmdrp.core.constants import SPEC_CHANNELS


ORIG_MASTER_DIR = os.getenv("LVM_MASTER_DIR")


def get_master_mjd(sci_mjd):

    masters_dir = sorted([f for f in os.listdir(ORIG_MASTER_DIR) if os.path.isdir(os.path.join(ORIG_MASTER_DIR, f))])
    masters_dir = [f for f in masters_dir if f[0]!='.']
    target_master = list(filter(lambda f: sci_mjd >= int(f), masters_dir))
    return int(target_master[-1])

@cloup.command(short_help='Run the Quick DRP', show_constraints=True)
@click.option('-e', '--expnum', type=int, help='an exposure number to reduce')
@click.option('-f', '--use-fiducial-master', is_flag=True, help='use fiducial master calibration frames')
def quick_reduction(expnum: int, use_fiducial_master: bool = False) -> None:
    """ Run the Quick DRP for a given exposure number.
    """
    # get target frames metadata
    sci_metadata = md.get_metadata(tileid="*", mjd="*", expnum=expnum, imagetyp="object")
    sci_metadata.sort_values("expnum", ascending=False, inplace=True)

    # define general metadata
    sci_tileid = sci_metadata["tileid"].unique()[0]
    sci_mjd = sci_metadata["mjd"].unique()[0]
    sci_expnum = sci_metadata["expnum"].unique()[0]
    log.info(f"Running Quick DRP for tile {sci_tileid} at MJD {sci_mjd} with exposure number {sci_expnum}")

    master_mjd = get_master_mjd(sci_mjd)
    log.info(f"target master MJD: {master_mjd}")

    # overwrite fiducial masters dir
    os.environ["LVM_MASTER_DIR"] = os.path.join(ORIG_MASTER_DIR, f"{master_mjd}")
    log.info(f"target master path: {os.getenv('LVM_MASTER_DIR')}")

    # make sure only one exposure number is being reduced
    sci_metadata.query("expnum == @sci_expnum", inplace=True)
    sci_metadata.sort_values("camera", inplace=True)

    # define arc lamps configuration per spectrograph channel
    arc_lamps = {"b": "hgne", "r": "neon", "z": "neon"}

    # run reduction loop for each science camera exposure
    for sci in sci_metadata.to_dict("records"):
        # define science camera
        sci_camera = sci["camera"]

        # define sci paths
        sci_path = path.full("lvm_raw", camspec=sci_camera, **sci)
        psci_path = path.full("lvm_anc", drpver=drpver, kind="p", imagetype=sci["imagetyp"], **sci)
        dsci_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=sci["imagetyp"], **sci)
        xsci_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=sci["imagetyp"], **sci)
        wsci_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype=sci["imagetyp"], **sci)
        hsci_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype=sci["imagetyp"], **sci)
        wskye_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype="sky_e", **sci)
        wskyw_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype="sky_w", **sci)
        hskye_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype="sky_e", **sci)
        hskyw_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype="sky_w", **sci)
        os.makedirs(os.path.dirname(hsci_path), exist_ok=True)
        # define current arc lamps to use for wavelength calibration
        lamps = arc_lamps[sci_camera[0]]
        
        # define calibration frames paths
        if use_fiducial_master:
            masters_path = os.getenv("LVM_MASTER_DIR")
            log.info(f"Using fiducial master calibration frames for {sci_camera} at $LVM_MASTER_DIR = {masters_path}")
            if masters_path is None:
                raise ValueError("LVM_MASTER_DIR environment variable is not defined")
            mpixmask_path = os.path.join(masters_path, f"lvm-mpixmask-{sci_camera}.fits")
            mbias_path = os.path.join(masters_path, f"lvm-mbias-{sci_camera}.fits")
            mdark_path = os.path.join(masters_path, f"lvm-mdark-{sci_camera}.fits")
            mtrace_path = os.path.join(masters_path, f"lvm-mtrace-{sci_camera}.fits")
            mwave_path = os.path.join(masters_path, f"lvm-mwave_{lamps}-{sci_camera}.fits")
            mlsf_path = os.path.join(masters_path, f"lvm-mlsf_{lamps}-{sci_camera}.fits")
            mflat_path = os.path.join(masters_path, f"lvm-mfiberflat-{sci_camera}.fits")
        else:
            log.info(f"Using master calibration frames from DRP version {drpver}, mjd = {sci_mjd}, camera = {sci_camera}")
            masters = md.match_master_metadata(target_mjd=sci_mjd,
                                               target_camera=sci_camera,
                                               target_imagetyp=sci["imagetyp"])
            mpixmask_path = path.full("lvm_master", drpver=drpver, kind="mpixmask", **masters["pixmask"].to_dict())
            mbias_path = path.full("lvm_master", drpver=drpver, kind="mbias", **masters["bias"].to_dict())
            mdark_path = path.full("lvm_master", drpver=drpver, kind="mdark", **masters["dark"].to_dict())
            mtrace_path = path.full("lvm_master", drpver=drpver, kind="mtrace", **masters["trace"].to_dict())
            mwave_path = path.full("lvm_master", drpver=drpver, kind=f"mwave_{lamps}", **masters["wave"].to_dict())
            mlsf_path = path.full("lvm_master", drpver=drpver, kind=f"mlsf_{lamps}", **masters["lsf"].to_dict())
            mflat_path = path.full("lvm_master", drpver=drpver, kind="mfiberflat", **masters["fiberflat"].to_dict())
        
        # preprocess frame
        image_tasks.preproc_raw_frame(in_image=sci_path, out_image=psci_path, in_mask=mpixmask_path)
        
        # detrend frame
        image_tasks.detrend_frame(in_image=psci_path, out_image=dsci_path, in_bias=mbias_path, in_dark=mdark_path, in_slitmap=Table(drp.fibermap.data))
        
        # extract 1d spectra
        image_tasks.extract_spectra(in_image=dsci_path, out_rss=xsci_path, in_trace=mtrace_path, method="aperture", aperture=3)
        
        # wavelength calibrate
        rss_tasks.create_pixel_table(in_rss=xsci_path, out_rss=wsci_path, arc_wave=mwave_path, arc_fwhm=mlsf_path)

        # apply fiberflat correction
        rss_tasks.apply_fiberflat(in_rss=wsci_path, out_rss=wsci_path, in_flat=mflat_path)

        # interpolate sky fibers
        sky_tasks.interpolate_sky(in_rss=wsci_path, out_sky=wskye_path, which="e")
        sky_tasks.interpolate_sky(in_rss=wsci_path, out_sky=wskyw_path, which="w")

        # quick sky subtraction
        sky_tasks.quick_sky_subtraction(in_rss=wsci_path, out_rss=wsci_path, in_skye=wskye_path, in_skyw=wskyw_path)

        # resample wavelength into uniform grid along fiber IDs
        iwave, fwave = SPEC_CHANNELS[sci_camera[0]]
        rss_tasks.resample_wavelength(in_rss=wsci_path, out_rss=hsci_path, method="linear", disp_pix=0.5, start_wave=iwave, end_wave=fwave, err_sim=10, parallel=0, extrapolate=False)
        rss_tasks.resample_wavelength(in_rss=wskye_path, out_rss=hskye_path, method="linear", disp_pix=0.5, start_wave=iwave, end_wave=fwave, err_sim=10, parallel=0, extrapolate=False)
        rss_tasks.resample_wavelength(in_rss=wskyw_path, out_rss=hskyw_path, method="linear", disp_pix=0.5, start_wave=iwave, end_wave=fwave, err_sim=10, parallel=0, extrapolate=False)

    # combine channels
    drp.combine_cameras(sci_tileid, sci_mjd, expnum=sci_expnum, spec=1)
    drp.combine_cameras(sci_tileid, sci_mjd, expnum=sci_expnum, spec=2)
    drp.combine_cameras(sci_tileid, sci_mjd, expnum=sci_expnum, spec=3)

    # combine spectrographs
    drp.combine_spectrographs(sci_tileid, sci_mjd, sci_expnum)
