#!/usr/bin/env python
# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Aug 9, 2023
# @Filename: quickdrp
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import click
import cloup

from astropy.table import Table

from lvmdrp import path, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.functions import run_drp as drp
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.core.constants import SPEC_CHANNELS


@cloup.command(short_help='Run the Quick DRP', show_constraints=True)
@click.option('-e', '--expnum', type=int, help='an exposure number to reduce')
def quick_reduction(expnum: int) -> None:
    """ Run the Quick DRP for a given exposure number.
    """
    # get target frames metadata
    sci_metadata = md.get_metadata(tileid="*", mjd="*", expnum=expnum)
    sci_metadata.sort_values("camera", inplace=True)

    # define general metadata
    sci_tileid = sci_metadata["tileid"].unique()[0]
    sci_mjd = sci_metadata["mjd"].unique()[0]
    sci_expnum = sci_metadata["expnum"].unique()[0]

    # define arc lamps configuration per spectrograph channel
    arc_lamps = {"b": "hgne", "r": "neon", "z": "neon"}

    # run reduction loop for each science camera exposure
    for sci in sci_metadata.to_dict("records"):
        
        # define sci paths
        sci_path = path.full("lvm_raw", camspec=sci["camera"], **sci)
        psci_path = path.full("lvm_anc", drpver=drpver, kind="p", imagetype=sci["imagetyp"], **sci)
        dsci_path = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=sci["imagetyp"], **sci)
        xsci_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=sci["imagetyp"], **sci)
        wsci_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype=sci["imagetyp"], **sci)
        hsci_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype=sci["imagetyp"], **sci)
        # define current arc lamps to use for wavelength calibration
        lamps = arc_lamps[sci["camera"][0]]
        
        # define calibration frames paths
        masters = md.match_master_metadata(target_mjd=sci["mjd"],
                                           target_camera=sci["camera"],
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
        
        # wavelength calibrate & resample
        iwave, fwave = SPEC_CHANNELS[sci["camera"][0]]
        rss_tasks.create_pixel_table(in_rss=xsci_path, out_rss=wsci_path, arc_wave=mwave_path, arc_fwhm=mlsf_path)
        rss_tasks.resample_wavelength(in_rss=wsci_path, out_rss=hsci_path, method="linear", disp_pix=0.5, start_wave=iwave, end_wave=fwave, err_sim=10, parallel=0)
        
        # apply fiberflat correction
        rss_tasks.apply_fiberflat(in_rss=hsci_path, out_rss=hsci_path, in_flat=mflat_path)

    # combine channels
    drp.combine_cameras(sci_tileid, sci_mjd, spec=1)
    drp.combine_cameras(sci_tileid, sci_mjd, spec=2)
    drp.combine_cameras(sci_tileid, sci_mjd, spec=3)

    # combine spectrographs
    drp.combine_spectrographs(sci_tileid, sci_mjd, sci_expnum)
