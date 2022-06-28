# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: main.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import sys
import yaml
import argparse
from argparse import Namespace
import datetime as dt
import numpy as np

from lvmdrp import image
from lvmdrp.core.constants import MASTER_CONFIG_PATH, FRAMES_PRIORITY, CALIBRATION_TYPES, PRODUCT_PATH
from lvmdrp.utils import get_master_name
from lvmdrp.utils.database import LAMP_NAMES, CalibrationFrames
from lvmdrp.utils.decorators import validate_fibers
from lvmdrp.utils.bitmask import QualityFlag, ReductionStatus
from lvmdrp.utils.namespace import Loader
from lvmdrp.functions import imageMethod, rssMethod


def load_master_config(master_config_path=MASTER_CONFIG_PATH):
    # if no path is given, load from hard-coded path
    with open(master_config_path, "r") as config_file:
        master_config = yaml.load(config_file, Loader=Loader)
    return master_config

def parse_arguments(config, args=None):
    if args is None:
        args = sys.argv[1:]
    # define cmdline arguments parser
    parser = argparse.ArgumentParser(
        prog="LVM data reduction pipeline",
        description="This pipeline takes raw frames and runs the reduction process to produce science-ready frames"
    )
    # config parser
    # parse
    cmd_args = parser.parse_args(args)
    # replace config parameters with cmdline arguments
    return config, cmd_args

def setup_reduction(config, metadata):
    # create a mapping of the target frame and the metadata/calibration frames needed to run the calibration
    redux_settings = Namespace()
    # TODO: add output file names to this mapping
    # copy basic metadata to this mapping from config
    redux_settings.ccd = metadata.ccd
    redux_settings.spec = metadata.spec
    redux_settings.mjd = metadata.mjd
    redux_settings.wl_range = config.WAVELENGTH_RANGES.__dict__[metadata.ccd[0]]
    redux_settings.pix2wave_map = os.path.join(config.LVM_DRP_CONFIG_PATH, config.PIX2WAVE_MAPS.__dict__[metadata.ccd])
    redux_settings.lamps = [name for name in metadata.__data__ if name in LAMP_NAMES and metadata.__data__[name]]
    # find type of reduction and calibration frames depending on the target image
    if metadata.imagetyp in FRAMES_PRIORITY:
        redux_settings.type = metadata.imagetyp
    else:
        raise ValueError(f"unrecognized image type '{metadata.imagetyp}'")

    # locate input frame in file system
    redux_settings.input_path = metadata.path
    # define output paths
    if redux_settings.type in CALIBRATION_TYPES:
        redux_settings.label = get_master_name(metadata.label, redux_settings.type, redux_settings.mjd)
        redux_settings.output_path = PRODUCT_PATH.format(
            path=config.LVM_SPECTRO_CALIB_PATH,
            label="{label}",
            kind="{kind}"
        )
        redux_settings.master_path = PRODUCT_PATH.format(
            path=config.LVM_SPECTRO_CALIB_PATH,
            label=redux_settings.label,
            kind="{kind}"
        )
    else:
        redux_settings.label = metadata.label
        redux_settings.output_path = PRODUCT_PATH.format(
            path=config.LVM_SPECTRO_REDUX_PATH,
            label=redux_settings.label,
            kind="{kind}"
        )
    # update reduction status
    metadata.status += "IN_PROGRESS"
    return metadata, redux_settings

# BUG: this function is doing too much! It should:
#      * look for calibrated analogs
#      * build a calibrated master
#      * it should be its own script
#      the DRP should handle the reduction of masters continuum and arc
def build_master(config, analogs_metadata, calibs_metadata, redux_settings):
    # bypass if all analog frames are already part of a master
    if (np.asarray([_.master_id for _ in analogs_metadata]) != None).all():
        return None, analogs_metadata
    
    calib_images = []
    for analog_metadata, calib_metadata in zip(analogs_metadata, calibs_metadata):
        # BUG: best way to calculate gain for each amplifier: series of flats and fit the slope for sigma_counts vs sqrt(mean_counts)
        # decorate preprocessing if necessary
        if redux_settings.type in ["continuum", "arc", "object"]:
            preproc = validate_fibers(["BAD_FIBERS"], config, "out_image")(imageMethod.preprocRawFrame_drp)
        else:
            preproc = imageMethod.preprocRawFrame_drp
        
        # preprocess analog frames
        proc_image, flags = preproc(
            in_image=analog_metadata.path,
            channel=redux_settings.ccd,
            out_image=redux_settings.output_path.format(label=analog_metadata.label, kind="pre"),
            boundary_x="1,2040",
            boundary_y="1,2040",
            positions="00,10,01,11",
            orientation="S,S,S,S",
            gain=config.GAIN, rdnoise=config.READ_NOISE, subtract_overscan=0
        )
        # update analogs metadata
        analog_metadata.naxis1 = proc_image._header["NAXIS1"]
        analog_metadata.naxis2 = proc_image._header["NAXIS2"]
        analog_metadata.flags += flags
        # only calibrate those frames that were reduced correctly
        if analog_metadata.flags != "OK":
            analog_metadata.status += "FAILED"
            continue
        else:
        analog_metadata.status += "FINISHED"
    
    master_bias = image.Image(data=np.zeros_like(proc_image._data))
    master_dark = image.Image(data=np.zeros_like(proc_image._data))
    master_flat = image.Image(data=np.ones_like(proc_image._data))
    # read master bias
    if "bias" in calib_metadata and calib_metadata["bias"]:
        master_bias = image.loadImage(calib_metadata["bias"].path)
    elif "bias" in calib_metadata:
        flags += "BAD_CALIBRATION_FRAMES"
    # read master dark
    if "dark" in calib_metadata and calib_metadata["dark"]:
        master_dark = image.loadImage(calib_metadata["dark"].path)
        master_dark._data *= analogs_metadata[0].exptime / calib_metadata["dark"].exptime
    elif "dark" in calib_metadata:
        flags += "BAD_CALIBRATION_FRAMES"
    # read master flat
    if "flat" in calib_metadata and calib_metadata["flat"]:
        master_flat = image.loadImage(calib_metadata["flat"].path)
        master_flat._data *= analogs_metadata[0].exptime / calib_metadata["flat"].exptime
    elif "flat" in calib_metadata:
        flags += "BAD_CALIBRATION_FRAMES"

    # normalize in case of flat calibration
    if redux_settings.type == "flat":
            proc_image = proc_image / np.median(proc_image._data)

    # run basic calibration for each analog
        calib_image = (proc_image - master_dark - master_bias) / master_flat
        calib_image.writeFitsData(redux_settings.output_path.format(label=analog_metadata.label, kind="calib"))
        calib_images.append(calib_image)
    
    # TODO: test and update database
    #   - test quality of master
    #   - add frame to master frames
    #   - add flags according to test results
    #   - add DB reference for preprocessed frames
    # BUG: set master & analogs reduction state before entering 'build_master' & put calibration state for both
    # save calibrated analogs & build master
    if len(calib_images) > 1:
        new_master = image.combineImages(calib_images, method="median")
        new_master.writeFitsData(redux_settings.master_path.format(kind="calib"))
        status = ReductionStatus["FINISHED"]
    else:
        flags += "POORLY_DEFINED_MASTER"
        status = ReductionStatus["FAILED"]
    # define new master metadata
    master_metadata = CalibrationFrames(
        mjd=analog_metadata.mjd,
        spec=analog_metadata.spec,
        ccd=analog_metadata.ccd,
        exptime=analog_metadata.exptime,
        imagetyp=analog_metadata.imagetyp,
        obstime=dt.datetime.now(),
        observat=analog_metadata.observat,
        naxis1=analog_metadata.naxis1,
        naxis2=analog_metadata.naxis2,
        label=redux_settings.label,
        path=redux_settings.master_path.format(kind="calib"),
        reduction_started=analog_metadata.reduction_started,
        reduction_finished=analog_metadata.reduction_finished,
        status=status,
        flags=flags
    )
    return master_metadata, analogs_metadata

def run_reduction_calib(config, metadata, calib_metadata, redux_settings):
    # decorate preprocessing if necessary
    if redux_settings.type in ["continuum", "arc", "object"]:
        preproc = validate_fibers(["BAD_FIBERS"], config, "out_image")(imageMethod.preprocRawFrame_drp)
    else:
        preproc = imageMethod.preprocRawFrame_drp
    
    proc_image, flags = preproc(
        in_image=redux_settings.input_path,
        channel=redux_settings.ccd,
        out_image=redux_settings.output_path.format(kind="pre"),
        boundary_x="1,2040",
        boundary_y="1,2040",
        positions="00,10,01,11",
        orientation="S,S,S,S",
        gain=config.GAIN, rdnoise=config.READ_NOISE
    )
    master_bias = image.Image(data=np.zeros_like(proc_image._data))
    master_dark = image.Image(data=np.zeros_like(proc_image._data))
    master_flat = image.Image(data=np.ones_like(proc_image._data))
    # read master bias
    if "bias" in calib_metadata and calib_metadata["bias"]:
        master_bias = image.loadImage(calib_metadata["bias"].path)
    elif "bias" in calib_metadata:
        flags += "BAD_CALIBRATION_FRAMES"
    # read master dark
    if "dark" in calib_metadata and calib_metadata["dark"]:
        master_dark = image.loadImage(calib_metadata["dark"].path)
        master_dark._data *= metadata.exptime / calib_metadata["dark"].exptime
    elif "dark" in calib_metadata:
        flags += "BAD_CALIBRATION_FRAMES"
    # read master flat
    if "flat" in calib_metadata and calib_metadata["flat"]:
        master_flat = image.loadImage(calib_metadata["flat"].path)
        master_flat._data *= metadata.exptime / calib_metadata["flat"].exptime
    elif "flat" in calib_metadata:
        flags += "BAD_CALIBRATION_FRAMES"

    # normalize in case of flat calibration
    if redux_settings.type == "flat":
        proc_image = proc_image / np.median(proc_image._data)

    # run basic calibration for each analog
    calib_image = (proc_image - master_dark - master_bias) / master_flat
    calib_image.writeFitsData(redux_settings.output_path.format(label=metadata.label, kind="calib"))

    metadata.naxis1 = calib_image._header["NAXIS1"]
    metadata.naxis2 = calib_image._header["NAXIS2"]
    metadata.flags += flags
    return metadata

def run_reduction_block(config, metadata, calib_metadata, redux_settings):
    # build calibration paths
    master_continuum_path = redux_settings.output_path.format(
        label=calib_metadata["continuum"].label,
        kind="{kind}"
    )
    target_frame_path = redux_settings.output_path.format(
        label=metadata.label,
        kind="{kind}"
    )
    _, flags = imageMethod.subtractStraylight_drp(
        image=target_frame_path.format(kind="cosmic"),
        trace=master_continuum_path.format(kind="trc"),
        stray_image=target_frame_path.format(kind="back"),
        clean_image=target_frame_path.format(kind="stray"),
        aperture=40, poly_cross=2, smooth_gauss=30
    )
    metadata.flags += flags
    _, flags = imageMethod.extractSpec_drp(
        image=target_frame_path.format(kind="stray"),
        trace=master_continuum_path.format(kind="trc"),
        out_rss=target_frame_path.format(kind="ms"),
        fwhm=master_continuum_path.format(kind="fwhm"),
        method="optimal", parallel="5"
    )
    metadata.flags += flags
    return metadata

def run_reduction_continuum(config, metadata, calib_metadata, redux_settings):

    # BUG: add continuum frames to CALIBRATION_FRAMES in DB
    target_frame_path = redux_settings.output_path.format(
        label=metadata.label,
        kind="{kind}"
    )
    _, flags = imageMethod.LACosmic_drp(
        image=target_frame_path.format(kind="calib"),
        out_image=target_frame_path.format(kind="cosmic"),
        increase_radius=1, flim="1.3", parallel='5'
    )
    metadata.flags += flags
    # BUG: verify outputs against expected values, skip calibration steps if needed & set corresponding flags
    _, flags = imageMethod.findPeaksAuto_drp(
        image=target_frame_path.format(kind="cosmic"),
        out_peaks_file=target_frame_path.format(kind="trace").replace(".fits", ".peaks"),
        disp_axis="X", threshold="5000", slice="3696", nfibers="41", median_box="1", median_cross="1", method="gauss", init_sigma="0.5", verbose=0
    )
    metadata.flags += flags
    _, flags = imageMethod.tracePeaks_drp(
        image=target_frame_path.format(kind="cosmic"),
        peaks_file=target_frame_path.format(kind="trace").replace(".fits", ".peaks"),
        trace_out=target_frame_path.format(kind="trc"),
        steps=30, method="gauss", threshold_peak=50, poly_disp=5, coadd=30, verbose=0
    )
    metadata.flags += flags
    _, flags = imageMethod.subtractStraylight_drp(
        image=target_frame_path.format(kind="cosmic"),
        trace=target_frame_path.format(kind="trc"),
        stray_image=target_frame_path.format(kind="back"),
        clean_image=target_frame_path.format(kind="stray"),
        aperture=40, poly_cross=2, smooth_gauss=30
    )
    metadata.flags += flags
    _, flags = imageMethod.traceFWHM_drp(
        image=target_frame_path.format(kind="stray"),
        trace=target_frame_path.format(kind="trc"),
        fwhm_out=target_frame_path.format(kind="fwhm"),
        blocks=32, steps=30, coadd=20, threshold_flux=50.0, poly_disp=5, clip="1.5,4.0"
    )
    metadata.flags += flags
    _, flags = imageMethod.extractSpec_drp(
        image=target_frame_path.format(kind="stray"),
        trace=target_frame_path.format(kind="trc"),
        out_rss=target_frame_path.format(kind="ms"),
        fwhm=target_frame_path.format(kind="fwhm"),
        parallel=5, method="optimal"
    )
    metadata.status += "FINISHED"
    metadata.flags += flags
    return metadata

def run_reduction_arc(config, metadata, calib_metadata, redux_settings):
    metadata = run_reduction_block(config, metadata=metadata, calib_metadata=calib_metadata, redux_settings=redux_settings)
    target_frame_path = redux_settings.output_path.format(
        label=metadata.label,
        kind="{kind}"
    )
    _, flags = rssMethod.detWaveSolution_drp(
        arc_rss=target_frame_path.format(kind="ms"),
        disp_rss=target_frame_path.format(kind="disp"),
        res_rss=target_frame_path.format(kind="res"),
        ref_line_file=redux_settings.pix2wave_map,
        aperture="7", poly_fwhm="-1,-1", poly_dispersion="-4", rel_flux_limits="0.2,2", flux_min="100.0", verbose="-1"
    )
    metadata.flags += flags
    _, flags = rssMethod.createPixTable_drp(
        rss_in=target_frame_path.format(kind="ms"),
        rss_out=target_frame_path.format(kind="rss"),
        arc_wave=target_frame_path.format(kind="disp"),
        arc_fwhm=target_frame_path.format(kind="res"),
        cropping=''
    )
    metadata.flags += flags
    _, flags = rssMethod.resampleWave_drp(
        rss_in=target_frame_path.format(kind="rss"),
        rss_out=target_frame_path.format(kind="disp_cor"),
        start_wave=redux_settings.wl_range[0], end_wave=redux_settings.wl_range[1], disp_pix="1.0", err_sim="0"
    )
    metadata.flags += flags
    metadata.status += "FINISHED"
    return metadata

def run_reduction_object(config, metadata, calib_metadata, redux_settings):

    _, flags = imageMethod.LACosmic_drp(
        image=redux_settings.output_path.format(kind="calib"),
        out_image=redux_settings.output_path.format(kind="cosmic"),
        increase_radius=1, flim="1.3", parallel='5'
    )
    metadata.flags += flags
    metadata = run_reduction_block(config, metadata=metadata, calib_metadata=calib_metadata, redux_settings=redux_settings)
    _, flags = rssMethod.createPixTable_drp(
        rss_in=redux_settings.output_path.format(kind="ms"),
        rss_out=redux_settings.output_path.format(kind="rss"),
        arc_wave=redux_settings.output_path.format(kind="disp"), arc_fwhm=redux_settings.output_path.format(kind="res"), cropping=''
    )
    metadata.flags += flags
    _, flags = rssMethod.resampleWave_drp(
        rss_in=redux_settings.output_path.format(kind="rss"),
        rss_out=redux_settings.output_path.format(kind="disp_cor"),
        start_wave=redux_settings.wl_range[0], end_wave=redux_settings.wl_range[1], disp_pix="1.0", err_sim="0"
    )
    metadata.flags += flags
    metadata.status += "FINISHED"
    return metadata