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
import numpy as np

from lvmdrp import image
from lvmdrp.core.constants import CALIBRATION_NAMES, FIELDS_TO_SKIP, INPUT_PATH, MASTER_CONFIG_PATH, CALIBRATION_TYPES, CONT_FIELDS, ARC_FIELDS, PRODUCT_PATH
from lvmdrp.utils import get_master_name
from lvmdrp.utils.bitmask import QualityFlag
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
    redux_settings.CCD = metadata.CCD
    redux_settings.MJD = metadata.MJD
    redux_settings.WL_RANGE = config.WAVELENGTH_RANGES.__dict__[metadata.CCD[0]]
    redux_settings.PIX2WAVE_MAP = os.path.join(config.LVM_DRP_CONFIG_PATH, config.PIX2WAVE_MAPS.__dict__[metadata.CCD])
    redux_settings.LAMPS = [field.lower() for field in metadata.__dict__ if field.lower() in list(CONT_FIELDS.keys())+list(ARC_FIELDS.keys()) if metadata.__dict__[field]]
    # find type of reduction and calibration frames depending on the target image
    #   - flat, dark, bias: calibration
    if metadata.IMAGETYP in CALIBRATION_TYPES:
        redux_settings.TYPE = metadata.IMAGETYP
        redux_settings.LABEL = get_master_name(label=metadata.LABEL, image_type=metadata.IMAGETYP, mjd=metadata.MJD)
        redux_settings.OUTPUT_PATH = PRODUCT_PATH.format(
            path=config.LVM_SPECTRO_CALIB_PATH,
            label=redux_settings.LABEL,
            kind="{kind}"
        )
    elif metadata.IMAGETYP =="arc" or metadata.IMAGETYP == "object":
    #   - any lamps are on:
        if redux_settings.LAMPS:
            has_cont = np.isin(list(CONT_FIELDS.keys()), redux_settings.LAMPS)
            has_arcs = np.isin(list(ARC_FIELDS.keys()), redux_settings.LAMPS)
    #     * continuum lamp or all lamps: continuum
            if metadata.IMAGETYP == "continuum" or has_cont.any() or has_arcs.all():
                redux_settings.TYPE = "continuum"
                metadata.IMAGETYP = "continuum"
                redux_settings.LABEL = get_master_name(label=metadata.LABEL, image_type=metadata.IMAGETYP, mjd=metadata.MJD)
    #     * some arc lamp are on: arc
            elif metadata.IMAGETYP == "arc" or (not has_cont.any() and has_arcs.any() and not has_arcs.all()):
                redux_settings.TYPE = "arc"
                metadata.IMAGETYP = "arc"
                redux_settings.LABEL = get_master_name(label=metadata.LABEL, image_type=metadata.IMAGETYP, mjd=metadata.MJD)
            else:
                raise ValueError(f"unrecognized case for lamps: '{redux_settings.LAMPS}'.")

            redux_settings.OUTPUT_PATH = PRODUCT_PATH.format(
                path=config.LVM_SPECTRO_CALIB_PATH,
                label=redux_settings.LABEL,
                kind="{kind}"
            )
    #   - object: object
        else:
            redux_settings.TYPE = "object"
            redux_settings.LABEL = metadata.LABEL
            redux_settings.OUTPUT_PATH = PRODUCT_PATH.format(
                calib_path=config.LVM_SPECTRO_REDUX_PATH,
                label=redux_settings.LABEL,
                kind="{kind}"
            )
    else:
        raise ValueError(f"unrecognized image type '{metadata.IMAGETYP}'")

    # locate input frame in file system
    # redux_settings.INPUT_PATH = get_input_path(pattern=INPUT_PATH, mjd=metadata.MJD, label=metadata.LABEL, paths=config.RAW_DATA_PATHS)
    redux_settings.INPUT_PATH = metadata.PATH

    # # decide where to store output frames in file system
    # if redux_settings.TYPE in CALIBRATION_TYPES+["continuum", "arc"]:
    #     redux_settings.OUTPUT_PATH = config.LVM_SPECTRO_CALIB_PATH
    # else:
    #     redux_settings.OUTPUT_PATH = config.LVM_SPECTRO_REDUX_PATH
    # update reduction status
    metadata.STATUS += "IN_PROGRESS"
    return metadata, redux_settings

def build_master(config, analogs_metadata, calib_metadata, frame_settings):
    
    # define master metadata by copying the basic fields from the analog metadata
    # BUG: set master & analogs reduction state before entering 'build_master' & put calibration state for both
    master_metadata = Namespace(**{field.upper(): analogs_metadata[0].__dict__.get(field.upper()) for field in CALIBRATION_NAMES if field not in FIELDS_TO_SKIP})

    # bypass if all analog frames are already part of a master
    if (np.asarray([_.MASTER_ID for _ in analogs_metadata]) != None).all():
        return master_metadata, analogs_metadata
    
    # BUG: subtract bias and darks on individual frames
    # take into account the exposure time
    # in case of exposure time mismatch assume linearity
    # preprocess analog frames
    frame_paths = []
    for analog_frame in analogs_metadata:
        # analog_in_path = get_input_path(pattern=INPUT_PATH, mjd=analog_frame.MJD, label=analog_frame.LABEL, paths=config.RAW_DATA_PATHS)
        analog_out_path = frame_settings.OUTPUT_PATH.format(kind="pre")
        proc_image, flags = imageMethod.preprocRawFrame_drp(
            in_image=analog_frame.PATH,
            channel=frame_settings.CCD,
            out_image=analog_out_path,
            boundary_x="1,2040",
            boundary_y="1,2040",
            positions="00,10,01,11",
            orientation="S,S,S,S",
            gain=config.GAIN, rdnoise=config.READ_NOISE
        )
        analog_frame.NAXIS1 = proc_image._header["NAXIS1"]
        analog_frame.NAXIS2 = proc_image._header["NAXIS2"]
        analog_frame.STATUS += "FINISHED"
        analog_frame.FLAGS += flags
        # only add those frames that were reduced correctly
        if "OK" in analog_frame.FLAGS: frame_paths.append(analog_out_path)
    
    # build masters
    # BUG: quick fix for the case of one analog
    master_out_path = frame_settings.OUTPUT_PATH.format(kind="calib")
    imageMethod.combineImages_drp(
        images=",".join(frame_paths if len(frame_paths) > 1 else 2*frame_paths),
        out_image=master_out_path,
        method="mean"
    )
    # initialize flags
    flags = QualityFlag["OK"]
    # get calibration paths
    calib_path = PRODUCT_PATH.format(
        path=config.LVM_SPECTRO_CALIB_PATH,
        label="{label}",
        kind="calib"
    )
    if frame_settings.TYPE == "bias":
        new_master = image.loadImage(master_out_path)
    elif frame_settings.TYPE == "dark":
        master_frame = image.loadImage(master_out_path)
        if calib_metadata["bias"].LABEL is not None:
            master_bias = image.loadImage(calib_path.format(label=calib_metadata["bias"].LABEL))
        else:
            master_bias = image.Image(data=np.zeros_like(master_frame._data))
            flags += "BAD_CALIBRATION_FRAMES"
        new_master = (master_frame - master_bias._data.mean())
        new_master.writeFitsData(master_out_path)
    elif frame_settings.TYPE == "flat":
        master_frame = image.loadImage(master_out_path)
        if calib_metadata["bias"].LABEL is not None:
            master_bias = image.loadImage(calib_path.format(label=calib_metadata["bias"].LABEL))
        else:
            master_bias = image.Image(data=np.zeros_like(master_frame._data))
            flags += "BAD_CALIBRATION_FRAMES"
        if calib_metadata["dark"].LABEL is not None:
            master_dark = image.loadImage(calib_path.format(label=calib_metadata["dark"].LABEL))
        else:
            master_dark = image.Image(data=np.zeros_like(master_frame._data))
            flags += "BAD_CALIBRATION_FRAMES"
        new_master = (master_frame - master_bias._data.mean() - master_dark._data.mean())
        new_master.writeFitsData(master_out_path)
    else:
        raise ValueError(f"unkown calibration type '{frame_settings.TYPE}'")
    # TODO: test and update database
    #   - test quality of master
    #   - add frame to master frames
    #   - add flags according to test results
    #   - add DB reference for preprocessed frames
    # BUG: update columns inherited from original frames metadata (remove 'path', remove 'obstime')
    # define new master metadata
    master_metadata.LABEL = frame_settings.LABEL
    master_metadata.NAXIS1 = new_master._header["NAXIS1"]
    master_metadata.NAXIS2 = new_master._header["NAXIS2"]
    master_metadata.STATUS += "FINISHED"
    master_metadata.FLAGS += flags

    return master_metadata, analogs_metadata

def run_reduction_calib(config, metadata, calib_metadata, frame_settings):
    
    target_frame, flags = imageMethod.preprocRawFrame_drp(
        in_image=frame_settings.INPUT_PATH,
        channel=frame_settings.CCD,
        out_image=frame_settings.OUTPUT_PATH.format(kind="pre"),
        boundary_x="1,2040",
        boundary_y="1,2040",
        positions="00,10,01,11",
        orientation="S,S,S,S",
        gain=config.GAIN, rdnoise=config.READ_NOISE
    )
    calib_path = PRODUCT_PATH.format(
        path=config.LVM_SPECTRO_CALIB_PATH,
        label="{label}",
        kind="calib"
    )
    if calib_metadata["bias"].LABEL is not None:
        master_bias = image.loadImage(calib_path.format(label=calib_metadata["bias"].LABEL))
    else:
        master_bias = image.Image(data=np.zeros_like(target_frame._data))
        flags += "BAD_CALIBRATION_FRAMES"
    if calib_metadata["dark"].LABEL is not None:
        master_dark = image.loadImage(calib_path.format(label=calib_metadata["dark"].LABEL))
    else:
        master_dark = image.Image(data=np.zeros_like(target_frame._data))
        flags += "BAD_CALIBRATION_FRAMES"
    if calib_metadata["flat"].LABEL is not None:
        master_flat = image.loadImage(calib_path.format(label=calib_metadata["flat"].LABEL))
    else:
        master_flat = image.Image(data=np.ones_like(target_frame._data))
        flags += "BAD_CALIBRATION_FRAMES"

    frame_calib = ((target_frame - master_bias._data.mean() - master_dark._data.mean())/master_flat)
    frame_calib.writeFitsData(frame_settings.OUTPUT_PATH.format(kind="calib"))

    metadata.NAXIS1 = frame_calib._header["NAXIS1"]
    metadata.NAXIS2 = frame_calib._header["NAXIS2"]
    metadata.STATUS += "FINISHED"
    metadata.FLAGS += flags
    return metadata

def run_reduction_block(config, metadata, calib_metadata, frame_settings):
    # build calibration paths
    calib_path = PRODUCT_PATH.format(path=config.LVM_SPECTRO_CALIB_PATH, label=calib_metadata["continuum"].LABEL, kind="{kind}")
    _, flags = imageMethod.subtractStraylight_drp(
        image=frame_settings.OUTPUT_PATH.format(kind="cosmic"),
        trace=calib_path.format(kind="trc"),
        stray_image=frame_settings.OUTPUT_PATH.format(kind="back"),
        clean_image=frame_settings.OUTPUT_PATH.format(kind="stray"),
        aperture=40, poly_cross=2, smooth_gauss=30
    )
    metadata.FLAGS += flags
    _, flags = imageMethod.extractSpec_drp(
        image=frame_settings.OUTPUT_PATH.format(kind="stray"),
        trace=calib_path.format(kind="trc"),
        out_rss=frame_settings.OUTPUT_PATH.format(kind="ms"),
        fwhm=calib_path.format(kind="fwhm"),
        method="optimal", parallel="5"
    )
    metadata.FLAGS += flags
    return metadata

def run_reduction_continuum(config, metadata, frame_settings):

    # BUG: add continuum frames to CALIBRATION_FRAMES in DB
    _, flags = imageMethod.LACosmic_drp(
        image=frame_settings.OUTPUT_PATH.format(kind="calib"),
        out_image=frame_settings.OUTPUT_PATH.format(kind="cosmic"),
        increase_radius=1, flim="1.3", parallel='5'
    )
    metadata.FLAGS += flags
    # BUG: verify outputs against expected values, skip calibration steps if needed & set corresponding flags
    _, flags = imageMethod.findPeaksAuto_drp(
        image=frame_settings.OUTPUT_PATH.format(kind="cosmic"),
        out_peaks_file=frame_settings.OUTPUT_PATH.format(kind="trace").replace(".fits", ".peaks"),
        disp_axis="X", threshold="5000", slice="3696", nfibers="41", median_box="1", median_cross="1", method="gauss", init_sigma="0.5", verbose=0
    )
    metadata.FLAGS += flags
    _, flags = imageMethod.tracePeaks_drp(
        image=frame_settings.OUTPUT_PATH.format(kind="cosmic"),
        peaks_file=frame_settings.OUTPUT_PATH.format(kind="trace").replace(".fits", ".peaks"),
        trace_out=frame_settings.OUTPUT_PATH.format(kind="trc"),
        steps=30, method="gauss", threshold_peak=50, poly_disp=5, coadd=30, verbose=0
    )
    metadata.FLAGS += flags
    _, flags = imageMethod.subtractStraylight_drp(
        image=frame_settings.OUTPUT_PATH.format(kind="cosmic"),
        trace=frame_settings.OUTPUT_PATH.format(kind="trc"),
        stray_image=frame_settings.OUTPUT_PATH.format(kind="back"),
        clean_image=frame_settings.OUTPUT_PATH.format(kind="stray"),
        aperture=40, poly_cross=2, smooth_gauss=30
    )
    metadata.FLAGS += flags
    _, flags = imageMethod.traceFWHM_drp(
        image=frame_settings.OUTPUT_PATH.format(kind="stray"),
        trace=frame_settings.OUTPUT_PATH.format(kind="trc"),
        fwhm_out=frame_settings.OUTPUT_PATH.format(kind="fwhm"),
        blocks=32, steps=30, coadd=20, threshold_flux=50.0, poly_disp=5, clip="1.5,4.0"
    )
    metadata.FLAGS += flags
    _, flags = imageMethod.extractSpec_drp(
        image=frame_settings.OUTPUT_PATH.format(kind="stray"),
        trace=frame_settings.OUTPUT_PATH.format(kind="trc"),
        out_rss=frame_settings.OUTPUT_PATH.format(kind="ms"),
        fwhm=frame_settings.OUTPUT_PATH.format(kind="fwhm"),
        parallel=5, method="optimal"
    )
    metadata.STATUS += "FINISHED"
    metadata.FLAGS += flags
    return metadata

def run_reduction_arc(config, metadata, calib_metadata, frame_settings):
    metadata = run_reduction_block(config, metadata=metadata, calib_metadata=calib_metadata, frame_settings=frame_settings)
    _, flags = rssMethod.detWaveSolution_drp(
        arc_rss=frame_settings.OUTPUT_PATH.format(kind="ms"),
        disp_rss=frame_settings.OUTPUT_PATH.format(kind="disp"),
        res_rss=frame_settings.OUTPUT_PATH.format(kind="res"),
        ref_line_file=frame_settings.PIX2WAVE_MAP,
        aperture="7", poly_fwhm="-1,-1", poly_dispersion="-4", rel_flux_limits="0.2,2", flux_min="100.0", verbose="-1"
    )
    metadata.FLAGS += flags
    _, flags = rssMethod.createPixTable_drp(
        rss_in=frame_settings.OUTPUT_PATH.format(kind="ms"),
        rss_out=frame_settings.OUTPUT_PATH.format(kind="rss"),
        arc_wave=frame_settings.OUTPUT_PATH.format(kind="disp"),
        arc_fwhm=frame_settings.OUTPUT_PATH.format(kind="res"),
        cropping=''
    )
    metadata.FLAGS += flags
    _, flags = rssMethod.resampleWave_drp(
        rss_in=frame_settings.OUTPUT_PATH.format(kind="rss"),
        rss_out=frame_settings.OUTPUT_PATH.format(kind="disp_cor"),
        start_wave=frame_settings.WL_RANGE[0], end_wave=frame_settings.WL_RANGE[1], disp_pix="1.0", err_sim="0"
    )
    metadata.FLAGS += flags
    metadata.STATUS += "FINISHED"
    return metadata

def run_reduction_object(config, metadata, calib_metadata, frame_settings):

    _, flags = imageMethod.LACosmic_drp(
        image=frame_settings.OUTPUT_PATH.format(kind="calib"),
        out_image=frame_settings.OUTPUT_PATH.format(kind="cosmic"),
        increase_radius=1, flim="1.3", parallel='5'
    )
    metadata.FLAGS += flags
    metadata = run_reduction_block(config, metadata=metadata, calib_metadata=calib_metadata, frame_settings=frame_settings)
    _, flags = rssMethod.createPixTable_drp(
        rss_in=frame_settings.OUTPUT_PATH.format(kind="ms"),
        rss_out=frame_settings.OUTPUT_PATH.format(kind="rss"),
        arc_wave=frame_settings.OUTPUT_PATH.format(kind="disp"), arc_fwhm=frame_settings.OUTPUT_PATH.format(kind="res"), cropping=''
    )
    metadata.FLAGS += flags
    _, flags = rssMethod.resampleWave_drp(
        rss_in=frame_settings.OUTPUT_PATH.format(kind="rss"),
        rss_out=frame_settings.OUTPUT_PATH.format(kind="disp_cor"),
        start_wave=frame_settings.WL_RANGE[0], end_wave=frame_settings.WL_RANGE[1], disp_pix="1.0", err_sim="0"
    )
    metadata.FLAGS += flags
    metadata.STATUS += "FINISHED"
    return metadata