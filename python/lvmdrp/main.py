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
from lvmdrp.utils.database import CalibrationFrames
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
    redux_settings.ccd = metadata.ccd
    redux_settings.mjd = metadata.mjd
    redux_settings.wl_range = config.WAVELENGTH_RANGES.__dict__[metadata.ccd[0]]
    redux_settings.pix2wave_map = os.path.join(config.LVM_DRP_CONFIG_PATH, config.PIX2WAVE_MAPS.__dict__[metadata.ccd])
    redux_settings.lamps = [field.lower() for field in metadata.__dict__ if field.lower() in list(CONT_FIELDS.keys())+list(ARC_FIELDS.keys()) if metadata.__dict__[field]]
    # find type of reduction and calibration frames depending on the target image
    #   - flat, dark, bias: calibration
    if metadata.imagetyp in CALIBRATION_TYPES:
        redux_settings.type = metadata.imagetyp
        redux_settings.label = get_master_name(label=metadata.label, image_type=metadata.imagetyp, mjd=metadata.mjd)
        redux_settings.output_path = PRODUCT_PATH.format(
            path=config.LVM_SPECTRO_CALIB_PATH,
            label=redux_settings.label,
            kind="{kind}"
        )
    elif metadata.imagetyp =="arc" or metadata.imagetyp == "object":
    #   - any lamps are on:
        if redux_settings.lamps:
            has_cont = np.isin(list(CONT_FIELDS.keys()), redux_settings.lamps)
            has_arcs = np.isin(list(ARC_FIELDS.keys()), redux_settings.lamps)
    #     * continuum lamp or all lamps: continuum
            if metadata.imagetyp == "continuum" or has_cont.any() or has_arcs.all():
                redux_settings.type = "continuum"
                metadata.imagetyp = "continuum"
                redux_settings.label = get_master_name(label=metadata.label, image_type=metadata.imagetyp, mjd=metadata.mjd)
    #     * some arc lamp are on: arc
            elif metadata.imagetyp == "arc" or (not has_cont.any() and has_arcs.any() and not has_arcs.all()):
                redux_settings.type = "arc"
                metadata.imagetyp = "arc"
                redux_settings.label = get_master_name(label=metadata.label, image_type=metadata.imagetyp, mjd=metadata.mjd)
            else:
                raise ValueError(f"unrecognized case for lamps: '{redux_settings.lamps}'.")

            redux_settings.output_path = PRODUCT_PATH.format(
                path=config.LVM_SPECTRO_CALIB_PATH,
                label=redux_settings.label,
                kind="{kind}"
            )
    #   - object: object
        else:
            redux_settings.type = "object"
            redux_settings.label = metadata.label
            redux_settings.output_path = PRODUCT_PATH.format(
                calib_path=config.LVM_SPECTRO_REDUX_PATH,
                label=redux_settings.label,
                kind="{kind}"
            )
    else:
        raise ValueError(f"unrecognized image type '{metadata.imagetyp}'")

    # locate input frame in file system
    # redux_settings.input_path = get_input_path(pattern=INPUT_PATH, mjd=metadata.mjd, label=metadata.label, paths=config.RAW_DATA_PATHS)
    redux_settings.input_path = metadata.path

    # # decide where to store output frames in file system
    # if redux_settings.type in CALIBRATION_TYPES+["continuum", "arc"]:
    #     redux_settings.output_path = config.LVM_SPECTRO_CALIB_PATH
    # else:
    #     redux_settings.output_path = config.LVM_SPECTRO_REDUX_PATH
    # update reduction status
    metadata.status += "IN_PROGRESS"
    return metadata, redux_settings

def build_master(config, analogs_metadata, calib_metadata, frame_settings):
    
    # define master metadata by copying the basic fields from the analog metadata
    # BUG: set master & analogs reduction state before entering 'build_master' & put calibration state for both
    master_metadata = CalibrationFrames(**analogs_metadata[0].__dict__)
    # bypass if all analog frames are already part of a master
    if (np.asarray([_.master_id for _ in analogs_metadata]) != None).all():
        return master_metadata, analogs_metadata
    
    # BUG: subtract bias and darks on individual frames
    # take into account the exposure time
    # in case of exposure time mismatch assume linearity
    # preprocess analog frames
    frame_paths = []
    for analog_frame in analogs_metadata:
        # analog_in_path = get_input_path(pattern=INPUT_PATH, mjd=analog_frame.mjd, label=analog_frame.label, paths=config.RAW_DATA_PATHS)
        analog_out_path = frame_settings.output_path.format(kind="pre")
        proc_image, flags = imageMethod.preprocRawFrame_drp(
            in_image=analog_frame.path,
            channel=frame_settings.ccd,
            out_image=analog_out_path,
            boundary_x="1,2040",
            boundary_y="1,2040",
            positions="00,10,01,11",
            orientation="S,S,S,S",
            gain=config.GAIN, rdnoise=config.READ_NOISE
        )
        analog_frame.naxis1 = proc_image._header["NAXIS1"]
        analog_frame.naxis2 = proc_image._header["NAXIS2"]
        analog_frame.status += "FINISHED"
        analog_frame.flags += flags
        # only add those frames that were reduced correctly
        if analog_frame.flags == "OK": frame_paths.append(analog_out_path)
    
    # build masters
    # BUG: quick fix for the case of one analog
    master_out_path = frame_settings.output_path.format(kind="calib")
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
    if frame_settings.type == "bias":
        new_master = image.loadImage(master_out_path)
    elif frame_settings.type == "dark":
        master_frame = image.loadImage(master_out_path)
        if calib_metadata["bias"].label is not None:
            master_bias = image.loadImage(calib_path.format(label=calib_metadata["bias"].label))
        else:
            master_bias = image.Image(data=np.zeros_like(master_frame._data))
            flags += "BAD_CALIBRATION_FRAMES"
        new_master = (master_frame - master_bias._data.mean())
        new_master.writeFitsData(master_out_path)
    elif frame_settings.type == "flat":
        master_frame = image.loadImage(master_out_path)
        if calib_metadata["bias"].label is not None:
            master_bias = image.loadImage(calib_path.format(label=calib_metadata["bias"].label))
        else:
            master_bias = image.Image(data=np.zeros_like(master_frame._data))
            flags += "BAD_CALIBRATION_FRAMES"
        if calib_metadata["dark"].label is not None:
            master_dark = image.loadImage(calib_path.format(label=calib_metadata["dark"].label))
        else:
            master_dark = image.Image(data=np.zeros_like(master_frame._data))
            flags += "BAD_CALIBRATION_FRAMES"
        new_master = (master_frame - master_bias._data.mean() - master_dark._data.mean())
        new_master.writeFitsData(master_out_path)
    else:
        raise ValueError(f"unkown calibration type '{frame_settings.type}'")
    # TODO: test and update database
    #   - test quality of master
    #   - add frame to master frames
    #   - add flags according to test results
    #   - add DB reference for preprocessed frames
    # BUG: update columns inherited from original frames metadata (remove 'path', remove 'obstime')
    # define new master metadata
    master_metadata.label = frame_settings.label
    master_metadata.naxis1 = new_master._header["NAXIS1"]
    master_metadata.naxis2 = new_master._header["NAXIS2"]
    master_metadata.status += "FINISHED"
    master_metadata.flags += flags

    return master_metadata, analogs_metadata

def run_reduction_calib(config, metadata, calib_metadata, frame_settings):
    
    target_frame, flags = imageMethod.preprocRawFrame_drp(
        in_image=frame_settings.input_path,
        channel=frame_settings.ccd,
        out_image=frame_settings.output_path.format(kind="pre"),
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
    frame_calib.writeFitsData(frame_settings.output_path.format(kind="calib"))

    metadata.naxis1 = frame_calib._header["NAXIS1"]
    metadata.naxis2 = frame_calib._header["NAXIS2"]
    metadata.status += "FINISHED"
    metadata.flags += flags
    return metadata

def run_reduction_block(config, metadata, calib_metadata, frame_settings):
    # build calibration paths
    calib_path = PRODUCT_PATH.format(path=config.LVM_SPECTRO_CALIB_PATH, label=calib_metadata["continuum"].LABEL, kind="{kind}")
    _, flags = imageMethod.subtractStraylight_drp(
        image=frame_settings.output_path.format(kind="cosmic"),
        trace=calib_path.format(kind="trc"),
        stray_image=frame_settings.output_path.format(kind="back"),
        clean_image=frame_settings.output_path.format(kind="stray"),
        aperture=40, poly_cross=2, smooth_gauss=30
    )
    metadata.flags += flags
    _, flags = imageMethod.extractSpec_drp(
        image=frame_settings.output_path.format(kind="stray"),
        trace=calib_path.format(kind="trc"),
        out_rss=frame_settings.output_path.format(kind="ms"),
        fwhm=calib_path.format(kind="fwhm"),
        method="optimal", parallel="5"
    )
    metadata.flags += flags
    return metadata

def run_reduction_continuum(config, metadata, frame_settings):

    # BUG: add continuum frames to CALIBRATION_FRAMES in DB
    _, flags = imageMethod.LACosmic_drp(
        image=frame_settings.output_path.format(kind="calib"),
        out_image=frame_settings.output_path.format(kind="cosmic"),
        increase_radius=1, flim="1.3", parallel='5'
    )
    metadata.flags += flags
    # BUG: verify outputs against expected values, skip calibration steps if needed & set corresponding flags
    _, flags = imageMethod.findPeaksAuto_drp(
        image=frame_settings.output_path.format(kind="cosmic"),
        out_peaks_file=frame_settings.output_path.format(kind="trace").replace(".fits", ".peaks"),
        disp_axis="X", threshold="5000", slice="3696", nfibers="41", median_box="1", median_cross="1", method="gauss", init_sigma="0.5", verbose=0
    )
    metadata.flags += flags
    _, flags = imageMethod.tracePeaks_drp(
        image=frame_settings.output_path.format(kind="cosmic"),
        peaks_file=frame_settings.output_path.format(kind="trace").replace(".fits", ".peaks"),
        trace_out=frame_settings.output_path.format(kind="trc"),
        steps=30, method="gauss", threshold_peak=50, poly_disp=5, coadd=30, verbose=0
    )
    metadata.flags += flags
    _, flags = imageMethod.subtractStraylight_drp(
        image=frame_settings.output_path.format(kind="cosmic"),
        trace=frame_settings.output_path.format(kind="trc"),
        stray_image=frame_settings.output_path.format(kind="back"),
        clean_image=frame_settings.output_path.format(kind="stray"),
        aperture=40, poly_cross=2, smooth_gauss=30
    )
    metadata.flags += flags
    _, flags = imageMethod.traceFWHM_drp(
        image=frame_settings.output_path.format(kind="stray"),
        trace=frame_settings.output_path.format(kind="trc"),
        fwhm_out=frame_settings.output_path.format(kind="fwhm"),
        blocks=32, steps=30, coadd=20, threshold_flux=50.0, poly_disp=5, clip="1.5,4.0"
    )
    metadata.flags += flags
    _, flags = imageMethod.extractSpec_drp(
        image=frame_settings.output_path.format(kind="stray"),
        trace=frame_settings.output_path.format(kind="trc"),
        out_rss=frame_settings.output_path.format(kind="ms"),
        fwhm=frame_settings.output_path.format(kind="fwhm"),
        parallel=5, method="optimal"
    )
    metadata.status += "FINISHED"
    metadata.flags += flags
    return metadata

def run_reduction_arc(config, metadata, calib_metadata, frame_settings):
    metadata = run_reduction_block(config, metadata=metadata, calib_metadata=calib_metadata, frame_settings=frame_settings)
    _, flags = rssMethod.detWaveSolution_drp(
        arc_rss=frame_settings.output_path.format(kind="ms"),
        disp_rss=frame_settings.output_path.format(kind="disp"),
        res_rss=frame_settings.output_path.format(kind="res"),
        ref_line_file=frame_settings.pix2wave_map,
        aperture="7", poly_fwhm="-1,-1", poly_dispersion="-4", rel_flux_limits="0.2,2", flux_min="100.0", verbose="-1"
    )
    metadata.flags += flags
    _, flags = rssMethod.createPixTable_drp(
        rss_in=frame_settings.output_path.format(kind="ms"),
        rss_out=frame_settings.output_path.format(kind="rss"),
        arc_wave=frame_settings.output_path.format(kind="disp"),
        arc_fwhm=frame_settings.output_path.format(kind="res"),
        cropping=''
    )
    metadata.flags += flags
    _, flags = rssMethod.resampleWave_drp(
        rss_in=frame_settings.output_path.format(kind="rss"),
        rss_out=frame_settings.output_path.format(kind="disp_cor"),
        start_wave=frame_settings.wl_range[0], end_wave=frame_settings.wl_range[1], disp_pix="1.0", err_sim="0"
    )
    metadata.flags += flags
    metadata.status += "FINISHED"
    return metadata

def run_reduction_object(config, metadata, calib_metadata, frame_settings):

    _, flags = imageMethod.LACosmic_drp(
        image=frame_settings.output_path.format(kind="calib"),
        out_image=frame_settings.output_path.format(kind="cosmic"),
        increase_radius=1, flim="1.3", parallel='5'
    )
    metadata.flags += flags
    metadata = run_reduction_block(config, metadata=metadata, calib_metadata=calib_metadata, frame_settings=frame_settings)
    _, flags = rssMethod.createPixTable_drp(
        rss_in=frame_settings.output_path.format(kind="ms"),
        rss_out=frame_settings.output_path.format(kind="rss"),
        arc_wave=frame_settings.output_path.format(kind="disp"), arc_fwhm=frame_settings.output_path.format(kind="res"), cropping=''
    )
    metadata.flags += flags
    _, flags = rssMethod.resampleWave_drp(
        rss_in=frame_settings.output_path.format(kind="rss"),
        rss_out=frame_settings.output_path.format(kind="disp_cor"),
        start_wave=frame_settings.wl_range[0], end_wave=frame_settings.wl_range[1], disp_pix="1.0", err_sim="0"
    )
    metadata.flags += flags
    metadata.status += "FINISHED"
    return metadata