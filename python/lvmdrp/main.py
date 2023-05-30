# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: main.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import yaml

from lvmdrp.core import image
from lvmdrp.core.constants import CALIBRATION_TYPES, FRAMES_PRIORITY
from lvmdrp.functions import imageMethod, rssMethod
from lvmdrp.utils.bitmask import ReductionStatus
from lvmdrp.utils.metadata import LAMP_NAMES, match_master_metadata
from lvmdrp.utils.decorators import validate_fibers


# TODO: define values in this dictionary as strings and move it to the constants module
REDUCTION_METHODS = {
    "pre": imageMethod.preprocRawFrame_drp,
    "calib": imageMethod.detrendFrame_drp,
    "cosmic": imageMethod.LACosmic_drp,
    "peaks": imageMethod.findPeaksAuto_drp,
    "trace": imageMethod.tracePeaks_drp,
    "stray": imageMethod.subtractStraylight_drp,
    "fwhm": imageMethod.traceFWHM_drp,
    "extract": imageMethod.extractSpec_drp,
    "wave": rssMethod.detWaveSolution_drp,
    "pixtable": rssMethod.createPixTable_drp,
    "waveres": rssMethod.resampleWave_drp,
}
STEPS_FILE_PARS = {
    "pre": {"in": ["in_image"], "out": ["out_image"]},
    "calib": {"in": ["in_image", "bias", "dark", "flat"], "out": ["out_image"]},
    "cosmic": {"in": ["image"], "out": ["out_image"]},
    "peaks": {"in": ["image"], "out": ["out_peaks_file"]},
    "trace": {"in": ["image", "peaks_file"], "out": ["trace_out"]},
    "stray": {"in": ["image", "trace"], "out": ["stray_image", "clean_image"]},
    "fwhm": {"in": ["image", "trace"], "out": ["fwhm_out"]},
    "extract": {"in": ["image", "trace"], "out": ["out_rss"]},
    "wave": {"in": ["arc_rss", "ref_line_file"], "out": ["disp_rss", "res_rss"]},
    "pixwave": {"in": ["rss_in", "arc_wave", "arc_fwhm"], "out": ["rss_out"]},
    "waveres": {"in": ["rss_in"], "out": ["rss_out"]},
}
REDUCTION_STEPS = {
    "bias": ["pre"],
    "dark": ["pre", "calib"],
    "flat": ["pre", "calib"],
    "continuum": ["pre", "calib", "cosmic", "peaks", "trace", "stray", "fwhm"],
    "arc": [
        "pre",
        "calib",
        "cosmic",
        "stray",
        "extract",
        "wave",
        "pixtable",
        "waveres",
    ],
    "object": ["pre", "calib", "cosmic", "stray", "extract", "pixtable", "waveres"],
}


def define_inout_paths(params, metadata, calib_metadata, prev_step, curr_step, path):
    # define previous and current step in/out param names
    curr_io_param_names = (
        STEPS_FILE_PARS[curr_step]["in"] + STEPS_FILE_PARS[curr_step]["out"]
    )
    # define previous and current step params
    curr_params = params[curr_step]
    # extract those params corresponding to in/out files
    curr_paths = {
        key: value for key, value in curr_params.items() if key in curr_io_param_names
    }
    # depending on the reduction type and step, set path params: path, label, kind
    if curr_step == "pre":
        curr_paths["in_image"] = curr_paths["in_image"].format(
            path=path, mjd=metadata.mjd, label=metadata.label
        )
        curr_paths["out_image"] = curr_paths["out_image"].format(
            path=path, label=metadata.label, kind=curr_step
        )
    elif curr_step == "calib":
        curr_paths["in_image"] = curr_paths["in_image"].format(
            path=path, label=metadata.label, kind="pre"
        )
        curr_paths["bias"] = (
            calib_metadata["bias"].path if calib_metadata["bias"] is not None else None
        )
        curr_paths["dark"] = (
            calib_metadata["dark"].path if calib_metadata["dark"] is not None else None
        )
        curr_paths["flat"] = (
            calib_metadata["flat"].path if calib_metadata["flat"] is not None else None
        )
        curr_paths["out_image"] = curr_paths["out_image"].format(
            path, label=metadata.label, kind=curr_step
        )
    elif curr_step == "peaks":
        curr_paths["image"] = curr_paths["image"].format(
            path=path, label=metadata.label, kind=prev_step
        )
        curr_paths["out_peaks_file"] = curr_paths["out_peaks_file"].format(
            path=path, label=metadata.label, kind=curr_step
        )
    elif curr_step == "trace":
        curr_paths["image"] = curr_paths["image"].format(
            path=path, label=metadata.label, kind=prev_step
        )
        curr_paths["peaks_file"] = curr_paths["peaks_file"].format(
            path=path, label=metadata.label, kind="peaks"
        )
        curr_paths["trace_out"] = curr_paths["trace_out"].format(
            path=path, label=metadata.label, kind=curr_step
        )
    elif curr_step == "stray":
        curr_paths["image"] = curr_paths["image"].format(
            path=path, label=metadata.label, kind=prev_step
        )
        curr_paths["trace"] = curr_paths["trace"].format(
            path=path, label=metadata.label, kind="trace"
        )
        curr_paths["stray_image"] = curr_paths["stray_image"].format(
            path=path, label=metadata.label, kind="back"
        )
        curr_paths["clean_image"] = curr_paths["clean_image"].format(
            path=path, label=metadata.label, kind=curr_step
        )
    elif curr_step == "fwhm":
        curr_paths["image"] = curr_paths["image"].format(
            path=path, label=metadata.label, kind=prev_step
        )
        curr_paths["trace"] = curr_paths["trace"].format(
            path=path, label=metadata.label, kind="trace"
        )
        curr_paths["fwhm_out"] = curr_paths["fwhm_out"].format(
            path=path, label=metadata.label, kind=curr_step
        )
    elif curr_step == "extract":
        curr_paths["image"] = curr_paths["image"].format(
            path=path, label=metadata.label, kind=prev_step
        )
        curr_paths["trace"] = curr_paths["trace"].format(
            path=path, label=metadata.label, kind="trace"
        )
        curr_paths["out_rss"] = curr_paths["out_rss"].format(
            path=path, label=metadata.label, kind=curr_step
        )
    elif curr_step == "wave":
        curr_paths["arc_rss"] = curr_paths["arc_rss"].format(
            path=path, label=metadata.label, kind=prev_step
        )
        curr_paths["disp_rss"] = curr_paths["disp_rss"].format(
            path=path, label=metadata.label, kind="disp"
        )
        curr_paths["res_rss"] = curr_paths["res_rss"].format(
            path=path, label=metadata.label, kind="res"
        )
        curr_paths["ref_line_file"] = calib_metadata["pix2wave"].path
    elif curr_step == "pixtable":
        curr_paths["rss_in"] = curr_paths["rss_in"].format(
            path=path, label=metadata.label, kind=prev_step
        )
        curr_paths["rss_out"] = curr_paths["rss_out"].format(
            path=path, label=metadata.label, kind=curr_step
        )
        curr_paths["arc_wave"] = curr_paths["arc_wave"].format(
            path=path, label=metadata.label, kind="disp"
        )
        curr_paths["arc_fwhm"] = curr_paths["arc_fwhm"].format(
            path=path, label=metadata.label, kind="res"
        )
    elif curr_step == "waveres":
        curr_paths["rss_in"] = curr_paths["rss_in"].format(
            path=path, label=metadata.label, kind=prev_step
        )
        curr_paths["rss_out"] = curr_paths["rss_out"].format(
            path=path, label=metadata.label, kind="rss"
        )

    return curr_paths


# TODO:
#   - define the paths to the input files
#   - define the paths to the output files
#   - define parameters to pass on each step


def parse_arguments(config, args=None):
    if args is None:
        args = sys.argv[1:]
    # define cmdline arguments parser
    parser = argparse.ArgumentParser(
        prog="LVM data reduction pipeline",
        description="This pipeline takes raw frames and runs the reduction process to produce science-ready frames",
    )
    # config parser
    # parse
    cmd_args = parser.parse_args(args)
    # replace config parameters with cmdline arguments
    return config, cmd_args


# BUG: redux settings should contain information about the whole reduction process
#   - which steps to carry on
#   - which parameters should be used in each step
#   - input and output paths well defined
#   - write current run config file (read only permissions)
#   - add possibility to run in dry mode, only producing the run config file
# BUG: handling paths from one step to the other
# OPTION 1:
#   - define a paths dictionary placeholder key: step, value: None
#   - fill the corresponding paths if the given step is present in the current run
#   - bypass steps with missing mandatory files
#   - set corresponding flags
# OPTION 2:
#   - create a dictionary with key: step and value: paths
#   - remove steps that are not present in the current run
#   - bypass steps with missing mandatory files
def setup_reduction(config, metadata):
    # create a mapping of the target frame and the metadata/calibration frames needed to run the calibration
    settings = Namespace()
    # copy basic metadata to this mapping from config
    settings.ccd = metadata.ccd
    settings.spec = metadata.spec
    settings.mjd = metadata.mjd
    settings.wl_range = config.WAVELENGTH_RANGES.__dict__[metadata.ccd[0]]
    settings.pix2wave_map = os.path.join(
        config.LVM_DRP_CONFIG_PATH, config.PIX2WAVE_MAPS.__dict__[metadata.ccd]
    )
    settings.lamps = [
        name
        for name in metadata.__data__
        if name in LAMP_NAMES and metadata.__data__[name]
    ]
    settings.label = metadata.label
    # find type of reduction and calibration frames depending on the target image
    if metadata.imagetyp in FRAMES_PRIORITY:
        settings.type = metadata.imagetyp
    else:
        raise ValueError(f"unrecognized image type '{metadata.imagetyp}'")

    # TODO: add output file names to this mapping
    settings.output_path = (
        config.LVM_SPECTRO_CALIB_PATH
        if settings.type in CALIBRATION_TYPES
        else config.LVM_SPECTRO_REDUX_PATH
    )

    # TODO: get calibration frames
    calib_metadata = match_master_metadata(metadata=metadata)

    settings.steps = {}
    settings.param = {}
    last_step = None
    for step in REDUCTION_STEPS[settings.type]:
        settings.param[step] = config.DRP_STEPS.__dict__[step].__dict__
        paths = define_inout_paths(
            params=settings.param,
            metadata=metadata,
            calib_metadata=calib_metadata,
            prev_step=last_step,
            curr_step=step,
            path=settings.output_path,
        )
        # add paths to settings
        step_file_pars = STEPS_FILE_PARS[step]
        for in_file_par in step_file_pars["in"]:
            settings.param[step][in_file_par] = paths[in_file_par]
        for out_file_par in step_file_pars["out"]:
            settings.param[step][out_file_par] = paths[out_file_par]
        settings.steps[step] = REDUCTION_METHODS[step]
        # keeping track of last step to know which output files were produced
        last_step = step

    return metadata, settings


def run(config, metadata, settings):
    # for each reduction step in settings:
    # check if step validation is needed
    # apply validation decorators
    # run step
    # update metadata
    return metadata


def run_reduction_calib(config, metadata, calib_metadata, settings):
    # decorate preprocessing if necessary
    if settings.type in ["continuum", "arc", "object"]:
        preproc = validate_fibers(["BAD_FIBERS"], config, "out_image")(
            imageMethod.preprocRawFrame_drp
        )
    else:
        preproc = imageMethod.preprocRawFrame_drp

    proc_image, flags = preproc(
        in_image=config.INPUT_FILE_PATTERN,
        channel=settings.ccd,
        out_image=settings.output_path.format(label=metadata.label, kind="pre"),
        boundary_x="1,2040",
        boundary_y="1,2040",
        positions="00,10,01,11",
        orientation="S,S,S,S",
        gain=config.GAIN,
        rdnoise=config.READ_NOISE,
    )
    metadata.status += "PREPROCESSED"
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
    if settings.type == "flat":
        proc_image = proc_image / np.median(proc_image._data)

    # run basic calibration for each analog
    calib_image = (proc_image - master_dark - master_bias) / master_flat
    calib_image.writeFitsData(
        settings.output_path.format(label=metadata.label, kind="calib")
    )

    metadata.naxis1 = calib_image._header["NAXIS1"]
    metadata.naxis2 = calib_image._header["NAXIS2"]
    metadata.status += "CALIBRATED"
    metadata.flags += flags
    return metadata


def run_reduction_block(config, metadata, calib_metadata, settings):
    # build calibration paths
    master_continuum_path = settings.output_path.format(
        label=calib_metadata["continuum"].label, kind="{kind}"
    )
    target_frame_path = settings.output_path.format(label=metadata.label, kind="{kind}")
    _, flags = imageMethod.subtractStraylight_drp(
        in_image=target_frame_path.format(kind="cosmic"),
        trace=master_continuum_path.format(kind="trc"),
        out_stray=target_frame_path.format(kind="back"),
        out_image=target_frame_path.format(kind="stray"),
        aperture=40,
        poly_cross=2,
        smooth_gauss=30,
    )
    metadata.flags += flags
    _, flags = imageMethod.extractSpec_drp(
        in_image=target_frame_path.format(kind="stray"),
        in_trace=master_continuum_path.format(kind="trc"),
        out_rss=target_frame_path.format(kind="ms"),
        fwhm=master_continuum_path.format(kind="fwhm"),
        method="optimal",
        parallel="5",
    )
    metadata.flags += flags
    return metadata


def run_reduction_continuum(config, metadata, calib_metadata, settings):
    # if bad flags bypass continuum calibration
    if "BAD_FIBERS" in metadata.flags:
        metadata.status = ReductionStatus["FAILED"]
        return metadata
    # BUG: add continuum frames to CALIBRATION_FRAMES in DB
    target_frame_path = settings.output_path.format(label=metadata.label, kind="{kind}")
    _, flags = imageMethod.LACosmic_drp(
        in_image=target_frame_path.format(kind="calib"),
        out_image=target_frame_path.format(kind="cosmic"),
        increase_radius=1,
        flim="1.3",
        parallel="5",
    )
    metadata.flags += flags
    # BUG: verify outputs against expected values, skip calibration steps if needed & set corresponding flags
    _, flags = imageMethod.findPeaksAuto_drp(
        in_image=target_frame_path.format(kind="cosmic"),
        out_peaks_file=target_frame_path.format(kind="trace").replace(
            ".fits", ".peaks"
        ),
        disp_axis="X",
        threshold="5000",
        slice="3696",
        nfibers="41",
        median_box="1",
        median_cross="1",
        method="gauss",
        init_sigma="0.5",
        verbose=0,
    )
    metadata.flags += flags
    _, flags = imageMethod.tracePeaks_drp(
        in_image=target_frame_path.format(kind="cosmic"),
        in_peaks=target_frame_path.format(kind="trace").replace(".fits", ".peaks"),
        out_trace=target_frame_path.format(kind="trc"),
        steps=30,
        method="gauss",
        threshold_peak=50,
        poly_disp=5,
        coadd=30,
        verbose=0,
    )
    metadata.flags += flags
    _, flags = imageMethod.subtractStraylight_drp(
        in_image=target_frame_path.format(kind="cosmic"),
        trace=target_frame_path.format(kind="trc"),
        out_stray=target_frame_path.format(kind="back"),
        out_image=target_frame_path.format(kind="stray"),
        aperture=40,
        poly_cross=2,
        smooth_gauss=30,
    )
    metadata.flags += flags
    _, flags = imageMethod.traceFWHM_drp(
        in_image=target_frame_path.format(kind="stray"),
        in_trace=target_frame_path.format(kind="trc"),
        out_fwhm=target_frame_path.format(kind="fwhm"),
        blocks=32,
        steps=30,
        coadd=20,
        threshold_flux=50.0,
        poly_disp=5,
        clip="1.5,4.0",
    )
    metadata.flags += flags
    _, flags = imageMethod.extractSpec_drp(
        in_image=target_frame_path.format(kind="stray"),
        in_trace=target_frame_path.format(kind="trc"),
        out_rss=target_frame_path.format(kind="ms"),
        fwhm=target_frame_path.format(kind="fwhm"),
        parallel=5,
        method="optimal",
    )
    metadata.status += "FINISHED"
    metadata.flags += flags
    return metadata


def run_reduction_arc(config, metadata, calib_metadata, settings):
    metadata = run_reduction_block(
        config, metadata=metadata, calib_metadata=calib_metadata, settings=settings
    )
    target_frame_path = settings.output_path.format(label=metadata.label, kind="{kind}")
    _, flags = rssMethod.detWaveSolution_drp(
        arc_rss=target_frame_path.format(kind="ms"),
        disp_rss=target_frame_path.format(kind="disp"),
        res_rss=target_frame_path.format(kind="res"),
        ref_line_file=settings.pix2wave_map,
        aperture="7",
        poly_fwhm="-1,-1",
        poly_dispersion="-4",
        rel_flux_limits="0.2,2",
        flux_min="100.0",
        verbose="-1",
    )
    metadata.flags += flags
    _, flags = rssMethod.createPixTable_drp(
        rss_in=target_frame_path.format(kind="ms"),
        rss_out=target_frame_path.format(kind="rss"),
        arc_wave=target_frame_path.format(kind="disp"),
        arc_fwhm=target_frame_path.format(kind="res"),
        cropping="",
    )
    metadata.flags += flags
    _, flags = rssMethod.resampleWave_drp(
        in_rss=target_frame_path.format(kind="rss"),
        out_rss=target_frame_path.format(kind="disp_cor"),
        start_wave=settings.wl_range[0],
        end_wave=settings.wl_range[1],
        disp_pix="1.0",
        err_sim="0",
    )
    metadata.flags += flags
    metadata.status += "FINISHED"
    return metadata


def run_reduction_object(config, metadata, calib_metadata, settings):
    _, flags = imageMethod.LACosmic_drp(
        in_image=settings.output_path.format(kind="calib"),
        out_image=settings.output_path.format(kind="cosmic"),
        increase_radius=1,
        flim="1.3",
        parallel="5",
    )
    metadata.flags += flags
    metadata = run_reduction_block(
        config, metadata=metadata, calib_metadata=calib_metadata, settings=settings
    )
    _, flags = rssMethod.createPixTable_drp(
        rss_in=settings.output_path.format(kind="ms"),
        rss_out=settings.output_path.format(kind="rss"),
        arc_wave=settings.output_path.format(kind="disp"),
        arc_fwhm=settings.output_path.format(kind="res"),
        cropping="",
    )
    metadata.flags += flags
    _, flags = rssMethod.resampleWave_drp(
        in_rss=settings.output_path.format(kind="rss"),
        out_rss=settings.output_path.format(kind="disp_cor"),
        start_wave=settings.wl_range[0],
        end_wave=settings.wl_range[1],
        disp_pix="1.0",
        err_sim="0",
    )
    metadata.flags += flags
    metadata.status += "FINISHED"
    return metadata


def run_sky_subtraction(config, rss, lsf_rss):
    # select sky fibers
    #   - use the fiber map to separate science, sky, std stars
    #   - select best method to build master sky (naive: combine all fibers, smart: best match to science in spectral space)
    # select science/standard stars fibers
    # build master sky
    # get cont_sky, lines_sky
    #   - read a selection of common lines
    #   - mask those lines
    #   - smooth masked master sky
    #   - residual = master_sky - smoothed spectrum
    #   - return sky_cont, sky_lines
    # interpolate/extrapolate sky using ESO
    # spatially interpolate sky continuum using the ESO sky model sky_cont_corr=0.5*( sky1_cont*(model_skysci/model_sky1) + sky2_cont*(model_skysci/model_sky2))
    # combine corrected versions of lines and continuum sky
    # for each fiber match LSF between sky and science
    # subtract the matched sky from science fibers
    # select faintest fiber in the science field refine sky subtraction with it
    # PCA refinement
    pass
