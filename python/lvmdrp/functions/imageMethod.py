#!/usr/bin/env python
# encoding: utf-8

from __future__ import annotations

import multiprocessing
import os
import sys
from itertools import product
from copy import deepcopy as copy
from multiprocessing import Pool
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

import numpy
import bottleneck as bn
from astropy.table import Table
from astropy.io import fits as pyfits
from astropy.visualization import simple_norm
from astropy.wcs import wcs
import astropy.io.fits as fits
from scipy import interpolate
from scipy import signal
from tqdm import tqdm

from typing import List, Tuple

from lvmdrp import log, __version__ as DRPVER
from lvmdrp.core.constants import CONFIG_PATH, SPEC_CHANNELS, ARC_LAMPS, LVM_REFERENCE_COLUMN, LVM_NBLOCKS, FIDUCIAL_PLATESCALE
from lvmdrp.utils.decorators import skip_on_missing_input_path, drop_missing_input_paths
from lvmdrp.utils.bitmask import QualityFlag
from lvmdrp.core.fiberrows import FiberRows, _read_fiber_ypix
from lvmdrp.core.image import (
    Image,
    _parse_ccd_section,
    _model_overscan,
    _remove_spikes,
    _fillin_valleys,
    _no_stepdowns,
    combineImages,
    glueImages,
    loadImage,
)
from lvmdrp.core.plot import plt, create_subplots, plot_detrend, plot_strips, plot_image_shift, plot_fiber_thermal_shift, save_fig
from lvmdrp.core.rss import RSS
from lvmdrp.core.spectrum1d import Spectrum1D, _spec_from_lines, _cross_match
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.utils.hdrfix import apply_hdrfix
from lvmdrp.utils.convert import dateobs_to_sjd, correct_sjd


NQUADS = 4
DEFAULT_IMAGETYP = "object"
DEFAULT_TRIMSEC = [
    "[1:2043, 2041:4080]",
    "[2078:4120, 2041:4080]",
    "[1:2043, 1:2040]",
    "[2078:4120, 1:2040]",
]
DEFAULT_BIASSEC = [
    "[2044:2060, 2041:4080]",
    "[2061:2077, 2041:4080]",
    "[2044:2060, 1:2040]",
    "[2061:2077, 1:2040]",
]
DEFAULT_BGSEC = [
    "[1:2043, 61:70]",
    "[1:2043, 61:70]",
    "[1:2043, 1991:2000]",
    "[1:2043, 1991:2000]",
]
# PAVAN'S FIT OF GAIN
DEFAULT_GAIN = {
    "b1": [2.71, 2.71, 2.69, 2.68],
    "b2": [2.62, 2.69, 2.69, 2.68],
    "b3": [2.71, 2.77, 2.73, 2.69],
    "r1": [2.75, 2.79, 2.74, 2.68],
    "r2": [2.64, 2.79, 2.67, 2.68],
    "r3": [2.74, 2.76, 2.81, 2.73],
    "z1": [2.76, 2.65, 2.78, 2.89],
    "z2": [2.70, 2.76, 2.81, 2.71],
    "z3": [2.75, 2.85, 2.79, 2.74]
}
DEFAULT_PTC_PATH = os.path.join(os.environ["LVMCORE_DIR"], "metrology", "PTC_fit.txt")

description = "Provides Methods to process 2D images"

__all__ = [
    "find_peaks_auto",
    "trace_peaks",
    "subtract_straylight",
    "traceFWHM_drp",
    "extract_spectra",
    "preproc_raw_frame",
    "detrend_frame",
    "create_master_frame",
]


def _nonlinearity_correction(ptc_params: None | numpy.ndarray, nominal_gain: float, quadrant: Image, iquad: int) -> Image:
    """calculates non-linearity correction for input quadrant

    Parameters
    ----------
    ptc_params : numpy.ndarray
        table with non-linearity correction parameters
    nominal_gain : float
        gain value of the input quadrant
    quadrant : Image
        input quadrant
    iquad : int
        quadrant number (1-index based)

    Returns
    -------
    Image
        gain map

    """
    quad = {1: "TL", 2: "TR", 3: "BL", 4: "BR"}[iquad]
    camera = quadrant._header["CCD"]


    if ptc_params is not None:
        row_qc = (ptc_params["C"] == camera) & (ptc_params["Q"] == quad)
        col_coeffs = "a1 a2 a3".split()

        a1, a2, a3 = ptc_params[row_qc][col_coeffs][0]
        log.info(f"calculating gain map using parameters: {a1 = :.2g}, {a2 = :.2g}, {a3 = :.2g}")
        gain_map = Image(data=1 / (a1 + a2*a3 * quadrant._data**(a3-1)))
        gain_map.setData(data=nominal_gain, select=numpy.isnan(gain_map._data), inplace=True)

        gain_med = bn.nanmedian(gain_map._data)
        gain_min, gain_max = bn.nanmin(gain_map._data), bn.nanmax(gain_map._data)
        log.info(f"gain map stats: {gain_med = :.2f} [{gain_min = :.2f}, {gain_max = :.2f}] ({nominal_gain = :.2f} e-/ADU)")
    else:
        log.warning("cannot apply non-linearity correction")
        log.info(f"using {nominal_gain = } (e-/ADU)")
        gain_map = Image(data=numpy.ones(quadrant._data.shape) * nominal_gain)
    return gain_map


def _create_peaks_regions(fibermap: Table, column: int = 2000) -> None:
    """creates a DS9 region file with the fiber peaks

    Parameters
    ----------
    fibermap : Table
        fibermap in the format given in lvmcore repository
    column : int, optional
        column number, by default 2000
    """
    for camspec in product("brz","123"):
        camera = "".join(camspec)
        out_region = f"peaks_{camera}.reg"

        fibermap_cam = fibermap[fibermap["spectrographid"] == int(camspec[1])]
        centers = fibermap_cam[f"ypix_{camspec[0]}"].data

        with open(out_region, "w") as reg_out:
            reg_out.write("# Region file format: DS9 version 4.1\n")
            reg_out.write(
                'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
            )
            reg_out.write("physical\n")
            for i in range(len(centers)):
                reg_out.write(
                    "# text(%.4f,%.4f) text={%i, %i}\n"
                    % (column+1, centers[i]+1, i + 1, centers[i])
                )


def _create_trace_regions(out_trace, table_data, table_poly, table_poly_all, label=None, display_plots=False):
    """Creates three DS9 region files with the trace data

    Parameters
    ----------
    out_trace : str
        output trace file
    table_data : numpy.ndarray
        trace data (measurements)
    table_poly : numpy.ndarray
        trace polynomial evaluated at the trace data
    table_poly_all : numpy.ndarray
        trace polynomial evaluated at all pixels
    label : str, optional
        label for the trace, by default None
    display_plots : bool, optional
        display plots, by default False
    """
    if len(table_data) == 0 or len(table_poly) == 0:
        return
    coords_file = out_trace.replace("calib", "ancillary").replace(".fits", "_coords.txt")
    os.makedirs(os.path.dirname(coords_file), exist_ok=True)
    poly_file = coords_file.replace("_coords.txt", "_poly.txt")
    poly_all_file = coords_file.replace("_coords.txt", "_poly_all.txt")
    log.info(f"writing trace data to files: {os.path.basename(coords_file)}, {os.path.basename(poly_file)} and {os.path.basename(poly_all_file)}")
    numpy.savetxt(coords_file, 1+table_data, fmt="%.5f")
    numpy.savetxt(poly_file, 1+table_poly, fmt="%.5f")
    numpy.savetxt(poly_all_file, 1+table_poly_all, fmt="%.5f")

    # plot trace fitting residuals
    if label is None:
        label = os.path.basename(out_trace).replace(".fits", "")

    fig, ax = create_subplots(1, 1, figsize=(10, 10))
    ax.axhspan(-1, 1, color="k", alpha=0.1)
    ax.axhspan(-10, 10, color="k", alpha=0.1)
    residuals = (table_poly[:, 1] - table_data[:, 1]) / table_data[:, 1] * 100
    ax.plot(table_data[:, 0], residuals, "o", ms=1, color="k")
    ax.set_ylim(-25, 25)
    ax.set_xlabel("y (pixel)")
    ax.set_ylabel("residuals (%)")
    ax.set_title(f"{label} residuals")
    save_fig(fig, out_trace, label="residuals", figure_path="qa", to_display=display_plots)


def _channel_combine_fiber_params(in_cent_traces, in_waves, add_overscan_columns=True):
    """Combines fiber centroid traces and wavelength model for a given spectrograph along the x-axis

    Given a set of centroid traces and wavelength models for a given spectrograph, this function
    combines them along the x-axis, adding overscan columns if requested. Additionally the regions
    past the dichroic are masked.

    Parameters
    ----------
    in_cent_traces : list
        list of input centroid trace files for the same spectrograph
    in_waves : list
        list of input wavelength files for the same spectrograph
    add_overscan_columns : bool, optional
        add overscan columns, by default True

    Returns
    -------
    list
        list of fiber centroid traces
    list
        list of wavelength models
    FiberRows
        channel stacked fiber centroid trace data
    FiberRows
        channel stacked wavelength data
    """
    channels = "brz"
    # read master trace and wavelength
    mtraces = [FiberRows() for _ in range(len(in_cent_traces))]
    mwaves = [FiberRows() for _ in range(len(in_waves))]
    channel_masks = []
    for i, (mtrace_path, mwave_path) in enumerate(zip(in_cent_traces, in_waves)):
        mtraces[i].loadFitsData(mtrace_path)
        mwaves[i].loadFitsData(mwave_path)

        # add columns for OS region
        if add_overscan_columns:
            os_region = numpy.zeros((mtraces[i]._data.shape[0], 34), dtype=int)
            trace_data = numpy.split(mtraces[i]._data, 2, axis=1)
            trace_mask = numpy.split(mtraces[i]._mask, 2, axis=1)
            mtraces[i]._data = numpy.concatenate([trace_data[0], os_region, trace_data[1]], axis=1)
            mtraces[i]._mask = numpy.concatenate([trace_mask[0], ~os_region.astype(bool), trace_mask[1]], axis=1)
            wave_data = numpy.split(mwaves[i]._data, 2, axis=1)
            wave_mask = numpy.split(mwaves[i]._mask, 2, axis=1)
            mwaves[i]._data = numpy.concatenate([wave_data[0], os_region, wave_data[1]], axis=1)
            mwaves[i]._mask = numpy.concatenate([wave_mask[0], ~os_region.astype(bool), wave_mask[1]], axis=1)

        # get pixels where spectropgraph is not blocked
        channel_mask = (SPEC_CHANNELS[channels[i]][0]>=mwaves[i]._data)|(mwaves[i]._data>=SPEC_CHANNELS[channels[i]][1])
        channel_masks.append(channel_mask)

        mtraces[i]._mask |= channel_mask
        mwaves[i]._mask |= channel_mask
    mtrace = FiberRows()
    mtrace.unsplit(mtraces)
    mwave = FiberRows()
    mwave.unsplit(mwaves)

    return mtraces, mwaves, mtrace, mwave


def _get_fiber_selection(traces, image_shape=(4080, 4120), y_widths=3):
    """Returns a mask with selected fibers within a given width

    Given a set of centroid traces, this function returns a mask with selected fibers
    within a given width.

    Parameters
    ----------
    traces : list
        list of fiber centroid traces
    image_shape : tuple, optional
        shape of the image, by default (4080, 4120)
    y_widths : int, optional
        width of the fiber trace, by default 3

    Returns
    -------
    numpy.ndarray
        fiber selection mask
    """
    images = [Image(data=numpy.zeros(image_shape), mask=numpy.zeros(image_shape, dtype=bool)) for _ in range(len(traces))]
    for i in range(len(traces)):
        images[i].maskFiberTraces(traces[i], aperture=y_widths, parallel=1)
    image = Image()
    image.unsplit(images)

    return image._mask


def _get_wave_selection(waves, lines_list, window):
    """Returns selection windows from a list of wavelengths

    Given a list of wavelengths, this function returns selection windows
    based on a given window size.

    Parameters
    ----------
    waves : numpy.ndarray
        wavelength data
    lines_list : list
        list of lines to select
    window : float
        window size

    Returns
    -------
    numpy.ndarray
        selection mask
    """
    hw = window / 2
    wave_selection = numpy.zeros_like(waves, dtype=bool)
    if len(wave_selection.shape) == 2:
        for ifiber in range(wave_selection.shape[0]):
            for wline in lines_list:
                if wline-hw < waves[ifiber].min() or wline+hw > waves[ifiber].max():
                    continue
                wave_selection[ifiber] |= (waves[ifiber] > wline-hw) & (waves[ifiber] < wline+hw)
    else:
        wave_selection = (waves > wline-hw) & (waves < wline+hw)
    return wave_selection


def _fix_fiber_thermal_shifts(image, trace_cent, trace_width=None, trace_amp=None, fiber_model=None,
                              columns=[500, 1000, 1500, 2000, 2500, 3000],
                              column_width=25, shift_range=[-5,5], axs=None):
    """Returns the updated fiber trace centroids after fixing the thermal shifts

    Parameters
    ----------
    image : Image
        the target image
    trace_cent : TraceMask
        the fiber trace centroids
    trace_width : TraceMask
        the fiber trace widths, defaults to None
    trace_amp : TraceMask
        the fiber trace amplitudes, defaults to None
    fiber_model : Image
        2D fiber model image, defaults to None
    columns : list
        list of columns to evaluate the continuum model, defaults to [500, 1000, 1500, 2000, 2500, 3000, 3500]
    column_width : int
        number of columns to add around the given columns, defaults to 25
    shift_range : list
        range of shifts to consider, defaults to [-5,5]

    Returns
    -------
    numpy.ndarray
        the calculated column shifts
    numpy.ndarray
        the mean shifts
    numpy.ndarray
        the standard deviation of the shifts
    TraceMask
        the updated fiber trace centroids
    Image
        the evaluated continuum model at the given columns
    """
    # generate the continuum model using the master traces only along the specific columns
    if fiber_model is None:
        fiber_model, _ = image.eval_fiber_model(trace_cent, trace_width, trace_amp=trace_amp, columns=columns, column_width=column_width)

    mjd = image._header["SMJD"]
    expnum = image._header["EXPOSURE"]
    camera = image._header["CCD"]

    # calculate thermal shifts
    column_shifts = image.measure_fiber_shifts(fiber_model, trace_cent, columns=columns, column_width=column_width, shift_range=shift_range, axs=axs)
    # shifts stats
    median_shift = numpy.nan_to_num(bn.nanmedian(column_shifts, axis=0))
    std_shift = numpy.nan_to_num(bn.nanstd(column_shifts, axis=0))
    if numpy.abs(median_shift) > 0.5:
        log.warning(f"large thermal shift measured: {','.join(map(str, column_shifts))} pixels for {mjd = }, {expnum = }, {camera = }")
        image.add_header_comment(f"large thermal shift: {','.join(map(str, column_shifts))} pixels {camera = }")
        log.warning(f"measured shifts median+/-stddev = {median_shift:.4f}+/-{std_shift:.4f} pixels")
    else:
        log.info(f"measured shifts median+/-stddev = {median_shift:.4f}+/-{std_shift:.4f} pixels for {mjd = }, {expnum = }, {camera = }")

    # apply average shift to the zeroth order trace coefficients
    trace_cent_fixed = copy(trace_cent)
    trace_cent_fixed._coeffs[:, 0] += median_shift
    trace_cent_fixed.eval_coeffs()

    return trace_cent_fixed, column_shifts, median_shift, std_shift, fiber_model


def _apply_electronic_shifts(images, out_images, drp_shifts=None, qc_shifts=None, custom_shifts=None, raw_shifts=None,
                             which_shifts="drp", apply_shifts=True, dry_run=False, display_plots=False):
    """Applies the chosen electronic pixel shifts to the images and plots the results

    Parameters
    ----------
    images : list
        list of input images
    out_images : list
        list of output images
    drp_shifts : numpy.ndarray
        DRP electronic pixel shifts, by default None
    qc_shifts : numpy.ndarray
        QC electronic pixel shifts, by default None
    custom_shifts : numpy.ndarray
        custom electronic pixel shifts, by default None
    raw_shifts : numpy.ndarray
        raw DRP electronic pixel shifts, by default None
    which_shifts : str
        chosen electronic pixel shifts, by default "drp"
    apply_shifts : bool
        apply the shifts, by default True
    dry_run : bool
        dry run mode (does not save corrected images), by default False
    display_plots : bool
        display plots, by default False

    Returns
    -------
    list
        list of corrected images
    numpy.ndarray
        the chosen electronic pixel shifts
    str
        name of the chosen electronic pixel shifts ('drp', 'qc' or 'custom')
    """
    images_out = [copy(image) for image in images]
    for image, image_out, out_image in zip(images, images_out, out_images):
        mjd = image._header.get("SMJD", image._header["MJD"])
        expnum, camera = image._header["EXPOSURE"], image._header["CCD"]
        imagetyp = image._header["IMAGETYP"]

        if which_shifts == "drp":
            this_shifts = drp_shifts
            image_color = "Blues"
        elif which_shifts == "qc":
            this_shifts = qc_shifts
            image_color = "Greens"
        elif which_shifts == "custom":
            this_shifts = custom_shifts
            image_color = "Purples"
        else:
            this_shifts = drp_shifts

        if apply_shifts and numpy.any(this_shifts != 0):
            shifted_rows = numpy.where(numpy.gradient(this_shifts) > 0)[0][1::2].tolist()
            log.info(f"applying shifts from rows {shifted_rows} ({numpy.sum(numpy.abs(this_shifts)>0)} affected rows)")
            for irow in range(len(this_shifts)):
                if this_shifts[irow] > 0:
                    image_out._data[irow, :] = numpy.roll(image._data[irow, :], int(this_shifts[irow]))

            if not dry_run:
                log.info(f"writing corrected image to {os.path.basename(out_image)}")
                image_out.writeFitsData(out_image)
                images_out.append(image_out)

            log.info(f"plotting results for {out_image}")
            fig, ax = create_subplots(to_display=display_plots, figsize=(15,7), sharex=True, layout="constrained")
            ax.set_title(f"{mjd = } - {expnum = } - {camera = } - {imagetyp = }", loc="left")
            y_pixels = numpy.arange(this_shifts.size)
            if raw_shifts is not None:
                ax.step(y_pixels, raw_shifts, where="mid", lw=0.5, color="0.9", label="raw DRP")
            ax.step(y_pixels, this_shifts, where="mid", color="k", lw=3)
            if drp_shifts is not None:
                ax.step(y_pixels, drp_shifts, where="mid", lw=1, color="tab:blue", label="DRP")
            if qc_shifts is not None:
                ax.step(y_pixels, qc_shifts, where="mid", lw=2, color="tab:green", label="QC")
            if custom_shifts is not None:
                ax.step(y_pixels, custom_shifts, where="mid", lw=2, color="tab:purple", label="custom shifts")
            ax.legend(loc="lower right", frameon=False)
            ax.set_xlabel("Y (pixel)")
            ax.set_ylabel("Shift (pixel)")
            plot_image_shift(ax, image._data, this_shifts, cmap="Reds")
            axis = plot_image_shift(ax, image_out._data, this_shifts, cmap=image_color, inset_pos=(0.14,1.0-0.32))
            plt.setp(axis, yticklabels=[], ylabel="")
            save_fig(
                fig,
                product_path=out_image,
                to_display=display_plots,
                figure_path="qa",
                label="pixel_shifts"
            )
        else:
            log.info(f"no shifts to apply, not need to write to {os.path.basename(out_image)}")
    return images_out, this_shifts, which_shifts


def select_lines_2d(in_images, out_mask, in_cent_traces, in_waves, lines_list=None, y_widths=3, wave_widths=0.6*5, image_shape=(4080, 4120), channels="brz", display_plots=False):
    """Selects spectral features based on a list of wavelengths from a 2D raw frame

    Given a list of raw frames, centroid traces and wavelength models for a given spectrograph,
    this function selects spectral features based on a list of wavelengths, creating a 2D mask
    with selected lines.

    Parameters
    ----------
    in_images : list
        list of input raw frames for the same spectrograph (brz)
    out_mask : str
        output mask file for the channel stacked frame
    in_cent_traces : list
        list of input centroid trace files for the same spectrograph
    in_waves : list
        list of input wavelength files for the same spectrograph
    lines_list : list, optional
        list of lines to select, by default None
    y_widths : int, optional
        width of the fiber trace, by default 3
    wave_widths : float, optional
        width of the wavelength window, by default 0.6*5
    image_shape : tuple, optional
        shape of the image, by default (4080, 4120)
    channels : str, optional
        spectrograph channels, by default "brz"
    display_plots : bool, optional
        display plots, by default False

    Returns
    -------
    numpy.ndarray
        2D mask with selected lines
    FiberRows
        channel stacked fiber centroid trace data
    FiberRows
        channel stacked wavelength data
    """
    # stack along x-axis traces and wavelengths, adding OS columns
    log.info(f"stacking centroid traces and wavelengths for {','.join(in_cent_traces)} and {','.join(in_waves)}")
    mtraces, mwaves, mtrace, mwave = _channel_combine_fiber_params(in_cent_traces, in_waves)

    # get fiber selection mask
    log.info(f"selecting fibers with {y_widths = } pixel")
    fiber_selection = _get_fiber_selection(mtraces, y_widths=y_widths, image_shape=image_shape)

    # parse lines list based on the image type
    if lines_list is None:
        image = loadImage(in_images[0])
        imagetyp = image._header["IMAGETYP"]
        if imagetyp == "arc":
            lines_list = ",".join([lamp.lower() for lamp in ARC_LAMPS if image._header.get(lamp, False)])
        elif imagetyp == "flat":
            lines_list = "sky"
            wave_widths = 5000
        else:
            lines_list = "sky"
        log.info(f"selecting sources for {imagetyp = } frame: {lines_list}")

    # parse lines list base on the given list
    if isinstance(lines_list, (list, tuple, numpy.ndarray)):
        log.info(f"selecting lines in given list {','.join(lines_list)}")
    elif isinstance(lines_list, str):
        sources = lines_list.split(",")
        lines_list = []
        lamps = list(map(str.lower, ARC_LAMPS))
        for source in sources:
            if source in lamps:
                # define reference lines path
                ref_table = os.path.join(CONFIG_PATH, "wavelength", f"lvm-reflines-{source}.txt")
                log.info(f"loading reference lines from '{ref_table}")
                # skip if missing
                if not os.path.isfile(ref_table):
                    log.warning(f"missing reference lines for {source = }")
                    continue
                table = numpy.genfromtxt(ref_table, usecols=(0, 1), skip_header=1)
                lines_list.append(table[table[:, 0]>=200, 1])
            elif source == "sky":
                ref_table = numpy.genfromtxt(os.path.join(os.getenv('LVMCORE_DIR'), 'etc', 'UVES_sky_lines.txt'), usecols=(1,4))
                lines_list.append(ref_table[ref_table[:, 1] > 2, 0])

        lines_list = numpy.unique(numpy.concatenate(lines_list))
        lines_list.sort()
        waves = mwave._data[mwave._data>0].flatten()
        lines_list = lines_list[(lines_list > waves.min()) & (lines_list < waves.max())]
        log.info(f"selecting lines in given sources {lines_list.tolist()}")

    # get lines selection mask
    log.info(f"selecting lines with {wave_widths = } angstrom")
    lines_mask = _get_wave_selection(mwave._data, lines_list=lines_list, window=wave_widths)
    lines_mask &= ~mwave._mask

    # make sky mask 2d
    log.info("interpolating mask into 2D frame")
    lines_mask_2d = numpy.zeros((image_shape[0], len(in_images)*image_shape[1]), dtype=bool)
    for icol in range(lines_mask_2d.shape[1]):
        for ifiber in range(mwave._data.shape[0]):
            lines_mask_2d[mtrace._data[ifiber, icol].astype(int), icol] = lines_mask[ifiber, icol]

        col = lines_mask_2d[:, icol]
        y = numpy.arange(lines_mask_2d.shape[0])
        if numpy.sum(col) == 0:
            continue
        f = interpolate.interp1d(y[col], col[col], kind="nearest", bounds_error=False, fill_value=0)
        lines_mask_2d[:, icol] = f(y).astype(bool)

    lines_mask_2d &= fiber_selection

    # write mask to file
    log.info(f"writing output mask to {os.path.basename(out_mask)}")
    mask_image = Image(data=lines_mask_2d.astype(int))
    mask_image.writeFitsData(out_mask)

    # plot results
    log.info("plotting results")
    fig, ax = create_subplots(1, 1, figsize=(15, 5))
    ax.imshow(lines_mask_2d, origin="lower", aspect="auto", cmap="binary_r")
    save_fig(
        fig,
        product_path=out_mask,
        to_display=display_plots,
        figure_path="qa",
        label="lines_mask_2d"
    )

    return lines_mask_2d, mtrace, mwave


def fix_pixel_shifts(in_images, out_images, ref_images, in_mask, report=None,
                     max_shift=10, threshold_spikes=0.6, flat_spikes=11,
                     fill_gaps=20, shift_rows=None, interactive=False, display_plots=False):
    """Corrects pixel shifts in raw frames based on reference frames and a selection of spectral regions

    Given a set of raw frames, reference frames and a mask, this function corrects pixel shifts
    based on the reference frames and a selection of spectral regions.

    Parameters
    ----------
    in_images : list
        list of input raw images for the same spectrograph (brz)
    out_images : str
        output pixel shifts file
    ref_images : list
        list of input reference images for the same spectrograph
    in_mask : str
        input mask file for the channel stacked frame
    report : dict, optional
        input report with keys (spec, expnum) and values (shift_rows, amount), by default None
    max_shift : int, optional
        maximum shift in pixels, by default 10
    threshold_spikes : float, optional
        threshold for spike removal, by default 0.6
    flat_spikes : int, optional
        width of the spike removal, by default 11
    fill_gaps : int, optional
        width of the gap filling, by default 20
    interactive : bool, optional
        interactive mode, by default False
    display_plots : bool, optional
        display plots, by default False

    Returns
    -------
    numpy.ndarray
        pixel shifts
    numpy.ndarray
        pixel correlations
    list
        list of corrected images
    """
    mask = loadImage(in_mask)

    log.info(f"loading reference image from {','.join(ref_images)}")
    image_ref = Image()
    image_ref.unsplit([loadImage(ref_image) for ref_image in ref_images])
    cdata = copy(image_ref._data)
    cdata = numpy.nan_to_num(cdata, nan=0) * mask._data

    # read all three detrended images and channel combine them
    log.info(f"loading input image from {','.join(in_images)}")
    image = Image()
    image.unsplit([loadImage(in_image) for in_image in in_images])
    rdata = copy(image._data)
    rdata = numpy.nan_to_num(rdata, nan=0) * mask._data

    # load input images and initialize output images
    images = [loadImage(in_image) for in_image in in_images]
    images_out = images

    # initialize custom shifts
    raw_shifts = None
    dshifts = None
    qshifts = None
    cshifts = None
    apply_shifts = True
    which_shifts = "drp"

    # calculate pixel shifts or use provided ones
    if shift_rows is not None:
        log.info("using user provided pixel shifts")
        cshifts = numpy.zeros(cdata.shape[0])
        for irow in shift_rows:
            cshifts[irow:] += 2
        corrs = numpy.zeros_like(cshifts)
        which_shifts = "custom"
    else:
        log.info("running row-by-row cross-correlation")
        dshifts, corrs = [], []
        for irow in range(rdata.shape[0]):
            cimg_row = cdata[irow]
            rimg_row = rdata[irow]
            if numpy.all(cimg_row == 0) or numpy.all(rimg_row == 0):
                dshifts.append(0)
                corrs.append(0)
                continue

            shift = signal.correlation_lags(cimg_row.size, rimg_row.size, mode="same")
            corr = signal.correlate(cimg_row, rimg_row, mode="same")

            mask = (numpy.abs(shift) <= max_shift)
            shift = shift[mask]
            corr = corr[mask]

            max_corr = numpy.argmax(corr)
            dshifts.append(shift[max_corr])
            corrs.append(corr[max_corr])
        dshifts = numpy.asarray(dshifts)
        corrs = numpy.asarray(corrs)

        dshifts = _remove_spikes(dshifts, width=flat_spikes, threshold=threshold_spikes)
        dshifts = _fillin_valleys(dshifts, width=fill_gaps)
        dshifts = _no_stepdowns(dshifts)

        # parse QC reports with the electronic pixel shifts
        if report is not None:
            shift_rows, amounts = report
            qshifts = numpy.zeros(cdata.shape[0])
            for irow, amount in zip(shift_rows, amounts[::-1]):
                qshifts[irow:] = amount

        # compare QC reports with the electronic pixel shifts
        if qshifts is not None:
            qshifted_rows = numpy.where(numpy.gradient(qshifts) > 0)[0][1::2].tolist()
            shifted_rows = numpy.where(numpy.gradient(dshifts) > 0)[0][1::2].tolist()
            log.info(f"QC reports shifted rows: {qshifted_rows}")
            log.info(f"DRP shifted rows: {shifted_rows}")
            if not numpy.all(qshifts == dshifts):
                _apply_electronic_shifts(images=images, out_images=out_images,
                                         drp_shifts=dshifts, qc_shifts=qshifts, raw_shifts=raw_shifts,
                                         which_shifts="drp", apply_shifts=True,
                                         dry_run=True, display_plots=display_plots)
                log.warning("QC reports and DRP do not agree on the shifted rows")
                [img.add_header_comment("QC reports and DRP do not agree on the shifted rows") for img in images]
                if interactive:
                    log.info("interactive mode enabled")
                    answer = input("apply [q]c, [d]rp or [c]ustom shifts: ")
                    if answer.lower() == "q":
                        log.info("choosing QC shifts")
                        shifts = qshifts
                        which_shifts = "qc"
                    elif answer.lower() == "d":
                        log.info("choosing DRP shifts")
                        shifts = dshifts
                        which_shifts = "drp"
                    elif answer.lower() == "c":
                        log.info("choosing custom shifts")
                        answer = input("provide comma-separated custom shifts and press enter: ")
                        shift_rows = numpy.array([int(_) for _ in answer.split(",")])
                        cshifts = numpy.zeros(cdata.shape[0])
                        for irow in shift_rows:
                            cshifts[irow:] += 2
                        shifts = cshifts
                        corrs = numpy.zeros_like(cshifts)
                        which_shifts = "custom"
                    apply_shifts = numpy.any(numpy.abs(shifts)>0)
                else:
                    log.warning(f"no shift will be applied to the images: {in_images}")
                    [img.add_header_comment("no shift will be applied to the images") for img in images]
                    apply_shifts = False

        elif (dshifts!=0).any() and interactive:
            shifted_rows = numpy.where(numpy.gradient(dshifts) > 0)[0][1::2].tolist()
            log.info(f"DRP shifted rows: {shifted_rows}")
            _apply_electronic_shifts(images=images, out_images=out_images,
                                     drp_shifts=dshifts, qc_shifts=qshifts, raw_shifts=raw_shifts,
                                     which_shifts="drp", apply_shifts=True,
                                     dry_run=True, display_plots=display_plots)
            if interactive:
                log.info("interactive mode enabled")
                answer = input("apply [d]rp, [c]ustom shifts or [n]one: ")
                if answer.lower() == "d":
                    log.info("choosing DRP shifts")
                    shifts = dshifts
                    which_shifts = "drp"
                elif answer.lower() == "c":
                    log.info("choosing custom shifts")
                    answer = input("provide comma-separated custom shifts and press enter: ")
                    shift_rows = numpy.array([int(_) for _ in answer.split(",")])
                    cshifts = numpy.zeros(cdata.shape[0])
                    for irow in shift_rows:
                        cshifts[irow:] += 2
                    shifts = cshifts
                    corrs = numpy.zeros_like(cshifts)
                    which_shifts = "custom"
                elif answer.lower() == "n":
                    log.info("choosing to apply no shift")
                    cshifts = numpy.zeros_like(cdata.shape[0])
                    shifts = cshifts
                    corrs = numpy.zeros_like(cshifts)
                    which_shifts = "custom"
                    apply_shifts = False

                apply_shifts = numpy.any(numpy.abs(shifts)>0)

    # apply pixel shifts to the images
    images_out, shifts, _, = _apply_electronic_shifts(images=images, out_images=out_images, raw_shifts=raw_shifts,
                                                      drp_shifts=dshifts, qc_shifts=qshifts, custom_shifts=cshifts,
                                                      which_shifts=which_shifts, apply_shifts=apply_shifts,
                                                      dry_run=False, display_plots=display_plots)

    return shifts, corrs, images_out


def addCCDMask_drp(image, mask, replaceError="1e10"):
    """
    Adds a mask image (containing only zeros and ones) as new FITS extension to the original image.
    Values of 1 in the mask image are considered as bad pixels. If the image contains already and error image as an
    extension the bad pixels can be replaced in the error array by a user defined value.

    Parameters
    ----------
    image: string
        Name of the FITS file to which the mask should be added
    mask: string
        Name of the FITS file containing the mask of a bad pixel (1 if bad pixel) to be added
    replace_error: strong of float, optional with default: '1e10'
        Error value for bad pixels in a possible error extension, will be ignored if empty

    Examples
    --------
    user:> lvmdrp image addCDDMask IMAGE.fits MASK.fits
    """

    replaceError = float(replaceError)
    img = loadImage(image)
    bad_pixel = loadImage(mask, extension_mask=0)
    if img._mask is not None:
        mask_comb = numpy.logical_or(img._mask, bad_pixel._mask)
    else:
        mask_comb = bad_pixel._mask
    if img._error is not None:
        img.setData(error=replaceError, select=mask_comb)
    img.setData(mask=mask_comb)
    img.writeFitsData(image)


# TODO: en un frame de ciencia colapsar 100 columnas centrales y correr la deteccion de picos
# TODO: comparar con picos del fiberflat, medir shift en las mismas 100 columnas
# TODO: tener en cuenta stretching, no solo cc
# TODO: agregar informacion del fibermap en la salida de esta funcion
# TODO: independientemente de cuantas fibras se detecten, el output tiene que tener todas las fibras + flags
# TODO: agregar informacion de posicion de las fibras al fibermap, para usar como referencia
# esta funcion se corre solo una vez o con frecuencia baja
@skip_on_missing_input_path(["in_image"])
def find_peaks_auto(
    in_image: str,
    out_peaks: str,
    out_region: str = None,
    slice: int = None,
    pixel_range: List[int] = [0, 4080],
    fibers_dmin: int = 5,
    threshold: int = 1.0,
    nfibers: int = None,
    disp_axis: str = "X",
    median_box: List[int] = [1, 10],
    method: str = "hyperbolic",
    init_sigma: float = 1.0,
    display_plots: bool = False,
):
    """
    Finds the exact subpixel cross-dispersion position of a given number of fibers at a certain dispersion column on the raw CCD frame.
    If a predefined number of pixel are expected, the initial threshold value for the minimum peak height will varied until the expected number
    pixels are detected.
    If instead the number of fibers is set to 0, all peaks above the threshold peak height value will be consider as fibers without further iterations.
    The results are stored in an ASCII file for further processing.

    Parameters
    ----------
    image: string
        Name of the Continuum FITS file in which the fiber position along cross-dispersion direction will be measured
    out_peaks : string
        Name of the ASCII file in which the resulting fiber peak positions are stored
    slice: string of integer, optional with default: ''
        Traces the peaks along a given dispersion slice column number. If empty, the dispersion column with the average maximum counts will be used
    pixel_range: string of integer, optional with default: '[0,4080]'
        Defines the range of pixels along the cross-dispersion axis to be considered for the peak finding
    nfibers: string of integer > 0
        Number of fibers for which need to be identified in cross-dispersion
    disp_axies: string of float, optional  with default: 'X'
        Define the dispersion axis, either 'X', 'x', or 0 for the  x axis or 'Y', 'y', or 1 for the y axis.
    threshold: string of float or integer  > 0
        Init threshold for the peak heights to be considered as a fiber peak.
    median_box: string of integer, optional  with default: '8'
        Defines a median smoothing box along dispersion axis to  reduce effects of cosmics or bad pixels
    method: string, optional with default: 'gauss'
        Set the method to measure the peaks positions, either 'gauss' or 'hyperbolic'.
    init_sigma: string of  float, optional with default: '1.0'
        Init guess for the  sigma width (in pixels units)  for the Gaussian fitting, only used if method 'gauss' is selected
    display_plots: string of integer (0 or 1), optional  with default: 1
        Show information during the processing on the command line (0 - no, 1 - yes)

    Examples
    --------
    user:> lvmdrp image findPeaksAuto IMAGE.fits OUT_PEAKS.txt 382  method='gauss', init_sigma=1.3
    """
    # TODO: read fibermap information (with the initial position of the fibers)
    # TODO: flag saturated fibers around those initial positions
    npeaks = nfibers

    # Load Image
    img = loadImage(in_image)

    # swap axes so that the dispersion axis goes along the x axis
    if disp_axis == "X" or disp_axis == "x":
        pass
    elif disp_axis == "Y" or disp_axis == "y":
        img.swapaxes()

    # perform median filtering along the dispersion axis to clean cosmic rays
    if 0 not in median_box:
        img = img.medianImg(median_box)

    # if no slice is given find the cross-dispersion cut with the highest signal
    if slice is None:
        log.info("collapsing image along Y-axis using a median statistic")
        median_cut = img.collapseImg(
            axis="y", mode="median"
        )  # median collapse of image along cross-dispersion axis
        maximum = median_cut.max()  # get maximum value along dispersion axis
        column = maximum[2]  # pixel position of maximum value
        log.info(f"selecting {column = } to locate fibers")
        cut = img.getSlice(column, axis="y")  # extract this column from image
    else:
        column = int(slice)  # convert column to integer value
        log.info(f"selecting {column = } to locate fibers")
        cut = img.getSlice(column, axis="y")  # extract this column from image

    # find location of peaks (local maxima) either above a fixed threshold or to reach a fixed number of peaks
    log.info("locating fibers")
    pixels, _, peaks = cut.findPeaks(
        pix_range=pixel_range, min_dwave=fibers_dmin, threshold=threshold, npeaks=npeaks
    )
    log.info(f"found {len(pixels)} fibers")

    # find the subpixel centroids of the peaks from the central 3 pixels using either a hyperbolic approximation
    # or perform a leastsq fit with a Gaussian
    log.info("refining fiber location")
    centers = cut.measurePeaks(pixels, method, init_sigma, threshold=0, max_diff=1.0)[0]
    # round the subpixel peak positions to their nearest integer value
    round_cent = numpy.round(centers).astype(int)
    log.info(f"final number of fibers found {len(round_cent)}")

    # write number of peaks and their position
    log.info(f"writing {os.path.basename(out_peaks)}")
    columns = [
        pyfits.Column(name="FIBER", format="I", array=numpy.arange(centers.size)),
        pyfits.Column(name="PIXEL", format="I", array=round_cent),
        pyfits.Column(name="SUBPIX", format="D", array=centers),
        pyfits.Column(name="QUALITY", format="I", array=numpy.zeros_like(centers)),
    ]
    table = pyfits.BinTableHDU().from_columns(columns)
    table.header["XPIX"] = (column, "X coordinate of the fibers [pix]")
    hdu_list = pyfits.HDUList([pyfits.PrimaryHDU(header=img._header), table])
    hdu_list.writeto(out_peaks, overwrite=True)
    # write .reg file for ds9
    if out_region is not None:
        with open(out_region, "w") as reg_out:
            reg_out.write("# Region file format: DS9 version 4.1\n")
            reg_out.write(
                'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
            )
            reg_out.write("physical\n")
            for i in range(len(centers)):
                reg_out.write(
                    "# text(%.4f,%.4f) text={%i, %i}\n"
                    % (column+1, centers[i]+1, i + 1, round_cent[i])
                )
        # with open(out_region.replace(".reg", ".txt"), "w") as txt_out:
        #     for i in range(len(centers)-1, -1, -1):
        #         txt_out.write("%i %i\n" % (i + 1, round_cent[i]))

    # plot figure
    fig, ax = create_subplots(to_display=display_plots, figsize=(15, 10))
    ax.step(cut._pixels, cut._data, "-k", lw=1, where="mid")
    ax.plot(pixels, peaks, "o", color="tab:red", mew=0, ms=5)
    ax.plot(
        centers,
        numpy.ones(len(centers)) * bn.nanmax(peaks) * 0.5,
        "x",
        mew=1,
        ms=7,
        color="tab:blue",
    )
    ax.set_xlabel("cross-dispersion axis (pix)")
    ax.set_ylabel("fiber profile")
    save_fig(
        fig,
        product_path=out_peaks,
        to_display=display_plots,
        figure_path="qa",
        label=None,
    )


def findPeaksOffset_drp(
    image,
    peaks_master,
    out_peaks,
    disp_axis="X",
    threshold="1500",
    median_box="8",
    median_cross="1",
    slice="",
    method="gauss",
    init_sigma="1.0",
    accuracy=1.2,
):
    threshold = float(threshold)
    median_box = int(median_box)
    median_cross = int(median_cross)
    init_sigma = float(init_sigma)

    # Load Image
    img = loadImage(image)

    # swap axes so that the dispersion axis goes along the x axis
    if disp_axis == "X" or disp_axis == "x":
        pass
    elif disp_axis == "Y" or disp_axis == "y":
        img.swapaxes()

    # perform median filtering along the dispersion axis to clean cosmic rays
    img = img.medianImg((median_cross, median_box))

    # if no slice is given find the cross-dispersion cut with the highest signal
    if slice == "":
        median_cut = img.collapseImg(
            axis="y", mode="median"
        )  # median collapse of image along cross-dispersion axis
        maximum = median_cut.max()  # get maximum value along dispersion axis
        column = maximum[2]  # pixel position of maximum value
        cut = img.getSlice(column, axis="y")  # extract this column from image
    else:
        column = int(slice)  # convert column to integer value
        cut = img.getSlice(column, axis="y")  # extract this column from image

    master_file = open(peaks_master, "r")
    lines = master_file.readlines()
    fiber = numpy.zeros(len(lines), dtype=numpy.int16)
    pixel = numpy.zeros(len(lines), dtype=numpy.int16)
    ref_pos = numpy.zeros(len(lines), dtype=numpy.float32)
    fib_qual = []
    for i in range(len(lines)):
        line = lines[i].split()
        fiber[i] = int(line[0])
        pixel[i] = int(line[1])
        ref_pos[i] = float(line[2])
        fib_qual.append(line[3])
    fib_qual = numpy.array(fib_qual)

    select_good = fib_qual == "GOOD"
    # npeaks=numpy.sum(select_good)
    # find location of peaks (local maxima) either above a fixed threshold or to reach a fixed number of peaks

    accepted = False
    offset = 0.0
    slope = 0.0

    ref_positions = ref_pos[select_good]
    while accepted is False:
        # if numpy.sum(select_weak)>0:
        #    select = numpy.logical_and(select_good_weak, select_weak)
        #    plt.plot(peaks_weak_good[0][select[select_good_weak]],peaks_weak_good[2] [select[select_good_weak]],'ob')
        peaks = cut.findPeaks(threshold=threshold)
        centers = cut.measurePeaks(
            peaks[0], method, init_sigma, threshold=0, max_diff=1.0
        )[0]
        plt.clf()
        plt.plot(cut._data, "-k")
        plt.plot(peaks[0], peaks[2], "or")
        plt.plot(centers, numpy.ones(len(centers)) * 2000.0, "xg")
        plt.plot(
            ref_positions + (ref_positions - ref_positions[0]) * slope + offset,
            numpy.ones(numpy.sum(select_good)) * 2000.0,
            "+b",
        )
        plt.show()
        print("New Threshold (%.1f):" % (threshold))
        line = sys.stdin.readline()
        try:
            threshold = float(line)
        except (TypeError, ValueError):
            accepted = True
        print("New Offset (%.1f):" % (offset))
        line = sys.stdin.readline()
        try:
            offset = float(line)
        except (TypeError, ValueError):
            pass
        print("New slope (%.1f):" % (slope))
        line = sys.stdin.readline()
        try:
            slope = float(line)
        except (TypeError, ValueError):
            pass

    # expect_first = ref_pos[select_good][0]
    # shift_peaks=peaks_good[0][0]-expect_first
    # if expect_first>=5 and shift_peaks>5:
    #    idx = numpy.indices(cut._data.shape)[0]
    # while ref_pos[select_good][-1]+shift_peaks+2>cut._data.shape[0]:
    #    last_fiber = idx[select_good][-1]

    # npeaks = numpy.sum(select_good)
    # peaks_good = cut.findPeaks(threshold=threshold, npeaks=npeaks)

    # centers = peaks_ref

    # round_cent = numpy.round(centers._data).astype('int16') # round the subpixel peak positions to their nearest integer value
    file_out = open(out_peaks, "w")

    file_out.write("%i\n" % (column))
    for i in range(len(ref_pos)):
        position = (ref_pos[i] - ref_positions[0]) * slope + offset + ref_pos[i]
        if select_good[i]:
            diff_arg = numpy.argmin(numpy.fabs(position - centers))
            diff = position - centers[diff_arg]
            if numpy.fabs(diff) < accuracy:
                file_out.write(
                    "%i %i %e %i\n"
                    % (
                        i + 1,
                        numpy.round(centers[diff_arg]).astype("int16"),
                        centers[diff_arg],
                        0,
                    )
                )
            else:
                file_out.write(
                    "%i %i %e %i\n"
                    % (i + 1, numpy.round(position).astype("int16"), position, 1)
                )
        else:
            file_out.write(
                "%i %i %e %i\n"
                % (i + 1, numpy.round(position).astype("int16"), position, 1)
            )
    file_out.close()


def findPeaksMaster_drp(
    image,
    peaks_master,
    out_peaks,
    disp_axis="X",
    threshold="1500",
    threshold_weak="500",
    median_box="8",
    median_cross="1",
    slice="",
    method="gauss",
    init_sigma="1.0",
    verbose="1",
):
    threshold = float(threshold)
    threshold_weak = float(threshold_weak)
    median_box = int(median_box)
    median_cross = int(median_cross)
    init_sigma = float(init_sigma)
    verbose = int(verbose)

    # Load Image
    img = loadImage(image)

    # swap axes so that the dispersion axis goes along the x axis
    if disp_axis == "X" or disp_axis == "x":
        pass
    elif disp_axis == "Y" or disp_axis == "y":
        img.swapaxes()

    # perform median filtering along the dispersion axis to clean cosmic rays
    img = img.medianImg((median_cross, median_box))

    # if no slice is given find the cross-dispersion cut with the highest signal
    if slice == "":
        median_cut = img.collapseImg(
            axis="y", mode="median"
        )  # median collapse of image along cross-dispersion axis
        maximum = median_cut.max()  # get maximum value along dispersion axis
        column = maximum[2]  # pixel position of maximum value
        cut = img.getSlice(column, axis="y")  # extract this column from image
    else:
        column = int(slice)  # convert column to integer value
        cut = img.getSlice(column, axis="y")  # extract this column from image

    master_file = open(peaks_master, "r")
    lines = master_file.readlines()
    fiber = numpy.zeros(len(lines), dtype=numpy.int16)
    pixel = numpy.zeros(len(lines), dtype=numpy.int16)
    ref_pos = numpy.zeros(len(lines), dtype=numpy.float32)
    fib_qual = []
    for i in range(len(lines)):
        line = lines[i].split()
        fiber[i] = int(line[0])
        pixel[i] = int(line[1])
        ref_pos[i] = float(line[2])
        fib_qual.append(line[3])
    fib_qual = numpy.array(fib_qual)

    select_good = fib_qual == "GOOD"
    npeaks = numpy.sum(select_good)
    # find location of peaks (local maxima) either above a fixed threshold or to reach a fixed number of peaks

    peaks_good = cut.findPeaks(threshold=threshold, npeaks=npeaks)
    expect_first = ref_pos[select_good][0]
    shift_peaks = peaks_good[0][0] - expect_first
    if expect_first >= 5 and shift_peaks > 5:
        idx = numpy.indices(cut._data.shape)[0]
        while ref_pos[select_good][-1] + shift_peaks + 2 > cut._data.shape[0]:
            last_fiber = idx[select_good][-1]
            select_good[last_fiber] = False
    npeaks = numpy.sum(select_good)
    peaks_good = cut.findPeaks(threshold=threshold, npeaks=npeaks)
    centers_good = cut.measurePeaks(
        peaks_good[0], method, init_sigma, threshold=0, max_diff=1.0
    )[0]
    peaks_ref = Spectrum1D(wave=fiber, data=ref_pos)

    shift_spec = Spectrum1D(
        wave=fiber[select_good], data=ref_pos[select_good] - centers_good
    )
    shift_spec.smoothPoly(order=-3, ref_base=fiber)
    centers = peaks_ref - shift_spec
    centers._data[select_good] = centers_good

    select_good_weak = numpy.logical_or(select_good, fib_qual == "WEAK")
    select_weak = fib_qual == "WEAK"
    npeaks = numpy.sum(select_good_weak)
    peaks_weak_good = cut.findPeaks(threshold=threshold_weak, npeaks=npeaks)
    centers_weak_good = cut.measurePeaks(
        peaks_weak_good[0], method, init_sigma, threshold=0, max_diff=1.0
    )[0]
    offset_weak = centers._data[select_good_weak] - centers_weak_good
    select_wrong = numpy.logical_not(
        numpy.logical_and(offset_weak > -0.5, offset_weak < 0.5)
    )
    offset_weak[select_wrong] = 0
    centers._data[select_good_weak] = centers._data[select_good_weak] - offset_weak
    round_cent = numpy.round(centers._data).astype(
        "int16"
    )  # round the subpixel peak positions to their nearest integer value
    file_out = open(out_peaks, "w")
    select_bad = numpy.logical_not(select_good_weak)
    file_out.write("%i\n" % (column))
    for i in range(len(round_cent)):
        file_out.write(
            "%i %i %e %i\n" % (i, round_cent[i], centers._data[i], int(select_bad[i]))
        )
    file_out.close()

    if verbose == 1:
        # control plot for the peaks NEED TO BE REPLACE BY A PROPER VERSION AND POSSIBLE IMPLEMENTAION FOR A GUI
        print("%i Fibers found" % (len(centers._data)))
        plt.plot(cut._data, "-k")
        plt.plot(peaks_good[0], peaks_good[2], "or")
        if numpy.sum(select_weak) > 0:
            select = numpy.logical_and(select_good_weak, select_weak)
            plt.plot(
                peaks_weak_good[0][select[select_good_weak]],
                peaks_weak_good[2][select[select_good_weak]],
                "ob",
            )
        plt.plot(centers._data, numpy.ones(len(centers._data)) * 2000.0, "xg")
        plt.show()


def findPeaksMaster2_drp(
    image,
    peaks_master,
    out_peaks,
    disp_axis="X",
    threshold="1500",
    threshold_weak="500",
    median_box="8",
    median_cross="1",
    slice="",
    method="gauss",
    init_sigma="1.0",
    border="4",
    verbose="1",
):
    threshold = float(threshold)
    threshold_weak = float(threshold_weak)
    border = int(border)
    median_box = int(median_box)
    median_cross = int(median_cross)
    init_sigma = float(init_sigma)
    verbose = int(verbose)

    # Load Image
    img = loadImage(image)

    # swap axes so that the dispersion axis goes along the x axis
    if disp_axis == "X" or disp_axis == "x":
        pass
    elif disp_axis == "Y" or disp_axis == "y":
        img.swapaxes()

    # perform median filtering along the dispersion axis to clean cosmic rays
    img = img.medianImg((median_cross, median_box))

    # if no slice is given find the cross-dispersion cut with the highest signal
    if slice == "":
        median_cut = img.collapseImg(
            axis="y", mode="median"
        )  # median collapse of image along cross-dispersion axis
        maximum = median_cut.max()  # get maximum value along dispersion axis
        column = maximum[2]  # pixel position of maximum value
        cut = img.getSlice(column, axis="y")  # extract this column from image
    else:
        column = int(slice)  # convert column to integer value
        cut = img.getSlice(column, axis="y")  # extract this column from image

    master_file = open(peaks_master, "r")
    lines = master_file.readlines()
    fiber = numpy.zeros(len(lines), dtype=numpy.int16)
    pixel = numpy.zeros(len(lines), dtype=numpy.int16)
    ref_pos = numpy.zeros(len(lines), dtype=numpy.float32)
    fib_qual = []
    for i in range(len(lines)):
        line = lines[i].split()
        # print(line)
        fiber[i] = int(line[0])
        pixel[i] = int(line[1])
        ref_pos[i] = float(line[2])
        fib_qual.append(line[3])
    fib_qual = numpy.array(fib_qual)

    select_good = fib_qual == "GOOD"
    # npeaks = numpy.sum(select_good)
    # find location of peaks (local maxima) either above a fixed threshold or to reach a fixed number of peaks

    peaks_good = []
    if bn.nanmax(cut._data) < threshold:
        threshold = bn.nanmax(cut._data) * 0.8
    while len(peaks_good) != numpy.sum(select_good):
        (peaks_good, temp, peaks_flux) = cut.findPeaks(threshold=threshold, npeaks=0)
        if peaks_good[0] < border:
            peaks_good = peaks_good[1:]
            peaks_flux = peaks_flux[1:]
        if peaks_good[-1] > len(cut._data) - border - 1:
            peaks_good = peaks_good[:-1]
            peaks_flux = peaks_flux[:-1]
        if peaks_good[0] > 10:
            expect_first = ref_pos[select_good][0]
            shift_peaks = peaks_good[0] - expect_first
        elif peaks_good[-1] + 10 <= len(cut._data) - 1:
            expect_last = ref_pos[select_good][-1]
            shift_peaks = peaks_good[-1] - expect_last

        # print peaks_good
        # ref_pos_temp = ref_pos[:] + shift_peaks
        select_good = (
            (fib_qual == "GOOD")
            & (numpy.rint(ref_pos + shift_peaks) > border)
            & (numpy.rint(ref_pos + shift_peaks) < len(cut._data) - border - 1)
        )
        # print (ref_pos+shift_peaks)[select_good],len(cut._data)
        if numpy.sum(select_good) > len(peaks_good):
            threshold = threshold / 1.02
        elif numpy.sum(select_good) < len(peaks_good):
            threshold = threshold * 1.05
            # print(threshold,numpy.sum(select_good),len(peaks_good),shift_peaks)
        # break
    centers_good = cut.measurePeaks(
        peaks_good, method, init_sigma, threshold=0, max_diff=1.0
    )[0]
    peaks_ref = Spectrum1D(wave=fiber, data=ref_pos)

    shift_spec = Spectrum1D(
        wave=fiber[select_good], data=ref_pos[select_good] - centers_good
    )
    shift_spec.smoothPoly(order=-3, ref_base=fiber)
    centers = peaks_ref - shift_spec
    centers._data[select_good] = centers_good

    round_cent = numpy.round(centers._data).astype(
        "int16"
    )  # round the subpixel peak positions to their nearest integer value
    file_out = open(out_peaks, "w")
    select_bad = numpy.logical_not(select_good)
    file_out.write("%i\n" % (column))
    for i in range(len(round_cent)):
        file_out.write(
            "%i %i %e %i\n" % (i, round_cent[i], centers._data[i], int(select_bad[i]))
        )
    file_out.close()

    if verbose == 1:
        # control plot for the peaks NEED TO BE REPLACE BY A PROPER VERSION AND POSSIBLE IMPLEMENTAION FOR A GUI
        print("%i Fibers found" % (len(centers._data)))
        plt.plot(cut._data, "-k")
        plt.plot(peaks_good, peaks_flux, "or")
        plt.plot(centers._data, numpy.ones(len(centers._data)) * 2000.0, "xg")
        plt.show()


# TODO: guardar tabla con los pixeles usados en el trazado (antes del fitting)
# TODO: guardar polinomio evaluado (con buen muestreo)
# TODO: graficar los coeficientes versus los puntos usados en el ajuste polinomial
def trace_peaks(
    in_image: str,
    out_trace: str,
    in_peaks: str = None,
    ref_column: int = 2000,
    correct_ref: bool = False,
    write_trace_data: bool = False,
    disp_axis: str = "X",
    method: str = "gauss",
    median_box: int = 10,
    median_cross: int = 1,
    steps: int = 30,
    coadd: int = 5,
    poly_disp: int = 6,
    init_sigma: float = 1.0,
    threshold: float = 0.5,
    max_diff: int = 1,
    display_plots: bool = True,
):
    """
    Traces the peaks of fibers along the dispersion axis. The peaks at a specific dispersion
    column had to be determined before. Two scheme of measuring the subpixel peak positionare
    available: A hyperbolic approximation or fitting a Gaussian profile to the brightest
    3 pixels of a peak. In both cases the resulting fiber traces along the dispersion axis are
    smoothed by modelling it with a polynomial function.

    Parameters
    ----------
    image: string
        Name of the Continuum exposure FITS file  used to trace the fibre positions
    peaks_file : string
        Name of peaks file containing previously estimated peak position for a certain cross-disperion profile at a specific dispersion column
    trace_out: string
        Name of the  FITS file in which the trace mask will be stored
    disp_axis: string of float, optional  with default: 'X'
        Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
    method: string, optional with default: 'gauss'
        Set the method to measure the peaks positions, either 'gauss' or 'hyperbolic'.
    median_box: string of integer, optional  with default: '7'
        Set a median box size for a median filtering in cross-dispersion direction (reduces artifiacts)
    steps : string of int, optional with default :'30'
        Steps in dispersions direction for which to measure the cross-dispersion fibre positions (saves times)
    coadd: string of integer, optional with default: '30'
        Coadd number of pixels in dispersion direction to increase the S/N of the data
    poly_disp: string of integer, optional with default: '-6'
        Order of the polynomial used to smooth the measured peak position along dispersion axis (positiv: normal polynomial, negative: Legandre polynomial)
    init_sigma: string of float, optional with default: '1.0'
        Initial guess for the width of the Gaussian profiles to measure the peak positions (only used in with method 'gauss')
    threshold: string of float, optional  with default: '100.0'
        Minimum contrast between peak height and the adjacent continuuml counts to be considered as a good measurement
    max_diff: string of float, optional with default: '1.0'
        Maximum difference between the peak position of each fiber in adjacent measurements (steps) along  dispersion direction
    verbose: string of integer (0 or 1), optional  with default: 1
        Show information during the processing on the command line (0 - no, 1 - yes)

    Examples
    --------
    user:> lvmdrp image tracePeaks IMAGE.fits OUT_PEAKS.txt x method=gauss steps=40 coadd=20 smooth_poly=-8
    """

    # load continuum image  from file
    log.info(f"using flat image {os.path.basename(in_image)} for tracing")
    img = loadImage(in_image)
    img.setData(data=numpy.nan_to_num(img._data), error=numpy.nan_to_num(img._error))

    # orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
    if disp_axis == "X" or disp_axis == "x":
        pass
    elif disp_axis == "Y" or disp_axis == "y":
        img.swapaxes()

    dim = img.getDim()
    # perform median filtering along the dispersion axis to clean cosmic rays
    if median_box != 0 or median_cross != 0:
        median_box = max(median_box, 1)
        median_cross = max(median_cross, 1)
        img = img.medianImg((median_cross, median_box))

    # coadd images along the dispersion axis to increase the S/N of the peaks
    if coadd != 0:
        coadd_kernel = numpy.ones(
            (1, coadd), dtype="uint8"
        )  # create convolution kernel array for coaddition
        img = img.convolveImg(coadd_kernel)  # perform convolution to coadd the signal
        threshold = (
            threshold * coadd
        )  # adjust the minimum contrast threshold for the peaks

    # load the initial positions of the fibers at a certain column
    if in_peaks is None:
        # read slitmap extension
        slitmap = img.getSlitmap()
        slitmap = slitmap[slitmap["spectrographid"] == int(img._header["CCD"][1])]

        channel = img._header["CCD"][0]
        positions = slitmap[f"ypix_{channel}"]
        fibers = positions.size

        # correct reference fiber positions
        profile = img.getSlice(ref_column, axis="y")._data
        if correct_ref:
            ypix = numpy.arange(profile.size)
            guess_heights = numpy.ones_like(positions) * bn.nanmax(profile)
            ref_profile = _spec_from_lines(positions, sigma=1.2, wavelength=ypix, heights=guess_heights)
            log.info(f"correcting guess positions for column {ref_column}")
            cc, bhat, mhat = _cross_match(
                ref_spec=ref_profile,
                obs_spec=profile,
                stretch_factors=numpy.linspace(0.7,1.3,5000),
                shift_range=[-100, 100])
            log.info(f"stretch factor: {mhat:.3f}, shift: {bhat:.3f}")
            positions = positions * mhat + bhat
        # set mask
        fibers_status = slitmap["fibstatus"]
        bad_fibers = (fibers_status == 1) | (profile[positions.astype(int)] < threshold)
        good_fibers = numpy.where(numpy.logical_not(bad_fibers))[0]
    else:
        ref_column, _, _, positions, bad_fibers = _read_fiber_ypix(in_peaks)
        fibers = positions.size
        good_fibers = numpy.where(numpy.logical_not(bad_fibers))[0]

    # create empty trace mask for the image
    trace = TraceMask()
    trace.createEmpty(data_dim=(fibers, dim[1]))
    trace.setFibers(fibers)
    trace._good_fibers = good_fibers
    # add the positions of the previous identified peaks
    trace.setSlice(
        ref_column, axis="y", data=positions, mask=numpy.zeros(len(positions), dtype="bool")
    )

    # select cross-dispersion ref_column for the measurements of the peaks
    # TODO: fix this mess with the steps
    first = numpy.arange(ref_column - 1, -1, -1)
    select_first = first % steps == 0
    second = numpy.arange(ref_column + 1, dim[1], 1)
    select_second = second % steps == 0
    # iterate towards index 0 along dispersion axis
    log.info("tracing fibers along dispersion axis")
    iterator = tqdm(
        first[select_first],
        total=select_first.sum(),
        desc=f"tracing fiber left from pixel {ref_column}",
        ascii=True,
        unit="pixel",
    )
    for i in iterator:
        cut_iter = img.getSlice(i, axis="y")  # extract cross-dispersion ref_column
        # infer pixel position of the previous ref_column
        # log.info(f"counter: {i}")
        if i == first[select_first][0]:
            pix = numpy.round(trace.getData()[0][:, ref_column]).astype("int16")
        else:
            pix = numpy.round(trace.getData()[0][:, i + steps]).astype("int16")

        # measure the peaks for the ref_column and store it in the trace
        centers = cut_iter.measurePeaks(pix, method, init_sigma, threshold=threshold, max_diff=float(max_diff))
        trace.setSlice(i, axis="y", data=centers[0], mask=centers[1])

    # iterate towards the last index along dispersion axis
    iterator = tqdm(
        second[select_second],
        total=select_second.sum(),
        desc=f"tracing fiber right from pixel {ref_column}",
        ascii=True,
        unit="pixel",
    )
    for i in iterator:
        cut_iter = img.getSlice(i, axis="y")  # extract cross-dispersion ref_column
        # infer pixel position of the previous ref_column
        if i == second[select_second][0]:
            pix = numpy.round(trace.getData()[0][:, ref_column]).astype("int16")
        else:
            pix = numpy.round(trace.getData()[0][:, i - steps]).astype("int16")

        # measure the peaks for the ref_column and store it in the trace
        centers = cut_iter.measurePeaks(
            pix, method, init_sigma, threshold=threshold, max_diff=float(max_diff)
        )
        trace.setSlice(i, axis="y", data=centers[0], mask=centers[1])

    # define trace data before polynomial smoothing
    trace_data = copy(trace)

    # mask zeros and data outside threshold and max_diff
    trace._mask |= (trace._data <= 0)
    # smooth all trace by a polynomial
    log.info(f"fitting trace with {numpy.abs(poly_disp)}-deg polynomial")
    table, table_poly, table_poly_all = trace.fit_polynomial(poly_disp, poly_kind="poly")
    # set bad fibers in trace mask
    trace._mask[bad_fibers] = True

    if write_trace_data:
        _create_trace_regions(out_trace, table, table_poly, table_poly_all, display_plots=display_plots)

    # linearly interpolate coefficients at masked fibers
    log.info(f"interpolating coefficients at {bad_fibers.sum()} masked fibers")
    x_pixels = numpy.arange(trace._data.shape[1])
    y_pixels = numpy.arange(trace._fibers)
    for column in range(trace._data.shape[1]):
        mask = trace._mask[:, column]
        for order in range(trace._coeffs.shape[1]):
            trace._coeffs[mask, order] = numpy.interp(y_pixels[mask], y_pixels[~mask], trace._coeffs[~mask, order])
    # evaluate trace at interpolated fibers
    for ifiber in y_pixels[bad_fibers]:
        poly = numpy.polynomial.Polynomial(trace._coeffs[ifiber, :])
        trace._data[ifiber, :] = poly(x_pixels)

    # create new header
    log.info(f"writing output trace at {os.path.basename(out_trace)}")
    new_header = img._header.copy()
    new_header["IMAGETYP"] = "trace"
    trace.setHeader(new_header)
    # write trace mask to file
    trace.writeFitsData(out_trace)

    # plot traces and data used in the fitting
    log.info("plotting traces and data used in the fitting")
    pix_ranges = [(0, 300), (1900, 2200), (3700, trace._data.shape[1])]
    fig, axs = plt.subplots(1, len(pix_ranges), figsize=(5 * len(pix_ranges), 10), sharey=True)

    figtitle = os.path.basename(out_trace.replace(".fits", ""))
    fig.suptitle(f"{figtitle}", size="large")

    img.apply_pixelmask()
    fiberflat_data = img._data

    pixels = numpy.arange(trace_data._data.shape[1])
    mask = (trace_data._data != 0).sum(axis=0).astype(bool)
    x = numpy.tile(pixels[mask], trace_data._fibers)
    y = trace_data._data[:, mask].flatten()

    for i, pix_range in enumerate(pix_ranges):
        axs[i].scatter(x, y, c="r", s=10)

        norm = simple_norm(fiberflat_data, stretch="asinh")
        axs[i].imshow(fiberflat_data, norm=norm, origin="lower", cmap="binary_r")

        for ifiber in range(trace._fibers):
            fiber = trace.getSpec(ifiber)
            axs[i].plot(fiber._pixels, fiber._data, color=plt.cm.rainbow(ifiber/trace._fibers), lw=0.5)

        axs[i].set_xlim(*pix_range)
        axs[i].set_ylim(3500, 4100)

    fig.tight_layout()
    save_fig(
        fig,
        product_path=out_trace,
        to_display=display_plots,
        figure_path="qa",
        label="traces",
    )

    return trace_data, trace


def glueCCDFrames_drp(
    images,
    out_image,
    boundary_x,
    boundary_y,
    positions,
    orientation,
    subtract_overscan="1",
    compute_error="1",
    gain="",
    rdnoise="",
):
    """
    Glue CCD subimages of different amplifiers  to a full science CCD images. The orientations of the sub images are taken into account as well as their overscan regions.
    A Poission error image can be automatically computed during this process. This requires that the GAIN and the Read-Out Noise are stored as header keywords in each
    subimage.

    Parameters
    --------------
    images: string
                    Comma-separated names of the FITS images containing the subimage to be combined
    out_image: string
                    Name of the FITS file  in which the combined image will be stored
    boundary_x : string of two comma-separated integers
                    Pixel boundaries of the subimages EXCLUDING the overscan regions along x axis (first pixel has index 1)
    boundary_y : string of two comma-separated integers
                    Pixel boundaries of the subimages EXCLUDING the overscan regionsalong y axis (first pixel has index 1)
    positions : string of two comma-separated  integer digits,
                    Describes the position of each sub image in colum/row format where the first digit describes the row and the second the column position.
                    '00' would correspond to the lower left corner in the combined CCD frame
    orientation: comma-separated strings
                    Describes how each subimage should be oriented before place into the glued CCD frame. Possible options are: 'S','T','X','Y','90','180'', and 270'
                    Their meaning are:
                    'S' : orientation is unchanged
                    'T' : the x and y axes are swapped
                    'X' : mirrored along the x axis
                    'Y' : mirrored along the y axis
                    '90' : rotated by 90 degrees
                    '180' : rotated by 180 degrees
                    '270' : rotated by 270 degrees
    subtract_overscan : string of integer ('0' or '1'), optional  with default: '1'
                    Should the median value of the overscan region be subtracted from the subimage before glueing, '1' - Yes, '0' - No
    compute_error : string of integer ('0' or '1'), optional  with default: '1'
                    Should the Poisson error included into the second extension, '1' - Yes, '0' - No
    gain : string, optional with default :''
                    Name of the FITS Header keyword for the gain value of the CCD, will be multiplied
    rdnoise: string, optional with default: ''
                    Name of the FITS Header keyword for the read out noise value

    Examples
    ----------------
    user:>  lvmdrp image glueCCDFrame FRAME1.fits, FRAME2.fits, FRAME3.fits, FRAME4.fits  FULLFRAME.fits  50,800 1,900  00,10,01,11 X,90,Y,180 gain='GAIN'
    """
    # convert input parameters to proper type
    images = images.split(",")
    bound_x = boundary_x.split(",")
    bound_y = boundary_y.split(",")
    orient = orientation.split(",")
    pos = positions.split(",")
    subtract_overscan = bool(int(subtract_overscan))
    compute_error = bool(int(compute_error))
    # create empty lists
    org_imgs = []  # list of images
    gains = []  # list of gains
    rdnoises = []  # list of read-out noises
    bias = []  # list of biasses

    for i in images:
        # load subimages from disc and append them to a list
        img = loadImage(i, extension_data=0)
        org_imgs.append(img)
        if gain != "":
            # get gain value
            try:
                gains.append(img.getHdrValue(gain))
            except KeyError:
                gains.append(float(gain))
        if rdnoise != "":
            # get read out noise value
            try:
                rdnoises.append(img.getHdrValue(rdnoise))
            except KeyError:
                rdnoises.append(float(rdnoise))
        else:
            rdnoises.append(0.0)

    for i in range(len(images)):
        # append the bias from the overscane region
        bias.append(org_imgs[i].cutOverscan(bound_x, bound_y, subtract_overscan))
        # multiplication with the gain factor
        if gain == "":
            mult = 1.0
        else:
            mult = gains[i]
        org_imgs[i] = org_imgs[i] * mult

        # change orientation of subimages
        org_imgs[i].orientImage(orient[i])
        if compute_error:
            org_imgs[i].computePoissonError(rdnoise=rdnoises[i])

    # create glued image
    full_img = glueImages(org_imgs, pos)

    # adjust FITS header information
    full_img.removeHdrEntries(["GAIN", "RDNOISE", "COMMENT", ""])
    # add gain keywords for the different subimages (CDDs/Amplifies)
    if gain != "":
        for i in range(len(org_imgs)):
            full_img.setHdrValue(
                "HIERARCH AMP%i GAIN" % (i + 1),
                gains[i],
                "Gain value of CCD amplifier %i" % (i + 1),
            )
    # add read-out noise keywords for the different subimages (CDDs/Amplifies)
    if rdnoise != "":
        for i in range(len(org_imgs)):
            full_img.setHdrValue(
                "HIERARCH AMP%i RDNOISE" % (i + 1),
                rdnoises[i],
                "Read-out noise of CCD amplifier %i" % (i + 1),
            )
    # add bias of overscan region for the different subimages (CDDs/Amplifies)
    for i in range(len(org_imgs)):
        if subtract_overscan:
            full_img.setHdrValue(
                "HIERARCH AMP%i OVERSCAN" % (i + 1),
                bias[i],
                "Overscan median (bias) of CCD amplifier %i" % (i + 1),
            )
    ##full_img.setHeader(header=header) # set the modified FITS Header
    # write out FITS file
    if compute_error:
        extension_error = 1
    else:
        extension_error = None
    full_img.writeFitsData(out_image, extension_error=extension_error)


def combineImages_drp(images, out_image, method="median", k="3.0"):
    # convert input parameters to proper type
    images = images.split(",")
    if len(images) == 1:
        file_list = open(images, "r")
        images = file_list.readlines()
        file_list.close()
    k = float(k)
    org_imgs = []
    for i in images:
        # load subimages from disc and append them to a list
        org_imgs.append(loadImage(i.replace("\n", "")))

    combined_img = combineImages(org_imgs, method=method, k=k)
    # write out FITS file
    combined_img.writeFitsData(out_image)


def subtract_straylight(
    in_image: str,
    in_cent_trace: str,
    out_image: str,
    out_stray: str = None,
    select_nrows: int|Tuple[int,int] = 3,
    aperture: int = 14,
    smoothing: int = 20,
    use_weights : bool = False,
    median_box: int = 11,
    gaussian_sigma: int = 20.0,
    parallel: int|str = "auto",
    display_plots: bool = False,
) -> Tuple[Image, Image, Image, Image]:
    """Subtracts a diffuse background signal (stray light) from the raw data

    It uses the regions between fiber to estimate the stray light signal and
    smoothes the result by a polyon in cross-disperion direction and afterwards
    a wide 2D Gaussian filter to reduce the introduction of low frequency
    noise.

    Parameters
    ----------
    in_image: str
        Name of the FITS image from which the stray light should be subtracted
    in_cent_trace: str
        Name of the  FITS file with the trace mask of the fibers
    out_image: str
        Name of the FITS file in which the straylight subtracted image is stored
    out_stray: str
        Name of the FITS file in which the pure straylight image is stored
    select_nrows: int or Tuple[int,int], optional with default: 30
        Number of rows at the top and bottom of the CCD to be used for the stray light estimation
    aperture: int, optional  with default: 14
        Size of the aperture around each fiber in cross-disperion direction assumed to contain signal from fibers
    smoothing: int, optional with default: 20
        Smoothing parameter for the spline fit to the background signal
    use_weights : bool, optional with default: False
        If True, the error of the image is used as weights in the spline fitting
    median_box: int, optional with default: 11
        Width of the median filter used to smooth the image along the dispersion axis
    gaussian_sigma : float, optional with default :20.0
        Width of the 2D Gaussian filter to smooth the measured background signal
    parallel : either int (>0) or  'auto', optional with default: 'auto'
        Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
    display_plots : bool, optional with default: False
        If True, the results are plotted and displayed

    Returns
    -------
    img: Image
        The original image
    img_fit: Image
        The polynomial fit to the background signal
    img_smooth: Image
        The smoothed background signal
    img_out: Image
        The stray light subtracted image
    """
    # load image data
    log.info(f"using image {os.path.basename(in_image)} for stray light subtraction")
    img = loadImage(in_image)
    unit = img._header["BUNIT"]

    # smooth image along dispersion axis with a median filter excluded NaN values
    if median_box is not None:
        log.info(f"median filtering image along dispersion axis with a median filter of width {median_box}")
        median_box = (1, max(1, median_box))
        img_median = img.replaceMaskMedian(*median_box, replace_error=None)
        img_median._data = numpy.nan_to_num(img_median._data)
        img_median = img_median.medianImg(median_box)
    else:
        img_median = copy(img)

    # load trace mask
    log.info(f"using centroids trace {os.path.basename(in_cent_trace)} to mask fibers")
    trace_mask = TraceMask()
    trace_mask.loadFitsData(in_cent_trace, extension_data=0)

    # update mask
    if img_median._mask is None:
        img_median._mask = numpy.zeros(img_median._data.shape, dtype=bool)
    img_median._mask = img_median._mask | numpy.isnan(img_median._data) | numpy.isinf(img_median._data) | (img_median._data == 0)

    # mask regions around each fiber within a given cross-dispersion aperture
    log.info(f"masking fibers with an aperture of {aperture} pixels")
    img_median.maskFiberTraces(trace_mask, aperture=aperture, parallel=parallel)

    # mask regions around the top and bottom of the CCD
    if isinstance(select_nrows, int):
        select_tnrows = select_nrows
        select_bnrows = select_nrows
    else:
        select_tnrows, select_bnrows = select_nrows
    log.info(f"selecting top {select_tnrows} and bottom {select_bnrows} rows of the CCD")
    # define indices for top/bottom fibers
    tfiber = numpy.ceil(trace_mask._data[0]).astype(int)
    bfiber = numpy.floor(trace_mask._data[-1]).astype(int)

    for icol in range(img_median._dim[1]):
        # mask top/bottom rows before/after first/last fiber
        img_median._mask[tfiber[icol]:, icol] = True
        img_median._mask[:bfiber[icol], icol] = True
        # unmask select_nrows around each region
        img_median._mask[(tfiber[icol]+aperture//2):(tfiber[icol]+aperture//2+select_tnrows), icol] = False
        img_median._mask[(bfiber[icol]-aperture//2-select_bnrows):(bfiber[icol]-aperture//2), icol] = False

    # fit the signal in unmaksed areas along cross-dispersion axis by a polynomial
    log.info(f"fitting spline with {smoothing = } to the background signal along cross-dispersion axis")
    img_fit = img_median.fitSpline(smoothing=smoothing, use_weights=use_weights, clip=(0.0, None))

    # median filter to reject outlying columns
    img_fit = img_fit.medianImg((1, 7))

    # smooth the results by 2D Gaussian filter of given width
    log.info(f"smoothing the background signal by a 2D Gaussian filter of width {gaussian_sigma}")
    img_stray = img_fit.convolveGaussImg(1, gaussian_sigma)

    # subtract smoothed background signal from original image
    log.info("subtracting the smoothed background signal from the original image")
    img_out = loadImage(in_image)
    img_out._data = img_out._data - img_stray._data

    # include header and write out file
    log.info(f"writing stray light subtracted image to {os.path.basename(out_image)}")
    img_out.setHeader(header=img.getHeader())
    img_out.writeFitsData(out_image)

    # plot results: polyomial fitting & smoothing, both with masked regions on
    log.info("plotting results")
    fig = plt.figure(figsize=(10, 10), layout="constrained")
    fig.suptitle(f"Stray Light Subtraction for frame {os.path.basename(in_image)}")
    fig.supxlabel("X (pixel)")
    fig.supylabel("Y (pixel)")
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_strayx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_strayy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_strayx.tick_params(axis="x", labelbottom=False)
    ax_strayy.tick_params(axis="y", labelleft=False)
    ax_strayx.set_ylabel(f"Counts ({unit})")
    ax_strayy.set_xlabel(f"Counts ({unit})")
    ax_strayx.set_yscale("asinh")
    ax_strayy.set_xscale("asinh")
    axins1 = inset_axes(ax, width="30%", height="2%", loc="upper right")
    axins1.tick_params(labelsize="small", labelcolor="tab:red")

    y_pixels = numpy.arange(img_median._data.shape[0])
    x_pixels = numpy.arange(img_median._data.shape[1])
    norm = simple_norm(data=img_stray._data, stretch="asinh")
    im = ax.imshow(img_stray._data, origin="lower", cmap="Greys_r", norm=norm, interpolation="none")
    cbar = fig.colorbar(im, cax=axins1, orientation="horizontal")
    cbar.set_label(f"Counts ({unit})", fontsize="small", color="tab:red")
    colors_x = plt.cm.coolwarm(numpy.linspace(0, 1, img_median._data.shape[0]))
    colors_y = plt.cm.coolwarm(numpy.linspace(0, 1, img_median._data.shape[1]))
    ax_strayx.fill_between(x_pixels, bn.nanmedian(img._error, axis=0), lw=0, fc="0.8")
    for iy in y_pixels:
        ax_strayx.plot(x_pixels, img_stray._data[iy], ",", color=colors_x[iy], alpha=0.2)
    ax_strayy.fill_betweenx(y_pixels, 0, bn.nanmedian(img._error, axis=1), lw=0, fc="0.8")
    for ix in x_pixels:
        ax_strayy.plot(img_stray._data[:, ix], y_pixels, ",", color=colors_y[ix], alpha=0.2)
    save_fig(fig, product_path=out_image, to_display=display_plots, figure_path="qa", label="straylight_model")

    # write out stray light image
    if out_stray is not None:
        log.info(f"writing stray light image to {os.path.basename(out_stray)}")
        hdus = pyfits.HDUList()
        hdus.append(pyfits.PrimaryHDU(img._data, header=img._header))
        hdus.append(pyfits.ImageHDU(img_out._data, name="CLEANED"))
        hdus.append(pyfits.ImageHDU(img_median._data, name="MASKED"))
        hdus.append(pyfits.ImageHDU(img_fit._data, name="SPLINE"))
        hdus.append(pyfits.ImageHDU(img_stray._data, name="SMOOTH"))
        hdus.writeto(out_stray, overwrite=True)

        # stray_model = fits.HDUList()
        # stray_model.append(fits.PrimaryHDU(data=img._data, header=img._header))
        # stray_model.append(fits.ImageHDU(data=img_stray._data, name="STRAY_CORR"))
        # stray_model.append(fits.ImageHDU(data=img._data-img_stray._data, name="STRAYLIGHT"))
        # stray_model.append(fits.ImageHDU(data=model._data, name="CONT_MODEL"))
        # stray_model.append(fits.ImageHDU(data=img_stray._data-model._data, name="STRAY_MODEL"))
        # stray_model.append(fits.ImageHDU(data=img._data-model._data, name="NOSTRAY_MODEL"))
        # stray_model.writeto(stray_path, overwrite=True)

    return img_median, img_fit, img_stray, img_out

def traceFWHM_drp(
    in_image,
    in_trace,
    out_fwhm,
    disp_axis="X",
    blocks="20",
    steps="100",
    coadd="10",
    median_box="10",
    median_cross="1",
    poly_disp="5",
    poly_kind="poly",
    threshold_flux="50.0",
    init_fwhm="2.0",
    clip="",
    parallel="auto",
):
    """
    Measures the FWHM of the cross-dispersion fiber profile across the CCD.  It assumes that the profiles have a Gaussian shape and that the width  is CONSTANT for
    a BLOCK of fibers in cross-dispersion direction.  If the FITS image contains an extension with the error, the error frame will be taken into account in the Gaussian fitting.
    To increase the speed only the cross-dispersion profiles at certain position along the dispersion axis with a certain distance (steps)
    in pixels are modelled. The FWHM are then extrapolate by fitting a polynomial of given order along the dispersion axis.

    Parameters
    ----------
    image: string
        Name of the Continuum FITS image from which the fiber profile width should be estimate.
    trace: string
        Name of the  FITS file representing the trace mask of the fibers
    fwhm_out: string
        Name of the FITS file in which the FWHM trace image will be stored
    disp_axis: string of float, optional  with default: 'X'
        Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
    blocks: string of integer, optional  with default: '20'
        Number of fiber blocks that are modelled simultaneously with the same FWHM in cross-dispersion direction.
        The actual number of fibers per block is roughly the total number of fibers divided by the number of blocks.
    steps : string of int, optional with default :'100'
        Steps in dispersions direction columns to measure the cross-dispersion fibre positions
    coadd: string of integer, optional with default: '10'
        Coadd number of pixels in dispersion direction to increase the S/N of the data
    poly_disp: string of integer, optional with default: '5'
        Order of the polynomial used to extrapolate the FWHM  values along dispersion direction for each block
        (positiv: normal polynomial, negativ: Legandre polynomial)
    poly_kind : string, optional with default: 'poly'
        the type of polynomial to use when extrapolating the FWHM
    threshold_flux: sting of float, optional with default: '50.0'
        Minimum integrated counts for a valid fiber per dispersion element
    init_fwhm: string of float, optional with default: '2.0'
        Initial guess of the cross-dispersion fiber FWHM
    clip: string of two comma separated floats, optional with default: ''
        Minimum and maximum number of FWHM in the resulting FWHM trace image. If some value are below or above
        the given limits they are replaced by those limits.
    parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
        Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
        for the given system is used.

    Examples
    --------
    user:> lvmdrp image traceFWHM IMAGE.fits TRACE.fits FWHM.fits x blocks=32 steps=50 poly_disp=20 clip=2,6 parallel=2
    """

    # convert input parameters to proper type
    steps = int(steps)
    blocks = int(blocks)
    poly_disp = int(poly_disp)
    init_fwhm = float(init_fwhm)
    coadd = int(coadd)
    median_box, median_cross = int(median_box), int(median_cross)
    threshold_flux = float(threshold_flux)
    if clip != "":
        clip = clip.split(",")
        clip = [float(clip[0]), float(clip[1])]
    else:
        clip = None

    # load image data
    img = loadImage(in_image)
    img.setData(data=numpy.nan_to_num(img._data), error=numpy.nan_to_num(img._error))

    # median_box, median_cross = 10, 1
    if median_box != 0 or median_cross != 0:
        median_box = max(median_box, 1)
        median_cross = max(median_cross, 1)
        img = img.medianImg((median_cross, median_box))

    img._mask[...] = False

    # plt.figure(figsize=(20,10))
    # plt.plot(img.getSlice(1300, axis="y")._error.tolist(), lw=0.6, color="0.7")
    # plt.plot(img.getSlice(1200, axis="y")._error.tolist(), lw=1)
    # plt.show()
    # return

    # orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
    if disp_axis == "X" or disp_axis == "x":
        pass
    elif disp_axis == "Y" or disp_axis == "y":
        img.swapaxes()
    dim = img.getDim()

    # coadd images along the dispersion axis to increase the S/N of the peaks
    if coadd != 0:
        coadd_kernel = numpy.ones(
            (1, coadd), dtype="uint8"
        )  # create convolution kernel array for coaddition
        img = img.convolveImg(coadd_kernel)  # perform convolution to coadd the signal
        threshold_flux = (
            threshold_flux * coadd
        )  # adjust threshold flux to the coadded signal

    # load trace
    trace_mask = TraceMask()
    trace_mask.loadFitsData(in_trace)

    orig_trace = copy(trace_mask)
    trace_mask._mask[...] = False

    # create a trace mask for the image
    traceFWHM = TraceMask()

    # define the cross-dispersion slices to be modelled with Gaussian profiles
    axis = numpy.arange(dim[1])
    select_steps = axis % steps == 0
    select_steps[-1] = True

    if parallel == "auto":
        fragments = multiprocessing.cpu_count()
    else:
        fragments = int(parallel)
    if fragments > 1:
        split_img = img.split(fragments, axis="X")
        split_trace = trace_mask.split(fragments, axis="X")
        pool = Pool()
        threads = []
        fwhm = []
        mask = []
        select = numpy.array_split(select_steps, fragments)
        for i in range(fragments):
            threads.append(
                pool.apply_async(
                    split_img[i].traceFWHM,
                    (
                        select[i],
                        split_trace[i],
                        blocks,
                        init_fwhm,
                        threshold_flux,
                        dim[0],
                    ),
                )
            )

        for i in range(fragments):
            result = threads[i].get()
            fwhm.append(result[0])
            mask.append(result[1])

        pool.close()
        pool.join()
        traceFWHM = TraceMask(data=numpy.concatenate(fwhm, axis=1), mask=numpy.concatenate(mask, axis=1))
    else:
        fwhm, mask = img.traceFWHM(select_steps, trace_mask, blocks, init_fwhm, threshold_flux, max_pix=dim[0])

    for ifiber in range(orig_trace._fibers):
        if orig_trace._mask[ifiber].all():
            continue
        good_pix = (~mask[ifiber]) & (~numpy.isnan(fwhm[ifiber])) & (fwhm[ifiber] != 0.0) & ((clip[0]<fwhm[ifiber]) & (fwhm[ifiber]<clip[1]))
        f_data = interpolate.interp1d(axis[good_pix], fwhm[ifiber, good_pix], kind="linear", bounds_error=False, fill_value="extrapolate")
        f_mask = interpolate.interp1d(axis[good_pix], mask[ifiber, good_pix], kind="nearest", bounds_error=False, fill_value=0)
        fwhm[ifiber] = f_data(axis)
        mask[ifiber] = f_mask(axis).astype(bool)



    traceFWHM = TraceMask(data=fwhm, mask=mask | orig_trace._mask)

    # smooth the FWHM trace with a polynomial fit along dispersion axis (uncertain pixels are not used)
    # traceFWHM.fit_polynomial(deg=poly_disp, poly_kind=poly_kind, clip=clip)

    # write out FWHM trace to FITS file
    traceFWHM.writeFitsData(out_fwhm)

    return fwhm[:, select_steps], mask[:, select_steps]


def offsetTrace_drp(
    image,
    trace,
    disp,
    lines,
    logfile,
    blocks="15",
    disp_axis="X",
    init_offset="0.0",
    size="20",
):
    """
                Measures the offset in the fiber trace in  cross-dispersion direction in an object raw frame compared to the traces measured from a continuum lamp frame.
                The measurements are stored in a ASCII logfile for futher processing and usage.

                Parameters
                --------------
                image: string
                                Name of the target FITS image which should be test for an offset in the tracing
                trace: string
                                Name of the  RSS FITS file representing the trace mask of the fibers
                disp: string
                                Name of the  RSS FITS file representing containing the wavelength solution for each pixel
                lines: comma separeted string of floats
                                Wavelength sequence of bright lines that can be used to compare the tracing
                logfile: string
                                Name of the output log file in which the measurements are stored in ASCII format
                blocks: string of integer, optional  with default: '20'
                                Number of fiber blocks that are modelled simultaneously with the same FWHM but with a variable offset compare to
                                the original central positions. The actual number of fibers per block is roughly the total number of fibers divided by
                                the number of blocks.
                disp_axis: string of float, optional  with default: 'X'
                                Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
                init_offset: string of float, optional with default: '0.0'
                        Initial guess of the cross-dispersion fiber trace offset in pixels
                size:  string of int (>0), optional with default: '20'
                        Number of pixels being coadd in dispersion direction centered on the wavelength corresponding to the wavelengthes
                        of the input lines

                Examples
                ----------------
    user:> lvmdrp image offsetTrace IMAGE.fits TRACE.fits DISP.fits  blocks=32 size=30
    """
    lines = lines.split(",")
    size = float(size)
    blocks = int(blocks)
    init_offset = float(init_offset)
    img = Image()
    img.loadFitsData(image)
    # orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
    if disp_axis == "X" or disp_axis == "x":
        pass
    elif disp_axis == "Y" or disp_axis == "y":
        img.swapaxes()

    trace_mask = TraceMask()
    trace_mask.loadFitsData(trace)

    dispersion_sol = FiberRows()
    dispersion_sol.loadFitsData(disp)

    # read log file to guess offset position
    try:
        log = open(logfile, "r")
        log_lines = log.readlines()
        i = 0
        # offset_files = []
        while i < len(log_lines):
            if len(log_lines[i].split()) == 1:
                i += 1
                offsets = []
            else:
                offsets.append(
                    bn.nanmedian(
                        numpy.array(log_lines[i + 2].split()[1:]).astype("float32")
                    )
                )
                i += 3
        log.close()
    except IOError:
        offsets = []
        for i in range(len(lines)):
            offsets.append(init_offset)

    log = open(logfile, "a")
    log.write("%s\n" % (image))
    off_trace_all = []
    for i in range(len(lines)):
        wave_line = float(lines[i])
        distance = numpy.abs(dispersion_sol._data - wave_line)
        central_pix = numpy.argmin(distance, 1)
        central_pos = trace_mask._data[numpy.arange(len(central_pix)), central_pix]
        fit = numpy.polyfit(central_pos, central_pix, 4)
        poly = numpy.polyval(fit, numpy.arange(img._data.shape[0]))
        line_pos = numpy.rint(poly).astype("int16")
        collapsed_data = numpy.zeros(len(line_pos), dtype=numpy.float32)

        for j in range(len(line_pos)):
            collapsed_data[j] = numpy.sum(
                img._data[j, line_pos[j] - size : line_pos[j] + size]
            )
        if img._error is not None:
            collapsed_error = numpy.zeros(len(line_pos), dtype=numpy.float32)
            for j in range(len(line_pos)):
                collapsed_error[j] = numpy.sqrt(
                    numpy.sum(
                        img._error[j, line_pos[j] - size : line_pos[j] + size] ** 2
                    )
                )
        else:
            collapsed_error = None
        trace_spec = Spectrum1D(
            wave=numpy.arange(len(collapsed_data)),
            data=collapsed_data,
            error=collapsed_error,
        )
        if trace_mask._mask is not None:
            mask = trace_mask._mask[numpy.arange(len(central_pix)), central_pix]
        else:
            mask = None
        out = trace_spec.measureOffsetPeaks(
            trace_mask._data[numpy.arange(len(central_pix)), central_pix],
            mask,
            blocks,
            init_offset=offsets[i],
            plot=-1,
        )
        off_trace_all.append(out[0])
        string_x = "%.3f" % (wave_line)
        string_y = "%.3f" % (wave_line)
        string_pix = "%.3f" % (wave_line)
        block_line_pos = numpy.array_split(line_pos, blocks)
        for j in range(len(out[0])):
            string_x += " %.3f" % (out[1][j])
            string_y += " %.3f" % (out[0][j])
            string_pix += " %.3f" % (bn.nanmedian(block_line_pos[j]))
        log.write(string_x + "\n")
        log.write(string_pix + "\n")
        log.write(string_y + "\n")
    off_trace_median = bn.nanmedian(numpy.array(off_trace_all))
    off_trace_rms = numpy.std(numpy.array(off_trace_all))
    off_trace_rms = "%.4f" % off_trace_rms if numpy.isfinite(off_trace_rms) else "NAN"
    img.setHdrValue(
        "HIERARCH PIPE FLEX YOFF",
        float("%.4f" % off_trace_median) * -1,
        "flexure offset in y-direction",
    )
    img.setHdrValue(
        "HIERARCH PIPE FLEX YRMS", off_trace_rms, "flexure rms in y-direction"
    )
    img.writeFitsHeader(image)
    log.close()


def offsetTrace2_drp(
    image,
    trace,
    trace_fwhm,
    disp,
    lines,
    logfile,
    blocks="15",
    disp_axis="X",
    min_offset="-2",
    max_offset="2",
    step_offset="0.1",
    size="20",
):
    """
    Measures the offset in the fiber trace in  cross-dispersion direction in an object raw frame compared to the traces measured from a continuum lamp frame.
    The measurements are stored in a ASCII logfile for futher processing and usage.

    Parameters
    --------------
    image: string
                    Name of the target FITS image which should be test for an offset in the tracing
    trace: string
                    Name of the  RSS FITS file representing the trace mask of the fibers
    disp: string
                    Name of the  RSS FITS file representing containing the wavelength solution for each pixel
    lines: comma separeted string of floats
                    Wavelength sequence of bright lines that can be used to compare the tracing
    logfile: string
                    Name of the output log file in which the measurements are stored in ASCII format
    blocks: string of integer, optional  with default: '20'
                    Number of fiber blocks that are modelled simultaneously with the same FWHM but with a variable offset compare to
                    the original central positions. The actual number of fibers per block is roughly the total number of fibers divided by
                    the number of blocks.
    disp_axis: string of float, optional  with default: 'X'
                    Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
    init_offset: string of float, optional with default: '0.0'
            Initial guess of the cross-dispersion fiber trace offset in pixels
    size:  string of int (>0), optional with default: '20'
            Number of pixels being coadd in dispersion direction centered on the wavelength corresponding to the wavelengthes
            of the input lines

    Examples
    ----------------
    user:> lvmdrp image offsetTrace IMAGE.fits TRACE.fits DISP.fits  blocks=32 size=30
    """
    lines = lines.split(",")
    size = int(size)
    blocks = int(blocks)
    min_offset = float(min_offset)
    max_offset = float(max_offset)
    step_offset = float(step_offset)

    img = Image()
    img.loadFitsData(image)
    # orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
    if disp_axis == "X" or disp_axis == "x":
        pass
    elif disp_axis == "Y" or disp_axis == "y":
        img.swapaxes()

    trace_mask = TraceMask()
    trace_mask.loadFitsData(trace)

    trace_fwhm_mask = TraceMask()
    trace_fwhm_mask.loadFitsData(trace_fwhm)

    dispersion_sol = FiberRows()
    dispersion_sol.loadFitsData(disp)

    log = open(logfile, "a")
    log.write("%s\n" % (image))
    off_trace_all = []
    for i in range(len(lines)):
        wave_line = float(lines[i])
        distance = numpy.abs(dispersion_sol._data - wave_line)
        central_pix = numpy.argmin(distance, 1)
        central_pos = trace_mask._data[numpy.arange(len(central_pix)), central_pix]
        central_fwhm = trace_fwhm_mask._data[
            numpy.arange(len(central_pix)), central_pix
        ]
        fit = numpy.polyfit(central_pos, central_pix, 4)
        poly = numpy.polyval(fit, numpy.arange(img._data.shape[0]))
        line_pos = numpy.rint(poly).astype("int16")
        collapsed_data = numpy.zeros(len(line_pos), dtype=numpy.float32)

        for j in range(len(line_pos)):
            collapsed_data[j] = numpy.sum(
                img._data[j, line_pos[j] - size : line_pos[j] + size]
            )
        if img._error is not None:
            collapsed_error = numpy.zeros(len(line_pos), dtype=numpy.float32)
            for j in range(len(line_pos)):
                collapsed_error[j] = numpy.sqrt(
                    numpy.sum(
                        img._error[j, line_pos[j] - size : line_pos[j] + size] ** 2
                    )
                )
        else:
            collapsed_error = None
        trace_spec = Spectrum1D(
            wave=numpy.arange(len(collapsed_data)),
            data=collapsed_data,
            error=collapsed_error,
        )
        if trace_mask._mask is not None:
            mask = trace_mask._mask[numpy.arange(len(central_pix)), central_pix]
        else:
            mask = None
        out = trace_spec.measureOffsetPeaks2(
            central_pos,
            mask,
            central_fwhm,
            blocks,
            min_offset,
            max_offset,
            step_offset,
            plot=-1,
        )
        off_trace_all.append(out[0] * -1)
        string_x = "%.3f" % (wave_line)
        string_y = "%.3f" % (wave_line)
        string_pix = "%.3f" % (wave_line)
        block_line_pos = numpy.array_split(line_pos, blocks)
        for j in range(len(out[0])):
            string_x += " %.3f" % (out[1][j])
            string_y += " %.3f" % (out[0][j] * -1)
            string_pix += " %.3f" % (bn.nanmedian(block_line_pos[j]))
        log.write(string_x + "\n")
        log.write(string_pix + "\n")
        log.write(string_y + "\n")

    off_trace_median = bn.nanmedian(numpy.array(off_trace_all))
    off_trace_rms = numpy.std(numpy.array(off_trace_all))
    img.setHdrValue(
        "HIERARCH PIPE FLEX YOFF",
        float("%.4f" % off_trace_median) * -1,
        "flexure offset in y-direction",
    )
    img.setHdrValue(
        "HIERARCH PIPE FLEX YRMS",
        float("%.4f" % off_trace_rms),
        "flexure rms in y-direction",
    )
    img.writeFitsHeader(image)
    log.close()


# TODO: suggestion from Oleg: test a voigt profile for the flux extraction
# it might be better in dealing with cross-talk
# TODO:
# * define lvm-frame ancillary product to replace for out_rss
@skip_on_missing_input_path(["in_image", "in_trace"])
def extract_spectra(
    in_image: str,
    out_rss: str,
    in_trace: str,
    in_fwhm: str = None,
    in_model: str = None,
    in_acorr: str = None,
    columns: List[int] = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000],
    column_width: int = 50,
    method: str = "optimal",
    aperture: int = 3,
    fwhm: float = 2.5,
    disp_axis: str = "X",
    replace_error: float = 1.0e10,
    display_plots: bool = False,
    parallel: str = "auto",
):
    """
    Extracts the flux for each fiber along the dispersion direction which is written into an RSS FITS file format.
    Either a simple aperture or an optimal extraction scheme may be used.
    In the optimal extraction scheme each cross-dispersion profile is fitted with independent Gaussian for each fiber where
    the position and fwhm is fixed according to the input parameters. This allows for a linear fitting scheme where only the fluxes per fiber
    are the only free parameter.

    Parameters
    --------------
    image: string
                    Name of the Continuum FITS image from which the fiber profile width should be estimate.
    trace: string
                    Name of the  FITS file representing the trace mask of the fibers
    out_rss: string
                    Name of the extracted RSS FITS file
    method: string, optional with default: 'optimal'
                    Available methods are either
                    1. 'optimal': using Gaussian profile fitting to extract the flux. The fwhm parameter needs to be set properly
                    2. 'aperture': simple aperture extraction. The aperture parameter needs to be set as desired
    aperture: string of integer (>0), optional with default: '7'
                    Size of the aperture around the peak position in cross-dispersion direction as used to integrate the flux.
                    Only used if method is set to 'aperture' otherwise this parameter is ignored.
    fwhm: string or string of float, optional with default: '2.5'
                    Set the FWHM in case of the Gaussian profile fitting in optimal method. Either a signle value or the name of a fits file containing
                    a spatially resolved FWHM map is provided. Only used if method is set to 'optimal', otherwise this parameter is ignored.
    disp_axis: string of float, optional  with default: 'X'
                    Define the dispersion axis, either 'X','x', or 0 for the  x axis or 'Y','y', or 1 for the y axis.
    parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
            Number of CPU cores used in parallel for the computation. If set to auto, the maximum number of CPUs
            for the given system is used.

    Examples
    ----------------
    user:> lvmdrp image extractSpec IMAGE.fits TRACE.fits RSS.fits optimal fwhm=FWHM.fits
    """

    if method == 'optimal':
        log.info(f"extracting fiber spectra using fiber profile fitting from {os.path.basename(in_image)}")
    else:
        log.info(f"extraction using aperture of {aperture} pixels")

    img = loadImage(in_image)
    mjd = img._header["SMJD"]
    camera = img._header["CCD"]
    expnum = img._header["EXPOSURE"]


    # orient image so that the cross-dispersion is along the first and the dispersion is along the second array axis
    if disp_axis == "X" or disp_axis == "x":
        pass
    elif disp_axis == "Y" or disp_axis == "y":
        img.swapaxes()

    trace_mask = TraceMask.from_file(in_trace)

    # load fiber model if given
    if in_model is not None and os.path.isfile(in_model):
        fiber_model = loadImage(in_model)
    else:
        fiber_model = None

    shift_range = [-4,4]
    fig = plt.figure(figsize=(15, 3*len(columns)), layout="constrained")
    fig.suptitle(f"Thermal fiber shifts for {mjd = }, {camera = }, {expnum = }")
    gs = GridSpec(len(columns)+1, 15, figure=fig)
    axs_cc, axs_fb = [], []
    for icol in range(len(columns)):
        axs_cc.append(fig.add_subplot(gs[icol, :3], sharex=axs_cc[-1] if icol > 0 else None))
        axs_fb.append(fig.add_subplot(gs[icol, 3:], sharex=axs_fb[-1] if icol > 0 else None, sharey=axs_fb[-1] if icol > 0 else None))

        if icol != len(columns)-1:
            axs_cc[-1].tick_params(labelbottom=False)
            axs_fb[-1].tick_params(labelbottom=False)
    ax_shift = fig.add_subplot(gs[-1:, :])
    axs_cc[0].set_title("Cross-correlation")
    axs_cc[-1].set_xlabel("Shift (pixel)")
    axs_fb[-1].set_xlabel("Y (pixel)")
    axs_cc[-1].set_xlim(shift_range)

    # fix centroids for thermal shifts
    log.info(f"measuring fiber thermal shifts @ columns: {','.join(map(str, columns))}")
    trace_mask, shifts, median_shift, std_shift, _ = _fix_fiber_thermal_shifts(img, trace_mask, 2.5,
                                                                               fiber_model=fiber_model,
                                                                               trace_amp=10000,
                                                                               columns=columns,
                                                                               column_width=column_width,
                                                                               shift_range=shift_range, axs=[axs_cc, axs_fb])
    # save columns measured for thermal shifts
    plot_fiber_thermal_shift(columns, shifts, median_shift, std_shift, ax=ax_shift)
    save_fig(fig, product_path=out_rss, to_display=display_plots, figure_path="qa", label="fiber_thermal_shifts")

    if method == "optimal":
        # check if fwhm trace is given and exists
        if in_fwhm is None or not os.path.isfile(in_fwhm):
            trace_fwhm = TraceMask()
            trace_fwhm.setData(data=numpy.ones(trace_mask._data.shape) * float(fwhm))
            trace_fwhm._coeffs = numpy.ones((trace_mask._data.shape[0], 1)) * float(fwhm)
        else:
            trace_fwhm = TraceMask.from_file(in_fwhm)

        # set up parallel run
        if parallel == "auto":
            fragments = multiprocessing.cpu_count()
        else:
            fragments = int(parallel)

        # run extraction algorithm
        if fragments > 1:
            split_img = img.split(fragments)
            split_trace = trace_mask.split(fragments)
            split_fwhm = trace_fwhm.split(fragments)
            pool = Pool()
            threads = []
            data = []
            error = []
            mask = []
            for i in range(fragments):
                threads.append(
                    pool.apply_async(
                        split_img[i].extractSpecOptimal, (split_trace[i], split_fwhm[i])
                    )
                )
            for i in range(fragments):
                result = threads[i].get()
                data.append(result[0])
                error.append(result[1])
                mask.append(result[2])
            pool.close()
            pool.join()
            data = numpy.concatenate(data, axis=1)
            if error[0] is not None:
                error = numpy.concatenate(error, axis=1)
            else:
                error = None
            if mask[0] is not None:
                mask = numpy.concatenate(mask, axis=1)
            else:
                mask = None
        else:
            (data, error, mask) = img.extractSpecOptimal(
                trace_mask, trace_fwhm, plot_fig=display_plots
            )
    elif method == "aperture":
        trace_fwhm = None

        (data, error, mask) = img.extractSpecAperture(trace_mask, aperture)

        # apply aperture correction given in in_acorr
        if in_acorr is not None and os.path.isfile(in_acorr):
            log.info(f"applying aperture correction in {os.path.basename(in_acorr)}")
            acorr = loadImage(in_acorr)
            data = data * acorr._data
            if error is not None:
                error = error * acorr._data
        else:
            log.warning("no aperture correction applied")
            img.add_header_comment("no aperture correction applied")

    # mask non-exposed standard fibers
    slitmap = img.getSlitmap()
    select_spec = slitmap["spectrographid"] == int(img._header["CCD"][1])
    slitmap_spec = slitmap[select_spec]
    exposed_selection = numpy.array(list(img._header["STD*ACQ"].values()))
    # mask fibers that are not exposed
    # TODO: use the more reliable routine get_exposed_std_fibers once is merged from addqa branch
    if len(exposed_selection) != 0:
        exposed_std = numpy.array(list(img._header["STD*FIB"].values()))[exposed_selection]
        mask |= (~(numpy.isin(slitmap_spec["orig_ifulabel"], exposed_std))&((slitmap_spec["telescope"] == "Spec")))[:, None]
        mask |= (slitmap_spec["fibstatus"] == 1)[:, None]

    # propagate thermal shift to slitmap
    channel = img._header['CCD'][0]
    slitmap[f"ypix_{channel}"] = slitmap[f"ypix_{channel}"].astype("float64")
    slitmap[f"ypix_{channel}"][select_spec] += bn.nanmedian(shifts, axis=0)

    if error is not None:
        error[mask] = replace_error
    rss = RSS(
        data=data,
        mask=mask,
        error=error,
        good_fibers=trace_mask._good_fibers,
        cent_trace=trace_mask,
        width_trace=trace_fwhm,
        header=img.getHeader(),
        slitmap=slitmap
    )
    rss.setHdrValue("NAXIS2", data.shape[0])
    rss.setHdrValue("NAXIS1", data.shape[1])
    rss.setHdrValue("DISPAXIS", 1)
    rss.setHdrValue(
        "HIERARCH FIBER CENT MIN",
        bn.nanmin(trace_mask._data),
    )
    rss.setHdrValue(
        "HIERARCH FIBER CENT MAX",
        bn.nanmax(trace_mask._data),
    )
    rss.setHdrValue(
        "HIERARCH FIBER CENT AVG",
        bn.nanmean(trace_mask._data) if data.size != 0 else 0,
    )
    rss.setHdrValue(
        "HIERARCH FIBER CENT MED",
        bn.nanmedian(trace_mask._data)
        if data.size != 0
        else 0,
    )
    rss.setHdrValue(
        "HIERARCH FIBER CENT SIG",
        bn.nanstd(trace_mask._data) if data.size != 0 else 0,
    )
    if method == "optimal":
        rss.setHdrValue(
            "HIERARCH FIBER WIDTH MIN",
            bn.nanmin(trace_fwhm._data),
        )
        rss.setHdrValue(
            "HIERARCH FIBER WIDTH MAX",
            bn.nanmax(trace_fwhm._data),
        )
        rss.setHdrValue(
            "HIERARCH FIBER WIDTH AVG",
            bn.nanmean(trace_fwhm._data) if data.size != 0 else 0,
        )
        rss.setHdrValue(
            "HIERARCH FIBER WIDTH MED",
            bn.nanmedian(trace_fwhm._data)
            if data.size != 0
            else 0,
        )
        rss.setHdrValue(
            "HIERARCH FIBER WIDTH SIG",
            bn.nanstd(trace_fwhm._data) if data.size != 0 else 0,
        )
    # save extracted RSS
    log.info(f"writing extracted spectra to {os.path.basename(out_rss)}")
    rss.writeFitsData(out_rss)


def calibrateSDSSImage_drp(file_in, file_out, field_file):
    """
                Converts the original SDSS image as retrieved from the DR into a photometrically calibrated image
                in untis of ??. Photometric information are taken from the corresponding SDSS field FITS file.

                Parameters
                --------------
                file_in: string
                                Name of the original SDSS image file
                file_out: string
                                Name of the  photometric calibrated FITS output file in units of ????
                field_file: string
                                Name of the corresponding SDSS field FITS file containing the photometric information

                Examples
                ----------------
    user:> lvmdrp image calibrateSDSSImage fpC-001453-g4-0030.fit.gz SDSS_calib.fits drField-001453-4-40-0030.fit
    """
    image = loadImage(file_in)
    calImage = image.calibrateSDSS(field_file)
    calImage.writeFitsData(file_out)


def subtractBias_drp(
    in_image,
    out_image,
    in_bias,
    compute_error="1",
    boundary_x="",
    boundary_y="",
    gain="",
    rdnoise="",
    subtract_light="0",
):
    subtract_light = bool(int(subtract_light))
    compute_error = bool(int(compute_error))
    image = loadImage(in_image)
    # print('image',image._data)
    bias_frame = loadImage(in_bias)
    # print('bias',bias_frame._data)

    clean = image - bias_frame
    # print('clean',clean._data)

    if gain != "":
        # get gain value
        try:
            gain = image.getHdrValue(gain)
        except KeyError:
            gain = float(gain)
        clean = clean * gain
        # print(clean._dim)

    if rdnoise != "":
        # get gain value
        try:
            rdnoise = image.getHdrValue(rdnoise)
        except KeyError:
            rdnoise = float(rdnoise)

    if compute_error:
        clean.computePoissonError(rdnoise=rdnoise)

    # if boundary_x != "":
    #     bound_x = boundary_x.split(",")
    # else:
    #     bound_x = [1, clean._dim[1]]
    # if boundary_y != "":
    #     bound_y = boundary_y.split(",")
    # else:
    #     bound_y = [1, clean._dim[0]]
    # straylight = clean.cutOverscan(bound_x, bound_y, subtract_light)
    # print(straylight)

    clean.writeFitsData(out_image)


def reprojectRSS_drp(
    stray, trace, fwhm_cross, fwhm_spect, wave, flux, sim_fwhm=0.5, method="linear"
):
    """
    Historic task used for debugging of the the extraction routine...
    """
    # label for outputs
    out_path = os.path.dirname(flux) or "./"
    out_name = os.path.basename(flux).replace(".fits", "")

    # read stray light map
    trace_stray = TraceMask()
    trace_stray.loadFitsData(stray, extension_data=0)
    # read trace
    trace_mask = TraceMask()
    trace_mask.loadFitsData(trace, extension_data=0)
    # read spatial fwhm
    trace_fwhm = TraceMask()
    trace_fwhm.loadFitsData(fwhm_cross, extension_data=0)
    # read spectral fwhm (lsf)
    spect_fwhm = TraceMask()
    spect_fwhm.loadFitsData(fwhm_spect, extension_data=0)
    # read wavelength solution
    trace_wave = TraceMask()
    trace_wave.loadFitsData(wave, extension_data=0)
    # read simulated RSS
    rss_flux = RSS()
    rss_flux.loadFitsData(flux)

    # TODO: implement interpolation in the cross-dispersion direction for:
    # 	- trace_fwhm
    # 	- spect_fwhm
    # 	- trace_wave
    # 	- trace_mask
    if trace_mask._data.shape[1] != rss_flux._data.shape[1]:
        trace_fwhm_res = numpy.zeros(
            (rss_flux._data.shape[0], trace_mask._data.shape[1])
        )
        spect_fwhm_res = numpy.zeros(
            (rss_flux._data.shape[0], trace_mask._data.shape[1])
        )
        trace_wave_res = numpy.zeros(
            (rss_flux._data.shape[0], trace_mask._data.shape[1])
        )
        trace_mask_res = numpy.zeros(
            (rss_flux._data.shape[0], trace_mask._data.shape[1])
        )
        cross_pixel = numpy.arange(trace_mask._data.shape[0])
        cross_pixel_res = numpy.linspace(
            0, trace_mask._data.shape[0] - 1, rss_flux._data.shape[0]
        )
        for i in range(trace_mask._data.shape[1]):
            if method == "spline":
                interp_trace_fwhm = interpolate.UnivariateSpline(
                    cross_pixel, trace_fwhm._data[:, i], s=0
                )
                interp_spect_fwhm = interpolate.UnivariateSpline(
                    cross_pixel, spect_fwhm._data[:, i], s=0
                )
                interp_wave = interpolate.UnivariateSpline(
                    cross_pixel, trace_wave._data[:, i], s=0
                )
                interp_mask = interpolate.UnivariateSpline(
                    cross_pixel, trace_mask._data[:, i], s=0
                )
                trace_fwhm_res[:, i] = interp_trace_fwhm(cross_pixel_res)
                spect_fwhm_res[:, i] = interp_spect_fwhm(cross_pixel_res)
                trace_wave_res[:, i] = interp_wave(cross_pixel_res)
                trace_mask_res[:, i] = interp_mask(cross_pixel_res)
            elif method == "linear":
                interp_trace_fwhm = interpolate.interpolate.interp1d(
                    cross_pixel, trace_fwhm._data[:, i]
                )
                interp_spect_fwhm = interpolate.interpolate.interp1d(
                    cross_pixel, spect_fwhm._data[:, i]
                )
                interp_wave = interpolate.interpolate.interp1d(
                    cross_pixel, trace_wave._data[:, i]
                )
                interp_mask = interpolate.interpolate.interp1d(
                    cross_pixel, trace_mask._data[:, i]
                )
                trace_fwhm_res[:, i] = interp_trace_fwhm(cross_pixel_res)
                spect_fwhm_res[:, i] = interp_spect_fwhm(cross_pixel_res)
                trace_wave_res[:, i] = interp_wave(cross_pixel_res)
                trace_mask_res[:, i] = interp_mask(cross_pixel_res)
            else:
                raise NotImplementedError(
                    f"interpolation method '{method}' not implemented"
                )
    else:
        trace_fwhm_res = trace_fwhm
        spect_fwhm_res = spect_fwhm
        trace_wave_res = trace_wave
        trace_mask_res = trace_mask

    trace_fwhm.setData(data=trace_fwhm_res)
    spect_fwhm.setData(data=spect_fwhm_res)
    trace_wave.setData(data=trace_wave_res)
    trace_mask.setData(data=trace_mask_res)
    # write new trace frames
    trace_fwhm.writeFitsData(filename=f"{out_path}/{out_name}.fwhm.fits")
    spect_fwhm.writeFitsData(filename=f"{out_path}/{out_name}.res.fits")
    trace_wave.writeFitsData(filename=f"{out_path}/{out_name}.disp.fits")
    trace_mask.writeFitsData(filename=f"{out_path}/{out_name}.trc.fits")

    # TODO: convert physical units into electrons
    # 	- read flux calibration factor
    # 	- apply factor to simulated spectra

    rss_flux_out = numpy.zeros((rss_flux._data.shape[0], trace_mask._data.shape[1]))
    for j in range(rss_flux._data.shape[0]):
        # extract the j-spectrum & set the original (simulated) fwhm
        spectrum = rss_flux[j]
        # BUG: resampling should be done after applying LSF to ensure the later is done in the most well-sampled data possible
        # resample to instrumental sampling
        spectrum = spectrum.resampleSpec(trace_wave_res[j], method="spline")
        # degrade spectral resolution to instrumental fwhm
        # BUG: there are cases in which instrumental resolution is better than simulation resolution
        spectrum = spectrum.smoothGaussVariable(
            numpy.sqrt(numpy.abs(spect_fwhm_res[j] ** 2 - sim_fwhm**2))
        )
        # transform to pixel space
        rss_flux_out[j] = spectrum._data

    out_2d = numpy.zeros(trace_stray._data.shape)
    pixel = numpy.arange(spect_fwhm_res.shape[1])
    fact = numpy.sqrt(2.0 * numpy.pi)
    for i in range(trace_mask_res.shape[1]):
        # re-project spectrum using the given instrumental setup
        sigma = trace_fwhm_res[:, i][None, :] / 2.354
        A = numpy.exp(
            -0.5 * ((pixel[:, None] - trace_mask_res[:, i][None, :]) / abs(sigma)) ** 2
        ) / (fact * abs(sigma))
        out_2d[:, i] = numpy.dot(A, rss_flux_out[:, i])

    # add stray light map
    out_2d = out_2d + trace_stray._data
    # TODO: add fiber-to-fiber transmission (fiberflat)
    # TODO: add random poissonian noise (bias+dark)
    # TODO: convert to ADU
    # store re-projected in FITS
    rep = pyfits.PrimaryHDU(out_2d)
    rep.writeto(f"{out_path}/{out_name}_2d.fits", overwrite=True)


def testres_drp(image, trace, fwhm, flux):
    """
    Historic task used for debugging of the the extraction routine...
    """
    img = Image()
    # t1 = time.time()
    img.loadFitsData(image, extension_data=0)
    trace_mask = TraceMask()
    trace_mask.loadFitsData(trace, extension_data=0)
    trace_fwhm = TraceMask()
    #   trace_fwhm.setData(data=numpy.ones(trace_mask._data.shape)*2.5)
    trace_fwhm.loadFitsData(fwhm, extension_data=0)

    trace_flux = TraceMask()
    trace_flux.loadFitsData(flux, extension_data=0)
    x = numpy.arange(img._dim[0])
    out = numpy.zeros(img._dim)
    fact = numpy.sqrt(2.0 * numpy.pi)
    for i in range(img._dim[1]):
        #  print i
        A = (
            1.0
            * numpy.exp(
                -0.5
                * (
                    (x[:, numpy.newaxis] - trace_mask._data[:, i][numpy.newaxis, :])
                    / abs(trace_fwhm._data[:, i][numpy.newaxis, :] / 2.354)
                )
                ** 2
            )
            / (fact * abs(trace_fwhm._data[:, i][numpy.newaxis, :] / 2.354))
        )
        spec = numpy.dot(A, trace_flux._data[:, i])
        out[:, i] = spec
        if i == 1000:
            plt.plot(spec, "-r")
            plt.plot(img._data[:, i], "ok")
            plt.show()

    hdu = pyfits.PrimaryHDU(img._data - out)
    hdu.writeto("res.fits", overwrite=True)
    hdu = pyfits.PrimaryHDU(out)
    hdu.writeto("fit.fits", overwrite=True)

    hdu = pyfits.PrimaryHDU((img._data - out) / img._data)
    hdu.writeto("res_rel.fits", overwrite=True)


# TODO: for arcs take short exposures for bright lines & long exposures for faint lines
# TODO: Argon: 10s + 300s
# TODO: Neon: 10s + 300s
# TODO: HgNe: 15s (particularly helpful for r and NIR strong lines) + 300s (not so many lines in NIR or r)
# TODO: Xenon: 300s
# TODO: correct non-linear region using the PTC
# TODO: Quartz lamp flat-fielding, 10 exptime is fine
@skip_on_missing_input_path(["in_image", "in_mask"])
def preproc_raw_frame(
    in_image: str,
    out_image: str,
    in_mask: str = None,
    assume_imagetyp: str = None,
    assume_trimsec: str = None,
    assume_biassec: str = None,
    assume_gain: list = None,
    assume_rdnoise: list = None,
    gain_prefix: str = "GAIN",
    rdnoise_prefix: str = "RDNOISE",
    subtract_overscan: bool = True,
    overscan_stat: str = "biweight",
    overscan_threshold: float = 3.0,
    overscan_model: str = "spline",
    replace_with_nan: bool = True,
    display_plots: bool = False,
):
    """produces a preprocessed frame given an LVM raw frame

    this taks performs the following steps:

        - identifies and extracts the overscan region, per quadrant
        - identifies extracts the science regions per quadrant
        - optionally subtracts the overscan regions in three possible modes:
            * median value (constant)
            * spline fit
            * polynomial fit
        - computes the saturated pixel mask
        - optionally propagates the "dead" and "hot" pixel mask into the processed frame

    Parameters
    ----------
    in_image : str
        input raw frame path
    out_image : str
        output preprocessed frame path
    in_mask : str, optional
        input pixel mask path, by default None
    assume_imagetyp : str, optional
        whether to assume this image type or use the one in header, by default None
    assume_trimsec : str, optional
        useful data section for each quadrant, by default None
    assume_biassec : str, optional
        overscan section for each quadrant, by default None
    assume_gain : list, optional
        gain values for each quadrant, by default None
    assume_rdnoise : list, optional
        read noise for each quadrant, by default None
    gain_prefix : str, optional
        gain keyword prefix, by default "GAIN"
    rdnoise_prefix : str, optional
        read noise keyword prefix, by default "RDNOISE"
    subtract_overscan : bool, optional
        whether to subtract the overscan for each quadrant or not, by default True
    overscan_stat : str, optional
        statistics to use when coadding pixels along the X axis, by default "biweight"
    overscan_threshold : float, optional
        number of standard deviations to reject pixels in overscan, by default 3.0
    overscan_model : str, optional
        model used to fit the overscan profile of each quadrant, by default "spline"
    replace_with_nan : bool, optional
        whether to replace masked pixels with NaNs or not, by default True
    display_plots : bool, optional
        whether to show plots on display or not, by default False
    """
    # load image
    log.info(f"starting preprocessing of raw image '{os.path.basename(in_image)}'")
    org_img = loadImage(in_image)
    org_header = org_img.getHeader()

    camera = org_header["CCD"]

    # fix the header with header fix file
    # convert real MJD to SJD
    try:
        sjd = int(dateobs_to_sjd(org_header.get("OBSTIME")))
        sjd = correct_sjd(in_image, sjd)
        org_header = apply_hdrfix(sjd, hdr=org_header) or org_header
    except ValueError as e:
        log.error(f"cannot apply header fix: {e}")

    # assume imagetyp or not
    if assume_imagetyp:
        log.warning(f"assuming IMAGETYP = '{assume_imagetyp}'")
        org_img.add_header_comment(f"assuming IMAGETYP = '{assume_imagetyp}'")
        org_header["IMAGETYP"] = assume_imagetyp
    elif "IMAGETYP" not in org_header:
        log.error(
            f"IMAGETYP not found in header. Assuming IMAGETYP = {DEFAULT_IMAGETYP}"
        )
        org_header["IMAGETYP"] = DEFAULT_IMAGETYP
    else:
        log.info(f"using header IMAGETYP = '{org_header['IMAGETYP']}'")

    # extract exptime
    exptime = org_header["EXPTIME"]
    log.info(f"exposure time {exptime} (s)")

    # extract TRIMSEC or assume default value
    if assume_trimsec:
        log.info(f"using given TRIMSEC = {assume_trimsec}")
        sc_sections = [sec for sec in assume_trimsec]
    elif not org_header["TRIMSEC?"]:
        log.warning(f"assuming TRIMSEC = {DEFAULT_TRIMSEC}")
        org_img.add_header_comment(f"assuming default TRIMSEC = {DEFAULT_TRIMSEC}")
        sc_sections = DEFAULT_TRIMSEC
    else:
        sc_sections = list(org_header["TRIMSEC?"].values())
        log.info(f"using header TRIMSEC = {sc_sections}")

    # extract BIASSEC or assume default value
    if assume_biassec:
        log.info(f"using given BIASSEC = {assume_biassec}")
        os_sections = [sec for sec in assume_biassec]
    elif not org_header["BIASSEC?"]:
        log.warning(f"assuming BIASSEC = {DEFAULT_BIASSEC}")
        org_img.add_header_comment(f"assuming default BIASSEC = {DEFAULT_BIASSEC}")
        os_sections = DEFAULT_BIASSEC
    else:
        os_sections = list(org_header["BIASSEC?"].values())
        log.info(f"using header BIASSEC = {os_sections}")

    # extract gain
    gain = numpy.ones(NQUADS)
    if assume_gain:
        log.info(f"using given GAIN = {assume_gain} (e-/ADU)")
        gain = numpy.asarray(assume_gain)
    elif not org_header[f"{gain_prefix}?"]:
        log.warning(f"assuming GAIN = {gain.tolist()} (e-/ADU)")
        org_img.add_header_comment(f"assuming default GAINs = {gain.tolist()}")
    else:
        # gain = numpy.asarray(DEFAULT_GAIN[org_header["CCD"]])
        gain = numpy.asarray([org_header[f"{gain_prefix}{iquad+1}"] for iquad in range(NQUADS)])

        if camera == "b1":
            gain[1] *= 1.036
        if camera == "b2":
            gain[1] *= 1.013
            gain[2] *= 1.011
        if camera == "b3":
            gain[1] *= 1.029
            gain[2] *= 1.012
        if camera == "r1":
            gain[1] *= 1.011
            gain[2] *= 1.027
        if camera == "r2":
            gain[1] *= 1.025
            gain[2] *= 1.017
        if camera == "r3":
            gain[1] *= 1.010
            gain[2] *= 1.020
        if camera == "z1":
            gain[1] *= 1.093
            gain[3] *= 1.063
        if camera == "z2":
            gain[0] *= 1.043
            gain[2] *= 1.089
        if camera == "z3":
            gain[3] /= 1.056

        log.info(f"using header GAIN = {gain.tolist()} (e-/ADU)")

    # initialize overscan stats, quadrants lists and, gains and rnoise
    os_bias_med, os_bias_std = numpy.zeros(NQUADS), numpy.zeros(NQUADS)
    sc_quads, os_quads = [], []
    os_profiles, os_models = [], []
    # process each quadrant
    for i, (sc_xy, os_xy) in enumerate(zip(sc_sections, os_sections)):
        # get overscan and science quadrant & convert to electron
        sc_quad = org_img.getSection(section=sc_xy)
        os_quad = org_img.getSection(section=os_xy)
        # subtract overscan bias from image if requested
        if subtract_overscan:
            if overscan_model not in ["const", "poly", "spline"]:
                log.warning(f"overscan model '{overscan_model}' not implemented, using 'spline'")
                org_img.add_header_comment(f"overscan model '{overscan_model}' not implemented, using 'spline'")
                overscan_model = "spline"
            if overscan_model == "spline":
                os_kwargs = {"nknots": 300}
            elif overscan_model == "poly":
                os_kwargs = {"deg": 9}
            else:
                os_kwargs = {}

            os_data, os_profile, os_model = _model_overscan(os_quad, axis=1, overscan_stat=overscan_stat, threshold=overscan_threshold, **os_kwargs)
            os_quad._data = os_data

            if numpy.isnan(os_data).any():
                os_nmask = numpy.isnan(os_data).sum()
                log.info(f"masked {os_nmask} ({os_nmask/os_data.size*100:.2f}%) pixels in overscan above {overscan_threshold} standard deviations")

            sc_quad = sc_quad - os_model

            os_profiles.append(os_profile)
            os_models.append(os_model)

        # compute overscan stats
        os_bias_med[i] = bn.nanmedian(os_quad._data, axis=None)
        os_bias_std[i] = bn.nanmedian(bn.nanstd(os_quad._data, axis=1), axis=None)
        log.info(
            f"median and standard deviation in OS quadrant {i+1}: "
            f"{os_bias_med[i]:.2f} +/- {os_bias_std[i]:.2f} (ADU)"
        )

        sc_quads.append(sc_quad)
        os_quads.append(os_quad)

    # extract rdnoise
    rdnoise = os_bias_std * gain
    if assume_rdnoise:
        log.info(f"using given RDNOISE = {assume_rdnoise} (e-)")
        rdnoise = numpy.asarray(assume_rdnoise)
    elif not org_header[f"{rdnoise_prefix}?"]:
        log.warning(f"assuming RDNOISE = {rdnoise.tolist()} (e-)")
        org_img.add_header_comment(f"assuming RDNOISE = {rdnoise.tolist()} (e-)")
    else:
        rdnoise = numpy.asarray([org_header[f"{rdnoise_prefix}{iquad+1}"] for iquad in range(NQUADS)])

        log.info(f"using header RDNOISE = {rdnoise.tolist()} (e-)")

    # join images
    QUAD_POSITIONS = ["01", "11", "00", "10"]
    proc_img = glueImages(sc_quads, positions=QUAD_POSITIONS)
    proc_img.setHeader(org_header)
    # update/set unit
    proc_img.setHdrValue("BUNIT", "adu", "physical units of the array values")
    # flip along dispersion axis
    try:
        ccd = org_header["CCD"]
    except KeyError:
        ccd = os.path.basename(in_image).split(".")[0].split("-")[1]
        org_header["CCD"] = ccd
    if ccd.startswith("z") or ccd.startswith("b"):
        log.info("flipping along X-axis")
        proc_img.orientImage("X")
        # NOTE: need to flip per quadrant quantities 1 <--> 2, 3 <--> 4
        gain2, gain1, gain4, gain3 = gain
        gain = [gain1, gain2, gain3, gain4]
        rdnoise2, rdnoise1, rdnoise4, rdnoise3 = rdnoise
        rdnoise = [rdnoise1, rdnoise2, rdnoise3, rdnoise4]

    # update header
    log.info("updating header with per quadrant stats")
    # add amplifier quadrants
    for i in range(NQUADS):
        ysize, xsize = sc_quads[i]._dim
        x, y = int(QUAD_POSITIONS[i][0]), int(QUAD_POSITIONS[i][1])
        proc_img.setHdrValue(
            f"HIERARCH AMP{i+1} TRIMSEC",
            f"[{x*xsize+1}:{xsize*(x+1)}, {y*ysize+1}:{ysize*(y+1)}]",
            f"Region of amp. {i+1}",
        )
    # add gain keywords for the different subimages (CCDs/Amplifiers)
    for i in range(NQUADS):
        proc_img.setHdrValue(
            f"HIERARCH AMP{i+1} {gain_prefix}",
            gain[i],
            f"Gain value of amp. {i+1} [electron/adu]",
        )
    # add read-out noise keywords for the different subimages (CCDs/Amplifiers)
    for i in range(NQUADS):
        proc_img.setHdrValue(
            f"HIERARCH AMP{i+1} {rdnoise_prefix}",
            rdnoise[i],
            f"Read-out noise of amp. {i+1} [electron]",
        )
    # add bias of overscan region for the different subimages (CCDs/Amplifiers)
    for i in range(NQUADS):
        proc_img.setHdrValue(
            f"HIERARCH AMP{i+1} OVERSCAN",
            os_bias_med[i],
            f"Overscan median of amp. {i+1} [adu]",
        )
    # add bias std of overscan region for the different subimages (CCDs/Amplifiers)
    for i in range(NQUADS):
        proc_img.setHdrValue(
            f"HIERARCH AMP{i+1} OVERSCAN_STD",
            os_bias_std[i],
            f"Overscan std of amp. {i+1} [adu]",
        )

    # load master pixel mask
    if in_mask and proc_img._header["IMAGETYP"] not in {"bias", "dark", "pixflat"}:
        log.info(f"loading master pixel mask from {os.path.basename(in_mask)}")
        master_mask = loadImage(in_mask)._mask.astype(bool)
    else:
        master_mask = numpy.zeros_like(proc_img._data, dtype=bool)

    # create pixel mask on the original image
    log.info("building pixel mask")
    proc_img._mask = master_mask
    # convert temp image to ADU for saturated pixel masking
    saturated_mask = proc_img._data >= 2**16
    proc_img._mask |= saturated_mask

    # log number of masked pixels
    nmasked = proc_img._mask.sum()
    log.info(f"{nmasked} ({nmasked / proc_img._mask.size * 100:.2g} %) pixels masked")

    # update masked pixels with NaNs if needed
    if replace_with_nan:
        log.info(f"replacing {nmasked} masked pixels with NaNs")
        proc_img.apply_pixelmask()

    # update data reduction quality flag
    drpqual = QualityFlag(0)
    if saturated_mask.sum() / proc_img._mask.size > 0.01:
        drpqual += "SATURATED"
    proc_img.setHdrValue("DRPQUAL", value=drpqual.value, comment="data reduction quality flag")

    # set drp tag version
    proc_img.setHdrValue("DRPVER", DRPVER, comment='data reduction pipeline software tag')

    # write out FITS file
    log.info(f"writing preprocessed image to {os.path.basename(out_image)}")
    proc_img.writeFitsData(out_image)

    # plot overscan strips along X and Y axes
    log.info("plotting results")
    # show column between ac and bd
    fig, axs = create_subplots(
        display_plots, nrows=2, ncols=1, figsize=(15, 10), sharex=True, sharey=False
    )
    axs[-1].set_xlabel("X (pixel)")
    fig.supylabel("median counts (ADU)")
    fig.suptitle("overscan cut along X-axis", size="xx-large")

    os_ab = glueImages(os_quads[:2], positions=["00", "10"])
    os_cd = glueImages(os_quads[2:], positions=["00", "10"])
    for i, os_quad in enumerate([os_ab, os_cd]):
        plot_strips(
            os_quad,
            axis=0,
            nstrip=1,
            ax=axs[i],
            mu_stat=bn.nanmedian,
            sg_stat=lambda x, axis: bn.nanmedian(numpy.std(x, axis=axis)),
            labels=True,
        )
        os_x, os_y = _parse_ccd_section(list(os_sections)[0])
        axs[i].axvline(os_x[1] - os_x[0], ls="--", color="0.5", lw=1)
        axs[i].set_title(f"overscan for quadrants {['12','34'][i]}", loc="left")
    save_fig(
        fig,
        product_path=out_image,
        to_display=display_plots,
        figure_path="qa",
        label="os_strips_12-34_x",
    )

    # show median counts along Y-axis
    fig, axs = create_subplots(
        to_display=display_plots,
        nrows=2,
        ncols=1,
        figsize=(15, 10),
        sharex=True,
        sharey=False,
    )
    axs[-1].set_xlabel("Y (pixel)")
    fig.supylabel("counts (ADU)")
    fig.suptitle("overscan cut along Y-axis", size="xx-large")
    os_ac = glueImages(os_quads[::2], positions=["00", "01"])
    os_bd = glueImages(os_quads[1::2], positions=["00", "01"])
    for i, os_quad in enumerate([os_ac, os_bd]):
        plot_strips(
            os_quad,
            axis=1,
            nstrip=1,
            ax=axs[i],
            mu_stat=bn.nanmedian,
            sg_stat=lambda x, axis: bn.nanmedian(numpy.std(x, axis=axis)),
            show_individuals=True,
            labels=True,
        )
        os_x, os_y = _parse_ccd_section(list(os_sections)[0])
        axs[i].axvline(os_y[1] - os_y[0], ls="--", color="0.5", lw=1)
        axs[i].set_title(f"overscan for quadrants {['13','24'][i]}", loc="left")
    save_fig(
        fig,
        product_path=out_image,
        to_display=display_plots,
        figure_path="qa",
        label="os_strips_13-24_y",
    )

    # show median counts for all quadrants along Y-axis
    fig, axs = create_subplots(
        to_display=display_plots, nrows=4, ncols=1, figsize=(15, 10), sharex=True
    )
    fig.supxlabel("Y (pixel)")
    fig.supylabel("counts (ADU)")
    fig.suptitle("overscan for all quadrants", size="xx-large")
    for i, os_quad in enumerate(os_quads):
        plot_strips(
            os_quad,
            axis=1,
            nstrip=1,
            ax=axs[i],
            mu_stat=bn.nanmedian,
            sg_stat=lambda x, axis: bn.nanmedian(numpy.std(x, axis=axis)),
            labels=True,
        )
        axs[i].step(
            numpy.arange(os_profiles[i].size), os_profiles[i], color="tab:orange", lw=1
        )
        axs[i].step(numpy.arange(os_profiles[i].size), os_models[i], color="k", lw=1)
        axs[i].axhline(
            bn.nanmedian(os_quad._data.flatten()) + rdnoise[i],
            ls="--",
            color="tab:purple",
            lw=1,
            label=f"median + {rdnoise_prefix}",
        )
        axs[i].set_title(f"quadrant {i+1}", loc="left")
    save_fig(
        fig,
        product_path=out_image,
        to_display=display_plots,
        figure_path="qa",
        label="os_strips",
    )

    return org_img, os_profiles, os_models, proc_img


def add_astrometry(
    in_image: str,
    out_image: str,
    in_agcsci_image: str,
    in_agcskye_image: str,
    in_agcskyw_image: str
):
    """
    uses WCS in AG camera coadd image to calculate RA,DEC of
    each fiber in each telescope and adds these to SLITMAP extension
    if AGC frames are not available it uses the POtelRA,POtelDEC,POtelPA

    Parameters

    in_image : str
        path to input image
    out_image : str
        path to output image
    in_agcsci_image : str
        path to Sci telescope AGC coadd master frame
    in_agcskye_image : str
        path to SkyE telescope AGC coadd master frame
    in_agcskyw_image : str
        path to SkyW telescope AGC coadd master frame
    """

    # print("**************************************")
    # print("**** ADDING ASTROMETRY TO SLITMAP ****")
    # print("**************************************")
    log.info(f"loading frame from {in_image}")
    #print(in_image)
    #print(out_image)
    #print(in_agcsci_image)
    #print(in_agcskye_image)
    #print(in_agcskyw_image)
    #print(in_agcspec_image)

    # reading slitmap
    org_img = loadImage(in_image)
    slitmap = org_img.getSlitmap()
    telescope=numpy.array(slitmap['telescope'].data)
    x=numpy.array(slitmap['xpmm'].data)
    y=numpy.array(slitmap['ypmm'].data)

    # selection mask for fibers from different telescopes
    selsci=(telescope=='Sci')
    selskye=(telescope=='SkyE')
    selskyw=(telescope=='SkyW')
    selspec=(telescope=='Spec')

    # read AGC coadd images and get RAobs, DECobs, and PAobs for each telescope
    agcfiledir={'sci':in_agcsci_image, 'skye':in_agcskye_image, 'skyw':in_agcskyw_image}

    def copy_guider_keyword(gdrhdr, keyword, img):
        '''Copy a keyword from a guider coadd header to an Image object Header'''
        inhdr = keyword in gdrhdr
        comment = gdrhdr.comments[keyword] if inhdr else ''
        img.setHdrValue(f'HIERARCH GDRCOADD {keyword}', gdrhdr.get(keyword), comment)

    def getobsparam(tel):
        if tel!='spec':
            if os.path.isfile(agcfiledir[tel]):
                mfagc=fits.open(agcfiledir[tel])
                mfheader=mfagc[1].header
                outw = wcs.WCS(mfheader)
                CDmatrix=outw.pixel_scale_matrix
                posangrad=-1*numpy.arctan(CDmatrix[1,0]/CDmatrix[0,0])
                PAobs=posangrad*180/numpy.pi
                IFUcencoords=outw.pixel_to_world(2500,1000)
                RAobs=IFUcencoords.ra.value
                DECobs=IFUcencoords.dec.value
                org_img.setHdrValue('ASTRMSRC', 'GDR coadd', comment='Source of astrometric solution: guider')
                copy_guider_keyword(mfheader, 'FRAME0  ', org_img)
                copy_guider_keyword(mfheader, 'FRAMEN  ', org_img)
                copy_guider_keyword(mfheader, 'NFRAMES ', org_img)
                copy_guider_keyword(mfheader, 'STACK0  ', org_img)
                copy_guider_keyword(mfheader, 'STACKN  ', org_img)
                copy_guider_keyword(mfheader, 'NSTACKED', org_img)
                copy_guider_keyword(mfheader, 'COESTIM ', org_img)
                copy_guider_keyword(mfheader, 'SIGCLIP ', org_img)
                copy_guider_keyword(mfheader, 'SIGMA   ', org_img)
                copy_guider_keyword(mfheader, 'OBSTIME0', org_img)
                copy_guider_keyword(mfheader, 'OBSTIMEN', org_img)
                copy_guider_keyword(mfheader, 'FWHM0   ', org_img)
                copy_guider_keyword(mfheader, 'FWHMN   ', org_img)
                copy_guider_keyword(mfheader, 'FWHMMED ', org_img)
                copy_guider_keyword(mfheader, 'COFWHM  ', org_img)
                copy_guider_keyword(mfheader, 'COFWHMST', org_img)
                copy_guider_keyword(mfheader, 'PACOEFFA', org_img)
                copy_guider_keyword(mfheader, 'PACOEFFB', org_img)
                copy_guider_keyword(mfheader, 'PAMIN   ', org_img)
                copy_guider_keyword(mfheader, 'PAMAX   ', org_img)
                copy_guider_keyword(mfheader, 'PADRIFT ', org_img)
                copy_guider_keyword(mfheader, 'ZEROPT  ', org_img)
                copy_guider_keyword(mfheader, 'SOLVED  ', org_img)
                copy_guider_keyword(mfheader, 'WARNPADR', org_img)
                copy_guider_keyword(mfheader, 'WARNTRAN', org_img)
                copy_guider_keyword(mfheader, 'WARNMATC', org_img)
                copy_guider_keyword(mfheader, 'WARNFWHM', org_img)
            else:
                RAobs=org_img._header.get(f'PO{tel}RA'.capitalize(), 0) or 0
                DECobs=org_img._header.get(f'PO{tel}DE'.capitalize(), 0) or 0
                PAobs=org_img._header.get(f'PO{tel}PA'.capitalize(), 0) or 0
                if numpy.any([RAobs, DECobs, PAobs]) == 0:
                    log.warning(f"some astrometry keywords for telescope '{tel}' are missing: {RAobs = }, {DECobs = }, {PAobs = }")
                    org_img.add_header_comment(f"no astromentry keywords '{tel}': {RAobs = }, {DECobs = }, {PAobs = }, using commanded")
                org_img.setHdrValue('ASTRMSRC', 'CMD position', comment='Source of astrometric solution: commanded position')
        else:
            RAobs=0
            DECobs=0
            PAobs=0
        return RAobs, DECobs, PAobs

    RAobs_sci, DECobs_sci, PAobs_sci = getobsparam('sci')
    RAobs_skye, DECobs_skye, PAobs_skye = getobsparam('skye')
    RAobs_skyw, DECobs_skyw, PAobs_skyw = getobsparam('skyw')
    RAobs_spec, DECobs_spec, PAobs_spec = getobsparam('spec')

    # Create fake IFU image WCS object for each telescope focal plane and use it to calculate RA,DEC of each fiber
    telcoordsdir={'sci':(RAobs_sci, DECobs_sci, PAobs_sci), 'skye':(RAobs_skye, DECobs_skye, PAobs_skye), 'skyw':(RAobs_skyw, DECobs_skyw, PAobs_skyw), 'spec':(RAobs_spec, DECobs_spec, PAobs_spec)}
    seldir={'sci':selsci, 'skye':selskye, 'skyw':selskyw, 'spec':selspec}

    RAfib=numpy.zeros(len(slitmap))
    DECfib=numpy.zeros(len(slitmap))

    def getfibradec(tel, platescale):
        RAobs, DECobs, PAobs = telcoordsdir[tel]
        pscale=0.01 # IFU image pixel scale in mm/pix
        skypscale=pscale*platescale/3600 # IFU image pixel scale in deg/pix
        npix=1800 # size of fake IFU image
        w = wcs.WCS(naxis=2) # IFU image wcs object
        w.wcs.crpix = [int(npix/2)+1, int(npix/2)+1]
        posangrad=PAobs*numpy.pi/180
        w.wcs.cd=numpy.array([[skypscale*numpy.cos(posangrad), -1*skypscale*numpy.sin(posangrad)],[-1*skypscale*numpy.sin(posangrad), -1*skypscale*numpy.cos(posangrad)]])
        w.wcs.crval = [RAobs,DECobs]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        # Calculate RA,DEC of each individual fiber
        sel=seldir[tel]
        xfib=x[sel]/pscale+int(npix/2) # pixel x coordinates of fibers
        yfib=y[sel]/pscale+int(npix/2) # pixel y coordinates of fibers
        fibcoords=w.pixel_to_world(xfib,yfib).to_table()
        RAfib[sel]=fibcoords['ra'].degree
        DECfib[sel]=fibcoords['dec'].degree

    log.info(f'Using Fiducial Platescale = {FIDUCIAL_PLATESCALE:.2f} "/mm')
    getfibradec('sci', platescale=FIDUCIAL_PLATESCALE)
    getfibradec('skye', platescale=FIDUCIAL_PLATESCALE)
    getfibradec('skyw', platescale=FIDUCIAL_PLATESCALE)
    getfibradec('spec', platescale=FIDUCIAL_PLATESCALE)

    # add coordinates to slitmap
    slitmap['ra']=RAfib
    slitmap['dec']=DECfib
    org_img._slitmap=slitmap

    log.info(f"writing RA,DEC to slitmap in image '{os.path.basename(out_image)}'")
    org_img.writeFitsData(out_image)




@skip_on_missing_input_path(["in_image"])
# @skip_if_drpqual_flags(["SATURATED"], "in_image")
def detrend_frame(
    in_image: str,
    out_image: str,
    in_bias: str = None,
    in_dark: str = None,
    in_pixelflat: str = None,
    in_nonlinearity: str = None,
    in_slitmap: Table = None,
    convert_to_e: bool = True,
    calculate_error: bool = True,
    replace_with_nan: bool = True,
    reject_cr: bool = True,
    median_box: list = [0, 0],
    display_plots: bool = False,
):
    """detrends input image by subtracting bias, dark and flatfielding

    Parameters
    ----------
    in_image : str
        path to input image
    out_image : str
        path to output detrended image
    in_bias : str, optional
        path to bias frame, by default None
    in_dark : str, optional
        path to dark frame, by default None
    in_pixelflat : str, optional
        path to pixelflat frame, by default None
    in_nonlinearity : str, optional
        path to non-linearity correction table, by default None
    in_slitmap: fits.BinTableHDU, optional
        FITS binary table containing the slitmap to be added to `out_image`, by default None
    calculate_error : bool, optional
        whether to calculate Poisson errors or not, by default True
    replace_with_nan : bool, optional
        whether to replace or not NaN values by zeros, by default True
    reject_cr : bool, optional
        whether to reject or not cosmic rays from detrended image, by default True
    median_box : tuple, optional
        size of the median box to refine pixel mask, by default [0,0]
    display_plots : str, optional
        whether to show plots on display or not, by default False
    """

    # TODO: Normalization of flats. This is for combining them right? Need to make sure median is not dominated by diferences in background.
    # We need bright pixels on fiber cores to be scaled to the same level.
    # TODO: Confirm that dark is not being flat fielded in current logic
    # TODO: What is the difference between "flat" and "flatfield"? Pixel flats should not be pixel flatted but regular flats (dome and twilight) yes.
    org_img = loadImage(in_image)
    exptime = org_img._header["EXPTIME"]
    img_type = org_img._header["IMAGETYP"].lower()
    log.info(
        "target frame parameters: "
        f"MJD = {org_img._header['MJD']}, "
        f"exptime = {exptime}, "
        f"camera = {org_img._header['CCD']}"
    )

    # skip detrending for bias frame
    if img_type == "bias":
        log.info(f"skipping detrending for bias frame: {img_type =}")
        return org_img, None, None, None, None, None

    # read master bias
    if img_type in ["bias"] or (in_bias is None or not os.path.isfile(in_bias)):
        if in_bias and not os.path.isfile(in_bias):
            log.warning(f"master bias '{in_bias}' not found. Using dummy bias")
            org_img.add_header_comment(f"master bias '{in_bias}' not found. Using dummy bias")
        mbias_img = Image(data=numpy.zeros_like(org_img._data))
    else:
        log.info(f"using bias calibration frame '{os.path.basename(in_bias)}'")
        mbias_img = loadImage(in_bias)

    # read master dark
    if img_type in ["bias", "dark"] or (in_dark is None or not os.path.isfile(in_dark)):
        if in_dark and not os.path.isfile(in_dark):
            log.warning(f"master dark '{in_dark}' not found. Using dummy dark")
            org_img.add_header_comment(f"master dark '{in_dark}' not found. Using dummy dark")
        mdark_img = Image(data=numpy.zeros_like(org_img._data))
    else:
        log.info(f"using dark calibration frame '{os.path.basename(in_dark)}'")
        mdark_img = loadImage(in_dark)
        mdark_img = mdark_img / mdark_img._header["EXPTIME"] * exptime

    # read master flat
    if img_type in ["bias", "dark", "pixflat"] or (
        in_pixelflat is None or not os.path.isfile(in_pixelflat)
    ):
        if in_pixelflat and not os.path.isfile(in_pixelflat):
            log.warning(f"master flat '{in_pixelflat}' not found. Using dummy flat")
            org_img.add_header_comment(f"master flat '{in_pixelflat}' not found. Using dummy flat")
        mflat_img = Image(data=numpy.ones_like(org_img._data))
    else:
        log.info(
            f"using pixelflat calibration frame '{os.path.basename(in_pixelflat)}'"
        )
        mflat_img = loadImage(in_pixelflat)

    # bias correct image
    if in_bias:
        log.info("subtracting master bias")
    bcorr_img = org_img - mbias_img

    # read in non_linearity correction table
    if in_nonlinearity is not None:
        ptc_params = numpy.loadtxt(in_nonlinearity, comments="#",
                                 dtype=[("C", "|U2",), ("Q", "|U2",),
                                        ("a1", float,), ("a2", float,),
                                        ("a3", float,)])
    else:
        ptc_params = None

    # convert to electrons if requested
    if convert_to_e:
        # calculate Poisson errors
        log.info("applying gain correction per quadrant")
        for i, quad_sec in enumerate(bcorr_img.getHdrValue("AMP? TRIMSEC").values()):
            log.info(f"processing quadrant {i+1}: {quad_sec}")
            # extract quadrant image
            quad = bcorr_img.getSection(quad_sec)
            # extract quadrant gain and rdnoise values
            gain = quad.getHdrValue(f"AMP{i+1} GAIN")
            rdnoise = quad.getHdrValue(f"AMP{i+1} RDNOISE")

            # non-linearity correction
            gain_map = _nonlinearity_correction(ptc_params, gain, quad, iquad=i+1)
            # gain-correct quadrant
            quad *= gain_map
            # propagate new NaNs to the mask
            quad._mask |= numpy.isnan(quad._data)

            quad.computePoissonError(rdnoise)
            bcorr_img.setSection(section=quad_sec, subimg=quad, inplace=True)
            log.info(f"median error in quadrant {i+1}: {bn.nanmedian(quad._error):.2f} (e-)")

        bcorr_img.setHdrValue("BUNIT", "electron", "physical units of the image")
    else:
        # convert to ADU
        log.info("leaving original ADU units")
        bcorr_img.setHdrValue("BUNIT", "adu", "physical units of the image")

    # complete image detrending
    if in_dark:
        log.info("subtracting master dark")
    elif in_dark and in_pixelflat:
        log.info("subtracting master dark and dividing by master pixelflat")

    detrended_img = (bcorr_img - mdark_img.convertUnit(to=bcorr_img._header["BUNIT"]))
    # NOTE: this is a hack to avoid the error propagation of the division in Image
    detrended_img._data = detrended_img._data / numpy.nan_to_num(mflat_img._data, nan=1.0)

    # propagate pixel mask
    log.info("propagating pixel mask")
    nanpixels = numpy.isnan(detrended_img._data)
    infpixels = numpy.isinf(detrended_img._data)
    detrended_img._mask = numpy.logical_or(org_img._mask, nanpixels)
    detrended_img._mask = numpy.logical_or(detrended_img._mask, infpixels)

    # reject cosmic rays
    if reject_cr:
        log.info("rejecting cosmic rays")
        rdnoise = detrended_img.getHdrValue("AMP1 RDNOISE")
        detrended_img.reject_cosmics(gain=1.0, rdnoise=rdnoise, rlim=1.3, iterations=5, fwhm_gauss=[2.75, 2.75],
                                     replace_box=[10,2], replace_error=1e6, verbose=True, inplace=True)

    # replace masked pixels with NaNs
    if replace_with_nan:
        log.info(f"replacing {detrended_img._mask.sum()} masked pixels with NaNs")
        detrended_img.apply_pixelmask()

    # normalize in case of pixel flat calibration
    # 'pixflat' is the imagetyp that a pixel flat can have
    if img_type == "pixflat":
        flat_array = numpy.ma.masked_array(
            detrended_img._data, mask=detrended_img._mask
        )
        detrended_img = detrended_img / numpy.ma.median(flat_array)

    # add slitmap information if given
    if in_slitmap is not None and detrended_img._header["IMAGETYP"] in {'flat', 'arc', 'object', 'science'}:
        log.info("adding slitmap information")
        detrended_img.setSlitmap(in_slitmap)
    else:
        log.warning("no slitmap information to be added")
        detrended_img.add_header_comment("no slitmap information to be added")

    # save detrended image
    log.info(f"writing detrended image to '{os.path.basename(out_image)}'")
    detrended_img.writeFitsData(out_image)

    # show plots
    log.info("plotting results")
    # detrending process
    fig, axs = create_subplots(
        to_display=display_plots,
        nrows=2,
        ncols=2,
        figsize=(15, 15),
        sharex=True,
        sharey=True,
    )
    plt.subplots_adjust(wspace=0.15, hspace=0.1)
    plot_detrend(ori_image=org_img, det_image=detrended_img, axs=axs, mbias=mbias_img, mdark=mdark_img, labels=True)
    save_fig(
        fig,
        product_path=out_image,
        to_display=display_plots,
        figure_path="qa",
        label="detrending",
    )

    return (
        org_img,
        mbias_img,
        mdark_img,
        mflat_img,
        detrended_img,
    )


@drop_missing_input_paths(["in_images"])
def create_master_frame(in_images: List[str], out_image: str, batch_size: int = 30, force_master: bool = True, master_mjd: int = None):
    """Combines the given calibration frames (bias, dark, or pixelflat) into a
    master calibration frame.

    When only one frame is given and `force_master==True`, it is still flagged
    as master, but a warning will be thrown.

    Parameters
    ----------
    in_images : List[str]
        list of paths to images that are going to be combined into a master frame
    out_image : str
        path to output master frame
    force_master : bool, optional
        whether to force or not creation of master frame, by default True, by default True

    Returns
    -------
    org_ims : List[Image]
        list of original images
    master_img : Image
        master image
    """
    if len(in_images) == 0:
        log.error("skipping master frame calculation, no input images given")
        return
    elif len(in_images) == 1:
        if not force_master:
            log.error(
                f"skipping master frame calculation, {len(in_images)} frame to combine"
            )
            return
        else:
            log.warning("building master frame with one image file")

    log.info(f"input frames: {','.join(in_images)}")

    # select only a maximum of `batch_size` images
    if len(in_images) > batch_size:
        log.info(f"selecting {batch_size} random images")
        in_images = numpy.random.choice(in_images, batch_size, replace=False)
    nexp = len(in_images)

    # load images
    org_imgs, imagetyps = [], []
    for in_image in in_images:
        img = loadImage(in_image)
        imagetyps.append(img._header["IMAGETYP"].lower())
        org_imgs.append(img)

    # check if all images have the same imagetyp
    master_type, counts = numpy.unique(imagetyps, return_counts=True)
    master_type = master_type[numpy.argmax(counts)]
    if numpy.any(master_type != numpy.asarray(imagetyps)):
        log.warning(f"not all imagetyp = {master_type}")

    # combine images
    log.info(f"combining {nexp} frames into master frame")
    if master_type == "bias":
        master_img = combineImages(org_imgs, method="median", normalize=False)
    elif master_type == "dark":
        master_img = combineImages(org_imgs, method="median", normalize=False)
    elif master_type == "pixflat":
        master_img = combineImages(
            [img / bn.nanmedian(img._data) for img in org_imgs],
            method="median",
            normalize=True,
            normalize_percentile=75,
        )
        master_img = master_img / master_img.medianImg(size=21, propagate_error=True)
    elif master_type == "arc":
        master_img = combineImages(
            org_imgs, method="median", normalize=True, normalize_percentile=99
        )
    elif master_type == "flat":
        master_img = combineImages(
            org_imgs, method="median", normalize=True, normalize_percentile=75
        )

    # write output master
    log.info(f"writing master frame to '{os.path.basename(out_image)}'")
    if master_mjd is not None:
        master_img._header["MJD"] = master_mjd
    master_img.writeFitsData(out_image)

    return org_imgs, master_img


# @skip_on_missing_input_path(["in_bias", "in_dark", "in_pixelflat"])
# @skip_if_drpqual_flags(["SATURATED"], "in_bias")
# @skip_if_drpqual_flags(["SATURATED"], "in_dark")
# @skip_if_drpqual_flags(["SATURATED"], "in_pixelflat")
# def create_pixelmask(
#     in_bias: str,
#     in_dark: str,
#     out_mask: str,
#     in_pixelflat: str = None,
#     median_box: int = [31, 31],
#     cen_stat: str = "median",
#     low_nsigma: int = 3,
#     high_nsigma: int = 7,
#     column_threshold: float = 0.3,
# ):
#     """create a pixel mask using a simple sigma clipping

#     Given a bias, dark, and pixelflat image, this function will calculate a
#     a pixel mask by performing the following steps:
#         * smooth images with a median filter set by `median_box`
#         * subtract smoothed images from original images
#         * calculate a sigma clipping mask using `cen_stat` and `low_/high_nsigma`
#         * mask whole column if fraction of masked pixels is above `column_threshold`
#         * combine all masks into a single mask

#     By using a low threshold we should be able to pick up weak bad columns, while the
#     high threshold should be able to pick up hot pixels.

#     Parameters
#     ----------
#     in_image : str
#         input image from which the pixel mask will be created
#     out_image : str
#         output image where the resulting pixel mask will be stored
#     cen_stat : str, optional
#         central statistic to use when sigma-clipping, by default "median"
#     nstd : int, optional
#         number of sigmas above which a pixel will be masked, by default 3
#     """
#     # verify of pixelflat exists, ignore if not
#     if in_pixelflat is not None and not os.path.isfile(in_pixelflat):
#         log.warning(f"pixel flat at '{in_pixelflat}' not found, ignoring")
#         in_pixelflat = None

#     imgs, med_imgs, masks = [], [], []
#     for in_image in filter(lambda i: i is not None, [in_bias, in_dark, in_pixelflat]):
#         img = loadImage(in_image)

#         log.info(f"creating pixel mask using '{os.path.basename(in_image)}'")

#         # define pixelmask image
#         mask = Image(data=numpy.ones_like(img._data), mask=numpy.zeros_like(img._data, dtype=bool))

#         quad_sections = img.getHdrValue("AMP? TRIMSEC").values()
#         for sec in quad_sections:
#             log.info(f"processing quadrant = {sec}")
#             quad = img.getSection(sec)
#             msk_quad = mask.getSection(sec)

#             # create a smoothed image using a median rolling box
#             log.info(f"smoothing image with median box {median_box = }")
#             med_quad = quad.medianImg(size=median_box)
#             # subtract that smoothed image from the master dark
#             quad = quad - med_quad

#             # calculate central value
#             # log.info(f"calculating central value using {cen_stat = }")
#             if cen_stat == "mean":
#                 cen = bn.nanmean(quad._data)
#             elif cen_stat == "median":
#                 cen = bn.nanmedian(quad._data)
#             log.info(f"central value = {cen = }")

#             # calculate standard deviation
#             # log.info("calculating standard deviation using biweight_scale")
#             std = biweight_scale(quad._data, M=cen, ignore_nan=True)
#             log.info(f"standard deviation = {std}")

#             # create pixel masks for low and high nsigmas
#             # log.info(f"creating pixel mask for {low_nsigma = } sigma")
#             badcol_mask = (quad._data < cen - low_nsigma * std)
#             badcol_mask |= (quad._data > cen + low_nsigma * std)
#             # log.info(f"creating pixel mask for {high_nsigma = } sigma")
#             if img._header["IMAGETYP"] != "bias":
#                 hotpix_mask = (quad._data < cen - high_nsigma * std)
#                 hotpix_mask |= (quad._data > cen + high_nsigma * std)
#             else:
#                 hotpix_mask = numpy.zeros_like(quad._data, dtype=bool)

#             # mask whole columns if fraction of masked pixels is above threshold
#             bad_columns = numpy.sum(badcol_mask, axis=0) > column_threshold * quad._dim[0]
#             log.info(f"masking {bad_columns.sum()} bad columns")
#             # reset mask to clean good pixels
#             badcol_mask[...] = False
#             # mask only bad columns
#             badcol_mask[:, bad_columns] = True

#             # combine bad column and hot pixel masks
#             msk_quad._mask = badcol_mask | hotpix_mask
#             log.info(f"masking {msk_quad._mask.sum()} pixels in total")

#             # set section to pixelmask image
#             mask.setSection(section=sec, subimg=msk_quad, inplace=True)

#         imgs.append(img)
#         masks.append(mask)

#     # define header for pixel mask
#     new_header = img._header
#     new_header["IMAGETYP"] = "pixmask"
#     new_header["EXPTIME"] = 0
#     new_header["DARKTIME"] = 0
#     # define image object to store pixel mask
#     new_mask = Image(data=mask._data, mask=numpy.any([mask._mask for mask in masks], axis=0), header=new_header)
#     new_mask.apply_pixelmask()
#     log.info(f"writing pixel mask to '{os.path.basename(out_mask)}'")
#     new_mask.writeFitsData(out_mask)

#     return imgs, med_imgs, masks


def create_pixelmask(in_short_dark, in_long_dark, out_pixmask, in_flat_a=None, in_flat_b=None,
                     dc_low=1.0, dc_high=10, dark_low=0.8, dark_high=1.2,
                     flat_low=0.2, flat_high=1.1, display_plots=False):
    """create a pixel mask using a simple sigma clipping

    Given a long and short dark image, this function will calculate a
    a pixel mask by performing the following steps:
        * calculate dark current
        * calculate ratio of darks
        * mask pixels with dark current below `dc_low`
        * mask pixels with ratio below `dark_low` or above `dark_high`

    Parameters
    ----------
    in_short_dark : str
        path to short dark image
    in_long_dark : str
        path to long dark image
    out_pixmask : str
        path to output pixel mask
    in_flat_a : str, optional
        path to flat A image, by default None
    in_flat_b : str, optional
        path to flat B image, by default None
    dc_low : float, optional
        reliable dark current threshold, by default 1.0
    dc_high : float, optional
        high dark current threshold, by default 10
    dark_low : float, optional
        lower ratio threshold, by default 0.8
    dark_high : float, optional
        upper ratio threshold, by default 1.2
    flat_low : float, optional
        lower flat threshold, by default 0.2
    flat_high : float, optional
        upper flat threshold, by default 1.1
    display_plots : bool, optional
        whether to show plots on display or not, by default False

    Returns
    -------
    pixmask : Image
        pixelmask image
    ratio_dark : Image
        ratio of darks image
    ratio_flat : Image
        ratio of flats image, None if no flats were given
    """

    # define dark current unit
    unit = "electron/s"

    # load short/long dark and convert to dark current
    log.info(f"loading short dark '{os.path.basename(in_short_dark)}'")
    short_dark = loadImage(in_short_dark).convertUnit(unit)
    log.info(f"loading long dark '{os.path.basename(in_long_dark)}'")
    long_dark = loadImage(in_long_dark).convertUnit(unit)

    if in_flat_a is not None:
        log.info(f"loading flat A '{os.path.basename(in_flat_a)}'")
        flat_a = loadImage(in_flat_a)
    else:
        flat_a = None
    if in_flat_b is not None:
        log.info(f"loading flat B '{os.path.basename(in_flat_b)}'")
        flat_b = loadImage(in_flat_b)
    else:
        flat_b = None

    # define exposure times
    short_exptime = short_dark._header["EXPTIME"]
    long_exptime = long_dark._header["EXPTIME"]
    log.info(f"short exposure time = {short_exptime}s")
    log.info(f"long exposure time = {long_exptime}s")

    # define ratio of darks
    ratio_dark = short_dark / long_dark

    # define quadrant sections
    sections = short_dark.getHdrValue("AMP? TRIMSEC")

    # define pixelmask image
    pixmask = Image(data=numpy.zeros_like(short_dark._data), mask=numpy.zeros_like(short_dark._data, dtype="bool"))

    # create histogram figure
    fig_dis, axs_dis = create_subplots(to_display=display_plots, nrows=2, ncols=2, figsize=(15,10), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    fig_dis.suptitle(os.path.basename(out_pixmask))
    # create ratio figure
    fig_rat, axs_rat = create_subplots(to_display=display_plots, nrows=2, ncols=2, figsize=(15, 10), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    fig_rat.suptitle(os.path.basename(out_pixmask))

    log.info(f"creating pixel mask using dark current threshold = {dc_low} {unit}")
    for iquad, section in enumerate(sections.values()):
        log.info(f"processing quadrant = {section}")
        # get sections
        squad = short_dark.getSection(section)
        lquad = long_dark.getSection(section)
        rquad = ratio_dark.getSection(section)
        mquad = pixmask.getSection(section)

        # define good current mask
        good_dc = (lquad._data > dc_low)
        log.info(f"selecting {good_dc.sum()} pixels with dark current > {dc_low} {unit}")
        # define quadrant pixelmask
        mask_hotpix = (squad._data > dc_high) | ((dark_low >= rquad._data) | (rquad._data >= dark_high))
        # set quadrant pixelmask
        log.info(f"masking {(good_dc & mask_hotpix).sum()} pixels with DC ratio < {dark_low} or > {dark_high}")
        mquad.setData(mask=good_dc & mask_hotpix)
        pixmask.setSection(section, mquad, inplace=True)

        # plot count distribution of short / long exposures
        log.info("plotting count distribution of short / long exposures")
        axs_dis[iquad].hist(squad._data.flatten(), bins=1000, label=f"{short_exptime}s", color="tab:blue", histtype="stepfilled", alpha=0.5)
        axs_dis[iquad].hist(lquad._data.flatten(), bins=1000, label=f"{long_exptime}s", color="tab:red", histtype="stepfilled", alpha=0.5)
        axs_dis[iquad].axvline(dc_low, lw=1, ls="--", color="0.2")
        axs_dis[iquad].set_title(f"quadrant {iquad+1}", loc="left")
        axs_dis[iquad].set_xscale("log")
        axs_dis[iquad].set_yscale("log")
        axs_dis[iquad].grid(color="0.9", ls="--", lw=0.5)
        # plot ratios and hot/cold pixel rejection
        log.info("plotting ratio of short / long exposures")
        axs_rat[iquad].axhspan(dark_low, dark_high, color="0.7", alpha=0.3)
        axs_rat[iquad].plot(squad._data[good_dc].flatten(), rquad._data[good_dc].flatten(), 'o', color="0.2", label='good DC')
        axs_rat[iquad].plot(squad._data[good_dc&mask_hotpix].flatten(), rquad._data[good_dc&mask_hotpix].flatten(), '.', color="tab:red", label='bad pixels')
        axs_rat[iquad].axhline(1, lw=1, ls="--", color="0.2", label="1:1 relationship")
        axs_rat[iquad].set_title(f"quadrant {iquad+1}", loc="left")
        axs_rat[iquad].set_yscale("log")
        axs_rat[iquad].grid(color="0.9", ls="--", lw=0.5)
    axs_dis[0].legend(loc="upper right")
    fig_dis.supxlabel(f'dark current ({unit})')
    fig_dis.supylabel('number of pixels')
    axs_rat[0].legend(loc="lower right")
    fig_rat.supxlabel(f"dark current ({unit}), {short_exptime}s exposure")
    fig_rat.supylabel("ratio, short / long exposure")
    save_fig(fig_dis, product_path=out_pixmask, to_display=display_plots, figure_path="qa", label="dc_hist")
    save_fig(fig_rat, product_path=out_pixmask, to_display=display_plots, figure_path="qa", label="dc_ratio")

    # mask pixels using flats ratio if possible
    ratio_flat = None
    if flat_a is not None and flat_b is not None:
        # median normalize flats
        median_box = 20
        log.info(f"normalizing flats background with {median_box = }")
        flat_a = flat_a / flat_a.medianImg(size=median_box)
        flat_b = flat_b / flat_b.medianImg(size=median_box)

        med_a, med_b = bn.nanmedian(flat_a._data), bn.nanmedian(flat_b._data)
        log.info(f"normalizing flats by median: {med_a = :.2f}, {med_b = :.2f}")
        flat_a = flat_a / med_a
        flat_b = flat_b / med_b

        # calculate ratio of flats
        ratio_flat = flat_a / flat_b
        ratio_med = bn.nanmedian(ratio_flat._data)
        ratio_min = bn.nanmin(ratio_flat._data)
        ratio_max = bn.nanmax(ratio_flat._data)
        log.info(f"calculating ratio of flats: {ratio_med = :.2f} [{ratio_min = :.2f}, {ratio_max = :.2f}]")

        # plot flats histograms
        log.info("plotting flats histograms")
        fig, ax = create_subplots(to_display=display_plots, figsize=(10,5))
        fig.suptitle(os.path.basename(out_pixmask))
        ax.axvspan(flat_low, flat_high, color="0.7", alpha=0.3)
        ax.hist(flat_a._data.flatten(), bins=1000, color="tab:blue", alpha=0.5, label=f"flat A ({os.path.basename(in_flat_a)})")
        ax.hist(flat_b._data.flatten(), bins=1000, color="tab:red", alpha=0.5, label=f"flat B ({os.path.basename(in_flat_b)})")
        ax.axvline(ratio_med, lw=1, ls="--", color="0.2")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(color="0.9", ls="--", lw=0.5)
        ax.legend(loc="upper right")
        ax.set_xlabel("flat")
        ax.set_ylabel("number of pixels")
        save_fig(fig, product_path=out_pixmask, to_display=display_plots, figure_path="qa", label="flat_hist")

        # create pixel mask using flat ratio
        flat_mask = (ratio_flat._data < flat_low) | (ratio_flat._data > flat_high)
        log.info(f"masking {flat_mask.sum()} pixels with flats ratio < {flat_low} or > {flat_high}")

        # update pixel mask
        pixmask.setData(mask=(pixmask._mask | flat_mask), inplace=True)

    # set masked pixels to NaN
    log.info(f"masked {pixmask._mask.sum()} pixels in total ({pixmask._mask.sum()/pixmask._mask.size*100:.2f}%)")
    pixmask.setData(data=numpy.nan, select=pixmask._mask)

    # write output mask
    log.info(f"writing pixel mask to '{os.path.basename(out_pixmask)}'")
    pixmask.writeFitsData(out_pixmask)

    return pixmask, ratio_dark, ratio_flat


def trace_centroids(in_image: str,
        out_trace_cent: str,
        correct_ref: bool = False,
        median_box: tuple = (1, 10),
        coadd: int = 5,
        method: str = "gauss",
        guess_fwhm: float = 2.5,
        counts_threshold: float = 500,
        max_diff: int = 5.0,
        ncolumns: int | Tuple[int] = 140,
        fit_poly: bool = False,
        poly_deg: int | Tuple[int] = 6,
        interpolate_missing: bool = True,
        display_plots: bool = False
    ) -> Tuple[TraceMask, TraceMask, TraceMask]:

    # load continuum image  from file
    log.info(f"using flat image {os.path.basename(in_image)} for tracing")
    img = loadImage(in_image)
    img.setData(data=numpy.nan_to_num(img._data), error=numpy.nan_to_num(img._error))

    # extract usefull metadata from the image
    channel = img._header["CCD"][0]

    # read slitmap extension
    slitmap = img.getSlitmap()
    slitmap = slitmap[slitmap["spectrographid"] == int(img._header["CCD"][1])]
    bad_fibers = slitmap["fibstatus"] == 1

    # perform median filtering along the dispersion axis to clean cosmic rays
    median_box = tuple(map(lambda x: max(x, 1), median_box))
    if median_box != (1, 1):
        log.info(f"performing median filtering with box {median_box} pixels")
        img = img.replaceMaskMedian(*median_box, replace_error=None)
        img._data = numpy.nan_to_num(img._data)
        img = img.medianImg(median_box, propagate_error=True)

    # coadd images along the dispersion axis to increase the S/N of the peaks
    if coadd != 0:
        log.info(f"coadding {coadd} pixels along the dispersion axis")
        coadd_kernel = numpy.ones((1, coadd), dtype="uint8")
        img = img.convolveImg(coadd_kernel)
        counts_threshold = counts_threshold * coadd

    # handle invalid error values
    img._error[img._mask|(img._error<=0)] = numpy.inf

    # calculate centroids for reference column
    if correct_ref:
        ref_cent = img.match_reference_column(ref_column=LVM_REFERENCE_COLUMN)
    else:
        ref_cent = img._slitmap[f"ypix_{channel}"].data

    # trace centroids in each column
    log.info(f"going to trace fiber centroids using {ncolumns} columns")
    centroids = img.trace_fiber_centroids(ref_column=LVM_REFERENCE_COLUMN, ref_centroids=ref_cent, mask_fibstatus=1,
                                          ncolumns=ncolumns, method=method, fwhm_guess=guess_fwhm,
                                          counts_threshold=counts_threshold, max_diff=max_diff)

    if fit_poly:
        # smooth all trace by a polynomial
        log.info(f"fitting centroid guess trace with {poly_deg}-deg polynomial")
        table_data, table_poly, table_poly_all = centroids.fit_polynomial(poly_deg, poly_kind="poly", min_samples_frac=0.5)
        _create_trace_regions(out_trace_cent, table_data, table_poly, table_poly_all, display_plots=display_plots)

        # set bad fibers in trace mask
        centroids._mask[bad_fibers] = True
        # linearly interpolate coefficients at masked fibers
        log.info(f"interpolating coefficients at {bad_fibers.sum()} masked fibers")
        centroids.interpolate_coeffs()
    else:
        log.info("interpolating centroid guess trace")
        centroids.interpolate_data(axis="X")

        # set bad fibers in trace mask
        centroids._mask[bad_fibers] = True
        log.info(f"interpolating data at {bad_fibers.sum()} masked fibers")
        centroids.interpolate_data(axis="Y")

    # write centroid if requested
    if out_trace_cent is not None:
        log.info(f"writing centroid trace to '{os.path.basename(out_trace_cent)}'")
        centroids.writeFitsData(out_trace_cent)
    return centroids, img


def trace_fibers(
    in_image: str,
    out_trace_cent: str,
    out_trace_amp: str,
    out_trace_fwhm: str,
    in_trace_cent_guess: str,
    out_model: str = None,
    out_ratio: str = None,
    correct_ref: bool = False,
    median_box: tuple = (1, 10),
    coadd: int = 5,
    method: str = "gauss",
    guess_fwhm: float = 2.5,
    counts_threshold: float = 500,
    max_diff: int = 5.0,
    ncolumns: int | Tuple[int] = 40,
    nblocks: int = 18,
    iblocks: list = [],
    fwhm_limits: Tuple[float] = (1.0, 3.5),
    fit_poly: bool = False,
    poly_deg: int | Tuple[int] = 6,
    interpolate_missing: bool = True,
    only_centroids: bool = False,
    use_given_centroids: bool = False,
    display_plots: bool = False
) -> Tuple[TraceMask, TraceMask, TraceMask]:
    """Trace fibers in a given image

    Given a continuum exposure, this function will trace the fibers
    and return the fiber centroids, flux, and FWHM.

    The first step is to perform a median filtering along the dispersion axis
    to clean cosmic rays. Then, the image is coadded using a Gaussian
    convolution along the dispersion axis to increase the S/N of the peaks. The
    fiber positions are extracted from the fibermap extension of the image
    header. The reference fiber positions are corrected using a
    cross-correlation between the reference fiber profile and the observed
    fiber profile.

    The first measurement of the fiber centroids is performed using individual
    Gaussian fittings per column in a selection of ncolumns across the X-axis.
    This first guess of the centroids is fitted with a polynomial.

    The centroids measured and fitted in the previous step are used to fit for
    the fiber profiles in the image along each ncolumns columns in the image. A
    number of nblocks of Gaussians are fitted simultaneously along each column.
    From each fitting the amplitude, centroid, and FWHM are estimated.

    Optionally, the amplitude, centroid, and FWHM traces can be fitted with
    polynomial functions or interpolated along the X-axis to fully sample the
    fibers.

    Optionally, bad/missing (non-illuminated) fibers can be interpolated using
    the information of the neighboring fibers. If the traces where fitted using
    polynomial functions, the bad/missing fibers are interpolated in the
    coefficients space. The interpolation of bad/missing fibers happens in data
    space otherwise.

    Parameters
    ----------
    in_image : str
        path to input image
    out_trace_cent : str
        path to output centroid trace
    out_trace_amp : str
        path to output amplitude trace
    out_trace_fwhm : str
        path to output FWHM trace
    in_trace_cent_guess : str
        path to input centroid guess trace
    correct_ref : bool, optional
        whether to correct reference fiber positions, by default False
    median_box : tuple, optional
        median box to use for cleaning cosmic rays, by default (1, 10)
    coadd : int, optional
        number of pixels to coadd along dispersion axis, by default 5
    method : str, optional
        method to use for tracing, by default "gauss"
    guess_fwhm : float, optional
        guess FWHM of fiber profiles, by default 3.0
    counts_threshold : float, optional
        threshold to use for fiber detection, by default 0.5
    max_diff : int, optional
        maximum difference between consecutive fiber positions, by default 1.5
    ncolumns : int or 2-tuple, optional
        number of columns to use for tracing, by default 18
    nblocks : int, optional
        number of blocks to use for tracing, by default 18
    iblocks : list, optional
        list of blocks to trace, by default []
    fwhm_limits: tuple, optional
        limits to use for FWHM fitting, by default (1.0, 3.5)
    fit_poly : bool, optional
        whether to fit a polynomial to the dispersion solution, by default False (interpolate in X axis)
    poly_deg : int or 3-tuple, optional
        degree of polynomial(s) to use when fitting amplitude, centroid and FWHM, by default 6
    intrpolate_missing : bool, optional
        whether to interpolate bad/missing fibers, by default True
    only_centroids : bool, optional
        whether to only trace centroids, by default False
    use_given_centroids : bool, optional
        whether to use given centroids, by default False
    display_plots : bool, optional
        whether to show plots on display or not, by default True

    Returns
    -------
    centroids : TraceMask
        fiber centroids
    flux : TraceMask
        fiber flux
    fwhm : TraceMask
        fiber FWHM

    Raises
    ------
    ValueError
        invalid polynomial degree
    ValueError
        invalid number of columns
    """
    # parse polynomial degrees
    if isinstance(poly_deg, (list, tuple)) and len(poly_deg) == 3:
            deg_amp, deg_cent, deg_fwhm = poly_deg
    elif isinstance(poly_deg, int):
        deg_amp = deg_cent = deg_fwhm = poly_deg
    else:
        raise ValueError(f"invalid polynomial degree: {poly_deg}")

    # load continuum image  from file
    log.info(f"using flat image {os.path.basename(in_image)} for tracing")
    img = loadImage(in_image)
    img.setData(data=numpy.nan_to_num(img._data), error=numpy.nan_to_num(img._error))

    # read slitmap extension
    slitmap = img.getSlitmap()
    slitmap = slitmap[slitmap["spectrographid"] == int(img._header["CCD"][1])]
    bad_fibers = slitmap["fibstatus"] == 1

    # perform median filtering along the dispersion axis to clean cosmic rays
    median_box = tuple(map(lambda x: max(x, 1), median_box))
    if median_box != (1, 1):
        log.info(f"performing median filtering with box {median_box} pixels")
        img = img.replaceMaskMedian(*median_box, replace_error=None)
        img._data = numpy.nan_to_num(img._data)
        img = img.medianImg(median_box, propagate_error=True)

    # coadd images along the dispersion axis to increase the S/N of the peaks
    if coadd != 0:
        log.info(f"coadding {coadd} pixels along the dispersion axis")
        coadd_kernel = numpy.ones((1, coadd), dtype="uint8")
        img = img.convolveImg(coadd_kernel)
        counts_threshold = counts_threshold * coadd

    # handle invalid error values
    img._error[img._mask|(img._error<=0)] = numpy.inf

    # trace centroids in each column
    log.info(f"loading guess fiber centroids from '{os.path.basename(in_trace_cent_guess)}'")
    centroids = TraceMask.from_file(in_trace_cent_guess)

    # initialize flux and FWHM traces
    trace_cent = copy(centroids)
    trace_amp = copy(centroids)
    trace_amp._header["IMAGETYP"] = "trace_amplitude"
    trace_fwhm = copy(centroids)
    trace_fwhm._header["IMAGETYP"] = "trace_fwhm"

    trace_amp, trace_cent, trace_fwhm, columns, mod_columns, residuals = img.trace_fiber_widths(centroids, ref_column=LVM_REFERENCE_COLUMN,
                                                                                                ncolumns=ncolumns, nblocks=LVM_NBLOCKS, iblocks=iblocks,
                                                                                                max_diff=max_diff,
                                                                                                fwhm_guess=guess_fwhm, fwhm_range=fwhm_limits,
                                                                                                counts_threshold=counts_threshold)

    # smooth all trace by a polynomial
    if fit_poly:

        # plt.figure()
        # data = copy(numpy.split(trace_amp._data, LVM_NBLOCKS, axis=0)[16][0])
        # # data[data==0] = numpy.nan
        # plt.plot(data, label="measured amp")

        log.info(f"fitting peak trace with {deg_amp}-deg polynomial")
        # constraints = [{'type': 'ineq', 'fun': lambda t, c: interpolate.splev(0, (t, c, deg_amp), der=1)},
        #                {'type': 'ineq', 'fun': lambda t, c: -interpolate.splev(trace_amp._data.shape[1], (t, c, deg_amp), der=1)}]
        table_data, table_poly, table_poly_all = trace_amp.fit_polynomial(deg_amp, poly_kind="poly", clip=(0.0,None), min_samples_frac=0.5)
        # table_data, table_poly, table_poly_all = trace_amp.fit_spline(degree=deg_amp, smoothing=0, constraints=constraints)
        _create_trace_regions(out_trace_amp, table_data, table_poly, table_poly_all, display_plots=display_plots)
        # plt.plot(numpy.split(trace_amp._data, LVM_NBLOCKS, axis=0)[16][0], label="fitted amp")
        # plt.show()

        log.info(f"fitting centroid trace with {deg_cent}-deg polynomial")
        table_data, table_poly, table_poly_all = trace_cent.fit_polynomial(deg_cent, poly_kind="poly", min_samples_frac=0.5)
        _create_trace_regions(out_trace_cent, table_data, table_poly, table_poly_all, display_plots=display_plots)

        log.info(f"fitting FWHM trace with {deg_fwhm}-deg polynomial")
        table_data, table_poly, table_poly_all = trace_fwhm.fit_polynomial(deg_fwhm, poly_kind="poly", clip=fwhm_limits, min_samples_frac=0.5)
        _create_trace_regions(out_trace_fwhm, table_data, table_poly, table_poly_all, display_plots=display_plots)

        # set bad fibers in trace mask
        trace_amp._mask[bad_fibers] = True
        trace_cent._mask[bad_fibers] = True
        trace_fwhm._mask[bad_fibers] = True

        # linearly interpolate coefficients at masked fibers
        if interpolate_missing:
            log.info(f"interpolating coefficients at {bad_fibers.sum()} masked fibers")
            trace_amp.interpolate_coeffs()
            trace_cent.interpolate_coeffs()
            trace_fwhm.interpolate_coeffs()
    else:
        # interpolate traces along X axis to fill in missing data
        log.info("interpolating traces along X axis to fill in missing data")
        trace_amp.interpolate_data(axis="X")
        trace_cent.interpolate_data(axis="X")
        trace_fwhm.interpolate_data(axis="X")
        # set bad fibers in trace mask
        trace_amp._mask[bad_fibers] = True
        trace_cent._mask[bad_fibers] = True
        trace_fwhm._mask[bad_fibers] = True

        if interpolate_missing:
            log.info(f"interpolating data at {bad_fibers.sum()} masked fibers")
            trace_amp.interpolate_data(axis="Y")
            trace_cent.interpolate_data(axis="Y")
            trace_fwhm.interpolate_data(axis="Y")

    # evaluate model image
    if out_model is not None and out_ratio is not None:
        log.info("evaluating model image")
        model, mratio = img.eval_fiber_model(trace_cent, trace_fwhm, trace_amp)
        model.writeFitsData(out_model)
        mratio.writeFitsData(out_ratio)
    else:
        model, mratio = None, None

    # write output traces
    log.info(f"writing amplitude trace to '{os.path.basename(out_trace_amp)}'")
    trace_amp.writeFitsData(out_trace_amp)
    log.info(f"writing centroid trace to '{os.path.basename(out_trace_cent)}'")
    trace_cent.writeFitsData(out_trace_cent)
    log.info(f"writing FWHM trace to '{os.path.basename(out_trace_fwhm)}'")
    trace_fwhm.writeFitsData(out_trace_fwhm)

    # plot results
    log.info("plotting results")
    camera = img._header["CCD"]
    # residuals
    fig, ax = create_subplots(to_display=display_plots, nrows=1, ncols=1, figsize=(15,7))
    fig.suptitle(f"Residuals of joint model for {camera = }")
    ax.plot(columns, residuals, "o", color="tab:red", ms=10)
    ax.axhline(0, color="0.2", ls="--", lw=1)
    ax.grid(ls="--", color="0.9", lw=0.5, zorder=0)
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("residuals (%)")
    save_fig(
        fig,
        product_path=out_trace_amp,
        to_display=display_plots,
        figure_path="qa",
        label="residuals_int_columns"
    )

    # profile models vs data
    fig, ax = create_subplots(to_display=display_plots, figsize=(15,7))
    fig.suptitle(f"Profile fitting residuals for {camera = }")
    fig.supylabel("residuals (%)")
    fig.supxlabel("Y (pixel)")

    colors = plt.cm.Spectral(numpy.linspace(0, 1, len(columns)))
    idx = numpy.argsort(columns)
    img_ = copy(img)
    for i in idx:
        icolumn = columns[i]

        img_slice = img_.getSlice(icolumn, axis="y")
        joint_mod = mod_columns[i](img_slice._pixels)
        img_slice._data[(img_slice._mask)|(joint_mod<=0)] = numpy.nan

        weights = img_slice._data / bn.nansum(img_slice._data) * 500
        residuals = (joint_mod - img_slice._data) / img_slice._data * 100
        ax.scatter(img_slice._pixels, residuals, s=weights, lw=0, color=colors[i])
        ax.set_ylim(-50, 50)
    save_fig(
        fig,
        product_path=out_trace_amp,
        to_display=display_plots,
        figure_path="qa",
        label="residuals_columns"
    )

    return centroids, trace_cent, trace_amp, trace_fwhm, img, model, mratio
