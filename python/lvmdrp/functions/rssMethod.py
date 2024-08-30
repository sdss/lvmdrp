#!/usr/bin/env python
# encoding: utf-8

import os
from copy import deepcopy as copy
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import numpy
import yaml
import bottleneck as bn
from tqdm import tqdm
from astropy import units as u
from astropy.constants import c
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.stats import biweight_scale
from numpy import polynomial
from scipy import interpolate, ndimage

from lvmdrp.utils.decorators import skip_on_missing_input_path, skip_if_drpqual_flags
from lvmdrp.utils.bitmask import ReductionStage
from lvmdrp.core.constants import CONFIG_PATH, ARC_LAMPS
from lvmdrp.core.cube import Cube
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.image import loadImage
from lvmdrp.core.passband import PassBand
from lvmdrp.core.plot import (plt, create_subplots, save_fig,
                              plot_wavesol_coeffs, plot_wavesol_residuals,
                              plot_wavesol_spec, plot_wavesol_wave,
                              plot_wavesol_lsf)
from lvmdrp.core.rss import RSS, _read_pixwav_map, loadRSS, lvmFrame, lvmFFrame, lvmCFrame
from lvmdrp.core.spectrum1d import Spectrum1D, _spec_from_lines, _cross_match_float
from lvmdrp.core.fluxcal import galExtinct
from lvmdrp.utils import flatten
from lvmdrp import log


description = "Provides Methods to process Row Stacked Spectra (RSS) files"

__all__ = [
    "determine_wavelength_solution",
    "create_pixel_table",
    "combine_rsss",
    "checkPixTable_drp",
    "correctPixTable_drp",
    "resample_wavelength",
    "includePosTab_drp",
    "join_spec_channels"
]


DONE_PASS = "gi"
DONE_MAGS = numpy.asarray([22, 21])
DONE_LIMS = numpy.asarray([1.7, 5.0])

# GB hand picked isolated bright lines across each channel which are not doublest in UVES atlas
# true wavelengths taken from UVES sky line atlas
REF_SKYLINES = {'b':[5577.346680], 'r':[6363.782715, 7358.680176, 7392.209961], 'z':[8399.175781, 8988.383789, 9552.546875, 9719.838867]}


def _linear_model(pars, xdata):
    """simple linear model

    Parameters
    ----------
    pars : array_like
        2-tuple for slope and y-offset
    xdata : array_like
        x-values to evaluate the model

    Returns
    -------
    array_like
        evaluated linear model
    """
    return pars[0] * xdata + pars[1]


def _illumination_correction(fiberflat, apply_correction=True):
    # define fiberflat spectrograph id
    specid = int(fiberflat._header["CCD"][1])
    # load fibermap and select fibers
    fibermap = Table(fiberflat._slitmap)
    fibermap = fibermap[fibermap["spectrographid"] == specid]

    # define fiber IDs for each telescope
    sci_fibers = fibermap["ifulabel"] == f"Sci{specid}"
    skw_fibers = fibermap["ifulabel"] == f"SkyW{specid}"
    ske_fibers = fibermap["ifulabel"] == f"SkyE{specid}"
    std_fibers = fibermap["ifulabel"] == f"Std{specid}"

    # define data and set to NaN bad pixels
    data = copy(fiberflat._data)
    data[(fiberflat._mask)|(data <= 0)] = numpy.nan

    # compute median factors
    sci_factor = numpy.nanmedian(data[sci_fibers, 1000:3000])
    skw_factor = numpy.nanmedian(data[skw_fibers, 1000:3000])
    ske_factor = numpy.nanmedian(data[ske_fibers, 1000:3000])
    std_factor = numpy.nanmedian(data[std_fibers, 1000:3000])
    norm = numpy.mean([sci_factor, skw_factor, ske_factor, std_factor])
    sci_factor /= norm
    skw_factor /= norm
    ske_factor /= norm
    std_factor /= norm

    # apply correction if requested
    if apply_correction:
        fiberflat[sci_fibers] *= sci_factor
        fiberflat[skw_fibers] *= skw_factor
        fiberflat[ske_fibers] *= ske_factor
        fiberflat[std_fibers] *= std_factor

    return fiberflat, dict(zip(("Sci", "SkyW", "SkyE", "Std"), (sci_factor, skw_factor, ske_factor, std_factor)))


def _make_arcline_axes(display_plots, pixel, ref_lines, ifiber, unit="e-", ncols=3, fig_shape=(5, 4)):
    nlines = len(pixel)
    nrows = int(numpy.ceil(nlines / ncols))
    fig, axs = create_subplots(to_display=display_plots,
                               nrows=nrows, ncols=ncols,
                               figsize=(fig_shape[0]*ncols, fig_shape[1]*nrows),
                               layout="constrained")
    fig.suptitle(f"Gaussian fitting for fiber {ifiber}")
    fig.supylabel(f"Counts ({unit}/pixel)")
    for i, ax in zip(range(nlines), axs):
        ax.set_title(f"line {ref_lines[i]:.2f} (Ang)")
        ax.set_xlabel("X (pixel)")

    return fig, axs


def mergeRSS_drp(files_in, file_out, mergeHdr="1"):
    """
    Different RSS are merged into a common file by extending the number of fibers.

    Note that the number of spectral pixel need to be the same and that all input RSS
    must have the same extensions.

    Parameters
    ----------
    files_in : string
        Comma-separates name of RSS FITS files to be merged into a signal RSS
    file_out : string
        Name of the merged RSS FITS file
    mergeHdr : string of integer (0 or 1), optional with default: '1'
        Flag to indicate if the header of the input RSS files are also merger and stored
        in the merged RSS. 1 if yes, 0 if not

    Examples
    --------
    user:> lvmdrp rss mergeRSS RSS1.fits,RSS2.fits,RSS3.fits RSS_OUT.fits
    """

    files = files_in.split(",")

    for i in range(len(files)):
        if i == 0:
            rss = loadRSS(files[i])
        else:
            rss_add = loadRSS(files[i])
            if mergeHdr == "0":
                rss.append(rss_add, append_hdr=False)
            else:
                rss.append(rss_add, append_hdr=True)
    rss.writeFitsData(file_out)


# TODO:
# * define ancillary product lvm-lxpeak for ref_line_file
# * define ancillary product lvm-arc (rss arc) for replace arc_rss
# * define ancillary product lvm-wave to contain wavelength solutions
# * merge disp_rss and res_rss products into lvmArc product, change variable to out_arc
@skip_on_missing_input_path(["in_arc"])
# @skip_if_drpqual_flags(["SATURATED"], "in_arc")
def determine_wavelength_solution(in_arcs: List[str]|str, out_wave: str, out_lsf: str,
                                  cont_niter: int = 3, cont_thresh: float = 0.999, cont_box_range: Tuple[int] = (50, 300),
                                  ref_fiber: int = 319, pixel: List[float] = [], ref_lines: List[float] = [],
                                  use_line: List[bool] = [],
                                  cc_correction: bool = True,
                                  cc_max_shift: int = 30,
                                  aperture: int = 12,
                                  fwhm_guess: float = 3.0,
                                  bg_guess: float = 0.0,
                                  flux_range: List[float] = [100.0, numpy.inf],
                                  cent_range: List[float] = [-4.0, 4.0],
                                  fwhm_range: List[float] = [2.0, 4.5],
                                  bg_range: List[float] = [-1000.0, numpy.inf],
                                  poly_disp: int = 6, poly_fwhm: int = 4,
                                  poly_cros: int = 0, poly_kinds: list = ['poly', 'poly', 'poly'],
                                  negative: bool = False,
                                  plot_fibers: List[int] = [],
                                  display_plots: bool = False):
    """
    Solves for the wavelength and the LSF using polynomial fitting

    Fits Gaussian + const profiles to a set of previously identified arc lines,
    then for each fiber fits a polynomial function for line(pix) vs
    line(ref_wave) to get the wavelength solution. This wavelength solution is
    used to estimate the instrumental resolution for each measured arc line.
    Similarly, a polynomial fitting is performed to each fiber in the plane
    FWHM(wave) vs line(wave).

    Parameters
    ----------
    in_arcs : list[str]|str
        Path or a list of paths to extracted arc exposures
    out_wave : str
        Path to output wavelength trace file
    out_lsf : str
        Path to output LSF trace file
    cont_niter : int, optional
        Number of iterations for the continuum fitting, by default 3
    cont_thresh : float, optional
        Threshold above which pixels get rejected in the continuum fitting, by default 0.999
    cont_box_range : tuple[int], optional
        range of box sizes in adaptive median filtering for the continuum fitting, by default (50, 300)
    ref_fiber : int, optional
        Reference fiber used in line identification, by default 319
    pixel : list[float], optional
        Pixel positions for reference arc lines, by default []
    ref_lines : list[float], optional
        Wavelengths for reference arc lines, by default []
    use_line : list[bool], optional
        List of boolean selection for given `pixel` and `ref_lines`, by default []
    cc_correction : bool, optional
        Perform cross-correlation correction to reference arc lines to account for instrumental shifts, by default True
    cc_max_shift : int, optional
        Maximum shift in pixels to reference lines, by default 30
    aperture : int, optional
        Range of pixels around arc lines guess centroid within which the Gaussian fitting will be performed, by default 12
    fwhm_guess : float, optional
        Guess for the FWHM (in pixel) of the arc lines during the Gaussian fitting, by default 3.0
    bg_guess : float, optional
        Guess for the local background around each arc line during Gaussian fitting, by default 0.0
    flux_range : list[float], optional
        Range within which the integrated flux of arc lines is allowed to be during Gaussian fitting, by default [800.0, inf]
    cent_range : list[float], optional
        Range of arc line centroids (in pixel) within which a line centroid is expected to be during Gaussian fitting, by default [-3.0, 3.0]
    fwhm_range : list[float], optional
        Range of FWHM (in pixel) allowed for arc lines during Gaussian fitting, by default [2.0, 3.5]
    bg_range : list[float], optional
        Range local background level allowed for arc lines during Gaussian fitting, by default [1000.0, inf]
    poly_disp : int, optional
        Polynomial degree for fiber wavelength solution fitting, by default 5
    poly_fwhm : int, optional
        Polynomial degree for fiber LSF solution fitting, by default 2
    poly_cros : int, optional
        Polynomial degree for cross-dispersion smoothing of FWHM(pixel) ( = 0 no smoothing), by default 2
    negative : bool, optional
        Assume absorption spectra, by default False
    plot_fibers : list[int], optional
        When debug_mode == True, this will show additional plots on the fitting of the listed fibers
    display_plots : bool, optional
        If True, the results are plotted and displayed

    Returns
    -------
    ref_lines : np.ndarray[float], nlines
        Reference lines used during the Gaussian fitting
    masked : np.ndarray[float], nfibers x nlines
        Masked reference arc lines for each fiber after Gaussian fitting
    cent_wave : np.ndarray[float], nfibers x nlines
        Pixel positions of reference arc lines for each fiber
    fwhm_wave : np.ndarray[float], nfibers x nlines
        FWHM (in pixel) of reference arc lines for each fiber
    arc : lvmdrp.core.rss.RSS
        Arc used to fit wavelength and LSF solutions
    wave_trace : lvmdrp.core.tracemask.TraceMask
        Trace object for wavelength solution
    fwhm_trace :
        Trace object for LSF solution
    """

    # convert parameters to the correct type
    kind_disp, kind_fwhm, kind_cros = poly_kinds.split(",") if isinstance(poly_kinds, str) else poly_kinds

    if isinstance(in_arcs, (list, tuple)):
        pass
    elif isinstance(in_arcs, str):
        in_arcs = [in_arcs]
    else:
        raise ValueError(f"wrong type for {in_arcs = }, it can be either a string or a list or tuple of")

    iarcs = []
    ilamps = []
    for in_arc in in_arcs:
        # initialize the extracted arc line frame
        log.info(f"reading arc from '{in_arc}'")
        arc = RSS.from_file(in_arc)

        camera = arc._header["CCD"]
        onlamp = ["ON", True, 'T', 1]
        lamps = [lamp.lower() for lamp in ARC_LAMPS if arc._header.get(lamp, "OFF") in onlamp]
        if len(lamps) == 0:
            log.error("no arc lamps were on during this exposure")
            continue

        # update current lamps
        ilamps.extend(lamps)
        # append arc
        iarcs.append(arc)

    # combine RSS objects
    arc = RSS()
    arc.combineRSS(iarcs, method="sum")
    unit = arc._header["BUNIT"]
    # update lamps status
    lamps = set(ilamps)

    # mask std fibers since they are not regularly illuminated during arc exposures
    fibermap = arc._slitmap[arc._slitmap["spectrographid"] == int(camera[1])]
    select = fibermap["telescope"] == "Spec"
    arc._mask[select] = True
    arc._data[select] = 0.0
    arc._error[select] = numpy.inf

    # subtract continuum
    if cont_niter > 0:
        log.info(f"fitting and subtracting continuum with parameters: {cont_niter = }, {cont_thresh = }, {cont_box_range = }")
        arc, _, _ = arc.subtract_continuum(niter=cont_niter, thresh=cont_thresh, median_box_range=cont_box_range)

    # replace NaNs
    mask = arc._mask | numpy.isnan(arc._data) | numpy.isnan(arc._error)
    mask |= (arc._data < 0.0) | (arc._error <= 0.0)
    arc._data[mask] = 0.0
    arc._error[mask] = numpy.inf

    # read reference lines
    if len(pixel) == 0 or len(ref_lines) == 0 or len(use_line) == 0:
        ilamps = [lamp.lower() for lamp in ARC_LAMPS]
        lamps_label = "_".join(sorted(lamps, key=lambda x: ilamps.index(x)))
        _, ref_fiber_, pixel, ref_lines, use_line = _read_pixwav_map(lamps_label, camera)
        # if no reference file for combined lamps exist, read individual files
        if ref_fiber is None:
            pixel_list, ref_lines_list, use_line_list = [], [], []
            for lamp in lamps:
                log.info(f"loading reference lines for {lamp = } in {camera = }")
                _, ref_fiber_, pixel, ref_lines, use_line = _read_pixwav_map(lamp, camera)

                # remove masked lines
                pixel = pixel[use_line]
                ref_lines = ref_lines[use_line]
                use_line = use_line[use_line]

                pixel_list.append(pixel)
                ref_lines_list.append(ref_lines)
                use_line_list.append(use_line)

            # combine all reference lines into a long array
            pixel = numpy.concatenate(pixel_list)
            ref_lines = numpy.concatenate(ref_lines_list)
            use_line = numpy.concatenate(use_line_list)
    else:
        log.info(f"using given reference lines: {ref_lines}")

    # sort lines by pixel position
    sort = numpy.argsort(pixel)
    pixel = pixel[sort]
    ref_lines = ref_lines[sort]
    use_line = use_line[sort]

    # apply cc correction to lines if needed
    if cc_correction or ref_fiber != ref_fiber_:
        log.info(f"running cross matching on all {pixel.size} identified lines")
        # determine maximum correlation shift
        pix_spec = _spec_from_lines(pixel, sigma=2.5/2.354, wavelength=arc._pixels)

        # fix cc_max_shift
        # cross-match spectrum and pixwav map
        cc, bhat, mhat = _cross_match_float(
            ref_spec=pix_spec,
            obs_spec=arc._data[ref_fiber],
            stretch_factors=numpy.linspace(0.8,1.2,10000),
            shift_range=[-cc_max_shift, cc_max_shift],
            normalize_spectra=False,
        )

        log.info(f"max CC = {cc:.2f} for strech = {mhat:.8f} and shift = {bhat:.8f}")
    else:
        mhat, bhat = 1.0, 0.0

    # remove bad lines
    pixel = pixel[use_line]
    ref_lines = ref_lines[use_line]
    use_line = use_line[use_line]
    nlines = len(pixel)

    # correct initial pixel map by shifting
    pixel = mhat * pixel + bhat

    if negative:
        log.info("flipping arc along flux direction")
        arc = -1 * arc + numpy.nanmedian(arc._data)

    # setup storage array
    wave_coeffs = numpy.zeros((arc._fibers, numpy.abs(poly_disp) + 1))
    lsf_coeffs = numpy.zeros((arc._fibers, numpy.abs(poly_fwhm) + 1))
    wave_sol = numpy.zeros((arc._fibers, arc._data.shape[1]), dtype=numpy.float32)
    wave_rms = numpy.zeros(arc._fibers, dtype=numpy.float32)
    lsf_sol = numpy.zeros((arc._fibers, arc._data.shape[1]), dtype=numpy.float32)
    lsf_rms = numpy.zeros(arc._fibers, dtype=numpy.float32)

    # measure the ARC lines with individual Gaussian across the CCD
    log.info(f"fitting arc lines for each fiber for {ref_fiber = } with parameter ranges:")
    log.info(f"   {flux_range = } {unit}")
    log.info(f"   {cent_range = } pixel")
    log.info(f"   {fwhm_range = } pixel")
    log.info(f"   {bg_range   = } {unit}")

    # initialize plots for arc lines fitting
    axs_fibers = {}
    axs_fibers[ref_fiber] = _make_arcline_axes(display_plots, pixel=pixel, ref_lines=ref_lines, ifiber=ref_fiber)
    if plot_fibers:
        for ifiber in plot_fibers:
            axs_fibers[ifiber] = _make_arcline_axes(display_plots, pixel=pixel, ref_lines=ref_lines, ifiber=ifiber)
    fibers, flux, cent_wave, fwhm, masked = arc.measureArcLines(
        ref_fiber,
        pixel,
        aperture=aperture,
        fwhm_guess=fwhm_guess,
        bg_guess=bg_guess,
        flux_range=flux_range,
        cent_range=cent_range,
        fwhm_range=fwhm_range,
        bg_range=bg_range,
        axs=axs_fibers,
    )
    for ifiber, (fig, axs) in axs_fibers.items():
        save_fig(
            fig,
            product_path=out_wave,
            to_display=display_plots,
            figure_path="qa",
            label=f"lines_fitting_{ifiber:04d}",
        )

    # numpy.savetxt("./pixels.txt", cent_wave)
    # numpy.savetxt("./flux.txt", flux)
    # numpy.savetxt("./fwhm.txt", fwhm)

    # smooth the FWHM values for each ARC line in cross-dispersion direction
    if poly_cros != 0:
        log.info(
            f"smoothing FWHM of guess lines along cross-dispersion axis using {poly_cros}-deg polynomials")
        for i in range(nlines):
            select = numpy.logical_and(
                numpy.logical_not(masked[:, i]), flux[:, i] > flux_range[0]
            )
            fwhm_med = ndimage.filters.median_filter(numpy.fabs(fwhm[select, i]), 4)
            msg = f'Failed to fit {kind_cros} for arc line {i}'
            if kind_cros not in ["poly", "legendre", "chebyshev"]:
                log.warning(f"invalid polynomial kind '{kind_cros}'. Falling back to 'poly'")
                arc.add_header_comment(f"invalid polynomial kind '{kind_cros}'. Falling back to 'poly'")
                kind_cros = "poly"

            if kind_cros == "poly":
                cros_cls = polynomial.Polynomial
            elif kind_cros == "legendre":
                cros_cls = polynomial.Legendre
            elif kind_cros == "chebyshev":
                cros_cls = polynomial.Chebyshev

            try:
                poly = cros_cls.fit(fibers[select], fwhm_med, deg=poly_cros)
            except ValueError as e:
                log.error(f'{msg}: {e}')
                continue
            except numpy.linalg.LinAlgError as e:
                log.error(f'{msg}: {e}')
                continue

            fwhm[:, i] = poly(fibers)

    # Determine the wavelength solution
    log.info(f"fitting wavelength using {poly_disp}-deg polynomials")

    # Iterate over the fibers
    good_fibers = numpy.ones(len(fibers), dtype="bool")
    for i in fibers:
        good_lines = ~masked[i]
        if good_lines.sum() <= poly_disp + 1:
            log.warning(f"fiber {i} has {good_lines.sum()} (< {poly_disp + 1 = }) good lines")
            arc.add_header_comment(f"fiber {i} has {good_lines.sum()} (< {poly_disp + 1 = }) good lines")
            good_fibers[i] = False
            continue

        if kind_disp not in ["poly", "legendre", "chebyshev"]:
            log.warning(("invalid polynomial kind " f"'{kind_disp = }'. Falling back to 'poly'"))
            arc.add_header_comment("invalid polynomial kind " f"'{kind_disp = }'. Falling back to 'poly'")
        if kind_disp == "poly":
            wave_cls = polynomial.Polynomial
        elif kind_disp == "legendre":
            wave_cls = polynomial.Legendre
        elif kind_disp == "chebyshev":
            wave_cls = polynomial.Chebyshev

        wave_poly = wave_cls.fit(cent_wave[i, good_lines], ref_lines[good_lines], deg=poly_disp)

        wave_coeffs[i, :] = wave_poly.convert().coef
        wave_sol[i, :] = wave_poly(arc._pixels)
        wave_rms[i] = numpy.nanstd(wave_poly(cent_wave[i, good_lines]) - ref_lines[good_lines])

    log.info(
        "finished wavelength fitting with median "
        f"RMS = {numpy.nanmedian(wave_rms):g} Angstrom "
        f"({numpy.nanmedian(wave_rms[:,None]/numpy.diff(wave_sol, axis=1)):g} pix)"
    )

    # Estimate the spectral resolution pattern
    dwave = numpy.fabs(numpy.gradient(wave_sol, axis=1))

    # Iterate over the fibers
    log.info(f"fitting LSF solutions using {poly_fwhm}-deg polynomials")
    for i in fibers:
        good_lines = ~masked[i]
        if good_lines.sum() <= poly_fwhm + 1:
            log.warning(f"fiber {i} has {good_lines.sum()} (< {poly_fwhm + 1 = }) good lines")
            arc.add_header_comment(f"fiber {i} has {good_lines.sum()} (< {poly_fwhm + 1 = }) good lines")
            good_fibers[i] = False
            continue

        dw = numpy.interp(cent_wave[i, good_lines], arc._pixels, dwave[i])
        fwhm_wave = dw * fwhm[i, good_lines]

        if kind_fwhm not in ["poly", "legendre", "chebyshev"]:
            log.warning(f"invalid polynomial kind '{kind_fwhm = }'. Falling back to 'poly'")
            arc.add_header_comment(f"invalid polynomial kind '{kind_fwhm = }'. Falling back to 'poly'")
            kind_fwhm = "poly"
        if kind_fwhm == "poly":
            fwhm_cls = polynomial.Polynomial
        elif kind_fwhm == "legendre":
            fwhm_cls = polynomial.Legendre
        elif kind_fwhm == "chebyshev":
            fwhm_cls = polynomial.Chebyshev

        fwhm_poly = fwhm_cls.fit(cent_wave[i, good_lines], fwhm_wave, deg=poly_fwhm)

        lsf_coeffs[i, :] = fwhm_poly.convert().coef
        lsf_sol[i, :] = fwhm_poly(arc._pixels)
        lsf_rms[i] = numpy.nanstd(fwhm_wave - fwhm_poly(cent_wave[i, good_lines]))

    log.info(
        "finished LSF fitting with median "
        f"RMS = {numpy.nanmedian(lsf_rms):g} Angstrom "
        f"({numpy.nanmedian(lsf_rms[:,None]/numpy.gradient(wave_sol, axis=1)):g} pix)"
    )

    # create plot of reference spectrum and wavelength fitting residuals
    fig, (ax_spec, ax_res) = create_subplots(to_display=display_plots, nrows=2, ncols=1, sharex=True, figsize=(15, 7), layout="constrained")

    ax_res = plot_wavesol_residuals(ref_fiber, ref_waves=ref_lines,
                                lines_pixels=cent_wave, poly_cls=wave_cls,
                                coeffs=wave_coeffs, ax=ax_res, labels=True)
    ax_spec = plot_wavesol_spec(ref_fiber, ref_pixels=pixel, aperture=aperture,
                                mhat=mhat, bhat=bhat, arc=arc, ax=ax_spec, labels=True)
    save_fig(
        fig,
        product_path=out_wave,
        to_display=display_plots,
        figure_path="qa",
        label="residuals_wave",
    )

    # plot wavelength fitting minus linear term
    fig_wave = plt.figure(figsize=(16, 10), layout="constrained")
    gs = gridspec.GridSpec(10, poly_disp + 1, figure=fig_wave)

    ax_sol_wave = fig_wave.add_subplot(gs[:5, :])
    ax_sol_wave = plot_wavesol_wave(xpix=arc._pixels, ref_waves=ref_lines,
                                    lines_pixels=cent_wave, wave_poly=wave_cls,
                                    wave_coeffs=wave_coeffs, ax=ax_sol_wave, labels=True)

    ax_coe_wave, ax_coe_lsf = [], []
    for i in range(poly_disp + 1):
        ax_coe_wave.append(fig_wave.add_subplot(gs[5:, i], sharey=None if i == 0 else ax_coe_wave[-1]))
        ax_coe_wave[-1].tick_params(labelleft=i == 0)
    ax_coe_wave = plot_wavesol_coeffs(numpy.arange(arc._fibers)[good_fibers], coeffs=wave_coeffs[good_fibers], axs=ax_coe_wave, labels=True)
    save_fig(fig_wave, product_path=out_wave, to_display=display_plots, figure_path='qa', label="fit_wave")

    # plot LSF fitting minus linear term
    fig_lsf = plt.figure(figsize=(16, 10), layout="constrained")
    gs = gridspec.GridSpec(10, poly_fwhm + 1, figure=fig_lsf)

    ax_sol_lsf = fig_lsf.add_subplot(gs[:5, :])
    ax_sol_lsf = plot_wavesol_lsf(xpix=arc._pixels, lsf=fwhm, lines_pixels=cent_wave,
                                  wave_poly=wave_cls, wave_coeffs=wave_coeffs, lsf_poly=fwhm_cls,
                                  lsf_coeffs=lsf_coeffs, ax=ax_sol_lsf, labels=True)

    for i in range(poly_fwhm + 1):
        ax_coe_lsf.append(fig_lsf.add_subplot(gs[5:, i], sharey=None if i == 0 else ax_coe_lsf[-1]))
        ax_coe_lsf[-1].tick_params(labelleft=i == 0)
    ax_coe_lsf = plot_wavesol_coeffs(numpy.arange(arc._fibers)[good_fibers], coeffs=lsf_coeffs[good_fibers], axs=ax_coe_lsf, color="tab:red", labels=True)
    save_fig(fig_lsf, product_path=out_wave, to_display=display_plots, figure_path='qa', label="fit_lsf")

    # update header
    log.info(
        f"updating header and writing wavelength/LSF to '{out_wave}' and '{out_lsf}'"
    )
    arc.setHdrValue(
        "HIERARCH PIPE DISP POLY",
        "%d" % (numpy.abs(poly_disp)),
        "Order of the dispersion polynomial",
    )
    arc.setHdrValue(
        "HIERARCH PIPE DISP RMS MEDIAN",
        "%.4f" % (numpy.median(wave_rms[good_fibers])),
        "Median RMS of disp sol",
    )
    arc.setHdrValue(
        "HIERARCH PIPE DISP RMS MIN",
        "%.4f" % (numpy.min(wave_rms[good_fibers])),
        "Min RMS of disp sol",
    )
    arc.setHdrValue(
        "HIERARCH PIPE DISP RMS MAX",
        "%.4f" % (numpy.max(wave_rms[good_fibers])),
        "Max RMS of disp sol",
    )
    arc.setHdrValue(
        "HIERARCH PIPE FWHM POLY",
        "%d" % (numpy.abs(poly_fwhm)),
        "Order of the resolution polynomial",
    )
    arc.setHdrValue(
        "HIERARCH PIPE DISP RMS MEDIAN",
        "%.4f" % (numpy.median(lsf_rms[good_fibers])),
        "Median RMS of disp sol",
    )
    arc.setHdrValue(
        "HIERARCH PIPE DISP RMS MIN",
        "%.4f" % (numpy.min(lsf_rms[good_fibers])),
        "Min RMS of disp sol",
    )
    arc.setHdrValue(
        "HIERARCH PIPE DISP RMS MAX",
        "%.4f" % (numpy.max(lsf_rms[good_fibers])),
        "Max RMS of disp sol",
    )

    mask = numpy.zeros(arc._data.shape, dtype=bool)
    mask[(~good_fibers)|(wave_coeffs==0).all(axis=1)] = True
    wave_trace = TraceMask(data=wave_sol, mask=mask, coeffs=wave_coeffs, header=arc._header.copy())
    wave_trace._samples = Table(data=cent_wave, names=ref_lines)
    wave_trace._header["IMAGETYP"] = "wave"
    mask = numpy.zeros(arc._data.shape, dtype=bool)
    mask[(~good_fibers)|(lsf_coeffs==0).all(axis=1)] = True
    fwhm_trace = TraceMask(data=lsf_sol, mask=mask, coeffs=lsf_coeffs, header=arc._header.copy())
    fwhm_trace._samples = Table(data=fwhm, names=ref_lines)
    fwhm_trace._header["IMAGETYP"] = "lsf"

    wave_trace.interpolate_coeffs()
    fwhm_trace.interpolate_coeffs()
    wave_trace.eval_coeffs()
    fwhm_trace.eval_coeffs()

    wave_trace.writeFitsData(out_wave)
    fwhm_trace.writeFitsData(out_lsf)

    return ref_lines, masked, cent_wave, fwhm_wave, arc, wave_trace, fwhm_trace

# method to apply shift in wavelength table based on comparison to skylines
def shift_wave_skylines(in_frame: str, out_frame: str, dwave: float = 8.0, skylinedict = REF_SKYLINES, display_plots: bool = False):
    """
    Applies shift to wavelength map extension based on sky line centroid measurements

    Parameters
    ----------
    in_frame : string
        Input RSS FITS file
    out_frame : string
        Output RSS FITS file with the shifted wavelength maps
    dwave : float, optional
        Wavelength window used to locate the sky line, by default 8.0
    skylinedict : dict[str, list[float]], optional
        Dictionary containing the list of reference sky lines per channel, by default REF_SKYLINES
    display_plots: bool, optional
        Display plots on screen, by default False
    """

    # print('************************************************')
    # print('***** CORRECTING WAVELENGTH USING SKYLINES *****')
    # print('************************************************')
    log.info("correcting wavelength using skylines")



    lvmframe = lvmFrame.from_file(in_frame)
    channel = lvmframe._header["CCD"][0]
    fiberid = lvmframe._slitmap['fiberid'].data
    # selection of which fibers belong to which spectrograph
    sel1 = lvmframe._slitmap['spectrographid'].data==1
    sel2 = lvmframe._slitmap['spectrographid'].data==2
    sel3 = lvmframe._slitmap['spectrographid'].data==3
    skylines = skylinedict[channel]

    # measure offsets
    offsets = numpy.ones((len(skylines), numpy.shape(lvmframe._data)[0])) * numpy.nan
    fiber_offset = numpy.ones(lvmframe._data.shape[0]) * numpy.nan
    iterator = tqdm(range(lvmframe._fibers), total=lvmframe._fibers, desc=f"measuring offsets using {len(skylines)} sky line(s)", ascii=True, unit="fiber")
    for ifiber in iterator:
        spec = lvmframe.getSpec(ifiber)
        if spec._mask.all() or lvmframe._slitmap[ifiber]["telescope"] == "Spec" or lvmframe._slitmap[ifiber]["fibstatus"] in [1, 2]:
            continue

        fwhm_guess = numpy.nanmean(numpy.interp(skylines, lvmframe._wave[ifiber], lvmframe._lsf[ifiber]))

        flux, sky_wave, fwhm, bg = spec.fitSepGauss(skylines, dwave, fwhm_guess, 0.0, [0, numpy.inf], [-2.5, 2.5], [fwhm_guess - 1.5, fwhm_guess + 1.5], [0.0, numpy.inf])
        if numpy.any(flux / bg < 0.7) or numpy.isnan([flux, sky_wave, fwhm]).any():
            continue

        offsets[:, ifiber] = sky_wave - skylines
        fiber_offset[ifiber] = numpy.nanmedian(offsets[:,ifiber], axis=0)

    # split per spectrographs
    specoffset = numpy.asarray(numpy.split(offsets, 3, axis=1))
    # fit smooth function to each spectrograph trend
    fiber_offset_mod = fiber_offset.copy()
    for spec_offset, spec in zip(numpy.split(fiber_offset, 3), [sel1, sel2, sel3]):
        mask = numpy.isfinite(spec_offset)
        t = numpy.linspace(
            fiberid[spec][mask][len(fiberid[spec][mask]) // 20],
            fiberid[spec][mask][-1 * len(fiberid[spec][mask]) // 20],
            20
        )
        median_offset = ndimage.median_filter(spec_offset[mask], 8)
        tck = interpolate.splrep(fiberid[spec][mask], median_offset, task=-1, t=t)
        fiber_offset_mod[spec] = interpolate.splev(fiberid[spec], tck)

    # Average offsets for different skylines in each channel, apply to trace, and write them in header
    meanoffset = numpy.nanmean(specoffset, axis=(1, 2)).round(4)
    log.info(f'Applying the offsets [Angstroms] in [1,2,3] spectrographs with means: {meanoffset}')
    lvmframe._wave_trace['COEFF'].data[:,0] -= fiber_offset_mod
    lvmframe._header[f'HIERARCH WAVE SKYOFF_{channel.upper()}1'] = (meanoffset[0], f'Mean sky line offset in {channel}1 [Angs]')
    lvmframe._header[f'HIERARCH WAVE SKYOFF_{channel.upper()}2'] = (meanoffset[1], f'Mean sky line offset in {channel}2 [Angs]')
    lvmframe._header[f'HIERARCH WAVE SKYOFF_{channel.upper()}3'] = (meanoffset[2], f'Mean sky line offset in {channel}3 [Angs]')

    wave_trace = TraceMask.from_coeff_table(lvmframe._wave_trace)
    lvmframe._wave = wave_trace.eval_coeffs()

    # write updated wobject
    log.info(f"writing updated wobject file '{os.path.basename(out_frame)}'")
    lvmframe._header["DRPSTAGE"] = (ReductionStage(lvmframe._header["DRPSTAGE"]) + "WAVELENGTH_SHIFTED").value
    lvmframe.writeFitsData(out_frame)

    # Make QA plots showing offsets for each sky line in each spectrograph
    fig, ax = plt.subplots()
    ax.hlines(+0.05, 1, 1944, linestyle=':', color='black', alpha=0.3)
    ax.hlines(-0.05, 1, 1944, linestyle=':', color='black', alpha=0.3)

    colors = plt.cm.coolwarm(numpy.linspace(0, 1, len(skylines)))
    for i in range(len(skylines)):
        ax.plot(fiberid[sel1], ndimage.median_filter(offsets[i,sel1], 5), color=colors[i], lw=1, alpha=0.7)
        ax.plot(fiberid[sel2], ndimage.median_filter(offsets[i,sel2], 5), color=colors[i], lw=1, alpha=0.7)
        ax.plot(fiberid[sel3], ndimage.median_filter(offsets[i,sel3], 5), color=colors[i], lw=1, alpha=0.7)
    ax.hlines(meanoffset[0], 1, 648, linestyle='-', color='tab:red', label='Spec1')
    ax.hlines(meanoffset[1], 1+648, 2*648, linestyle='-', color='tab:green', label='Spec2')
    ax.hlines(meanoffset[2], 1+2*648, 1944, linestyle='-', color='tab:blue', label='Spec3')
    ax.plot(fiberid[sel1], fiber_offset_mod[sel1], color='0.2')
    ax.plot(fiberid[sel2], fiber_offset_mod[sel2], color='0.2')
    ax.plot(fiberid[sel3], fiber_offset_mod[sel3], color='0.2')
    ax.hlines(0, 1, 1944, linestyle='--', color='black', alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.4,0.4)
    ax.set_title(f'{lvmframe._header["EXPOSURE"]} - {channel} - {numpy.round(skylines, 2)}')
    ax.set_xlabel('Fiber ID')
    ax.set_ylabel(r'$\Delta \lambda [\AA]$')

    save_fig(
        fig,
        product_path=out_frame,
        to_display=display_plots,
        figure_path="qa",
        label=f"skylineshift_{channel}")



# TODO:
# * merge arc_wave and arc_lsfs into lvmArc product, change variable name to in_arc
# @skip_on_missing_input_path(["in_rss", "in_waves", "in_lsfs"])
# @skip_if_drpqual_flags(["EXTRACTBAD", "BADTRACE"], "in_rss")
def create_pixel_table(in_rss: str, out_rss: str, in_waves: str, in_lsfs: str):
    """
    Applies the wavelength and the spectral resolution (LSF) to an RSS

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file
    out_rss : string
        Output RSS FITS file with the wavelength and spectral resolution pixel
        table added as extensions
    in_waves : string
        RSS FITS file containing the wavelength solutions
    in_lsfs : string, optional with default: ''
        RSS FITS file containing the spectral resolution (LSF in FWHM)
    """
    rss = RSS.from_file(in_rss)

    wave_traces = [TraceMask.from_file(in_wave) for in_wave in in_waves]
    wave_trace = TraceMask.from_spectrographs(*wave_traces)
    rss.set_wave_trace(wave_trace)

    lsf_traces = [TraceMask.from_file(in_lsfs) for in_lsfs in in_lsfs]
    lsf_trace = TraceMask.from_spectrographs(*lsf_traces)
    rss.set_lsf_trace(lsf_trace)
    rss._header["DRPSTAGE"] = (ReductionStage(rss._header["DRPSTAGE"]) + "WAVELENGTH_CALIBRATED").value
    rss.writeFitsData(out_rss)

    return rss


def checkPixTable_drp(
    in_rss, ref_lines, logfile, blocks="15", init_back="100.0", aperture="10"
):
    """
    Measures the offset in dispersion axis between the target and the calibration RSS.

    It compares the wavelength of emission lines (i.e. night sky line) in the
    RSS as measured by Gaussian fitting with their reference wavelength. The
    offset in pixels is computed and stored in a log file for future
    processing.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file
    ref_lines : string
        Comma-separated list of emission lines to be fit in the RSS for each fiber
    logfile : string
        Output ASCII logfile that stores the position and deviation from the
        expected wavelength for each reference emission line
    blocks: string of integer, optional  with default: '20'
        Number of fiber blocks over which the fitted central wavelength are
        averaged to increase the accuary of the measurement. The actual number
        of fibers per block is roughly the total number of fibers divided by
        the number of blocks.
    init_back : string of float, optional with default: '100.0'
        The initial guess for a constant background level included in the
        Gaussian model. If this parameter is empty, no background level is
        fitted and set to zero instead.
    aperture : string of integer (>0), optional with default: '10'
        Number of pixel used for the fitting of each emission line centered on
        the pixel position of the line's expected wavelength

    Examples
    --------
    user:> lvmdrp rss checkPixTable RSS_IN.fits 4500.0,5577.4,6300.3 OFFSETWAVE.log
    user:> lvmdrp rss checkPixTable RSS_IN.fits 4500.0,5577.4,6300.3 OFFSETWAVE.log /
    > aperture=14
    """
    centres = numpy.array(ref_lines.split(",")).astype("float")
    init_back = float(init_back)
    aperture = float(aperture)
    nblocks = int(blocks)
    rss = RSS.from_file(in_rss)
    fit_wave = numpy.zeros((len(rss), len(centres)), dtype=numpy.float32)
    good_fiber = numpy.zeros(len(rss), dtype="bool")
    offset_pix = numpy.zeros((len(rss), len(centres)), dtype=numpy.float32)
    log = open(logfile, "a")
    log.write("%s\n" % (in_rss))

    for i in range(len(rss)):
        spec = rss[i]
        disp_pix = spec._wave[1:] - spec._wave[:-1]
        numpy.insert(disp_pix, 0, disp_pix[0])

        plot = False
        if i == -1:
            plot = True
        else:
            plot = False
        out = spec.fitSepGauss(
            centres, aperture, init_back=init_back, plot=plot, warning=False
        )
        good_fiber[i] = out[0] != 0.0
        fit_wave[i, :] = out[len(centres) : 2 * len(centres)]
        for j in range(len(centres)):
            idx = numpy.argmin(numpy.abs(fit_wave[i, j] - spec._wave))
            offset_pix[i, j] = (fit_wave[i, j] - centres[j]) / disp_pix[idx]

    blocks = numpy.array_split(numpy.arange(0, len(rss)), nblocks)
    blocks_good = numpy.array_split(good_fiber, nblocks)
    for j in range(len(centres)):
        log.write(
            "%.3f %.3f %.3f %.3f \n"
            % (
                centres[j],
                numpy.median(fit_wave[good_fiber, j]),
                numpy.median(fit_wave[good_fiber, j]) - centres[j],
                numpy.std(fit_wave[good_fiber, j]),
            )
        )
        for i in range(len(blocks)):
            log.write(" %.3f" % numpy.mean(blocks[i]))
        log.write("\n")
        for i in range(len(blocks)):
            if numpy.sum(blocks_good[i]) > 0:
                log.write(
                    " %.3f" % numpy.median(offset_pix[blocks[i][blocks_good[i]], j])
                )
            else:
                log.write(" 0.0")
        log.write("\n")
        for i in range(len(blocks)):
            if numpy.sum(blocks_good[i]) > 0:
                log.write(
                    " %.3f"
                    % (
                        numpy.median(fit_wave[blocks[i][blocks_good[i]], j])
                        - centres[j]
                    )
                )
            else:
                log.write(" 0.0")
        log.write("\n")

    off_disp_median = numpy.median(offset_pix[good_fiber, :])
    off_disp_rms = numpy.std(offset_pix[good_fiber, :])
    off_disp_median = (
        float("%.4f" % off_disp_median)
        if numpy.isfinite(off_disp_median)
        else str(off_disp_median)
    )
    off_disp_rms = (
        float("%.4f" % off_disp_rms)
        if numpy.isfinite(off_disp_rms)
        else str(off_disp_rms)
    )
    rss.setHdrValue(
        "HIERARCH PIPE FLEX XOFF", off_disp_median, "flexure offset in x-direction"
    )
    rss.setHdrValue(
        "HIERARCH PIPE FLEX XRMS", off_disp_rms, "flexure rms in x-direction"
    )
    rss.writeFitsHeader(in_rss)
    log.close()


def correctPixTable_drp(
    in_rss,
    out_rss,
    logfile,
    ref_id,
    smooth_poly_cross="",
    smooth_poly_disp="",
    poly_disp="6",
    verbose="0",
):
    """
    Corrects the RSS wavelength vectors for offsets in dispersion axis due to flexures

    The offfsets need to be determined beforehand via the checkPixTable task.
    The offsets can be smoothed and/or extrapolated along the dispersion axis,
    but a global median offset is strongly recommended due to measurement
    inaccuracies.

    Parameters
    --------------
    in_rss : string
        Input RSS FITS file
    out_rss : string
        Output RSS FITS file with corrected wavelength pixel table
    logfile : string
        Input ASCII logfile containing the previously measured offset for
        certain reference emission line in dispersion direction
    ref_id : string
        Reference ID under which the offsets are stored in the logfile for this
        specific RSS
    smooth_poly_cross : string of integer, optional with default: ''
        Degree of the polynomial which is used to smooth the offset value for
        each reference emission line as a function of fiber number (i.e. along
        cross-disperion direction on the CCD) (positiv: normal polynomial,
        negative: Legandre polynomial) No smoothing is performed if this
        parameter is empty.
    smooth_poly_disp : string of integer, optional with default: ''
        Degree of the polynomial which is used to extrapolated the offsets
        along the wavelength direction for each block of fibers individually.
        (positiv: normal polynomial, negative: Legandre polynomial) A median
        value of all measured shifts is used if this parameter is empty.
    poly_disp : string of integer (>0), optional with default: '6'
        Degree of polynomial used to construct the wavelength solution. This is
        needed to properly shift the wavelength table according to the offset
        in pixel units.
    verbose: string of integer (0 or 1), optional  with default: 1
        Show information during the processing on the command line (0 - no, 1 -
        yes)

    Examples
    --------
    user:> lvmdrp rss correctPixTable RSS_in.fits RSS_out.fits OFFSETWAVE.log /
    > RSS_REF_ID poly_disp=7
    """

    poly_disp = int(poly_disp)
    verbose = int(verbose)

    rss = loadRSS(in_rss)
    log = open(logfile, "r")
    log_lines = log.readlines()
    m = 0
    offsets = []
    for i in range(len(log_lines)):
        if ref_id in log_lines[i]:
            ref_wave = []
            offsets = []
            m = 1
        if m == 1 and len(log_lines[i].split()) == 4:
            ref_wave.append(float(log_lines[i].split()[0]))
            fibers = numpy.array(log_lines[i + 1].split()).astype("float")
            offset_pix = numpy.array(log_lines[i + 2].split()).astype("float")
            # offset_wave = numpy.array(log_lines[i + 3].split()).astype("float")

            if smooth_poly_cross == "":
                offsets.append(offset_pix)

            else:
                smooth_poly_cross = int(smooth_poly_cross)
                spec = Spectrum1D(data=offset_pix, wave=fibers)
                spec.smoothPoly(
                    order=smooth_poly_cross, ref_base=numpy.arange(rss._fibers)
                )
                if verbose == 1:
                    plt.plot(fibers, offset_pix, "o")
                    plt.plot(numpy.arange(rss._fibers), spec._data)
                offsets.append(spec._data)

        if len(log_lines[i].split()) == 1 and (ref_id not in log_lines[i]) and m == 1:
            m = 0
    if verbose == 1:
        plt.show()
    offsets = numpy.array(offsets)
    ref_wave = numpy.array(ref_wave)
    for i in range(rss._fibers):
        spec = rss[i]
        if smooth_poly_disp == "":
            off = numpy.median(offsets.flatten())
        else:
            smooth_poly_disp = int(smooth_poly_disp)
            if smooth_poly_disp == "":
                off = numpy.median(offsets[i])
            else:
                off = Spectrum1D(wave=ref_wave, data=offsets[:, i])
                off.smoothPoly(smooth_poly_disp, ref_base=spec._wave)
                if i == -1:
                    plt.plot(ref_wave, offsets[:, i], "ok")
                    plt.plot(off._wave, off._data, "-r")
                    plt.show()
                off = off._data
        new_wave = Spectrum1D(spec._pixels + off, spec._wave)
        new_wave.smoothPoly(poly_disp, ref_base=spec._pixels)
        spec._wave = new_wave._data
        rss[i] = spec
    rss.writeFitsData(out_rss)

# TODO: aplicar correccion a la solucion de longitud de onda comparando lineas de cielo
# TODO: hacer esto antes de hacer el rasampling en wl
@skip_on_missing_input_path(["in_rss"])
@skip_if_drpqual_flags(["BADTRACE", "EXTRACTBAD"], "in_rss")
def resample_wavelength(in_rss: str, out_rss: str, method: str = "linear",
                        wave_range: Tuple[float,float] = None, wave_disp: float = None,
                        helio_vel: float = 0.0, helio_vel_keyword: str = "HELIO_RV",
                        convert_to_density: bool = False) -> RSS:
    """Resamples the RSS wavelength solutions to a common wavelength solution

    A common wavelength solution is computed for the RSS by resampling the
    wavelength solution of each fiber to a common wavelength grid. The
    resampling is performed using a linear or spline interpolation scheme.
    Additionally barycentric correction in velocity can be applied if
    information is supplied as input parameter or in the header of the input
    RSS.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file where the wavelength is stored as a pixel table
    out_rss : string
        Output RSS FITS file with a common wavelength solution
    method : string, optional with default: 'linear'
        Interpolation scheme used for the spectral resampling of the data.
        Available options are:
            - linear
            - spline
    wave_range : string of float, optional with default: None
        Wavelength range of the common resampled wavelength solution. If the
        parameter is empty, the wavelength range of the input RSS is used.
    wave_disp : string of float, optional with default: None
        Dispersion per pixel for the common resampled wavelength solution.
        The "optimal" dispersion will be used if the parameter is empty.
    helio_vel : string of float, optional with default: 0.0
        Heliocentric velocity in km/s. If the parameter is empty, the value
        stored in the header of the input RSS is used.
    helio_vel_keyword : string, optional with default: 'HELIO_RV'
        Keyword in the header of the input RSS where the heliocentric velocity
        is stored.
    convert_to_density : string of boolean, optional with default: False
        If True, the resampled RSS will be converted to density units.

    Returns
    -------
    RSS : lvmdrp.core.rss.RSS
        Resampled RSS with a common wavelength solution
    """

    # load input RSS
    log.info(f"reading target data from '{os.path.basename(in_rss)}'")
    rss = loadRSS(in_rss)

    # define wavelength grid
    if wave_range is None or len(wave_range) < 2:
        start_wave = numpy.min(rss._wave)
        end_wave = numpy.max(rss._wave)
        wave_range = (start_wave, end_wave)
    if wave_disp is None:
        wave_disp = numpy.min(rss._wave[:, 1:] - rss._wave[:, :-1])
    log.info(f"using wavelength range {wave_range = } angstrom and {wave_disp = } angstrom pixel size")

    # apply heliocentric velocity correction
    if helio_vel is None or helio_vel == 0.0:
        helio_vel = rss._header.get(helio_vel_keyword)
        if helio_vel is None:
            helio_vel = 0.0
            log.warning(f"no heliocentric velocity found in header by keywords {helio_vel_keyword = }, assuming {helio_vel = } km/s")
            rss.add_header_comment(f"no heliocentric velocity {helio_vel_keyword = }, assuming {helio_vel = } km/s")
    else:
        log.info(f"applying heliocentric velocity correction of {helio_vel = } km/s")

    rss._wave = rss._wave * (1 + helio_vel / c.to("km/s").value)

    # resample the wavelength solution
    log.info(f"resampling the wavelength solution using {method = } interpolation")
    new_rss = rss.rectify_wave(wave_range=wave_range, wave_disp=wave_disp, method=method, return_density=convert_to_density)

    # write output RSS
    log.info(f"writing resampled RSS to '{os.path.basename(out_rss)}'")
    new_rss.writeFitsData(out_rss)

    return new_rss


def match_resolution(in_rss, out_rss, target_fwhm=None, min_fwhm=0.1, plot_fibers=[0,300,600,900,1200,1400,1700], display_plots=False):
    """
    Homogenise the LSF of the RSS to a common spectral resolution (FWHM)

    This routine downgrades the RSS LSF with a Gaussian kernel of the
    corresponding width. A pixel table with the spectral resolution needs to be
    present in the RSS. If the spectral resolution is higher than than the
    target spectral resolution for certain pixel, the spectra is degraded to
    `min_fwhm` value.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file with a pixel table for the spectral resolution
    out_rss : string
        Output RSS FITS file with a homogenised spectral resolution
    target_fwhm : float, optional
        Spectral resolution in FWHM Agnstroms to which the RSS will be homogenised, by default None
    min_fwhm : float, optional
        Minimum spectral resolution in FWHM allowed, by default 0.1 Angstrom
    plot_fibers : list[int], optional
        List of fiber indices to plot, by default [0,300,600,900,1200,1400,1700]
    display_plots : bool, optional
        Show plot on screen or not, by default False

    Returns
    -------
    new_rss : lvmdrp.core.rss.RSS
        New RSS with homogenised LSF
    """
    rss = RSS.from_file(in_rss)
    camera = rss._header["CCD"]
    expnum = rss._header["EXPOSURE"]

    new_rss = rss.match_lsf(target_fwhm, min_fwhm=min_fwhm)
    new_rss._lsf = None
    new_rss.setHdrValue("HIERARCH WAVE RES", target_fwhm, "spectral resolution (FWHM) [Angstrom]")
    new_rss.writeFitsData(out_rss)

    if plot_fibers:
        fig, ax = create_subplots(to_display=display_plots, figsize=(15,5), layout="constrained")
        fig.suptitle(f"Matched LSF for {camera = }, {expnum = }")
        for ifiber in plot_fibers:
            wave = rss._wave if len(rss._wave.shape) == 1 else rss._wave[ifiber]
            ln, = ax.step(wave, rss._data[ifiber], lw=1, where="mid", alpha=0.5)
            ax.step(wave, new_rss._data[ifiber], lw=1, where="mid", color=ln.get_color(), label=ifiber)
        ax.legend(loc=1, frameon=False, title="Fiber Idx", ncols=7)
        save_fig(fig, to_display=display_plots, product_path=out_rss, figure_path="qa", label="match_res")

    return new_rss


def splitFibers_drp(in_rss, splitted_out, contains):
    """
    Subtracts a (sky) spectrum, which was stored as a FITS file, from the whole RSS.
    The error will be propagated if the spectrum AND the RSS contain error information.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file including a position table as an extension
    splitted_out : string
        Comma-separated list of output RSS FITS files
    contains : string
        Comma-Separated list of fiber "types" included in the respective output
        RSS file. Available fiber types are OBJ, SKY and CAL, corresponding to
        target fibers, dedicated sky fibers and calibration fibers,
        respectively. If more than one type of fibers should be contained in
        one of the splitted RSS, they need to be ";" separated.

    Examples
    ----------------
    user:> lvmdrp rss splitFibers RSS_IN.fits RSS_OBJ.fits,RSS_SKY.fits SKY,OBJ
    user:> lvmdrp rss splitFibers RSS_IN.fits RSS_OBJ_SKY.fits,RSS_CAL.fits SKY;OBJ,SKY
    """
    contains = contains.split(",")
    splitted_out = splitted_out.split(",")
    rss = RSS.from_file(in_rss)
    splitted_rss = rss.splitFiberType(contains)
    for i in range(len(splitted_rss)):
        splitted_rss[i].writeFitsData(splitted_out[i])


# TODO: for twilight fiber flats, normalize the individual flats before combining to
# remove the time dependence
def create_fiberflat(in_rsss: List[str], out_rsss: List[str], median_box: int = 0,
                     gaussian_kernel: int = 5,
                     poly_deg: int = 0, poly_kind: str = "poly",
                     clip_range: List[float] = None,
                     wave_range: List[float] = None,
                     illumination_corr: bool = False,
                     display_plots: bool = False) -> RSS:
    """computes a fiberflat from a wavelength calibrated continuum exposure

    This function computes a fiberflat from a extracted and wavelength calibrated
    continuum exposure. The fiberflat is computed by dividing the continuum
    exposure by the median spectrum of the continuum exposure. The fiberflat
    can be smoothed with a median filter and a gaussian kernel. The fiberflat
    can be fitted with a polynomial along the dispersion axis for each fiber
    individually. The fiberflat can be clipped to a given range of relative
    transmission values.


    Parameters
    ----------
    in_rss : list
        paths to a extracted and wavelength calibrated continuum exposures for one spectrograph channel
    out_rss : list
        paths to the outputs fiberflat
    median_box : int, optional
        size along dispersion direction (angstroms) of the median box, by default 0
    gaussian_kernel : int, optional
        standard deviation of the Gaussian kernel along dispersion direction (angstroms), by default 5
    poly_deg : int, optional
        polynomial degree used to fit the fiberflat, by default 0
    poly_kind : str, optional
        type of polynomial used to fit the fiberflat, by default "poly"
    clip_range : List[float], optional
        range of valid values for the fiber flat, by default None
    wave_range : List[float], optional
        wavelength range where the median spectrum is computed, by default None
    illumination_corr : bool, optional
        whether to apply an illumination correction to the fiberflat, by default False
    display_plots : bool, optional
        whether to display or not the diagnostic plots, by default False

    Returns
    -------
    RSS
        computed fiberflat spectrograph-combined
    """
    # read continuum exposure
    wave, data, error, mask = [], [], [], []
    headers = []
    fibers = []
    j = 1
    for in_rss in in_rsss:
        log.info(f"reading continuum exposure from {os.path.basename(in_rss)}")
        rss = loadRSS(in_rss)
        wave.append(rss._wave)
        data.append(rss._data)
        error.append(rss._error)
        mask.append(rss._mask)
        fibers.append(numpy.zeros(rss._fibers) + j)
        headers.append(rss._header)
        j += 1
    rss = RSS(wave=numpy.vstack(wave), data=numpy.vstack(data), error=numpy.vstack(error), mask=numpy.vstack(mask))
    rss._data = numpy.clip(rss._data, 0, None)
    fibers = numpy.vstack(fibers).flatten()

    # extract useful metadata
    channel = headers[0]["CCD"][0]
    unit = headers[0]["BUNIT"]

    # wavelength calibration check
    if rss._wave is None:
        log.error(f"RSS {os.path.basename(in_rss)} has not been wavelength calibrated")
        return None
    else:
        wdelt = numpy.diff(rss._wave, axis=1).mean()

    # copy original data into output fiberflat object
    fiberflat = copy(rss)
    fiberflat._error = None

    # apply median smoothing to data
    if median_box > 0:
        median_box_pix = int(median_box / wdelt)
        log.info(f"applying median smoothing with box size {[1, median_box]} angstroms ({[1, median_box_pix]} pixels)")
        fiberflat._data = ndimage.filters.median_filter(fiberflat._data, (1, median_box_pix))

    # calculate median spectrum
    log.info(f"caculating normalization in full wavelength range ({fiberflat._wave.min():.2f} - {fiberflat._wave.max():.2f} angstroms)")
    norm = bn.nanmedian(fiberflat._data, axis=0)
    norm_wave = bn.nanmedian(fiberflat._wave, axis=0)

    # clip wavelength range for median spectrum
    if wave_range is not None:
        log.info(f"limiting wavelength range to {wave_range[0]:.2f} - {wave_range[1]:.2f} angstroms")
        wave_select = (wave_range[0] <= norm_wave) & (norm_wave <= wave_range[1])
        norm[~wave_select] = numpy.nan

    # normalize fibers where norm has valid values
    log.info(f"computing fiberflat across {fiberflat._fibers} fibers and {(~numpy.isnan(norm)).sum()} wavelength bins")
    normalized = fiberflat._data / norm[None, :]
    fiberflat._data = normalized
    fiberflat._mask |= normalized <= 0

    # apply clipping
    if clip_range is not None:
        log.info(f"cliping fiberflat to range {clip_range[0]:.2f} - {clip_range[1]:.2f}")
        fiberflat._data = numpy.clip(fiberflat._data, clip_range[0], clip_range[1])

    # apply gaussian smoothing
    if gaussian_kernel > 0:
        gaussian_kernel_pix = int(gaussian_kernel / wdelt)
        log.info(f"applying gaussian smoothing with kernel size {gaussian_kernel} angstroms ({gaussian_kernel_pix} pixels)")
        for ifiber in range(rss._fibers):
            spec = fiberflat.getSpec(ifiber)
            spec.smoothSpec(gaussian_kernel_pix, method="gauss")
            fiberflat._data[ifiber, :] = spec._data

    # polynomial smoothing
    if poly_deg != 0:
        log.info(f"applying polynomial fitting with degree {poly_deg} and kind '{poly_kind}'")
        for ifiber in range(fiberflat._fibers):
            spec = fiberflat.getSpec(ifiber)
            spec.smoothPoly(deg=poly_deg, poly_kind=poly_kind)
            fiberflat._data[ifiber, :] = spec._data

    # interpolate masked pixels in fiberflat
    for ifiber in range(fiberflat._fibers):
        wave, data, mask = fiberflat._wave[ifiber], fiberflat._data[ifiber], fiberflat._mask[ifiber]
        mask |= ~numpy.isfinite(data)
        if numpy.sum(~mask) == 0:
            continue
        fiberflat._data[ifiber, mask] = interpolate.interp1d(wave[~mask], data[~mask], bounds_error=False, assume_sorted=True)(wave[mask])
        fiberflat._mask[ifiber, mask] = False

    # create diagnostic plots
    log.info("creating diagnostic plots for fiberflat")
    fig, axs = create_subplots(to_display=display_plots, nrows=3, ncols=1, figsize=(12, 15), sharex=True)
    # plot original continuum exposure, fiberflat and corrected fiberflat per fiber
    colors = plt.cm.Spectral(numpy.linspace(0, 1, fiberflat._fibers))
    rss._data[rss._mask] = numpy.nan
    stdev_ori = biweight_scale(rss._data, axis=0, ignore_nan=True)[1950:2050].mean()
    stdev_new = biweight_scale(rss._data/fiberflat._data, axis=0, ignore_nan=True)[1950:2050].mean()
    log.info(f"flatfield statistics: {stdev_ori = :.2f}, {stdev_new = :.2f} ({stdev_new/stdev_ori*100:.2f}%)")
    for ifiber in range(fiberflat._fibers):
        # input data
        axs[0].step(rss._wave[ifiber], rss._data[ifiber], color=colors[ifiber], alpha=0.5, lw=1)
        # fiberflat
        axs[1].step(fiberflat._wave[ifiber], fiberflat._data[ifiber], lw=1, color=colors[ifiber])
        # corrected fiberflat
        axs[2].step(fiberflat._wave[ifiber], rss._data[ifiber] / fiberflat._data[ifiber], lw=1, color=colors[ifiber])
    # plot median spectrum
    axs[0].step(norm_wave, norm, color="0.1", lw=2, label="median spectrum")
    axs[2].step(norm_wave, norm, color="0.1", lw=2, label="median spectrum")
    # add labels and titles and set axis limits
    ymax = bn.nanmean(norm) + bn.nanstd(rss._data) * 3
    axs[0].set_ylim(0, ymax)
    axs[0].set_ylabel(f"counts ({unit})")
    axs[0].set_title("median spectrum", loc="left")
    axs[1].set_ylim(0, 3.0)
    axs[1].set_ylabel("relative transmission")
    axs[1].set_title("fiberflat", loc="left")
    axs[2].set_ylim(0, ymax)
    axs[2].set_ylabel(f"corr. counts ({unit})")
    axs[2].set_title("corrected fiberflat", loc="left")
    axs[2].set_xlabel("wavelength (angstroms)")
    fig.suptitle(f"fiberflat creation for {channel = }", fontsize=16)
    # display/save plots
    save_fig(
        fig,
        product_path=out_rsss[0],
        to_display=display_plots,
        figure_path="qa",
        label="fiberflat"
    )

    for i in range(len(in_rsss)):
        log.info(f"writing fiberflat to {os.path.basename(out_rsss[i])}")
        spec_mask = (fibers == (i+1))
        wave_cam = fiberflat._wave[spec_mask, :]
        data_cam = fiberflat._data[spec_mask, :]
        mask_cam = fiberflat._mask[spec_mask, :]

        fiberflat_cam = RSS(wave=wave_cam, data=data_cam, mask=mask_cam, header=headers[i])

        # apply illumination correction if requested
        if illumination_corr:
            _, factors = _illumination_correction(fiberflat_cam, apply_correction=True)
            log.info(f"telescope illumination correction factors: {factors}")

        # perform some statistic about the fiberflat
        if fiberflat_cam._mask is not None:
            select = numpy.logical_not(fiberflat_cam._mask)
        else:
            select = fiberflat_cam._data == fiberflat_cam._data
        min = bn.nanmin(fiberflat_cam._data[select])
        max = bn.nanmax(fiberflat_cam._data[select])
        mean = bn.nanmean(fiberflat_cam._data[select])
        median = bn.nanmedian(fiberflat_cam._data[select])
        std = bn.nanstd(fiberflat_cam._data[select])
        log.info(f"fiberflat statistics: {min = :.3f}, {max = :.3f}, {mean = :.2f}, {median = :.2f}, {std = :.3f}")

        fiberflat_cam.setHdrValue(
            "HIERARCH PIPE FLAT MIN", float("%.3f" % (min)), "Mininum fiberflat value"
        )
        fiberflat_cam.setHdrValue(
            "HIERARCH PIPE FLAT MAX", float("%.3f" % (max)), "Maximum fiberflat value"
        )
        fiberflat_cam.setHdrValue(
            "HIERARCH PIPE FLAT AVR", float("%.2f" % (mean)), "Mean fiberflat value"
        )
        fiberflat_cam.setHdrValue(
            "HIERARCH PIPE FLAT MED", float("%.2f" % (median)), "Median fiberflat value"
        )
        fiberflat_cam.setHdrValue(
            "HIERARCH PIPE FLAT STD", float("%.3f" % (std)), "rms of fiberflat values"
        )
        fiberflat_cam._header["BUNIT"] = "dimensionless"
        fiberflat_cam._header["IMAGETYP"] = "fiberflat"
        fiberflat_cam.writeFitsData(out_rsss[i])

    return fiberflat


def correctTraceMask_drp(trace_in, trace_out, logfile, ref_file, poly_smooth=""):
    """
    Corrects the trace mask of the central fiber position for possible offsets
    in cross-dispersion direction due to flexure effects with respect to the
    calibration frames taken for this object. The offfsets need to be
    determined beforehand via the offsetTrace task. The offsets can be smoothed
    and/or extrapolated along the dispersion axis.

    Parameters
    ----------
    trace_in : string
        Input RSS FITS file containing the traces of the fiber position on the CCD
    trace_out : string
        Output RSS FITS file with offset corrected fiber position traces
    logfile : string
        Input ASCII logfile containing the previously measured offset for
        certain reference emission line in cross-dispersion direction
    ref_file : string
        Reference file under which the offsets are stored in the logfile for
        this specific RSS
    poly_smooth: string of integer, optional with default: ''
        Degree of the polynomial which is used to smooth/extrapolate the
        offsets as a function of wavelength (positiv: normal polynomial,
        negative: Legandre polynomial) No smoothing is performed if this
        parameter is empty an a median offset is used instead.

    Examples
    --------
    user:> lvmdrp rss correctTraceMask TRACE_IN.fits TRACE_OUT.fits OFFSET_TRACE.log /
     > REF_File_name

    user:> lvmdrp rss correctTraceMask TRACE_IN.fits TRACE_OUT.fits OFFSET_TRACE.log /
    > REF_File_name poly_smooth= -6
    """
    log = open(logfile, "r")
    log_lines = log.readlines()
    i = 0
    while i < len(log_lines):
        split = log_lines[i].split()
        z = 1
        if len(split) == 1 and split[0] == ref_file:
            offsets = []
            lines = []
            cross_pos = []
            disp_pos = []
            while i + z < len(log_lines):
                split1 = log_lines[i + z].split()
                split2 = log_lines[i + z + 1].split()
                split3 = log_lines[i + z + 2].split()
                if len(split1) > 1:
                    offsets.append(numpy.array(split3[1:]).astype("float32"))
                    cross_pos.append(numpy.array(split1[1:]).astype("float32"))
                    disp_pos.append(numpy.array(split2[1:]).astype("float32"))
                    lines.append(float(split3[0]))
                else:
                    # incomplete = False
                    break
                z += 3
            break
        i += 1

        log.close()
    offsets = numpy.array(offsets)
    cross_pos = numpy.array(cross_pos)
    disp_pos = numpy.array(disp_pos)
    trace = TraceMask.from_file(trace_in)

    if poly_smooth == "":
        trace = trace + (numpy.median(offsets.flatten()) * -1)
    else:
        split_trace = trace.split(offsets.shape[1], axis="y")
        offset_trace = TraceMask()
        offset_trace.createEmpty(data_dim=trace._data.shape)
        for j in range(len(split_trace)):
            offset_spec = Spectrum1D(wave=disp_pos[:, j], data=offsets[:, j])
            wave = numpy.arange(trace._data.shape[1])
            offset_spec.smoothPoly(
                order=int(poly_smooth),
                start_wave=wave[0],
                end_wave=wave[-1],
                ref_base=wave,
            )
            if j > 0:
                corr_trace = split_trace[j] + (offset_spec * -1)
                offset_trace.append(corr_trace)
            else:
                offset_trace = split_trace[j] + (offset_spec * -1)

        trace = offset_trace

    trace.writeFitsData(trace_out)


def apply_fiberflat(in_rss: str, out_frame: str, in_flat: str, clip_below: float = 0.0) -> RSS:
    """applies fiberflat correction to target RSS file

    This function applies a fiberflat correction to a target RSS file. The
    fiberflat correction is computed by the create_fiberflat function. The
    fiberflat correction is applied by dividing the target RSS by the fiberflat
    RSS. The fiberflat RSS needs to have the same number of fibers and the same
    wavelength grid as the target RSS.

    Parameters
    ----------
    in_rss : str
        input RSS file path to be corrected
    out_frame : str
        output lvmFrame file path with fiberflat correction applied
    in_flat : str
        input RSS file path to the fiberflat
    clip_below : float, optional
        minimum relative transmission considered. Values below will be masked, by default 0.0

    Returns
    -------
    RSS
        fiberflat corrected RSS
    """
    # load target data
    log.info(f"reading target data from {os.path.basename(in_rss)}")
    rss = RSS.from_file(in_rss)

    # compute initial variance
    ifibvar = bn.nanmean(bn.nanvar(rss._data, axis=0))

    # load fiberflat
    log.info(f"reading fiberflat from {os.path.basename(in_flat)}")
    flat = RSS.from_file(in_flat)
    if flat._wave is None:
        flat.set_wave_trace(rss._wave_trace)
        flat.set_wave_array()

    # check if fiberflat has the same number of fibers as the target data
    if rss._fibers != flat._fibers:
        log.error(f"number of fibers in target data ({rss._fibers}) and fiberflat ({flat._fibers}) do not match")
        return None

    # check if fiberflat has the same wavelength grid as the target data
    if not numpy.isclose(rss._wave, flat._wave).all():
        log.warning("target data and fiberflat have different wavelength grids")
        rss.add_header_comment("target data and fiberflat have different wavelength grids")

    # apply fiberflat
    log.info(f"applying fiberflat correction to {rss._fibers} fibers with minimum relative transmission of {clip_below}")
    for i in range(flat._fibers):
        # extract fibers spectra
        spec_flat = flat.getSpec(i)
        spec_data = rss.getSpec(i)

        # interpolate fiberflat to target wavelength grid to fill in missing values
        if not numpy.isclose(spec_flat._wave, spec_data._wave).all():
            deltas = spec_flat._wave - spec_data._wave
            log.warning(f"at fiber {i} resampling fiberflat: {numpy.min(deltas):.4f} - {numpy.max(deltas):.4f}")
            rss.add_header_comment(f"at fiber {i} resampling fiberflat: {numpy.min(deltas):.4f} - {numpy.max(deltas):.4f}")
            spec_flat = spec_flat.resampleSpec(spec_data._wave, err_sim=5)

        # apply clipping
        select_clip_below = (spec_flat < clip_below) | numpy.isnan(spec_flat._data)
        spec_flat._data[select_clip_below] = 1
        # if spec_flat._mask is not None:
        #     spec_flat._mask[select_clip_below] = True

        # correct
        spec_new = spec_data / spec_flat._data
        rss.setSpec(i, spec_new)

    # compute final variance
    ffibvar = bn.nanmean(bn.nanvar(rss._data, axis=0))

    # load ancillary data
    log.info(f"writing lvmFrame to {os.path.basename(out_frame)}")

    # create lvmFrame
    lvmframe = lvmFrame(
        data=rss._data,
        error=rss._error,
        mask=rss._mask,
        cent_trace=rss._cent_trace,
        width_trace=rss._width_trace,
        wave_trace=rss._wave_trace,
        lsf_trace=rss._lsf_trace,
        slitmap=rss._slitmap,
        superflat=flat._data
    )
    lvmframe.set_header(orig_header=rss._header, flatname=os.path.basename(in_flat), ifibvar=ifibvar, ffibvar=ffibvar)
    lvmframe._header["DRPSTAGE"] = (ReductionStage(lvmframe._header["DRPSTAGE"]) + "FLATFIELDED").value
    lvmframe.writeFitsData(out_frame)

    return rss, lvmframe


def combine_rsss(in_rsss, out_rss, method="mean"):
    """combines the given RSS list to a single RSS using a statistic

    Parameters
    ----------
    in_rsss : array_like
        list of RSS file paths
    out_rss : str
        output RSS file path
    method : str, optional
        statistic to use for combining the RSS objects, by default "mean"
    """
    rss_list = []
    for i in in_rsss:
        # load subimages from disc and append them to a list
        rss = loadRSS(i)
        rss_list.append(rss)
    # combined_header = combineHdr(rss_list)
    combined_rss = RSS()
    combined_rss.combineRSS(rss_list, method=method)
    # combined_rss.setHeader(header=combined_header._header)
    # write out FITS file
    combined_rss.writeFitsData(out_rss)


def apertureFluxRSS_drp(
    in_rss, center_x, center_y, arc_radius, hdr_prefix, flux_type="mean,3900,4600"
):
    """combines a selection of fibers within a set aperture into a single spectrum

    Parameters
    ----------
    in_rss : str
        intput RSS file path
    center_x : float
        X coordinate of the aperture center
    center_y : float
        Y coordinate of the aperture center
    arc_radius : float
        aperture radius in spaxels
    hdr_prefix : str
        header prefix
    flux_type : str, optional
        collapse resulting spectrum, by default "mean,3900,4600"
    """
    flux_type = flux_type.split(",")
    center_x = float(center_x)
    center_y = float(center_y)
    arc_radius = float(arc_radius)
    # load subimages from disc and append them to a list
    rss = loadRSS(in_rss)

    spec = rss.createAperSpec(center_x, center_y, arc_radius)
    if flux_type[0] == "mean" or flux_type[0] == "sum" or flux_type[0] == "median":
        start_wave = float(flux_type[1])
        end_wave = float(flux_type[2])
        flux_spec = spec.collapseSpec(
            method=flux_type[0], start=start_wave, end=end_wave
        )

        rss.setHdrValue(
            hdr_prefix + " APER FLUX",
            flux_spec[0],
            flux_type[0] + " flux from %.0f to %.0f" % (start_wave, end_wave),
        )
        rss.setHdrValue(
            hdr_prefix + " APER ERROR",
            flux_spec[1],
            flux_type[0] + " error from %.0f to %.0f" % (start_wave, end_wave),
        )
    else:
        passband = PassBand()
        passband.loadTxtFile(
            flux_type[0], wave_col=int(flux_type[1]), trans_col=int(flux_type[2])
        )
        flux_spec = passband.getFluxPass(spec)
        #  print flux_spec
        rss.setHdrValue(
            hdr_prefix + " APER FLUX",
            float("%.3f" % flux_spec[0]),
            flux_type[0].split("/")[-1].split(".")[0]
            + " band flux (%.1farcsec diameter)" % (2 * arc_radius),
        )
        if flux_spec[1] is not None:
            rss.setHdrValue(
                hdr_prefix + " APER ERR",
                float("%.3f" % flux_spec[1]),
                flux_type[0].split("/")[-1].split(".")[0]
                + " band error (%.1farcsec diameter)" % (2 * arc_radius),
            )
    rss.writeFitsData(in_rss)


def matchFluxRSS_drp(
    rsss,
    center_x,
    center_y,
    arc_radius,
    hdr_prefixes,
    start_wave="3800",
    end_wave="4600",
    polyorder="2",
    verbose="0",
):
    """matches the flux level of the given RSS list within an aperture

    Parameters
    ----------
    rsss : array_like
        input RSS file paths
    center_x : float
        X coordinate of fiber center
    center_y : float
        Y coordinate of fiber center
    arc_radius : float
        aperture radius in spaxels
    hdr_prefixes : str
        header prefix
    start_wave : str, optional
        initial wavelength value, by default "3800"
    end_wave : str, optional
        final wavelength value, by default "4600"
    polyorder : str, optional
        polynomial degree, by default "2"
    verbose : str, optional
        whether to show information and plots or not, by default "0"
    """
    verbose = int(verbose)
    list_rss = rsss.split(",")
    center_x = float(center_x)
    center_y = float(center_y)
    if start_wave != "":
        start_wave = float(start_wave)
    else:
        start_wave = None
    if end_wave != "":
        end_wave = float(end_wave)
    else:
        end_wave = None
    hdr_prefixes = hdr_prefixes.split(",")
    arc_radius = float(arc_radius)
    polyorder = int(polyorder)
    specs = []
    fluxes = []
    for i in range(len(list_rss)):
        # load subimages from disc and append them to a list
        rss = loadRSS(list_rss[i])
        specs.append(rss.createAperSpec(center_x, center_y, arc_radius))
        fluxes.append(numpy.median(specs[i]._data))

    order = numpy.argsort(fluxes)
    #   print fluxes, order
    for i in range(len(list_rss)):
        rss = loadRSS(list_rss[i])
        ratio = specs[order[-1]] / specs[i]
        coeff = ratio.smoothPoly(
            order=polyorder, start_wave=start_wave, end_wave=end_wave
        )
        rss = rss * ratio
        rss._data = rss._data.astype(numpy.float32)
        rss._error = rss._error.astype(numpy.float32)
        if start_wave is not None:
            rss.setHdrValue(
                hdr_prefixes[i] + " RELFLUX START",
                start_wave,
                "Start wave for poly fit",
            )
        if end_wave is not None:
            rss.setHdrValue(
                hdr_prefixes[i] + " RELFLUX END", end_wave, "End wave for poly fit"
            )
        for m in range(len(coeff)):
            rss.setHdrValue(
                hdr_prefixes[i] + " RELFLUX POLY%i" % (m),
                "%.3E" % (coeff[len(coeff) - 1 - m]),
                "Polynomial coefficient",
            )

        rss.writeFitsData(list_rss[i])
        if verbose == 1:
            plt.plot(specs[i]._wave, (specs[order[-1]] / specs[i])._data, "-k")
            plt.plot(specs[i]._wave, ratio._data, "-r")
            # plt.plot((specs[i])._data,'-k')
            # plt.plot((specs[i]*ratio)._data,'-r')
    if verbose == 1:
        plt.show()


def includePosTab_drp(in_rss, position_table, offset_x="0.0", offset_y="0.0"):
    """
    Adds an ASCII file position table as a FITS table extension to the RSS file.
    An offset may be applied to the fiber positions in x and y direction independently.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file for which the position table will be added
    position_table : string
        Input position table ASCII file name
    offset_x : string of float, optional with default: '0.0'
        Offset applied to the fiber positions in x direction before being added
        to the RSS.
    offset_y : string of float, optional with default: '0.0'
        Offset applied to the fiber positions in y direction before being added
        to the RSS.

    Examples
    --------
    user:> lvmdrp rss includePosTab RSS.fits POSTAB.txt
    user:> lvmdrp rss includePosTab RSS.fits POSTAB.txt  offset_x=-5.0 offset_y=3.0
    """
    offset_x = float(offset_x)
    offset_y = float(offset_y)
    rss = RSS.from_file(in_rss)
    rss.loadTxtPosTab(position_table)
    rss.offsetPosTab(offset_x, offset_y)
    rss.writeFitsData(in_rss)


def copyPosTab_drp(in_rss, out_rss):
    """
    Copies the position table FITS extension from one RSS to another RSS FITS file.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file from which the position table will be taken.
    out_rss : string
        Output RSS FITS file (must already exist) in which the position table
        will be added.

    Examples
    --------
    user:> lvmdrp rss copyPosTab RSS1.fits RSS2.fits
    """
    rss1 = RSS.from_file(in_rss)
    rss2 = RSS.from_file(out_rss)
    rss2._shape = rss1._shape
    rss2._size = rss1._size

    rss2._arc_position_x = rss1._arc_position_x
    rss2._arc_position_y = rss1._arc_position_y
    rss2._good_fibers = rss1._good_fibers
    rss2._fiber_type = rss1._fiber_type
    rss2.writeFitsData(out_rss)


def offsetPosTab_drp(in_rss, offset_x, offset_y):
    """
    Applies an offset to the fiber positions in x and y direction independently.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file in which the position table will be changed by fiber offsets
    offset_x : string of float, optional with default: '0.0'
        Offset applied to the fiber positions in x direction.
    offset_y : string of float, optional with default: '0.0'
        Offset applied to the fiber positions in y direction.

    Examples
    --------
    user:> lvmdrp rss offsetPosTab RSS.fits offset_x=-5.0 offset_y=3.0
    """
    offset_x = float(offset_x)
    offset_y = float(offset_y)
    rss = RSS.from_file(in_rss)
    rss.offsetPosTab(offset_x, offset_y)
    rss.writeFitsData(in_rss)


def rotatePosTab_drp(in_rss, angle="0.0"):
    """
    Applies an offset to the fiber positions in x and y direction independently.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file in which the position table will be rotated around
        the bundle zero-point 0,0 by an angle
    angle : string of float, optional with default: '0.0'
        Angle applied to rotate the fiber positions counter-clockwise

    Examples
    --------
    user:> lvmdrp rss  RSS.fits rotate=152.0
    """
    angle = float(angle)
    rss = loadRSS(in_rss)
    new_pos = rss.rotatePosTab(angle)
    rss.setPosTab(new_pos)
    rss.writeFitsData(in_rss)


def createCube_drp(
    in_rss,
    cube_out,
    position_x="",
    position_y="",
    ref_pos_wave="",
    int_ref="1",
    mode="inverseDistance",
    resolution="1.0",
    sigma="1.0",
    radius_limit="5.0",
    min_fibers="3",
    slope="2",
    bad_threshold="0.01",
    replace_error="1e10",
    flip_x="0",
    flip_y="0",
    full_field="0",
    store_cover="0",
    parallel="auto",
    verbose="0",
):
    """create cube from given RSS

    Parameters
    ----------
    in_rss : str
        input RSS file path
    cube_out : str
        output cube file path
    position_x : str, optional
        input RSS file path for X coordinates, by default ""
    position_y : str, optional
        input RSS file path for Y coordinates, by default ""
    ref_pos_wave : str, optional
        reference wavelength, by default ""
    int_ref : str, optional
        reference kind, by default "1"
    mode : str, optional
        metric to use, by default "inverseDistance"
    resolution : str, optional
        spatial resolution, by default "1.0"
    sigma : str, optional
        standard deviation for the Gaussian weights, by default "1.0"
    radius_limit : str, optional
        maximum radius value, by default "5.0"
    min_fibers : str, optional
        minimum number of fibers, by default "3"
    slope : str, optional
        sharpness of the Gaussian weights, by default "2"
    bad_threshold : str, optional
        fraction of bad spaxels, by default "0.01"
    replace_error : str, optional
        error replacement for bad spaxels, by default "1e10"
    flip_x : str, optional
        whether to flip positions along X axis, by default "0"
    flip_y : str, optional
        whether to flip positions along Y axis, by default "0"
    full_field : str, optional
        whether to use the whole field or not, by default "0"
    store_cover : str, optional
        whether to store the cover or not, by default "0"
    parallel : str, optional
        whether to run in parallel or not, by default "auto"
    verbose : str, optional
        whether to show information and plots or not, by default "0"
    """
    resolution = float(resolution)
    sigma = float(sigma)
    radius_limit = float(radius_limit)
    min_fibers = int(min_fibers)
    slope = float(slope)
    bad_threshold = float(bad_threshold)
    flip_x = int(flip_x)
    flip_y = int(flip_y)
    int_ref = int(int_ref)
    replace_error = float(replace_error)
    verbose = int(verbose)
    store_cover = bool(store_cover)
    if position_x == "":
        pos_x = None
    else:
        # pos_x = Spectrum1D()
        # pos_x.loadFitsData(position_x)
        pos_x = loadRSS(position_x)
    if position_y == "":
        pos_y = None
    else:
        # pos_y = Spectrum1D()
        # pos_y.loadFitsData(position_y)
        pos_y = loadRSS(position_y)
    if ref_pos_wave != "":
        ref_pos_wave = float(ref_pos_wave)
    else:
        ref_pos_wave = None
    rss = loadRSS(in_rss)
    if flip_x == 1:
        rss._arc_position_x = -1 * rss._arc_position_x
    if flip_y == 1:
        rss._arc_position_y = -1 * rss._arc_position_y

    if int(full_field) == 0:
        full_field = False
    elif int(full_field) == 1:
        full_field = True

    if parallel == "auto":
        cpus = cpu_count()
    else:
        cpus = int(parallel)

    if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
        idx = numpy.argmin(numpy.fabs(pos_x._wave - ref_pos_wave))
        ref_x = pos_x._data[0, idx]
        ref_y = pos_y._data[0, idx]
        if int_ref == 1:
            ref_x = numpy.rint(ref_x)
            ref_y = numpy.rint(ref_y)
        offset_x = (pos_x._data - ref_x) * -1
        if flip_x == 1:
            offset_x = offset_x * -1
        offset_y = (pos_y._data - ref_y) * -1
        if flip_y == 1:
            offset_y = offset_y * -1

        if verbose == 1:
            print(ref_x, ref_y)
            plt.plot(pos_x._wave, offset_x[0, :], "-k")
            plt.plot(pos_y._wave, offset_y[0, :], "-b")
            plt.show()

        if rss._shape == "C":
            if full_field:
                min_x = (
                    numpy.min(rss._arc_position_x + numpy.min(offset_x)) - rss._size[0]
                )
                max_x = (
                    numpy.max(rss._arc_position_x + numpy.max(offset_x)) + rss._size[0]
                )
                min_y = (
                    numpy.min(rss._arc_position_y + numpy.min(offset_y)) - rss._size[1]
                )
                max_y = (
                    numpy.max(rss._arc_position_y + numpy.max(offset_y)) + rss._size[1]
                )
            else:
                min_x = numpy.min(rss._arc_position_x) - rss._size[0]
                max_x = numpy.max(rss._arc_position_x) + rss._size[0]
                min_y = numpy.min(rss._arc_position_y) - rss._size[1]
                max_y = numpy.max(rss._arc_position_y) + rss._size[1]
                dim_x = numpy.rint(float(max_x - min_x) / resolution)
                dim_y = numpy.rint(float(max_y - min_y) / resolution)
            if int_ref == 2:
                ref_x = numpy.argsort(
                    numpy.fabs(
                        min_x + numpy.arange(dim_x) * resolution - offset_x[0, idx]
                    )
                )[0]
                ref_y = numpy.argsort(
                    numpy.fabs(
                        min_y + numpy.arange(dim_y) * resolution - offset_y[0, idx]
                    )
                )[0]
                off_x = (min_x + numpy.arange(dim_x)[ref_x] * resolution) - offset_x[
                    0, idx
                ]
                off_y = (min_y + numpy.arange(dim_y)[ref_y] * resolution) - offset_y[
                    0, idx
                ]
                offset_x = offset_x + off_x
                offset_y = offset_y + off_y
                ref_x += 1
                ref_y += 1
            dim_x = int(dim_x)
            dim_y = int(dim_y)
            min_x = float(min_x)
            min_y = float(min_y)
            max_x = float(max_x)
            max_y = float(max_y)

        elif rss._shape == "R":
            if not full_field:
                min_x = numpy.round(numpy.min(rss._arc_position_x), 4)
                max_x = numpy.round(numpy.max(rss._arc_position_x), 4)
                min_y = numpy.round(numpy.min(rss._arc_position_y), 4)
                max_y = numpy.round(numpy.max(rss._arc_position_y), 4)
                dim_x = (
                    numpy.round(numpy.rint(float(max_x - min_x) / resolution), 4) + 1
                )
                dim_y = (
                    numpy.round(numpy.rint(float(max_y - min_y) / resolution), 4) + 1
                )
                dim_x = int(dim_x)
                dim_y = int(dim_y)
                min_x = float(min_x)
                min_y = float(min_y)
            else:
                min_x = numpy.round(
                    numpy.min(
                        rss._arc_position_x[:, numpy.newaxis] + offset_x * resolution
                    ),
                    4,
                )
                max_x = numpy.round(
                    numpy.max(
                        rss._arc_position_x[:, numpy.newaxis] + offset_x * resolution
                    ),
                    4,
                )
                min_y = numpy.round(
                    numpy.min(
                        rss._arc_position_y[:, numpy.newaxis] + offset_y * resolution
                    ),
                    4,
                )
                max_y = numpy.round(
                    numpy.max(
                        rss._arc_position_y[:, numpy.newaxis] + offset_y * resolution
                    ),
                    4,
                )
                dim_x = (
                    numpy.round(numpy.rint(float(max_x - min_x) / resolution), 4) + 1
                )
                dim_y = (
                    numpy.round(numpy.rint(float(max_y - min_y) / resolution), 4) + 1
                )
                dim_x = int(dim_x)
                dim_y = int(dim_y)
                min_x = float(min_x)
                min_y = float(min_y)

        # needed to make sure the the c-code is compiled
        dummy_rss = RSS(
            data=numpy.zeros((rss._fibers, 2), dtype=numpy.float32),
            error=numpy.zeros((rss._fibers, 2), dtype=numpy.float32),
            mask=numpy.zeros((rss._fibers, 2), dtype=bool),
            wave=rss._wave[:2],
            shape=rss._shape,
            size=rss._size,
            arc_position_x=rss._arc_position_x,
            arc_position_y=rss._arc_position_y,
            good_fibers=rss._good_fibers,
            fiber_type=rss._fiber_type,
        )
        dummy_rss.createCubeInterDAR_new(
            offset_x,
            offset_y,
            min_x,
            max_x,
            min_y,
            max_y,
            dim_x,
            dim_y,
            mode=mode,
            sigma=sigma,
            resolution=resolution,
            radius_limit=radius_limit,
            min_fibers=min_fibers,
            slope=slope,
            bad_threshold=bad_threshold,
            full_field=full_field,
            replace_error=replace_error,
            store_cover=store_cover,
        )
    # set the dimension for the final array\

    if cpus > 1:
        pool = Pool(cpus)
        threads = []
        part_rss = rss.splitRSS(cpus)
        if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
            offset_x = RSS(data=offset_x, wave=rss._wave)
            offset_y = RSS(data=offset_y, wave=rss._wave)
            part_offsets_x = offset_x.splitRSS(cpus)
            part_offsets_y = offset_y.splitRSS(cpus)

        data = []
        error = []
        error_weight = []
        mask = []
        cover = []

        for i in range(cpus):
            if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
                threads.append(
                    pool.apply_async(
                        part_rss[i].createCubeInterDAR_new,
                        args=(
                            part_offsets_x[i]._data,
                            part_offsets_y[i]._data,
                            min_x,
                            max_x,
                            min_y,
                            max_y,
                            dim_x,
                            dim_y,
                            mode,
                            sigma,
                            radius_limit,
                            resolution,
                            min_fibers,
                            slope,
                            bad_threshold,
                            full_field,
                            replace_error,
                            store_cover,
                        ),
                    )
                )
            else:
                threads.append(
                    pool.apply_async(
                        part_rss[i].createCubeInterpolation,
                        args=(
                            mode,
                            sigma,
                            radius_limit,
                            resolution,
                            min_fibers,
                            slope,
                            bad_threshold,
                            replace_error,
                            store_cover,
                        ),
                    )
                )
        pool.close()
        pool.join()

        for i in range(cpus):
            cube = threads[i].get()
            if i == 0:
                header = cube._header
            data.append(cube._data)
            error.append(cube._error)
            error_weight.append(cube._error_weight)
            mask.append(cube._mask)
            cover.append(cube._cover)

        data = numpy.concatenate(data)
        if rss._error is not None:
            error = numpy.concatenate(error)
            error_weight = numpy.concatenate(error_weight)
        else:
            error = None
            error_weight = None
        mask = numpy.concatenate(mask)
        if store_cover and mode == "inverseDistance":
            cover = numpy.concatenate(cover)
        else:
            cover = None

        cube = Cube(
            data=data,
            error=error,
            mask=mask,
            wave=rss._wave,
            error_weight=error_weight,
            header=header,
            cover=cover,
        )
    else:
        if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
            # print(rss.getHdrValue('CRVAL1'), rss.getHdrValue('CDELT1'))
            cube = rss.createCubeInterDAR_new(
                offset_x,
                offset_y,
                min_x,
                max_x,
                min_y,
                max_y,
                dim_x,
                dim_y,
                mode=mode,
                sigma=sigma,
                resolution=resolution,
                radius_limit=radius_limit,
                min_fibers=min_fibers,
                slope=slope,
                bad_threshold=bad_threshold,
                replace_error=replace_error,
                store_cover=store_cover,
            )
        else:
            # print(pos_x,pos_y,ref_pos_wave,'test')
            cube = rss.createCubeInterpolation(
                mode=mode,
                sigma=sigma,
                resolution=resolution,
                radius_limit=radius_limit,
                min_fibers=min_fibers,
                slope=slope,
                bad_threshold=bad_threshold,
                replace_error=replace_error,
                store_cover=store_cover,
            )

    #   Cube.writeFitsData('dat_'+cube_out, extension_data=0)
    #  Cube.writeFitsData('err_'+cube_out, extension_error=0)
    #  Cube.writeFitsData('mask_'+cube_out, extension_mask=0)
    if pos_x is not None and pos_y is not None and ref_pos_wave is not None:
        cube.setHdrValue("CRPIX1", ref_x, "Ref pixel for WCS")
        cube.setHdrValue("CRPIX2", ref_y, "Ref pixel for WCS")
    cube.writeFitsData(cube_out)


def correctGalExtinct_drp(in_rss, out_rss, Av, Rv="3.1", verbose="0"):
    """
    Corrects the wavelength calibrated RSS for the effect of galactic extinction using
    the galactic extinction curve from Cardelli et al. (1989).

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file
    out_rss : string
        Output RSS FITS file with the corrected spectra
    Av : string of float
        V-band galactic extinction in magnitudes along the line of sight
    Rv : string of float, optional with default: '3.1'
        average E(B-V)/A(V) ratio
    verbose: string of integer (0 or 1), optional  with default: 1
        Show information during the processing on the command line (0 - no, 1 - yes)

    Examples
    --------
    user:> lvmdrp rss correctGalExtinct RSS_IN.fits RSS_OUT.fits 0.33
    """

    Av = float(Av)
    Rv = float(Rv)

    verbose = int(verbose)
    rss = loadRSS(in_rss)

    if len(rss._wave.shape) == 1:
        galExtCurve = galExtinct(rss._wave, Rv)
        Alambda = galExtCurve * Av
        if verbose == 1:
            plt.plot(1.0 / 10 ** (Alambda._data / -2.5))
            plt.show()
        rss_corr = rss * (1.0 / 10 ** (Alambda / -2.5))
    rss_corr.writeFitsData(out_rss)


def splitFile_drp(
    in_rss, data="", error="", mask="", wave="", fwhm="", position_table=""
):
    """
    Copies the different extension of the RSS into separate files.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file
    data : string, optional with default: ''
        Ouput FITS file name containing only the data RSS in its primary extension
    error : string, optional with default: ''
        Ouput FITS file name containing only the error RSS in its primary extension
    mask : string, optional with default: ''
        Ouput FITS file name containing only the bad pixel mask RSS in its
        primary extension
    wave : string, optional with default: ''
        Ouput FITS file name containing only the wavelength RSS in its primary extension
    fwhm : string, optional with default: ''
        Ouput FITS file name containing only the spectral resolution RSS in its
        primary extension
    position_table : string, optional with default: ''
        Ouput ASCII file of the position table in E3D format

    Examples
    ----------------
    user:> lvmdrp rss splitFile RSS_IN.fits DATA_RSS.fits
    user:> lvmdrp rss splitFile RSS_IN.fits mask=MASK_RSS.fits position_table=POSTAB.txt
    """
    rss = loadRSS(in_rss)

    if data != "" and rss._data is not None:
        rss.writeFitsData(data, extension_data=0, include_PT=False)

    if error != "" and rss._error is not None:
        rss.writeFitsData(error, extension_error=0, include_PT=False)

    if mask != "" and rss._mask is not None:
        rss.writeFitsData(mask, extension_mask=0, include_PT=False)

    if position_table != "" and rss._arc_position_x is not None:
        rss.writeTxtPosTab(position_table)


def maskFibers_drp(in_rss, out_rss, fibers, replace_error="1e10"):
    """masks fibers in the given RSS

    Parameters
    ----------
    in_rss : str
        input RSS file path
    out_rss : str
        output RSS file path
    fibers : array_like
        list of fibers to mask
    replace_error : str, optional
        replacement for masked error pixels, by default "1e10"
    """
    replace_error = float(replace_error)
    mask_fibers = fibers.split(",")

    rss = loadRSS(in_rss)
    for i in range(len(mask_fibers)):
        mfibers = mask_fibers[i].split("-")
        if len(mfibers) == 2:
            for f in range(int(mfibers[0]), int(mfibers[1]) + 1, 1):
                rss.maskFiber(f + 1, replace_error=replace_error)
        else:
            rss.maskFiber(int(mask_fibers[i]) - 1, replace_error=replace_error)
    rss.writeFitsData(out_rss)


def maskNAN_drp(in_rss, replace_error="1e12"):
    """mask NaN values in given RSS

    Parameters
    ----------
    in_rss : str
        input RSS file path
    replace_error : str, optional
        replacement for masked error pixels, by default "1e12"
    """
    rss = loadRSS(in_rss)
    select = numpy.isnan(rss._data)
    if numpy.sum(select) > 0:
        for i in range(rss._fibers):
            if numpy.sum(select[i, :]):
                rss._data[i, :] = 0
                if rss._error is not None:
                    rss._error[i, :] = float(replace_error)
                if rss._mask is not None:
                    rss._mask[i, :] = True
        rss.writeFitsData(in_rss)


def registerSDSS_drp(
    in_rss,
    out_rss,
    sdss_file,
    sdss_field,
    filter,
    ra,
    dec,
    hdr_prefix,
    search_box="20.0,2.6",
    step="1.0,0.2",
    offset_x="0.0",
    offset_y="0.0",
    quality_figure="",
    angle_key="SPA",
    parallel="auto",
    verbose="0",
):
    """
    Copies the different extension of the RSS into separate files.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file
    out_rss : string
        Output RSS FITS file
    sdss_file : string
        Original SDSS file in a given filter band that contains the object
        given in in_rss
    sdss_field : string
        Corresponding SDSS field calibration field for photometric calibration
    filter : string
        Filter response curve correponding to the SDSS file and covered by the
        data. The number of columns containing the wavelength and transmission
        are followed comma separated
    ra : string of float
        Right ascension of reference point to center the IFU in degrees
    dec: string of float
        Declination of reference point to center the IFU in degrees
    hdr_prefix : string
        Prefix for the FITS keywords in which the measurement parameters are
        stored. Need to start with 'HIERARCH'
    search_box : string list of floats with default '20.0,2.6'
        Search box size for subsequent iterations to construct the chi-square
        plane of the matching
    step : string list of floats with default '1.0,0.2'
        Sampling for subsequent iterations to construct the chi-square plane of
        the matching
    offset_x : string of float with default '0.0'
        Inital guess for the offset in x (right ascension ) direction
    offset_y : string of float with default '0.0'
        Inital guess for the offset in y (declination ) direction
    quality_figure : string with default  ''
        Name of the output quality control figure. If empty no figure will be produced
    parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
        Number of CPU cores used in parallel for the computation. If set to
        auto, the maximum number of CPUs for the given system is used.
    verbose: string of integer (0 or 1), optional  with default: 1
        Show information during the processing on the command line (0 - no, 1 - yes)

    Examples
    ----------------
    user:> lvmdrp rss registerSDSS RSS_IN.fits RSS_OUT.fits SDSS_r_IMG.fits /
    > SDSS_FIELD.fit sloan_r.dat,0,1 234.0 20.3 'HIERARCH TEST'

    user:> lvmdrp rss registerSDSS RSS_IN.fits RSS_OUT.fits SDSS_r_IMG.fits /
    > SDSS_FIELD.fit sloan_r.dat,0,1 234.0 20.3 'HIERARCH TEST'  search_box=20,2 /
    > step=2,0.5 quality_figure='test.png' parralel=3 verbose=1
    """

    search_box = numpy.array(search_box.split(",")).astype(numpy.float32)
    step = numpy.array(step.split(",")).astype(numpy.float32)
    offset_x = float(offset_x)
    offset_y = float(offset_y)
    verbose = int(verbose)

    rss = loadRSS(in_rss)

    filter = filter.split(",")
    posTab = rss.getPositionTable()
    # fiber_area = numpy.pi * posTab._size[0] ** 2
    img = loadImage(sdss_file)
    sdssimg = img.calibrateSDSS(sdss_field)
    spa = -1 * img.getHdrValue(angle_key)
    scale = 0.396
    # sdssimg._header.verify('fix')
    wcs = WCS(sdssimg._header)
    pix_coordinates = flatten(wcs.world_to_pix(ra, dec))
    passband = PassBand()
    passband.loadTxtFile(filter[0], wave_col=int(filter[1]), trans_col=int(filter[2]))

    best_offset_x = offset_x
    best_offset_y = offset_y
    for i in range(len(search_box)):
        if verbose == 1:
            print("Start iteration %d" % (i + 1))
            print(
                "Searchbox %.2f arcsec with sampling of %.2f arcsec"
                % (search_box[i], step[i])
            )
        if i > 0:
            offset_x = best_offset_x
            offset_y = best_offset_y
        (
            offsets_xIFU,
            offsets_yIFU,
            chisq,
            scale_flux,
            AB_flux,
            valid_fibers,
        ) = rss.registerImage(
            sdssimg,
            passband,
            search_box[i],
            step[i],
            pix_coordinates[0] + 1,
            pix_coordinates[1] + 1,
            scale,
            spa,
            offset_x,
            offset_y,
            parallel=parallel,
        )
        # idx = numpy.indices(chisq.shape)
        # select_valid = numpy.max(valid_fibers)-2<valid_fibers
        # select_best=numpy.min(chisq[select_valid])==chisq
        select_best = numpy.min(chisq) == chisq
        best_offset_x = offsets_xIFU[select_best][0]
        best_offset_y = offsets_yIFU[select_best][0]
        best_chisq = chisq[select_best][0]
        best_scale = scale_flux[select_best][0]
        best_valid = valid_fibers[select_best][0]

        if verbose == 1:
            print("Best offset in RA: %.2f" % (-1 * best_offset_x))
            print("Best offset in DEC: %.2f" % (-1 * best_offset_y))
            print("Minimum Chi-square: %.1f" % (best_chisq))
            print("Valid fibers: %.1f" % (best_valid))
            print("Photometric scale factor: %.3f" % (best_scale))

    rss = loadRSS(in_rss)
    if rss._size is not None:
        rss.offsetPosTab(-1 * best_offset_x, -1 * best_offset_y)
    rss = rss * best_scale
    rss._data = rss._data.astype(numpy.float32)
    # rss_error = rss._error.astype(numpy.float32)
    rss.setHdrValue(
        hdr_prefix + " PIPE OFFX",
        float("%.2f" % (best_offset_x)),
        "IFU RA offset from ref coordinate",
    )
    rss.setHdrValue(
        hdr_prefix + " PIPE OFFY",
        float("%.2f" % (best_offset_y)),
        "IFU DEC offset from ref coordinate",
    )
    rss.setHdrValue(
        hdr_prefix + " PIPE CHISQ",
        float("%.2f" % (best_chisq)),
        "CHISQ of image matching",
    )
    rss.setHdrValue(
        hdr_prefix + " PIPE VALIDFIB",
        int("%d" % (best_valid)),
        "Valid fibers for image matching",
    )
    rss.setHdrValue(
        hdr_prefix + " PIPE PHOTSCL",
        float("%.3f" % (best_scale)),
        "photometric scale factor",
    )
    rss.writeFitsData(out_rss)

    if quality_figure != "" or verbose == 1:
        flux = sdssimg.extractApertures(
            posTab,
            pix_coordinates[0],
            pix_coordinates[1],
            scale,
            angle=spa,
            offset_arc_x=best_offset_x,
            offset_arc_y=best_offset_y,
        )

        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_axes([0.01, 0.08, 0.3, 0.79])
        ax2 = fig.add_axes([0.32, 0.08, 0.3, 0.79])
        x_pos = rss._arc_position_x + best_offset_x
        y_pos = rss._arc_position_y + best_offset_y
        select_nan = numpy.isnan(AB_flux)
        norm = matplotlib.colors.LogNorm(
            vmin=numpy.min(AB_flux[numpy.logical_not(select_nan)]),
            vmax=numpy.max(AB_flux[numpy.logical_not(select_nan)]),
        )
        XY = numpy.hstack(
            ((x_pos).ravel()[:, numpy.newaxis], (y_pos).ravel()[:, numpy.newaxis])
        )
        circ = matplotlib.collections.CircleCollection(
            [60] * len(y_pos),
            offsets=XY,
            transOffset=ax1.transData,
            norm=norm,
            cmap=matplotlib.cm.gist_stern_r,
        )

        AB_flux[select_nan] = 1e-30
        circ.set_array(AB_flux.ravel())
        ax1.add_collection(circ)
        ax1.autoscale_view()
        ax1.set_xlim(-40, 40)
        ax1.set_ylim(-40, 40)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("CALIFA r band", fontsize=18, fontweight="bold")

        XY = numpy.hstack(
            ((x_pos).ravel()[:, numpy.newaxis], (y_pos).ravel()[:, numpy.newaxis])
        )
        circ2 = matplotlib.collections.CircleCollection(
            [60] * len(y_pos),
            offsets=XY,
            transOffset=ax2.transData,
            norm=norm,
            cmap=matplotlib.cm.gist_stern_r,
        )
        select_nan = numpy.isnan(flux[0])
        flux[0][select_nan] = 1e-30
        circ2.set_array((flux[0] / best_scale).ravel())
        ax2.add_collection(circ2)
        ax2.autoscale_view()
        ax2.set_xlim(-40, 40)
        ax2.set_ylim(-40, 40)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("SDSS best-match CALIFA map", fontsize=18, fontweight="bold")

        ax3 = fig.add_axes([0.66, 0.08, 0.35, 0.83])
        norm = matplotlib.colors.LogNorm(
            vmin=100.0 / float(best_valid), vmax=numpy.max(chisq)
        )
        chi_map = ax3.imshow(
            chisq.T,
            origin="lower",
            interpolation="nearest",
            norm=norm,
            extent=[
                offsets_xIFU[0, 0] - step[i] / 2.0,
                offsets_xIFU[-1, 0] + step[i] / 2.0,
                offsets_yIFU[0, 0] - step[i] / 2.0,
                offsets_yIFU[0, -1] + step[i] / 2.0,
            ],
        )
        ax3.plot(best_offset_x, best_offset_y, "ok", ms=8)
        plt.colorbar(chi_map, ax=ax3, pad=0.0)
        ax3.set_xlabel("offset in RA [arcsec]", fontsize=18)
        ax3.set_ylabel("offset in DEC [arcsec]", fontsize=18)
        ax3.minorticks_on()
        for line in (
            ax3.xaxis.get_ticklines()
            + ax3.yaxis.get_ticklines()
            + ax3.xaxis.get_minorticklines()
            + ax3.yaxis.get_minorticklines()
        ):
            line.set_markeredgewidth(2.0)
        ax3.set_title(
            "$\mathbf{\chi^2}$ matching for offsets", fontsize=18, fontweight="bold"
        )
        ax3.set_xlim(
            [offsets_xIFU[0, 0] - step[i] / 2.0, offsets_xIFU[-1, 0] + step[i] / 2.0]
        )
        ax3.set_ylim(
            [offsets_yIFU[0, 0] - step[i] / 2.0, offsets_yIFU[0, -1] + step[i] / 2.0]
        )
        if quality_figure != "":
            plt.savefig(quality_figure)
        if verbose == 1:
            plt.show()


def DAR_registerSDSS_drp(
    in_rss,
    sdss_file,
    sdss_field,
    ra,
    dec,
    out_prefix,
    ref_wave,
    coadd="150",
    step="150",
    smooth_poly="3",
    resolution="0.3,0.05",
    guess_x="0.0",
    guess_y="0.0",
    start_wave="",
    end_wave="",
    parallel="auto",
    verbose="0",
):
    resolution = numpy.array(resolution.split(",")).astype(numpy.float32)
    search_box = resolution * 5
    guess_x = float(guess_x)
    guess_y = float(guess_y)
    coadd = int(coadd)
    step = int(step)
    smooth_poly = int(smooth_poly)
    ref_wave = float(ref_wave)
    verbose = int(verbose)

    if start_wave == "":
        start_wave = None
    else:
        start_wave = float(start_wave)
    if end_wave == "":
        end_wave = None
    else:
        end_wave = float(end_wave)

    rss = loadRSS(in_rss)
    posTab = rss.getPositionTable()
    fiber_area = numpy.pi * posTab._size[0] ** 2
    img = loadImage(sdss_file)
    sdssimg = img.calibrateSDSS(sdss_field)
    spa = -1 * img.getHdrValue("spa")
    scale = 0.396
    wcs = WCS(sdssimg._header)
    pix_coordinates = wcs.world_to_pixel(ra, dec)
    steps = int(numpy.rint(rss._res_elements / step))
    mean_wave = numpy.zeros(steps)
    position_x = numpy.zeros(steps, dtype=numpy.float32)
    position_y = numpy.zeros(steps, dtype=numpy.float32)

    passbands = []
    for m in range(steps):
        filter = numpy.zeros(rss._res_elements)
        filter[step * m : step * (m + 1)] = 1.0
        select_wave = filter > 0.0
        mean_wave[m] = numpy.mean(rss._wave[select_wave])
        passbands.append(PassBand(wave=rss._wave, data=filter))

    diff = (mean_wave - ref_wave) ** 2
    select_start = diff == numpy.min(diff)
    idx_pass = numpy.arange(steps)
    select_pass = idx_pass[select_start][0]
    passband = passbands[select_pass]
    (flux_rss, error_rss) = passband.getFluxRSS(rss)
    flux_rss = flux_rss * fiber_area
    error_rss = error_rss * fiber_area

    # rss_mag = passband.fluxToMag(flux_rss)
    # AB_flux = 10 ** (rss_mag / -2.5)
    # AB_eflux = error_rss * (AB_flux / flux_rss)
    # good_rss = flux_rss / error_rss > 3.0
    for i in range(len(search_box)):
        result = rss.registerImage(
            sdssimg,
            passband,
            search_box[i],
            resolution[i],
            pix_coordinates[0],
            pix_coordinates[1],
            scale,
            spa,
            guess_x,
            guess_y,
            parallel=parallel,
        )
        guess_x = result[0]
        guess_y = result[1]
    position_x[select_start] = guess_x
    position_y[select_start] = guess_y

    select_blue = mean_wave < mean_wave[select_start]
    select_red = mean_wave > mean_wave[select_start]
    for m in range(idx_pass[select_blue][-1], idx_pass[select_blue][0] - 1, -1):
        result = rss.registerImage(
            sdssimg,
            passbands[m],
            search_box[-1],
            resolution[-1],
            pix_coordinates[0],
            pix_coordinates[1],
            scale,
            spa,
            position_x[m + 1],
            position_y[m + 1],
            parallel=parallel,
        )
        position_x[m] = result[0]
        position_y[m] = result[1]

    for m in range(idx_pass[select_red][0], idx_pass[select_red][-1] + 1, 1):
        result = rss.registerImage(
            sdssimg,
            passbands[m],
            search_box[-1],
            resolution[-1],
            pix_coordinates[0],
            pix_coordinates[1],
            scale,
            spa,
            position_x[m - 1],
            position_y[m - 1],
            parallel=parallel,
        )
        position_x[m] = result[0]
        position_y[m] = result[1]

    spec_y = Spectrum1D(data=position_y, wave=mean_wave)
    spec_x = Spectrum1D(data=position_x, wave=mean_wave)
    spec_x.smoothPoly(
        order=smooth_poly, ref_base=rss._wave, start_wave=start_wave, end_wave=end_wave
    )
    spec_y.smoothPoly(
        order=smooth_poly, ref_base=rss._wave, start_wave=start_wave, end_wave=end_wave
    )
    spec_x.writeFitsData(out_prefix + ".cent_x.fits")
    spec_y.writeFitsData(out_prefix + ".cent_y.fits")
    if verbose == 1:
        plt.plot(mean_wave, position_x, "ob")
        plt.plot(spec_x._wave, spec_x._data, "-b")
        plt.plot(mean_wave, position_y, "ok")
        plt.plot(spec_y._wave, spec_y._data, "-k")
        plt.show()


def stack_spectrographs(in_rsss: List[str], out_rss: str) -> RSS:
    """Stacks the given RSS list spectrograph-wise

    Given a list of RSS files, this function stacks them spectrograph-wise
    (i.e. the RSS objects are stacked along the fiber ID axis). The output
    RSS object will have the full set of fibers (i.e. 1944).

    Parameters
    ----------
    in_rsss : List[str]
        list of RSS file paths
    out_rss : str
        output RSS file path

    Returns
    -------
    RSS
        stacked RSS object
    """

    rsss = [loadRSS(in_rss) for in_rss in in_rsss]

    log.info(f"stacking frames in {','.join([os.path.basename(in_rss) for in_rss in in_rsss])} along fiber ID axis")
    try:
        rss_out = RSS.from_spectrographs(*rsss)
    except TypeError as e:
        log.error(f'Cannot stack spectrographs: {e}')
        return

    # write output
    log.info(f"writing stacked RSS to {os.path.basename(out_rss)}")
    rss_out._header["DRPSTAGE"] = (ReductionStage(rss_out._header["DRPSTAGE"]) + "SPECTROGRAPH_STACKED").value
    rss_out.writeFitsData(out_rss)

    return rss_out


def join_spec_channels(in_fframes: List[str], out_cframe: str, use_weights: bool = True):
    """Stitch together the three RSS channels (brz) into a single RSS.

    Given a list of three rss files (one per channel), this function
    stitches them together into a single RSS file. The output RSS file
    will have the same number of fibers as the input RSS files, but
    the wavelength range will be the union of the wavelength ranges
    of the input RSS files.

    Parameters
    ----------
    in_rsss : array_like
        list of RSS file paths for each spectrograph channel
    out_rss : str
    use_weights : bool, optional
        use inverse variance weights for channel combination, by default True

    Returns
    -------
    RSS
        combined RSS
    """

    # read all three channels
    log.info(f"loading RSS files: {', '.join([os.path.basename(in_rss) for in_rss in in_fframes])}")
    fframes = [lvmFFrame.from_file(in_rss) for in_rss in in_fframes]
    # set masked pixels to NaN
    [fframe.apply_pixelmask() for fframe in fframes]

    # combine channels
    new_rss = RSS.from_channels(*fframes, use_weights=use_weights)

    cframe = lvmCFrame(data=new_rss._data, error=new_rss._error, mask=new_rss._mask, header=new_rss._header,
                       wave=new_rss._wave, lsf=new_rss._lsf,
                       sky_east=new_rss._sky_east, sky_east_error=new_rss._sky_east_error,
                       sky_west=new_rss._sky_west, sky_west_error=new_rss._sky_west_error,
                       slitmap=new_rss._slitmap)

    cframe._header["DRPSTAGE"] = (ReductionStage(cframe._header["DRPSTAGE"]) + "CHANNEL_COMBINED").value

    # write output RSS
    if out_cframe is not None:
        log.info(f"writing output RSS to {os.path.basename(out_cframe)}")
        cframe.writeFitsData(out_cframe)

    return cframe

# TODO: from Law+2016
# 	* normalize each fiber to unity
# 	* merge all fiber spectra into a single (supersampled) spectrum
# 	* fit with a basis-spline function, this is the superflat
# 	* evaluate the fitted function in the individual fiber wavelengths
# 	* normalize each evaluated superflat by the individual fiberflat
# 	* fit each normalized fiber with a bspline
# 	* interpolate across bad pixels
def createMasterFiberFlat_drp(
    in_fiberflat,
    out_masterflat,
    weighted=True,
    degree=3,
    smooth=3,
    start_wave=None,
    end_wave=None,
):
    """create mater fiberflat from RSS fiberflat

    Parameters
    ----------
    in_fiberflat : str
        input RSS file path
    out_masterflat : str
        output RSS file path
    weighted : bool, optional
        whether to use errors as weights or not, by default True
    degree : int, optional
        degree of the B-spline, by default 3
    smooth : int, optional
        degree of the smoothing B-spline, by default 3
    start_wave : float, optional
        initial wavelength value, by default None
    end_wave : float, optional
        final wavelength value, by default None
    """
    fiberflat = RSS.from_file(in_fiberflat)

    if len(fiberflat._wave.shape) == 1:
        # cannot create master flat with homogeneous wavelength sampled RSS
        return None
    else:
        superflat = fiberflat.create1DSpec()
        print(superflat)

        # good pixels in superflat
        good_pix = superflat._data > 0

        # define weights
        if weighted:
            weights = 1 / superflat._error[good_pix]
        else:
            weights = None

        knots, coeffs, deg = interpolate.splrep(
            superflat._wave[good_pix],
            superflat._data[good_pix],
            w=weights,
            s=None,
            k=degree,
            xb=start_wave,
            xe=end_wave,
        )
        print(knots, coeffs, deg)
        superflat_func = interpolate.BSpline(knots, coeffs, deg, extrapolate=False)
        print(superflat_func)

        fiberflats = []
        for ifiber in range(fiberflat._fibers):
            print(ifiber)
            fiber = fiberflat.getSpec(ifiber)
            select = numpy.logical_not(fiber._mask)
            wave_ori = fiber._wave

            norm = superflat_func(fiber._wave[select]) * (1 / fiber[select])
            norm.smoothSpec(smooth, method="BSpline")
            norm = norm.resampleSpec(wave_ori)
            fiberflats.append(norm)

        master_fiberflat = RSS.from_spectra1d(fiberflats)

        master_fiberflat.writeFitsData(out_masterflat)


@skip_on_missing_input_path(["in_std", "in_sky", "in_biases", "in_fiberflat", "in_arc", "ref_values"])
def quickQuality(
    in_sci,
    in_std,
    in_sky,
    in_biases,
    in_fiberflat,
    in_arc,
    out_report,
    ref_values,
    pct_level=98,
):
    """builds a quality report from the quick DRP outputs

    Parameters
    ----------
    in_std : str
        input RSS file path for standard stars
    in_sky : str
        input RSS file path for sky
    in_biases : array_like
        list of input image file path for biases
    in_fiberflat : str
        input RSS file path for fiberflat
    in_arc : str
        input RSS file path for arc
    out_report : str
        output file path for final report
    ref_values : str
        input YAML file contaning the reference values
    pct_level : int, optional
        percentile used in report, by default 98
    """
    # TODO: load reference values for qualitative quality flags
    ref_values = yaml.safe_load(open(ref_values, "r"))

    # load passbands
    responses = []
    for passband in DONE_PASS:
        # read passband
        response = PassBand()
        # TODO: use a combination of broadbands (avoid emission line contamination) and narrowbands around specific lines
        response.loadTxtFile(
            os.path.join(CONFIG_PATH, f"{passband}_passband.txt"),
            wave_col=1,
            trans_col=2,
        )
        responses.append(response)
    npassband = len(responses)

    # bias quality
    # 	- exposure name
    # 	- percentile counts in each quadrant and channel
    # 	- temperature of specs room (need for a bias?)
    # 	- UT (of observation or reduction?)
    # 	- qualitative quality (GOOD, BAD)
    frame_name = []
    avg_count, std_count, pct_count = [], [], []
    temps, times, flags = [], [], []
    for in_bias in in_biases:
        bias_img = loadImage(in_bias)

        # extract frame name
        frame_name = os.path.basename(in_bias).split(".")[0]
        _, camera, expnum = frame_name.split("-")
        frame_name.append(frame_name)

        # compute statistics
        quads_avg, quads_std, quads_pct = [], [], []
        for section in bias_img._header["AMP? TRIMSEC"]:
            quad = bias_img.getSection(section)
            quads_avg.append(numpy.mean(quad._data))
            quads_std.append(numpy.std(quad._data))
            quads_pct.append(numpy.percentile(quad._data, q=pct_level))
        avg_count.append(quads_avg)
        std_count.append(quads_std)
        pct_count.append(quads_pct)

        # extract other quantities
        temps.append(bias_img["TRUSTEMP"])
        times.append(Time(bias_img["DATE-OBS"], scale="tai"))
        # TODO: set quality flags
        flags.append("GOOD")

    # build bias table
    bias_table = Table(
        data={
            "frame_name": frame_name,
            "avg_count": numpy.asarray(avg_count),
            "std_count": numpy.asarray(std_count),
            f"{pct_level}p_count": numpy.asarray(pct_count),
            "temp": numpy.asarray(temps),
            "time": numpy.asarray(times),
            "flag": numpy.asarray(flags),
        },
        names=[
            "frame_name",
            "avg_count",
            "std_count",
            f"{pct_level}p_count",
            "temp",
            "time",
            "flag",
        ],
        units=[None, u.electron, u.electron, u.electron, u.Celsius, "UT", None],
    )

    # fiberflat
    # 	- exposure name
    # 	- fiber recovery (good/total)
    # 	- exp. time
    # 	- temperature of specs room
    # 	- UT
    # 	- median flux in e- in all given passpands
    # 	- quality (GOOD, BAD)
    # arc
    # 	- exposure name
    # 	- exp. time
    # 	- temperature of specs room
    # 	- UT
    # 	- wavelength center for each channel
    # 	- quality (GOOD, BAD)
    # std
    # 	- exposure name
    # 	- fiber recovery (good/total)
    # 	- exp. time
    # 	- temperature of specs room
    # 	- UT
    # 	- sky level for each channel
    # 	- S/N**2 for each channel
    # 	- quality (GOOD, BAD)
    frame_name, fiber_flavor = [], []
    flx, err, snr, sn2, mag = [], [], [], [], []
    temps, times, rfibs, expos, flags = [], [], [], [], []
    for kind, in_fiber in zip(
        ["flat", "arc", "sci", "std", "sky"],
        [in_fiberflat, in_arc, in_sci, in_std, in_sky],
    ):
        rss = loadRSS(in_fiber)
        snr_rss = copy(rss)
        snr_rss._data = rss._data / rss._error

        # extract frame name
        frame_name = os.path.basename(in_fiber).split(".")[0]
        frame_name.extend([frame_name] * rss._fibers)
        # extract fiber flavor
        fiber_flavor.extend([kind] * rss._fibers)
        # extract passband fluxes, errors and SN2
        flx_pass = numpy.zeros((rss._fibers, npassband))
        err_pass = numpy.zeros((rss._fibers, npassband))
        mag_pass = numpy.zeros((rss._fibers, npassband))
        snr_pass = numpy.zeros((rss._fibers, npassband))
        sn2_pass = numpy.zeros((rss._fibers, npassband))
        for i in enumerate(responses):
            flx_pass[:, i], err_pass[:, i], _, _, _ = responses[i].getFluxRSS(rss)
            snr_pass[:, i], _, _, _, _ = responses[i].getFluxRSS(snr_rss)
            mag_pass[:, i] = responses[i].fluxToMag(flx_pass)
            snr_pass[:, i] = snr_pass
            sn2_pass[:, i] = snr_pass**2
        flx.extend(flx_pass)
        err.extend(err_pass)
        snr.extend(snr_pass)
        sn2.extend(sn2_pass)
        mag.extend(mag_pass)

        # extract other quantities
        temps.extend([rss._header["TRUSTEMP"]] * rss._fibers)
        times.extend([Time(rss._header["OBS-TIME"], scale="tai")] * rss._fibers)
        rfibs.extend(rss._good_fibers / rss._fibers)
        expos.extend([rss._header["EXPTIME"]] * rss._fibers)
        # TODO: set quality flags
        flags.extend(["GOOD"] * rss._fibers)

        # calculations specific to each image type
        if kind == "flat":
            # TODO: calculate fiber transparency
            # 	- pull flat information from past (fiducial) fiberflats
            # 	- calculate fiber ratio between two consecutive (in time) flats
            # 	- calculate the median/mean statistic
            pass
        elif kind == "arc":
            # TODO:
            # 	- fraction of lines detected vs expected
            # 	- lines used to calculate wavelength solution
            # 	- distribution of polynomial coefficients for wavelength
            # 	- distribution of polynomial coefficients for LSF
            pass
        elif kind == "std":
            # TODO:
            # 	- calculate flux for each fiber
            # 	- calculate magnitud for each fiber
            # 	- calculate SN2 for each spec/channel
            pass
        elif kind == "sky":
            # TODO:
            # 	- calculate sky flux in passbands
            pass
        else:
            pass

    # build fibers table
    fiber_table = Table(
        data={
            "frame_name": frame_name,
            "fiber_flavor": fiber_flavor,
            "flx": flx,
            "err": err,
            "snr": snr,
            "sn2": sn2,
            "temp": temps,
            "time": times,
            "fibers_frac": rfibs,
            "exp_time": expos,
            "flag": flags,
        },
        unit=[
            None,
            None,
            u.electron,
            u.electron,
            None,
            None,
            u.Celcius,
            "UT",
            None,
            u.second,
            None,
        ],
    )

    # observation depth
    # depth = numpy.zeros(len(DONE_PASS))
    # for i, passband in enumerate(DONE_PASS):
    #     std_table = fiber_table[fiber_table["fiber_flavor"] == "std"]
    #     model = partial(_linear_model, xdata=std_table["mag"])
    #     res = least_squares(lambda pars: model(pars) - std_table["snr"], x0=(1, 0))

    #     depth[i] = _linear_model(res.x, DONE_MAGS[i]) ** 2

    # dump all tables in a multi-extension FITS
    metadata = fits.PrimaryHDU()
    # metadata["DONE"] = ((depth >= DONE_LIMS).all(), "is this tile ID done?")
    report_fits = fits.HDUList(
        [metadata] + [fits.table_to_hdu(bias_table), fits.table_to_hdu(fiber_table)]
    )
    report_fits[0].name = "BIAS"
    report_fits[1].name = "FLAT/ARC/STD/SKY"
    report_fits.writeto(out_report, overwrite=True)
