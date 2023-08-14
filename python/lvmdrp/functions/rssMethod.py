#!/usr/bin/env python
# encoding: utf-8

import os
from copy import deepcopy as copy
from multiprocessing import Pool, cpu_count
from typing import List

import matplotlib
import matplotlib.gridspec as gridspec
import numpy
import yaml
import bottleneck as bn
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from numpy import polynomial
from scipy import interpolate, ndimage
from scipy.optimize import least_squares

from lvmdrp.utils.decorators import skip_on_missing_input_path, drop_missing_input_paths, skip_if_drpqual_flags
from lvmdrp.core.constants import CONFIG_PATH, ARC_LAMPS
from lvmdrp.core.cube import Cube
from lvmdrp.core.fiberrows import FiberRows
from lvmdrp.core.image import loadImage
from lvmdrp.core.passband import PassBand
from lvmdrp.core.plot import plt, create_subplots, save_fig, plot_wavesol_residuals, plot_wavesol_coeffs
from lvmdrp.core.rss import RSS, _read_pixwav_map, glueRSS, loadRSS
from lvmdrp.core.spectrum1d import Spectrum1D, wave_little_interpol, _spec_from_lines, _cross_match
from lvmdrp.external import ancillary_func
from lvmdrp.utils import flatten
from lvmdrp import log


description = "Provides Methods to process Row Stacked Spectra (RSS) files"

__all__ = [
    "determine_wavelength_solution",
    "create_pixel_table",
    "combineRSS_drp",
    "checkPixTable_drp",
    "correctPixTable_drp",
    "resample_wavelength",
    "includePosTab_drp",
    "join_spec_channels"
]


DONE_PASS = "gi"
DONE_MAGS = numpy.asarray([22, 21])
DONE_LIMS = numpy.asarray([1.7, 5.0])


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
def determine_wavelength_solution(in_arcs: List[str], out_wave: str, out_lsf: str,
                                  ref_fiber: int = 319, pixel: List[float] = [], ref_lines: List[float] = [],
                                  poly_disp: int = 3, poly_fwhm: int = 5,
                                  poly_cros: int = 3, poly_kinds: list = ['poly', 'poly', 'poly'],
                                  init_back: float = 10.0, aperture: int = 10,
                                  flux_min: float = 10.0, fwhm_max: float = 5.0,
                                  rel_flux_limits: list = [0.001, 100.0], fiberflat: str = "",
                                  negative: bool = False, cc_correction: bool = True,
                                  cc_max_shift: int = 40,
                                  display_plots: bool = False):
    """
    Solves for the wavelength and the LSF using polynomial fitting

    Measures the pixel position of emission lines in wavelength UNCALIBRATED
    for all fibers of the RSS. Starting from the initial guess of pixel
    positions for a given fiber, the program measures the position using
    Gaussian fitting to the first and last fiber of the RSS. The best fit
    emission line position of the previous fiber are used as guess parameters.
    Certain criterion can be imposed to reject certain measurements and flag
    those as bad. They will be ignored for the dispersion solution, which is
    estimated for each fiber independently. Two RSS FITS file containing the
    wavelength pixel table and the FWHM pixel table will be stored.

    Parameters
    --------------
    arc_rss : string
        Input RSS FITS file name of the uncalibrated arc lamp exposure
    prefix_out : string
        PREFIX for the output RSS file containing the wavelength RSS pixel table (PREFIX.disp.fits) and
        the spectral resolution (FWHM) RSS pixel table (PREFIX.res.fits)
    ref_line_file : string, optional with default: ''
        ASCII file name containing the number of the reference fiber in the first row,
        reference wavelength of emission line, its rough centroid pixel position a flag if the width of the
        line should be considered for the spectral resolution measurements (space separated) in
        each subsquent row.
        If no ASCII file is provided those information must be given in the ref_fiber, pixel and ref_lines parameters.
    ref_fiber : string of integer, optional with default: ''
        Number of the fiber in the RSS for which the rough guess for their centroid pixel position (x-direction) are given.
        Only used if no ASCII file is given.
    pixel : string of integers, optional with default: ''
        Comma-separated list of rough centroid pixel position for each emission line for the corresponding reference fiber.
        Only used if no ASCII file is given.
    ref_lines : string of floats, optional with default: ''
        Comma-separated list of reference emission-line wavelength. Need to be same number of values as for the pixel guess
        Only used if no ASCII file is given.
    poly_dispersion : string of integer, optional with default: '-5'
        Degree of polynomial used to construct the wavelength solution for each fiber. (positiv: normal polynomial, negative: Legandre polynomial)
    poly_fwhm : string of two integers, optional with default: '-3,-5'
        First integer is the degree of polynomial used to smooth the measured FWHM of each line as a function of fiber number (cross-dispersion).
        Second integer is the degree of polynomial used to subsquently extrapolate the line FWHM across the disperion direction,
        (positiv: normal polynomial, negative: Legandre polynomial)
    init_back : string of float, optinal with default: '10.0'
        Initial guess for the constant background level that can be fitted in addition to the Gaussian for each line.
        If this parameter is left empty, the background level is fixed to zero.
    aperture : string of integer, optional with default: '13'
        Aperture centered on the guess of the pixel position from which pixel with the maximum flux is used as the guess for the Gaussian fitting.
        This is also the size of the fitted region for each line.
    flux_min : string of float, optional with default: '200.0'
        Required minimum integrated flux of the best-fit Gaussian model to be considered as a reliable value.
        The measurement for this emission line for the specific fiber is masked if it falls below this threshold.
    fwhm_max : string of float, optional with default: '10.0'
        Maximum FWHM of the best-fit Gaussian model to be considered as a reliable value.
    rel_flux_limits : string of two floats, optional with default: '0.1,5.0'
        Required relative integrated fluxes with respect to the measured fluxes  for the reference fiber.
        If relative fluxes are outside this range, they will be masked.
    negative : boolean, optiona with default False
        whether to flip dark along the flux axis or not
    plot: string of integer (0 or 1), optional  with default: 1
        Show information during the processing on the command line (0 - no, 1 - yes)

    Examples
    --------
    user:> lvmdrp rss detWaveSolution ARC_RSS.fits arc REF_FILE.txt /
    > poly_dispersion='-7' poly_fwhm='-4,-5'

    user:> lvmdrp rss detWaveSolution ARC_RSS.fits arc ref_fiber=100 /
    > pixel=200,500,1000 ref_lines=3000.0,5000.0,8000.0 flux_min=100.0
    """

    # convert parameters to the correct type
    kind_disp, kind_fwhm, kind_cros = poly_kinds.split(",") if isinstance(poly_kinds, str) else poly_kinds

    if isinstance(in_arcs, (list, tuple)):
        pass
    else:
        in_arcs = [in_arcs]

    if fiberflat != "":
        fiberflat = fiberflat.split(",")

    iarcs = []
    ilamps = []
    for in_arc in in_arcs:
        # initialize the extracted arc line frame
        log.info(f"reading arc from '{in_arc}'")
        arc = RSS()
        arc.loadFitsData(in_arc)

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
    # update lamps status
    lamps = set(ilamps)

    # replace NaNs
    mask = numpy.isnan(arc._data) | numpy.isnan(arc._error)
    mask |= (arc._data < 0) | (arc._error < 0)
    arc._data[mask] = 0
    arc._error[mask] = 0

    # read reference lines
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

    # sort lines by pixel position
    sort = numpy.argsort(pixel)
    pixel = pixel[sort]
    ref_lines = ref_lines[sort]
    use_line = use_line[sort]
    nlines = len(pixel)

    # apply cc correction to lines if needed
    if cc_correction or ref_fiber != ref_fiber_:
        log.info(f"running cross matching on {pixel.size} good lines")
        # determine maximum correlation shift
        pix_spec = _spec_from_lines(pixel, sigma=2, wavelength=arc._pixels)

        # fix cc_max_shift
        cc_max_shift = max(cc_max_shift, 50)
        # cross-match spectrum and pixwav map
        cc, bhat, mhat = _cross_match(
            ref_spec=pix_spec,
            obs_spec=arc._data[ref_fiber],
            stretch_factors=numpy.linspace(0.8,1.2,10000),
            shift_range=[-cc_max_shift, cc_max_shift]
        )
        
        log.info(f"max CC = {cc:.2f} for strech = {mhat:.2f} and shift = {bhat:.2f}")
    else:
        mhat, bhat = 1.0, 0.0

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
    fwhm_sol = numpy.zeros((arc._fibers, arc._data.shape[1]), dtype=numpy.float32)
    fwhm_rms = numpy.zeros(arc._fibers, dtype=numpy.float32)

    # measure the ARC lines with individual Gaussian across the CCD
    log.info(
        f"measuring arc lines for each fiber from reference fiber {ref_fiber}, "
        f"{flux_min = }, {fwhm_max = } and relative flux limits {rel_flux_limits}"
        )

    # TODO: run peak finder without gaussian fitting in a small running window
    # centers = cut_iter.measurePeaks(
    #     pix, method, init_sigma, threshold=threshold, max_diff=float(max_diff)
    # )

    # initialize plots for arc lines fitting
    ncols = 3
    nrows = int(numpy.ceil(nlines / ncols))
    fig, axs = create_subplots(to_display=display_plots, nrows=nrows, ncols=ncols, figsize=(6*ncols, 6*nrows))
    fig.suptitle("Gaussian fitting")
    fig.supylabel("counts (e-/pixel)")
    for i, ax in zip(range(nlines), axs):
        # ax.axvline(pixel[i], ls="--", lw=1, color="tab:red")
        ax.set_title(f"line @ {pixel[i]:.1f} (pixel) - {ref_lines[i]:.2f} (angstrom)")
        ax.set_xlabel("X (pixel)")
    # axs = None
    fibers, flux, cent_wave, fwhm, masked = arc.measureArcLines(
        ref_fiber,
        pixel,
        aperture=aperture,
        init_back=init_back,
        flux_min=flux_min,
        fwhm_max=fwhm_max,
        rel_flux_limits=rel_flux_limits,
        axs=axs,
    )
    save_fig(
        fig,
        product_path=out_wave,
        to_display=display_plots,
        figure_path="qa",
        label="lines_fitting",
    )

    if fiberflat != "":
        log.info("computing fiberflat from measured lines")
        norm_flux = numpy.zeros_like(ref_lines)
        for n in range(len(ref_lines)):
            norm_flux[n] = numpy.nanmean(flux[numpy.logical_not(masked[:, n]), n])
        flat_flux = numpy.nanmean(flux / norm_flux[numpy.newaxis, :], 1)
        log.info(
            f"assuming wavelength range [{fiberflat[0]}, {fiberflat[1]}] and sampling {fiberflat[2]} AA"
        )
        wave = numpy.arange(
            float(fiberflat[0]),
            float(fiberflat[1]) + float(fiberflat[2]),
            float(fiberflat[2]),
        )
        norm = numpy.ones((flux.shape[0], len(wave)), dtype=numpy.float32)
        norm = norm * flat_flux[:, numpy.newaxis]
        rss_flat = RSS(wave=wave, data=norm, header=arc.getHeader())
        log.info(f"storing fiberflat in '{fiberflat[3]}'")
        rss_flat.writeFitsData(f"{fiberflat[3]}.fits")

    # smooth the FWHM values for each ARC line in cross-dispersion direction
    if poly_cros != 0:
        log.info(
            f"smoothing FWHM of guess lines along cross-dispersion axis using {poly_cros}-deg polynomials")
        for i in range(nlines):
            select = numpy.logical_and(
                numpy.logical_not(masked[:, i]), flux[:, i] > flux_min
            )
            fwhm_med = ndimage.filters.median_filter(numpy.fabs(fwhm[select, i]), 4)
            msg = f'Failed to fit {kind_cros} for arc line {i}'
            if kind_cros not in ["poly", "legendre", "chebyshev"]:
                log.warning(f"invalid polynomial kind '{kind_cros}'. Falling back to 'poly'")
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
    log.info(
        f"fitting wavelength solutions using {poly_disp}-deg polynomials"
    )

    # Iterate over the fibers
    good_fibers = numpy.zeros(len(fibers), dtype="bool")
    nmasked = numpy.zeros(len(fibers), dtype="uint16")
    for i in fibers:
        masked_lines = masked[i, use_line]
        nmasked[i] = numpy.sum(masked_lines)

        if nmasked[i] == 0:
            good_fibers[i] = True
        elif nmasked[i] == len(masked_lines):
            log.warning(f"fiber {i} has all lines masked")
            good_fibers[i] = False
        # select = numpy.logical_not(masked_lines)

        if kind_disp not in ["poly", "legendre", "chebyshev"]:
            log.warning(
                ("invalid polynomial kind " f"'{kind_disp = }'. Falling back to 'poly'")
            )
        if kind_disp == "poly":
            wave_cls = polynomial.Polynomial
        elif kind_disp == "legendre":
            wave_cls = polynomial.Legendre
        elif kind_disp == "chebyshev":
            wave_cls = polynomial.Chebyshev
        
        wave_poly = wave_cls.fit(cent_wave[i, use_line], ref_lines[use_line], deg=poly_disp)

        wave_coeffs[i, :] = wave_poly.convert().coef
        wave_sol[i, :] = wave_poly(arc._pixels)
        wave_rms[i] = numpy.std(wave_poly(cent_wave[i, use_line]) - ref_lines[use_line])

    log.info(
        "finished wavelength fitting with median "
        f"RMS = {numpy.median(wave_rms):g} AA "
        f"({numpy.median(wave_rms[:,None]/numpy.diff(wave_sol, axis=1)):g} pix)"
    )

    # Estimate the spectral resolution pattern
    dwave = wave_sol[:, 1:] - wave_sol[:, :-1]
    cent_round = numpy.round(cent_wave).astype(int)

    # Iterate over the fibers
    log.info(f"fitting LSF solutions using {poly_fwhm}-deg polynomials")
    for i in fibers:
        fwhm_wave = numpy.fabs(dwave[i, cent_round[i, :]]) * fwhm[i, :]

        if kind_fwhm not in ["poly", "legendre", "chebyshev"]:
            log.warning(
                f"invalid polynomial kind '{kind_fwhm = }'. Falling back to 'poly'"
            )
            kind_fwhm = "poly"
        if kind_fwhm == "poly":
            fwhm_cls = polynomial.Polynomial
        elif kind_fwhm == "legendre":
            fwhm_cls = polynomial.Legendre
        elif kind_fwhm == "chebyshev":
            fwhm_cls = polynomial.Chebyshev
        
        fwhm_poly = fwhm_cls.fit(cent_wave[i, use_line], fwhm_wave[use_line], deg=poly_fwhm)

        lsf_coeffs[i, :] = fwhm_poly.convert().coef
        fwhm_sol[i, :] = fwhm_poly(arc._pixels)
        fwhm_rms[i] = numpy.std(fwhm_wave[use_line] - fwhm_poly(cent_wave[i, use_line]))

    log.info(
        "finished LSF fitting with median "
        f"RMS = {numpy.median(fwhm_rms):g} AA "
        f"({numpy.median(fwhm_rms[:,None]/numpy.diff(wave_sol, axis=1)):g} pix)"
    )

    # create plot of polynomial coefficients
    fig, axs = create_subplots(to_display=display_plots, nrows=wave_coeffs.shape[1], figsize=(10, 15), sharex=True)
    # TODO: use ypix for the fibers instead of fiber ids
    axs = plot_wavesol_coeffs(numpy.arange(arc._fibers), coeffs=wave_coeffs, axs=axs, labels=True)
    save_fig(
        fig,
        product_path=out_wave,
        to_display=display_plots,
        figure_path="qa",
        label="coeffs_wave",
    )
    # create plot of wavelength fitting residuals
    fig, ax = create_subplots(to_display=display_plots, figsize=(15, 7))
    axs = plot_wavesol_residuals(lines_pixels=pixel, lines_waves=ref_lines, model_waves=wave_cls(wave_coeffs[ref_fiber])(pixel), ax=ax, labels=True)
    save_fig(
        fig,
        product_path=out_wave,
        to_display=display_plots,
        figure_path="qa",
        label="residuals_wave",
    )

    # create plot of polynomial fittings
    fig = plt.figure(figsize=(16, 10), tight_layout=True)
    gs = gridspec.GridSpec(10, max(poly_disp + 1, poly_fwhm + 1))

    ax_spec = fig.add_subplot(gs[:3, :])
    ax_spec.tick_params(labelbottom=False)
    # ax_spec.set_yscale("log")
    ax_sol_wave = fig.add_subplot(gs[3:6, :], sharex=ax_spec)
    ax_sol_fwhm = ax_sol_wave.twinx()
    ax_sol_wave.tick_params("y", labelcolor="tab:blue")
    ax_sol_fwhm.tick_params("y", labelcolor="tab:red")
    ax_coe_wave, ax_coe_fwhm = [], []
    for i in range(poly_disp + 1):
        ax_coe_wave.append(fig.add_subplot(gs[6:8, i]))
    for i in range(poly_fwhm + 1):
        ax_coe_fwhm.append(fig.add_subplot(gs[8:, i]))

    # add reference spectrum plot with reference lines & corrected lines
    good_pix = ~arc._mask
    for pix in pixel:
        ax_spec.axvspan(
            pix - (aperture - 1) // 2,
            pix + (aperture - 1) // 2,
            numpy.nanmin((arc._data * good_pix)[ref_fiber]),
            numpy.nanmax((arc._data * good_pix)[ref_fiber]),
            fc="0.7",
            alpha=0.5,
        )
    ax_spec.vlines(
        (pixel - bhat) / mhat,
        numpy.nanmin((arc._data * good_pix)[ref_fiber]),
        numpy.nanmax((arc._data * good_pix)[ref_fiber]),
        color="tab:red",
        lw=0.5,
        label="orig. ref. lines",
    )
    ax_spec.vlines(
        pixel,
        numpy.nanmin((arc._data * good_pix)[ref_fiber]),
        numpy.nanmax((arc._data * good_pix)[ref_fiber]),
        color="tab:blue",
        lw=0.5,
        label=f"corr. lines ({mhat = :.2f}, {bhat = :.2f})",
    )
    ax_spec.step(arc._pixels, (arc._data * good_pix)[ref_fiber], color="0.2", lw=1)
    ax_spec.set_title(f"reference arc spectrum {ref_fiber}", loc="left")
    ax_spec.set_ylabel("count (e-/pix)")
    ax_spec.legend(loc=1)

    # add coefficients plots
    for icoef in range(poly_disp + 1):
        data = wave_coeffs[:, icoef]
        mean, std = data.mean(), data.std()
        ax_coe_wave[icoef].hist(data, bins=100, fc="tab:blue")
        ax_coe_wave[icoef].text(
            0.05,
            0.95,
            f"{mean = :g}\n{std = :g}",
            va="top",
            ha="left",
            transform=ax_coe_wave[icoef].transAxes,
        )
        ax_coe_wave[icoef].tick_params(labelsize="x-small")
        if icoef == 0:
            ax_coe_wave[icoef].set_title("wavelength coefficients", loc="left")
    for icoef in range(poly_fwhm + 1):
        data = lsf_coeffs[:, icoef]
        mean, std = data.mean(), data.std()
        ax_coe_fwhm[icoef].hist(data, bins=100, fc="tab:red")
        ax_coe_fwhm[icoef].text(
            0.05,
            0.95,
            f"{mean = :g}\n{std = :g}",
            va="top",
            ha="left",
            transform=ax_coe_fwhm[icoef].transAxes,
        )
        ax_coe_fwhm[icoef].tick_params(labelsize="x-small")
        if icoef == 0:
            ax_coe_fwhm[icoef].set_title("LSF coefficients", loc="left")

    # add wavelength and LSF solutions plot
    ax_sol_wave.fill_between(
        arc._pixels,
        wave_sol.mean(0) - wave_sol.std(0),
        wave_sol.mean(0) + wave_sol.std(0),
        lw=0,
        fc="tab:blue",
        alpha=0.5,
    )
    ax_sol_wave.plot(arc._pixels, wave_sol.mean(0), lw=1, color="tab:blue")
    for i in fibers:
        ax_sol_wave.plot(
            cent_wave[i, use_line],
            ref_lines[use_line],
            ",",
            color="tab:blue",
        )
    ax_sol_wave.set_xlabel("dispersion axis (pix)")
    ax_sol_wave.set_ylabel("wavelength (AA)")
    ax_sol_wave.set_title(
        f"wavelength solutions with a {poly_disp}-deg polynomial",
        loc="left",
        color="tab:blue",
    )

    ax_sol_fwhm.fill_between(
        arc._pixels,
        fwhm_sol.mean(0) - fwhm_sol.std(0),
        fwhm_sol.mean(0) + fwhm_sol.std(0),
        lw=0,
        fc="tab:red",
        alpha=0.5,
    )
    ax_sol_fwhm.plot(arc._pixels, fwhm_sol.mean(0), lw=1, color="tab:red")
    for i in fibers:
        fwhm_wave = numpy.fabs(dwave[i, cent_round[i, :]]) * fwhm[i, :]

        ax_sol_fwhm.plot(
            cent_wave[i, use_line],
            fwhm_wave[use_line],
            ",",
            color="tab:red",
        )
    ax_sol_fwhm.set_ylabel("FWHM LSF (AA)")
    ax_sol_fwhm.set_title(
        f"LSF solutions with a {poly_fwhm}-deg polynomial",
        loc="right",
        color="tab:red",
    )

    fig.tight_layout()
    save_fig(fig, product_path=out_wave, to_display=display_plots, figure_path='qa', label="fit_wave")


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
        "%.4f" % (numpy.median(fwhm_rms[good_fibers])),
        "Median RMS of disp sol",
    )
    arc.setHdrValue(
        "HIERARCH PIPE DISP RMS MIN",
        "%.4f" % (numpy.min(fwhm_rms[good_fibers])),
        "Min RMS of disp sol",
    )
    arc.setHdrValue(
        "HIERARCH PIPE DISP RMS MAX",
        "%.4f" % (numpy.max(fwhm_rms[good_fibers])),
        "Max RMS of disp sol",
    )
    mask = numpy.zeros(arc._data.shape, dtype=bool)
    mask[~good_fibers] = True
    wave_trace = FiberRows(data=wave_sol, mask=mask, coeffs=wave_coeffs, header=arc._header.copy())
    wave_trace._header["IMAGETYP"] = "wave"
    fwhm_trace = FiberRows(data=fwhm_sol, mask=mask, coeffs=lsf_coeffs, header=arc._header.copy())
    fwhm_trace._header["IMAGETYP"] = "lsf"

    wave_trace.writeFitsData(out_wave)
    fwhm_trace.writeFitsData(out_lsf)


# TODO:
# * merge arc_wave and arc_fwhm into lvmArc product, change variable name to in_arc
@skip_on_missing_input_path(["in_rss", "arc_wave", "arc_fwhm"])
@skip_if_drpqual_flags(["EXTRACTBAD", "BADTRACE"], "in_rss")
def create_pixel_table(in_rss: str, out_rss: str, arc_wave: str, arc_fwhm: str = "",
                       cropping: list = None):
    """
    Applies the wavelength and possibly also the spectral resolution (FWHM) to an RSS

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file
    out_rss : string
        Output RSS FITS file with the wavelength and spectral resolution pixel
        table added as extensions
    arc_wave : string
        RSS FITS file containing the wavelength pixel table in its primary
        (0th) extension
    arc_fwhm : string, optional with default: ''
        RSS FITS file containing the spectral resolution (FWHM) pixel table in
        its primary (0th) extension. No spectral resolution will not be added
        if the string is empty.

    Examples
    --------
    user:> lvmdrp rss createPixTable RSS_IN.fits RSS_OUT.fits WAVE.fits
    user:> lvmdrp rss createPixTable RSS_IN.fits RSS_OUT.fits WAVE.fits FWHM.fits
    """
    rss = RSS()
    rss.loadFitsData(in_rss)
    if cropping:
        crop_start = int(cropping[0]) - 1
        crop_end = int(cropping[1]) - 1
    else:
        crop_start = 0
        crop_end = rss._data.shape[1] - 1
    wave_trace = FiberRows()
    wave_trace.loadFitsData(arc_wave)
    rss.setWave(wave_trace.getData()[0][:, crop_start:crop_end])
    rss._data = rss._data[:, crop_start:crop_end]
    if rss._error is not None:
        rss._error = rss._error[:, crop_start:crop_end]
    if rss._mask is not None:
        rss._mask = rss._mask[:, crop_start:crop_end]

    try:
        rss.copyHdrKey(wave_trace, "HIERARCH PIPE DISP RMS MEDIAN")
        rss.copyHdrKey(wave_trace, "HIERARCH PIPE DISP RMS MIN")
        rss.copyHdrKey(wave_trace, "HIERARCH PIPE DISP RMS MAX")
    except KeyError:
        pass

    if arc_fwhm != "":
        fwhm_trace = FiberRows()
        fwhm_trace.loadFitsData(arc_fwhm)
        rss.setInstFWHM(fwhm_trace.getData()[0][:, crop_start:crop_end])
    rss.writeFitsData(out_rss)


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
    rss = RSS()
    rss.loadFitsData(in_rss)
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
def resample_wavelength(in_rss: str, out_rss: str, method: str = "spline",
                        start_wave: float = None, end_wave: float = None,
                        disp_pix: float = None, err_sim: int = 500,
                        replace_error: float = 1.e10, correctHvel: float = None,
                        compute_densities: bool = False, extrapolate: bool = True,
                        parallel: str = "auto"):
    """
    Resamples the RSS wavelength solutions a common wavelength solution for each fiber

    A Monte Carlo scheme can be used to propagte the error to the resample spectrum.
    Note that correlated noise is not taken into account with the procedure.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file where the wavelength is stored as a pixel table
    out_rss : string
        Output RSS FITS file with a common wavelength solution
    method : string, optional with default: 'spline'
        Interpolation scheme used for the spectral resampling of the data.
        Available options are:
            - spline
            - linear
    start_wave : string of float, optional with default: ''
        Start wavelength for the common resampled wavelength solution.
        The "optimal" wavelength will be used if the paramter is empty.
    endt_wave : string of float, optional with default: ''
        End wavelength for the common resampled wavelength solution
        The "optimal" wavelength will be used if the paramter is empty.
    disp_pix : string of float, optional with default: ''
        Dispersion per pixel for the common resampled wavelength solution.
        The "optimal" dispersion will be used if the paramter is empty.
    err_sim : string of integer (>0), optional with default: '500'
        Number of Monte Carlo simulation per fiber in the RSS to estimate the
        error of the resampled spectrum. If err_sim is set to 0, no error will
        be estimated for the resampled RSS.
    replace_error: strong of float, optional with default: '1e10'
        Error value for bad pixels resampled data, will be ignored if empty
    parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
        Number of CPU cores used in parallel for the computation. If set to
        auto, the maximum number of CPUs for the given system is used.

    Examples
    --------
    user:> lvmdrp rss resampleWave RSS_in.fits RSS_out.fits
    user:> lvmdrp rss resampleWave RSS_in.fits RSS_out.fits start_wave=3700.0 /
    > end_wave=7000.0 disp_pix=2.0 err_sim=0
    """

    rss = loadRSS(in_rss)

    if not start_wave:
        start_wave = numpy.min(rss._wave)
    else:
        start_wave = float(start_wave)
    if not disp_pix:
        disp_pix = numpy.min(rss._wave[:, 1:] - rss._wave[:, :-1])
    else:
        disp_pix = float(disp_pix)

    if not end_wave:
        end_wave = numpy.max(rss._wave)
    else:
        end_wave = float(end_wave)

    if not correctHvel:
        offset_vel = 0.0
    else:
        try:
            offset_vel = float(correctHvel)
        except ValueError:
            offset_vel = rss.getHdrValue(correctHvel)

    ref_wave = numpy.arange(start_wave, end_wave + disp_pix - 0.001, disp_pix)
    rss._wave = rss._wave * (1 + offset_vel / 300000.0)

    if extrapolate:
        collapsed_spec = rss.create1DSpec().resampleSpec(
            ref_wave, method="linear", err_sim=err_sim
        )
    else:
        collapsed_spec = None

    data = numpy.zeros((rss._fibers, len(ref_wave)), dtype=numpy.float32)
    if rss._error is not None and err_sim != 0:
        error = numpy.zeros((rss._fibers, len(ref_wave)), dtype=numpy.float32)
    else:
        error = None
    if rss._inst_fwhm is not None:
        inst_fwhm = numpy.zeros((rss._fibers, len(ref_wave)), dtype=numpy.float32)
    else:
        inst_fwhm = None
    mask = numpy.zeros((rss._fibers, len(ref_wave)), dtype="bool")
    if compute_densities:
        width_pix = numpy.zeros_like(rss._data)
        width_pix[:, :-1] = numpy.fabs(rss._wave[:, 1:] - rss._wave[:, :-1])
        width_pix[:, -1] = width_pix[:, -2]
        rss._data = rss._data / width_pix
        if rss._error is not None:
            rss._error = rss._error / width_pix
    if rss._wave is not None and len(rss._wave.shape) == 2:
        if parallel == "auto":
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus)
            result_spec = []
            for i in range(rss._fibers):
                spec = rss.getSpec(i)
                result_spec.append(
                    pool.apply_async(
                        spec.resampleSpec,
                        args=(ref_wave, method, err_sim, replace_error, collapsed_spec),
                    )
                )
            pool.close()
            pool.join()

        for i in range(rss._fibers):
            if cpus > 1:
                spec = result_spec[i].get()
            else:
                spec = rss.getSpec(i)
                spec = spec.resampleSpec(
                    ref_wave, method, err_sim, replace_error, collapsed_spec
                )
            data[i, :] = spec._data
            if rss._error is not None and err_sim != 0:
                error[i, :] = spec._error
            if rss._inst_fwhm is not None:
                inst_fwhm[i, :] = spec._inst_fwhm
            mask[i, :] = spec._mask
        resamp_rss = RSS(
            data=data,
            wave=ref_wave,
            inst_fwhm=inst_fwhm,
            header=rss.getHeader(),
            error=error,
            mask=mask,
        )

    resamp_rss.writeFitsData(out_rss)


def matchResolution_drp(in_rss, out_rss, targetFWHM, parallel="auto"):
    """
    Homogenise the LSF of the RSS to a common spectral resolution (FWHM)

    This task smooths the RSS with a Gaussian kernel of the corresponding
    width. A pixel table with the spectral resolution needs to be present in
    the RSS. If the spectral resolution is higher than than the target spectral
    resolution for certain pixel, no smoothing is applied for those pixels.

    Parameters
    ----------
    in_rss : string
        Input RSS FITS file with a pixel table for the spectral resolution
    out_rss : string
        Output RSS FITS file with a homogenised spectral resolution
    targetFWHM : string of float
        Spectral resolution in FWHM to which the RSS shall be homogenised
    parallel: either string of integer (>0) or  'auto', optional with default: 'auto'
        Number of CPU cores used in parallel for the computation. If set to
        auto, the maximum number of CPUs for the given system is used.

    Examples
    --------
    user:> lvmdrp rss matchResolution RSS_in.fits RSS_out.fits 6.0
    """
    targetFWHM = float(targetFWHM)
    rss = RSS()
    rss.loadFitsData(in_rss)

    smoothFWHM = numpy.zeros_like(rss._inst_fwhm)
    select = rss._inst_fwhm < targetFWHM
    smoothFWHM[select] = numpy.sqrt(targetFWHM**2 - rss._inst_fwhm[select] ** 2)

    if parallel == "auto":
        cpus = cpu_count()
    else:
        cpus = int(parallel)

    if cpus > 1:
        pool = Pool(cpus)
        threads = []
        for i in range(len(rss)):
            threads.append(
                pool.apply_async(rss[i].smoothGaussVariable, ([smoothFWHM[i, :]]))
            )

        for i in range(len(rss)):
            rss[i] = threads[i].get()
        pool.close()
        pool.join()
    else:
        for i in range(len(rss)):
            rss[i] = rss[i].smoothGaussVariable(smoothFWHM[i, :])
    rss._inst_fwhm = None
    rss.setHdrValue(
        "HIERARCH PIPE SPEC RES", targetFWHM, "FWHM in A of spectral resolution"
    )
    rss.writeFitsData(out_rss)


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
    rss = RSS()
    rss.loadFitsData(in_rss)
    splitted_rss = rss.splitFiberType(contains)
    for i in range(len(splitted_rss)):
        splitted_rss[i].writeFitsData(splitted_out[i])


# TODO: for twilight fiber flats, normalize the individual flats before combining to
# remove the time dependence
def create_fiberflat(in_rss: str, out_rss: str, median_box: int = 0,
                     gaussian_kernel: int = 5,
                     poly_deg: int = 0, poly_kind: str = "poly",
                     clip_range: List[float] = None,
                     wave_range: List[float] = None,
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
    in_rss : str
        path to a extracted and wavelength calibrated continuum exposure
    out_rss : str
        path to the output fiberflat
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
    display_plots : bool, optional
        whether to display or not the diagnostic plots, by default False

    Returns
    -------
    RSS
        computed fiberflat
    """
    # read continuum exposure
    log.info(f"reading continuum exposure from {os.path.basename(in_rss)}")
    rss = loadRSS(in_rss)

    # wavelength calibration check
    if rss._wave is None:
        log.error(f"RSS {os.path.basename(in_rss)} has not been wavelength calibrated")
        return None
    elif len(rss._wave.shape) != 1:
        log.error(f"RSS {os.path.basename(in_rss)} has not been resampled to a common wavelength grid")
        return None
    else:
        wdelt = rss._wave[1] - rss._wave[0]

    # copy original data into output fiberflat object
    fiberflat = copy(rss)
    fiberflat._error = None

    # apply median smoothing to data
    if median_box > 0:
        median_box_pix = int(median_box / wdelt)
        log.info(f"applying median smoothing with box size {[1, median_box]} angstroms ({[1, median_box_pix]} pixels)")
        rss._data = ndimage.filters.median_filter(rss._data, (1, median_box_pix))
    
    # calculate normalization within a given window or on the full array
    if wave_range is not None:
        log.info(f"caculating normalization in wavelength range {wave_range[0]:.2f} - {wave_range[1]:.2f} angstroms")
        wave_select = (wave_range[0] <= rss._wave) & (wave_range[1] <= rss._wave)
        norm = numpy.median(rss._data[wave_select, :], axis=0)
    else:
        log.info(f"caculating normalization in full wavelength range ({rss._wave[0]:.2f} - {rss._wave[-1]:.2f} angstroms)")
        norm = bn.nanmedian(rss._data, axis=0)

    # normalize fibers where norm has valid values
    select = norm > 0
    log.info(f"computing fiberflat across {rss._fibers} fibers and {numpy.sum(select)} wavelength bins")
    normalized = numpy.zeros_like(rss._data)
    normalized[:, select] = rss._data[:, select] / norm[select][None, :]
    fiberflat._data = normalized
    fiberflat._mask |= normalized <= 0

    # apply clipping
    if clip_range is not None:
        log.info(f"cliping fiberflat to range {clip_range[0]:.2f} - {clip_range[1]:.2f}")
        mask = numpy.logical_or(fiberflat._data < clip_range[0], fiberflat._data > clip_range[1])
        if fiberflat._mask is not None:
            mask = numpy.logical_or(fiberflat._mask, mask)
        fiberflat.setData(mask=mask)

    # apply gaussian smoothing
    if gaussian_kernel > 0:
        gaussian_kernel_pix = int(gaussian_kernel / wdelt)
        log.info(f"applying gaussian smoothing with kernel size {gaussian_kernel} angstroms ({gaussian_kernel_pix} pixels)")
        for ifiber in range(rss._fibers):
            spec = fiberflat.getSpec(ifiber)
            spec.smoothSpec(gaussian_kernel, method="gauss")
            fiberflat._data[ifiber, :] = spec._data

    # polynomial smoothing
    if poly_deg != 0:
        log.info(f"applying polynomial fitting with degree {poly_deg} and kind '{poly_kind}'")
        for ifiber in range(fiberflat._fibers):
            spec = fiberflat.getSpec(ifiber)
            spec.smoothPoly(deg=poly_deg, poly_kind=poly_kind)
            fiberflat._data[ifiber, :] = spec._data
    
    # create diagnostic plots
    log.info("creating diagnostic plots for fiberflat")
    fig, axs = create_subplots(to_display=display_plots, nrows=3, ncols=1, figsize=(12, 15), sharex=True)
    # plot original continuum exposure, fiberflat and corrected fiberflat per fiber
    colors = plt.cm.Spectral(numpy.linspace(0, 1, fiberflat._fibers))
    for ifiber in range(fiberflat._fibers):
        # input data
        axs[0].step(rss._wave, rss._data[ifiber], color=colors[ifiber], alpha=0.5, lw=1)
        # fiberflat
        axs[1].step(fiberflat._wave, fiberflat._data[ifiber], lw=1, color=colors[ifiber])
        # corrected fiberflat
        axs[2].step(fiberflat._wave, rss._data[ifiber] / fiberflat._data[ifiber], lw=1, color=colors[ifiber])
    # plot median spectrum
    axs[0].step(fiberflat._wave, norm, color="0.1", lw=2, label="median spectrum")
    # add labels and titles
    axs[0].set_ylabel("counts (e-/s)")
    axs[0].set_title("median spectrum", loc="left")
    axs[1].set_ylabel("relative transmission")
    axs[1].set_title("fiberflat", loc="left")
    axs[2].set_ylabel("corr. counts (e-/s)")
    axs[2].set_title("corrected fiberflat", loc="left")
    axs[2].set_xlabel("wavelength (angstroms)")
    fig.suptitle(f"fiberflat creation from continuum frame '{os.path.basename(in_rss)}'", fontsize=16)
    # display/save plots
    save_fig(
        fig,
        product_path=out_rss,
        to_display=display_plots,
        figure_path="qa",
        label="fiberflat"
    )

    # perform some statistic about the fiberflat
    if fiberflat._mask is not None:
        select = numpy.logical_not(fiberflat._mask)
    else:
        select = fiberflat._data == fiberflat._data
    min = bn.nanmin(fiberflat._data[select])
    max = bn.nanmax(fiberflat._data[select])
    mean = bn.nanmean(fiberflat._data[select])
    median = bn.nanmedian(fiberflat._data[select])
    std = bn.nanstd(fiberflat._data[select])
    log.info(f"fiberflat statistics: {min = :.3f}, {max = :.3f}, {mean = :.2f}, {median = :.2f}, {std = :.3f}")

    log.info(f"writing fiberflat to {os.path.basename(out_rss)}")
    fiberflat.setHdrValue(
        "HIERARCH PIPE FLAT MIN", float("%.3f" % (min)), "Mininum fiberflat value"
    )
    fiberflat.setHdrValue(
        "HIERARCH PIPE FLAT MAX", float("%.3f" % (max)), "Maximum fiberflat value"
    )
    fiberflat.setHdrValue(
        "HIERARCH PIPE FLAT AVR", float("%.2f" % (mean)), "Mean fiberflat value"
    )
    fiberflat.setHdrValue(
        "HIERARCH PIPE FLAT MED", float("%.2f" % (median)), "Median fiberflat value"
    )
    fiberflat.setHdrValue(
        "HIERARCH PIPE FLAT STD", float("%.3f" % (std)), "rms of fiberflat values"
    )
    fiberflat._header["CUNIT"] = "dimensionless"
    fiberflat._header["IMAGETYP"] = "fiberflat"
    fiberflat.writeFitsData(out_rss)
    
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
    trace = FiberRows()
    trace.loadFitsData(trace_in)

    if poly_smooth == "":
        trace = trace + (numpy.median(offsets.flatten()) * -1)
    else:
        split_trace = trace.split(offsets.shape[1], axis="y")
        offset_trace = FiberRows()
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


def apply_fiberflat(in_rss: str, out_rss: str, in_flat: str, clip_below: float = 0.2) -> RSS:
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
    out_rss : str
        output RSS file path with fiberflat correction applied
    in_flat : str
        input RSS file path to the fiberflat
    clip_below : float, optional
        minimum relative transmission considered. Values below will be masked, by default 0.2

    Returns
    -------
    RSS
        fiberflat corrected RSS
    """
    # load target data
    log.info(f"reading target data from {os.path.basename(in_rss)}")
    rss = RSS()
    rss.loadFitsData(in_rss)
    
    # load fiberflat
    log.info(f"reading fiberflat from {os.path.basename(in_flat)}")
    flat = RSS()
    flat.loadFitsData(in_flat)

    # check if fiberflat has the same number of fibers as the target data
    if rss._fibers != flat._fibers:
        log.error(f"number of fibers in target data ({rss._fibers}) and fiberflat ({flat._fibers}) do not match")
        return None
    
    # check if fiberflat has the same wavelength grid as the target data
    if not numpy.array_equal(rss._wave, flat._wave):
        log.error("target data and fiberflat have different wavelength grids")
        return None

    # apply fiberflat
    log.info(f"applying fiberflat correction to {rss._fibers} fibers with minimum relative transmission of {clip_below}")
    for i in range(flat._fibers):
        # extract fibers spectra
        spec_flat = flat.getSpec(i)
        spec_data = rss.getSpec(i)

        # interpolate fiberflat to target wavelength grid to fill in missing values
        flat_resamp = spec_flat.resampleSpec(spec_data._wave, err_sim=0)
        
        # apply clipping
        select_clip_below = (flat_resamp < clip_below) | numpy.isnan(flat_resamp._data)
        flat_resamp._data[select_clip_below] = 0
        flat_resamp._mask[select_clip_below] = True

        # correct
        spec_new = spec_data / flat_resamp
        rss.setSpec(i, spec_new)
    
    # write out corrected RSS
    log.info(f"writing fiberflat corrected RSS to {os.path.basename(out_rss)}")
    rss.writeFitsData(out_rss)

    return rss


def combineRSS_drp(in_rsss, out_rss, method="mean"):
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
    # convert input parameters to proper type
    list_rss = in_rsss.split(",")

    rss_list = []
    for i in list_rss:
        # load subimages from disc and append them to a list
        rss = loadRSS(i)
        rss_list.append(rss)
    # combined_header = combineHdr(rss_list)
    combined_rss = RSS()
    combined_rss.combineRSS(rss_list, method=method)
    # combined_rss.setHeader(header=combined_header._header)
    # write out FITS file
    combined_rss.writeFitsData(out_rss)


def glueRSS_drp(rsss, out_rss):
    """concatenates the given RSS list to a single RSS

    Parameters
    ----------
    rsss : array_like
        list of RSS file paths
    out_rss : str
        output RSS file path
    """
    list_rss = rsss.split(",")
    glueRSS(list_rss, out_rss)


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
    rss = RSS()
    rss.loadFitsData(in_rss)
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
    rss1 = RSS()
    rss1.loadFitsData(in_rss)
    rss2 = RSS()
    rss2.loadFitsData(out_rss)
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
    rss = RSS()
    rss.loadFitsData(in_rss)
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
        galExtCurve = ancillary_func.galExtinct(rss._wave, Rv)
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


def join_spec_channels(in_rss: List[str], out_rss: str):
    """combine the given RSS list through the overlaping wavelength range

    Run once per exposure, for one spectrograph at a time.
    in_rss is a list of 3 files, one for each channel, for a given
    exposure and spectrograph id.

    Parameters
    ----------
    in_rss : array_like
        list of RSS file paths for each spectrograph channel
    out_rss : str
        output RSS file path

    Returns
    -------
    RSS
        combined RSS
    """

    # read all three channels
    log.info(f"loading RSS files: {in_rss}")
    rsss = [loadRSS(rss_path) for rss_path in in_rss]
    # set masked pixels to NaN
    [rss.apply_pixelmask() for rss in rsss]

    # get wavelengths
    log.info("computing best wavelength array")
    waves = [rss._wave for rss in rsss]
    # compute the combined wavelengths
    new_wave = wave_little_interpol(waves)
    sampling = numpy.diff(new_wave)
    log.info(f"new wavelength sampling: min = {sampling.min()}, max = {sampling.max()}")

    # define interpolators
    log.info("interpolating RSS data in new wavelength array")
    fluxes_f = [interpolate.interp1d(rss._wave, rss._data, axis=1, bounds_error=False, fill_value=numpy.nan) for rss in rsss]
    errors_f = [interpolate.interp1d(rss._wave, rss._error, axis=1, bounds_error=False, fill_value=numpy.nan) for rss in rsss]
    masks_f = [interpolate.interp1d(rss._wave, rss._mask, axis=1, kind="nearest", bounds_error=False, fill_value=numpy.nan) for rss in rsss]
    lsfs_f = [interpolate.interp1d(rss._wave, rss._inst_fwhm, axis=1, bounds_error=False, fill_value=numpy.nan) for rss in rsss]
    # evaluate interpolators
    fluxes = numpy.asarray([f(new_wave) for f in fluxes_f])
    errors = numpy.asarray([f(new_wave) for f in errors_f])
    masks = numpy.asarray([f(new_wave) for f in masks_f])
    lsfs = numpy.asarray([f(new_wave) for f in lsfs_f])

    # define weights for channel combination
    log.info("calculating weights for channel combination")
    weights = 1.0 / errors ** 2
    norms = bn.nansum(weights, axis=0)
    weights = weights / norms[None, :, :]

    # channel-combine RSS data
    log.info("combining channel data")
    new_data = bn.nansum(fluxes * weights, axis=0)
    new_inst_fwhm = bn.nansum(lsfs * weights, axis=0)
    new_error = numpy.sqrt(1 / bn.nansum(weights * norms[None, :, :], axis=0))
    new_mask = bn.nansum(masks, axis=0).astype(bool)

    # create RSS
    log.info(f"writing output RSS to {os.path.basename(out_rss)}")
    new_hdr = rsss[0]._header.copy()
    new_hdr["CCD"] = ",".join([rss._header["CCD"] for rss in rsss])
    new_hdr["CDELT1"] = new_wave[1] - new_wave[0]
    new_hdr["CRVAL1"] = new_wave[0]
    new_hdr["NAXIS1"] = len(new_wave)
    new_rss = RSS(data=new_data, error=new_error, mask=new_mask, wave=new_wave, inst_fwhm=new_inst_fwhm, header=new_hdr)
    # write output RSS
    new_rss.writeFitsData(out_rss)

    return new_rss

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
    fiberflat = RSS()
    fiberflat.loadFitsData(in_fiberflat)

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
