# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jan 25, 2023
# @Filename: fluxCalMethod
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

# FROM RENBIN'S PAPER (RENBIN+2016) ---------------------------------------------------------------
# TODO: measure the PSF from the guider
#   * stack image guider to obtain the effective PSF for the science frames
#   * fit with a double Gaussian each guiding star with varying A and sigma
#   * pick the sharpest PSF among the N in-focus guiding stars as reference PSF, (p_0) (check that conditions justifying this choice hold for LVM)
#   * choose a effective guiding wavelength (5400 AA)
# TODO: predict wavelength-dependent PSF
#   * seeing as a function of wavelength:
#       p_lambda(r) = p_0 * [r * (lambda/lambda_0)^(1/5)]
#   * interpolate Gunn+2006 to obtain focus offset as a function of wavelength
#   * convolve PSF with a ring kernel (k_[lambda,d](r)) of radii offset*1/n offset*1/N, where f/n is the telescope beam, N is the size of the secondary mirror and d is the distance to the center of the plate
#   * convolve again with a circular step function with radius = radius of the fiber
# TODO: stellar typing
#   * divide the sky-subtracted std stars by the throughput vector (average throughput from tens of previous run)
#   * select fiber with the maximum integrated flux across the wavelength range for each mini-bundle as the reference spectrum
#   * fit a model to each reference spectrum
#   * scale each model spectrum to match the PSF r-band magnitude of individual std stars
# TODO: fitting flux ratios
#   * select fiber with the maximum integrated flux from 3500 to 10500AA
#   * integrate the flux in eight wide wavelength windows
#   * calculate the ratio between each fiber and the reference fiber in each window
#   * run a MCMC to fit for x,y of the star and scaling and rotation of the DAR vector
#   * given a set of the above parameters, compute the expected flux ratios from the PSF model and repeat until minimum chi-square is reached
#   * scale the PSF to smaller and larger sizes and compute the minimum chi-square by fitting a quadratic function, the best fit is found among all the fibers
#   * run the MCMC again using the best PSF
# TODO: deriving throughput loss
#   * compute the fraction of the flux of the PSF that is covered by the fiber as a function of wavelength
#   * compute the expected flux of the star (frac_PSF * model_star)
#   * divide the observed spectra by the expected flux (obs_star / (frac_PSF * model_star))
# -------------------------------------------------------------------------------------------------

# FROM THE CODE BASE (PY3D) -----------------------------------------------------------------------

import os
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import interpolate
from scipy import stats
from scipy import ndimage
from scipy.ndimage import median_filter
import re

from astropy.stats import biweight_location, biweight_scale
from astropy.table import Table
from astropy.io import fits

from lvmdrp.core.rss import RSS, loadRSS, lvmFFrame
from lvmdrp.core.spectrum1d import Spectrum1D
import lvmdrp.core.fluxcal as fluxcal
from lvmdrp.core.sky import get_sky_mask_uves, get_z_continuum_mask
from lvmdrp import log

from lvmdrp.core.plot import plt, create_subplots, save_fig
from lvmdrp.core.constants import MASTERS_DIR

description = "provides flux calibration tasks"

__all__ = [
    "createSensFunction_drp",
    "createSensFunction2_drp",
    "quickFluxCalibration_drp",
    "correctTelluric_drp",
]

def apply_fluxcal(in_rss: str, out_fframe: str, method: str = 'STD', display_plots: bool = False):
    """applies flux calibration to spectrograph-combined data

    Parameters
    ----------
    in_rss : str
        input RSS file
    out_rss : str
        output RSS file
    method : str
        'STD' - apply calibration inferred from standard stars  (default)
        'SCI' - apply calibration inferred from field stars in science ifu (fallback if STD not available)
        'MOD' - apply calibration inferred from stellar atmosphere models
        'NONE' - do not apply flux calibration
    display_plots : bool, optional

    Returns
    -------
    rss : RSS
        flux-calibrated RSS object
    """

    assert (method in ['STD', 'SCI', 'MOD', 'NONE']), 'Fluxcal method must be either STD, SCI, MOD or NONE'

    # read all three channels
    log.info(f"loading RSS file {os.path.basename(in_rss)}")
    rss = loadRSS(in_rss)

    # initialize the lvmFFrame object
    fframe = lvmFFrame(data=rss._data, error=rss._error, mask=rss._mask, header=rss._header,
                       wave=rss._wave, lsf=rss._lsf,
                       sky_east=rss._sky_east, sky_east_error=rss._sky_east_error,
                       sky_west=rss._sky_west, sky_west_error=rss._sky_west_error,
                       fluxcal_std=rss._fluxcal_std, fluxcal_sci=rss._fluxcal_sci,
                       fluxcal_mod=rss._fluxcal_mod, slitmap=rss._slitmap)

    # check for flux calibration data
    fframe.setHdrValue("FLUXCAL", 'NONE', "flux-calibration method")
    if method == "NONE":
        log.info("skipping flux calibration")
        fframe.writeFitsData(out_fframe)
        return fframe

    expnum = fframe._header["EXPOSURE"]
    channel = fframe._header["CCD"]

    # set masked pixels to NaN
    fframe.apply_pixelmask()
    # load fibermap and filter for current spectrograph
    slitmap = fframe._slitmap

    # define exposure time factors
    exptimes = np.zeros(len(slitmap))
    exptimes[
        (slitmap["targettype"] == "science") | (slitmap["targettype"] == "SKY")
    ] = fframe._header["EXPTIME"]
    if len(fframe._header["STD*EXP"]) == 0:
        exptimes[slitmap["telescope"] == "Spec"] = fframe._header["EXPTIME"] / 12
        log.warning(f"missing standard stars exposure time, assuming exptime = {fframe._header['EXPTIME'] / 12}s")
        fframe.add_header_comment(f"missing standard stars exposure time, assuming exptime = {fframe._header['EXPTIME'] / 12} s")
    else:
        for std_hd in fframe._fluxcal_std.colnames:
            exptime = fframe._header[f"{std_hd[:-3]}EXP"]
            fiberid = fframe._header[f"{std_hd[:-3]}FIB"]
            exptimes[slitmap["orig_ifulabel"] == fiberid] = exptime

    # apply joint sensitivity curve
    fig, ax = create_subplots(to_display=display_plots, figsize=(10, 5))
    fig.suptitle(f"Flux calibration for {expnum = }, {channel = }")
    log.info(f"computing joint sensitivity curve for channel {channel}")
    # calculate exposure time factors
    # std_exp = np.asarray([fframe._header.get(f"{std_hd[:-3]}EXP", 1.0) for std_hd in fframe._fluxcal.colnames])
    # weights = std_exp / std_exp.sum()
    # TODO: reject sensitivity curves based on the overall shape by normalizing using a median curve
    # calculate the biweight mean sensitivity

    # if instructed, use standard stars
    if method == 'STD':
        log.info("flux-calibratimg using STD standard stars")
        sens_arr = fframe._fluxcal_std.to_pandas().values  # * (std_exp / std_exp.sum())[None]
        sens_ave = biweight_location(sens_arr, axis=1, ignore_nan=True)
        sens_rms = biweight_scale(sens_arr, axis=1, ignore_nan=True)

        # fix case of all invalid values
        if (sens_ave == 0).all() or np.isnan(sens_ave).all() or (sens_ave<0).any():
            log.warning("all standard star sensitivities are <=0 or NaN, falling back to SCI stars")
            rss.add_header_comment("all standard star sensitivities are <=0 or NaN, falling back to SCI stars")
            method = 'SCI'  # fallback to sci field stars
        else:
            fframe.setHdrValue("FLUXCAL", 'STD', "flux-calibration method")

    # fall back to science ifu field stars if above failed or if instructed to use this method
    if method == 'SCI':
        log.info("flux-calibratimg using SCI field stars")
        sens_arr = fframe._fluxcal_sci.to_pandas().values  # * (std_exp / std_exp.sum())[None]
        sens_ave = biweight_location(sens_arr, axis=1, ignore_nan=True)
        sens_rms = biweight_scale(sens_arr, axis=1, ignore_nan=True)

        # fix case of all invalid values
        if (sens_ave == 0).all() or np.isnan(sens_ave).all():
            log.warning("all field star sensitivities are zero or NaN, can't calibrate")
            rss.add_header_comment("all field star sensitivities are zero or NaN, can't calibrate")
            sens_ave = np.ones_like(sens_ave)
            sens_rms = np.zeros_like(sens_rms)
        else:
            fframe.setHdrValue("FLUXCAL", 'SCI', "flux-calibration method")

    if method == 'MOD':
        log.info("flux-calibratimg using model stellar spectra")
        sens_arr = fframe._fluxcal_mod.to_pandas().values  # * (std_exp / std_exp.sum())[None]
        sens_ave = biweight_location(sens_arr, axis=1, ignore_nan=True)
        sens_rms = biweight_scale(sens_arr, axis=1, ignore_nan=True)

        # fix case of all invalid values
        if (sens_ave == 0).all() or np.isnan(sens_ave).all():
            log.warning("all sensitivities from models are zero or NaN, can't calibrate")
            rss.add_header_comment("all sensitivities from model are zero or NaN, can't calibrate")
            sens_ave = np.ones_like(sens_ave)
            sens_rms = np.zeros_like(sens_rms)
        else:
            fframe.setHdrValue("FLUXCAL", 'MOD', "flux-calibration method")

    if method != 'NONE':
        # update the fluxcal extension
        fframe._fluxcal_std["mean"] = sens_ave
        fframe._fluxcal_std["rms"] = sens_rms

        ax.set_title(f"flux calibration for {channel = } with {method = }", loc="left")
        for j in range(sens_arr.shape[1]):
            std_hd = fframe._fluxcal_std.colnames[j][:-3]
            std_id = fframe._header.get(f"{std_hd}FIB", "unknown")

            ax.plot(fframe._wave, sens_arr[:, j], "-", lw=1, label=std_id)
        ax.plot(fframe._wave, sens_ave, "-r", lw=2, label="mean")
        ax.set_yscale("log")
        ax.set_xlabel("wavelength (Angstrom)")
        ax.set_ylabel("sensitivity [(ergs/s/cm^2/A) / (e-/s/A)]")
        ax.legend(loc="upper right")
        fig.tight_layout()
        save_fig(fig, product_path=out_fframe, to_display=display_plots, figure_path="qa", label="fluxcal")

    # flux-calibrate and extinction correct data
    # Note that we assume a constant extinction curve here!
    log.info(f"Extinction correcting science and sky spectra, curve {os.getenv('LVMCORE_DIR')+'/etc/lco_extinction.txt'}")
    txt = np.genfromtxt(os.getenv("LVMCORE_DIR") + "/etc/lco_extinction.txt")
    lext, ext = txt[:, 0], txt[:, 1]
    ext = np.interp(fframe._wave, lext, ext)
    sci_secz = fframe._header["TESCIAM"]

    # optionally sky flux calibration
    if method == 'NONE':
        log.info("skipping flux calibration")
        fframe._data /= exptimes[:, None]
        fframe._error /= exptimes[:, None]
        if fframe._sky is not None:
            fframe._sky /= exptimes[:, None]
        if fframe._sky_error is not None:
            fframe._sky_error /= exptimes[:, None]
        if fframe._sky_east is not None:
            fframe._sky_east /= exptimes[:, None]
        if fframe._sky_east_error is not None:
            fframe._sky_east_error /= exptimes[:, None]
        if fframe._sky_west is not None:
            fframe._sky_west /= exptimes[:, None]
        if fframe._sky_west_error is not None:
            fframe._sky_west_error /= exptimes[:, None]
        fframe.setHdrValue("FLUXCAL", 'NONE', "flux-calibration method")
        fframe.setHdrValue("BUNIT", "electron/s/A", "flux units")
    else:
        log.info("flux-calibrating data science and sky spectra")
        fframe._data *= sens_ave * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        fframe._error *= sens_ave * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        if fframe._sky is not None:
            fframe._sky *= sens_ave * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        if fframe._sky_error is not None:
            fframe._sky_error *= sens_ave * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        if fframe._sky_east is not None:
            fframe._sky_east *= sens_ave * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        if fframe._sky_east_error is not None:
            fframe._sky_east_error *= sens_ave * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        if fframe._sky_west is not None:
            fframe._sky_west *= sens_ave * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        if fframe._sky_west_error is not None:
            fframe._sky_west_error *= sens_ave * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        fframe.setHdrValue("BUNIT", "ergs/s/cm^2/A", "flux units")

    log.info(f"writing output file in {os.path.basename(out_fframe)}")
    fframe.writeFitsData(out_fframe)

    return fframe


def linear_to_logscale(wl, flux):
    wl_log = np.log(wl)
    wl_log_step = np.min((wl_log-np.roll(wl_log,1))[1:])
    n_elements = np.ceil((np.max(wl_log) - np.min(wl_log))/wl_log_step).astype(int)
    wl_log_regular = np.linspace(np.min(wl_log), np.max(wl_log), n_elements)
    rec = np.isfinite(flux)
    flux_log = np.interp(wl_log_regular, np.log(wl[rec]), flux[rec])
    return wl_log_regular, flux_log

def logscale_to_linear(wl_regular, wl_log, flux_log, shift=0):
    rec = np.isfinite(flux_log)
    #print(np.exp(wl_log[rec]+shift))
    flux = np.interp(wl_regular, np.exp(wl_log[rec]+shift), flux_log[rec])
    return flux


def smoothSpec_old(self, size, method="gauss", mode="nearest"):
    """
    Smooth the spectrum

    Parameters
    --------------
    size : float or int
        Size of the smooth window or Gaussian width (sigma)
    method : string, optional with default: 'gauss'
        Available methods are 'gauss' - convolution with Gaussian kernel or
        'median' - median smoothing of the spectrum
    mode :  string, optional with default: 'nearest'
        Set the mode how to handle the boundarys within the convolution
        Possilbe modes are: reflect, constant, nearest, mirror,  wrap
    """
    if method == "gauss":
        # filter with Gaussian kernel
        median_filt = ndimage.filters.gaussian_filter1d(self['flux'], size, mode=mode)
    elif method == "median":
        # filter with median filter
        median_filt = ndimage.filters.median_filter(self['flux'], size, mode=mode)
    elif method == "BSpline":
        smooth = interpolate.splrep(
            self['wavelength'],
            self['flux'],
            #w=1.0 / np.sqrt(np.fabs(self['flux'])),
            s=size,
        )
        median_filt = interpolate.splev(self['wavelength'], smooth, der=0)
    return median_filt


def model_selection(in_rss, GAIA_CACHE_DIR=None, width=3, plot=True):
    """ Selection of the stellar atmosphere model spectra (POLLUX database, AMBRE library)
    Read all the models already convolved with Gaia LSF and normalized
    Correct observed standard spectrum for the atmospheric extinction
    Fit continuum to observed (corrected for the extinction) standard spectra (in 3 channels separately)
    Normalise observed standard spectra and stitch 3 channels together
    Fit continuum to Gaia spectra
    Multuply Gaia continuum and normalise observed standard spectra
    Mask telluric lines
    Normalise to total sum = 1
    Find the best-fit model from the set of models convolved with Gaia LSF and nolmalised (to total sum = 1)
    with chi-square method
    Check for possible velocity offsets - CAN BE USED LATER TO ADD CORRECTION FOR VELOCITIES
    Find the conversion coefficient between model units and Gaia units:
        Read the best-fit model with good resolution, non-normalised
        Convolve with Gaia LSF
        Calculate median fot stdflux/model_convolved_to_gaia - we will use this coefficient


    :param in_rss:
    :param GAIA_CACHE_DIR:
    :param width:
    :return:
    best_fit_models
        names of best-fit models
    model_to_gaia_median
        array with conversion coefficients between model units and Gaia units
    """
    # TODO: think about uniting this code and the fluxcal code that iterates over cameras?
    # TODO: find a place under the calib directory structure for the stellar models
    # TODO: telluric list should go in lvmcore
    # models_dir = '/Users/amejia/Downloads/stellar_models/'
    models_dir = os.path.join(MASTERS_DIR, "stellar_models")
    template_model = 'M_p6250g4.0z-0.25t1.0_a-0.10c0.00n0.00o-0.10r0.00s0.00_VIS.fits'
    telluric_file = os.path.join(os.getenv("LVMCORE_DIR"), 'etc', 'telluric_lines.txt')  # wavelength regions with Telluric
    # absorptions based on KPNO data (unknown source) with a 1% transmission threshold this file is used as a mask for
    # the fit of standard stars - from Alfredo.
    # https://github.com/desihub/desispec/blob/main/py/desispec/data/arc_lines/telluric_lines.txt
    telluric_tab = Table.read(telluric_file, format='ascii.fixed_width_two_line')
    # mask_for_fit = telluric_tab

    model_names = [f for f in listdir(join(models_dir, 'median_normalized_logscale')) if
                   isfile(join(models_dir, 'median_normalized_logscale', f)) and (f.lower().endswith('.fits'))]
    model_specs_norm = []

    # read the downsampled to 2A, normalized, log-wavelength model grid
    n_models = len(model_names)
    log.info(f'Number of models: {n_models}')
    for i in range(n_models):
        with fits.open(join(models_dir, 'median_normalized_logscale', model_names[i]), memmap=False) as hdul:
            convolved_tmp = hdul[0].data
        model_specs_norm.append(convolved_tmp)
    model_specs_norm = np.array(model_specs_norm)

    GAIA_CACHE_DIR = "./" if GAIA_CACHE_DIR is None else GAIA_CACHE_DIR
    log.info(f"Using Gaia CACHE DIR '{GAIA_CACHE_DIR}'")

    # Parameters for continuum fit
    nknots = 10
    median_box = 30
    niter = 10
    mask_bands = ([3060, 3110], [3200, 3300], [3785, 3805], [3820, 3840], [3870, 3980],
                  [4080, 4120], [4180, 4550], [4800, 4900], [6450, 6700], [8400, 8900],
                  [8950, 9050], [9200, 9250], [9500, 9600], [9950, 10150], [10750, 11150])

    rss = []
    w = []
    ext = []
    normalized_spectra_all_bands = []
    normalized_spectra_unconv_all_bands = []
    std_errors_all_bands = []
    lsf_all_bands = []
    std_spectra_all_bands = [] ## contains original std spectra for all stars in ALL band

    for b in range(len(in_rss)):
        std_spectra = []  # contains original std spectra for all stars in each band
        normalized_spectra = []
        normalized_spectra_unconv = []
        std_errors = []
        lsf = []
        fibers = []
        #log.info(f"loading input RSS file '{os.path.basename(in_rss[b])}'")
        rss_tmp = RSS.from_file(in_rss[b])

        # get the list of standards from the header
        try:
            stds = fluxcal.retrieve_header_stars(rss=rss_tmp)
        except KeyError:
            pass
            # log.warning(f"no standard star metadata found in '{in_rss}', skipping sensitivity measurement")
            # rss.add_header_comment(f"no standard star metadata found in '{in_rss}', skipping sensitivity measurement")
            # rss.set_fluxcal(fluxcal=res_std, source='std')
            # rss.writeFitsData(in_rss)
            # TODO: fix this, this seems to be copy-pasted from the gaia code
            # return res_std, mean_std, rms_std, rss

        # wavelength array
        w_tmp = rss_tmp._wave
        w.append(w_tmp)

        # load the sky masks
        channel = rss_tmp._header['CCD']
        #w = rss._wave

        m = get_sky_mask_uves(w[b], width=width)
        m2 = None
        if channel == "z":
            m2 = get_z_continuum_mask(w_tmp)

        master_sky = rss_tmp.eval_master_sky()
        # iterate over standard stars
        gaia_ids = []
        for s in stds:
            nn, fiber, gaia_id, exptime, secz = s  # unpack standard star tuple
            gaia_ids.append(gaia_id)
            fibers.append(fiber)

            # find the fiber with our spectrum of that Gaia star, if it is not in the current spectrograph, continue
            select = rss_tmp._slitmap["orig_ifulabel"] == fiber
            fibidx = np.where(select)[0]

            log.info(f"standard fiber '{fiber}', index '{fibidx}', star '{gaia_id}', exptime '{exptime:.2f}', secz '{secz:.2f}'")

            # subtract sky spectrum and divide by exptime
            spec_tmp = rss_tmp._data[fibidx[0], :]
            error_tmp = rss_tmp._error[fibidx[0], :]
            lsf_tmp = rss_tmp._lsf[fibidx[0], :]
            if np.nanmean(spec_tmp) < 100:
                log.warning(f"fiber {fiber} @ {fibidx[0]} has counts < 100 e-, skipping")
                #rss.add_header_comment(f"fiber {fiber} @ {fibidx[0]} has counts < 100 e-, skipping")
                continue

            spec_tmp = (rss_tmp._data[fibidx[0],:] - master_sky._data[fibidx[0],:])/exptime

            # interpolate over bright sky lines and nan values
            mask_bad = ~np.isfinite(spec_tmp)
            spec_tmp = fluxcal.interpolate_mask(w_tmp, spec_tmp, m | mask_bad, fill_value="extrapolate")
            if channel == "z":
                spec_tmp = fluxcal.interpolate_mask(w_tmp, spec_tmp, ~m2 | mask_bad, fill_value="extrapolate")

            # extinction correction
            # load extinction curve
            # Note that we assume a constant extinction curve here!
            txt = np.genfromtxt(os.getenv("LVMCORE_DIR") + "/etc/lco_extinction.txt")
            lext, ext = txt[:, 0], txt[:, 1]
            ext = np.interp(w_tmp, lext, ext)

            # correct for extinction
            spec_ext_corr = spec_tmp.copy()
            spec_ext_corr *= 10 ** (0.4 * ext * secz)
            pxsize = abs(np.nanmedian(w_tmp - np.roll(w_tmp, -1)))
            lsf_conv = np.sqrt(np.clip(2 ** 2 - lsf_tmp ** 2, 0.1, None))/pxsize  # as model spectra were already convolved with lsf=2.0 A,
            # we need to degrade our observed std spectra. Also, convert it to pixels
            mask_bad = ~np.isfinite(spec_tmp)
            mask_lsf = ~np.isfinite(lsf_conv)
            lsf_conv_interpolated = fluxcal.interpolate_mask(w_tmp, lsf_conv, mask_lsf, fill_value="extrapolate")

            # # degrade observed std spectra
            spec_tmp_convolved = fluxcal.lsf_convolve(spec_tmp, lsf_conv_interpolated, w_tmp)

            # Obtain continuum with 160A median filter and normalize spectra
            # best_continuum, continuum_models, masked_pixels, knots = fit_continuum_std(w_tmp,
            #                                                                            spec_tmp_convolved,
            #                                                                            mask_bands=mask_bands,
            #                                                                            threshold=0.5,niter=niter,
            #                                                                            nknots=nknots,
            #                                                                            median_box=median_box)
            std_spec_conv = Table(data=[w_tmp, spec_tmp_convolved],
                                         names=['wave', 'flux'])
            best_continuum = smoothSpec_old(std_spec_conv, int(160/0.5), method="median")
            #std_errors.append(error_tmp/best_continuum)
            error_tmp = 1 / error_tmp**0.5
            std_errors.append(error_tmp / best_continuum)
            normalized_spectra.append(spec_tmp_convolved/best_continuum) # normalized std spestra degraded to 2A for all
                                                                        # standards in each channel
            best_continuum = ndimage.filters.median_filter(spec_tmp, int(160/0.5), mode="nearest")
            normalized_spectra_unconv.append(spec_tmp/best_continuum)
            lsf.append(lsf_tmp) # initial std spec LSF for all standards in each channel
            std_spectra.append(spec_ext_corr)
            #print(nn, fiber)
        #print('!!! one band',std_spectra)

        normalized_spectra_all_bands.append(normalized_spectra) # normalized std spestra degraded to 2A for all
                                                                        # standards and all channels together
        normalized_spectra_unconv_all_bands.append(normalized_spectra_unconv)
        std_errors_all_bands.append(std_errors)
        lsf_all_bands.append(lsf) # initial std spec LSF for all standards and all channel together
        std_spectra_all_bands.append(std_spectra)
    #print('!!!', std_spectra_all_bands)

    # Stitch wavelength arrays in brz together
    wave_b = np.round(w[0],1)
    wave_r = np.round(w[1],1)
    wave_z = np.round(w[2], 1)
    mask_b_norm = (wave_b < min(wave_r))
    mask_r_norm = (wave_r <= 7540)
    mask_z_norm = (wave_z > 7540)
    std_wave_all = np.concatenate((wave_b[mask_b_norm], wave_r[mask_r_norm], wave_z[mask_z_norm]))

    # mask only tellurics - used for calculation of conversion from model to Gaia units
    mask_tellurics = np.zeros_like(std_wave_all, dtype=bool)
    for i in range(len(telluric_tab)):
        mask_tellurics = mask_tellurics | ((std_wave_all > telluric_tab['Start'][i] - 10) & (
                    std_wave_all < telluric_tab['End'][i] + 10))

    # table with masks for tellurics, overlaps between channels, and bluest part og the spectra - used for model matchind
    br_overlap_start = 5775
    br_overlap_end = 5825
    rz_overlap_start = 7520 #7520
    rz_overlap_end = 7580 #7570
    mask_line_start = 8620
    mask_line_end = 8690
    mask_for_fit = telluric_tab
    mask_for_fit['Start'] = mask_for_fit['Start'] - 10
    mask_for_fit['End'] = mask_for_fit['End'] + 10
    mask_for_fit.add_row([3500,3800]) #mask the bluest part of the spectra - prev.[3500,3715]
    mask_for_fit.add_row([br_overlap_start, br_overlap_end])
    mask_for_fit.add_row([rz_overlap_start, rz_overlap_end])
    mask_for_fit.add_row([mask_line_start, mask_line_end])
    # print(mask_for_fit)

    model_to_gaia_median = []
    best_fit_models = []
    log_shift_brz_all = []
    gaia_flux_interpolated = []
    # Stitch normalized spectra in brz together
    for i in range(len(stds)):
        std_normalized_all_convolved = np.concatenate((normalized_spectra_all_bands[0][i][mask_b_norm],
                                             normalized_spectra_all_bands[1][i][mask_r_norm],
                                             normalized_spectra_all_bands[2][i][mask_z_norm]))
        # lsf_all (initial std lsf) - will be used to convolve good res models for sens curve calculation
        lsf_all = np.concatenate((lsf_all_bands[0][i][mask_b_norm],
                                             lsf_all_bands[1][i][mask_r_norm],
                                             lsf_all_bands[2][i][mask_z_norm]))
        std_norm_unconv = np.concatenate((normalized_spectra_unconv_all_bands[0][i][mask_b_norm], normalized_spectra_unconv_all_bands[1][i][mask_r_norm],
                              normalized_spectra_unconv_all_bands[2][i][mask_z_norm]))
        # TODO: switch to new resampling code
        log_std_wave_all, flux_std_unconv_logscale = linear_to_logscale(std_wave_all, std_norm_unconv)
        log_std_wave_all, flux_std_logscale = linear_to_logscale(std_wave_all, std_normalized_all_convolved)
        std_errors_normalized_all = np.concatenate((std_errors_all_bands[0][i][mask_b_norm],
                                                    std_errors_all_bands[1][i][mask_r_norm],
                                                    std_errors_all_bands[2][i][mask_z_norm]))
        log_std_wave_all, log_std_errors_normalized_all = linear_to_logscale(std_wave_all, std_errors_normalized_all)

        # load Gaia BP-RP spectrum from cache, or download from webapp, and fit the continuum to Gaia spec
        try:
            gw, gf = fluxcal.retrive_gaia_star(gaia_ids[i], GAIA_CACHE_DIR=GAIA_CACHE_DIR)
            stdflux = np.interp(std_wave_all, gw, gf)  # interpolate to our wavelength grid
            gaia_flux_interpolated.append(stdflux)
        except fluxcal.GaiaStarNotFound as e:
            log.warning(e)
            rss_tmp.add_header_comment(f"Gaia star {gaia_id} not found")
            continue
        # best_continuum, continuum_models, masked_pixels, knots = fit_continuum_std(std_wave_all, stdflux,
        #                                                                            mask_bands=mask_bands,
        #                                                                            threshold=0.1, niter=niter,
        #                                                                            nknots=nknots,
        #                                                                            median_box=median_box)
        std_spec = Table(data=[std_wave_all, stdflux], names=['wave', 'flux'])
        best_continuum = smoothSpec_old(std_spec, int(160 / 0.5), method="median")
        normalized_std_on_gaia_cont_single_tmp = best_continuum*std_normalized_all_convolved
        normalized_std_on_gaia_cont_single_tmp[mask_tellurics] = np.nan
        normalized_std_on_gaia_cont_single = normalized_std_on_gaia_cont_single_tmp / np.nansum(normalized_std_on_gaia_cont_single_tmp)

        # mask tellurics, channels overlaps, and bluest part of the spectra in log scale
        mask_good = np.zeros_like(log_std_wave_all, dtype=bool)
        for wave_masks in range(len(mask_for_fit)):
            mask_good = mask_good | ((log_std_wave_all > np.log(mask_for_fit['Start'][wave_masks]))
                                     & (log_std_wave_all < np.log(mask_for_fit['End'][wave_masks])))
        mask_good = ~mask_good & np.isfinite(flux_std_logscale) #~mask_tellurics_log & ~mask_wave

        # canonical f-type model: Teff=6500, logg=4, Fe/H=-1.5 or something like that
        # Check the possible velocity offsets IN LOGSCALE
        # Now we use the model template with Teff=6250, logg=4.0, Fe/H=-0.25
        with fits.open(join(models_dir, 'median_normalized_logscale', template_model), memmap=False) as hdul: #previous -> 'normalized_logscale'
            template = hdul[0].data
        log_model_wave_all = log_std_wave_all
        flux_model_logscale =template

        log_shift_full = fluxcal.derive_vecshift(flux_std_logscale[mask_good],
                                        flux_model_logscale[mask_good], max_ampl=50)*np.median(log_std_wave_all - np.roll(log_std_wave_all, 1))
        vel_shift_full = log_shift_full * 3e5

        flux_std_logscale_shifted = np.interp((log_std_wave_all - log_shift_full), log_std_wave_all, flux_std_logscale)

        chi2 = [np.nansum(((flux_std_logscale_shifted[mask_good] -
                            model_specs_norm[model_ind][mask_good]) / log_std_errors_normalized_all[mask_good]) ** 2) /
                np.sum(mask_good) for model_ind in range(n_models)]
        best_id = np.argmin(chi2)
        # print(f'chi2: {np.argmin(chi2)}')
        # print(f'Model: {model_names[best_id]}')
        model_params = re.split('[a-z]+', model_names[best_id], flags=re.IGNORECASE)
        # print(model_params)

        # TODO: remove the second part of the code that runs per camera

        log.info(f"GAIA id:{gaia_ids[i]}. Best model is: {best_id}, {model_names[best_id]}")
        best_fit_models.append(model_names[best_id])


        # Conversion coefficient model to gaia units
        with fits.open(join(models_dir, 'good_res_new', model_names[best_id])) as hdul:
            model_flux = hdul[0].data
            hdr = hdul[0].header
        n_steps = int((9850 - 3550) / 0.05) + 1
        model_wave = np.linspace(3550, 9850, n_steps)

        mask_model = (model_wave >= min(std_wave_all)) & (model_wave <= max(std_wave_all))
        model_wave = model_wave[mask_model]
        model_flux = model_flux[mask_model]

        # Gaia LSF
        gaia_lsf_path = os.getenv("LVMCORE_DIR") + "/etc/Gaia_BPRP_resolution.txt"
        gaia_lsf_table_tmp = Table.read(gaia_lsf_path, format='ascii',
                                        names=['wavelength', 'resolution'])
        gaia_lsf_table_tmp['wavelength'][len(gaia_lsf_table_tmp['wavelength']) - 1] = gaia_lsf_table_tmp['wavelength'][
                                                                                          len(
                                                                                              gaia_lsf_table_tmp[
                                                                                                  'wavelength']) - 1] * 10
        gaia_lsf_table_tmp['linewidth'] = gaia_lsf_table_tmp['wavelength'] / gaia_lsf_table_tmp['resolution']
        gaia_lsf_table_bp = gaia_lsf_table_tmp[0:10]
        gaia_lsf_table_rp = gaia_lsf_table_tmp[10:17]
        wave_bprp_mean = (max(gaia_lsf_table_bp['wavelength']) + min(gaia_lsf_table_rp['wavelength'])) / 2
        # print(wave_bprp_mean)
        mask_wl_bp = (std_wave_all < wave_bprp_mean)
        mask_wl_rp = (std_wave_all >= wave_bprp_mean)
        gaia_lsf_bp = np.interp(std_wave_all[mask_wl_bp], gaia_lsf_table_bp['wavelength'], gaia_lsf_table_bp['linewidth'])
        gaia_lsf_rp = np.interp(std_wave_all[mask_wl_rp], gaia_lsf_table_rp['wavelength'], gaia_lsf_table_rp['linewidth'])
        gaia_lsf = np.concatenate((gaia_lsf_bp, gaia_lsf_rp))

        # resample model to the same step
        model_flux_resampled = np.interp(std_wave_all, model_wave, model_flux)
        good_model_to_std_lsf = np.sqrt(lsf_all ** 2 - 0.3 ** 2) # to degrade good resolution model to std lsf for plots
        model_convolved_spec_lsf = fluxcal.lsf_convolve(model_flux_resampled, good_model_to_std_lsf, std_wave_all)
        model_tmp = Table(data=[std_wave_all, model_convolved_spec_lsf], names=['wave', 'flux'])
        best_continuum = smoothSpec_old(model_tmp, int(160 / 0.5), method="median")
        model_norm_convolved_spec_lsf = model_convolved_spec_lsf / best_continuum
        # print(model_flux_resampled)
        # print(model_convolved_spec_lsf)
        log_std_wave_all_tmp, log_model_norm_convolved_spec_lsf = linear_to_logscale(std_wave_all, model_norm_convolved_spec_lsf)
        model_shifted_norm_convolved_spec_lsf = logscale_to_linear(std_wave_all, log_std_wave_all_tmp,
                                                                   log_model_norm_convolved_spec_lsf, log_shift_full)

        # convolve model to gaia lsf
        # TODO: make sure we do this once
        model_convolved_to_gaia = fluxcal.lsf_convolve(model_flux_resampled, gaia_lsf, std_wave_all)
        model_to_gaia = stdflux/model_convolved_to_gaia
        model_to_gaia_median.append(np.median(model_to_gaia))

        if plot:

            fig = plt.figure(figsize=(14, 24))

            plt.subplot(511)
            plt.title(label=f'Gaia ID: {gaia_ids[i]}. Model: {model_names[best_id]}',fontsize=14)
            plt.plot(log_std_wave_all, flux_std_unconv_logscale, label=f'Observed standard spectrum from fiber '
                                                                f'{fibers[i]}, continuum normalized')
            sigma1 = flux_std_logscale + log_std_errors_normalized_all
            sigma2 = flux_std_logscale - log_std_errors_normalized_all
            plt.plot(log_std_wave_all, sigma2, '--', color='grey', lw=0.5)
            plt.plot(log_std_wave_all, sigma1, '--', color='grey', lw=0.5)
            plt.fill_between(log_std_wave_all, sigma1, sigma2, alpha=0.2, color='blue')

            # plt.plot(std_wave_all,
            #         np.interp(std_wave_all, std_wave_all*(1+vel_offset_b/3e5), normalized_std_on_gaia_cont_single), label='Shifted')
            plt.plot(log_model_wave_all+log_shift_full, log_model_norm_convolved_spec_lsf, label='Best-fit model spectrum, '
                                                'continuum normalized and convolved with std LSF', alpha=0.7) # shifted
            for n_mask, mask_box in enumerate(mask_for_fit):
                if n_mask == 0:
                    plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey',
                                label='Mask used for model matching')
                else:
                    plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')
            xlim = [8.18, 9.2]
            # xlim = [3600,9800]
            ylim = [0.1,1.6]
            plt.text((xlim[1] - xlim[0]) * 0.05 + xlim[0], (ylim[1] - ylim[0]) * 0.9 + ylim[0], f'Best-fit model: '
                                f'Teff = {model_params[2]}, log(g) = {model_params[3]}, [Fe/H] = {model_params[4]},'
                                f'Vel. correction = {vel_shift_full:.2f} km/s', size=14)
            plt.text((xlim[1] - xlim[0]) * 0.15 + xlim[0], (ylim[1] - ylim[0]) * 0.82 + ylim[0],
                                f'chi2 = {np.argmin(chi2)}', size=14)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel("wavelength [A]", size=14)
            show_wl = np.arange(3500, 10000, 500)
            plt.xticks(np.log(show_wl), labels=show_wl.astype(str), size=14)
            plt.yticks(fontsize=14)
            plt.legend(loc="lower right", fontsize=14)

            plt.subplot(512)
            plt.plot(log_std_wave_all, flux_std_unconv_logscale, label='Observed')
            plt.plot(log_std_wave_all, sigma2, '--', color='grey', lw=0.5)
            plt.plot(log_std_wave_all, sigma1, '--', color='grey', lw=0.5)
            plt.fill_between(log_std_wave_all, sigma1, sigma2, alpha=0.2, color='blue')
            plt.plot(log_model_wave_all+log_shift_full, log_model_norm_convolved_spec_lsf, label='Model shifted', alpha=0.7)
            #plt.plot(log_std_wave_all-log_shift_b, flux_model_logscale, label='Model shifted')
            for mask_box in mask_for_fit:
                plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')
            #plt.legend()
            xlim = [8.24, 8.38]
            show_wl = np.arange(3700, 4400, 100)
            plt.xticks(np.log(show_wl), labels=show_wl.astype(str), size=14)
            plt.yticks(fontsize=14)
            ylim = [0.1,1.6]
            # ylim = [0.9, 1.2]
            plt.text((xlim[1] - xlim[0]) * 0.03 + xlim[0], (ylim[1] - ylim[0]) * 0.9 + ylim[0], 'b channel', size=14)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel("wavelength [A]", size=14)

            plt.subplot(513)
            plt.plot(log_std_wave_all, flux_std_unconv_logscale, label='Observed')
            #plt.plot(log_model_wave_all, flux_model_logscale, label='Model')
            plt.plot(log_std_wave_all, sigma2, '--', color='grey', lw=0.5)
            plt.plot(log_std_wave_all, sigma1, '--', color='grey', lw=0.5)
            plt.fill_between(log_std_wave_all, sigma1, sigma2, alpha=0.2, color='blue')
            plt.plot(log_std_wave_all+log_shift_full, log_model_norm_convolved_spec_lsf, label='Model shifted', alpha=0.7)
            for mask_box in mask_for_fit:
                plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')
            #plt.legend()
            # xlim = [8.66, 8.8]
            # xlim = [8.66, 8.92] #~whole channel
            xlim = [8.69, 8.8]
            ylim = [0.2, 1.5]
            show_wl = np.arange(5700, 6700, 100)
            plt.xticks(np.log(show_wl), labels=show_wl.astype(str), size=14)
            plt.yticks(fontsize=14)
            plt.text((xlim[1] - xlim[0]) * 0.03 + xlim[0], (ylim[1] - ylim[0]) * 0.9 + ylim[0], 'r channel', size=14)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel("wavelength [A]", size=14)

            plt.subplot(514)
            plt.plot(log_std_wave_all, flux_std_unconv_logscale, label='Observed')
            plt.plot(log_std_wave_all, sigma2, '--', color='grey', lw=0.5)
            plt.plot(log_std_wave_all, sigma1, '--', color='grey', lw=0.5)
            plt.fill_between(log_std_wave_all, sigma1, sigma2, alpha=0.2, color='blue')
            #plt.plot(log_model_wave_all, flux_model_logscale, label='Model')
            plt.plot(log_std_wave_all+log_shift_full, log_model_norm_convolved_spec_lsf, label='Model shifted', alpha=0.7)
            for mask_box in mask_for_fit:
                plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')
            #plt.legend()
            # xlim = [9.02, 9.16]
            xlim = [9.035, 9.1]
            ylim = [0.2, 1.5]
            show_wl = np.arange(8300, 9500, 100)
            plt.xticks(np.log(show_wl), labels=show_wl.astype(str), size=14)
            plt.yticks(fontsize=14)
            plt.text((xlim[1] - xlim[0]) * 0.03 + xlim[0], (ylim[1] - ylim[0]) * 0.9 + ylim[0], 'z channel', size=14)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel("wavelength [A]", size=14)
            # plt.ylabel(size=14)

            plt.subplot(515)
            # plt.plot(std_wave_all, normalized_std_on_gaia_cont_single_tmp, linewidth=1.5,
            #          label='Continuum from GAIA spectrum * observed absorptions')
            # plt.plot(std_wave_all, model_convolved_spec_lsf * np.median(model_to_gaia), label='Best-fit model',
            #          linewidth=1.5, alpha=0.7)
            # plt.plot(std_wave_all, std_norm_unconv)
            # plt.plot(std_wave_all, model_shifted_norm_convolved_spec_lsf)
            plt.plot(std_wave_all, std_norm_unconv/model_shifted_norm_convolved_spec_lsf, label='Observed normalised/model normalised')
            # plt.plot(log_std_wave_all+log_shift_full, log_model_norm_convolved_spec_lsf, label='Model shifted', alpha=0.7)
            #plt.plot(log_std_wave_all+log_shift_z, flux_model_logscale, label='Model shifted')
            for mask_box in mask_for_fit:
               plt.axvspan((mask_box[0]), (mask_box[1]), alpha=0.2, color='grey')
            plt.legend(fontsize=14)
            plt.xlim(3800,5000)
            #plt.gca().set_ylim(bottom=0)
            plt.ylim(0.5, 1.6)
            plt.xlabel("wavelength [A]", size=14)
            # plt.ylabel("Flux, erg/s/cm^2/A", size=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            #plt.show()
            fig_path = in_rss[0]
            fig_path = f"{fig_path.replace('lvm-hobject-b', 'lvm-hobject')}"
            save_fig(plt.gcf(), product_path=fig_path, to_display=False, figure_path="qa/model_matching", label=f"matching_std{i}")

        # calculating sensitivity curves
    for n_chan, chan in enumerate('brz'):
        # load input RSS
        log.info(f"loading input RSS file '{os.path.basename(in_rss[n_chan])}'")
        rss = RSS.from_file(in_rss[n_chan])

        # define dummy sensitivity array in (ergs/s/cm^2/A) / (e-/s/A) for standard star fibers
        colnames = [f"{std_fib[:-3]}SEN" for std_fib in rss._header["STD*FIB"]]
        if len(colnames) == 0:
            NSTD = 15
            colnames = [f"STD{i}SEN" for i in range(1, NSTD + 1)]
        res_mod = Table(np.full(w[n_chan].size, np.nan, dtype=list(zip(colnames, ["f8"] * len(colnames)))))
        # mean_mod, rms_mod = np.full(w.size, np.nan), np.full(w.size, np.nan)

        if plot:
            plt.subplot
            fig1 = plt.figure(1)
            frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
            frame1.set_xticklabels([])

        for i in range(len(stds)):
            sens_tmp = calc_sensitivity_from_model(w[n_chan], std_spectra_all_bands[n_chan][i], lsf_all_bands[n_chan][i],
                                                   model_names[best_id], model_to_gaia_median[i], log_shift_full)
            wgood, sgood = fluxcal.filter_channel(w[n_chan], sens_tmp, 3, method='savgol')
            if chan == 'b':
                win = 150
                ylim = [0, 0.3e-11]
            elif chan == 'r':
                win = 70
                ylim = [0, 0.5e-12]
            else:
                win = 15
                ylim = [0, 0.5e-12]
            s = interpolate.make_smoothing_spline(wgood, sgood, lam=win)
            sens0 = s(w[n_chan]).astype(np.float32)

            # calculate the normalization of the average (known) sensitivity curve in a broad band
            lvmflux = fluxcal.spec_to_LVM_flux(chan, w[n_chan], std_spectra_all_bands[n_chan][i]*sens0)
            gaia_flux = fluxcal.spec_to_LVM_flux(chan, std_wave_all, gaia_flux_interpolated[i])
            sens_coef = gaia_flux/lvmflux
            #print(f'lvmflux={lvmflux}, gaia_flux={gaia_flux}, converted to gaia flux = {lvmflux*sens_coef}')


            res_mod[f"STD{i}SEN"] = s(w[n_chan]).astype(np.float32)*sens_coef
            sens = sens0*sens_coef

            fig_path = in_rss[n_chan]
            if plot:
                plt.plot(wgood, sgood*sens_coef, ".k", markersize=2, zorder=-999)
                plt.plot(w[n_chan], sens, linewidth=1, zorder=-999)


        res_mod_pd = res_mod.to_pandas().values
        rms_mod = biweight_scale(res_mod_pd, axis=1, ignore_nan=True)
        mean_mod = biweight_location(res_mod_pd, axis=1, ignore_nan=True)

        label = rss._header['CCD']
        rss.setHdrValue(f"MODSENM{label}", np.nanmean(mean_mod[1000:3000]), f"Mean model sensitivity in {chan}")
        rss.setHdrValue(f"MODSENR{label}", np.nanmean(rms_mod[1000:3000]), f"Mean model sensitivity rms in {chan}")
        log.info(f"Mean model sensitivity in {chan} : {np.nanmean(mean_mod[1000:3000])}")

        print(f"product_path = {in_rss[n_chan]}")
        if plot:
            plt.ylabel("sensitivity [(ergs/s/cm^2/A) / (e-/s/A)]")
            plt.xlabel("wavelength [A]")
            plt.ylim(1e-14, 0.1e-11)
            plt.semilogy()
            fig1.add_axes((0.1, 0.1, 0.8, 0.2))
            plt.plot([w[n_chan][0], w[n_chan][-1]], [0.05, 0.05], color="k", linewidth=1, linestyle="dotted")
            plt.plot([w[n_chan][0], w[n_chan][-1]], [-0.05, -0.05], color="k", linewidth=1, linestyle="dotted")
            plt.plot([w[n_chan][0], w[n_chan][-1]], [0.1, 0.1], color="k", linewidth=1, linestyle="dashed")
            plt.plot([w[n_chan][0], w[n_chan][-1]], [-0.1, -0.1], color="k", linewidth=1, linestyle="dashed")
            plt.plot(w[n_chan], rms_mod / mean_mod)
            plt.plot(w[n_chan], -rms_mod / mean_mod)
            plt.ylim(-0.2, 0.2)
            plt.ylabel("relative residuals")
            plt.xlabel("wavelength [A]")
            save_fig(plt.gcf(), product_path=in_rss[n_chan], to_display=False, figure_path="qa", label="fluxcal_mod")

        # update sensitivity extension
        log.info('appending FLUXCAL_MOD table')
        rss.set_fluxcal(fluxcal=res_mod, source='mod')
        rss.writeFitsData(in_rss[n_chan])

    return best_fit_models, model_to_gaia_median


def fit_continuum_std(spectrum_wave, spectrum_flux, mask_bands=([4830,4900],), niter=3, threshold=0.5, nknots=100, median_box=10, **kwargs):
    """Modified version of fit_continuum function
    Fit a continuum to a spectrum using a spline interpolation

    Given a spectrum, this function fits a continuum using a spline
    interpolation and iteratively masks outliers below a given threshold of the
    fitted spline.

    Parameters
    ----------
    spectrum_wave : wavelength array
    spectrum_flux : flux array
    mask_bands : list
        List of wavelength bands to mask
    median_box : int
        Size of the median filter box
    niter : int
        Number of iterations to fit the continuum
    threshold : float or tuple of floats
        Threshold to mask outliers, if tuple, the first element is the lower
        threshold and the second element is the upper threshold

    Returns
    -------
    best_continuum : np.ndarray
        Best fit continuum
    continuum_models : list
        List of continuum models for each iteration
    masked_pixels : np.ndarray
        Masked pixels in all iterations
    knots : np.ndarray
        Spline knots
    """

    # define main arrays
    wave = spectrum_wave.copy()
    data = spectrum_flux.copy()

    # define spline fitting parameters
    nknots = kwargs.pop("nknots", nknots)
    knots = np.linspace(wave[wave.size // nknots], wave[-1 * wave.size // nknots], nknots)
    if mask_bands:
        mask = np.ones_like(knots, dtype="bool")
        for iwave, fwave in mask_bands:
            mask[(iwave <= knots) & (knots <= fwave)] = False
        knots = knots[mask]
    kwargs.update([("t", knots)])
    kwargs.update([("task", -1)])

    spectrum_flux = median_filter(spectrum_flux, size=median_box)
    mask = np.isnan(spectrum_flux)

    # fit first spline
    f = interpolate.splrep(wave, data, **kwargs)
    spline = interpolate.splev(spectrum_wave, f)

    # iterate to mask outliers and update spline
    continuum_models = []
    if threshold is not None and isinstance(threshold, (float, int)):
        threshold = (threshold, np.inf)
    masked_pixels = mask
    for i in range(niter):
        residuals = spline - spectrum_flux#spectrum._data
        mask = spline - threshold[0] * np.nanstd(residuals) > spectrum_flux#spectrum._data
        mask |= spline + threshold[1] * np.nanstd(residuals) < spectrum_flux#spectrum._data

        # add new outliers to mask
        masked_pixels |= mask

        # update spline
        f = interpolate.splrep(spectrum_wave[~masked_pixels], spectrum_flux[~masked_pixels], **kwargs)
        new_spline = interpolate.splev(spectrum_wave, f)
        continuum_models.append(new_spline)
        if np.mean(np.abs(new_spline - spline) / spline) <= 0.01:
            break
        else:
            spline = new_spline

    best_continuum = continuum_models.pop()
    return best_continuum, continuum_models, masked_pixels, knots


def calc_sensitivity_from_model(wl, obs_spec, spec_lsf, best_model='', model_to_gaia_median=1, model_log_shift=0):
    """
    Calculate the sensitivity curves using the model spectra
    First convert model spectra to log scale, apply the "velocity shift" found in the model_selection function in log
    spece, and convert back to linear space. After that the sensitivity curve is calculated.
    :param wl: wavelength array for the observed spectrum
    :param obs_spec: observed standard spectum (sky-subtracted and corrected for extinction)
    :param spec_lsf: spectrograph lsf
    :param best_model: model to use in calculation
    :param model_to_gaia_median: coefficient from model to GAIA units
    :param model_log_shift: shift of the model spectra relative to observed in log space
    :return: sensitivity curve
    """

    # read the best-fit model and convolve with spectrograph LSF
    # model_dir = '/Users/amejia/Downloads/stellar_models/'
    models_dir = os.path.join(MASTERS_DIR, "stellar_models")

    with fits.open(join(models_dir, 'good_res_new', best_model)) as hdul:
        model_flux = hdul[0].data
        hdr = hdul[0].header
    #model_flux = best_fit_model['flux']
    n_steps = int((9850-3550) / 0.05) + 1
    model_wave = np.linspace(3550, 9850, n_steps)

    # apply the model shift relative to observed spectra in log space
    log_model_wave, flux_model_logscale = linear_to_logscale(model_wave, model_flux)
    flux_model_shifted = logscale_to_linear(model_wave, log_model_wave, flux_model_logscale, shift=model_log_shift)

    # #resample model to the same step
    model_flux_resampled = np.interp(wl, model_wave, flux_model_shifted)
    spec_lsf = np.sqrt(spec_lsf**2 - 0.3**2)  # as model spectra were already convolved with lsf=0.3, we need to account for this

    # # convolve model to spec lsf
    # TODO: make sure we do this once
    model_convolved_spec_lsf = fluxcal.lsf_convolve(model_flux_resampled, spec_lsf, wl)
    sens = model_convolved_spec_lsf * model_to_gaia_median / obs_spec

    return sens


def standard_sensitivity(stds, rss, GAIA_CACHE_DIR, ext, res, plot=False, width=3):
    # load the sky masks
    channel = rss._header['CCD']
    w = rss._wave

    m = get_sky_mask_uves(w, width=width)
    m2 = None
    if channel == "z":
        m2 = get_z_continuum_mask(w)

    master_sky = rss.eval_master_sky()

    # iterate over standard stars, derive sensitivity curve for each
    for i, s in enumerate(stds):
        nn, fiber, gaia_id, exptime, secz = s  # unpack standard star tuple

        # find the fiber with our spectrum of that Gaia star, if it is not in the current spectrograph, continue
        select = rss._slitmap["orig_ifulabel"] == fiber
        fibidx = np.where(select)[0]

        log.info(f"standard fiber '{fiber}', index '{fibidx}', star '{gaia_id}', exptime '{exptime:.2f}', secz '{secz:.2f}'")

        # load Gaia BP-RP spectrum from cache, or download from webapp
        try:
            gw, gf = fluxcal.retrive_gaia_star(gaia_id, GAIA_CACHE_DIR=GAIA_CACHE_DIR)
            stdflux = np.interp(w, gw, gf)  # interpolate to our wavelength grid
        except fluxcal.GaiaStarNotFound as e:
            log.warning(e)
            rss.add_header_comment(f"Gaia star {gaia_id} not found")
            continue

        # subtract sky spectrum and divide by exptime
        spec = rss._data[fibidx[0], :]
        lsf = rss._lsf[fibidx[0], :]
        if np.nanmean(spec) < 100:
            log.warning(f"fiber {fiber} @ {fibidx[0]} has counts < 100 e-, skipping")
            rss.add_header_comment(f"fiber {fiber} @ {fibidx[0]} has counts < 100 e-, skipping")
            continue

        spec = (rss._data[fibidx[0],:] - master_sky._data[fibidx[0],:])/exptime  #TODO: check exptime?

        # interpolate over bright sky lines
        spec = fluxcal.interpolate_mask(w, spec, m, fill_value="extrapolate")
        if channel == "z":
            spec = fluxcal.interpolate_mask(w, spec, ~m2, fill_value="extrapolate")

        # correct for extinction
        spec *= 10 ** (0.4 * ext * secz)

        # TODO: fit continuum to instrumental std spectrum (stdflux) and normalize
        # TODO: mask telluric absorption lines from stdflux
        # TODO: match gaia spectrum and stdflux against a set of theoretical stellar templates
        # TODO: downgrade best fit template to instrumental LSF and calculate sensitivity curve (after lifting telluric mask)

        # divide to find sensitivity and smooth
        # Here we can choose if we want to use Gaia or model spectra to get the sensitivity curves
        # if mode == "GAIA":
        # sens = stdflux / spec
        # else:
        #     sens = calc_sensitivity_from_model(w, spec, spec_lsf=lsf, best_model=model_list[i],
        #                                        model_to_gaia_median=model_coef[i], model_log_shift = model_log_shifts[i])
        # if mode == "GAIA":
        # wgood, sgood = fluxcal.filter_channel(w, sens, 2)
        # else:
        #     if channel == 'b':
        #         wgood, sgood = fluxcal.filter_channel(w, sens, 3, method='savgol')
        #     elif channel == 'r':
        #         wgood, sgood = fluxcal.filter_channel(w, sens, 3, method='savgol')
        #     else:
        #         wgood = w[np.isfinite(sens)]
        #         sgood = sens[np.isfinite(sens)]

        # sens_gaia = stdflux / spec
        # wgood_gaia, sgood_gaia = fluxcal.filter_channel(w, sens_gaia, 2)
        #
        # # if mode == "GAIA":
        # s = interpolate.make_smoothing_spline(wgood_gaia, sgood_gaia, lam=1e4)
        # else:
        #     if channel == 'b':
        #         win = 150
        #     elif channel == 'r':
        #         win = 70
        #     else:
        #         win = 15
        #     s = interpolate.make_smoothing_spline(wgood, sgood, lam=win)
        # s_gaia = interpolate.make_smoothing_spline(wgood_gaia, sgood_gaia, lam=win)

        # divide to find sensitivity and smooth
        sens = stdflux / spec
        wgood, sgood = fluxcal.filter_channel(w, sens, 2)
        s = interpolate.make_smoothing_spline(wgood, sgood, lam=1e4)
        res[f"STD{nn}SEN"] = s(w).astype(np.float32)

        # caluculate SDSS g band magnitudes for QC
        mAB_std = np.round(fluxcal.spec_to_LVM_mAB(channel, w, stdflux), 2)
        mAB_obs = np.round(fluxcal.spec_to_LVM_mAB(channel, w[np.isfinite(spec)], spec[np.isfinite(spec)]), 2)
        # update input file header
        label = channel.upper()
        rss.setHdrValue(f"STD{nn}{label}AB", mAB_std, f"Gaia AB mag in {channel}-band")
        rss.setHdrValue(f"STD{nn}{label}IN", mAB_obs, f"Obs AB mag in {channel}-band")
        log.info(f"AB mag in LVM_{channel}: Gaia {mAB_std:.2f}, instrumental {mAB_obs:.2f}")

        # if plot:
        #     # fig = plt.figure(figsize=(16, 6))
        #     # plt.plot(wgood, sgood, ".k", markersize=2, zorder=-999)
        #     plt.plot(w, sens, ".k", markersize=2, zorder=-999)
        #     plt.plot(w, res[f"STD{nn}SEN"], label='sens. curve (after shift correction)')
        #     #plt.plot(w, s_gaia(w).astype(np.float32), linewidth=2, color='red', label='old sensitivity curve')
        #     # plt.ylim(0, 0.1e-11)
        #     # plt.legend()
        #     # plt.show()
        if plot:
            plt.plot(wgood, sgood, ".k", markersize=2, zorder=-999)
            plt.plot(w, res[f"STD{nn}SEN"], linewidth=1)

    return rss, res


def science_sensitivity(rss, res_sci, ext, GAIA_CACHE_DIR, NSCI_MAX=15, r_spaxel=(32.0/2)/3600.0, width=3, plot=False):
    '''
    Scale the (assumed known) average sensitivity function of LVMi using GAIA XP spectra of
    bright stars found in the science IFU. Scaling is based on a "broad band" Gaussian filter
    in b,r,i channels.

    First query for any gaia stars in the science IFU, then use the nearest fiber
    spectrum to measure an average flux offset between the data and the XP spectra.
    '''

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # get data from lvmFrame
    fluxes = rss._data
    obswave = rss._wave
    header = rss._header
    slitmap = rss._slitmap
    cfib = slitmap[slitmap['fiberid']==975]  # central fiber (in spectrograph 2, xpmm=ypmm=0)
    ra = cfib['ra'].value[0]
    dec = cfib['dec'].value[0]
    # ra = header['POSCIRA']
    # dec = header['POSCIDE']
    expnum = header['EXPOSURE']
    exptime = header['EXPTIME']
    channel = header['CCD']
    secz = header['TESCIAM']

    m = get_sky_mask_uves(obswave, width=width)
    m2 = None
    if channel == "z":
        m2 = get_z_continuum_mask(obswave)

    # get GAIA data, potentially cached
    r, calibrated_spectra, sampling = fluxcal.get_XP_spectra(expnum, ra, dec, plot=False, lim_mag=13.5,
                                                             n_spec=NSCI_MAX, GAIA_CACHE_DIR=GAIA_CACHE_DIR)
    gwave = sampling*10 # to A
    for i in range(len(calibrated_spectra)):
        # W/micron/m^2 -> in erg/s/cm^2/A
        calibrated_spectra.iloc[i].flux *= 100

    # read the mean sensitivity curve
    mean_sens = fluxcal.get_mean_sens_curves(os.getenv("LVMCORE_DIR") + "/sensitivity")

    # we will need the science fibers and the sky fibers
    scifibs = slitmap[slitmap['targettype'] == 'science']
    #skyfibs = slitmap[slitmap['targettype'] == 'SKY']

    master_sky = rss.eval_master_sky()

    # locate the science ifu fibers the stars are in
    for i in range(len(calibrated_spectra)):
        data = r[i]
        d = np.sqrt((data['ra']-scifibs['ra'])**2 + (data['dec']-scifibs['dec'])**2) # in degrees
        fib = np.where(d<r_spaxel)[0] # there can only be zero or one fiber with a distance cut smaller than a fiber diameter
        if fib.size > 0:
            # if we found a star in a fiber
            gflux = calibrated_spectra.iloc[i].flux

            fibidx = scifibs['fiberid'][fib] - 1

            # skip star if the fiber is dead
            if rss._mask[fibidx[0]].all():
                continue

            dmin = d[fib] * 3600 # convert to arcsec

            log.info(f"science fiberid '{scifibs['fiberid'][fib][0]}', star '{data['SOURCE_ID']}', secz '{secz:.2f}'")

            # correction for Evelyn's effect
            radius_fac = np.interp(dmin, np.array([0,4,6,8,10,12,14,16]), 10**(-0.4*np.array([0.0,0.0,0.05,0.12,0.15,0.2,0.2,0.2])))

            # subtract sky, correct for radius effect
            obsflux = (fluxes[fibidx[0],:] - master_sky._data[fibidx[0],:])/radius_fac # observed flux of star
            obsflux = fluxcal.interpolate_mask(obswave, obsflux, ~np.isfinite(obsflux))
            # interpolate over bright sky lines
            obsflux = fluxcal.interpolate_mask(obswave, obsflux, m, fill_value="extrapolate")
            if channel == "z":
                obsflux = fluxcal.interpolate_mask(obswave, obsflux, ~m2, fill_value="extrapolate")

            # correct for extinction
            obsflux *= 10 ** (0.4 * ext * secz)
            obsflux /= exptime

            # calculate the normalization of the average (known) sensitivity curve in a broad band
            lvmflux = fluxcal.spec_to_LVM_flux(channel, obswave, obsflux)
            sens = fluxcal.spec_to_LVM_flux(channel, gwave, gflux) / lvmflux
            res_sci[f"STD{i+1}SEN"] = (sens*np.interp(obswave, mean_sens[channel]['wavelength'],
                                                               mean_sens[channel]['sens'])).astype(np.float32)

            mAB_std = np.round(fluxcal.spec_to_LVM_mAB(channel, gwave, gflux), 2)
            mAB_obs = np.round(fluxcal.spec_to_LVM_mAB(channel, obswave, obsflux), 2)
            # update input file header
            cam = channel.upper()
            rss.setHdrValue(f"SCI{i+1}{cam}AB", mAB_std, f"Gaia AB mag in {channel}-band")
            rss.setHdrValue(f"SCI{i+1}{cam}IN", mAB_obs, f"Obs AB mag in {channel}-band")
            rss.setHdrValue(f"SCI{i+1}ID", data['SOURCE_ID'], f"Field star {i+1} Gaia source ID")
            rss.setHdrValue(f"SCI{i+1}FIB", scifibs['fiberid'][fib][0], f"Field star {i+1} fiber id")
            rss.setHdrValue(f"SCI{i+1}RA", data['ra'], f"Field star {i+1} RA")
            rss.setHdrValue(f"SCI{i+1}DE", data['dec'], f"Field star {i+1} DEC")
            log.info(f"AB mag in LVM_{channel}: Gaia {mAB_std:.2f}, instrumental {mAB_obs:.2f}")

            # calibrate and plot against the stars for debugging:
            if plot:
                plt.plot(obswave, np.interp(obswave, gwave, gflux)/obsflux, '.',
                         color=colors[i%len(colors)] , markersize=2, zorder=-999)
                plt.plot(obswave, res_sci[f"STD{i+1}SEN"], color=colors[i%len(colors)], linewidth=2)

    return rss, res_sci


def fluxcal_standard_stars(in_rss, plot=True, GAIA_CACHE_DIR=None, mode='GAIA', model_list=[], model_coef=[], model_log_shifts=[]):
    """
    Create sensitivity functions for LVM data using the 12 spectra of stars observed through
    the Spec telescope.

    Uses Gaia BP-RP spectra for calibration. To be replaced or extended by using fitted stellar
    atmmospheres.

    mode = 'GAIA' (old behavior) or 'model' (uses stellar atmosphere models)
    """
    GAIA_CACHE_DIR = "./" if GAIA_CACHE_DIR is None else GAIA_CACHE_DIR
    log.info(f"Using Gaia CACHE DIR '{GAIA_CACHE_DIR}'")

    # load input RSS
    log.info(f"loading input RSS file '{os.path.basename(in_rss)}'")
    rss = RSS.from_file(in_rss)

    # wavelength array
    w = rss._wave

    # define dummy sensitivity array in (ergs/s/cm^2/A) / (e-/s/A) for standard star fibers
    colnames = [f"{std_fib[:-3]}SEN" for std_fib in rss._header["STD*FIB"]]
    if len(colnames) == 0:
        NSTD = 15
        colnames = [f"STD{i}SEN" for i in range(1, NSTD + 1)]
    res_std = Table(np.full(w.size, np.nan, dtype=list(zip(colnames, ["f8"] * len(colnames)))))
    mean_std, rms_std = np.full(w.size, np.nan), np.full(w.size, np.nan)

    # load extinction curve
    # Note that we assume a constant extinction curve here!
    txt = np.genfromtxt(os.getenv("LVMCORE_DIR") + "/etc/lco_extinction.txt")
    lext, ext = txt[:, 0], txt[:, 1]
    ext = np.interp(w, lext, ext)

    # get the list of standards from the header
    try:
        stds = fluxcal.retrieve_header_stars(rss=rss)
    except KeyError:
        log.warning(f"no standard star metadata found in '{in_rss}', skipping sensitivity measurement")
        rss.add_header_comment(f"no standard star metadata found in '{in_rss}', skipping sensitivity measurement")
        rss.set_fluxcal(fluxcal=res_std, source='std')
        rss.writeFitsData(in_rss)
        return res_std, mean_std, rms_std, rss

    # early stop if not standards exposed in current spectrograph
    if len(stds) == 0:
        log.warning(f"no standard stars found in '{in_rss}', skipping sensitivity measurement")
        rss.add_header_comment(f"no standard stars found in '{in_rss}', skipping sensitivity measurement")
        rss.set_fluxcal(fluxcal=res_std, source='std')
        rss.writeFitsData(in_rss)
        return res_std, mean_std, rms_std, rss

    if plot:
        plt.subplot
        fig1 = plt.figure(1)
        frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
        frame1.set_xticklabels([])

    # standard fibers sensitivity curves
    rss, res_std = standard_sensitivity(stds, rss, GAIA_CACHE_DIR, ext, res_std, plot=plot)
    res_std_pd = res_std.to_pandas().values
    ngood_std = res_std_pd.shape[1]-2-np.isnan(res_std_pd.sum(axis=0)).sum()
    if ngood_std < 8:
        log.warning("less than 8 good standard fibers, skipping standard calibration")
        rss.add_header_comment("less than 8 good standard fibers, skipping standard calibration")
        res_std[:] = np.nan
        rss.set_fluxcal(fluxcal=res_std, source='std')
        rss.writeFitsData(in_rss)
        return res_std, mean_std, rms_std, rss

    rms_std = biweight_scale(res_std_pd, axis=1, ignore_nan=True)
    mean_std = biweight_location(res_std_pd, axis=1, ignore_nan=True)

    label = rss._header['CCD']
    channel = label.lower()
    rss.setHdrValue(f"STDSENM{label}", np.nanmean(mean_std[1000:3000]), f"Mean stdstar sensitivity in {channel}")
    rss.setHdrValue(f"STDSENR{label}", np.nanmean(rms_std[1000:3000]), f"Mean stdstar sensitivity rms in {channel}")
    log.info(f"Mean stdstar sensitivity in {channel} : {np.nanmean(mean_std[1000:3000])}")

    print(f"product_path = {in_rss}")
    if plot:
        plt.ylabel("sensitivity [(ergs/s/cm^2/A) / (e-/s/A)]")
        plt.xlabel("wavelength [A]")
        plt.ylim(1e-14, 0.1e-11)
        plt.semilogy()
        fig1.add_axes((0.1, 0.1, 0.8, 0.2))
        plt.plot([w[0], w[-1]], [0.05, 0.05], color="k", linewidth=1, linestyle="dotted")
        plt.plot([w[0], w[-1]], [-0.05, -0.05], color="k", linewidth=1, linestyle="dotted")
        plt.plot([w[0], w[-1]], [0.1, 0.1], color="k", linewidth=1, linestyle="dashed")
        plt.plot([w[0], w[-1]], [-0.1, -0.1], color="k", linewidth=1, linestyle="dashed")
        plt.plot(w, rms_std / mean_std)
        plt.plot(w, -rms_std / mean_std)
        plt.ylim(-0.2, 0.2)
        plt.ylabel("relative residuals")
        plt.xlabel("wavelength [A]")
        save_fig(plt.gcf(), product_path=in_rss, to_display=False, figure_path="qa", label="fluxcal_std")

    # update sensitivity extension
    log.info('appending FLUXCAL_STD table')
    rss.set_fluxcal(fluxcal=res_std, source='std')
    rss.writeFitsData(in_rss)

    return res_std, mean_std, rms_std, rss


def fluxcal_sci_ifu_stars(in_rss, plot=True, GAIA_CACHE_DIR=None, NSCI_MAX=15):
    """
    Scale the (assumed known) average sensitivity function using XP spectra of stars
    found in the science IFU. Create a calibration table analogue to the one returned
    by `fluxcal_standards`.

    Uses Gaia BP-RP spectra for calibration. To be replaced or extended by using fitted stellar
    atmmospheres.
    """
    GAIA_CACHE_DIR = "./" if GAIA_CACHE_DIR is None else GAIA_CACHE_DIR
    log.info(f"Using Gaia CACHE DIR '{GAIA_CACHE_DIR}'")

    # load input RSS
    log.info(f"loading input RSS file '{os.path.basename(in_rss)}'")
    rss = RSS.from_file(in_rss)

    # wavelength array
    w = rss._wave

    # define dummy sensitivity array in (ergs/s/cm^2/A) / (e-/s/A) for standard star fibers
    colnames = [f"STD{i}SEN" for i in range(1, NSCI_MAX + 1)]
    res_sci = Table(np.full(w.size, np.nan, dtype=list(zip(colnames, ["f8"] * len(colnames)))))
    mean_sci, rms_sci = np.full(w.size, np.nan), np.full(w.size, np.nan)

    # load extinction curve
    # Note that we assume a constant extinction curve here!
    txt = np.genfromtxt(os.getenv("LVMCORE_DIR") + "/etc/lco_extinction.txt")
    lext, ext = txt[:, 0], txt[:, 1]
    ext = np.interp(w, lext, ext)

    if plot:
        plt.subplot
        fig1 = plt.figure(1)
        frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
        frame1.set_xticklabels([])

    # science fibers with Gaia stars sensitivity curves
    rss, res_sci = science_sensitivity(rss, res_sci, ext, GAIA_CACHE_DIR, NSCI_MAX=NSCI_MAX, plot=plot)
    rms_sci = biweight_scale(res_sci.to_pandas().values, axis=1, ignore_nan=True)
    mean_sci = biweight_location(res_sci.to_pandas().values, axis=1, ignore_nan=True)

    label = rss._header['CCD']
    channel = label.lower()
    rss.setHdrValue(f"SCISENM{label}", np.nanmean(mean_sci[1000:3000]), f"Mean scistar sensitivity in {channel}")
    rss.setHdrValue(f"SCISENR{label}", np.nanmean(rms_sci[1000:3000]), f"Mean scistar sensitivity rms in {channel}")
    log.info(f"Mean scistar sensitivity in {channel} : {np.nanmean(mean_sci[1000:3000])}")

    if plot:
        plt.ylabel("sensitivity [(ergs/s/cm^2/A) / (e-/s/A)]")
        plt.xlabel("wavelength [A]")
        plt.ylim(1e-14, 0.1e-11)
        plt.semilogy()
        fig1.add_axes((0.1, 0.1, 0.8, 0.2))
        plt.plot([w[0], w[-1]], [0.05, 0.05], color="k", linewidth=1, linestyle="dotted")
        plt.plot([w[0], w[-1]], [-0.05, -0.05], color="k", linewidth=1, linestyle="dotted")
        plt.plot([w[0], w[-1]], [0.1, 0.1], color="k", linewidth=1, linestyle="dashed")
        plt.plot([w[0], w[-1]], [-0.1, -0.1], color="k", linewidth=1, linestyle="dashed")
        plt.plot(w, rms_sci / mean_sci)
        plt.plot(w, -rms_sci / mean_sci)
        plt.ylim(-0.2, 0.2)
        plt.ylabel("relative residuals")
        plt.xlabel("wavelength [A]")
        save_fig(plt.gcf(), product_path=in_rss, to_display=False, figure_path="qa", label="fluxcal_sciifu")

    # update sensitivity extension
    log.info('appending FLUXCAL_SCI table')
    rss.set_fluxcal(fluxcal=res_sci, source='sci')
    rss.writeFitsData(in_rss)

    return res_sci, mean_sci, rms_sci, rss


def createSensFunction_drp(
    in_rss,
    out_throughput,
    ref_spec,
    airmass,
    exptime,
    smooth_poly="5",
    smooth_ref="6.0",
    smooth_ref2="6.0",
    median_filt="0",
    coadd="1",
    extinct_v="0.0",
    extinct_curve="mean",
    aper_correct="1.0",
    ref_units="1e-16",
    target_units="1e-16",
    column_wave="0",
    column_flux="1",
    delimiter="",
    header="1",
    split="",
    mask_wave="",
    mask_telluric="",
    overlap="100",
    out_star="",
    verbose="0",
):
    smooth_poly = int(smooth_poly)
    smooth_ref = float(smooth_ref)
    smooth_ref2 = float(smooth_ref2)
    median_filt = int(median_filt)
    coadd = int(coadd)
    ref_units = float(ref_units)
    target_units = float(target_units)
    aper_correct = float(aper_correct)
    column_wave = int(column_wave)
    column_flux = int(column_flux)
    header = int(header)
    if mask_wave != "":
        mask_wave = np.array(mask_wave.split(",")).astype("float32")
    else:
        mask_wave = None

    if mask_telluric != "":
        mask_telluric = np.array(mask_telluric.split(",")).astype("float32")
    else:
        mask_telluric = None
    verbose = int(verbose)

    ref_star_spec = Spectrum1D()

    if coadd > 0:
        rss = RSS()
        rss.loadFitsData(in_rss)
        select = rss.selectSpec(min=0, max=coadd, method="median")
        star_rss = rss.subRSS(select)
        star_spec = star_rss.create1DSpec(method="sum") / aper_correct
    else:
        star_spec = Spectrum1D()
        if ".fits" in in_rss:
            star_spec.loadFitsData(in_rss)
        elif ".txt" in in_rss:
            star_spec.loadTxtData(in_rss)

    try:
        extinct_v = rss.getHdrValue(extinct_v)
    except KeyError:
        extinct_v = float(extinct_v)

    try:
        airmass = rss.getHdrValue(airmass)
    except KeyError:
        airmass = float(airmass)

    try:
        exptime = rss.getHdrValue(exptime)
    except KeyError:
        exptime = float(exptime)

    if (
        extinct_curve == "mean"
        or extinct_curve == "summer"
        or extinct_curve == "winter"
    ):
        extinct = 10 ** (
            fluxcal.extinctCAHA(star_spec._wave, extinct_v, type=extinct_curve)
            * airmass
            * -0.4
        )
    elif extinct_curve == "Paranal":
        extinct = 10 ** (
            fluxcal.extinctParanal(star_spec._wave) * airmass * -0.4
        )
    else:
        extinct = Spectrum1D()
        extinct.loadTxtData(extinct_curve)
        extinct = 10 ** (extinct * airmass * -0.4)
        extinct = extinct.resampleSpec(star_spec._wave)
    ref_star_spec.loadSTDref(
        ref_spec,
        column_wave=column_wave,
        column_flux=column_flux,
        delimiter=delimiter,
        header=header,
    )
    ref_star_resamp = ref_star_spec.resampleSpec(star_spec._wave, method="linear")

    ref_star_resamp.smoothSpec(
        smooth_ref / 2.354 / (star_spec._wave[1] - star_spec._wave[0])
    )
    if out_star != "":
        star_out = open(out_star, "w")
        for i in range(star_spec._dim):
            star_out.write("%i %.3f %e\n" % (i, star_spec._wave[i], star_spec._data[i]))
        star_out.close()

    star_spec.smoothSpec(smooth_ref)
    # print(exptime,extinct._wave,star_spec._wave)
    star_corr = star_spec / extinct / exptime

    throughput = ref_star_resamp / star_corr
    if mask_wave is not None:
        regions = len(mask_wave) / 2
        for i in range(regions):
            select_region = np.logical_and(
                throughput._wave > mask_wave[i * 2],
                throughput._wave < mask_wave[i * 2 + 1],
            )
            select_blue = np.logical_and(
                throughput._wave > mask_wave[i * 2] - 20,
                throughput._wave < mask_wave[i * 2],
            )
            select_red = np.logical_and(
                throughput._wave > mask_wave[i * 2 + 1],
                throughput._wave < mask_wave[i * 2 + 1] + 20,
            )
            line_par = stats.linregress(
                [mask_wave[i * 2] - 10, mask_wave[i * 2 + 1] + 10],
                [
                    np.median(throughput._data[select_blue]),
                    np.median(throughput._data[select_red]),
                ],
            )

            throughput._data[select_region] = (
                line_par[0] * throughput._wave[select_region] + line_par[1]
            ).astype("float32")
            # select = np.logical_and(throughput._wave>mask_wave[i*2], throughput._wave<mask_wave[i*2+1])
            # throughput._mask[select]=True
    if mask_telluric is not None:
        star_telluric1 = star_rss.create1DSpec(method="sum")
        star_telluric2 = star_rss.create1DSpec(method="sum")
        regions = len(mask_telluric) / 2
        for i in range(regions):
            select_region = np.logical_and(
                star_telluric1._wave > mask_telluric[i * 2],
                star_telluric1._wave < mask_telluric[i * 2 + 1],
            )
            select_blue = np.logical_and(
                star_telluric1._wave > mask_telluric[i * 2] - 20,
                star_telluric1._wave < mask_telluric[i * 2],
            )
            select_red = np.logical_and(
                star_telluric1._wave > mask_telluric[i * 2 + 1],
                star_telluric1._wave < mask_telluric[i * 2 + 1] + 20,
            )
            line_par = stats.linregress(
                [mask_telluric[i * 2] - 10, mask_telluric[i * 2 + 1] + 10],
                [
                    np.median(star_telluric1._data[select_blue]),
                    np.median(star_telluric1._data[select_red]),
                ],
            )
            star_telluric2._data[select_region] = (
                line_par[0] * star_telluric1._wave[select_region] + line_par[1]
            ).astype("float32")
        telluric_spec = (star_telluric1 / star_telluric2) ** (1.0 / airmass)
        telluric_spec.writeFitsData("telluric_spec.fits")
    good_pix = np.logical_not(throughput._mask)
    if median_filt > 0:
        throughput.smoothSpec(median_filt, method="median")
    if verbose == 1:
        plt.plot(
            throughput._wave[good_pix][10:-10], throughput._data[good_pix][10:-10], "-k"
        )
    if split == "":
        mask = throughput._mask
        throughput_s = 1.0 / Spectrum1D(
            wave=throughput._wave, data=throughput._data, mask=mask
        )
        throughput_s.smoothPoly(smooth_poly)
        mask = np.logical_or(throughput_s._mask, throughput_s._data <= 0)
        throughput_s = 1.0 / throughput_s
        throughput_s._mask = mask
        if verbose == 1:
            plt.plot(throughput_s._wave, throughput_s._data, "-r")
            plt.plot(throughput_s._wave, throughput._data / throughput_s._data, "-g")
            # sens_test_out = open('test_sens.txt', 'w')
            # for i in range(throughput_s._dim):
            #     sens_test_out.write('%i %.2f %e %e %e\n'%(i, throughput_s._wave[i], throughput._data[i], throughput_s._data[i], throughput._data[i]/throughput_s._data[i]))
            # sens_test_out.close()
    else:
        split = float(split)
        overlap = float(overlap)
        select = throughput._wave > split
        mask = throughput._mask[select]
        mask[-10:] = True
        throughput_s2 = Spectrum1D(
            wave=throughput._wave[select], data=throughput._data[select], mask=mask
        )
        throughput_s2.smoothPoly(smooth_poly)

        select = throughput._wave < split + overlap
        mask = throughput._mask[select]
        mask[-10:] = True
        throughput_s1 = Spectrum1D(
            wave=throughput._wave[select], data=throughput._data[select], mask=mask
        )
        throughput_s1.smoothPoly(smooth_poly)

        if verbose == 1:
            plt.plot(throughput_s1._wave, throughput_s1._data, "-r")
            plt.plot(throughput_s2._wave, throughput_s2._data, "-r")
    if verbose == 1:
        plt.show()

    throughput_s.writeFitsData(out_throughput)


def createSensFunction2_drp(
    in_rss,
    out_sens,
    ref_spec,
    airmass,
    exptime,
    smooth_bspline="0.3",
    smooth_ref="6.0",
    smooth_ref2="6.0",
    median_filt="0",
    coadd="1",
    extinct_v="0.0",
    extinct_curve="mean",
    aper_correct="1.0",
    ref_units="1e-16",
    target_units="1e-16",
    column_wave="0",
    column_flux="1",
    delimiter="",
    header="1",
    mask_wave="",
    out_star="",
    verbose="0",
):
    smooth_bspline = float(smooth_bspline)
    smooth_ref = float(smooth_ref)
    smooth_ref2 = float(smooth_ref2)
    median_filt = int(median_filt)
    coadd = int(coadd)
    ref_units = float(ref_units)
    target_units = float(target_units)
    aper_correct = float(aper_correct)
    column_wave = int(column_wave)
    column_flux = int(column_flux)
    header = int(header)
    if mask_wave != "":
        mask_wave = np.array(mask_wave.split(",")).astype("float32")
    else:
        mask_wave = None
    verbose = int(verbose)

    rss = RSS()
    if coadd > 0:
        rss.loadFitsData(in_rss)
        select = rss.selectSpec(min=0, max=coadd, method="median")
        star_rss = rss.subRSS(select)
        star_spec = star_rss.create1DSpec(method="sum") / aper_correct
    else:
        star_spec = Spectrum1D()
        if ".fits" in in_rss:
            star_spec.loadFitsData(in_rss)
        elif ".txt" in in_rss:
            star_spec.loadTxtData(in_rss)

    try:
        extinct_v = rss.getHdrValue(extinct_v)
    except (KeyError, TypeError):  # KeyError or TypeError:
        extinct_v = float(extinct_v)

    try:
        airmass = rss.getHdrValue(airmass)
    except (KeyError, TypeError):  # KeyError or TypeError:
        airmass = float(airmass)

    try:
        exptime = rss.getHdrValue(exptime)
    except (KeyError, TypeError):  # KeyError or TypeError:
        exptime = float(exptime)

    if (
        extinct_curve == "mean"
        or extinct_curve == "summer"
        or extinct_curve == "winter"
    ):
        extinct = 10 ** (
            fluxcal.extinctCAHA(star_spec._wave, extinct_v, type=extinct_curve)
            * airmass
            * -0.4
        )
    elif extinct_curve == "Paranal":
        extinct = 10 ** (
            fluxcal.extinctParanal(star_spec._wave) * airmass * -0.4
        )
    else:
        extinct = Spectrum1D()
        extinct.loadTxtData(extinct_curve)
        extinct = 10 ** (extinct * airmass * -0.4)
        extinct = extinct.resampleSpec(star_spec._wave)

    ref_star_spec = Spectrum1D()
    ref_star_spec.loadSTDref(
        ref_spec,
        column_wave=column_wave,
        column_flux=column_flux,
        delimiter=delimiter,
        header=header,
    )
    ref_star_resamp = ref_star_spec.resampleSpec(star_spec._wave, method="linear")

    ref_star_resamp.smoothSpec(
        smooth_ref / 2.354 / (star_spec._wave[1] - star_spec._wave[0])
    )
    if out_star != "":
        star_out = open(out_star, "w")
        for i in range(star_spec._dim):
            star_out.write("%i %.3f %e\n" % (i, star_spec._wave[i], star_spec._data[i]))
        star_out.close()

    star_spec.smoothSpec(smooth_ref)
    star_corr = star_spec / extinct / exptime

    throughput = ref_star_resamp / star_corr
    if mask_wave is not None:
        regions = len(mask_wave) / 2
        for i in range(regions):
            select_region = np.logical_and(
                throughput._wave > mask_wave[i * 2],
                throughput._wave < mask_wave[i * 2 + 1],
            )
            select_blue = np.logical_and(
                throughput._wave > mask_wave[i * 2] - 20,
                throughput._wave < mask_wave[i * 2],
            )
            select_red = np.logical_and(
                throughput._wave > mask_wave[i * 2 + 1],
                throughput._wave < mask_wave[i * 2 + 1] + 20,
            )
            line_par = stats.linregress(
                [mask_wave[i * 2] - 10, mask_wave[i * 2 + 1] + 10],
                [
                    np.median(throughput._data[select_blue]),
                    np.median(throughput._data[select_red]),
                ],
            )

            throughput._data[select_region] = (
                line_par[0] * throughput._wave[select_region] + line_par[1]
            ).astype("float32")

    good_pix = np.logical_not(throughput._mask)
    if median_filt > 0:
        throughput.smoothSpec(median_filt, method="median")
    if verbose == 1:
        plt.plot(
            throughput._wave[good_pix][10:-10],
            1.0 / throughput._data[good_pix][10:-10],
            "-k",
        )

    mask = throughput._mask
    # mask[:10]=True
    # mask[-10:]=True
    throughput_s = Spectrum1D(
        wave=throughput._wave, data=1.0 / throughput._data, mask=mask
    )
    throughput_s.smoothSpec(smooth_bspline, method="BSpline")
    if verbose == 1:
        plt.plot(throughput_s._wave, throughput_s._data, "-r")
        plt.plot(
            throughput_s._wave, (1.0 / throughput._data) / throughput_s._data, "-g"
        )
        sens_test_out = open("test_sens.txt", "w")
        for i in range(throughput_s._dim):
            sens_test_out.write(
                "%i %.2f %e %e %e\n"
                % (
                    i,
                    throughput_s._wave[i],
                    throughput._data[i],
                    throughput_s._data[i],
                    throughput._data[i] / throughput_s._data[i],
                )
            )
        sens_test_out.close()
        plt.show()
    throughput_s = 1.0 / throughput_s

    # need to replace with XML output
    out = open(out_sens, "w")
    for i in range(throughput._dim):
        out.write("%i %.3f %e\n" % (i, throughput_s._wave[i], throughput_s._data[i]))
    out.close()


def quickFluxCalibration_drp(
    in_rss,
    out_rss,
    in_throughput,
    airmass,
    exptime,
    extinct_v="0.0",
    extinct_curve="mean",
    ref_units="1e-16",
    target_units="1e-16",
    norm_sb_fib="",
):
    ref_units = float(ref_units)
    target_units = float(target_units)
    rss = RSS()
    rss.loadFitsData(in_rss)
    if norm_sb_fib == "":
        norm_sb_fib = 1.0
    else:
        norm_sb_fib = np.pi * float(norm_sb_fib) ** 2
    try:
        airmass = rss.getHdrValue(airmass)
    except KeyError:
        airmass = float(airmass)

    try:
        exptime = rss.getHdrValue(exptime)
    except KeyError:
        exptime = float(exptime)

    try:
        extinct_v = rss.getHdrValue(extinct_v)
    except KeyError:
        extinct_v = float(extinct_v)

    troughput_file = open(in_throughput, "r")
    lines = troughput_file.readlines()
    wave_sens = np.zeros(len(lines), dtype=np.float32)
    sens_dat = np.zeros(len(lines), dtype=np.float32)
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 3:
            wave_sens[i] = float(line[1])
            sens_dat[i] = float(line[2])
        elif len(line) == 2:
            wave_sens[i] = float(line[0])
            sens_dat[i] = float(line[1])
    throughput = Spectrum1D(wave=wave_sens, data=sens_dat)
    if len(rss._wave.shape) == 1:
        if (
            extinct_curve == "mean"
            or extinct_curve == "summer"
            or extinct_curve == "winter"
        ):
            extinct = 10 ** (
                fluxcal.extinctCAHA(rss._wave, extinct_v, type=extinct_curve)
                * airmass
                * -0.4
            )
        elif extinct_curve == "Paranal":
            extinct = 10 ** (fluxcal.extinctParanal(rss._wave) * airmass * -0.4)
        else:
            extinct = Spectrum1D()
            extinct.loadTxtData(extinct_curve)
            extinct = 10 ** (extinct * airmass * -0.4)
            extinct = extinct.resampleSpec(rss._wave, method="spline")
        throughput_resamp = throughput.resampleSpec(rss._wave, method="spline")

        for j in range(rss._fibers):
            rss[j] = (
                (rss[j] / extinct / exptime / norm_sb_fib)
                * throughput_resamp
                * (ref_units / target_units)
            )
    #        print exptime
    rss.writeFitsData(out_rss)


def correctTelluric_drp(in_rss, out_rss, telluric_spectrum, airmass="AIRMASS"):
    """
    Corrects the wavelength calibrated RSS for the effect of telluric absoroption using
    a transmission spectrum generated from a star.

    Parameters
    --------------
    in_rss : string
            Input RSS FITS file
    out_rss : string
            Output RSS FITS file with the corrected spectra
    telluric_spectrum : string
            FITS file of the telluric transmission spectrum
    airmass : string or string of float, optional with default: 'AIRMASS'
            Airmass for the target observation.
            Either a corresponding header keyword or a float value may be used.

    Examples
    ----------------
    user:> lvmdrp rss correctTelluric in_rss.fits out_rss.fits TELL_SPEC.fits
    user:> lvmdrp rss correctTelluric in_rss.fits out_rss.fits TELL_SPEC.fits  1.4
    """
    rss = loadRSS(in_rss)
    telluric = Spectrum1D()
    telluric.loadFitsData(telluric_spectrum)
    telluric._mask = None
    telluric._error = None

    try:
        airmass = rss.getHdrValue(airmass)
    except KeyError:
        airmass = float(airmass)
    if len(rss._wave.shape) == 1:
        telluric_resamp = telluric.resampleSpec(rss._wave)
        rss_corr = rss * (1.0 / (telluric_resamp ** (airmass)))

    elif len(rss._wave.shape) == 2:
        rss_corr = rss
        for i in range(len(rss._fibers)):
            spec = rss[i]
            telluric_resamp = telluric.resampleSpec(spec._wave)
            rss_corr[i] = spec * (1.0 / (telluric_resamp ** (airmass)))
    rss_corr.writeFitsData(out_rss)


# -------------------------------------------------------------------------------------------------
