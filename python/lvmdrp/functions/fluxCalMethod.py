# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jan 25, 2023
# @Filename: fluxCalMethod
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM


import os
# from os import listdir
# from os.path import isfile, join
import numpy as np
from scipy import interpolate
from scipy import ndimage
# from scipy.ndimage import median_filter
from scipy.signal import find_peaks
# import re
import pandas as pd

from astropy import units as u
from astropy.stats import biweight_location, biweight_scale
from astropy.table import Table
from astropy.io import fits
from astroquery.gaia import Gaia

from lvmdrp.core.rss import RSS, loadRSS, lvmFFrame
from lvmdrp.core.spectrum1d import Spectrum1D
import lvmdrp.core.fluxcal as fluxcal
from lvmdrp.core.sky import get_sky_mask_uves, get_z_continuum_mask
from lvmdrp import log

from lvmdrp.core.plot import plt, create_subplots, save_fig
from lvmdrp.core.constants import MASTERS_DIR

description = "provides flux calibration tasks"


def apply_fluxcal(in_rss: str, out_fframe: str, method: str = 'MOD', display_plots: bool = False):
    """applies flux calibration to spectrograph-combined data

    Parameters
    ----------
    in_rss : str
        input RSS file
    out_rss : str
        output RSS file
    method : str
        'STD' - apply calibration inferred from standard stars
        'SCI' - apply calibration inferred from field stars in science ifu (fallback if STD not available)
        'MOD' - apply calibration inferred from stellar atmosphere models  (default)
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

    expnum = fframe._header["EXPOSURE"]
    channel = fframe._header["CCD"]

    # set masked pixels to NaN
    fframe.apply_pixelmask()
    # load fibermap and filter for current spectrograph
    slitmap = fframe._slitmap

    # apply joint sensitivity curve
    fig, ax = create_subplots(to_display=display_plots, figsize=(10, 5))
    fig.suptitle(f"Flux calibration for {expnum = }, {channel = }")
    log.info(f"computing joint sensitivity curve for channel {channel}")
    # calculate exposure time factors
    # std_exp = np.asarray([fframe._header.get(f"{std_hd[:-3]}EXP", 1.0) for std_hd in fframe._fluxcal.colnames])
    # weights = std_exp / std_exp.sum()
    # TODO: reject sensitivity curves based on the overall shape by normalizing using a median curve
    # calculate the biweight mean sensitivity

    # update the fluxcal extension
    fframe._fluxcal_std["mean"] = biweight_location(fframe._fluxcal_std.to_pandas().values, axis=1, ignore_nan=True) * u.Unit("erg / (ct cm2)")
    fframe._fluxcal_std["rms"] = biweight_scale(fframe._fluxcal_std.to_pandas().values, axis=1, ignore_nan=True) * u.Unit("erg / (ct cm2)")

    fframe._fluxcal_mod["mean"] = biweight_location(fframe._fluxcal_mod.to_pandas().values, axis=1, ignore_nan=True) * u.Unit("erg / (ct cm2)")
    fframe._fluxcal_mod["rms"] = biweight_scale(fframe._fluxcal_mod.to_pandas().values, axis=1, ignore_nan=True) * u.Unit("erg / (ct cm2)")

    fframe._fluxcal_sci["mean"] = biweight_location(fframe._fluxcal_sci.to_pandas().values, axis=1, ignore_nan=True) * u.Unit("erg / (ct cm2)")
    fframe._fluxcal_sci["rms"] = biweight_scale(fframe._fluxcal_sci.to_pandas().values, axis=1, ignore_nan=True) * u.Unit("erg / (ct cm2)")

    # check for flux calibration data
    if method == "NONE":
        log.info("skipping flux calibration")
        fframe.setHdrValue("FLUXCAL", 'NONE', "flux calibration method")
        fframe.writeFitsData(out_fframe)
        return fframe

    # if instructed, use standard stars
    if method == 'STD':
        log.info("calculating sensitivity using STD standard stars")

        sens_arr = fframe._fluxcal_std.to_pandas().values[:, :-2]
        sens_ave = fframe._fluxcal_std["mean"].value

        # fall back to science field if all invalid values
        if (sens_ave == 0).all() or np.isnan(sens_ave).all() or (sens_ave<0).any():
            log.warning("all standard star sensitivities are <=0 or NaN, falling back to SCI stars")
            rss.add_header_comment("all standard star sensitivities are <=0 or NaN, falling back to SCI stars")
            method = 'SCI'
        # fall back to science field if less than 8 standard stars
        elif (~np.isnan(sens_arr).all(axis=0)).sum() < 8:
            log.warning("less than 8 good standard fibers, falling back to science field calibration")
            rss.add_header_comment("less than 8 good standard fibers, falling back to science field calibration")
            method = "SCI"

    if method == 'MOD':
        log.info("calculating sensitivity using model stellar spectra")

        sens_arr = fframe._fluxcal_mod.to_pandas().values[:, :-2]
        sens_ave = fframe._fluxcal_mod["mean"].value

        # fall back to science field if all invalid values
        if (sens_ave == 0).all() or np.isnan(sens_ave).all():
            log.warning("all standard star sensitivities are <=0 or NaN, falling back to SCI stars")
            rss.add_header_comment("all standard star sensitivities are <=0 or NaN, falling back to SCI stars")
            method = 'SCI'
        # fall back to science field if less than 8 standard stars
        elif (~np.isnan(sens_arr).all(axis=0)).sum() < 8:
            log.warning(f"{np.isnan(sens_arr).all(axis=0).sum()} good standard fibers")
            log.warning("less than 8 good standard fibers, falling back to science field calibration")
            rss.add_header_comment("less than 8 good standard fibers, falling back to science field calibration")
            method = "SCI"

    # fall back to science ifu field stars if above failed or if instructed to use this method
    if method == 'SCI':
        log.info("calculating sensitivity using SCI field stars")

        sens_arr = fframe._fluxcal_sci.to_pandas().values[:, :-2]
        sens_ave = fframe._fluxcal_sci["mean"].value

        # fix case of all invalid values
        if (sens_ave == 0).all() or np.isnan(sens_ave).all():
            log.warning("all field star sensitivities are zero or NaN, can't calibrate")
            rss.add_header_comment("all field star sensitivities are zero or NaN, can't calibrate")
            # sens_ave = np.ones_like(sens_ave)
            # sens_rms = np.zeros_like(sens_rms)

    # final check on sensitivities
    if method == "STD" and np.nanmean(fframe._fluxcal_std["mean"]) > 1e-12:
        method = "SCI"
        sens_arr = fframe._fluxcal_sci.to_pandas().values[:, :-2]
        sens_ave = fframe._fluxcal_sci["mean"].value
        log.warning("standard calibration has average sensitivity > 1e-12, falling back to science field calibration")
        rss.add_header_comment("standard calibration has average sensitivity > 1e-12, falling back to science field calibration")
    if method == "SCI" and np.nanmean(fframe._fluxcal_sci["mean"]) > 1e-12:
        method = "STD"
        sens_arr = fframe._fluxcal_std.to_pandas().values[:, :-2]
        sens_ave = fframe._fluxcal_std["mean"].value
        log.warning("science field calibration has average sensitivity > 1e-12, falling back to standard calibration")
        rss.add_header_comment("science field calibration has average sensitivity > 1e-12, falling back to standard calibration")
    # if method == 'MOD':
    #     log.info("flux-calibratimg using model stellar spectra")
    #     sens_arr = fframe._fluxcal_mod.to_pandas().values  # * (std_exp / std_exp.sum())[None]
    #     sens_ave = biweight_location(sens_arr, axis=1, ignore_nan=True)
    #     sens_rms = biweight_scale(sens_arr, axis=1, ignore_nan=True)

    if np.nanmean(sens_ave) > 1e-12 or np.isnan(sens_ave).all() or (sens_ave == 0).all():
        method = "NONE"
        log.warning("template matching, standard and science field calibration yield unreliable average sensitivity, skipping flux calibration")
        rss.add_header_comment("template matching, standard and science field calibration yield unreliable average sensitivity, skipping flux calibration")

    fframe.setHdrValue("FLUXCAL", method, "flux calibration method")

    if method != 'NONE':
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
    sci_secz = fframe._header["SCIAM"]

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
        for std_hd in fframe._fluxcal_std.colnames[:-2]:
            exptime = fframe._header[f"{std_hd[:-3]}EXP"]
            fiberid = fframe._header[f"{std_hd[:-3]}FIB"]
            exptimes[slitmap["orig_ifulabel"] == fiberid] = exptime

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
        fframe.setHdrValue("FLUXCAL", 'NONE', "flux calibration method")
        fframe.setHdrValue("BUNIT", "electron / (Angstrom s)", "physical units of the array values")
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
        fframe.setHdrValue("BUNIT", "erg / (Angstrom s cm2)", "physical units of the array values")

    log.info(f"writing output file in {os.path.basename(out_fframe)}")
    fframe.writeFitsData(out_fframe)

    return fframe


def linear_to_logscale(wl, flux):
    wl = np.float64(wl)
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

def prepare_spec(in_rss, width=3):
    '''
    Preparation of the standard star spectra for subsequent model matching.
    Read the standard star spectra and errors
    Subtract the master sky, divide by exptime
    Correct for atmospheric extinction
    Convolve to 2.3A (the same as low resolution models)
    Fit continuum (160A median filter) and normalize the spectra
    :param in_rss:
    :param width:
    :return:
    w:
        wavelength array
    gaia_ids:
        gaia_ids (needed for QA plots)
    fibers
        fiber ids (P1-1,P1-2, etc; needed for QA plots)
    std_spectra_all_bands:
        standard star spectra for all bands (uncolvolved, unnormalized, needed for sens. curves), ext. corrected
    normalized_spectra_unconv_all_bands:
        normalized unconvolved std spectra (needed for QA model matching plots), ext. corrected
    normalized_spectra_all_bands:
        normalizes std spectra convolved to 2.3A (needed for model matching), ext. corrected
    std_errors_all_bands:
        error array
    lsf_all_bands:
        LVM spectrograph LSFs for standard star fibers (needed to convolve good res. models before calculation of the sens. curve)
    '''
    w = [] # wavelength arrays
    ext = []
    normalized_spectra_all_bands = []
    normalized_spectra_unconv_all_bands = []
    std_errors_all_bands = []
    lsf_all_bands = []
    std_spectra_all_bands = [] ## contains original std spectra for all stars in ALL band
    fibers_all_bands = []
    gaia_ids_all_bands = []
    nns_all_bands = []
    # keys = ['fiber_0', 'good_flux_0', 'fiber_1', 'good_flux_1', 'fiber_2', 'good_flux_2']
    # check_bad_fluxes = {key: [] for key in keys}

    for b in range(len(in_rss)):
        #log.info(f"loading input RSS file '{os.path.basename(in_rss[b])}'")
        rss_tmp = RSS.from_file(in_rss[b])

        # get the list of standards from the header
        try:
            stds = fluxcal.retrieve_header_stars(rss=rss_tmp)
        except KeyError:
            pass
            log.warning(f"no standard star metadata found in '{in_rss}', skipping sensitivity measurement")
            # rss.add_header_comment(f"no standard star metadata found in '{in_rss}', skipping sensitivity measurement")
            # rss.set_fluxcal(fluxcal=res_std, source='std')
            # rss.writeFitsData(in_rss)
            # TODO: fix this, this seems to be copy-pasted from the gaia code
            # return res_std, mean_std, rms_std, rss
            stds = []

        # wavelength array
        w_tmp = rss_tmp._wave
        w.append(w_tmp)

        # load the sky masks
        channel = rss_tmp._header['CCD']
        m = get_sky_mask_uves(w[b], width=width)
        m2 = None
        if channel == "z":
            m2 = get_z_continuum_mask(w_tmp)

        master_sky = rss_tmp.eval_master_sky()

        # iterate over standard stars
        std_spectra = []  # contains original std spectra for all stars in each band
        normalized_spectra = []
        normalized_spectra_unconv = []
        std_errors = []
        lsf = []
        fibers = []
        gaia_ids = []
        nns = []

        for s in stds:
            nn, fiber, gaia_id, exptime, secz = s  # unpack standard star tuple

            # find the fiber with our spectrum of that Gaia star, if it is not in the current spectrograph, continue
            select = rss_tmp._slitmap["orig_ifulabel"] == fiber
            fibidx = np.where(select)[0]

            log.info(f"standard fiber '{fiber}', index '{fibidx}', star '{gaia_id}', exptime '{exptime:.2f}', secz '{secz:.2f}'")

            # subtract sky spectrum and divide by exptime
            spec_tmp = rss_tmp._data[fibidx[0], :]
            error_tmp = rss_tmp._error[fibidx[0], :]
            lsf_tmp = rss_tmp._lsf[fibidx[0], :]
            # check_bad_fluxes[f'fiber_{b}'].append(fiber)
            if np.nanmean(spec_tmp) < 100:
                log.warning(f"fiber {fiber} @ {fibidx[0]} has counts < 100 e-, skipping")
                # check_bad_fluxes[f'good_flux_{b}'].append(False)
                if b > 0:
                    for ind_b in range(b):
                        try:
                            idx = fibers_all_bands[ind_b].index(fiber)
                        except ValueError:
                            continue
                        del fibers_all_bands[ind_b][idx]
                        del normalized_spectra_all_bands[ind_b][idx]
                        del normalized_spectra_unconv_all_bands[ind_b][idx]
                        del std_errors_all_bands[ind_b][idx]
                        del lsf_all_bands[ind_b][idx]
                        del std_spectra_all_bands[ind_b][idx]
                        del gaia_ids_all_bands[ind_b][idx]
                        del nns_all_bands[ind_b][idx]
                # #rss.add_header_comment(f"fiber {fiber} @ {fibidx[0]} has counts < 100 e-, skipping")
                continue
            if b > 0:
                good_fiber = True
                for ind_b in range(b):
                    try:
                        idx = fibers_all_bands[ind_b].index(fiber)
                    except ValueError:
                        good_fiber = False
                        continue
                if not good_fiber:
                    continue
            gaia_ids.append(gaia_id)
            fibers.append(fiber)
            nns.append(nn)
            # check_bad_fluxes[f'good_flux_{b}'].append(True)

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
            lsf_conv = np.sqrt(np.clip(2.3 ** 2 - lsf_tmp ** 2, 0.1, None))/pxsize  # as model spectra were already convolved with lsf=2.3 A,
            # we need to degrade our observed std spectra. Also, convert it to pixels
            mask_bad = ~np.isfinite(spec_tmp)
            mask_lsf = ~np.isfinite(lsf_conv)
            lsf_conv_interpolated = fluxcal.interpolate_mask(w_tmp, lsf_conv, mask_lsf, fill_value="extrapolate")

            # # degrade observed std spectra
            spec_tmp_convolved = fluxcal.lsf_convolve(spec_ext_corr, lsf_conv_interpolated, w_tmp)

            # Obtain continuum with 160A median filter and normalize spectra
            best_continuum = ndimage.filters.median_filter(spec_tmp_convolved, int(160/0.5), mode="nearest")
            error_tmp = 1 / error_tmp**0.5
            std_errors.append(error_tmp / best_continuum)
            normalized_spectra.append(spec_tmp_convolved/best_continuum) # normalized std spestra degraded to 2.3A for all
                                                                        # standards in each channel
            best_continuum = ndimage.filters.median_filter(spec_ext_corr, int(160/0.5), mode="nearest")
            normalized_spectra_unconv.append(spec_ext_corr/best_continuum)
            lsf.append(lsf_tmp) # initial std spec LSF for all standards in each channel
            std_spectra.append(spec_ext_corr)

        normalized_spectra_all_bands.append(normalized_spectra) # normalized std spectra degraded to 2.3A for all
                                                                        # standards and all channels together
        normalized_spectra_unconv_all_bands.append(normalized_spectra_unconv)
        std_errors_all_bands.append(std_errors)
        lsf_all_bands.append(lsf) # initial std spec LSF for all standards and all channel together
        std_spectra_all_bands.append(std_spectra) # corrected for extinction
        fibers_all_bands.append(fibers)
        gaia_ids_all_bands.append(gaia_ids)
        nns_all_bands.append(nns)

    return w, nns_all_bands[0], gaia_ids_all_bands[0], fibers, std_spectra_all_bands, normalized_spectra_unconv_all_bands, normalized_spectra_all_bands, std_errors_all_bands, lsf_all_bands

def model_selection(in_rss, GAIA_CACHE_DIR=None, width=3, plot=True):
    """ Selection of the stellar atmosphere model spectra (POLLUX database, AMBRE library)
    The model spectra: -1.5 <= Z <= 0.75; 3 <= logg <= 5; 5500 <0 Teff <= 8000; microturb_vel = 1.0; O/Fe = 0.0
    Read file with models that contains:
    - good res. (0.3A), non-normalized models - will be used to get the sensitivity curves
    - logscale, low-resolution (convolved to 2.3A), normalized models - will be used for model matching
    - model parameters
    Correct observed standard spectrum for the atmospheric extinction
    Fit continuum to observed (corrected for the extinction) standard spectra (in 3 channels separately)
    Normalise observed standard spectra and stitch 3 channels together
    Find the vel. shift relative to template model
    Shift the standard star spectrum in logscale and find the best-fit model using chi-square
    Mask regions with high chi-square values and find the best-fit model using chi-square
    Find the conversion coefficient between model units and Gaia units:
        Read the best-fit model with good resolution, non-normalised
        Convolve with Gaia LSF
        Calculate median fot stdflux/model_convolved_to_gaia - we will use this coefficient
    QA plots
    Calculate the sensitivity curves

    :param in_rss:
    :param GAIA_CACHE_DIR:
    :param width:
    :return:
    """
    # TODO: think about uniting this code and the fluxcal code that iterates over cameras?
    # TODO: find a place under the calib directory structure for the stellar models
    # TODO: telluric list should go in lvmcore
    # models_dir = '/Users/amejia/Downloads/stellar_models/'
    models_dir = os.path.join(MASTERS_DIR, "stellar_models")
    telluric_file = os.path.join(os.getenv("LVMCORE_DIR"), 'etc', 'telluric_lines.txt')  # wavelength regions with Telluric
    # absorptions based on KPNO data (unknown source) with a 1% transmission threshold this file is used as a mask for
    # the fit of standard stars - from Alfredo.
    # https://github.com/desihub/desispec/blob/main/py/desispec/data/arc_lines/telluric_lines.txt
    telluric_tab = Table.read(telluric_file, format='ascii.fixed_width_two_line')

    with fits.open(name=models_dir + '/lvm-models_ambre-all.fits') as model:
        model_good = model[0].data
        model_norm = model[1].data
        model_info = pd.DataFrame(model[2].data)
    model_names = model_info['Model_name'].to_list()
    n_models = len(model_names)
    log.info(f'Number of models: {n_models}')

    GAIA_CACHE_DIR = "./" if GAIA_CACHE_DIR is None else GAIA_CACHE_DIR
    log.info(f"Using Gaia CACHE DIR '{GAIA_CACHE_DIR}'")

    # Prepare the spectra
    (w, nns, gaia_ids, fibers, std_spectra_all_bands, normalized_spectra_unconv_all_bands, normalized_spectra_all_bands,
     std_errors_all_bands, lsf_all_bands) = prepare_spec(in_rss, width=width)

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

    # table with masks for tellurics, overlaps between channels, and bluest part og the spectra - used for model matching
    br_overlap_start = 5775
    br_overlap_end = 5825
    rz_overlap_start = 7520 #7520
    rz_overlap_end = 7580 #7570
    mask_line_start = 8620
    mask_line_end = 8690
    mask_for_fit = telluric_tab
    mask_for_fit['Start'] = mask_for_fit['Start'] - 10
    mask_for_fit['End'] = mask_for_fit['End'] + 10
    mask_for_fit.add_row([3500,3800]) #mask the bluest part of the spectra
    mask_for_fit.add_row([br_overlap_start, br_overlap_end])
    mask_for_fit.add_row([rz_overlap_start, rz_overlap_end])
    mask_for_fit.add_row([mask_line_start, mask_line_end])

    model_to_gaia_median = []
    best_fit_models = []
    gaia_flux_interpolated = []
    gaia_Teff = []
    gaia_logg = []
    gaia_z = []

    # Stitch normalized spectra in brz together
    for i, nn in enumerate(nns):
        std_norm_unconv = np.concatenate((normalized_spectra_unconv_all_bands[0][i][mask_b_norm],
                                          normalized_spectra_unconv_all_bands[1][i][mask_r_norm],
                                          normalized_spectra_unconv_all_bands[2][i][mask_z_norm]))
        std_normalized_all_convolved = np.concatenate((normalized_spectra_all_bands[0][i][mask_b_norm],
                                             normalized_spectra_all_bands[1][i][mask_r_norm],
                                             normalized_spectra_all_bands[2][i][mask_z_norm]))
        std_errors_normalized_all = np.concatenate((std_errors_all_bands[0][i][mask_b_norm],
                                                    std_errors_all_bands[1][i][mask_r_norm],
                                                    std_errors_all_bands[2][i][mask_z_norm]))
        # lsf_all (initial std lsf) - will be used to convolve good res models for sens curve calculation
        lsf_all = np.concatenate((lsf_all_bands[0][i][mask_b_norm],
                                             lsf_all_bands[1][i][mask_r_norm],
                                             lsf_all_bands[2][i][mask_z_norm]))
        log_std_wave_all, flux_std_unconv_logscale = linear_to_logscale(std_wave_all, std_norm_unconv)
        log_std_wave_all, flux_std_logscale = linear_to_logscale(std_wave_all, std_normalized_all_convolved)
        log_std_wave_all, log_std_errors_normalized_all = linear_to_logscale(std_wave_all, std_errors_normalized_all)

        # mask tellurics, channels overlaps, and bluest part of the spectra in log scale
        mask_good = np.zeros_like(log_std_wave_all, dtype=bool)
        for wave_masks in range(len(mask_for_fit)):
            mask_good = mask_good | ((log_std_wave_all > np.log(mask_for_fit['Start'][wave_masks]))
                                     & (log_std_wave_all < np.log(mask_for_fit['End'][wave_masks])))
        mask_good = ~mask_good & np.isfinite(flux_std_logscale) #~mask_tellurics_log & ~mask_wave

        # canonical f-type model: Teff=6500, logg=4, Fe/H=-1.5 or something like that
        # Check the possible velocity offsets IN LOGSCALE
        # Now we use the model template with Teff=6250, logg=3.5, Fe/H=-0.5
        template_index =  model_info.index[(model_info['Teff'] == 6250) & (model_info['logg']==3.5) & (model_info['Z']==0.5)][0]
        template = model_norm[template_index]
        log_model_wave_all = log_std_wave_all
        flux_model_logscale = template

        log_shift_full = fluxcal.derive_vecshift(flux_std_logscale[mask_good], flux_model_logscale[mask_good],
                                            max_ampl=3)*np.median(log_std_wave_all - np.roll(log_std_wave_all, 1))
        vel_shift_full = log_shift_full * 3e5
        flux_std_logscale_shifted = np.interp((log_std_wave_all - log_shift_full), log_std_wave_all, flux_std_logscale)

        best_id, chi2_bestfit, chi2_wave_bestfit_0 = chi2_model_matching(flux_std_logscale_shifted,
                                                                            log_std_errors_normalized_all,
                                                                            model_norm, mask_good)
        log.info(f'Initial chi2={chi2_bestfit:.2f}, initial model {best_id}')
        mask_chi2 = ~np.zeros_like(chi2_wave_bestfit_0, dtype=bool)

        chi2_threshold = 20
        peak_width = 10
        peaks, properties = find_peaks(chi2_wave_bestfit_0, height=chi2_threshold, width=[1, peak_width])
        for peak in peaks:
            width = int(properties["widths"][np.where(peaks == peak)][0])  # Use detected width
            start = max(0, peak - width)
            end = min(len(chi2_wave_bestfit_0), peak + width)
            mask_chi2[start:end] = False

        combined_mask = np.zeros_like(mask_good, dtype=bool)
        combined_mask[mask_good] = mask_chi2
        mask_upd = mask_good & combined_mask

        best_id, chi2_bestfit, chi2_wave_bestfit = chi2_model_matching(flux_std_logscale_shifted,
                                                                         log_std_errors_normalized_all,
                                                                         model_norm, mask_upd)

        npix_masked = len(chi2_wave_bestfit_0) - len(chi2_wave_bestfit)
        log.info(f'Masked {npix_masked} pixels')
        log.info(f'After additional masking {chi2_bestfit:.2f}')

        log.info(f"GAIA id:{gaia_ids[i]}. Best model is: {best_id}, {model_names[best_id]}")
        best_fit_models.append(model_names[best_id])

        # Conversion coefficient model to gaia units
        model_flux = model_good[best_id]
        n_steps = int((9850 - 3550) / 0.05) + 1
        model_wave = np.linspace(3550, 9850, n_steps)

        mask_model = (model_wave >= min(std_wave_all)) & (model_wave <= max(std_wave_all))
        model_wave = model_wave[mask_model]
        model_flux = model_flux[mask_model]

        # resample model to the same step
        model_flux_resampled = np.interp(std_wave_all, model_wave, model_flux)
        good_model_to_std_lsf = np.sqrt(lsf_all ** 2 - 0.3 ** 2)/0.5 # to degrade good resolution model to std lsf for plots
        model_convolved_spec_lsf = fluxcal.lsf_convolve(model_flux_resampled, good_model_to_std_lsf, std_wave_all)
        best_continuum = ndimage.filters.median_filter(model_convolved_spec_lsf, int(160/0.5), mode="nearest")
        model_norm_convolved_spec_lsf = model_convolved_spec_lsf / best_continuum
        log_std_wave_all_tmp, log_model_norm_convolved_spec_lsf = linear_to_logscale(std_wave_all, model_norm_convolved_spec_lsf)
        # model_shifted_norm_convolved_spec_lsf = logscale_to_linear(std_wave_all, log_std_wave_all_tmp,
        #                                                            log_model_norm_convolved_spec_lsf, log_shift_full)

        # Gaia LSF
        gaia_lsf_path = os.getenv("LVMCORE_DIR") + "/etc/Gaia_BPRP_resolution.txt"
        gaia_lsf_table_tmp = Table.read(gaia_lsf_path, format='ascii',
                                        names=['wavelength', 'resolution'])
        # gaia_lsf_table_tmp['wavelength'][len(gaia_lsf_table_tmp['wavelength']) - 1] = gaia_lsf_table_tmp['wavelength'][
        #                                                                                   len(
        #                                                                                       gaia_lsf_table_tmp[
        #                                                                                           'wavelength']) - 1] * 10
        gaia_lsf_table_tmp['linewidth'] = gaia_lsf_table_tmp['wavelength'] / gaia_lsf_table_tmp['resolution']
        gaia_lsf_table_bp = gaia_lsf_table_tmp[0:10]
        gaia_lsf_table_rp = gaia_lsf_table_tmp[10:17]
        wave_bprp_mean = (max(gaia_lsf_table_bp['wavelength']) + min(gaia_lsf_table_rp['wavelength'])) / 2
        mask_wl_bp = (std_wave_all < wave_bprp_mean)
        mask_wl_rp = (std_wave_all >= wave_bprp_mean)
        gaia_lsf_bp = np.interp(std_wave_all[mask_wl_bp], gaia_lsf_table_bp['wavelength'], gaia_lsf_table_bp['linewidth'])
        gaia_lsf_rp = np.interp(std_wave_all[mask_wl_rp], gaia_lsf_table_rp['wavelength'], gaia_lsf_table_rp['linewidth'])
        gaia_lsf = np.concatenate((gaia_lsf_bp, gaia_lsf_rp))

        # load Gaia BP-RP spectrum from cache, or download from webapp, and fit the continuum to Gaia spec
        try:
            gw, gf = fluxcal.retrive_gaia_star(gaia_ids[i], GAIA_CACHE_DIR=GAIA_CACHE_DIR)
            stdflux = np.interp(std_wave_all, gw, gf)  # interpolate to our wavelength grid
            gaia_flux_interpolated.append(stdflux)

            job = Gaia.launch_job(f"SELECT teff_gspspec, logg_gspspec, mh_gspspec FROM gaiadr3.astrophysical_parameters WHERE source_id = {gaia_ids[i]} ")
            r = job.get_results()
            gaia_Teff.append(r['teff_gspspec'])
            gaia_logg.append(r['logg_gspspec'])
            gaia_z.append(r['mh_gspspec'])

        except fluxcal.GaiaStarNotFound as e:
            gaia_Teff.append(np.nan)
            gaia_logg.append(np.nan)
            gaia_z.append(np.nan)
            gaia_flux_interpolated.append(np.ones_like(std_wave_all) * np.nan)
            model_to_gaia_median.append(np.nan)
            log.warning(f"Gaia star {gaia_ids[i]} not found: {e}")
            # rss_tmp.add_header_comment(f"Gaia star {gaia_ids[i]} not found")
            continue

        # convolve model to gaia lsf
        model_convolved_to_gaia = fluxcal.lsf_convolve(model_flux_resampled, gaia_lsf/0.5, std_wave_all)
        model_to_gaia = stdflux/model_convolved_to_gaia
        model_to_gaia_median.append(np.median(model_to_gaia))

        # prepare dictionaries to plot QA plots for model matching
        fig_path = in_rss[0]
        fiber_params = {'i':i,'fiber_id':fibers[i]}
        gaia_params = {'gaia_id':gaia_ids[i],'gaia_Teff':gaia_Teff[i][0],'gaia_logg':gaia_logg[i][0],'gaia_z':gaia_z[i][0]}
        model_params = {'model_name':model_names[best_id], 'model_Teff':model_info['Teff'][best_id],
                        'model_logg': model_info['logg'][best_id],'model_z':model_info['Z'][best_id]}
        matching_params = {'vel_shift':vel_shift_full, 'log_vel_shift':log_shift_full, 'npix_masked':npix_masked,
                           'peaks':peaks, 'properties':properties, 'chi2_threshold':chi2_threshold,
                           'chi2_bestfit':chi2_bestfit, 'chi2_wave_bestfit':chi2_wave_bestfit,
                           'chi2_wave_bestfit_0':chi2_wave_bestfit_0, 'model_to_gaia':model_to_gaia}
        mask_dict = {'mask_for_fit':mask_for_fit, 'mask_good':mask_good, 'mask_chi2':mask_chi2}
        wave_arrays = {'std_wave_all':std_wave_all, 'log_std_wave_all':log_std_wave_all,
                       'log_model_wave_shifted':log_model_wave_all + log_shift_full}

        if plot:
            qa_model_matching(fig_path,
                              fiber_params = fiber_params,
                              gaia_params=gaia_params,
                              model_params=model_params,
                              matching_params=matching_params,
                              wave_arrays=wave_arrays,
                              stdflux=stdflux,
                              flux_std_unconv_logscale=flux_std_unconv_logscale,
                              log_std_errors_normalized_all=log_std_errors_normalized_all,
                              model_flux_resampled=model_flux_resampled,
                              log_model_norm_convolved_spec_lsf = log_model_norm_convolved_spec_lsf,
                              model_convolved_to_gaia=model_convolved_to_gaia,
                              mask_dict=mask_dict)


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

        for i, nn in enumerate(nns):
            if np.isnan(model_to_gaia_median[i]):
                continue
            # !now telluric correction does not work!
            std_telluric_corrected = correct_tellurics(w[n_chan], std_spectra_all_bands[n_chan][i], lsf_all_bands[n_chan][i], in_rss[n_chan], chan)
            sens_tmp = calc_sensitivity_from_model(w[n_chan], std_telluric_corrected, lsf_all_bands[n_chan][i],
                                                   model_good[best_id], model_to_gaia_median[i], log_shift_full) #model_names[best_id]
            wgood, sgood = fluxcal.filter_channel(w[n_chan], sens_tmp, 3, method='savgol')
            # if chan == 'b':
            #     win = 150
            #     ylim = [0, 0.3e-11]
            # elif chan == 'r':
            #     win = 70
            #     ylim = [0, 0.5e-12]
            # else:
            #     win = 15
            #     ylim = [0, 0.5e-12]
            s = interpolate.make_smoothing_spline(wgood, sgood, lam=1e4) #lam=win
            sens0 = s(w[n_chan]).astype(np.float32)
            # wgood, sgood = fluxcal.filter_channel(w, sens, 2) #for std
            # s = interpolate.make_smoothing_spline(wgood, sgood, lam=1e4)

            # calculate the normalization of the average (known) sensitivity curve in a broad band
            lvmflux = fluxcal.spec_to_LVM_flux(chan, w[n_chan], std_spectra_all_bands[n_chan][i]*sens0)
            gaia_flux = fluxcal.spec_to_LVM_flux(chan, std_wave_all, gaia_flux_interpolated[i])
            sens_coef = gaia_flux/lvmflux
            #print(f'lvmflux={lvmflux}, gaia_flux={gaia_flux}, converted to gaia flux = {lvmflux*sens_coef}')

            res_mod[f"STD{nn}SEN"] = s(w[n_chan]).astype(np.float32)*sens_coef
            sens = sens0*sens_coef

            # fig_path = in_rss[n_chan]
            if plot:
                plt.plot(wgood, sgood*sens_coef, ".k", markersize=2, zorder=-999)
                plt.plot(w[n_chan], sens, linewidth=1, zorder=-999, label = fibers[i])

                plt.legend()

        res_mod_pd = res_mod.to_pandas().values
        rms_mod = biweight_scale(res_mod_pd, axis=1, ignore_nan=True)
        mean_mod = biweight_location(res_mod_pd, axis=1, ignore_nan=True)

        label = rss._header['CCD']
        mean_mod_band = np.nanmean(mean_mod[1000:3000])
        rms_mod_band = np.nanmean(rms_mod[1000:3000])
        mean_mod_band = -999.9 if np.isnan(mean_mod_band) else mean_mod_band
        rms_mod_band = -999.9 if np.isnan(rms_mod_band) else rms_mod_band
        rss.setHdrValue(f"MODSENM{label}", mean_mod_band, f"Mean model sensitivity in {chan}")
        rss.setHdrValue(f"MODSENR{label}", rms_mod_band, f"Mean model sensitivity rms in {chan}")
        log.info(f"Mean model sensitivity in {chan} : {mean_mod_band}")

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
        log.info(f'appending FLUXCAL_MOD table to {in_rss[n_chan]}')
        rss.set_fluxcal(fluxcal=res_mod, source='mod')
        rss.writeFitsData(in_rss[n_chan])

    return

def chi2_model_matching(std_spectra, std_errors, model_norm, mask):
    """
    Find best-fit model with chi-square
    :param std_spectra:
    :param std_errors:
    :param model_norm:
    :param mask:
    :return:
    best_id:
        id of the best fit model
    chi2_bestfit:
        reduced chi square
    chi2_wave_bestfit:
        chi2 values by wavelength
    """
    chi2 = [np.nansum((std_spectra[mask] - model_norm[model_ind][mask]) ** 2 / (std_errors[mask] ** 2
                                                            + (0.05 * model_norm[model_ind][mask]) ** 2)) /
                                                            len(std_spectra) for model_ind in range(len(model_norm))]
    best_id = np.argmin(chi2)
    chi2_bestfit = np.nansum((std_spectra[mask] - model_norm[best_id][mask]) ** 2 / (std_errors[mask] ** 2 + (0.05 *
                                                            model_norm[best_id][mask]) ** 2)) / len(std_spectra)
    chi2_wave_bestfit = (std_spectra[mask] - model_norm[best_id][mask]) ** 2 / (std_errors[mask] ** 2 + (0.05 *
                                                            model_norm[best_id][mask]) ** 2)
    return best_id, chi2_bestfit, chi2_wave_bestfit

def qa_model_matching(fig_path, fiber_params = None, gaia_params = None, model_params = None, matching_params = None,
                      wave_arrays = None, stdflux = None, flux_std_unconv_logscale = None,
                      log_std_errors_normalized_all = None, model_flux_resampled = None,
                      log_model_norm_convolved_spec_lsf = None, model_convolved_to_gaia = None, mask_dict = None):

    plt.figure(figsize=(14, 27))

    plt.subplot(611)
    plt.title(label=f'Gaia ID: {gaia_params["gaia_id"]}. Model: {model_params["model_name"]}', fontsize=14)
    plt.plot(wave_arrays['log_std_wave_all'], flux_std_unconv_logscale, label=f'Observed standard spectrum from fiber '
                                                               f'{fiber_params["fiber_id"]}, continuum normalized', linewidth=1)
    sigma1 = flux_std_unconv_logscale + log_std_errors_normalized_all  # flux_std_logscale
    sigma2 = flux_std_unconv_logscale - log_std_errors_normalized_all
    # plt.plot(log_std_wave_all, sigma2, '--', color='grey', lw=0.1)
    # plt.plot(log_std_wave_all, sigma1, '--', color='grey', lw=0.1)
    # plt.fill_between(log_std_wave_all, sigma1, sigma2, alpha=0.2, color='blue')

    plt.plot(wave_arrays['log_model_wave_shifted'], log_model_norm_convolved_spec_lsf,
             label='Best-fit model spectrum, continuum normalized and convolved with std LSF', alpha=0.7, linewidth=1)  # shifted
    sigma1_model = log_model_norm_convolved_spec_lsf + (0.05 * log_model_norm_convolved_spec_lsf)
    sigma2_model = log_model_norm_convolved_spec_lsf - (0.05 * log_model_norm_convolved_spec_lsf)

    for n_mask, mask_box in enumerate(mask_dict["mask_for_fit"]):
        if n_mask == 0:
            plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey',
                        label='Mask used for model matching')
        else:
            plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')

    xlim = [8.18, 9.2]
    # xlim = [3600,9800] model_params[2]
    ylim = [0.0, 1.6]

    # for n_mask, mask_box in enumerate(mask_chi2):
    #     if n_mask == 0:
    #         plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='red',
    #                     label='Chi square mask')
    #     else:
    #         plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='red')

    for n_peak, peak in enumerate(matching_params["peaks"]):
        if n_peak == 0:
            width = int(matching_params["properties"]["widths"][np.where(matching_params["peaks"] == peak)][0])  # Use detected width
            start = max(0, peak - width)
            end = min(len(matching_params["chi2_wave_bestfit_0"]) - 1, peak + width)
            plt.axvspan(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][start],
                        wave_arrays['log_std_wave_all'][mask_dict['mask_good']][end],
                        ymin=ylim[0], ymax=ylim[1], alpha=0.2, color='red', label='Chi square mask')
        else:
            width = int(matching_params["properties"]["widths"][np.where(matching_params["peaks"] == peak)][0])  # Use detected width
            start = max(0, peak - width)
            end = min(len(matching_params["chi2_wave_bestfit_0"]) - 1, peak + width)
            plt.axvspan(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][start],
                        wave_arrays['log_std_wave_all'][mask_dict['mask_good']][end],
                        ymin=ylim[0], ymax=ylim[1], alpha=0.2, color='red')

    plt.text((xlim[1] - xlim[0]) * 0.05 + xlim[0], (ylim[1] - ylim[0]) * 0.9 + ylim[0], f'Best-fit model: '
                                                                                        f'Teff = {model_params["model_Teff"]}, '
                                                                                        f'log(g) = {model_params["model_logg"]}, '
                                                                                        f'[Fe/H] = {model_params["model_z"]},'
                                                                                        f'Vel. correction = '
                                                                                        f'{matching_params["vel_shift"]:.2f} km/s',
             size=14)
    plt.text((xlim[1] - xlim[0]) * 0.15 + xlim[0], (ylim[1] - ylim[0]) * 0.82 + ylim[0],
             f'Reduced chi2 = {matching_params["chi2_bestfit"]:.4f}', size=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("wavelength [A]", size=14)
    plt.ylabel("Normalized Flux", size=14)
    show_wl = np.arange(3500, 10000, 500)
    plt.xticks(np.log(show_wl), labels=show_wl.astype(str), size=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower right", fontsize=14)

    plt.subplot(612)
    plt.plot(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][mask_dict['mask_chi2']] +
             matching_params["log_vel_shift"], matching_params["chi2_wave_bestfit"],
             label=f'chi2, threshold for masking chi2 = {matching_params["chi2_threshold"]}; '
                   f'{matching_params["npix_masked"]} pixels were masked', linewidth=1)
    for n_mask, mask_box in enumerate(mask_dict['mask_for_fit']):
        if n_mask == 0:
            plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')
        else:
            plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')
            # mask_chi2[start:end] = True

            # plt.fill_between(log_std_wave_all[mask_good]+log_shift_full, -1,
            #                  1.1*np.max(chi2_wave_bestfit), where=(chi2_wave_bestfit_0 >= chi2_threshold), color='gray', alpha=0.2)

    ylim = [-1, 1.1 * np.max(matching_params["chi2_wave_bestfit"])]

    for peak in matching_params["peaks"]:
        width = int(matching_params["properties"]["widths"][np.where(matching_params["peaks"] == peak)][0])  # Use detected width
        start = max(0, peak - width)
        end = min(len(matching_params["chi2_wave_bestfit_0"]) - 1, peak + width)
        plt.axvspan(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][start],
                    wave_arrays['log_std_wave_all'][mask_dict['mask_good']][end],
                    ymin=ylim[0], ymax=ylim[1], alpha=0.2, color='red')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("wavelength [A]", size=14)
    plt.ylabel("Chi2", size=14)
    show_wl = np.arange(3500, 10000, 500)
    plt.xticks(np.log(show_wl), labels=show_wl.astype(str), size=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.subplot(613)
    plt.plot(wave_arrays['log_std_wave_all'], flux_std_unconv_logscale, linewidth=1)
    plt.plot(wave_arrays['log_std_wave_all'], sigma2, '--', color='grey', lw=0.1)
    plt.plot(wave_arrays['log_std_wave_all'], sigma1, '--', color='grey', lw=0.1)
    plt.fill_between(wave_arrays['log_std_wave_all'], sigma1, sigma2, alpha=0.2, color='blue')

    plt.plot(wave_arrays['log_model_wave_shifted'], log_model_norm_convolved_spec_lsf, alpha=0.7, linewidth=1)
    plt.plot(wave_arrays['log_model_wave_shifted'], sigma2_model, '--', color='grey', lw=0.1)
    plt.plot(wave_arrays['log_model_wave_shifted'], sigma1_model, '--', color='grey', lw=0.1)
    plt.fill_between(wave_arrays['log_model_wave_shifted'], sigma1_model, sigma2_model, alpha=0.2, color='orange')
    # plt.plot(log_std_wave_all-log_shift_b, flux_model_logscale, label='Model shifted')
    for mask_box in mask_dict['mask_for_fit']:
        plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')
    plt.plot(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][mask_dict['mask_chi2']] + matching_params["log_vel_shift"],
             matching_params["chi2_wave_bestfit"] / 100, label='chi2/100', linewidth=1)

    # xlim = [np.min(log_std_wave_all), 8.2]
    # xlim = [8.24, 8.38]
    xlim = [np.log(3900),np.log(4200)]
    # ylim = [0.1,1.6]
    ylim = [-0.1, 1.6]
    # show_wl = np.arange(3700, 4400, 100)
    show_wl = np.arange(3900, 4200, 100)

    for peak in matching_params["peaks"]:
        width = int(matching_params["properties"]["widths"][np.where(matching_params["peaks"] == peak)][0])  # Use detected width
        start = max(0, peak - width)
        end = min(len(matching_params["chi2_wave_bestfit_0"]) - 1, peak + width)
        plt.axvspan(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][start],
                    wave_arrays['log_std_wave_all'][mask_dict['mask_good']][end],
                    ymin=ylim[0], ymax=ylim[1], alpha=0.2, color='red')

    plt.legend(fontsize=14)
    # show_wl = np.arange(5000, 6000, 100)
    plt.xticks(np.log(show_wl), labels=show_wl.astype(str), size=14)
    plt.yticks(fontsize=14)
    plt.text((xlim[1] - xlim[0]) * 0.03 + xlim[0], (ylim[1] - ylim[0]) * 0.9 + ylim[0], 'b channel', size=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("wavelength [A]", size=14)
    plt.ylabel("Normalized Flux", size=14)

    plt.subplot(614)
    plt.plot(wave_arrays['log_std_wave_all'], flux_std_unconv_logscale, label='Observed', linewidth=1)
    # plt.plot(log_model_wave_all, flux_model_logscale, label='Model')
    plt.plot(wave_arrays['log_std_wave_all'], sigma2, '--', color='grey', lw=0.1)
    plt.plot(wave_arrays['log_std_wave_all'], sigma1, '--', color='grey', lw=0.1)
    plt.fill_between(wave_arrays['log_std_wave_all'], sigma1, sigma2, alpha=0.2, color='blue')

    plt.plot(wave_arrays['log_std_wave_all'] + matching_params["log_vel_shift"], log_model_norm_convolved_spec_lsf,
             label='Model shifted', alpha=0.7, linewidth=1)
    plt.plot(wave_arrays['log_model_wave_shifted'], sigma2_model, '--', color='grey', lw=0.1)
    plt.plot(wave_arrays['log_model_wave_shifted'], sigma1_model, '--', color='grey', lw=0.1)
    plt.fill_between(wave_arrays['log_model_wave_shifted'], sigma1_model, sigma2_model, alpha=0.2, color='orange')

    for mask_box in mask_dict['mask_for_fit']:
        plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')
    plt.plot(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][mask_dict['mask_chi2']] + matching_params["log_vel_shift"],
             matching_params["chi2_wave_bestfit"] / 100, label='chi2/100', linewidth=1)
    # plt.axvline(x=np.log(6563), color='grey', linewidth=0.5, linestyle='--')
    # plt.legend()
    # xlim = [8.66, 8.92] #~whole channel
    # xlim = [8.69, 8.8]
    xlim = [np.log(6400), np.log(6600)]
    # ylim = [0.2, 1.5]
    ylim = [-0.1, 1.5]
    # show_wl = np.arange(5700, 6700, 100)
    show_wl = np.arange(6400, 6600, 100)

    for peak in matching_params["peaks"]:
        width = int(matching_params["properties"]["widths"][np.where(matching_params["peaks"] == peak)][0])  # Use detected width
        start = max(0, peak - width)
        end = min(len(matching_params["chi2_wave_bestfit_0"]) - 1, peak + width)
        plt.axvspan(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][start],
                    wave_arrays['log_std_wave_all'][mask_dict['mask_good']][end],
                    ymin=ylim[0], ymax=ylim[1], alpha=0.2, color='red')
    plt.xticks(np.log(show_wl), labels=show_wl.astype(str), size=14)
    plt.yticks(fontsize=14)
    plt.text((xlim[1] - xlim[0]) * 0.03 + xlim[0], (ylim[1] - ylim[0]) * 0.9 + ylim[0], 'r channel', size=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("wavelength [A]", size=14)
    plt.ylabel("Normalized Flux", size=14)

    plt.subplot(615)
    plt.plot(wave_arrays['log_std_wave_all'], flux_std_unconv_logscale, label='Observed', linewidth=1)
    plt.plot(wave_arrays['log_std_wave_all'], sigma2, '--', color='grey', lw=0.1)
    plt.plot(wave_arrays['log_std_wave_all'], sigma1, '--', color='grey', lw=0.1)
    plt.fill_between(wave_arrays['log_std_wave_all'], sigma1, sigma2, alpha=0.2, color='blue')
    # plt.plot(log_model_wave_all, flux_model_logscale, label='Model')

    plt.plot(wave_arrays['log_model_wave_shifted'], log_model_norm_convolved_spec_lsf, label='Model shifted', alpha=0.7,
             linewidth=1)
    plt.plot(wave_arrays['log_model_wave_shifted'], sigma2_model, '--', color='grey', lw=0.1)
    plt.plot(wave_arrays['log_model_wave_shifted'], sigma1_model, '--', color='grey', lw=0.1)
    plt.fill_between(wave_arrays['log_model_wave_shifted'], sigma1_model, sigma2_model, alpha=0.2, color='orange')

    for mask_box in mask_dict['mask_for_fit']:
        plt.axvspan(np.log(mask_box[0]), np.log(mask_box[1]), alpha=0.2, color='grey')
    plt.plot(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][mask_dict['mask_chi2']] + matching_params["log_vel_shift"],
             matching_params["chi2_wave_bestfit"] / 100, label='chi2/100', linewidth=1)
    # if do_mask:
    #     plt.fill_between(log_std_wave_all[mask_good] + log_shift_full, chi2_wave_bestfit.min(),
    #                  chi2_wave_bestfit.max(), where=(chi2_wave_bestfit_0 >= chi2_threshold), color='gray',
    #                  alpha=0.2,
    #                  label="Masked Region")
    # plt.legend()
    # xlim = [9.035, 9.1]
    xlim = [np.log(8400), np.log(8650)]
    ylim = [-0.1, 1.5]
    # show_wl = np.arange(8300, 9500, 100)
    show_wl = np.arange(8400, 8650, 100)

    for peak in matching_params["peaks"]:
        width = int(matching_params["properties"]["widths"][np.where(matching_params["peaks"] == peak)][0])  # Use detected width
        start = max(0, peak - width)
        end = min(len(matching_params["chi2_wave_bestfit_0"]) - 1, peak + width)
        plt.axvspan(wave_arrays['log_std_wave_all'][mask_dict['mask_good']][start],
                    wave_arrays['log_std_wave_all'][mask_dict['mask_good']][end],
                    ymin=ylim[0], ymax=ylim[1], alpha=0.2, color='red')

    plt.xticks(np.log(show_wl), labels=show_wl.astype(str), size=14)
    plt.yticks(fontsize=14)
    plt.text((xlim[1] - xlim[0]) * 0.03 + xlim[0], (ylim[1] - ylim[0]) * 0.9 + ylim[0], 'z channel', size=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("wavelength [A]", size=14)
    plt.ylabel("Normalized Flux", size=14)
    # plt.ylabel(size=14)

    plt.subplot(616)
    # plt.plot(std_wave_all, normalized_std_on_gaia_cont_single_tmp, linewidth=1.5,
    #          label='Continuum from GAIA spectrum * observed absorptions')
    # plt.plot(std_wave_all, model_convolved_spec_lsf * np.median(model_to_gaia), label='Best-fit model',
    #          linewidth=1.5, alpha=0.7)
    # plt.plot(std_wave_all, std_norm_unconv)
    # plt.plot(std_wave_all, model_shifted_norm_convolved_spec_lsf)
    # plt.plot(std_wave_all, std_norm_unconv/model_shifted_norm_convolved_spec_lsf, label='Observed normalised/model normalised')
    plt.plot(wave_arrays['std_wave_all'], model_flux_resampled * np.mean(matching_params["model_to_gaia"]), label='Model', linewidth=1)
    plt.plot(wave_arrays['std_wave_all'], model_convolved_to_gaia * np.mean(matching_params["model_to_gaia"]), label='Model, convolved with Gaia LSF',
             linewidth=1)
    plt.plot(wave_arrays['std_wave_all'], stdflux,
             label=f'Gaia, Teff={gaia_params["gaia_Teff"]:.0f}, logg={gaia_params["gaia_logg"]:.1f}, [Fe/H]={gaia_params["gaia_z"]:.1f}',
             linewidth=1)
    # plt.plot(log_std_wave_all+log_shift_full, log_model_norm_convolved_spec_lsf, label='Model shifted', alpha=0.7)
    # plt.plot(log_std_wave_all+log_shift_z, flux_model_logscale, label='Model shifted')
    # for mask_box in mask_for_fit:
    #    plt.axvspan((mask_box[0]), (mask_box[1]), alpha=0.2, color='grey')
    plt.legend(fontsize=14)
    plt.xlim(3500, 10000)
    # plt.gca().set_ylim(bottom=0)
    # plt.ylim(0.5, 1.6)
    plt.xlabel("wavelength [A]", size=14)
    plt.ylabel("Flux, erg/s/cm^2/A", size=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.show()
    # fig_path = in_rss[0]
    fig_path = f"{fig_path.replace('lvm-hobject-b', 'lvm-hobject')}"
    save_fig(plt.gcf(), product_path=fig_path, to_display=False, figure_path="qa/model_matching",
                 label=f"matching_std{fiber_params['i']}")

    return

def calc_sensitivity_from_model(wl, obs_spec, spec_lsf, model_flux=[], model_to_gaia_median=1, model_log_shift=0):
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
    # models_dir = os.path.join(MASTERS_DIR, "stellar_models")

    n_steps = int((9850-3550) / 0.05) + 1
    model_wave = np.linspace(3550, 9850, n_steps)

    # apply the model shift relative to observed spectra in log space
    log_model_wave, flux_model_logscale = linear_to_logscale(model_wave, model_flux)
    flux_model_shifted = logscale_to_linear(model_wave, log_model_wave, flux_model_logscale, shift=model_log_shift)

    #resample model to the same step
    model_flux_resampled = np.interp(wl, model_wave, flux_model_shifted)
    spec_lsf = np.sqrt(spec_lsf**2 - 0.3**2)/0.5  # as model spectra were already convolved with lsf=0.3, we need to account for this

    # convolve model to spec lsf after vel. shift
    model_convolved_spec_lsf = fluxcal.lsf_convolve(model_flux_resampled, spec_lsf, wl)
    # first multiply to model_to_gaia_median to be able to compare sens. curve with STD and SCI methods
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
        # lsf = rss._lsf[fibidx[0], :]
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
        res[f"STD{nn}SEN"] = s(w).astype(np.float32) * u.Unit("erg / (ct cm2)")

        # caluculate SDSS g band magnitudes for QC
        mAB_std = np.round(fluxcal.spec_to_LVM_mAB(channel, w, stdflux), 2)
        mAB_obs = np.round(fluxcal.spec_to_LVM_mAB(channel, w[np.isfinite(spec)], spec[np.isfinite(spec)]), 2)
        # update input file header
        label = channel.upper()
        rss.setHdrValue(f"STD{nn}{label}AB", mAB_std, f"Gaia AB mag in {channel}-band")
        rss.setHdrValue(f"STD{nn}{label}IN", mAB_obs, f"obs AB mag in {channel}-band")
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
    secz = header["SCIAM"]

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

    # filter the GAIA stars to avoid multiple stars in a single fiber
    # locate the science ifu fibers the stars are in
    fibs = np.zeros(len(calibrated_spectra)) - 1
    for i in range(len(calibrated_spectra)):
        data = r[i]
        d = np.sqrt((data['ra']-scifibs['ra'])**2 + (data['dec']-scifibs['dec'])**2) # in degrees
        fib = np.where(d<r_spaxel)[0] # there can only be zero or one fiber with a distance cut smaller than a fiber diameter
        if fib.size > 0:
            fibs[i] = fib

    # locate the science ifu fibers the stars are in
    for i in range(len(calibrated_spectra)):
        data = r[i]
        d = np.sqrt((data['ra']-scifibs['ra'])**2 + (data['dec']-scifibs['dec'])**2) # in degrees
        fib = np.where(d<r_spaxel)[0] # there can only be zero or one fiber with a distance cut smaller than a fiber diameter
        if fib.size > 0:
            # skip if the there are multiple stars in this fiber
            assert(fibs[i] != -1)
            if np.count_nonzero(fibs == fib) > 1:
                log.info(f"dropping gaia star {data['source_id']} in fiber {fib}, multiple stars")
                continue
            # if we found a single star in a fiber
            gflux = calibrated_spectra.iloc[i].flux

            fibidx = scifibs['fiberid'][fib] - 1

            # skip star if the fiber is dead
            if rss._mask[fibidx[0]].all():
                continue

            dmin = d[fib] * 3600 # convert to arcsec

            log.info(f"science fiberid '{scifibs['fiberid'][fib][0]}', star '{data['source_id']}', secz '{secz:.2f}'")

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
            sens *= np.interp(obswave, mean_sens[channel]['wavelength'], mean_sens[channel]['sens'])
            res_sci[f"SCI{i+1}SEN"] = sens.astype(np.float32) * u.Unit("erg / (ct cm2)")
            # reject sensitivity that yield negative instrumental magnitude
            if lvmflux <= 0:
                res_sci[f"SCI{i+1}SEN"][:] = np.nan

            mAB_std = np.round(fluxcal.spec_to_LVM_mAB(channel, gwave, gflux), 2)
            mAB_obs = np.round(fluxcal.spec_to_LVM_mAB(channel, obswave, obsflux), 2)
            # update input file header
            cam = channel.upper()
            rss.setHdrValue(f"SCI{i+1}{cam}AB", mAB_std, f"Gaia AB mag in {channel}-band")
            rss.setHdrValue(f"SCI{i+1}{cam}IN", mAB_obs, f"obs AB mag in {channel}-band")
            rss.setHdrValue(f"SCI{i+1}ID", data['source_id'], f"field star {i+1} Gaia source ID")
            rss.setHdrValue(f"SCI{i+1}FIB", scifibs['fiberid'][fib][0], f"field star {i+1} fiber id")
            rss.setHdrValue(f"SCI{i+1}RA", data['ra'], f"field star {i+1} RA")
            rss.setHdrValue(f"SCI{i+1}DE", data['dec'], f"field star {i+1} DEC")
            log.info(f"AB mag in LVM_{channel}: Gaia {mAB_std:.2f}, instrumental {mAB_obs:.2f}")

            # calibrate and plot against the stars for debugging:
            if plot:
                plt.plot(obswave, np.interp(obswave, gwave, gflux)/obsflux, '.',
                         color=colors[i%len(colors)] , markersize=2, zorder=-999)
                plt.plot(obswave, res_sci[f"SCI{i+1}SEN"], color=colors[i%len(colors)], linewidth=2)

    return rss, res_sci


def fluxcal_standard_stars(in_rss, plot=True, GAIA_CACHE_DIR=None):
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
    res_std = Table(np.full(w.size, np.nan, dtype=list(zip(colnames, ["f8"] * len(colnames)))), units=[u.Unit("erg / (ct cm2)")]*len(colnames))
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

    rms_std = biweight_scale(res_std_pd, axis=1, ignore_nan=True)
    mean_std = biweight_location(res_std_pd, axis=1, ignore_nan=True)

    label = rss._header['CCD']
    channel = label.lower()

    mean_std_band = np.nanmean(mean_std[1000:3000])
    rms_std_band = np.nanmean(rms_std[1000:3000])
    mean_std_band = -999.9 if np.isnan(mean_std_band) else mean_std_band
    rms_std_band = -999.9 if np.isnan(rms_std_band) else rms_std_band
    rss.setHdrValue(f"STDSENM{label}", mean_std_band, f"mean stdstar sensitivity in {channel}")
    rss.setHdrValue(f"STDSENR{label}", rms_std_band, f"mean stdstar sensitivity rms in {channel}")
    log.info(f"Mean stdstar sensitivity in {channel} : {mean_std_band}")

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
    log.info(f'appending FLUXCAL_STD table to {in_rss}')
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
    colnames = [f"SCI{i}SEN" for i in range(1, NSCI_MAX + 1)]
    res_sci = Table(np.full(w.size, np.nan, dtype=list(zip(colnames, ["f8"] * len(colnames)))), units=[u.Unit("erg / (ct cm2)")]*len(colnames))
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
    mean_sci_band = np.nanmean(mean_sci[1000:3000])
    rms_sci_band = np.nanmean(rms_sci[1000:3000])
    mean_sci_band = -999.9 if np.isnan(mean_sci_band) else mean_sci_band
    rms_sci_band = -999.9 if np.isnan(rms_sci_band) else rms_sci_band
    rss.setHdrValue(f"SCISENM{label}", mean_sci_band, f"mean scistar sensitivity in {channel}")
    rss.setHdrValue(f"SCISENR{label}", rms_sci_band, f"mean scistar sensitivity rms in {channel}")
    log.info(f"Mean scistar sensitivity in {channel} : {mean_sci_band}")

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
    log.info(f'appending FLUXCAL_SCI table to {in_rss}')
    rss.set_fluxcal(fluxcal=res_sci, source='sci')
    rss.writeFitsData(in_rss)

    return res_sci, mean_sci, rms_sci, rss

def correct_tellurics(wave, std_spec, lsf, in_rss, chan):
    """
    Do we need airmass correction?
    :param std_spec:
    :param lsf:
    :return:
    """
    std_telluric_corrected = std_spec.copy()
    log.warning("Tellurics correction is not implemented yet. Skipping correction.")
    return std_telluric_corrected

    telluric_file = os.path.join(os.getenv("LVMCORE_DIR"), "etc", "skytable.fits")
    # telluric_lines = '/Users/jane/Science/LVMFluxCalib/notebooks/atmabs.txt'  # wavelength regions with Telluric
    # absorptions based on KPNO data (unknown source) with a 1% transmission threshold this file is used as a mask for
    # the fit of standard stars - from Alfredo.
    # https://github.com/desihub/desispec/blob/main/py/desispec/data/arc_lines/telluric_lines.txt
    # telluric_lines_tab = Table.read(telluric_lines, format='ascii.fixed_width_two_line')

    with fits.open(telluric_file) as hdul:
        data = hdul[1].data
        # hdr = hdul[1].header
    telluric_table = Table(data)
    telluric_table['lam'] *= 10
    tell_continuum = ndimage.filters.median_filter(telluric_table.as_array()['trans'], int(1500 / 1), mode="nearest")
    telluric_table['trans_norm'] = telluric_table['trans'] / tell_continuum

    # if chan == 'b':
    #     xlim = [3600,5800]
    # elif chan == 'r':
    #     xlim = [6850,7050]
    # elif chan == 'z':
    #     xlim = [7580,7720]

    # fig = plt.figure(figsize=(15, 10))
    # plt.subplot(211)
    # plt.plot(telluric_table['lam'], telluric_table['trans_norm'], label='normalized tellurics convolved with LVM LSF')
    # plt.xlabel('Wavelength, $\AA$')
    # plt.ylabel('Flux')
    # plt.xlim(xlim)
    # for mask_box in telluric_lines_tab:
    #     plt.axvspan((mask_box[0]), (mask_box[1]), alpha=0.2, color='grey')
    #
    # plt.subplot(212)
    # plt.plot(wave, std_spec)
    # plt.xlabel('Wavelength, $\AA$')
    # plt.ylabel('Flux')
    # plt.xlim(xlim)
    # for mask_box in telluric_lines_tab:
    #     plt.axvspan((mask_box[0]), (mask_box[1]), alpha=0.2, color='grey')

    # fig_path = in_rss[0]
    # fig_path = f"{fig_path.replace('lvm-hobject-b', 'lvm-hobject')}"
    # save_fig(plt.gcf(), product_path=fig_path, to_display=False, figure_path="qa/telluric_correction",
    #          label=f"matching_std{0}_tmp")

    return std_telluric_corrected


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
