# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Jan 25, 2023
# @Filename: fluxCalMethod
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM


import os
import time
# from os import listdir
# from os.path import isfile, join
import numpy as np
from scipy import interpolate
from scipy import ndimage
# from scipy.ndimage import median_filter
from scipy.signal import find_peaks
# import re
import pandas as pd

from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from astropy import units as u
from astropy.stats import biweight_location, biweight_scale
from astropy.table import Table
from astropy.io import fits
from astroquery.gaia import Gaia

from lmfit import minimize, Parameters

from lvmdrp.core.rss import RSS, loadRSS, lvmFFrame
from lvmdrp.core.spectrum1d import Spectrum1D
import lvmdrp.core.fluxcal as fluxcal
from lvmdrp.core.sky import get_sky_mask_uves, get_z_continuum_mask
from lvmdrp import log

from lvmdrp.core.plot import plt, create_subplots, save_fig
from lvmdrp.core.constants import MASTERS_DIR

description = "provides flux calibration tasks"

# Telluric-free continuum regions for PWV fitting (wavelengths in Angstroms)
# Regions between these windows contain telluric absorption (O2, H2O bands)
TELLURIC_FREE_REGIONS = {
    'r': [[6750, 6860], [7080, 7145], [7400, 7460]],  # O2 B-band ~6860-7080, H2O ~7145-7400
    'z': [[7780, 7860], [8050, 8085], [8440, 8480]],  # O2 A-band ~7600-7780, H2O ~8085-8440
}


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
    sci_secz = fframe._header["SCIAM"]
    skye_secz = fframe._header["SKYEAM"]
    skyw_secz = fframe._header["SKYWAM"]

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
        sens_ave_sci = sens_ave
        sens_ave_skye = sens_ave
        sens_ave_skyw = sens_ave

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

        # Incorporate into the sensitivity array the telluric bands absorption
        # for estimated average PWV and given zenith angle

        # Read median PWV from header
        pwv = fframe._header.get("PWV_MED", None)

        if pwv is not None and np.isfinite(pwv):
            log.info(f"Applying telluric correction with PWV = {pwv:.2f} mm")

            # Initialize TelluricCalculator and compute transmission
            telluric_corrector = fluxcal.TelluricCalculator()

            # Use median LSF across all fibers for the telluric correction (in Angstroms)
            # TODO: LSF should be used per fiber
            lsf_median = np.nanmedian(fframe._lsf, axis=0)

            # Compute telluric transmission matched to data wavelength grid with LSF convolution
            # LSF is passed in wavelength units (Angstroms)
            # TODO: need to use LSF per fiber
            telluric_trans_sci = telluric_corrector.match_to_data(fframe._wave, lsf_median, pwv, airmass=sci_secz, lsf_in_wavelength=True)
            telluric_trans_skye = telluric_corrector.match_to_data(fframe._wave, lsf_median, pwv, airmass=skye_secz, lsf_in_wavelength=True)
            telluric_trans_skyw = telluric_corrector.match_to_data(fframe._wave, lsf_median, pwv, airmass=skyw_secz, lsf_in_wavelength=True)

            # Divide sensitivity curve by atmospheric molecular transmission
            sens_ave_sci = sens_ave / telluric_trans_sci
            sens_ave_skye = sens_ave / telluric_trans_skye
            sens_ave_skyw = sens_ave / telluric_trans_skyw
            # sens_arr_sci = sens_arr / telluric_trans_sci[:, np.newaxis] # Not used for the moment

        else:
            log.warning(
                "PWV_MED not found in header or invalid, skipping correction "
                "of sensitivity curve for telluric absorption")

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
        sens_ave_sci = sens_ave
        sens_ave_skye = sens_ave
        sens_ave_skyw = sens_ave

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
        fframe._data *= sens_ave_sci * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        fframe._error *= sens_ave_sci * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        if fframe._sky is not None:
            # TODO: NEED TO UNDERSTAND: what sensetivity curve and airmass (secz) should be used for sky frame
            fframe._sky *= sens_ave_sci * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        if fframe._sky_error is not None:
            fframe._sky_error *= sens_ave_sci * 10 ** (0.4 * ext * (sci_secz)) / exptimes[:, None]
        if fframe._sky_east is not None:
            fframe._sky_east *= sens_ave_skye * 10 ** (0.4 * ext * (skye_secz)) / exptimes[:, None]
        if fframe._sky_east_error is not None:
            fframe._sky_east_error *= sens_ave_skye * 10 ** (0.4 * ext * (skye_secz)) / exptimes[:, None]
        if fframe._sky_west is not None:
            fframe._sky_west *= sens_ave_skyw * 10 ** (0.4 * ext * (skyw_secz)) / exptimes[:, None]
        if fframe._sky_west_error is not None:
            fframe._sky_west_error *= sens_ave_skyw * 10 ** (0.4 * ext * (skyw_secz)) / exptimes[:, None]
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
    std_spectra_orig_all_bands = []
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

        # Load sky emission line mask (telluric bands are NOT masked - handled in model_selection)
        channel = rss_tmp._header['CCD']
        m = get_sky_mask_uves(w[b], width=width)

        master_sky = rss_tmp.eval_master_sky()

        # iterate over standard stars
        std_spectra = []  # contains original std spectra for all stars in each band
        std_spectra_orig = []
        normalized_spectra = []
        normalized_spectra_unconv = []
        std_errors = []
        lsf = []
        fibers = []
        gaia_ids = []
        zenith_angles = []
        nns = []

        for s in stds:
            nn, fiber, gaia_id, exptime, secz, zenith_angle = s  # unpack standard star tuple

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
            zenith_angles.append(zenith_angle)
            nns.append(nn)
            # check_bad_fluxes[f'good_flux_{b}'].append(True)

            spec_orig = (rss_tmp._data[fibidx[0],:] - master_sky._data[fibidx[0],:]) / exptime

            # Interpolate over bright sky emission lines and bad pixels
            # Note: telluric bands are NOT masked here - telluric correction is applied later in model_selection()
            mask_bad = ~np.isfinite(spec_orig)
            spec_tmp = fluxcal.interpolate_mask(w_tmp, spec_orig, m | mask_bad, fill_value="extrapolate")

            # extinction correction
            # load extinction curve
            # Note that we assume a constant extinction curve here!
            txt = np.genfromtxt(os.getenv("LVMCORE_DIR") + "/etc/lco_extinction.txt")
            lext, ext = txt[:, 0], txt[:, 1]
            ext = np.interp(w_tmp, lext, ext)

            # correct for extinction
            spec_ext_corr = spec_tmp.copy()
            spec_orig_ext_corr = spec_orig.copy()
            extinction_correction = 10 ** (0.4 * ext * secz)
            spec_ext_corr *= extinction_correction
            spec_orig_ext_corr *= extinction_correction
            pxsize = abs(np.nanmedian(w_tmp - np.roll(w_tmp, -1)))
            lsf_conv = np.sqrt(np.clip(2.3 ** 2 - lsf_tmp ** 2, 0.1, None))/pxsize  # as model spectra were already convolved with lsf=2.3 A,
            # we need to degrade our observed std spectra. Also, convert it to pixels
            mask_bad = ~np.isfinite(spec_tmp)
            mask_lsf = ~np.isfinite(lsf_conv)
            lsf_conv_interpolated = fluxcal.interpolate_mask(w_tmp, lsf_conv, mask_lsf, fill_value="extrapolate")

            # # degrade observed std spectra
            spec_tmp_convolved = fluxcal.lsf_convolve_fast(spec_ext_corr, lsf_conv_interpolated)

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
            std_spectra_orig.append(spec_orig_ext_corr)

        normalized_spectra_all_bands.append(normalized_spectra) # normalized std spectra degraded to 2.3A for all
                                                                        # standards and all channels together
        normalized_spectra_unconv_all_bands.append(normalized_spectra_unconv)
        std_errors_all_bands.append(std_errors)
        lsf_all_bands.append(lsf) # initial std spec LSF for all standards and all channel together
        std_spectra_all_bands.append(std_spectra) # corrected for extinction
        std_spectra_orig_all_bands.append(std_spectra_orig) # original std spectra without masking tellurics
        fibers_all_bands.append(fibers)
        gaia_ids_all_bands.append(gaia_ids)
        nns_all_bands.append(nns)

    return (
        w,
        nns_all_bands[0],
        gaia_ids_all_bands[0],
        fibers,
        std_spectra_all_bands,
        normalized_spectra_unconv_all_bands,
        normalized_spectra_all_bands,
        std_errors_all_bands,
        lsf_all_bands,
        std_spectra_orig_all_bands,
        zenith_angles,
        stds
    )


def _residual_function(params, y_obs, valid_mask, za, telluric_corrector,
                       waves, lsfs, return_components=False):
    """Compute residuals between observed and model transmission."""

    pwv = params['pwv'].value

    model_r = telluric_corrector.match_to_data(waves[0], lsfs[0], pwv, za)
    model_z = telluric_corrector.match_to_data(waves[1], lsfs[1], pwv, za)

    resid_r = y_obs[0] - model_r
    resid_z = y_obs[1] - model_z

    if return_components:
        return resid_r, resid_z, model_r, model_z

    resid = np.concatenate((resid_r[valid_mask[0]], resid_z[valid_mask[1]]))
    return resid


def _qa_plot_pwv_calculation(fig_out, wave_channels, y_data, model_trans, residuals,
                             obs_ratio, poly_continuum, regions_info,
                             result, zenith_angle, rms, sky_masks=None,
                             save_png=False, std_info=None):
    """Create QA plot for PWV calculation. Pure visualization - all data pre-computed."""

    # Ensure output directory exists and prepare file paths
    output_path = Path(fig_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = output_path.with_suffix('.html')
    png_path = output_path.with_suffix('.png')

    # Create figure with two panels
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=False,
                        vertical_spacing=0.075,
                        row_heights=[0.5, 0.5],
                        subplot_titles=('PWV fitting', 'Continuum Fitting'))

    channel_names = ['r', 'z']
    data_color_masked = 'black'
    data_color_full = 'orange'
    model_color = 'red'
    residual_color = 'gray'

    # Upper panel: PWV fitting
    for ich in range(2):
        wave_ch = wave_channels[ich]

        # Plot good pixels only (black)
        y_good = y_data[ich].copy()
        if sky_masks is not None:
            y_good[sky_masks[ich]] = np.nan
        fig.add_trace(go.Scatter(x=wave_ch, y=y_good, mode='lines',
                                 line=dict(color=data_color_masked, width=1),
                                 name='Data' if ich == 0 else None,
                                 showlegend=(ich == 0)), row=1, col=1)

        # Plot masked pixels only (orange), with edge pixels for visual continuity
        if sky_masks is not None:
            # Expand mask by 1 pixel on each edge for visual continuity
            mask_expanded = sky_masks[ich].copy()
            mask_expanded[1:] |= sky_masks[ich][:-1]   # add left neighbor
            mask_expanded[:-1] |= sky_masks[ich][1:]   # add right neighbor
            y_masked = y_data[ich].copy()
            y_masked[~mask_expanded] = np.nan
            fig.add_trace(go.Scatter(x=wave_ch, y=y_masked, mode='lines',
                                     line=dict(color=data_color_full, width=1),
                                     name='Masked (possibly sky lines cotaminated)' if ich == 0 else None,
                                     showlegend=(ich == 0)), row=1, col=1)

        # Plot best-fit model
        fig.add_trace(go.Scatter(x=wave_ch, y=model_trans[ich], mode='lines',
                                 line=dict(color=model_color, width=1.5),
                                 name='Best-fit' if ich == 0 else None,
                                 showlegend=(ich == 0)), row=1, col=1)

        # Plot residuals
        fig.add_trace(go.Scatter(x=wave_ch, y=residuals[ich], mode='lines',
                                 line=dict(color=residual_color, width=1),
                                 name='Resid.' if ich == 0 else None,
                                 showlegend=(ich == 0)), row=1, col=1)

    # Add zero line
    fig.add_hline(y=0, line=dict(color='indigo', width=1, dash='dot'), row=1, col=1)

    # Lower panel: continuum fitting
    for ich in range(2):
        wave_ch = wave_channels[ich]

        # Plot good pixels only (black)
        obs_good = obs_ratio[ich].copy()
        if sky_masks is not None:
            obs_good[sky_masks[ich]] = np.nan
        fig.add_trace(go.Scatter(x=wave_ch, y=obs_good,
                                 mode='lines', showlegend=(ich == 0),
                                 line=dict(color=data_color_masked, width=1),
                                 name='Obs/Stellar model' if ich == 0 else None,
                                 legend='legend2'), row=2, col=1)

        # Plot masked pixels only (orange), with edge pixels for visual continuity
        if sky_masks is not None:
            mask_expanded = sky_masks[ich].copy()
            mask_expanded[1:] |= sky_masks[ich][:-1]   # add left neighbor
            mask_expanded[:-1] |= sky_masks[ich][1:]   # add right neighbor
            obs_masked = obs_ratio[ich].copy()
            obs_masked[~mask_expanded] = np.nan
            fig.add_trace(go.Scatter(x=wave_ch, y=obs_masked,
                                     mode='lines', showlegend=(ich == 0),
                                     line=dict(color=data_color_full, width=1),
                                     name='Masked (possibly sky lines cotaminated)' if ich == 0 else None,
                                     legend='legend2'), row=2, col=1)

        # Plot polynomial continuum fit
        fig.add_trace(go.Scatter(x=wave_ch, y=poly_continuum[ich],
                                 mode='lines', showlegend=(ich == 0),
                                 line=dict(color=model_color, width=2),
                                 name='Continuum fit' if ich == 0 else None,
                                 legend='legend2'), row=2, col=1)

        # Mark continuum regions
        for region_idx, (wmin, wmax) in enumerate(regions_info[ich]):
            fig.add_vrect(x0=wmin, x1=wmax, fillcolor='gray', opacity=0.1,
                          line_width=0, row=2, col=1, name='Poly fit regions', legend='legend2')

    fig.update_xaxes(title_text="Wavelength (Å)", row=2, col=1)
    fig.update_yaxes(title_text="Obs. spec. / (Template * Mpoly)", row=1, col=1)
    fig.update_yaxes(title_text="Obs. spec. / Template", row=2, col=1)
    fig.update_yaxes(range=[-4 * rms, 1 + 4 * rms], row=1, col=1)

    fig.update_layout(
        legend=dict(orientation='h', xanchor='left', yanchor='bottom',
                    x=0.0, y=0.98, bgcolor='rgba(255, 255, 255, 0.9)',
                    font=dict(size=10)),
        legend2=dict(orientation='h', xanchor='left', yanchor='top',
                     x=0.0, y=0.465, bgcolor='rgba(255, 255, 255, 0.9)',
                     font=dict(size=10)),
        plot_bgcolor='rgba(245, 245, 245, 1)',
        paper_bgcolor='white'
    )

    # Add title with fit results
    pwv_val = result.params['pwv'].value
    pwv_err = result.params['pwv'].stderr if result.params['pwv'].stderr is not None else 0.0
    s_idx, s_fiber, s_gaia_id, s_exptime, s_secz, s_zenith_angle = std_info
    title_text = (
        f"#{s_idx} | {s_fiber} | GAIA ID: {s_gaia_id} | T<sub>EXP</sub> = {s_exptime:.1f}s | "
        f"secz = {s_secz:.2f} | Zenith Angle = {zenith_angle:.1f}°<br>"
        f"<span style='color:red;'>PWV = {pwv_val:.3f} ± {pwv_err:.3f} mm</span> | "
        f"N function evaluations: {result.nfev}"
    )

    fig.update_layout(margin=dict(l=60, r=20, t=60, b=60),
                      title=dict(text=title_text,
                                 x=0.5, xanchor='center',
                                 y=0.987, yanchor='top',
                                 font=dict(size=14, weight=600)),
                      width=1200, height=800)

    # Save to HTML
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    log.info(f"PWV QA plot saved to {html_path}")

    if save_png:
        # Save to PNG using kaleido (if available) which is very slow!
        png_path = output_path.with_suffix('.png')
        try:
            fig.write_image(str(png_path), format='png', width=1200, height=800, scale=1)
            log.info(f"PWV QA plot saved to {png_path}")
        except Exception as e:
            log.warning(f"Could not save PNG file: {e}. Only HTML saved.")


def calc_pwv(wave, spec, lsf, stellar_model, telluric_corrector,
             pwv_init=3.0, pwv_bounds=(0.5, 50.0), zenith_angle=0.0,
             continuum_poly_deg=2, sky_mask_width=3, fig_out=None, std_info=None):
    """
    Estimate precipitable water vapor (PWV) by fitting telluric absorption bands.
    Fits standard star spectra in r and z channels using O2 and H2O absorption
    regions to determine optimal PWV value.

    Args:
        wave: Wavelength arrays for [r_channel, z_channel]
        spec: Observed spectra for [r_channel, z_channel]
        lsf: Line spread functions for [r_channel, z_channel]
        stellar_model: Model stellar spectra for [r_channel, z_channel]
        telluric_corrector: TelluricCalculator instance for telluric transmission calculations
        pwv_init: Initial PWV guess in mm (default: 3.0)
        pwv_bounds: PWV fitting bounds in mm (default: (0.5, 50.0))
        zenith_angle: Observation zenith angle in degrees (default: 0.0)
        continuum_poly_deg: Polynomial degree for continuum fitting (default: 2)
        sky_mask_width: Width for sky emission line masking (default: 3)

    Returns:
        tuple: (pwv_value, pwv_error) in mm
    """

    # Use module-level telluric-free regions
    regions_info = [TELLURIC_FREE_REGIONS['r'], TELLURIC_FREE_REGIONS['z']]

    # Set wavelength range for efficiency (only compute transmission where needed)
    wave_min = TELLURIC_FREE_REGIONS['r'][0][0] - 50
    wave_max = TELLURIC_FREE_REGIONS['z'][-1][1] + 50
    telluric_corrector.set_wave_range(wave_min, wave_max)

    # Build wavelength masks for continuum fitting and full regions
    continuum_masks = []
    full_masks = []
    sky_masks_full = []  # sky masks for full fitting regions

    for ich, regions in enumerate(regions_info):
        wave_ch = wave[ich]

        # Full region spanning all continuum windows
        full_mask = (wave_ch >= regions[0][0]) & (wave_ch <= regions[-1][1])

        # Continuum regions only
        cont_mask = np.zeros(len(wave_ch), dtype=bool)
        for wmin, wmax in regions:
            cont_mask |= (wave_ch >= wmin) & (wave_ch <= wmax)

        # Sky emission line mask (True = masked/bad pixel)
        sky_mask = get_sky_mask_uves(wave_ch, width=sky_mask_width)

        # Apply sky mask to continuum mask (exclude sky lines from continuum fitting)
        cont_mask = cont_mask & ~sky_mask

        continuum_masks.append(cont_mask)
        full_masks.append(full_mask)
        sky_masks_full.append(sky_mask[full_mask])  # sky mask for full region only

    # Compute normalized absorption depth for each channel
    y_data, valid_masks, poly_continuum_fits, obs_stellar_ratios = [], [], [], []

    for ich in range(2):
        cont_msk = continuum_masks[ich]
        full_msk = full_masks[ich]

        # Observed / stellar ratio (for continuum fitting and QA plot)
        obs_stellar_ratio = spec[ich][full_msk] / stellar_model[ich][full_msk]
        obs_stellar_ratios.append(obs_stellar_ratio)

        # Fit polynomial to continuum ratio (only in continuum regions)
        ratio_cont = spec[ich][cont_msk] / stellar_model[ich][cont_msk]
        finite_mask = np.isfinite(ratio_cont)
        poly_coef = np.polyfit(wave[ich][cont_msk][finite_mask],
                               ratio_cont[finite_mask],
                               continuum_poly_deg)
        poly_continuum = np.polyval(poly_coef, wave[ich][full_msk])
        poly_continuum_fits.append(poly_continuum)

        # Normalized absorption depth (1.0 = no absorption)
        normalized_depth = obs_stellar_ratio / poly_continuum
        y_data.append(normalized_depth)
        # Valid mask: finite values AND not masked by sky lines
        valid_masks.append(np.isfinite(normalized_depth) & ~sky_masks_full[ich])

    # Setup and run PWV optimization
    params = Parameters()
    params.add('pwv', value=pwv_init, min=pwv_bounds[0], max=pwv_bounds[1])

    wave_channels = [wave[0][full_masks[0]], wave[1][full_masks[1]]]
    lsf_channels = [lsf[0][full_masks[0]], lsf[1][full_masks[1]]]

    result = minimize(_residual_function, params, method='leastsq',
                      args=(y_data, valid_masks, zenith_angle, telluric_corrector,
                            wave_channels, lsf_channels))

    if fig_out is not None:
        # Get best-fit model and residuals
        resid_r, resid_z, model_r, model_z = _residual_function(
            result.params, y_data, valid_masks, zenith_angle,
            telluric_corrector, wave_channels, lsf_channels,
            return_components=True
        )

        # Prepare data for QA plot
        model_trans = [model_r, model_z]
        residuals = [resid_r, resid_z]

        # Calculate RMS for Y-axis scaling
        rms = np.nanstd(np.concatenate(residuals))

        # Call pure visualization function
        _qa_plot_pwv_calculation(
            fig_out, wave_channels, y_data, model_trans, residuals,
            obs_stellar_ratios, poly_continuum_fits, regions_info,
            result, zenith_angle, rms, sky_masks=sky_masks_full,
            save_png=False, std_info=std_info)

    # Reset wavelength range to full model
    telluric_corrector.reset_wave_range()

    return result.params['pwv'].value, result.params['pwv'].stderr


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

    with fits.open(name=models_dir + '/lvm-models_AMBRE_for_LVM_3000_11000.fits') as model:
        model_good = model['FLUX'].data
        model_wave = model['WAVE'].data
        model_norm = model['FLUX_NORM'].data
        model_norm_wave = model['WAVE_LOG'].data
        model_info = pd.DataFrame(model['MODEL_INFO'].data)
    model_names = model_info['Model_name'].to_list()
    n_models = len(model_names)
    log.info(f'Number of models: {n_models}')

    # Initialize TelluricCalculator to handle atmospheric transmission calculations
    skymodel_path = os.path.join(models_dir, 'lvm-model_transmission_Palace_SkyModel_step0.2.fits')
    telluric_corrector = fluxcal.TelluricCalculator(skymodel_path)

    GAIA_CACHE_DIR = "./" if GAIA_CACHE_DIR is None else GAIA_CACHE_DIR
    log.info(f"Using Gaia CACHE DIR '{GAIA_CACHE_DIR}'")

    # Read Calibration GAIA stars table and create index on source_id for quick
    # record retrieval
    # https://sdss-wiki.atlassian.net/wiki/spaces/LVM/pages/14460157/Calibration+Stars
    gaia_stars = Table.read(models_dir + '/lvm-many_Gaia_stars_5-9_ftype_v4.fits', format='fits')
    gaia_stars.add_index('source_id')

    # Prepare the spectra
    (w, nns, gaia_ids, fibers, std_spectra_all_bands, normalized_spectra_unconv_all_bands,
     normalized_spectra_all_bands, std_errors_all_bands, lsf_all_bands,
     std_spectra_orig_all_bands, zenith_angles, std_info) = prepare_spec(in_rss, width=width)

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
    pwv_values, pwv_errors = [], []
    stack_stellar_model, stack_telluric_trans = [], []

    # Loop over standard stars, stitch normalized spectra in brz together
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
        vel_shift_full = log_shift_full * 299792.458
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

        #######################################################################
        # Rebin and convolve bestfit stellar model for all below calculations
        #######################################################################

        # LSF in pixels for brz channels
        lsf_pixels = [
            lsf_all_bands[q][i] / np.diff( fluxcal.edges_from_centers(w[q]) ) for q in range(3)
        ]

        # Account for the velocity shift determined above
        model_wave_ref = model_wave / (1.0 - vel_shift_full / 299792.45)

        stellar_model = [
            fluxcal.rebin_and_convolve(w[q], model_wave_ref, model_flux, lsf_pixels[q]) for q in [0, 1, 2]
        ]
        stack_stellar_model.append(stellar_model)

        #######################################################################
        # PWV calculation
        #######################################################################

        # Path to QA Plotly plot for PWV calculation (without extension)
        fig_out_pwv = in_rss[0].replace('lvm-hobject-b', 'qa/pwv_calc/lvm-hobject').replace('.fits', f"_pwv_std{i}")

        # Prepare data for r and z channels, which will be used to estimate PWV
        spec_rz = [
            std_spectra_orig_all_bands[1][i],
            std_spectra_orig_all_bands[2][i]
        ]

        try:
            t_start = time.time()
            # use only r and z channels
            pwv, pwv_err = calc_pwv(w[1:], spec_rz, lsf_pixels[1:], stellar_model[1:],
                                    telluric_corrector,
                                    zenith_angle=zenith_angles[i], fig_out=fig_out_pwv,
                                    std_info=std_info[i])
            log.info(f"Estimated for star # {i} GAIA ID {gaia_ids[i]} PWV = {pwv:.2f} +/- {pwv_err:.2f} mm ({time.time() - t_start:.2f} sec)")

            # Calculate telluric transmission for all channels
            telluric_trans = [
                telluric_corrector.match_to_data(w[k], lsf_pixels[k], pwv, zenith_angles[i]) for k in range(3)
            ]

        except Exception as e:
            log.error(f"Failed to calculate PWV for star # {i} GAIA ID {gaia_ids[i]}: {e}")
            pwv = np.nan
            pwv_err = np.nan
            telluric_trans = [ np.ones_like(w[k]) for k in range(3) ] # dummy unity values

        pwv_values.append(pwv)
        pwv_errors.append(pwv_err)
        stack_telluric_trans.append(telluric_trans)


        #######################################################################
        # Preparation for QA plots
        #######################################################################

        # resample model to the same step
        model_flux_resampled = np.interp(std_wave_all, model_wave, model_flux)
        good_model_to_std_lsf = np.sqrt(lsf_all ** 2 - 0.3 ** 2) # to degrade good resolution model to std lsf for plots
        # if wave vector provided, it converts LSF from wavelengths to pixels
        model_convolved_spec_lsf = fluxcal.lsf_convolve_fast(model_flux_resampled, good_model_to_std_lsf, std_wave_all)
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

            # Try to get stellar parameters from the local table first
            try:
                # Used indexed column source_id, See where table was read
                gaia_rec = gaia_stars.loc[gaia_ids[i]]
                teff, logg, z = gaia_rec['teff_gspspec'], gaia_rec['logg_gspspec'], gaia_rec['mh_gspspec']
            except KeyError:
                # If entry not found in local table, then call external Gaia service
                job = Gaia.launch_job(f"SELECT teff_gspspec, logg_gspspec, mh_gspspec FROM gaiadr3.astrophysical_parameters WHERE source_id = {gaia_ids[i]} ")
                r = job.get_results()
                teff, logg, z = r['teff_gspspec'][0], r['logg_gspspec'][0], r['mh_gspspec'][0]

        except fluxcal.GaiaStarNotFound as e:
            stdflux = np.full_like(std_wave_all, np.nan)
            teff, logg, z = np.nan, np.nan, np.nan
            model_to_gaia_median.append(np.nan)
            log.warning(f"Gaia star {gaia_ids[i]} not found: {e}")
        finally:
            gaia_flux_interpolated.append(stdflux)
            gaia_Teff.append(teff)
            gaia_logg.append(logg)
            gaia_z.append(z)

        # Skip star with no Gaia parameters
        if np.isnan(teff):
            continue

        # Keep Eugenia's implementation for a reference after a minor bug fix
        # (missing GAIA LSF conversion to pixels).
        # gaia_lsf_pix = gaia_lsf / np.diff(fluxcal.edges_from_centers(std_wave_all))
        # model_convolved_to_gaia = fluxcal.lsf_convolve(model_flux_resampled, gaia_lsf/0.5_pix, gw)
        # model_to_gaia = stdflux / model_convolved_to_gaia
        # model_to_gaia_median.append(np.median(model_to_gaia))


        # In the block below, we rebin the stellar template spectrum to the GAIA
        # wavelength grid, then convolve it with the GAIA LSF.

        # Rebin the model spectrum to the GAIA wavelength grid.
        # The GAIA grid (gw) extends beyond the model range, the extended parts are filled with np.nan.
        model_flux_gaia_rebinned = fluxcal.fluxconserve_rebin(gw, model_wave, model_flux)

        # Replace nan by 0.0 to make possible convolution
        # TODO: must be fixed when stellar templates grid will be extended
        model_flux_gaia_rebinned[~np.isfinite(model_flux_gaia_rebinned)] = 0.0

        # Interpolate GAIA LSF to the extended grid
        gaia_lsf_gw_ang = np.interp(gw, gaia_lsf_table_tmp['wavelength'], gaia_lsf_table_tmp['linewidth'])

        # Convert GAIA LSF in Angstroms to pixels of GAIA spectrum
        gaia_lsf_gw_pix = gaia_lsf_gw_ang / np.diff( fluxcal.edges_from_centers(gw) )
        model_flux_gaia_convolved = fluxcal.lsf_convolve_fast(model_flux_gaia_rebinned, gaia_lsf_gw_pix)

        model2gaia_factor = np.median(gf / model_flux_gaia_convolved)
        model_to_gaia_median.append(model2gaia_factor)


        # prepare dictionaries to plot QA plots for model matching
        fig_path = in_rss[0]
        fiber_params = {'i': i, 'fiber_id': fibers[i]}
        gaia_params = {
            'gaia_id': gaia_ids[i],
            'gaia_Teff': gaia_Teff[i],
            'gaia_logg': gaia_logg[i],
            'gaia_z': gaia_z[i]
        }
        model_params = {
            'model_name': model_names[best_id],
            'model_Teff': model_info['Teff'][best_id],
            'model_logg': model_info['logg'][best_id],
            'model_z': model_info['Z'][best_id]
        }
        matching_params = {
            'vel_shift': vel_shift_full,
            'log_vel_shift': log_shift_full,
            'npix_masked': npix_masked,
            'peaks': peaks,
            'properties': properties,
            'chi2_threshold': chi2_threshold,
            'chi2_bestfit': chi2_bestfit,
            'chi2_wave_bestfit': chi2_wave_bestfit,
            'chi2_wave_bestfit_0': chi2_wave_bestfit_0,
            'model_flux_gaia_convolved': model_flux_gaia_convolved,
            'model2gaia_factor': model2gaia_factor
        }
        mask_dict = {
            'mask_for_fit': mask_for_fit,
            'mask_good': mask_good,
            'mask_chi2': mask_chi2
        }
        wave_arrays = {
            'std_wave_all': std_wave_all,
            'log_std_wave_all': log_std_wave_all,
            'log_model_wave_shifted': log_model_wave_all + log_shift_full,
            'gaia_wave': gw
        }

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
                              model_flux_gaia_convolved=model_flux_gaia_convolved,
                              mask_dict=mask_dict)

    # calculating sensitivity curves by channels
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

            # Telluric absorption correction on unmasked spectrum
            std_telluric_corrected = std_spectra_orig_all_bands[n_chan][i] / stack_telluric_trans[i][n_chan]

            # Mask bad pixels and sky emission lines AFTER telluric correction
            mask_bad = ~np.isfinite(std_telluric_corrected)
            mask_skylines = get_sky_mask_uves(w[n_chan], width=3)
            std_masked = fluxcal.interpolate_mask(
                w[n_chan], std_telluric_corrected, mask_bad | mask_skylines, fill_value="extrapolate"
            )

            # Calculate sensitivity curve using masked spectrum
            sens_tmp = stack_stellar_model[i][n_chan] * model_to_gaia_median[i] / std_masked

            # Filter and interpolate sensitivity curve
            wgood, sgood = fluxcal.filter_channel(w[n_chan], sens_tmp, 3, method='savgol')
            s = interpolate.make_smoothing_spline(wgood, sgood, lam=1e4)
            sens0 = s(w[n_chan]).astype(np.float32)

            # Calculate flux normalization using telluric-corrected masked spectrum
            lvmflux = fluxcal.spec_to_LVM_flux(chan, w[n_chan], std_masked * sens0)
            gaia_flux = fluxcal.spec_to_LVM_flux(chan, std_wave_all, gaia_flux_interpolated[i])
            sens_coef = gaia_flux / lvmflux
            #print(f'lvmflux={lvmflux}, gaia_flux={gaia_flux}, converted to gaia flux = {lvmflux*sens_coef}')

            res_mod[f"STD{nn}SEN"] = sens0 * sens_coef
            sens = sens0 * sens_coef

            # fig_path = in_rss[n_chan]
            if plot:
                plt.plot(wgood, sgood * sens_coef, ".k", markersize=2, zorder=-999)
                plt.plot(w[n_chan], sens, linewidth=1, zorder=-999, label = fibers[i])
                plt.legend()

            # add PWV values and errors into header
            s_idx, s_fiber, s_gaia_id, *_ = std_info[i]
            rss.setHdrValue(f"S{s_idx}_PWV", pwv_values[i], f"PWV value for GAIA {s_gaia_id} {s_fiber}")
            rss.setHdrValue(f"S{s_idx}_PWVE", pwv_errors[i], "PWV error")

        # Add PWV averaged values into header
        pwv_mean = np.nanmean(np.asarray(pwv_values))
        pwv_median = np.nanmedian(np.asarray(pwv_values))
        pwv_std = np.nanstd(np.asarray(pwv_values))
        pwv_mean_err = np.nanmean(np.asarray(pwv_errors))
        pwv_median_err = np.nanmedian(np.asarray(pwv_errors))

        rss.setHdrValue(f"PWV_MEAN", pwv_mean, "Mean PWV value based on all standard stars")
        rss.setHdrValue(f"PWV_MED", pwv_median, "Median PWV value")
        rss.setHdrValue(f"PWV_STD", pwv_std, "Standard deviation of PWV values")
        rss.setHdrValue(f"PWVE_MNE", pwv_mean_err, "Mean of PWV errors")
        rss.setHdrValue(f"PWVE_MED", pwv_median_err, "Median of PWV errors")

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

def qa_model_matching(fig_path, fiber_params=None, gaia_params=None, model_params=None, matching_params=None,
                      wave_arrays=None, stdflux=None, flux_std_unconv_logscale=None,
                      log_std_errors_normalized_all=None, model_flux_resampled=None,
                      log_model_norm_convolved_spec_lsf=None, model_flux_gaia_convolved=None, mask_dict=None):

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
    plt.plot(wave_arrays['std_wave_all'], model_flux_resampled * matching_params["model2gaia_factor"], label='Model', linewidth=1)
    plt.plot(wave_arrays['gaia_wave'], model_flux_gaia_convolved * matching_params["model2gaia_factor"], label='Model, convolved with Gaia LSF',
             linewidth=1)

    teff_val = float(np.ma.filled(gaia_params["gaia_Teff"], np.nan))
    logg_val = float(np.ma.filled(gaia_params["gaia_logg"], np.nan))
    feh_val = float(np.ma.filled(gaia_params["gaia_z"], np.nan))

    plt.plot(wave_arrays['std_wave_all'], stdflux,
             label=f'Gaia, Teff={teff_val:.0f}, logg={logg_val:.1f}, [Fe/H]={feh_val:.1f}',
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

# TODO: can be removed
def calc_sensitivity_from_model(wl, obs_spec, spec_lsf, model_wave, model_flux=[], model_to_gaia_median=1, model_log_shift=0):
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

    # apply the model shift relative to observed spectra in log space
    log_model_wave, flux_model_logscale = linear_to_logscale(model_wave, model_flux)
    flux_model_shifted = logscale_to_linear(model_wave, log_model_wave, flux_model_logscale, shift=model_log_shift)

    #resample model to the same step
    model_flux_resampled = np.interp(wl, model_wave, flux_model_shifted)
    spec_lsf = np.sqrt(spec_lsf**2 - 0.3**2) # as model spectra were already convolved with lsf=0.3, we need to account for this

    # convolve model to spec lsf after vel. shift
    model_convolved_spec_lsf = fluxcal.lsf_convolve_fast(model_flux_resampled, spec_lsf, wl)
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
        nn, fiber, gaia_id, exptime, secz, _ = s  # unpack standard star tuple

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

# TODO: can be removed
def correct_tellurics(wave, std_spec, std_spec_orig, lsf, in_rss, chan, plot=False):
    """
    Do we need airmass correction?
    :param std_spec:
    :param lsf:
    :return:
    """
    if plot:
        plt.plot(wave, std_spec_orig, label='orig')
        plt.plot(wave, std_spec, label='interpolated')
        plt.legend()

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
