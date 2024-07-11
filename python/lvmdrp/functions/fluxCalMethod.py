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
import numpy as np
from scipy import interpolate
from scipy import stats

from astropy.stats import biweight_location, biweight_scale
from astropy.table import Table

from lvmdrp.core.rss import RSS, loadRSS, lvmFFrame
from lvmdrp.core.spectrum1d import Spectrum1D
import lvmdrp.core.fluxcal as fluxcal 
from lvmdrp.core.sky import get_sky_mask_uves, get_z_continuum_mask
from lvmdrp import log

from lvmdrp.core.plot import plt, create_subplots, save_fig


description = "provides flux calibration tasks"

__all__ = [
    "createSensFunction_drp",
    "createSensFunction2_drp",
    "quickFluxCalibration_drp",
    "correctTelluric_drp",
]

def apply_fluxcal(in_rss: str, out_fframe: str, skip_fluxcal: bool = False, display_plots: bool = False):
    """applies flux calibration to spectrograph-combined data

    Parameters
    ----------
    in_rss : str
        input RSS file
    out_rss : str
        output RSS file
    skip_fluxcal : bool, optional
        whether to skip flux calibration, by default False
    display_plots : bool, optional

    Returns
    -------
    rss : RSS
        flux-calibrated RSS object
    """
    # read all three channels
    log.info(f"loading RSS file {os.path.basename(in_rss)}")
    rss = loadRSS(in_rss)

    # initialize the lvmFFrame object
    fframe = lvmFFrame(data=rss._data, error=rss._error, mask=rss._mask, header=rss._header,
                       wave=rss._wave, lsf=rss._lsf,
                       sky_east=rss._sky_east, sky_east_error=rss._sky_east_error,
                       sky_west=rss._sky_west, sky_west_error=rss._sky_west_error,
                       fluxcal_std=rss._fluxcal_std, fluxcal_sci=rss._fluxcal_sci, slitmap=rss._slitmap)

    # check for flux calibration data
    if np.isnan(fframe._fluxcal_std.to_pandas().values).all():
        log.warning("no standard star metadata found, skipping flux calibration")
        fframe.setHdrValue("FLUXCAL", False, "flux-calibrated?")
        fframe.writeFitsData(out_fframe)
        return rss

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
    sens_arr = fframe._fluxcal_std.to_pandas().values  # * (std_exp / std_exp.sum())[None]
    sens_ave = biweight_location(sens_arr, axis=1, ignore_nan=True)
    sens_rms = biweight_scale(sens_arr, axis=1, ignore_nan=True)

    # fix case of all invalid values
    if (sens_ave == 0).all() or np.isnan(sens_ave).all():
        log.warning(
            "all sensitivity values are zero or NaN, impossible to flux-calibrate"
        )
        sens_ave = np.ones_like(sens_ave)
        sens_rms = np.zeros_like(sens_rms)
        fframe.setHdrValue("FLUXCAL", False, "flux-calibrated?")

    # update the fluxcal extension
    fframe._fluxcal_std["mean"] = sens_ave
    fframe._fluxcal_std["rms"] = sens_rms

    ax.set_title(f"{channel = }", loc="left")
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
    save_fig(
        fig,
        product_path=out_fframe,
        to_display=display_plots,
        figure_path="qa",
        label="fluxcal",
    )
    # flux-calibrate and extinction correct data
    # Note that we assume a constant extinction curve here!
    log.info(
        f"Extinction correcting science and sky spectra, curve {os.getenv('LVMCORE_DIR')+'/etc/lco_extinction.txt'}"
    )
    txt = np.genfromtxt(os.getenv("LVMCORE_DIR") + "/etc/lco_extinction.txt")
    lext, ext = txt[:, 0], txt[:, 1]
    ext = np.interp(fframe._wave, lext, ext)
    sci_secz = fframe._header["TESCIAM"]

    # optionally sky flux calibration
    if skip_fluxcal:
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
        fframe.setHdrValue("FLUXCAL", False, "flux-calibrated?")
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
        fframe.setHdrValue("FLUXCAL", True, "flux-calibrated?")
        fframe.setHdrValue("BUNIT", "ergs/s/cm^2/A", "flux units")

    log.info(f"writing output file in {os.path.basename(out_fframe)}")
    fframe.writeFitsData(out_fframe)

    return fframe


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
    for s in stds:
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
            continue

        # subtract sky spectrum and divide by exptime
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
        sens = stdflux / spec
        wgood, sgood = fluxcal.filter_channel(w, sens, 2)
        s = interpolate.make_smoothing_spline(wgood, sgood, lam=1e4)
        res[f"STD{nn}SEN"] = s(w).astype(np.float32)

        # caluculate SDSS g band magnitudes for QC
        mAB_std = np.round(fluxcal.spec_to_LVM_mAB(channel, w, stdflux), 2)
        mAB_obs = np.round(fluxcal.spec_to_LVM_mAB(channel, w[np.isfinite(spec)], spec[np.isfinite(spec)]), 2)
        # update input file header
        label = channel.upper()
        rss.setHdrValue(f"STD{nn}{label}AB", mAB_std, f"Gaia AB mag in {label}-band")
        rss.setHdrValue(f"STD{nn}{label}IN", mAB_obs, f"Obs AB mag in {label}-band")
        log.info(f"AB mag in LVM_{channel}: Gaia {mAB_std:.2f}, instrumental {mAB_obs:.2f}")

        if plot:
            plt.plot(wgood, sgood, ".k", markersize=2, zorder=-999)
            plt.plot(w, res[f"STD{nn}SEN"], linewidth=1)
            # plt.ylim(0,0.1e-11)
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
                                                             n_spec=NSCI_MAX, GAIA_CACHE_DIR='./gaia_cache')
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
            # TODO: understand exptime, is the data already per s?
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
            rss.setHdrValue(f"SCI{i+1}{cam}AB", mAB_std, f"Gaia AB mag in {cam}-band")
            rss.setHdrValue(f"SCI{i+1}{cam}IN", mAB_obs, f"Obs AB mag in {cam}-band")
            rss.setHdrValue(f"SCI{i+1}ID", data['SOURCE_ID'], f"Field star {i+1} Gaia source ID")
            rss.setHdrValue(f"SCI{i+1}FIB", scifibs['fiberid'][fib][0], f"Field star {i+1} fiber id")
            rss.setHdrValue(f"SCI{i+1}RA", data['ra'], f"Field star {i+1} RA")
            rss.setHdrValue(f"SCI{i+1}DE", data['dec'], f"Field star {i+1} DEC")
            log.info(f"AB mag in LVM_{cam}: Gaia {mAB_std:.2f}, instrumental {mAB_obs:.2f}")

            # calibrate and plot against the stars for debugging:
            if plot:
                plt.plot(obswave, np.interp(obswave, gwave, gflux)/obsflux, '.', 
                         color=colors[i%len(colors)] , markersize=2, zorder=-999)
                plt.plot(obswave, res_sci[f"STD{i+1}SEN"], color=colors[i%len(colors)], linewidth=2)

    return rss, res_sci


def fluxcal_standard_stars(in_rss, plot=True, GAIA_CACHE_DIR=None):
    """
    Create sensitivity functions for LVM data using the 12 spectra of stars observed through
    the Spec telescope.

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
        rss.writeFitsData(in_rss)
        return res_std, mean_std, rms_std, rss

    # early stop if not standards exposed in current spectrograph
    if len(stds) == 0:
        log.warning(f"no standard stars found in '{in_rss}', skipping sensitivity measurement")
        rss.writeFitsData(in_rss)
        return res_std, mean_std, rms_std, rss

    if plot:
        plt.subplot
        fig1 = plt.figure(1)
        frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
        frame1.set_xticklabels([])

    # standard fibers sensitivity curves
    rss, res_std = standard_sensitivity(stds, rss, GAIA_CACHE_DIR, ext, res_std, plot=plot)
    rms_std = biweight_scale(res_std.to_pandas().values, axis=1, ignore_nan=True)
    mean_std = biweight_location(res_std.to_pandas().values, axis=1, ignore_nan=True)

    cam = rss._header['CCD'].upper()
    rss.setHdrValue(f"STDSENM{cam}", np.nanmean(mean_std[1000:3000]), f"Mean stdstar sensitivity in {cam}")
    rss.setHdrValue(f"STDSENR{cam}", np.nanmean(rms_std[1000:3000]), f"Mean stdstar sensitivity rms in {cam}")
    log.info(f"Mean stdstar sensitivity in {cam} : {np.nanmean(mean_std[1000:3000])}")

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
    rss.set_fluxcal(fluxcal=res_std)
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

    cam = rss._header['CCD'].upper()
    rss.setHdrValue(f"SCISENM{cam}", np.nanmean(mean_sci[1000:3000]), f"Mean scistar sensitivity in {cam}")
    rss.setHdrValue(f"SCISENR{cam}", np.nanmean(rms_sci[1000:3000]), f"Mean scistar sensitivity rms in {cam}")
    log.info(f"Mean scistar sensitivity in {cam} : {np.nanmean(mean_sci[1000:3000])}")

    if plot:
        # TODO: Fix this!
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
    rss.set_fluxcal(fluxcal=res_sci)
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
