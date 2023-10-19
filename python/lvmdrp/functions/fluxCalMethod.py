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
from scipy import signal

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.stats import biweight_location, biweight_scale
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from lvmdrp.core.rss import RSS, loadRSS
from lvmdrp.core.spectrum1d import Spectrum1D
from lvmdrp.external import ancillary_func
from lvmdrp.functions import skyMethod
from lvmdrp import log

from lvmdrp.core.plot import plt, create_subplots, save_fig


description = "provides flux calibration tasks"

__all__ = [
    "createSensFunction_drp",
    "createSensFunction2_drp",
    "quickFluxCalibration_drp",
    "correctTelluric_drp",
]

def apply_fluxcal(in_rss: str, out_rss: str, display_plots: bool = False):
    """applies flux calibration to spectrograph-combined data

    Parameters
    ----------
    in_rss : str
        input RSS file
    out_rss : str
        output RSS file
    display_plots : bool, optional

    Returns
    -------
    rss : RSS
        flux-calibrated RSS object
    """
    # read all three channels
    log.info(f"loading RSS file {os.path.basename(in_rss)}")
    rss = loadRSS(in_rss)
    expnum = rss._header["EXPOSURE"]
    channel = rss._header["CCD"][0]
    # set masked pixels to NaN
    rss.apply_pixelmask()

    # apply joint sensitivity curve
    fig, ax = create_subplots(to_display=display_plots, figsize=(15, 5))
    fig.suptitle(f"Flux calibration for {expnum = }, {channel = }")
    log.info(f"computing joint sensitivity curve for channel {channel}")
    # scale by exposure time (each std star has slightly different exposure time)
    sens_arr = rss._fluxcal.to_pandas().values
    sens_ave = biweight_location(sens_arr, axis=1, ignore_nan=True)
    sens_rms = biweight_scale(sens_arr, axis=1, ignore_nan=True)

    rss._fluxcal["mean"] = sens_ave
    rss._fluxcal["rms"] = sens_rms

    ax.set_title(f"{channel = }", loc="left")
    for j in range(sens_arr.shape[1]):
        std_hd = rss._fluxcal.colnames[j][:-3]
        std_id = rss._header[f"{std_hd}FIB"]

        ax.plot(rss._wave, sens_arr[:,j], "-", lw=1, label=std_id)
    ax.plot(rss._wave, sens_ave, "-r", lw=2, label="mean")
    ax.set_yscale("log")
    ax.set_xlabel("wavelength (Angstrom)")
    ax.set_ylabel("sensitivity [(ergs/s/cm^2/A) / e-]")
    ax.legend(loc="upper right")
    save_fig(fig, product_path=out_rss, to_display=display_plots, figure_path="qa", label="fluxcal")

    # flux-calibrate data
    log.info("flux-calibrating data science and sky spectra")
    rss._data *= sens_ave
    rss._error *= sens_ave
    rss._sky *= sens_ave
    rss._sky_error *= sens_ave

    log.info(f"writing output file in {os.path.basename(out_rss)}")
    rss.writeFitsData(out_rss)

    return rss

def fluxcal_Gaia(camera, in_rss, plot=True, GAIA_CACHE_DIR=None):
    '''
    Flux calibrate LVM data using the 12 spectra of stars observed through
    the Spec telescope.

    Uses Gaia BP-RP spectra for calibration. To be replaced or extended by using fitted stellar 
    atmmospheres.
    '''
    GAIA_CACHE_DIR = './' if GAIA_CACHE_DIR is None else GAIA_CACHE_DIR
    log.info(f"Using Gaia CACHE DIR '{GAIA_CACHE_DIR}'")

    # get the list of standards from the header
    try:
        stds = retrieve_header_stars(in_rss)
    except KeyError:
        log.warning("no standard star information found, skipping flux calibration")
        return

    # load input RSS
    log.info(f"loading input RSS file '{os.path.basename(in_rss)}'")
    rss = RSS()
    rss.loadFitsData(in_rss)

    sci_exptime = rss._header['EXPTIME']
    sci_exptime = rss._header['TESCIAM']

    # load fibermap and filter for current spectrograph
    slitmap = rss._slitmap[rss._slitmap["spectrographid"] == int(camera[1])]

    # wavelength array
    w = rss._wave

    # load extinction curve
    # Note that we assume a constant extinction curve here!
    txt = np.genfromtxt(os.getenv('LVMCORE_DIR')+'/etc/lco_extinction.txt')
    lext, ext = txt[:,0], txt[:,1]
    ext = np.interp(w, lext, ext)

    if plot:
        plt.subplot
        res = []
        fig1 = plt.figure(1)
        frame1 = fig1.add_axes((.1,.3,.8,.6))
        frame1.set_xticklabels([])

    # load the sky masks
    m = skyMethod.get_sky_mask_uves(w, width=3)
    m2 = None
    if camera[0] == 'z':
        m2 = skyMethod.get_z_continuum_mask(w)

    # iterate over standard stars, derive sensitivity curve for each
    res = []
    res_columns = []
    for s in stds:
        nn, fiber, gaia_id, exptime, secz = s   # unpack standard star tuple
        
        # find the fiber with our spectrum of that Gaia star, if it is not in the current spectrograph, continue
        select = (slitmap["orig_ifulabel"] == fiber)
        fibidx = np.where(select)[0]
        if len(fibidx) == 0:
            continue

        log.info(f"Standard fiber '{fiber}', index '{fibidx}', star '{gaia_id}', exptime '{exptime:.2f}', secz '{secz:.2f}'")

        # load Gaia BP-RP spectrum from cache, or download from webapp
        try:
            gw, gf = ancillary_func.retrive_gaia_star(gaia_id, GAIA_CACHE_DIR=GAIA_CACHE_DIR)
            stdflux = np.interp(w, gw, gf)   # interpolate to our wavelength grid
        except ancillary_func.GaiaStarNotFound as e:
            log.warning(e)
            continue
    
        # divide by our exptime for that standard
        spec = rss._data[fibidx[0],:]/exptime
        
        # interpolate over bright sky lines
        spec = ancillary_func.interpolate_mask(w, spec, m, fill_value='extrapolate')
        if camera[0] == 'z':
            spec = ancillary_func.interpolate_mask(w, spec, ~m2, fill_value='extrapolate')
        
        # correct for extinction
        spec *= 10**(0.4*ext*secz)

        # TODO: mask telluric spectral regions

        # divide to find sensitivity and smooth
        sens = stdflux/spec        
        wgood, sgood = filter_channel(w, sens, 2)
        s = interpolate.make_smoothing_spline(wgood, sgood, lam=1e4)        
        r = s(w).astype(np.float32)
        res.append(r)
        res_columns.append(Table.Column(name=f"STD{nn}SEN", dtype="f8", data=r))

        # caluculate SDSS g band magnitudes for QC
        # put in header as
        # STDNNBAB, RAB, ZAB and STDNNBIN, RIN, ZIN
        mAB_std = ancillary_func.spec_to_LVM_mAB(camera, w, stdflux)
        mAB_obs = ancillary_func.spec_to_LVM_mAB(camera, w[np.isfinite(spec)], spec[np.isfinite(spec)])
        log.info(f"AB mag in LVM_{camera[0]}: Gaia {mAB_std:.2f}, instrumental {mAB_obs:.2f}")

        if plot:
            plt.plot(wgood, sgood, 'r.', markersize=4)
            plt.plot(w, s(w), linewidth=0.5)
            #plt.ylim(0,0.1e-11)

    res = np.array(res)            # list of sensitivity functions in (ergs/s/cm^2/A) / e-
    rms = biweight_scale(res, axis=0, ignore_nan=True)
    mean = biweight_location(res, axis=0, ignore_nan=True)

    if plot:
        plt.ylabel('sensitivity [(ergs/s/cm^2/A) / e-]')
        plt.xlabel('wavelength [A]')
        plt.ylim(1e-14, 0.1e-11)
        plt.semilogy()
        frame2 = fig1.add_axes((.1,.1,.8,.2))        
        plt.plot([w[0],w[-1]], [0.05, 0.05], color='k', linewidth=1, linestyle='dotted')
        plt.plot([w[0],w[-1]], [-0.05, -0.05], color='k', linewidth=1, linestyle='dotted')
        plt.plot([w[0],w[-1]], [0.1, 0.1], color='k', linewidth=1, linestyle='dashed')
        plt.plot([w[0],w[-1]], [-0.1, -0.1], color='k', linewidth=1, linestyle='dashed')
        plt.plot(w, rms/mean)
        plt.plot(w, -rms/mean)
        plt.ylim(-0.2, 0.2)
        plt.ylabel('relative residuals')
        plt.xlabel('wavelength [A]')

    save_fig(plt.gcf(), product_path=in_rss, to_display=False, figure_path="qa", label="fluxcal")

    # update input file header
    if camera[0] == "b":
        rss.setHdrValue(f"STD{nn}BAB", mAB_std, "AB mag in B-band")
        rss.setHdrValue(f"STD{nn}BIN", mAB_obs, "AB mag in B-band")
    elif camera[0] == "r":
        rss.setHdrValue(f"STD{nn}RAB", mAB_std, "AB mag in R-band")
        rss.setHdrValue(f"STD{nn}RIN", mAB_obs, "AB mag in R-band")
    elif camera[0] == "z":
        rss.setHdrValue(f"STD{nn}ZAB", mAB_std, "AB mag in Z-band")
        rss.setHdrValue(f"STD{nn}ZIN", mAB_obs, "AB mag in Z-band")
    # add sensitivity extension
    sens_table = Table()
    sens_table.add_columns(res_columns)
    rss.set_fluxcal(fluxcal=sens_table)
    rss.writeFitsData(in_rss)

    return res, mean, rms, rss

def retrieve_header_stars(in_rss):
    '''
    Retrieve fiber, Gaia ID, exposure time and airmass for the 12 standard stars in the header.
    return a list of tuples of the above quatities.
    '''
    lco = EarthLocation(lat=-29.008999964*u.deg, lon=-70.688663912*u.deg, height=2800*u.m)   
    with fits.open(in_rss) as hdub:
        h = hdub[0].header
    # retrieve the data for the 12 standards from the header
    stddata = []
    for i in range(12):
        stdi = 'STD'+str(i+1)
        if h[stdi+'ACQ']:
            gaia_id = h[stdi+'ID']
            fiber = h[stdi+'FIB']
            obstime = Time(h[stdi+'T0'])
            exptime = h[stdi+'EXP']
            c = SkyCoord(float(h[stdi+'RA']), float(h[stdi+'DE']), unit="deg") 
            stdT = c.transform_to(AltAz(obstime=obstime,location=lco))  
            secz = stdT.secz.value
            #print(gid, fib, et, secz)
        stddata.append((i+1, fiber, gaia_id, exptime, secz))
    return stddata

def mean_absolute_deviation(vals):
    '''
    Robust estimate of RMS
    - see https://en.wikipedia.org/wiki/Median_absolute_deviation
    '''
    mval = np.nanmedian(vals)
    rms = 1.4826*np.nanmedian(np.abs(vals-mval))
    return mval, rms
    #ok=np.abs(vals-mval)<4*rms

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    normal_cutoff = float(cutoff_freq) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass') 
    y = signal.filtfilt(b, a, data)
    return y

def filter_channel(w, f, k=3):
    c = np.where(np.isfinite(f))
    s = butter_lowpass_filter(f[c], 0.01, 2)
    res = s - f[c]
    #plt.plot(w[c], f[c], 'k.')
    #plt.plot(w[c], s, 'b-')
    mres, rms = mean_absolute_deviation(res)
    good = np.where(np.abs(res-mres)<k*rms)
    #plt.plot(w[c][good], f[c][good], 'r.', markersize=5)
    return w[c][good], f[c][good]



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
    except:
        extinct_v = float(extinct_v)

    try:
        airmass = rss.getHdrValue(airmass)
    except:
        airmass = float(airmass)

    try:
        exptime = rss.getHdrValue(exptime)
    except:
        exptime = float(exptime)

    if (
        extinct_curve == "mean"
        or extinct_curve == "summer"
        or extinct_curve == "winter"
    ):
        extinct = 10 ** (
            ancillary_func.extinctCAHA(star_spec._wave, extinct_v, type=extinct_curve)
            * airmass
            * -0.4
        )
    elif extinct_curve == "Paranal":
        extinct = 10 ** (
            ancillary_func.extinctParanal(star_spec._wave) * airmass * -0.4
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
    except:  # KeyError or TypeError:
        extinct_v = float(extinct_v)

    try:
        airmass = rss.getHdrValue(airmass)
    except:  # KeyError or TypeError:
        airmass = float(airmass)

    try:
        exptime = rss.getHdrValue(exptime)
    except:  # KeyError or TypeError:
        exptime = float(exptime)

    if (
        extinct_curve == "mean"
        or extinct_curve == "summer"
        or extinct_curve == "winter"
    ):
        extinct = 10 ** (
            ancillary_func.extinctCAHA(star_spec._wave, extinct_v, type=extinct_curve)
            * airmass
            * -0.4
        )
    elif extinct_curve == "Paranal":
        extinct = 10 ** (
            ancillary_func.extinctParanal(star_spec._wave) * airmass * -0.4
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
                ancillary_func.extinctCAHA(rss._wave, extinct_v, type=extinct_curve)
                * airmass
                * -0.4
            )
        elif extinct_curve == "Paranal":
            extinct = 10 ** (ancillary_func.extinctParanal(rss._wave) * airmass * -0.4)
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
