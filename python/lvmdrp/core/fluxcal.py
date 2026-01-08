# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez (adapted from MaNGA IDL code)
# @Date: Jan 27, 2023
# @Filename: fluxcal.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import numpy as np
from scipy import signal
from scipy.integrate import simpson
from scipy import interpolate
import requests
import pandas as pd
import bottleneck as bn
import os.path as path
import pathlib
from tqdm import tqdm


import gaiaxpy
from astroquery.gaia import Gaia

from astropy.time import Time
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u

from lvmdrp import log
from lvmdrp.core.constants import LVM_ELEVATION, LVM_LON, LVM_LAT, MASTERS_DIR, STELLAR_TEMP_PATH
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.utils.paths import get_calib_paths
from lvmdrp.core.spectrum1d import Spectrum1D, convolution_matrix
from lvmdrp.core.rss import RSS

from scipy.sparse import csr_matrix


def get_mean_sens_curves(sens_dir):
    return {'b':pd.read_csv(f'{sens_dir}/mean-sens-b.csv', names=['wavelength', 'sens']),
            'r':pd.read_csv(f'{sens_dir}/mean-sens-r.csv', names=['wavelength', 'sens']),
            'z':pd.read_csv(f'{sens_dir}/mean-sens-z.csv', names=['wavelength', 'sens'])}

def retrieve_header_stars(rss):
    """
    Retrieve fiber, Gaia ID, exposure time and airmass for the 12 standard stars in the header.
    return a list of tuples of the above quantities.
    """
    lco = EarthLocation(lat=LVM_LAT, lon=LVM_LON, height=LVM_ELEVATION * u.m)
    h = rss._header
    slitmap = rss._slitmap
    # retrieve the data for the 12 standards from the header
    stddata = []
    for i in range(12):
        stdi = "STD" + str(i + 1)
        if h[stdi + "ACQ"] and h[stdi + "FIB"] in slitmap["orig_ifulabel"]:
            gaia_id = h[stdi + "ID"]
            if gaia_id is None:
                log.warning(f"{stdi} acquired but Gaia ID is {gaia_id}")
                rss.add_header_comment(f"{stdi} acquired but Gaia ID is {gaia_id}")
                continue
            fiber = h[stdi + "FIB"]
            obstime = Time(h[stdi + "T0"])
            exptime = h[stdi + "EXP"]
            c = SkyCoord(float(h[stdi + "RA"]), float(h[stdi + "DE"]), unit="deg")
            stdT = c.transform_to(AltAz(obstime=obstime, location=lco))
            secz = stdT.secz.value
            zenith_angle = 90 - stdT.alt.value
            # print(gid, fib, et, secz)
            stddata.append((i + 1, fiber, gaia_id, exptime, secz, zenith_angle))
    return stddata


class GaiaStarNotFound(Exception):
    """
    Signal that the star has no BP-RP spectrum
    """

    pass


def retrive_gaia_star(gaiaID, GAIA_CACHE_DIR):
    """
    Load or download and load from cache the XP spectrum of a gaia star, converted to erg/s/cm^2/A
    """
    # create cache dir if it does not exist
    pathlib.Path(GAIA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if path.exists(GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + ".csv") is True:
        # read the tables from our cache
        gaiaflux = Table.read(GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + ".csv", format="csv")
        gaiawave = Table.read(GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + "_sampling.csv", format="csv")
    else:
        # need to download from Gaia archive
        CSV_URL = ("https://gea.esac.esa.int/data-server/data?RETRIEVAL_TYPE=XP_CONTINUOUS&ID=Gaia+DR3+"
            + str(gaiaID)
            + "&format=CSV&DATA_STRUCTURE=RAW")
        FILE = GAIA_CACHE_DIR + "/XP_" + str(gaiaID) + "_RAW.csv"

        with requests.get(CSV_URL, stream=True) as r:
            r.raise_for_status()
            if len(r.content) < 2:
                raise GaiaStarNotFound(f"Gaia DR3 {gaiaID} has no BP-RP spectrum!")
            with open(FILE, "w") as f:
                f.write(r.content.decode("utf-8"))

        # convert coefficients to sampled spectrum
        _, _ = gaiaxpy.calibrate(FILE, output_path=GAIA_CACHE_DIR,\
                                 output_file="gaia_spec_" + str(gaiaID), output_format="csv")
        # read the flux and wavelength tables
        gaiaflux = Table.read(GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + ".csv", format="csv")
        gaiawave = Table.read(GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + "_sampling.csv", format="csv")

    # make numpy arrays from whatever weird objects the Gaia stuff creates
    wave = np.fromstring(gaiawave["pos"][0][1:-1], sep=",") * 10  # in Angstrom
    # W/s/micron -> in erg/s/cm^2/A
    flux = (1e7 * 1e-1 * 1e-4 * np.fromstring(gaiaflux["flux"][0][1:-1], sep=","))
    return wave, flux


def get_XP_spectra(expnum, ra_tile, dec_tile, lim_mag=14.0, n_spec=15, GAIA_CACHE_DIR='./gaia_cache', plot=False):
    '''
    mjd, tileid, central ra and dec, query for brightest GAIA stars in the science IFU,
    cache their IDs, cache their XP spectra, and return a table with all the data
    '''
    if GAIA_CACHE_DIR is None or path.exists(GAIA_CACHE_DIR + f'/{expnum}_ids.ecsv') is False:
        print('querying for ids ...')
        r_ifu = np.sqrt(3.0)/2 * (30.2/2) / 60.0 # inner radius of hexagon in degrees for margin
        select_tile = f'DISTANCE({ra_tile}, {dec_tile}, ra, dec) < {r_ifu} '
        job = Gaia.launch_job(f"SELECT TOP {n_spec} * FROM gaiadr3.gaia_source_lite WHERE "
                              + select_tile + f"AND phot_g_mean_mag < {lim_mag} AND has_xp_continuous = 'True' ORDER BY phot_g_mean_mag ASC ")
        r = job.get_results()
        if GAIA_CACHE_DIR is not None:
            #print('writing '+GAIA_CACHE_DIR + f'/{expnum}_ids.ecsv')
            r.write(GAIA_CACHE_DIR + f'/{expnum}_ids.ecsv', overwrite=True)
    else:
        #print('reading '+GAIA_CACHE_DIR + f'/{expnum}_ids.ecsv')
        r = Table.read(GAIA_CACHE_DIR + f'/{expnum}_ids.ecsv')
    #
    # get XP spectra and cache the calibrated spectra
    #

    cols = r.colnames
    new_cols = [col.lower() for col in cols]
    r.rename_columns(cols, new_cols)

    sampling=np.arange(336., 1021., 2.)
    ids = [line['source_id'] for line in r]
    if GAIA_CACHE_DIR is None or path.exists(GAIA_CACHE_DIR + f'/{expnum}_XP_spec.pickle') is False:
        calibrated_spectra, _ = gaiaxpy.calibrate(ids, truncation=False, save_file=False)
        if GAIA_CACHE_DIR is not None:
            #print('writing '+GAIA_CACHE_DIR + f'/{expnum}_XP_spec.pickle')
            calibrated_spectra.to_pickle(GAIA_CACHE_DIR + f'/{expnum}_XP_spec.pickle')
    else:
        #print('reading '+GAIA_CACHE_DIR + f'/{expnum}_XP_spec.csv')
        calibrated_spectra = pd.read_pickle(GAIA_CACHE_DIR + f'/{expnum}_XP_spec.pickle')

    # calibrated_spectra
    if(plot):
        gaiaxpy.plot_spectra(calibrated_spectra, sampling=sampling, multi=True, show_plot=True, output_path=None, legend=False)
    # calibrated_spectra *= 100  # W/m^2 -> erg/s/cm^
    # astropy.Table ['SOURCE_ID, 'ra', 'dec', ...], ]pandas.DataFrame ['source_id', 'flux', 'flux_error'], np.ndarray
    return r, calibrated_spectra, sampling


def mean_absolute_deviation(vals):
    """
    Robust estimate of RMS
    - see https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    mval = bn.nanmedian(vals)
    rms = 1.4826 * bn.nanmedian(np.abs(vals - mval))
    return mval, rms
    # ok=np.abs(vals-mval)<4*rms


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    normal_cutoff = float(cutoff_freq) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype="lowpass")
    y = signal.filtfilt(b, a, data)
    return y


def cos_apod(nsample, perc=10.):
    y=np.ones(nsample)
    nperc=int(np.round(nsample*perc/100))
    x=np.sin(np.pi/2/nperc*np.arange(nperc))
    y[:nperc]=x
    y[-nperc:]=np.flip(x)
    return y


def derive_vecshift(vec, vec_ref, max_ampl=30, oversample_bin=20):
    """
    Derive shift of 1D-array vec from vec_ref using cross-correlation;
    both arrays assumed to be normalized
    if max_ampl is set then maximum shift is max_ampl
    """
    nsamples = min([len(vec), len(vec_ref)])
    vec[~np.isfinite(vec)] = np.nanmedian(vec)
    vec_ref[~np.isfinite(vec_ref)] = np.nanmedian(vec_ref)
    vec = signal.resample_poly(cos_apod(nsamples) * (vec[:nsamples]), oversample_bin, 1)
    vec_ref = signal.resample_poly(cos_apod(nsamples) * (vec_ref[:nsamples]), oversample_bin, 1)
    xcorr = signal.correlate(vec, vec_ref)
    if max_ampl:
        max_ampl = min([(nsamples * oversample_bin - 1), int(np.floor(max_ampl * oversample_bin))])
        xcorr = xcorr[nsamples * oversample_bin - (max_ampl + 1): nsamples * oversample_bin + max_ampl]
    else:
        max_ampl = nsamples * oversample_bin - 1
    dt = np.arange(- max_ampl, max_ampl + 1)
    shift = dt[xcorr.argmax()] / oversample_bin
    return shift



def filter_channel(w, f, k=3, method='lowpass'):
    c = np.where(np.isfinite(f))
    if method == 'lowpass':
        s = butter_lowpass_filter(f[c], 0.01, 2)
    elif method == 'savgol':
        s = signal.savgol_filter(f[c], 5, 3)
    res = s - f[c]
    # plt.plot(w[c], f[c], 'k.')
    # plt.plot(w[c], s, 'b-')
    mres, rms = mean_absolute_deviation(res)
    good = np.where(np.abs(res - mres) < k * rms)
    # plt.plot(w[c][good], f[c][good], 'r.', markersize=5)
    return w[c][good], f[c][good]


sdss_g_w = np.array(
    [
        3630,
        3640,
        3680,
        3780,
        3880,
        3980,
        4080,
        4180,
        4280,
        4380,
        4480,
        4580,
        4680,
        4780,
        4880,
        4980,
        5080,
        5180,
        5280,
        5380,
        5480,
        5580,
        5680,
        5780,
        5880,
        5980,
    ]
)
sdss_g_f = np.array(
    [
        0.0000,
        0.0000,
        0.0013,
        0.0055,
        0.0500,
        0.1629,
        0.2609,
        0.3105,
        0.3385,
        0.3596,
        0.3736,
        0.3863,
        0.3973,
        0.4019,
        0.4073,
        0.4147,
        0.4201,
        0.4147,
        0.3233,
        0.1043,
        0.0128,
        0.0024,
        0.0010,
        0.0003,
        0.0000,
        0.0000,
    ]
)

def LVM_phot_filter(channel, w):
    """
    LVM photometric system: Gaussian filter with sigma 250A centered in channels
    at 4500, 6500, and 8500A
    """
    if channel == "b":
        return np.exp(-0.5 * ((w - 4500) / 250) ** 2)
    elif channel == "r":
        return np.exp(-0.5 * ((w - 6500) / 250) ** 2)
    elif channel == "z":
        return np.exp(-0.5 * ((w - 8500) / 250) ** 2)
    else:
        raise Exception(f"Unknown filter '{channel}'")


def spec_to_mAB(lam, spec, lamf, filt):
    """
    Calculate AB magnitude in filter (lamf, filt) given a spectrum
    (lam, spec) in ergs/s/cm^2/A
    """
    c_AAs = 2.99792458e18  # Speed of light in Angstrom/s
    filt_int = np.interp(lam, lamf, filt)  # Interpolate to common wavelength axis
    I1 = simpson(y=spec * filt_int * lam, x=lam)
    I2 = simpson(y=filt_int / lam, x=lam)
    fnu = I1 / I2 / c_AAs  # Average flux density
    mab = -2.5 * np.log10(fnu) - 48.6  # AB magnitude
    if np.isnan(mab):
        mab = -9999.9
    return mab


def integrate_flux_in_filter(lam, spec, lamf, filt):
    """
    Calculate average flux in filter (lamf, filt) given a spectrum
    (lam, spec)
    """
    filt_int = np.interp(lam, lamf, filt)  # Interpolate to common wavelength axis
    return simpson(y=spec * filt_int, x=lam) / simpson(y=filt_int, x=lam)


def spec_to_LVM_flux(channel, w, f):
    """
    Return average flux in the LVM photometric system
    """
    return integrate_flux_in_filter(w, f, w, LVM_phot_filter(channel, w))

def spec_to_LVM_mAB(channel, w, f):
    """
    LVM photometric system: Gaussian filter with sigma 250A centered in channels
    at 4500, 6500, and 8500A
    """
    return spec_to_mAB(w, f, w, LVM_phot_filter(channel, w))


def sky_flux_in_filter(cam, skyfibs, obswave, percentile=75):
    '''
    Given an lvmFrame, calculate the median flux in the LVM photometric system of the
    lowest 'percentile' of sky fibers.

    Used for sky subtraction of the photometry of stars for sci IFU self calibration.
    '''
    nfiber = skyfibs.shape[0]
    flux = np.full(nfiber, np.nan)
    for i in range(nfiber):
        obsflux = skyfibs[i,:]
        f = np.isfinite(obsflux)
        if np.any(f):
            obsflux = interpolate_mask(obswave, obsflux, ~f)
            flux[i] = spec_to_LVM_flux(cam, obswave, obsflux)

    limidx = int(nfiber*percentile/100.0)
    skies = np.argsort(flux)[1:limidx]
    return bn.nanmedian(flux[skies])


def interpolate_mask(x, y, mask, kind="linear", fill_value=0):
    """
    :param x, y: numpy arrays, samples and values
    :param mask: boolean mask, True for masked values
    :param method: interpolation method, one of linear, nearest,
    nearest-up, zero, slinear, quadratic, cubic, previous, or next.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    if not np.any(mask):
        return y
    known_x, known_v = x[~mask], y[~mask]
    missing_x = x[mask]
    missing_idx = np.where(mask)

    f = interpolate.interp1d(known_x, known_v, kind=kind, fill_value=fill_value, bounds_error=False)
    yy = y.copy()
    yy[missing_idx] = f(missing_x)

    return yy


def lsf_convolve(data, diff_fwhm, wave_lsf_interp):
    """Degrade resolution of given spectrum
    """

    new_data = data.copy()
    sigmas = diff_fwhm / 2.354

    # setup kernel
    pixels = np.ceil(3 * max(sigmas))
    pixels = np.arange(-pixels, pixels)
    kernel = np.asarray([np.exp(-0.5 * (pixels / sigmas[iw]) ** 2) for iw in range(data.size)])
    kernel = convolution_matrix(kernel)
    new_data = kernel @ data

    return new_data


def lsf_convolve_fast(data, diff_fwhm, waves=None, n_sigma=4, edges_reflect=True):
    """
    Degrade spectrum resolution using sparse matrix convolution (~40x faster than lsf_convolve).

    Parameters
    ----------
    data : np.ndarray
        Input flux array
    diff_fwhm : np.ndarray
        FWHM array for each pixel (same length as data); in pixels, or in wavelength units if waves provided
    waves : np.ndarray, optional
        Wavelength array; if provided, converts diff_fwhm from wavelength units to pixels
    n_sigma : int, optional
        Kernel size in units of sigma
    edges_reflect : bool, optional
        If True, reflect edges to avoid edge effects in convolution

    Returns
    -------
    np.ndarray
        Convolved flux array
    """
    # convert FWHM from wavelength units to pixels if wavelength array provided
    if waves is not None:
        # this calculates the sizes of pixels in wavelength units, can be used for irregular grids
        delta_wave = np.diff(edges_from_centers(waves))
        sigmas = (diff_fwhm / delta_wave) / 2.3548 # 2*np.sqrt(2*np.log(2))
    else:
        sigmas = diff_fwhm / 2.3548

    # determine kernel size based on maximum sigma (nsigma-sigma coverage)
    max_sigma = np.max(sigmas)
    npix_half = int(np.ceil(n_sigma * max_sigma))
    npix = 2 * npix_half + 1  # full size of the kernel

    # handle edge reflection to avoid edge effects
    if edges_reflect:
        # get kernel sizes at edges
        npix_half_left = int(np.ceil(n_sigma * sigmas[0]))
        npix_half_right = int(np.ceil(n_sigma * sigmas[-1]))

        # reflect edges
        data_extended = np.concatenate([
            data[npix_half_left:0:-1],  # left reflection (reversed)
            data,
            data[-2:-npix_half_right-2:-1]  # right reflection (reversed)
        ])

        # extend sigmas array using edge values
        sigmas_extended = np.concatenate([
            np.full(npix_half_left, sigmas[0]),
            sigmas,
            np.full(npix_half_right, sigmas[-1])
        ])

        offset = npix_half_left
    else:
        data_extended = data
        sigmas_extended = sigmas
        offset = 0

    # create pixel grid for kernel
    x_px = np.arange(-npix_half, npix_half + 1, dtype=float)

    # compute Gaussian kernels for all pixels using broadcasting
    # shape: (n_pixels, npix)
    yy = x_px[None, :] / sigmas_extended[:, None]

    kernels = np.exp(-0.5 * yy**2)
    # normalize kernels numerically
    kernels = kernels / kernels.sum(axis=1, keepdims=True)

    # build sparse matrix indices
    x_size = data_extended.size
    rows2d = np.broadcast_to(np.arange(x_size)[:, None], (x_size, npix))
    cols2d = rows2d + x_px.astype(int)

    # clip columns to valid range [0, x_size)
    valid_mask = (cols2d >= 0) & (cols2d < x_size)
    rows_flat = rows2d[valid_mask]
    cols_flat = cols2d[valid_mask]
    data_flat = kernels[valid_mask]

    # construct sparse convolution matrix
    matrix_csr = csr_matrix((data_flat, (rows_flat, cols_flat)), shape=(x_size, x_size))

    # apply convolution
    new_data = matrix_csr @ data_extended

    # remove reflected edges if they were added
    if edges_reflect:
        new_data = new_data[offset:offset + data.size]

    return new_data


def get_worst_resolution(delta_fwhm=1.0):
    """Get worst possible resolution + delta_fwhm from available LVM long-term calibrations
    """
    # get all available calibration epochs
    calib_mjds = sorted([int(p) for p in os.listdir(MASTERS_DIR) if p.isdigit() and int(p) >= 60177])

    worst_res = 0
    for calib_mjd in calib_mjds:
        calib_paths = get_calib_paths(mjd=calib_mjd, flavors={"lsf"}, from_sanbox=True)
        for _, lsf_path in calib_paths["lsf"].items():
            lsf = TraceMask.from_file(lsf_path)

            mask = np.isnan(lsf._data)
            if lsf._mask is not None:
                mask |= lsf._mask
            worst_ = lsf._data[~mask].max()
            if worst_ > worst_res:
                worst_res = worst_

    # add 1 Angstrom so that we don't get collapsed Gaussians
    worst_res += delta_fwhm

    return worst_res


def create_stellar_templates(target_fwhm, models_dir=STELLAR_TEMP_PATH, model_fwhm=0.3, model_sampling=0.05):
    """Create stellar templates with given resolution in FWHM"""
    # read the best-fit model and convolve with spectrograph LSF
    n_steps = int((9800-3600) / model_sampling) + 1
    model_wave = np.linspace(3600, 9800, n_steps)
    model_lsf = np.ones_like(model_wave) * target_fwhm

    models_dir = os.path.join(models_dir, 'good_res')
    models_path = [os.path.join(models_dir, models_name) for models_name in os.listdir(models_dir) if models_name.endswith(".fits")]
    log.info(f"loading stellar templates from '{models_dir}', found: {len(models_path)} templates")

    log.info(f"assuming wavelength sampling of {model_sampling = } and spectral FWHM {model_fwhm = } Angstroms")

    new_models = []
    iterator = tqdm(models_path, desc="degrading models resolution", ascii=True, unit="spectrum")
    for model_path in iterator:
        try:
            with fits.open(model_path, memmap=False) as hdul:
                model_flux = hdul[0].data
        except OSError as e:
            log.error(f"while reading {model_path}: {e}")
            continue
        diff_lsf = np.sqrt(model_lsf**2 - model_fwhm**2)

        # convolve model to spec lsf
        new_models.append(lsf_convolve(model_flux, diff_lsf, model_wave))

    new_header = fits.Header()
    new_header["MODPATH"] = (models_dir, "directory of original models")
    new_header["INISAMP"] = (model_sampling, "initial wavelength sampling [Angstrom]")
    new_header["INIFWHM"] = (model_fwhm, "initial resolution in FWHM [Angstrom]")
    new_header["FINFWHM"] = (target_fwhm, "final resolution in FWHM [Angstrom]")

    rss_models = RSS(data=np.asarray(new_models), wave=model_wave, lsf=model_lsf, header=new_header)

    out_models = os.path.join(models_dir, "lvm-stellar-templates.fits")
    log.info(f"writing models to '{out_models}'")
    rss_models.writeFitsData(out_models)

    return rss_models


def extinctLaSilla(wave):
    # digitized version of LaSilla extinctin curve from
    w = [
        3520.83333,
        3562.50000,
        3979.16667,
        4489.58333,
        4802.08333,
        5312.50000,
        5614.58333,
        5760.41667,
        6041.66667,
        6572.91667,
        7145.83333,
        7541.66667,
        8052.08333,
        8770.83333,
        9781.25000,
        10197.91667,
    ]
    f = [
        0.53533,
        0.52174,
        0.34511,
        0.22283,
        0.18071,
        0.14402,
        0.13315,
        0.14130,
        0.11685,
        0.07880,
        0.05299,
        0.04348,
        0.03533,
        0.02717,
        0.01902,
        0.02038,
    ]
    spec_raw = Spectrum1D(wave=w, data=f)
    return spec_raw.resampleSpec(wave)


def extinctCAHA(wave, extinct_v, type="mean"):
    if type == "mean":
        data = 0.0935 * (wave / 5450.0) ** (-4) + (
            ((0.8 * extinct_v) - 0.0935) * (wave / 5450.0) ** (-0.8)
        )

    elif type == "winter" or type == "summer":
        if type == "winter":
            (f1, f2, f3) = (1.02, 0.94, 0.29)

        elif type == "summer":
            (f1, f2, f3) = (1.18, 4.52, 0.19)
        k1 = f1 * 7.25e-3 * (wave / 10000.0) ** (-4)
        k2 = f2 * 0.006 * (wave / 10000.0) ** (-0.8)
        k3 = f3 * 0.015 * np.exp(-((wave - 6000.0) / 1200.0))
        data = k1 + k2 + k3
        scale_idx = np.argsort((wave - 5500.0) ** 2)[0]
        scale_offset = extinct_v - data[scale_idx]
        data = data + scale_offset

    spec = Spectrum1D(wave=wave, data=data)
    return spec


def extinctParanal(wave):
    wave_base = np.concatenate(
        (
            np.arange(3325, 6780, 50),
            np.array([7060, 7450, 7940, 8500, 8675, 8850, 10000]),
        )
    )
    extinct = np.array(
        [
            0.686,
            0.606,
            0.581,
            0.552,
            0.526,
            0.504,
            0.478,
            0.456,
            0.430,
            0.409,
            0.386,
            0.378,
            0.363,
            0.345,
            0.330,
            0.316,
            0.298,
            0.285,
            0.274,
            0.265,
            0.253,
            0.241,
            0.229,
            0.221,
            0.212,
            0.204,
            0.198,
            0.190,
            0.185,
            0.182,
            0.176,
            0.169,
            0.162,
            0.157,
            0.156,
            0.153,
            0.146,
            0.143,
            0.141,
            0.139,
            0.139,
            0.134,
            0.133,
            0.131,
            0.129,
            0.127,
            0.128,
            0.130,
            0.134,
            0.132,
            0.124,
            0.122,
            0.125,
            0.122,
            0.117,
            0.115,
            0.108,
            0.104,
            0.102,
            0.099,
            0.095,
            0.092,
            0.085,
            0.086,
            0.083,
            0.081,
            0.076,
            0.072,
            0.068,
            0.064,
            0.064,
            0.048,
            0.042,
            0.032,
            0.030,
            0.029,
            0.022,
        ]
    )
    spec_raw = Spectrum1D(wave=wave_base, data=extinct)
    spec = spec_raw.resampleSpec(wave)
    return spec


def galExtinct(wave, Rv):
    m = wave / 10000.0
    x = 1.0 / m
    y = x - 1.82
    ax = (
        1
        + (0.17699 * y)
        - (0.50447 * y**2)
        - (0.02427 * y**3)
        + (0.72085 * y**4)
        + (0.01979 * y**5)
        - (0.77530 * y**6)
        + (0.32999 * y**7)
    )
    bx = (
        (1.41338 * y)
        + (2.28305 * y**2)
        + (1.07233 * y**3)
        - (5.38434 * y**4)
        - (0.62251 * y**5)
        + (5.30260 * y**6)
        - (2.09002 * y**7)
    )

    Arat = (ax + (bx / Rv)).astype(np.float32)
    spec = Spectrum1D(wave=wave, data=Arat)
    return spec


def edges_from_centers(centers):
    """
    Ancillary function for fluxconserve_rebin to calculate bin edges from bin centers.
    """
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5*(centers[1:] + centers[:-1])
    edges[0]  = centers[0]  - 0.5*(centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5*(centers[-1] - centers[-2])
    return edges


def fluxconserve_rebin(output_wave, input_wave, input_flux, normalize=True):
    """
    Flux-conserving rebin using linear inteprolation of the commulative function.
    Preserves input data type while performing calculations in float64.
    Returns np.nan if output_wave is outside the range of input_wave.
    """
    # Store original dtype
    original_dtype = input_flux.dtype

    # Convert to float64 for calculations
    input_flux_f64 = input_flux.astype(np.float64)

    edges_out = edges_from_centers(output_wave)
    edges_in = edges_from_centers(input_wave)
    cdf_in = np.r_[0.0, np.cumsum(input_flux_f64, dtype=np.float64)]
    cdf_out = np.interp(edges_out, edges_in, cdf_in)

    if normalize:
        # in this case output spectrum will be on the same level of the input
        x_edges_in = np.arange(len(edges_in))
        x_edges_out = np.interp(edges_out, edges_in, x_edges_in, left=np.nan, right=np.nan)
        norm_factors = np.diff(x_edges_out)
    else:
        # spectrum level will be higer because of integration over pixels
        norm_factors = np.diff(edges_out)

    output = np.diff(cdf_out) / norm_factors

    # Convert back to original dtype
    return output.astype(original_dtype)


def rebin_and_convolve(wave_target, wave_source, flux_source, lsf, lsf_in_wavelength=False):
    """
    Rebin and convolve spectrum to target wavelength grid.
    """
    rebinned = fluxconserve_rebin(wave_target, wave_source, flux_source)
    if lsf_in_wavelength:
        convolved = lsf_convolve_fast(rebinned, lsf, waves=wave_target)
    else:
        convolved = lsf_convolve_fast(rebinned, lsf)
    return convolved


class TelluricCalculator:
    """
    Handles telluric transmission calculations using a precomputed high resolution
    transmission curve generated with the Palace and SkyModel models.

    Encapsulates the sky model data and provides methods to compute atmospheric
    transmission as a function of precipitable water vapor (PWV) and zenith angle.
    """

    def __init__(self, skymodel_path=None):
        """
        Initialize TelluricCalculator by loading the Palace Sky Model.

        Parameters
        ----------
        skymodel_path : str, optional
            Full path to the SkyModel transmission FITS table.
            If None, uses default: MASTERS_DIR/stellar_models/transmission_from_Palace_SkyModel_step0.2.fits
        """
        self._load_sky_model(skymodel_path)

    def _load_sky_model(self, skymodel_path=None):
        """Load the Palace Sky Model transmission table."""
        if skymodel_path is None:
            model_path = os.path.join(MASTERS_DIR, 'stellar_models', 'transmission_from_Palace_SkyModel_step0.2.fits')
        else:
            model_path = skymodel_path

        log.info(f"Loading telluric transmission model from '{model_path}'")

        skytab = fits.getdata(model_path)
        # Store full model data
        self._wave_air_full = skytab['wave_air']
        self._trans_ma_full = skytab['trans_ma']
        self._fH2O_full = skytab['fH2O']

        # Initialize working arrays to full model
        self.wave_air = self._wave_air_full
        self.trans_ma = self._trans_ma_full
        self.fH2O = self._fH2O_full
        self._wave_range_set = False

    def set_wave_range(self, wave_min, wave_max):
        """
        Set working wavelength range to improve computational efficiency.
        """
        mask = (self._wave_air_full >= wave_min) & (self._wave_air_full <= wave_max)
        self.wave_air = self._wave_air_full[mask]
        self.trans_ma = self._trans_ma_full[mask]
        self.fH2O = self._fH2O_full[mask]
        self._wave_range_set = True

    def reset_wave_range(self):
        """Reset wavelength range to full model coverage."""
        self.wave_air = self._wave_air_full
        self.trans_ma = self._trans_ma_full
        self.fH2O = self._fH2O_full
        self._wave_range_set = False

    def calc_transmission(self, pwv, zenith_angle=None, airmass=None):
        """
        Calculate atmospheric transmission for given PWV and zenith angle or airmass.

        According to formulae 4 and 5 from Noll et al. 2025
        PALACE v1.0: Paranal Airglow Line And Continuum Emission model
        """
        if zenith_angle is None and airmass is None:
            raise ValueError("Either zenith_angle or airmass must be provided")

        if airmass is None:
            cosz = np.cos(np.deg2rad(zenith_angle))
            airmass = 1.0 / (cosz + 0.025 * np.exp(-11.0 * cosz))

        rpwv = pwv / 2.5 - 1
        return np.power(self.trans_ma, (1.0 + rpwv * self.fH2O) * airmass)

    def match_to_data(self, wave_target, lsf, pwv, zenith_angle=None, airmass=None, lsf_in_wavelength=False):
        """
        Calculate telluric transmission and match to observed data wavelength grid,
        including rebinning and convolution with LSF.
        """
        # Calculate transmission at high resolution model grid
        trans_hr = self.calc_transmission(pwv, zenith_angle=zenith_angle, airmass=airmass)

        # Rebin and convolve to target wavelength grid
        trans_rebinned = rebin_and_convolve(wave_target, self.wave_air, trans_hr, lsf, lsf_in_wavelength=lsf_in_wavelength)

        return trans_rebinned