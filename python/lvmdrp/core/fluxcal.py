# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez (adapted from MaNGA IDL code)
# @Date: Jan 27, 2023
# @Filename: fluxcal.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os
import numpy as np
import pathlib
from contextlib import redirect_stdout
from scipy import signal
from scipy.integrate import simpson
from scipy import interpolate
import pandas as pd
import bottleneck as bn
from tqdm import tqdm

import pyvo as vo
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

import warnings

# pandas mute warnings about .swapaxes method
warnings.filterwarnings("ignore", message='.*DataFrame.swapaxes.*')

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


class GaiaXPSpectra(object):
    """Manage Gaia XP spectrophotometry retrieval, caching, and loading.

    This class fetches Gaia DR3 XP coefficient data, converts coefficients into
    sampled spectra, caches results locally, and loads cached spectra on demand.
    """

    def __init__(self, cache_dir="./gaia_cache"):
        """Initialize the GaiaXPSpectra cache manager.

        Parameters
        ----------
        cache_dir : str, optional
            Directory where Gaia XP spectrum cache files are stored.
        """
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._wave_sampling = np.arange(336, 1021, 2)

    def _get_xp_spectrum_paths(self, source_id):
        """Return the expected cache file paths for a Gaia source.

        Parameters
        ----------
        source_id : int or str
            Gaia source identifier.

        Returns
        -------
        tuple[pathlib.Path, pathlib.Path]
            Paths for the wavelength sampling and spectrum CSV files.
        """
        output_name = f"gaia_spec_{source_id}"
        wave_path = self.cache_dir / f"{output_name}_sampling.csv"
        spec_path = self.cache_dir / f"{output_name}.csv"
        return wave_path, spec_path

    def _xp_spectrum_exists(self, source_id):
        """Return whether a cached XP spectrum exists for the source.

        Parameters
        ----------
        source_id : int or str
            Gaia source identifier.

        Returns
        -------
        bool
            True if both sampling and spectrum cache files exist.
        """
        wave_path, spec_path = self._get_xp_spectrum_paths(source_id)
        return wave_path.exists() and spec_path.exists()

    def _not_in_cache(self, source_ids):
        """Filter source IDs that are not already cached.

        Parameters
        ----------
        source_ids : iterable
            Gaia source identifiers to check.

        Returns
        -------
        list
            IDs that are missing from the local cache.
        """
        new_ids = list(filter(lambda source_id: not self._xp_spectrum_exists(source_id), source_ids))
        return new_ids

    def _convert_to_csg(self, wave_xp, spectra_xp):
        """Convert Gaia XP spectrum units to wavelength in Angstrom and flux in CGS.

        Parameters
        ----------
        wave_xp : ndarray
            Wavelength sampling array in microns.
        spectra_xp : ndarray
            Flux array in W/s/micron.

        Returns
        -------
        tuple[ndarray, ndarray]
            Converted wavelength and flux arrays.
        """
        # micron -> A
        wave_xp *= 10
        # W/s/micron -> erg/s/cm^2/A
        spectra_xp *= 1e7 * 1e-1 * 1e-4
        return wave_xp, spectra_xp

    def fetch_xp_coeffs(self, source_ids):
        """Fetch Gaia XP coefficients from the Gaia TAP service.

        Parameters
        ----------
        source_ids : int, str, or sequence
            Gaia DR3 source identifier or identifiers.

        Returns
        -------
        pandas.DataFrame
            Coefficients for the requested XP spectra.
        """

        # define origin DB
        tap_service = vo.dal.TAPService("https://gaia.aip.de/tap")

        # define query for XP coefficients
        if isinstance(source_ids, (int, str)):
            source_ids = [source_ids]
        id_string = ", ".join(map(str, source_ids))
        GAIA_XP_QUERY = f"""
            SELECT * FROM gaiadr3.xp_continuous_mean_spectrum
            WHERE source_id IN ({id_string})
        """

        # summit query
        job = tap_service.run_sync(GAIA_XP_QUERY)

        # parse pandas dataframe and strip masked arrays
        coeffs = job.to_table().to_pandas()
        for col in coeffs.columns:
            if coeffs[col].dtype == 'object':
                coeffs[col] = coeffs[col].apply(lambda x: np.array(x.data) if isinstance(x, np.ma.MaskedArray) else x)

        return coeffs

    def cache_xp_spectra(self, coeffs, convert_to_cgs=True):
        """Calibrate Gaia XP coefficients to spectra and save them to cache.

        Parameters
        ----------
        coeffs : pandas.DataFrame
            Result of :meth:`fetch_xp_coeffs` containing XP coefficients.
        convert_to_cgs : bool, optional
            If True, convert outputs to CGS units.

        Returns
        -------
        tuple[ndarray, ndarray]
            Wavelength sampling and stacked spectrum array for the last source.
        """
        # calibrate gaia XP coefficients into spectra
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            spectra_xp = []
            print(coeffs)
            coeffs_list = np.split(coeffs, coeffs.shape[0])
            for coeff in coeffs_list:
                spectrum_xp, wave_xp = gaiaxpy.calibrate(coeff, sampling=self._wave_sampling,
                                                         truncation=False, save_file=True, output_path=self.cache_dir,
                                                         output_file=f"gaia_spec_{coeff.iloc[0].source_id}", output_format="csv")
                spectra_xp.append(spectrum_xp.loc[0, "flux"])
        spectra_xp = np.row_stack(spectra_xp)

        if convert_to_cgs:
            return self._convert_to_csg(wave_xp, spectra_xp)

        return wave_xp, spectra_xp

    def load_xp_spectra(self, source_ids, convert_to_cgs=True):
        """Load cached XP spectra for one or more Gaia source IDs.

        Parameters
        ----------
        source_ids : int or sequence
            Gaia source identifier or identifiers.
        convert_to_cgs : bool, optional
            If True, convert loaded spectra to CGS units.

        Returns
        -------
        tuple[ndarray, ndarray]
            Wavelength sampling and one- or two-dimensional flux array.
        """

        if isinstance(source_ids, (list, tuple, set, np.ndarray)):
            spectra_xp = []
            for source_id in source_ids:
                wave_xp, spectrum = self.load_xp_spectra(source_ids=source_id, convert_to_cgs=False)
                spectra_xp.append(spectrum)
            spectra_xp = np.row_stack(spectra_xp)

            if convert_to_cgs:
                return self._convert_to_csg(wave_xp, spectra_xp)

            return wave_xp, spectra_xp

        if not self._xp_spectrum_exists(source_ids):
            raise ValueError(f"{source_ids = } not present in cache directory {self.cache_dir}")

        wavelength_path, spectrum_path = self._get_xp_spectrum_paths(source_ids)
        wave_xp = Table.read(wavelength_path, format="csv")
        spectrum_xp = Table.read(spectrum_path, format="csv")

        # make numpy arrays from whatever weird objects the Gaia stuff creates
        wave_xp = np.fromstring(wave_xp["pos"][0][1:-1], sep=",")
        spectrum_xp = np.atleast_2d(np.fromstring(spectrum_xp["flux"][0][1:-1], sep=","))

        if convert_to_cgs:
            return self._convert_to_csg(wave_xp, spectrum_xp)
        return wave_xp, spectrum_xp

    def fetch_sources(self, label, ra, dec, lim_mag=14.0, n_ids=15, ignore_cache=False):
        """Fetch nearby Gaia sources with XP spectra and cache the query results.

        Parameters
        ----------
        label : str
            Cache label used to write the source ID file.
        ra : float
            Tile center right ascension in degrees.
        dec : float
            Tile center declination in degrees.
        lim_mag : float, optional
            Maximum Gaia G magnitude for selected sources.
        n_ids : int, optional
            Number of sources to retrieve.
        ignore_cache : bool, optional
            If True, ignore cached source lists and refetch from Gaia.

        Returns
        -------
        astropy.table.Table
            Gaia source table for the selected nearby objects.
        """

        IFU_INNER_RADIUS = np.sqrt(3.0)/2 * (30.2/2) / 60.0
        TILE_GAIA_QUERY = f"""
            SELECT TOP {n_ids} * FROM gaiadr3.gaia_source_lite WHERE
            DISTANCE({ra}, {dec}, ra, dec) < {IFU_INNER_RADIUS}
            AND phot_g_mean_mag < {lim_mag} AND has_xp_continuous = 'True' ORDER BY phot_g_mean_mag ASC"""

        cache_path = self.cache_dir / f"{label}_ids.ecsv"
        if not ignore_cache and cache_path.exists():
            gaia_table = Table.read(cache_path, format="ascii.ecsv")
        else:
            job = Gaia.launch_job(TILE_GAIA_QUERY)
            gaia_table = job.get_results()
            gaia_table.write(cache_path, format="ascii.ecsv", serialize_method="data_mask", overwrite=True)

        # clean up columns
        cols = gaia_table.colnames
        new_cols = [col.lower() for col in cols]
        gaia_table.rename_columns(cols, new_cols)

        return gaia_table

    def fetch_xp_spectra(self, source_ids, ignore_cache=False):
        """Fetch and cache XP spectra for requested Gaia source IDs.

        Parameters
        ----------
        source_ids : sequence
            Gaia source identifiers whose spectra should be retrieved.
        ignore_cache : bool, optional
            If True, force download even if cached spectra exist.

        Returns
        -------
        list
            Source IDs that were newly downloaded and cached.
        """

        # identify new sources if any present in cache or ignore cache and download all
        new_ids = source_ids if ignore_cache else self._not_in_cache(source_ids)
        if len(new_ids) != 0:
            log.info(f"downloading {len(new_ids)} new XP spectra coefficients")
            coeffs = self.fetch_xp_coeffs(new_ids)
            self.cache_xp_spectra(coeffs)

        return new_ids


def get_xp_spectra_from_tile(expnum, ra, dec, lim_mag=14.0, n_spectra=15, return_table=False, cache_only=False, convert_to_cgs=True, cache_dir="./gaia_cache", ignore_cache=False):
    """Retrieve Gaia XP spectra for a tile centered at RA/Dec.

    Parameters
    ----------
    expnum : str
        Exposure number or tile label used for cache naming.
    ra : float
        Right ascension of the tile center in degrees.
    dec : float
        Declination of the tile center in degrees.
    lim_mag : float, optional
        Maximum Gaia G magnitude for returned sources.
    n_spectra : int, optional
        Number of Gaia sources to retrieve.
    return_table : bool, optional
        If True, also return the Gaia source table.
    cache_only : bool, optional
        If True, only download and cache spectra without loading them.
    convert_to_cgs : bool, optional
        If True, convert loaded spectra to CGS units.
    cache_dir : str, optional
        Directory used for Gaia cache files.
    ignore_cache : bool, optional
        If True, ignore existing cache and refetch all data.

    Returns
    -------
    tuple
        If return_table is False, returns (wave_xp, spectra_xp).
        If return_table is True, returns (wave_xp, spectra_xp, gaia_table).
    """
    gaia = GaiaXPSpectra(cache_dir=cache_dir)

    log.info(f"fetching {n_spectra} Gaia sources for {expnum = } with G < {lim_mag} mag")
    gaia_table = gaia.fetch_sources(expnum, ra, dec, lim_mag=lim_mag, n_ids=n_spectra, ignore_cache=ignore_cache)

    # download new XP spectra
    source_ids = gaia_table["source_id"].filled().data
    new_ids = gaia.fetch_xp_spectra(source_ids, ignore_cache=ignore_cache)
    # return if only requested caching
    if cache_only:
        log.info(f"cached {len(new_ids)} for {expnum = }, returning after running with {cache_only = }")
        return

    # load XP spectra, only the old ones
    log.info(f"loading {len(source_ids)} Gaia XP spectra with {convert_to_cgs = }")
    wave_xp, spectra_xp = gaia.load_xp_spectra(source_ids, convert_to_cgs=convert_to_cgs)

    if return_table:
        return wave_xp, spectra_xp, gaia_table
    return wave_xp, spectra_xp


def get_xp_spectra_from_ids(source_ids, cache_only=False, convert_to_cgs=True, cache_dir="./gaia_cache", ignore_cache=False):
    """Fetch Gaia XP spectra for the specified source IDs.

    Parameters
    ----------
    source_ids : int or sequence
        Gaia source identifier(s) to retrieve.
    cache_only : bool, optional
        If True, only cache the spectra and do not load them.
    convert_to_cgs : bool, optional
        If True, convert loaded spectra to CGS units.
    cache_dir : str, optional
        Directory used for Gaia cache files.
    ignore_cache : bool, optional
        If True, re-download spectra even if cached.

    Returns
    -------
    tuple[ndarray, ndarray]
        Wavelength sampling and spectra array unless cache_only is True.
    """

    if isinstance(source_ids, int):
        source_ids = [source_ids]

    gaia = GaiaXPSpectra(cache_dir=cache_dir)

    # download new XP spectra
    new_ids = gaia.fetch_xp_spectra(source_ids, ignore_cache=ignore_cache)

    # return if only requested caching
    if cache_only:
        log.info(f"cached {len(new_ids)}, returning after running with {cache_only = }")
        return

    # load XP spectra, only the old ones
    log.info(f"loading {len(source_ids)} Gaia XP spectra with {convert_to_cgs = }")
    wave_xp, spectra_xp = gaia.load_xp_spectra(source_ids, convert_to_cgs=convert_to_cgs)

    return wave_xp, spectra_xp


def get_stellar_params(source_ids):
    """Retrieve stellar physical parameters for Gaia source IDs.

    This function first attempts to read stellar parameters from a local
    calibration table in the masters directory. If any requested source IDs are
    missing from the local table, it queries Gaia DR3 astrophysical parameters
    and falls back to a dummy parameter table if the external query fails.

    Parameters
    ----------
    source_ids : sequence
        Gaia source identifiers for which to retrieve stellar parameters.

    Returns
    -------
    pandas.DataFrame
        Stellar parameters indexed by source_id, including teff_gspspec,
        logg_gspspec, and mh_gspspec.
    """
    # Read Calibration GAIA stars table and create index on source_id for quick
    # record retrieval
    # https://sdss-wiki.atlassian.net/wiki/spaces/LVM/pages/14460157/Calibration+Stars
    params_path = pathlib.Path(MASTERS_DIR) / "stellar_models" / "lvm-many_Gaia_stars_5-9_ftype_v4-all.fits"

    SOURCE_IDS = ", ".join(map(str, source_ids))
    COLUMN_NAMES = ["source_id", "teff_gspspec", "logg_gspspec", "mh_gspspec"]
    DUMMY_TABLE = pd.DataFrame(index=source_ids, columns=COLUMN_NAMES[1:])
    DUMMY_TABLE.index.name = "source_id"

    gaia_stars = Table.read(params_path, format='fits').to_pandas()
    gaia_stars = gaia_stars.filter(items=COLUMN_NAMES)
    gaia_stars.set_index('source_id', drop=True, inplace=True)

    # Try to get stellar parameters from the local table first
    try:
        # Used indexed column source_id, See where table was read
        stellar_params = gaia_stars.loc[source_ids]
        log.info(f"found all {len(source_ids)} standard stars physical parameters in {params_path}")
    except KeyError as e:
        log.warning(e.args[0])
        # If entry not found in local table, then call external Gaia service
        try:
            job = Gaia.launch_job(f"SELECT source_id, teff_gspspec, logg_gspspec, mh_gspspec FROM gaiadr3.astrophysical_parameters WHERE source_id IN ({SOURCE_IDS})")
            stellar_params = job.get_results().to_pandas().set_index("source_id")
            stellar_params = stellar_params.loc[source_ids]
            log.info(f"fetched {len(source_ids)} standard stars stellar parameters")
        except Exception as e:
            log.warning(f"{e}, returning dummy parameters")
            stellar_params = DUMMY_TABLE.copy()
    return stellar_params


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
            If None, uses default: MASTERS_DIR/stellar_models/lvm-model_transmission_Palace_SkyModel_step0.2-all.fits
        """
        self._load_sky_model(skymodel_path)

    def _load_sky_model(self, skymodel_path=None):
        """Load the Palace Sky Model transmission table."""
        if skymodel_path is None:
            model_path = os.path.join(MASTERS_DIR, 'stellar_models', 'lvm-model_transmission_Palace_SkyModel_step0.2-all.fits')
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