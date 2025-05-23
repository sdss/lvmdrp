import os.path as path
import pathlib

import numpy
import requests
from astropy.table import Table
from gaiaxpy import calibrate
from scipy import interpolate
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps

from lvmdrp.core.spectrum1d import Spectrum1D


sdss_g_w = numpy.array(
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
sdss_g_f = numpy.array(
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


def spec_to_mAB(lam, spec, lamf, filt):
    """
    Calculate AB magnitude in filter (lamf, filt) given a spectrum
    (lam, spec) in ergs/s/cm^2/A
    """
    c_AAs = 2.99792458e18  # Speed of light in Angstrom/s
    filt_int = numpy.interp(lam, lamf, filt)  # Interpolate to common wavelength axis
    I1 = simps(spec * filt_int * lam, lam)
    I2 = simps(filt_int / lam, lam)
    fnu = I1 / I2 / c_AAs  # Average flux density
    mab = -2.5 * numpy.log10(fnu) - 48.6  # AB magnitude
    if numpy.isnan(mab):
        mab = -9999.9
    return mab


def spec_to_LVM_mAB(channel, w, f):
    """
    LVM photometric system: Gaussian filter with sigma 250A centered in channels
    at 4500, 6500, and 8500A
    """
    if channel == "b":
        return spec_to_mAB(w, f, w, numpy.exp(-0.5 * ((w - 4500) / 250) ** 2))
    elif channel == "r":
        return spec_to_mAB(w, f, w, numpy.exp(-0.5 * ((w - 6500) / 250) ** 2))
    else:
        return spec_to_mAB(w, f, w, numpy.exp(-0.5 * ((w - 8500) / 250) ** 2))


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
    if not numpy.any(mask):
        return y
    known_x, known_v = x[~mask], y[~mask]
    missing_x = x[mask]
    missing_idx = numpy.where(mask)

    f = interpolate.interp1d(known_x, known_v, kind=kind, fill_value=fill_value)
    yy = y.copy()
    yy[missing_idx] = f(missing_x)

    return yy


class GaiaStarNotFound(Exception):
    """
    Signal that the star has no BP-RP spectrum
    """

    pass


def retrive_gaia_star(gaiaID, GAIA_CACHE_DIR):
    """
    Load or download and load from cache the spectrum of a gaia star, converted to erg/s/cm^2/A
    """
    # create cache dir if it does not exist
    pathlib.Path(GAIA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if path.exists(GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + ".csv") is True:
        # read the tables from our cache
        gaiaflux = Table.read(
            GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + ".csv", format="csv"
        )
        gaiawave = Table.read(
            GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + "_sampling.csv", format="csv"
        )
    else:
        # need to download from Gaia archive
        CSV_URL = (
            "https://gea.esac.esa.int/data-server/data?RETRIEVAL_TYPE=XP_CONTINUOUS&ID=Gaia+DR3+"
            + str(gaiaID)
            + "&format=CSV&DATA_STRUCTURE=RAW"
        )
        FILE = GAIA_CACHE_DIR + "/XP_" + str(gaiaID) + "_RAW.csv"

        with requests.get(CSV_URL, stream=True) as r:
            r.raise_for_status()
            if len(r.content) < 2:
                raise GaiaStarNotFound(f"Gaia DR3 {gaiaID} has no BP-RP spectrum!")
            with open(FILE, "w") as f:
                f.write(r.content.decode("utf-8"))

        # convert coefficients to sampled spectrum
        _, _ = calibrate(
            FILE,
            output_path=GAIA_CACHE_DIR,
            output_file="gaia_spec_" + str(gaiaID),
            output_format="csv",
        )
        # read the flux and wavelength tables
        gaiaflux = Table.read(
            GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + ".csv", format="csv"
        )
        gaiawave = Table.read(
            GAIA_CACHE_DIR + "/gaia_spec_" + str(gaiaID) + "_sampling.csv", format="csv"
        )

    # make numpy arrays from whatever weird objects the Gaia stuff creates
    wave = numpy.fromstring(gaiawave["pos"][0][1:-1], sep=",") * 10  # in Angstrom
    flux = (
        1e7 * 1e-1 * 1e-4 * numpy.fromstring(gaiaflux["flux"][0][1:-1], sep=",")
    )  # W/s/micron -> in erg/s/cm^2/A
    return wave, flux


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
        k3 = f3 * 0.015 * numpy.exp(-((wave - 6000.0) / 1200.0))
        data = k1 + k2 + k3
        scale_idx = numpy.argsort((wave - 5500.0) ** 2)[0]
        scale_offset = extinct_v - data[scale_idx]
        data = data + scale_offset

    spec = Spectrum1D(wave=wave, data=data)
    return spec


def extinctParanal(wave):
    wave_base = numpy.concatenate(
        (
            numpy.arange(3325, 6780, 50),
            numpy.array([7060, 7450, 7940, 8500, 8675, 8850, 10000]),
        )
    )
    extinct = numpy.array(
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

    Arat = (ax + (bx / Rv)).astype(numpy.float32)
    spec = Spectrum1D(wave=wave, data=Arat)
    return spec
