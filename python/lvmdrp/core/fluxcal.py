# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez (adapted from MaNGA IDL code)
# @Date: Jan 27, 2023
# @Filename: fluxcal.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import numpy as np
from scipy import signal

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u

from lvmdrp import log


def retrieve_header_stars(rss):
    """
    Retrieve fiber, Gaia ID, exposure time and airmass for the 12 standard stars in the header.
    return a list of tuples of the above quatities.
    """
    lco = EarthLocation(
        lat=-29.008999964 * u.deg, lon=-70.688663912 * u.deg, height=2800 * u.m
    )
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
                continue
            fiber = h[stdi + "FIB"]
            obstime = Time(h[stdi + "T0"])
            exptime = h[stdi + "EXP"]
            c = SkyCoord(float(h[stdi + "RA"]), float(h[stdi + "DE"]), unit="deg")
            stdT = c.transform_to(AltAz(obstime=obstime, location=lco))
            secz = stdT.secz.value
            # print(gid, fib, et, secz)
            stddata.append((i + 1, fiber, gaia_id, exptime, secz))
    return stddata


def mean_absolute_deviation(vals):
    """
    Robust estimate of RMS
    - see https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    mval = np.nanmedian(vals)
    rms = 1.4826 * np.nanmedian(np.abs(vals - mval))
    return mval, rms
    # ok=np.abs(vals-mval)<4*rms


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    normal_cutoff = float(cutoff_freq) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype="lowpass")
    y = signal.filtfilt(b, a, data)
    return y


def filter_channel(w, f, k=3):
    c = np.where(np.isfinite(f))
    s = butter_lowpass_filter(f[c], 0.01, 2)
    res = s - f[c]
    # plt.plot(w[c], f[c], 'k.')
    # plt.plot(w[c], s, 'b-')
    mres, rms = mean_absolute_deviation(res)
    good = np.where(np.abs(res - mres) < k * rms)
    # plt.plot(w[c][good], f[c][good], 'r.', markersize=5)
    return w[c][good], f[c][good]

