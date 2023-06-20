#!/usr/bin/env python
# encoding: utf-8

from astropy.time import Time


def dateobs_to_mjd(value: str) -> float:
    """ Convert observation time to MJD

    Converts isoformat datetime to an MJD.  For example,
    '2023-06-19T22:52:00.981' from a OBSTIME or DATEOBS
    header keyword.

    Parameters
    ----------
    value : str
        the isoformat datetime

    Returns
    -------
    float
        the Modified Julian Date
    """
    return Time(value).mjd


def dateobs_to_sjd(value: str) -> float:
    """ Convert observation time to SJD

    Converts isoformat datetime to an SJD.  For example,
    '2023-06-19T22:52:00.981' from a OBSTIME or DATEOBS
    header keyword.


    Parameters
    ----------
    value : str
        the isoformat datetime

    Returns
    -------
    float
        the SDSS Julian Date
    """
    return mjd_to_sjd(dateobs_to_mjd(value))


def mjd_to_sjd(value: float) -> float:
    """ Convert the Modified Julian Date to SDSS Julian Date

    Parameters
    ----------
    value : float
        the MJD

    Returns
    -------
    float
        the SJD
    """
    return value + 0.4


def sjd_to_mjd(value: float) -> float:
    """ Convert the SDSS Julian Date to Modified Julian Date

    Parameters
    ----------
    value : float
        the SJD

    Returns
    -------
    float
        the MJD
    """
    return value - 0.4
