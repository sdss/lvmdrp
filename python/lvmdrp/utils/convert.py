#!/usr/bin/env python
# encoding: utf-8

import pathlib
from typing import Union
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
    return Time(value.strip()).mjd


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


def correct_sjd(path: pathlib.Path, sjd: int) -> int:
    """ Correct the SJD

    Correct the SJD for early frames with the expected
    directory SJD.  This is only for early data cases
    where the SJD and MJD were incorrect/swapped in the
    headers, behaviors for MJDs 60007-60112.

    Parameters
    ----------
    path : pathlib.Path
        the raw frame path
    sjd : int
        the current computed SJD

    Returns
    -------
    int
        the correct SJD
    """
    path = pathlib.Path(path)

    # check for master and raw
    if not path.parent.stem.isdigit():
        # master
        exp_sjd = int(path.parent.parent.stem)
    else:
        # raw
        exp_sjd = int(path.parent.stem)
    return sjd if exp_sjd == sjd else exp_sjd


def tileid_grp(tileid: Union[int, str]) -> str:
    """ Convert a tile id to a tile group

    This is for manual use without ``sdss_access``.  If the
    group definition changes, the definition in the
    ``tilegrp`` function in ``sdss_access.path.path.py``
    also needs updating.

    The raw_metadata code uses a tileid of "*" for
    pattern matching.  In this case, we use "*XX" for
    the tile group.

    Parameters
    ----------
    tileid : Union[int, str]
        the LVM tile id

    Returns
    -------
    str
        the LVM tile id group
    """
    return '*XX' if tileid == '*' else f'{int(tileid) // 1000:0>4d}XX'

