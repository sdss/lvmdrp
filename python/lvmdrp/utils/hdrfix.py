#!/usr/bin/env python
# encoding: utf-8

import functools
import os
import pathlib
import fnmatch
import yaml

import pandas as pd
from astropy.io import fits
from lvmdrp import path, log


def get_hdrfix_path(mjd: int) -> str:
    """ Get a header fix file path

    Builds the path the lvmHdrFix file for a given
    MJD in LVMCORE.

    Parameters
    ----------
    mjd : int
        The MJD to look up

    Returns
    -------
    str
        the path to the header fix file

    Raises
    ------
    ValueError
        when the LVMCORE_DIR envvar is not properly set
    """
    if lvmcore := os.getenv("LVMCORE_DIR"):
        return pathlib.Path(lvmcore) / "hdrfix" / f"{mjd}/lvmHdrFix-{mjd}.yaml"
    else:
        raise ValueError('LVMCORE_DIR environment variable not found.  Please set up the repo.')


# @functools.lru_cache(maxsize=256)
def read_hdrfix_file(mjd: int) -> pd.DataFrame:
    """ Read a header fix file

    Reads a header fix file and returns it as a
    pandas dataframe.  Each row in the df is an
    item in the "fixes" list of hdrFix yaml file.

    Parameters
    ----------
    mjd : int
        The MJD to look up

    Returns
    -------
    pd.DataFrame
        the contents of the header fix file
    """
    # get the file path
    path = get_hdrfix_path(mjd)
    if not path.exists():
        return

    # read the file and convert it to a pandas dataframe
    with open(path, 'r') as f:
        return pd.json_normalize(yaml.safe_load(f)['fixes'])


def write_hdrfix_file(mjd: int, fileroot: str, keyword: str, value: str):
    """ Write or update a new header fix file

    Write a new, or update an existing, header fix file. For the given
    input MJD, add a new header fix entry for the given ``fileroot`` frame
    match pattern, and new header ``keyword`` and ``value``.

    Parameters
    ----------
    mjd : int
        The MJD to update or write
    fileroot : str
        The fileroot pattern
    keyword : str
        The header keyword to add
    value : str
        The new value of the header key
    """
    # get the file path
    path = get_hdrfix_path(mjd)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    read_hdrfix_file.cache_clear()
    fix = read_hdrfix_file.__wrapped__(mjd)
    if fix is None or fix.empty:
        fix = pd.DataFrame.from_dict([{'fileroot': fileroot, 'keyword': keyword, 'value': value}])
    else:
        new_fix = {'fileroot': fileroot, 'keyword': keyword, 'value': value}
        if (fix == new_fix).all(1).any():
            log.info(f"skipping header fix for {mjd = }, {fileroot = }: {keyword} = '{value}', already exist")
            return
        fix.loc[len(fix)] = new_fix

    # get schema
    schema = [{'description': 'the raw frame file root with * as wildcard', 'dtype': 'str',
               'name': 'fileroot'},
              {'description': 'the name of the header keyword to fix', 'dtype': 'str',
               'name': 'keyword'},
              {'description': 'the value of the header keyword to update', 'dtype': 'str',
               'name': 'value'}]

    # get data
    data = fix.to_dict(orient='records')
    data = {'schema': schema, 'fixes': data}

    # write the file
    with open(path, 'w+') as f:
        f.write(yaml.safe_dump(data, sort_keys=False, indent=2))
    log.info(f"created header fix for {mjd = }, {fileroot = }: {keyword} = '{value}'")


def apply_hdrfix(mjd: int, camera: str = None, expnum: int = None,
                 filename: str = None, hdr: fits.Header = None) -> fits.Header:
    """ Apply a header fix file to a header object

    Applies any header corrections found for the given file or MJD
    Corrections include any incorrect or missing header keywords.

    The input must be either a filepath to the raw sdR frame, an MJD +
    header object, or the MJD + camera + exposre number of the raw sdR
    frame file.

    Parameters
    ----------
    mjd : int
        the MJD of the observation
    camera : str, optional
        the camera of the observation, by default None
    expnum : int, optional
        the exposure number of the observation, by default None
    filename : str, optional
        the filepath to the raw sdR frame file, by default None
    hdr : fits.Header, optional
        the FITS primary header object, by default None

    Returns
    -------
    fits.Header
        a corrected FITS primary header object

    Raises
    ------
    ValueError
        when the necessary inputs are not supplied
    """

    if filename:
        matches = path.extract('lvm_raw', filename)
        mjd = int(matches.get('mjd'))
        camera = matches.get('camspec')
        expnum = int(matches.get('expnum'))
        hdr = fits.getheader(filename)
    elif hdr:
        # assume these are correct in the headers, may need to change this
        # mjd = hdr.get("MJD")
        expnum = hdr.get("EXPOSURE")
        camera = hdr.get("CCD")
    elif not (mjd and camera and expnum):
        raise ValueError('Either filename, hdr, or mjd, camera, expnum must be specified.')

    # always check the hdr tile id and correct it
    # get the tile id; set null tile ids -999 to 11111
    tileid = hdr.get("TILE_ID") or hdr.get("TILEID", 11111)
    tileid = 11111 if tileid in (-999, 999, None) else tileid
    hdr['TILE_ID'] = tileid

    # add QA header keywords default values
    hdr['QAQUAL'] = ('GOOD', 'string value for raw data quality flag')
    hdr['QAFLAG'] = ('0000', 'bitmask value for raw data quality flag')

    # read the hdr fix file
    fix = read_hdrfix_file(mjd)

    # if not hdr fix file, exit
    if fix is None or fix.empty:
        return hdr

    # Create the current file string
    hemi = 's' if hdr['OBSERVAT'] == 'LCO' else 'n'
    current_file = f'sdR-{hemi}-{camera}-{expnum:0>8}'

    # Apply the header fixes
    for _, row in fix.iterrows():
        if fnmatch.fnmatch(current_file, row['fileroot']):
            hdr[row['keyword']] = row['value']
            log.info(f'Applying header fix on {current_file} for key: {row["keyword"]}, value: {row["value"]}.')

    # fix typing in QAFLAG keyword
    # hdr['QAQUAL'] = ('GOOD', 'string value for raw data quality flag')
    # hdr.set('QAFLAG', QAFlag(int(hdr['QAFLAG'], base=2)), 'bitmask value for raw data quality flag')

    return hdr
