#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
import re
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


def write_hdrfix_file(mjd: int):
    # get the file path
    path = get_hdrfix_path(mjd)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    #data = df.to_dict(orient='records')
    data = {}

    # write the file
    with open(path, 'w') as f:
        f.write(yaml.safe_dump(data))


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
        mjd = hdr.get("MJD")
        expnum = hdr.get("EXPOSURE")
        camera = hdr.get("CCD")
    elif not (mjd and camera and expnum):
        raise ValueError('Either filename, hdr, or mjd, camera, expnum must be specified.')

    # read the hdr fix file
    fix = read_hdrfix_file(mjd)

    # if not hdr fix file, exit
    if fix is None or fix.empty:
        return hdr

    # find matching files
    for fileroot, key, val in zip(fix['fileroot'], fix['keyword'], fix['value']):
        root = f'60010/{fileroot}{"" if fileroot.endswith("*") else "*"}'
        files = pathlib.Path(os.getenv('LVM_DATA_S')).rglob(root)

        pattern = re.compile(f'{camera}-{expnum:0>8}')
        sub = filter(pattern.search, map(str, files))

        # apply the header fix
        for file in sub:
            stem = pathlib.Path(file).stem
            log.info(f'Applying header fix on {stem} for key: {key}, value: {val}.')
            hdr[key] = val

    return hdr

