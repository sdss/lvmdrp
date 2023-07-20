#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
import pickle
import re
import subprocess
import zipfile
from typing import Union
from glob import glob

import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from lvmdrp import log
from lvmdrp.utils.hdrfix import apply_hdrfix


FC = [0.88, 0.94]
GAINS = {
    "b": [1.018, 1.006, 1.048, 1.048],
    "r": [1.4738, 1.5053, 1.9253, 1.5122],
    "z": [1.4738, 1.5053, 1.9253, 1.5122],
}
RDNOISES = {
    "b": 4 * [2.0 * FC[0] * 0.56],
    "r": 4 * [2.0 * FC[0] * 0.56],
    "z": 4 * [2.0 * FC[0] * 0.56],
}


def parse_sdr_name(sdr_name):
    """Return camera and expnum from a given raw frame path/name"""
    name_parts = os.path.basename(sdr_name).split(".")[0].split("-")[1:]
    if len(name_parts) == 2:
        camera, expnum = name_parts
    elif len(name_parts) == 3:
        hemi, camera, expnum = name_parts
    else:
        raise ValueError(f"unkown name formatting {os.path.basename(sdr_name)}")
    return camera, expnum


def fetch_example_data(url, name, dest_path, ext="zip"):
    """Download 2D examples data"""
    file_name = f"{name}.{ext}"
    file_path = os.path.join(dest_path, file_name)

    if os.path.isdir(dest_path)==False:
        log.info("Creating destination directory %s" % (dest_path))
        os.makedirs(dest_path)

    if os.path.exists(os.path.join(dest_path, name)):
        log.info("example data already exists")
        return

    log.info(f"downloading example data to {file_path}")
    process = subprocess.Popen(
        f"curl {url}/{file_name} --output {file_path}".split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    for stdout_line in iter(process.stdout.readline, ""):
        log.info(stdout_line[:-1])
    process.stdout.close()
    returncode = process.wait()
    if returncode == 0:
        log.info("successfully downloaded example data")
    else:
        log.error("error while downloading example data")
        log.error("full report:")
        log.error(process.stderr.decode("utf-8"))
    with zipfile.ZipFile(file_path, "r") as src_compressed:
        src_compressed.extractall(dest_path)
    os.remove(file_path)


def get_frames_metadata(mjd: Union[str, int] = None, path: str = None, suffix: str = "fits",
                        ignore_cache: bool = False) -> Table:
    """ Extract metadata from the 2d raw frames

    Builds an Astropy table containing extracted metadata for each of the 2d raw sdR
    frame files.  Globs for all files in the ``mjd`` subdirectory of the raw LVM data.
    If no mjd is specified, searches all of them. Can optionally override the raw
    data "LVM_DATA_S" directory with the ``path`` keyword argument.  Writes the table
    to a pickle cache and if found, will load the content from there.  The cache is written
    to either the LVM_SANDBOX folder or the user home directory.

    Parameters
    ----------
    mjd : Union[str, int], optional
        The MJD of the data sub-directory to search in, by default None
    path : str, optional
        The raw data path to override the default, by default None
    suffix : str, optional
        The raw data file suffix, by default "fits"
    ignore_cache : bool, optional
        Flag to ignore the pickle cache, by default False

    Returns
    -------
    astropy.Table
        An astropy Table of metadata
    """

    # load from a cache file
    cache_path = pathlib.Path(os.getenv("LVM_SANDBOX", os.path.expanduser('~')))
    cache_file = cache_path / f"lvm_metadata_{mjd}.pkl"
    if cache_file.exists() and not ignore_cache:
        log.info(f"loading cached metadata from '{cache_file}'")
        return pickle.load(open(cache_file, "rb"))

    # look up raw data in the relevant MJD path
    raw_data_path = path or os.getenv("LVM_DATA_S")
    raw_frame = f'{mjd}/sdR*{suffix}*' if mjd else f'*/sdR*{suffix}*'
    frames = list(pathlib.Path(raw_data_path).rglob(raw_frame))

    # build the table data
    frames_table = Table(
        names=["imagetyp", "spec", "mjd", "camera", "expnum", "exptime", "tileid",
               "quality", "path", "name"],
        dtype=[str, str, int, str, str, float, int, str, str, str],
    )
    for frame_path in tqdm(frames, ascii=True, unit='files', total=len(frames)):
        try:
            header = fits.getheader(frame_path, ext=0)
        except OSError as e:
            log.error(f'Cannot read FITS header: {e}')
            continue
        mjd = header.get("MJD")

        # apply any header fix or if none, use old header
        header = apply_hdrfix(mjd, hdr=header) or header

        # get certain keys
        imagetyp = header.get("FLAVOR", header.get("IMAGETYP"))
        camera, expnum = parse_sdr_name(frame_path)
        spec = f"sp{camera[-1]}"
        exptime = header["EXPTIME"]
        tileid = header.get("TILEID", 1111)
        qual = header.get("QUALITY", "excellent")
        frames_table.add_row([imagetyp, spec, mjd, camera, expnum, exptime,
                              tileid, qual, str(frame_path), frame_path.stem])
    log.info(f"successfully extracted metadata for {len(frames_table)} frames.")

    # create the cache file
    if not cache_file.parent.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"caching metadata to '{cache_file}'")
    pickle.dump(frames_table, open(cache_file, "wb"))

    return frames_table


def get_masters_metadata(path_pattern, **kwargs):
    """return master metadata given a path where master calibration frames are stored"""
    path_params = re.findall(r"\{(\w+)\}", path_pattern)
    params = dict.fromkeys(path_params, "*")
    params.update(kwargs)

    masters_path = path_pattern.format(**params)
    masters_path = glob(masters_path)

    metadata = []
    for path in masters_path:
        name = os.path.basename(path).split(".")[0]
        metadata.append(name.split("-")[1:])

    metadata = pd.DataFrame(columns=path_params, data=metadata)
    metadata = metadata.apply(lambda s: pd.to_numeric(s, errors="ignore"), axis="index")
    metadata["path"] = masters_path

    return metadata


def fix_lamps_metadata(metadata, lamp_names, inplace=True):
    """fix arc lamps to be ON for consistent exposure numbers

    Parameters
    ----------
    metadata : pd.DataFrame
        frames metadata
    lamp_names : list_like
        list of names for lamps found in the metadata
    inplace : bool, optional
        whether fix is in place or not, by default True

    Returns
    -------
    pd.DataFrame
        frames metadata with arc lamps fixed
    """
    if inplace:
        md = metadata
    else:
        md = metadata.copy()

    for lamp_name in lamp_names:
        # get unique exposure number where lamp_name is ON
        expnums = md[md[lamp_name]].expnum.drop_duplicates().tolist()
        # set lamp_name to ON for defined expnums
        md.loc[md.expnum.isin(expnums), lamp_name] = True

    if not inplace:
        return md
