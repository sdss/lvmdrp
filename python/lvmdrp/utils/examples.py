#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
import pickle
import subprocess
import zipfile
from typing import Union

from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from lvmdrp import log


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
    cache_file = cache_path / "lvm_raw_frames_table.pkl"
    if cache_file.exists() and not ignore_cache:
        log.info(f"loading cached metadata from '{cache_file}'")
        return pickle.load(open(cache_file, "rb"))

    # look up raw data in the relevant MJD path
    raw_data_path = path or os.getenv("LVM_DATA_S")
    raw_frame = f'{mjd}/sdR*{suffix}*' if mjd else f'*/sdR*{suffix}*'
    frames = list(pathlib.Path(raw_data_path).rglob(raw_frame))

    # build the table data
    frames_table = Table(
        names=["imagetyp", "spec", "mjd", "camera", "expnum", "exptime", "tileid", "path"],
        dtype=[str, str, int, str, str, float, int, str],
    )
    for frame_path in tqdm(frames, ascii=True, unit='files', total=len(frames)):
        header = fits.getheader(frame_path, ext=0)
        mjd = header.get("MJD")
        imagetyp = header.get("FLAVOR", header.get("IMAGETYP"))
        camera, expnum = parse_sdr_name(frame_path)
        spec = f"sp{camera[-1]}"
        exptime = header["EXPTIME"]
        tileid = header.get("TILEID", 1111)
        frames_table.add_row([imagetyp, spec, mjd, camera, expnum, exptime, tileid, str(frame_path)])
    log.info(f"successfully extracted metadata for {len(frames_table)} frames.")

    # create the cache file
    if not ignore_cache:
        if not cache_file.parent.exists():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"caching metadata to '{cache_file}'")
        pickle.dump(frames_table, open(cache_file, "wb"))

    return frames_table
