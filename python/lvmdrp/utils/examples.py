import os
import pickle
import subprocess
import zipfile
from glob import glob
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from lvmdrp.utils import logger


examples_logger = logger.get_logger(name=__name__)


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
    os.makedirs(dest_path, exist_ok=True)
    if not os.path.exists(os.path.join(dest_path, name)):
        examples_logger.info(f"downloading example data to {file_path}")
        process = subprocess.Popen(
            f"curl {url}/{file_name} --output {file_path}".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        for stdout_line in iter(process.stdout.readline, ""):
            examples_logger.info(stdout_line[:-1])
        process.stdout.close()
        returncode = process.wait()
        if returncode == 0:
            examples_logger.info("successfully downloaded example data")
        else:
            examples_logger.error("error while downloading example data")
            examples_logger.error("full report:")
            examples_logger.error(process.stderr.decode("utf-8"))
        with zipfile.ZipFile(file_path, "r") as src_compressed:
            src_compressed.extractall(dest_path)
        os.remove(file_path)
    else:
        examples_logger.info("example data already exists")


def get_frames_metadata(path, suffix=".fits.gz", ignore_cache=False):
    """Return astropy.table.Table containing useful metadata from 2D raw frames"""
    CACHE_PATH = os.path.join(path, "frames_table.pkl")
    if os.path.isfile(CACHE_PATH) and not ignore_cache:
        examples_logger.info(f"loading cached metadata from '{CACHE_PATH}'")
        return pickle.load(open(CACHE_PATH, "rb"))

    frames = [
        os.path.join(root, frame_name)
        for root, _, frame_names in os.walk(path)
        for frame_name in frame_names
        if frame_name.endswith(suffix)
    ]
    examples_logger.info(f"extracting metadata from {len(frames)} frames")
    frames_table = Table(
        names=[
            "imagetyp",
            "spec",
            "mjd",
            "camera",
            "expnum",
            "exptime",
            "argon",
            "neon",
            "ldls",
            "hgne",
            "xenon",
            "path",
        ],
        dtype=[str, str, int, str, str, float, bool, bool, bool, bool, bool, str],
    )
    for frame_path in tqdm(frames, ascii=True):
        try:
            header = fits.getheader(frame_path, ext=0)
        except Exception:
            examples_logger.error(f"error while reading frame '{frame_path}'")
            continue
        mjd = header.get("MJD")
        imagetyp = header.get("FLAVOR", header.get("IMAGETYP"))
        camera, expnum = parse_sdr_name(frame_path)
        spec = f"sp{camera[-1]}"
        exptime = header["EXPTIME"]
        argon = header.get("ARGON", "OFF") == "ON"
        neon = header.get("NEON", "OFF") == "ON"
        ldls = header.get("LDLS", "OFF") == "ON"
        hgne = header.get("HGNE", "OFF") == "ON"
        xenon = header.get("XENON", "OFF") == "ON"
        frames_table.add_row(
            [
                imagetyp,
                spec,
                mjd,
                camera,
                expnum,
                exptime,
                argon,
                neon,
                ldls,
                hgne,
                xenon,
                frame_path,
            ]
        )
    examples_logger.info("successfully extracted metadata")

    examples_logger.info(f"caching metadata to '{CACHE_PATH}'")
    pickle.dump(frames_table, open(CACHE_PATH, "wb"))

    return frames_table


def get_masters_metadata(
    path_pattern, mjd=None, kind=None, camera=None, exptime=None, ext="fits"
):
    """return master metadata given a path where master calibration frames are stored"""
    masters_path = path_pattern.format(
        mjd=mjd or "*",
        kind=kind or "*",
        camera=camera or "*",
        exptime=exptime or "*",
        ext=ext,
    )
    masters_path = glob(masters_path)

    metadata = []
    for path in masters_path:
        p = os.path.basename(path).split(".")[0]
        mjd, kind, camera, exptime = p.split("-")[1:]
        metadata.append(
            [int(mjd) if mjd != "super" else mjd, kind, camera, float(exptime), path]
        )
    return pd.DataFrame(
        columns=["mjd", "kind", "camera", "exptime", "path"], data=metadata
    )
