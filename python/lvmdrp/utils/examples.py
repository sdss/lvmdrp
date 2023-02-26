

import os
import subprocess
import zipfile

from astropy.table import Table
from astropy.io import fits
from functools import lru_cache
from tqdm import tqdm

from . import logger

examples_logger = logger.get_logger(name=__name__)


def parse_sdr_name(sdr_name):
    """Return camera and expnum from a given raw frame path/name"""
    camera, expnum = os.path.basename(sdr_name).split(".")[0].split("-")[1:]
    return camera, expnum

def fetch_example_data(url, compressed_name, dest_path):
    """Download 2D examples data"""
    if not os.path.exists(os.path.join(dest_path, "data", "data_simulator")):
        compressed_path = os.path.join(dest_path, compressed_name)
        examples_logger.info(f"downloading example data to {compressed_path}")
        process = subprocess.Popen(f"curl {url} --output {compressed_path}".split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
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
        with zipfile.ZipFile(compressed_path, "r") as src_compressed:
            src_compressed.extractall(dest_path)
        os.remove(compressed_path)
    else:
        examples_logger.info("example data already exists")

@lru_cache(maxsize=128)
def get_frames_metadata(path):
    """Return astropy.table.Table containing useful metadata from 2D raw frames"""
    frames = [os.path.join(root, frame_name) for root, _, frame_names in os.walk(path) for frame_name in frame_names if frame_name.endswith(".fit.gz")]
    examples_logger.info(f"extracting metadata from {len(frames)} frames")
    frames_table = Table(names=["imagetyp", "spec", "camera", "expnum", "exptime", "path"], dtype=[str, str, str, str, float, str])
    for frame_path in tqdm(frames, ascii=True):
        header = fits.getheader(frame_path, ext=0)
        exptime = header["EXPTIME"]
        camera, expnum = parse_sdr_name(frame_path)
        spec = f"sp{camera[-1]}"
        imagetyp = frame_path.split(os.sep)[-2]
        frames_table.add_row([imagetyp, spec, camera, expnum, exptime, frame_path])
    examples_logger.info("successfully extracted metadata")
    return frames_table