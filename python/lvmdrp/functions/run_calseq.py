# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: Feb 1, 2024
# @Filename: run_calseq.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

# * learn the current calibration sequence:
#   - bias/dark
#   - pixel flat
#   - fiber flat and twilights
#   - arcs
# * calibrate sequence:
#   - bias/dark/pixflat (2D)
#   - traces (centroids and widths)
#   - fiber flat (1D)
#   - illumination corrections
#   - wavelength fitting
# * produce QA plots
# * write calibration files
#   - master bias/dark/pixflat
#   - master pixelmask
#   - master traces
#   - master fiberflat (with illumination corrections)
#

import os
import yaml
import warnings
import numpy as np
import bottleneck as bn
import pandas as pd
from glob import glob
from itertools import product
from pprint import pformat
from copy import deepcopy as copy
from datetime import datetime
from shutil import copy2, rmtree
from astropy.io import fits
from astropy.table import Table
from scipy import interpolate
from typing import Union, Tuple, List, Dict
from collections.abc import Callable
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.utils import metadata as md
from lvmdrp.utils import hdrfix
from lvmdrp.utils.convert import tileid_grp
from lvmdrp.utils.paths import get_master_mjd, get_calib_paths, group_calib_paths, get_frames_paths
from lvmdrp.core.plot import create_subplots, save_fig
from lvmdrp.core import dataproducts as dp
from lvmdrp.core.constants import (
    LVM_NFIBERS,
    LVM_NCOLS,
    LVM_NBLOCKS,
    CAMERAS,
    SPEC_CHANNELS,
    CON_LAMPS,
    ARC_LAMPS,
    LVM_REFERENCE_COLUMN,
    STD_FIBER_LABELS,
    CALIBRATION_TYPES,
    CALIBRATION_PRODUCTS,
    CALIBRATION_NEEDS,
    SKYLINES_FIBERFLAT,
    CONTINUUM_FIBERFLAT,
    MASTERS_DIR,
    PIXELSHIFTS_PATH)
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.core.image import loadImage
from lvmdrp.core.rss import RSS, lvmFrame
from lvmdrp.core.fit_profile import gaussians

from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.main import start_logging, get_config_options, read_fibermap, reduce_2d, reduce_1d
from lvmdrp.functions.run_twilights import lvmFlat, to_native_wave, fit_fiberflat, combine_twilight_sequence, fit_skyline_flatfield


SLITMAP = read_fibermap(as_table=True)
MASTER_CON_LAMPS = {"b": "ldls", "r": "ldls", "z": "quartz"}
MASK_BANDS = {
    "b": [(3910, 4000), (4260, 4330)],
    "r": [(6840,6960)],
    "z": [(7570, 7700)]
}
COUNTS_THRESHOLDS = {"ldls": 1000, "quartz": 1000}
FIBER_MEASURING_CONFIG = {
        "counts": {"mode": "lsq", "method": "dogbox", "loss": "linear", "xtol": 1e-3, "ftol": 1e-3},
        "centroids": {"mode": "lsq", "method": "dogbox", "loss": "linear", "xtol": 1e-3, "ftol": 1e-3},
        "sigmas": {"mode": "lsq", "method": "dogbox", "loss": "linear", "xtol": 1e-3, "ftol": 1e-3}}
FIBER_SMOOTHING_CONFIG = {
    "counts": ("spline", {"smoothing": None, "use_weights": True, "nsigmas": np.inf, "min_samples_frac": 0.7}),
    "centroids": ("polynomial", {"deg": 5, "nsigmas": np.inf, "min_samples_frac": 0.7}),
    "sigmas": ("polynomial", {"deg": 8, "nsigmas": np.inf, "min_samples_frac": 0.7})}


CALIBRATION_EPOCHS_PATH = os.path.join(os.getenv("LVMCORE_DIR"), "calibrations", "calibration-epochs.yaml")

STRAYLIGHT_PARS = dict(
    select_nrows=(10,10), use_weights=True, aperture=11,
    x_bins=60, x_bounds=("data","data"), y_bounds=(0.0,0.0),
    x_nbound=10, y_nbound=5, clip=(0.0,None),
    nsigma=1.0, smoothing=90, median_box=None)


def _reject_pixelshifted(frames, pixelshifts_path=PIXELSHIFTS_PATH):
    pixelshifts = pd.read_parquet(pixelshifts_path)

    # filter pixelshifts by exposure numbers in frames
    expnums = frames.expnum.unique()
    selection = pixelshifts.exp_no.isin(expnums)
    pixelshifts = pixelshifts.loc[selection]

    log.info(f"removing {len(pixelshifts)} exposures affected by pixel shifts: {pixelshifts.exp_no.values}")

    return frames.query("expnum not in @pixelshifts.exp_no")
    # return frames.query("expnum not in @pixelshifts.exp_no and spec not in @pixelshifts.spec")


def refresh_pixelshifts_file(mjd, drpver=drpver, pixelshifts_path=PIXELSHIFTS_PATH, dry_run=False):
    if not os.path.exists(pixelshifts_path):
        log.error(f"pixel shifts file not found: {pixelshifts_path}")
        return

    pixelshifts = pd.read_parquet(pixelshifts_path)
    npixelshifts = len(pixelshifts)

    paths = path.expand("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="e", imagetype="*", expnum="????????", camera="*")
    if len(paths) == 0:
        log.info(f"no pixel shifts detected for {mjd = } with DRP '{drpver}'")
        return

    params = []
    log.info(f"found {len(paths) // 3} spectrograph frames with shifted pixels")
    for p in paths:
        parts = p.replace(os.environ["LVM_SPECTRO_REDUX"]+"/", "").replace(".fits", "").split("/")
        image_params = parts[-1].split("-")
        params.append({"MJD": int(parts[3]), "exp_no": int(image_params[-1]), "spec": f"sp{image_params[2][-1]}", "exp_type": image_params[1][1:]})

    new_pixelshifts = pd.DataFrame(params).drop_duplicates(subset=["MJD", "exp_no", "spec"], ignore_index=True)
    records = new_pixelshifts.to_string(index=None).split("\n")
    for record in records:
        log.info(f"   {record}")

    pixelshifts = pd.concat((pixelshifts, new_pixelshifts), axis="index", ignore_index=True)
    pixelshifts.drop_duplicates(subset=["MJD", "exp_no", "spec"], ignore_index=True, inplace=True)
    pixelshifts.sort_values("MJD", inplace=True)
    pixelshifts.reset_index(drop=True, inplace=True)

    nadded = len(pixelshifts) - npixelshifts

    log.info(f"added {nadded} pixel shift detections to {pixelshifts_path}")
    if not dry_run:
        pixelshifts.to_parquet(pixelshifts_path)


def _get_standards_ring(ring):
    if ring == "primary":
        standards_sequence = (label for label in STD_FIBER_LABELS if label.startswith("P1"))
    elif ring == "secondary":
        standards_sequence = (label for label in STD_FIBER_LABELS if label.startswith("P2"))
    else:
        standards_sequence = STD_FIBER_LABELS.copy()
    return sorted(standards_sequence, key=lambda s: int(s.split("-")[1]))


def get_standards_sequence(frames, camera, ring="primary"):

    slitmap = SLITMAP.copy()
    slitmap = slitmap[slitmap["spectrographid"] == int(camera[1])]

    # all standards
    standards = set(slitmap[slitmap["telescope"] == "Spec"]["orig_ifulabel"])
    # targeted standards in all spectrographs (i.e., all standards in a given ring or both rings)
    standards_ring = set(_get_standards_ring(ring=ring))
    # selection of standards present in the current camera spectrograph
    standards_selection = standards.intersection(standards_ring)

    return frames.query("camera == @camera and calibfib in @standards_selection")


def choose_sequence(frames, calibration, kind="longterm", ring="primary"):
    """Chooses the right calibration frame sequence depending on the calibration type and length

    Parameters
    ----------
    frames : pd.DataFrame
        Data frame containing the calibrations metadata
    calibration : str
        Calibration type, either 'bias', 'trace', 'wave', 'dome' or 'twilight'
    kind : str, optional
        Calibration sequence length, either 'nightly' (short) or 'longterm' (long), by default "longterm"
    ring : str, optional
        Standard fibers ring to select, either 'primary', 'secondary', 'both'. By default 'primary'

    Returns
    -------
    pd.DataFrame
        Data frame containing the selection of individual exposures

    Raises
    ------
    ValueError
        `ring` has the incorrect value ('primary', 'secondary' or 'both')
    ValueError
        `calibration` has the incorrect value ('bias', 'trace', 'wave', 'dome' or 'twilight')
    ValueError
        `kind` has the incorrect value ('nightly', 'longterm')
    """

    # TODO: merge per camera iterator output (get_standards_sequence)
    # TODO: implement possibility to give a selection of exposure numbers

    if ring not in {"primary", "secondary", "both"}:
        raise ValueError(f"Invalid value for `ring`: {ring}. Expected either 'primary', 'secondary' or 'both'")
    if not isinstance(calibration, str) or calibration not in CALIBRATION_TYPES:
        raise ValueError(f"Invalid value for `calibration`: {calibration}. Expected one of {','.join(CALIBRATION_TYPES)}")
    if not isinstance(kind, str) or kind not in {"nightly", "longterm"}:
        raise ValueError(f"Invalid value for `kind`: {kind}. Expected either 'longterm' or 'nightly'")

    nstandards = 12 if ring in {"primary", "secondary"} else 24
    EXPECTED_SEQUENCE_LENGTH = {
        "bias": 7,
        "trace": 2 if kind == "nightly" else nstandards,
        "dome": 2 if kind == "nightly" else nstandards,
        "wave": 2 if kind == "nightly" else nstandards,
        "twilight": nstandards}

    # reject frames affected by shifted/missing pixels
    chosen_frames = _reject_pixelshifted(frames)

    if kind == "nightly" and calibration != "twilight":
        chosen_frames = chosen_frames.head(EXPECTED_SEQUENCE_LENGTH[calibration])
        return chosen_frames, chosen_frames.expnum.unique()

    if calibration == "bias":
        return chosen_frames, chosen_frames.expnum.unique()

    standards_sequence = _get_standards_ring(ring=ring)
    chosen_frames.query("calibfib in @standards_sequence", inplace=True)
    chosen_expnums = chosen_frames.groupby("calibfib").first().reset_index().sort_values("calibfib").expnum.unique()
    chosen_frames.query("expnum in @chosen_expnums", inplace=True)
    return chosen_frames.reset_index(drop=True), chosen_expnums


def get_sequence_iterator(frames, camera):
    """Returns standards sequence dictionary with exposed fiber and block IDs

    Parameters
    ----------
    expnums : list
        List of exposure numbers in the sequence
    camera : str
        Camera name (e.g. "b1")

    Returns
    -------
    dict
        Dictionary with the exposed standard fiber IDs for each exposure in the sequence
    """

    slitmap = SLITMAP.copy()
    slitmap = slitmap[slitmap["spectrographid"] == int(camera[1])]
    spec_select = slitmap["telescope"] == "Spec"
    ids_std = slitmap[spec_select]["orig_ifulabel"]

    exposed_stds, block_idxs = {}, np.arange(LVM_NBLOCKS).tolist()
    for _, frame in frames.iterrows():
        # get block ID for exposed standard fiber
        fiber_par = slitmap[slitmap["orig_ifulabel"] == frame.calibfib]
        block_idx = int(fiber_par["blockid"][0][1:])-1
        if block_idx in block_idxs:
            block_idxs.remove(block_idx)

        exposed_stds[frame.expnum] = (frame.calibfib, [block_idx])

    # handle case of no standard fiber exposed
    if len(exposed_stds) == 0:
        block_idxs = []
        exposed_stds[frames.expnums[0]] = (None, block_idxs)

    # add missing blocks for first exposure
    if len(block_idxs) > 0:
        expnum = list(exposed_stds.keys())[0]
        exposed_stds[expnum] = (exposed_stds[expnum][0], sorted(exposed_stds[expnum][1]+block_idxs))

    # list unexposed standard fibers
    unexposed_stds = [fiber for fiber in ids_std if fiber not in list(zip(*exposed_stds.values()))[0]]

    return exposed_stds, unexposed_stds


def get_fibers_signal(mjd, camera, expnum, imagetyp="flat"):
    img_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, expnum=expnum, camera=camera, imagetype=imagetyp, kind="d")
    img = loadImage(img_path)

    img._data = np.nan_to_num(img._data, posinf=0, neginf=0)
    img._error = np.nan_to_num(img._error, nan=np.inf, neginf=np.inf)

    fiberpos = img.match_reference_column()
    fiberpos = fiberpos.round().astype(int)
    data = img._data[fiberpos]
    error = img._error[fiberpos]

    snr = data / error

    log.info(f"average signal = {np.nanmean(data)}")
    log.info(f"average SNR = {np.nanmean(snr)}")
    log.info(f"standard deviation SNR = {np.nanstd(snr)}")

    return fiberpos, img


def get_exposed_standards(expnums, camera, ref_column=LVM_REFERENCE_COLUMN, ncolumns=100, trust_header=True, axs={}):
    """Determine standard fiber exposed on dome flats for a given calibration epoch

    Parameters
    ----------
    expnums : list[int]
        list of exposure numbers
    camera : str
        camera e.g., b1, r3
    ref_column : int, optional
        reference column used to locate fiber centroids, by default LVM_REFERENCE_COLUMN
    ncolumns : int, optional
        number of columns combined around `ref_column`, by default 100
    trust_header : bool, optional
        whether to trust CALIBFIB header

    Returns
    -------
    pd.DataFrame
        dataframe containing the analysed dome flats
    dict[(int,int), str]
        dictionary mapping MJD, exposure number to exposed standard fiber

    Raises
    ------
    ValueError
        if the value of camera does not match LVM's
    """
    if camera not in CAMERAS:
        raise ValueError(f"Invalid value for `camera`: {camera}. Expected one of {', '.join(CAMERAS)}")

    dframe_paths = [path.expand("lvm_anc", drpver=drpver, tileid=11111, mjd="*", kind="d", imagetype="*", camera=camera, expnum=expnum) for expnum in expnums]
    dframe_paths = [dframe_path[0] for dframe_path in dframe_paths if dframe_path != []]
    if len(dframe_paths) == 0:
        log.error(f"no detrended frames found for {camera = }, {expnums = }, skipping exposed standard detection")
        return {expnum: (None, np.inf) for expnum in expnums}

    NIMAGES = 10
    log.info(f"combining a maximum of {NIMAGES} exposures: {expnums[:NIMAGES]}")
    images = [image_tasks.loadImage(dframe_path) for dframe_path in dframe_paths[:NIMAGES]]
    cimage = image_tasks.combineImages(images, normalize=False, background_subtract=False)
    log.info(f"correcting reference fiber positions @ {ref_column} +/- {ncolumns//2} columns")
    fiber_pos, profile, mhat, bhat = cimage.match_reference_column(ref_column, width=ncolumns, return_all=True)
    log.info(f"stretch factor: {mhat:.3f}, shift: {bhat:.3f}")

    axs = axs or {}
    if "profile" in axs:
        axs["profile"].set_title(f"Exposed standard fibers for expnums = {expnums.min()} - {expnums.max()} | {camera = }", fontsize="x-large")
        axs["profile"].vlines(fiber_pos.round().astype("int"), 0, np.nanmax(profile._data), lw=1, color="tab:red")
        axs["profile"].step(profile._pixels, profile._data, where="mid", lw=1, color="0.2")

    log.info(f"detecting exposed standard fibers for {camera = }:")
    exposed_stds = {}
    axs_exposed = axs.get("exposed", [None] * len(dframe_paths))
    for i, dframe_path in enumerate(dframe_paths):
        if not os.path.isfile(dframe_path):
            continue
        dframe = image_tasks.loadImage(dframe_path)
        camera = dframe._header["CCD"]
        expnum = dframe._header["EXPOSURE"]

        # skip exposed standard identification if header keyword exists
        exposed_std = dframe._header.get("CALIBFIB")
        if exposed_std is not None and exposed_std in dframe._slitmap["orig_ifulabel"] and trust_header:
            log.info(f"    camera exposure {expnum}, skipping and using header value: {exposed_std.__str__():<5s}")
            exposed_stds[expnum] = (exposed_std, np.inf)
            continue

        # NOTE: test the possibility to match the fiber positions for each image
        # log.info(f"correcting reference fiber positions @ {ref_column}")
        # fiber_pos, profile, mhat, bhat = dframe.match_reference_column(ref_column, width=ncolumns, return_all=True)
        # log.info(f"stretch factor: {mhat:.3f}, shift: {bhat:.3f}")
        # _, ax_profile = plt.subplots(figsize=(15,5), layout="tight")
        # ax_profile.vlines(fiber_pos.round().astype("int"), 0, np.nanmax(profile._data), lw=1, color="tab:red")
        # ax_profile.step(profile._pixels, profile._data, where="mid", lw=1, color="0.2")
        # NOTE: test the possibility to measure the fiber positions to have a better estimate of the peak SNR

        try:
            exposed_std, snr, fiber_pos, _, _ = dframe.get_exposed_std(ref_column=ref_column, width=ncolumns, nsigmas=50, fiber_pos=fiber_pos, trust_errors=True, ax=axs_exposed[i])
            exposed_pos = fiber_pos[dframe._filter_slitmap()["orig_ifulabel"].data == exposed_std]
            log.info(f"    camera exposure {expnum}, found exposed standard fiber: {exposed_std.__str__():<5s}")
        except Exception as e:
            log.error(f"while analysing frame at {dframe_path}: {e}")
            exposed_stds[expnum] = (exposed_std, np.inf)
            continue

        exposed_stds[expnum] = (exposed_std, snr[exposed_pos][0] if exposed_std is not None else None)

    return exposed_stds


def create_calibfib_hdrfix(mjd, calibration, ref_column=LVM_REFERENCE_COLUMN, ncolumns=500, trust_header=True, skip_done=True, display_plots=False):
    """Creates header fixes for CALIBFIB when it is missing in dome/twilight flats and arcs frames

    NOTE: The implementation of CALIBFIB became live Nov 22nd, 2023

    Parameters
    ----------
    mjd : int
        MJD of the night to analyse
    calibration : str
        Type of calibration to analyse, either 'trace', 'wave' or 'twilight'
    ref_column : int, optional
        Reference column used approximate fiber centroids, by default LVM_REFERENCE_COLUMN
    ncolumns : int, optional
        Number of columns to combine around reference column, by default 500
    skip_done : bool, optional
        If frames are already detrended this step is skipped, by default True
    display_plots : bool, optional
        Display plots to screen, by default False
    """
    def choose_stds(exposed_stds):
        df = pd.DataFrame.from_dict(exposed_stds)
        fibers = df.map(lambda x: x[0] if hasattr(x, "__getitem__") else np.nan)
        snrs = df.map(lambda x: x[1] if hasattr(x, "__getitem__") else np.nan)
        chosen_stds = {expnum: fibers.loc[expnum, camera] if not pd.isnull(camera) else None for expnum, camera in snrs.idxmax(axis="columns").items()}
        return chosen_stds

    log.info(f"going to create CALIBFIB header fixes for '{calibration}' calibrations on {mjd = }")
    frames = pd.concat([md.get_calibrations_metadata(mjds=mjd, calibration=calibration, camera=camera) for camera in CAMERAS], ignore_index=True)
    if frames.empty:
        log.error(f"no calibration frames found for calibration = '{calibration}' | {mjd = }, skipping CALIBFIB header fix creation")
        return
    if trust_header:
        frames = frames.loc[~frames.calibfib.isin(STD_FIBER_LABELS)]
    if frames.empty:
        log.info(f"all frames for calibration = '{calibration}' | {mjd = }, skipping CALIBFIB header fix creation")
        return

    expnums = frames.expnum.unique()
    calibs = get_calib_paths(mjd=mjd, version=drpver, flavors=CALIBRATION_NEEDS[calibration], from_sandbox=True)
    reduce_2d(mjds=mjd, calibrations=calibs, expnums=expnums, cameras=CAMERAS, add_astro=False, reject_cr=False, sub_straylight=False, skip_done=skip_done)

    exposed_stds = {}
    for camera in CAMERAS:
        ncols = 3
        nrows = np.ceil(len(expnums) / ncols).astype("int")
        fig = plt.figure(figsize=(15,1+4*nrows))
        gs = GridSpec(nrows+1, ncols)
        ax_profile = fig.add_subplot(gs[0, :])

        axs = []
        for i, j in product(range(nrows), range(ncols)):
            if not axs:
                ax = fig.add_subplot(gs[i+1, j])
            else:
                ax = fig.add_subplot(gs[i+1, j], sharex=axs[0], sharey=axs[0])
            ax.label_outer()
            axs.append(ax)

        stds = get_exposed_standards(expnums=expnums, camera=camera, ref_column=ref_column, ncolumns=ncolumns, axs={"profile": ax_profile, "exposed": axs})
        exposed_stds[camera] = stds

        fig.tight_layout()
        fig.align_xlabels(axs)
        fig.align_ylabels(axs)
        save_fig(fig,
                 product_path=path.full("lvm_anc", drpver=drpver, tileid=11111,
                                         mjd=mjd, camera=camera, expnum=0,
                                         kind="d", imagetype=calibration),
                 to_display=display_plots,
                 figure_path="qa",
                 label="exposed_std_fiber")

    expnum_std = choose_stds(exposed_stds=exposed_stds)
    log.info(f"going to write header fixes for CALIBFIB on {len(expnum_std.keys())} exposures")
    for expnum, std in expnum_std.items():
        hdrfix.write_hdrfix_file(mjd=mjd, fileroot=f"sdR-*-*-{expnum:>08d}", keyword="CALIBFIB", value=std)


def create_imagetyp_hdrfix(mjd, calibration):
    IMAGETYPES_MAPPING = {"bias": "bias", "trace": "flat", "wave": "arc", "twilight": "flat", "dome": "flat"}
    imagetyp = IMAGETYPES_MAPPING[calibration]

    frames = md.get_calibrations_metadata(mjds=mjd, calibration=calibration)
    frames.query("imagetyp != @imagetyp", inplace=True)
    expnums = frames.expnum.unique()
    nexpnums = len(expnums)
    if nexpnums == 0:
        log.info(f"no need to apply header fixes for {calibration = }, IMAGETYP = '{imagetyp}', on MJD = {mjd}")
        return
    log.info(f"going to write header fixes for {calibration = }, IMAGETYP = '{imagetyp}', on {nexpnums} exposures")
    for expnum in expnums:
        hdrfix.write_hdrfix_file(mjd=mjd, fileroot=f"sdR-*-*-{expnum:>08d}", keyword="IMAGETYP", value=IMAGETYPES_MAPPING[calibration])


def create_qaqual_bad_hdrfix(mjd, expnums):

    frames = [md.get_calibrations_metadata(mjds=mjd, expnums=expnums, calibration=calibration) for calibration in CALIBRATION_TYPES]
    frames = pd.concat(frames, ignore_index=True)
    expnums = frames.expnum.unique()
    nexpnums = len(expnums)
    if nexpnums == 0:
        log.info(f"no need to apply header fixes QAQUAL = 'BAD' on MJD = {mjd}")
        return
    log.info(f"going to write header fixes QAQUAL = 'BAD' on {nexpnums} exposures")
    for expnum in expnums:
        hdrfix.write_hdrfix_file(mjd=mjd, fileroot=f"sdR-*-*-{expnum:>08d}", keyword="QAQUAL", value="BAD")


def _parse_list(items_str):
    # handle list
    if isinstance(items_str, str) and "," in items_str:
        items = [int(i) for i in items_str.split(",") if i]
    # handle range
    elif isinstance(items_str, str) and "-" in items_str:
        item_i, item_f = items_str.split("-")
        if not item_i or not item_f:
            items = []
        else:
            items = list(range(int(item_i), int(item_f)+1))
    # handle rest
    else:
        items = [int(items_str)]

    return items


def load_calibration_epochs(epochs_path=None, filter_by=None, verbose=True):
    epochs_path = epochs_path or CALIBRATION_EPOCHS_PATH
    if not os.path.exists(epochs_path):
        log.error(f"calibration epochs file not found: {epochs_path}")
        return

    with open(epochs_path) as f:
        epochs = yaml.safe_load(f)["epochs"]

    if verbose:
        log.info(f"found {len(epochs)} calibration epochs:")
        for mjd in epochs:
            log.info(f"  {mjd}: {epochs[mjd]}")

    if filter_by is not None and isinstance(filter_by, (list, tuple)):
        epochs = {mjd: epochs[mjd] for mjd in filter_by if mjd in epochs}
        if len(epochs) == 0:
            log.error(f"epoch(s) {filter_by} not found in calibration epochs file: '{epochs_path}'")
            return epochs
        if verbose:
            log.info(f"after filtering by MJD = {filter_by}, {len(epochs)} remaining epoch(s):")
            for mjd in epochs:
                log.info(f"  {mjd}: {epochs[mjd]}")
    return epochs


def get_calibration_epoch(mjd, flavors=None, trigger=None, comment=None):
    if flavors is None:
        return {flavor: _parse_list(mjd) for flavor in CALIBRATION_TYPES}

    calibs_mjds = {flavor: _parse_list(source_mjd) for flavor, source_mjd in flavors.items()}
    return calibs_mjds


def _load_shift_report(mjd):
    """Reads QC reports with the electronic pixel shifts"""

    with open(os.path.join(os.environ["LVM_SANDBOX"], "shift_monitor", f"shift_{mjd}.txt"), "r") as f:
        lines = f.readlines()[2:]

    shifts_report = {}
    for line in lines:
        cols = line[:-1].split()
        if not cols:
            continue
        _, exp, _, spec = cols[:4]
        exp = int(exp)
        spec = spec[-1]
        shifts = np.array([int(_) for _ in cols[4:]])
        shifts_report[(spec, exp)] = (shifts[::2], shifts[1::2])

    return shifts_report


def _get_reference_expnum(frame, ref_frames):
    """Get reference frame for a given frame

    Given a frame and a set of reference frames, get the reference frame for the
    given frame. This routine will return the reference frame with the closest
    exposure number to the given frame.

    Parameters:
    ----------
    frame : pd.Series
        Frame metadata
    ref_frames : pd.DataFrame
        Reference frames metadata

    Returns:
    -------
    pd.Series
        Reference frame metadata
    """
    if frame.imagetyp == "flat" and frame.ldls|frame.quartz:
        refs = ref_frames.query("imagetyp == 'flat' and (ldls|quartz)")
    elif frame.imagetyp == "flat":
        refs = ref_frames.query("imagetyp == 'flat' and not (ldls|quartz)")
    else:
        refs = ref_frames.query("imagetyp == @frame.imagetyp")

    ref_expnums = refs.expnum.unique()
    if len(ref_expnums) < 2:
        warnings.warn(f"no reference frame found for {frame.imagetyp}, found only {len(ref_expnums)} exposure(s)")
        return None
    idx = np.argmin(np.abs(ref_expnums-frame.expnum))
    if idx > 0:
        idx -= 1
    if idx == 0:
        idx += 1
    return ref_expnums[idx]


def _clean_ancillary(mjd, expnums=None, flavors="all"):
    """Clean ancillary files

    Given a set of MJDs and (optionally) exposure numbers, clean the ancillary
    files for the given flavor of frames. This routine will remove the ancillary
    files for the given flavor of frames in the corresponding calibration
    directory in the `masters_mjd` or by default in the smallest MJD in `mjds`.

    Parameters:
    ----------
    mjd : int
        MJD to clean
    expnums : list
        List of exposure numbers to clean
    flavors : list, tuple, set or str
        type of, defaults to "all"
    """
    # filter by target image types
    all_flavors = {"bias", "dark", "flat", "arc", "cent", "amp", "width", "stray"}
    if not set(flavors).issubset(flavors):
        raise ValueError(f"Invalid flavor: '{flavors}'. Must be one of {all_flavors} or 'all'")

    ancillary_dir = os.path.join(os.getenv("LVM_SPECTRO_REDUX"), drpver, "0011XX", "11111", str(mjd), "ancillary")
    if flavors == "all":
        rmtree(ancillary_dir)
        return

    for flavor in flavors:
        # remove ancillary files
        ancillary_paths = path.expand("lvm_anc", drpver=drpver, mjd=mjd, tileid=11111, kind='*', imagetype=flavor, camera="*", expnum="*")
        [os.remove(ancillary_path) for ancillary_path in ancillary_paths]

    if not os.listdir(ancillary_dir):
        os.rmdir(ancillary_dir)


def _link_pixelmasks():
    """Creates a symbolic link of fiducial pixel flats and masks to current version directory"""
    tileid = 11111
    tilegrp = tileid_grp(tileid)
    pixelmasks_version_path = os.path.join(os.getenv('LVM_SPECTRO_REDUX'), f"{drpver}/{tilegrp}/{tileid}/pixelmasks")
    if os.path.isdir(pixelmasks_version_path):
        log.info(f"link to pixel flats and masks already exists, {pixelmasks_version_path}")
        return

    pixelmasks_path = os.path.join(MASTERS_DIR, "pixelmasks")
    log.info(f"linking pixel flats and masks to {pixelmasks_version_path}")
    os.symlink(src=pixelmasks_path,
                dst=pixelmasks_version_path,
                target_is_directory=True)


def _get_ring_expnums(expnums_ldls, expnums_qrtz, ring_size=12, sort_expnums=False):
    """Split expnums into primary and secondary ring expnums

    Given a set of MJDs and (optionally) exposure numbers, split the expnums
    into primary and secondary ring expnums. This routine will return the
    primary and secondary ring expnums for the given expnums.

    Parameters:
    ----------
    expnums_ldls : list
        List of LDLS expnums
    expnums_qrtz : list
        List of quartz expnums
    ring_size : int
        Size of the primary ring
    sort_expnums : bool
        Sort expnums

    Returns:
    -------
    expnum_params : dict
        Dictionary with the expnums parameters
    """

    # sort expnums
    if sort_expnums:
        expnums_ldls = sorted(expnums_ldls)
        expnums_qrtz = sorted(expnums_qrtz)

    # split expnums into primary and secondary ring expnums
    pri_ldls_expnums = expnums_ldls[:ring_size]
    pri_qrtz_expnums = expnums_qrtz[:ring_size]
    sec_ldls_expnums = expnums_ldls[ring_size:]
    sec_qrtz_expnums = expnums_qrtz[ring_size:]

    # define expnum parameters
    expnum_params = {camera: [] for camera in ["b1", "b2", "b3", "r1", "r2", "r3", "z1", "z2", "z3"]}
    for ring, ring_expnums in enumerate([(pri_ldls_expnums, pri_qrtz_expnums), (sec_ldls_expnums, sec_qrtz_expnums)]):
        for channel, expnums in [("b", ring_expnums[0]), ("r", ring_expnums[0]), ("z", ring_expnums[1])]:
            for fiber, expnum in enumerate(expnums):
                if expnum is None:
                    continue
                # define fiber ID
                # TODO: change this to use CALIBFIB header keyword
                fiber_str = f"P{ring+1}-{fiber+1}"
                # get spectrograph where current fiber is plugged
                fiber_par = SLITMAP[SLITMAP["orig_ifulabel"] == fiber_str]
                block_id = int(fiber_par["blockid"][0][1:])-1
                specid = fiber_par["spectrographid"][0]
                # define camera exposure
                camera = f"{channel}{specid}"

                # define exposure parameters
                expnum_params[camera].append((expnum, [block_id], fiber_str))

    # add missing blocks for first exposure
    for camera in expnum_params:
        if len(expnum_params[camera]) == 0:
            continue
        expnums, block_ids, fiber_strs = zip(*expnum_params[camera])
        block_ids = list(zip(*block_ids))[0]
        missing_block_ids = list(set(range(18)) - set(block_ids))
        filled_block_ids = list(block_ids)[0:1] + missing_block_ids
        expnum_params[camera][0] = (expnums[0], sorted(filled_block_ids), fiber_strs[0])

    return expnum_params


def _get_crosstalk(cent, fwhm, ifiber, jcolumn, ypixels=None, nfibers=1):
    """Calculates the crosstalk between a reference fiber and its neighboring fibers

    This assumes the fibers have a Gaussian profile and that they can be
    treated as vectors, where the cross-talk is calculated as the scalar
    projection of fibers `ifiber-1` and `ifiber+1`.

    Parameters
    ----------
    cent : TraceMask
        TraceMask object containing the centroid positions of the fibers.
    fwhm : TraceMask
        TraceMask object containing the FWHM values of the fibers.
    ifiber : int
        Index of the reference fiber for which to compute crosstalk.
    jcolumn : int
        Column index (typically along the dispersion axis) at which to evaluate the crosstalk.
    ypixels : array-like, optional
        Array of pixel positions along the spatial axis. If None, defaults to np.arange(0, 4080).
    nfibers : int, optional
        Number of neighboring fibers on each side to include in the crosstalk calculation. Default is 1.

    Returns
    -------
    crosstalk : np.ndarray
        Array of crosstalk values (in percent) from each neighboring fiber to the reference fiber.

    Notes
    -----
    The crosstalk is computed as the projection of the neighboring fiber profiles onto the normalized
    reference fiber profile at the specified column. The result is expressed as a percentage.
    """
    if ypixels is None:
        ypixels = np.arange(0, 4080)

    ref_cent = cent._data[[ifiber], jcolumn]
    ref_width = fwhm._data[[ifiber], jcolumn] / 2.354
    ref_amp = np.asarray([1.0])
    ref_fiber = gaussians([ref_amp, ref_cent, ref_width], ypixels)

    nei_ifibers = [ifiber - i for i in range(nfibers, 0, -1)] + [ifiber + i for i in range(1, nfibers + 1)]
    nei_cents = cent._data[nei_ifibers, jcolumn]
    nei_widths = fwhm._data[nei_ifibers, jcolumn] / 2.354
    nei_amps = np.ones_like(nei_ifibers)

    nei_fibers = gaussians([nei_amps, nei_cents, nei_widths], ypixels)

    ref_unit = ref_fiber / np.sqrt(np.dot(ref_fiber, ref_fiber))
    crosstalk = np.dot(nei_fibers, ref_unit) * 100

    return crosstalk


def _log_dry_run(frames, calibs=None, settings=None, caller=None, show_calibrations=False):
    if frames.empty:
        log.error("empty list of frames")
        return
    lamps = list(map(lambda s: s.lower(), CON_LAMPS + ARC_LAMPS))
    df = frames.copy()
    df["lamps"] = df.filter(items=lamps).apply(lambda r: ','.join(r.index[r.values].tolist()) or None, axis="columns")
    df.sort_values(["lamps", "expnum"], inplace=True)
    df.sort_values("calibfib", key=lambda s: s.str.split("-").str[-1].astype(int, errors="ignore"), inplace=True)
    df = df.filter(["mjd", "tileid", "expnum", "imagetyp", "qaqual", "calibfib", "lamps"]).drop_duplicates()
    records = df.to_string(index=None).split("\n")
    if caller is not None:
        log.info(f"dry run of '{caller}' with {len(df)} exposures:")
    for record in records:
        log.info(f"   {record}")
    if show_calibrations and calibs is not None:
        log.info("with calibrations:")
        for r in pformat(calibs).split("\n"):
            log.info(r)


def _create_wavelengths_60177(use_longterm_cals=True, skip_done=True, dry_run=False):
    """Reduce arc sequence for MJD = 60177"""
    pixwav = {"z1": np.asarray([
    [88.57, 7488.8712, 1],
    [111.79, 7503.8690, 1],
    [128.59, 7514.6520, 1],
    [161.43, 7535.7741, 0],
    [237.58, 7584.6800, 1],
    [454.78, 7724.6233, 0],
    [840.33, 7967.3400, 1],
    [902.66, 8006.1570, 0],
    [916.05, 8014.7860, 0],
    [1109.70, 8136.4054, 1],
    [1222.22, 8206.3400, 0],
    [1262.80, 8231.6350, 0],
    [1372.72, 8300.3258, 0],
    [1478.12, 8365.7466, 1],
    [1497.42, 8377.6080, 0],
    [1573.44, 8424.6400, 1],
    [1688.33, 8495.3598, 0],
    [1768.50, 8544.6958, 1],
    [1844.78, 8591.2584, 0],
    [1915.81, 8634.6470, 0],
    [2029.76, 8704.1116, 0],
    [2087.89, 8739.3900, 0],
    [2220.99, 8819.4110, 0],
    [2369.50, 8908.7300, 1],
    [2387.51, 8919.5006, 1],
    [2406.58, 8930.8300, 1],
    [2600.14, 9045.4500, 0],
    [2775.63, 9148.6716, 1],
    [2866.74, 9201.7591, 0],
    [2994.24, 9275.5196, 1],
    [3164.63, 9373.3078, 0],
    [3256.41, 9425.3788, 1],
    [3364.76, 9486.6818, 0],
    [3473.13, 9547.4049, 1],
    [3671.55, 9657.7860, 1],
    [3721.46, 9685.3200, 1],
    [3781.02, 9718.1600, 1],
    [3902.11, 9784.5030, 1],
    [3930.05, 9799.7000, 0]]),
    "z2": np.asarray([
    [91.50, 7488.8712, 1],
    [114.69, 7503.8690, 1],
    [131.48, 7514.6520, 1],
    [164.29, 7535.7741, 0],
    [240.36, 7584.6800, 1],
    [457.35, 7724.6233, 0],
    [842.51, 7967.3400, 1],
    [904.78, 8006.1570, 0],
    [918.15, 8014.7860, 0],
    [1111.61, 8136.4054, 1],
    [1224.01, 8206.3400, 0],
    [1264.56, 8231.6350, 0],
    [1374.37, 8300.3258, 0],
    [1479.65, 8365.7466, 1],
    [1498.94, 8377.6080, 0],
    [1574.89, 8424.6400, 1],
    [1689.66, 8495.3598, 0],
    [1769.74, 8544.6958, 1],
    [1845.95, 8591.2584, 0],
    [1916.91, 8634.6470, 0],
    [2030.75, 8704.1116, 0],
    [2088.82, 8739.3900, 0],
    [2221.79, 8819.4110, 0],
    [2370.14, 8908.7300, 1],
    [2388.14, 8919.5006, 1],
    [2407.19, 8930.8300, 1],
    [2600.55, 9045.4500, 0],
    [2775.87, 9148.6716, 1],
    [2866.89, 9201.7591, 0],
    [2994.25, 9275.5196, 1],
    [3164.48, 9373.3078, 0],
    [3256.16, 9425.3788, 1],
    [3364.41, 9486.6818, 0],
    [3472.66, 9547.4049, 1],
    [3670.89, 9657.7860, 1],
    [3720.75, 9685.3200, 1],
    [3780.25, 9718.1600, 1],
    [3901.22, 9784.5030, 1],
    [3929.13, 9799.7000, 0]]),
    "z3": np.asarray([
    [68.74, 7488.8712, 1],
    [92.02, 7503.8690, 1],
    [108.87, 7514.6520, 1],
    [141.79, 7535.7741, 0],
    [218.13, 7584.6800, 1],
    [435.89, 7724.6233, 0],
    [822.42, 7967.3400, 1],
    [884.91, 8006.1570, 0],
    [898.33, 8014.7860, 0],
    [1092.48, 8136.4054, 1],
    [1205.28, 8206.3400, 0],
    [1245.97, 8231.6350, 0],
    [1356.17, 8300.3258, 0],
    [1461.83, 8365.7466, 1],
    [1481.18, 8377.6080, 0],
    [1557.40, 8424.6400, 1],
    [1672.58, 8495.3598, 0],
    [1752.95, 8544.6958, 1],
    [1829.43, 8591.2584, 0],
    [1900.64, 8634.6470, 0],
    [2014.88, 8704.1116, 0],
    [2073.16, 8739.3900, 0],
    [2206.60, 8819.4110, 0],
    [2355.48, 8908.7300, 1],
    [2373.54, 8919.5006, 1],
    [2392.66, 8930.8300, 1],
    [2586.71, 9045.4500, 0],
    [2762.65, 9148.6716, 1],
    [2853.99, 9201.7591, 0],
    [2981.81, 9275.5196, 1],
    [3152.64, 9373.3078, 0],
    [3244.65, 9425.3788, 1],
    [3353.28, 9486.6818, 0],
    [3461.92, 9547.4049, 1],
    [3660.85, 9657.7860, 1],
    [3710.89, 9685.3200, 1],
    [3770.60, 9718.1600, 1],
    [3892.00, 9784.5030, 1],
    [3920.01, 9799.7000, 0]])}

    mjd = 60177
    expnums = range(3453, 3466+1)

    frames = md.get_calibrations_metadata(mjds=mjd, calibration="wave", expnums=expnums)

    # define master paths for target frames
    calibs = get_calib_paths(mjd, version=drpver, longterm_cals=use_longterm_cals, flavors=CALIBRATION_NEEDS["wave"])

    if dry_run:
        _log_dry_run(frames, calibs=calibs, settings=None, caller=_create_wavelengths_60177.__name__)
        return

    reduce_2d(mjd, calibrations=calibs, expnums=expnums, assume_imagetyp="arc", reject_cr=False,
              add_astro=False, sub_straylight=False, skip_done=skip_done)

    lamps = [lamp.lower() for lamp in ARC_LAMPS]
    xarc_paths = {"b1": [], "b2": [], "b3": [], "r1": [], "r2": [], "r3": [], "z1": [], "z2": [], "z3": []}
    for lamp in lamps:
        arc_analogs = frames.loc[frames[lamp]].groupby(["camera",])
        for camera in arc_analogs.groups:
            arcs = arc_analogs.get_group((camera,))
            expnum_str = f"{arcs.expnum.min():>08}_{arcs.expnum.max():>08}"

            # define master frame path
            carc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype=f"arc_{lamp}", camera=camera, expnum=expnum_str)
            xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype=f"arc_{lamp}", camera=camera, expnum=expnum_str)
            os.makedirs(os.path.dirname(carc_path), exist_ok=True)
            darc_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="arc", **arc) for arc in arcs.to_dict("records")]
            xarc_paths[camera].append(xarc_path)

            # create master arc (2D image)
            if skip_done and os.path.exists(carc_path):
                log.info(f"skipping {carc_path}, file already exists")
            else:
                image_tasks.create_master_frame(in_images=darc_paths, out_image=carc_path)

            # extract combined (master) arc
            if skip_done and os.path.exists(xarc_path):
                log.info(f"skipping {xarc_path}, file already exists")
            else:
                image_tasks.extract_spectra(in_image=carc_path, out_rss=xarc_path,
                                            in_trace=calibs["centroids"][camera],
                                            in_sigma=calibs["sigmas"][camera],
                                            in_model=calibs["model"][camera])

    expnum_str = f"{frames.expnum.min():>08}_{frames.expnum.max():>08}"
    for camera in np.sort(frames.camera.unique()):
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=camera, expnum=expnum_str)

        # coadd arcs
        if skip_done and os.path.isfile(xarc_path):
            log.info(f"skipping {xarc_path}, file already exists")
        else:
            rss_tasks.combine_rsss(in_rsss=xarc_paths[camera], out_rss=xarc_path, method="sum")

        # fit wavelength solution
        mwave_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mwave")
        mlsf_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mlsf")
        pixels = pixwav[camera][:, 0] if camera in pixwav else []
        waves = pixwav[camera][:, 1] if camera in pixwav else []
        use_lines = pixwav[camera][:, 2].astype(bool) if camera in pixwav else []
        ref_lines, _, cent_wave, _, rss, wave_trace, fwhm_trace = rss_tasks.determine_wavelength_solution(in_arcs=xarc_paths[camera], out_wave=mwave_path, out_lsf=mlsf_path,
                                                                                                          pixel=pixels, ref_lines=waves, use_line=use_lines,
                                                                                                          flux_range=[800, np.inf], cent_range=[-1.5, 1.5], fwhm_range=[2.0, 4.5])

        lvmarc = lvmArc(data=rss._data, error=rss._error, mask=rss._mask, header=rss._header,
                        ref_wave=ref_lines, cent_line=cent_wave,
                        wave_trace=wave_trace, lsf_trace=fwhm_trace)
        lvmarc.writeFitsData(path.full("lvm_frame", mjd=mjd, tileid=11111, drpver=drpver, expnum=expnum_str, kind=f'Arc-{camera}'))

    for channel in "brz":
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=channel, expnum=expnum_str)
        harc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="h", imagetype="arc", camera=channel, expnum=expnum_str)
        mwave_paths = [path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}{spec+1}", kind="mwave") for spec in range(3)]
        mlsf_paths = [path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}{spec+1}", kind="mlsf") for spec in range(3)]

        # stack spectragraphs
        xarc_paths = sorted(path.expand("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=f"{channel}?", expnum=expnum_str))
        rss_tasks.stack_spectrographs(in_rsss=xarc_paths, out_rss=xarc_path)

        # apply wavelength solution to arcs and rectify
        rss_tasks.create_pixel_table(in_rss=xarc_path, out_rss=harc_path, in_waves=mwave_paths, in_lsfs=mlsf_paths)
        rss_tasks.resample_wavelength(in_rss=harc_path, out_rss=harc_path, method="linear", wave_range=SPEC_CHANNELS[channel], wave_disp=0.5)


def _copy_fiberflats_from(mjd, mjd_dest=60177, use_longterm_cals=True):
    """Copies twilight fiberflats from given MJD to MJD destination

    Parameters
    ----------
    mjd : int
        MJD of calibration epoch from which the twilight fiberflats will be copied
    mjd_dest : int
        MJD where copied twilight fiberflats will be stored
    use_longterm_cals : bool, optional
        Whether to use long-term calibration frames or not, defaults to True
    """

    # get source fiberflats
    fiberflat_paths = get_calib_paths(mjd, version=drpver, longterm_cals=use_longterm_cals)
    fiberflat_paths = group_calib_paths(fiberflat_paths["fiberflat_twilight"])

     # define master paths for target frames
    calibs = get_calib_paths(mjd_dest, version=drpver, longterm_cals=use_longterm_cals)
    mwave_paths = group_calib_paths(calibs["wave"])
    mlsf_paths = group_calib_paths(calibs["lsf"])

    log.info(f"going to copy twilight fiberflats from {mjd = } to {mjd_dest = }")
    for channel in "brz":
        log.info(f"preparing wavelength for new fiberflats: {mwave_paths[channel]}, {mlsf_paths[channel]}")
        mwaves = [TraceMask.from_file(mwave_path) for mwave_path in mwave_paths[channel]]
        mwave = TraceMask.from_spectrographs(*mwaves)
        mlsfs = [TraceMask.from_file(mlsf_path) for mlsf_path in mlsf_paths[channel]]
        mlsf = TraceMask.from_spectrographs(*mlsfs)

        fiberflat_path = fiberflat_paths[channel][0]
        log.info(f"loading reference fiberflat from {fiberflat_path}")
        fiberflat = RSS.from_file(fiberflat_path)

        # interpolate fiberflats to mjd_ wavelengths
        log.info("resampling fiberflat to new wavelengths")
        new_fiberflat = copy(fiberflat)
        new_fiberflat._header["MJD"] = mjd_dest
        new_fiberflat._header["SMJD"] = mjd_dest
        for ifiber in range(fiberflat._fibers):
            old_wave = fiberflat._wave[ifiber]
            new_wave = mwave._data[ifiber]
            old_flat = fiberflat._data[ifiber]

            new_fiberflat._data[ifiber] = interpolate.interp1d(old_wave, old_flat, bounds_error=False, fill_value="extrapolate")(new_wave)
            if new_fiberflat._error is not None:
                new_fiberflat._error[ifiber] = interpolate.interp1d(old_wave, fiberflat._error[ifiber], bounds_error=False, fill_value="extrapolate")(new_wave)
            if new_fiberflat._mask is not None:
                new_fiberflat._mask[ifiber] = interpolate.interp1d(old_wave, fiberflat._mask[ifiber].astype(int), bounds_error=False, kind="nearest", fill_value="extrapolate")(new_wave)
                new_fiberflat._mask[ifiber] = new_fiberflat._mask[ifiber].astype(bool)

        # update wavelength traces
        new_fiberflat.set_wave_trace(mwave)
        new_fiberflat.set_lsf_trace(mlsf)
        new_fiberflat.set_wave_array(mwave._data)
        new_fiberflat.set_lsf_array(mlsf._data)

        # store new fiberflat
        new_fiberflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_dest, camera=channel, kind="mfiberflat_twilight")
        log.info(f"writing new fiberflat to {new_fiberflat_path}")
        new_fiberflat.writeFitsData(new_fiberflat_path)


def copy_longterm_calibrations(mjd, flavors=None, dry_run=False):
    """Copies long-term calibrations from versioned path to sandbox

    Parameters
    ----------
    mjd : int
        MJD for the source calibrations to copy from
    flavors : str, optional
        Types of calibration (e.g., wave, bias), by default None (all calibrations)
    dry_run : bool, optional
        log information about source and
    """
    # handle possible acceptable flavors
    if isinstance(flavors, (list, tuple, set, np.ndarray)):
        flavors = set(flavors)
    elif isinstance(flavors, str) and flavors in flavors:
        flavors = {flavors}
    elif flavors is None:
        flavors = CALIBRATION_PRODUCTS.difference({"pixmask", "pixflat", "trace_guess", "amp", "fiberflat_dome"})
    else:
        raise ValueError(f"kind must be one of {flavors}")

    # filter out non-needed calibrations
    flavors = set(flavors).difference({"pixmask", "pixflat", "trace_guess", "amp", "fiberflat_dome"})

    log.info(f"going to copy calibrations: {flavors}")
    for flavor in flavors:
        src_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind=f"m{flavor}", camera="*"))
        if not src_paths:
            log.error(f"no paths found for {flavor = }: {src_paths}")
        for src_path in src_paths:
            camera = os.path.basename(src_path).split(".")[0].split("-")[-1]
            dst_path = path.full("lvm_calib", mjd=mjd, kind=flavor, camera=camera)

            dst_exists = os.path.isfile(dst_path)
            if dry_run:
                src_mtime = datetime.fromtimestamp(os.path.getmtime(src_path))
                dst_mtime = datetime.fromtimestamp(os.path.getmtime(dst_path)) if dst_exists else None
                log.info(f"source/destination for {flavor = }, {camera = }:")
                log.info(f"   {src_mtime.strftime('%a %d %b %Y, %I:%M:%S%p')} {src_path}")
                log.info(f"   {dst_mtime.strftime('%a %d %b %Y, %I:%M:%S%p') if dst_exists else None} {dst_path}")
                if src_mtime > dst_mtime:
                    log.info("   > source is newer than destination")
                elif src_mtime <= dst_mtime:
                    log.warning("   < source is older than destination")
                continue
            try:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                copy2(src_path, dst_path)
                log.info(f"copied {src_path} into {dst_path}")
            except PermissionError as e:
                log.error(f"error while copying {src_path}: {e}")


def messup_frame(mjd, expnum, spec="1", shifts=[1500, 2000, 3500], shift_size=-2, undo_messup=False):
    """ Mess up a frame by shifting the data by a given amount along given rows

    Parameters
    ----------
    mjd : int
        the MJD of the frame
    expnum : int
        the exposure number of the frame
    spec : str
        the spectrograph number
    shifts : list
        the rows to shift
    shift_size : int
        the amount to shift the data by
    undo_messup : bool
        whether to undo the mess up of the frame

    Returns
    -------
    list
        the messed up frames
    """
    specid = f"sp{spec}"
    frames = md.get_frames_metadata(mjd)
    frames.query("expnum == @expnum and spec == @specid", inplace=True)
    log.info(f"messing up frames for spectrograph = {specid}")

    rframe_paths = sorted(path.expand("lvm_raw", hemi="s", camspec=f"?{spec}", mjd=mjd, expnum=expnum))
    rframe_ori_paths = [rframe_path.replace(".fits.gz", "_good.fits.gz") for rframe_path in rframe_paths]
    original_exists = all([os.path.exists(rframe_ori_path) for rframe_ori_path in rframe_ori_paths])
    if undo_messup and original_exists:
        log.info(f"undoing mess up of frames {','.join(rframe_paths)}")
        [copy2(rframe_ori_path, rframe_path) for rframe_ori_path, rframe_path in zip(rframe_ori_paths, rframe_paths)]
        [os.remove(rframe_ori_path) for rframe_ori_path in rframe_ori_paths]
        return
    elif undo_messup:
        log.info(f"cannot undo mess up of frames {','.join(rframe_paths)} because original frames do not exist")
        return

    log.info(f"messing up frames {','.join(rframe_paths)} with {shifts = } and {shift_size = } pixels")
    if not original_exists:
        [copy2(rframe_path, rframe_ori_path) for rframe_path, rframe_ori_path in zip(rframe_paths, rframe_ori_paths)]

    messed_up_frames = []
    for rframe_path, rframe_ori_path in zip(rframe_paths, rframe_ori_paths):
        rframe = loadImage(rframe_ori_path)

        messup_frame = copy(rframe)
        for shift in shifts:
            messup_frame._data[shift:] = np.roll(messup_frame._data[shift:], shift_size, axis=1)

        log.info(f"saving messed up frame to {rframe_path}")
        messup_frame.writeFitsData(rframe_path)

        messed_up_frames.append(messup_frame)

    return messed_up_frames


def fix_raw_pixel_shifts(mjd, expnums=None, ref_expnums=None, use_longterm_cals=True, specs="123", imagetyps=None,
                         y_widths=5, wave_list=None, wave_widths=0.6*5, max_shift=10, flat_spikes=11,
                         threshold_spikes=np.inf, shift_rows=None, interactive=False, skip_done=False,
                         display_plots=False):
    """Attempts to fix pixel shifts in a list of raw frames

    Given an MJD and (optionally) exposure numbers, fix the pixel shifts in a
    list of 2D frames. This routine will store the fixed frames in the
    corresponding calibration directory in the `use_longterm_cals` or by default `mjd`.

    Parameters:
    ----------
    mjd : float
        MJD to reduce
    expnums : list
        List of exposure numbers to look for pixel shifts
    ref_expnums : list
        List of reference exposure numbers to use as reference for good frames, by default None
    use_longterm_cals : bool
        Whether to use long-term calibration frames or not, defaults to True
    specs : str
        Spectrograph channels
    imagetyps : list
        List of image types to analyse, by default None (any image type)
    y_widths : int
        Width of the fibers along y-axis, by default 5
    wave_list : list
        List of lines to use for the wavelength calibration, by default None
    wave_widths : float
        Width of the wavelength axis for the lines, by default 0.6*5
    max_shift : int
        Maximum shift in pixels, by default 10
    flat_spikes : int
        Number of flat spikes, by default 11
    threshold_spikes : float
        Threshold for spikes, by default np.inf
    shift_rows : dict
        Rows to shift, by default None
    interactive : bool
        Interactive mode when report and measured shifts are different, by default False
    skip_done : bool
        Skip pipeline steps that have already been done
    display_plots : bool
        Display plots, by default False
    """

    if shift_rows is None:
        shift_rows = {}
    elif not isinstance(shift_rows, dict):
        raise ValueError("shift_rows must be a dictionary with keys (spec, expnum) and values a list of rows to shift")

    # get target frames & reference frames metadata
    frames = md.get_frames_metadata(mjd)
    if imagetyps is not None:
        frames.query("imagetyp in @imagetyps", inplace=True)
    if expnums is not None:
        frames.query("expnum in @expnums", inplace=True)
    ref_frames = md.get_frames_metadata(mjd)
    if imagetyps is not None:
        ref_frames.query("imagetyp in @imagetyps", inplace=True)
    if ref_expnums is not None:
        ref_frames.query("expnum in @ref_expnums", inplace=True)

    if use_longterm_cals:
        masters_mjd = get_master_mjd(mjd)
        masters_path = os.path.join(MASTERS_DIR, str(masters_mjd))

    ref_imagetyps = set(ref_frames.imagetyp)
    imagetyps = set(frames.imagetyp)
    if not imagetyps.issubset(ref_imagetyps):
        raise ValueError(f"the following image types are not present in the reference frames: {imagetyps - ref_imagetyps}")

    shifts_path = os.path.join(os.getenv('LVM_SANDBOX'), 'shift_monitor', f'shift_{mjd}.txt')
    shifts_report = {}
    if os.path.isfile(shifts_path):
        shifts_report = _load_shift_report(mjd)

    expnums_grp = frames.groupby("expnum")
    for spec in specs:
        for expnum in expnums_grp.groups:
            frame = expnums_grp.get_group(expnum).iloc[0]

            # find suitable reference frame for current frame
            ref_expnum = _get_reference_expnum(frame, ref_frames)
            if ref_expnum is None:
                warnings.warn(f"missing reference frame for {frame.expnum = } of {frame.imagetyp = }")
                continue

            rframe_paths = sorted(path.expand("lvm_raw", hemi="s", camspec=f"?{spec}", mjd=mjd, expnum=expnum))
            rframe_paths = [rframe_path for rframe_path in rframe_paths if ".gz" in rframe_path]

            # use fixed reference if exist, else use original raw frame
            cframe_paths = sorted([path.full("lvm_anc", drpver=drpver, tileid=frame.tileid, mjd=mjd, kind="e", imagetype=frame.imagetyp, expnum=ref_expnum, camera=f"{channel}{spec}") for channel in "brz"])
            if not all([os.path.exists(cframe_path) for cframe_path in cframe_paths]):
                cframe_paths = sorted(path.expand("lvm_raw", hemi="s", camspec=f"?{spec}", mjd=mjd, expnum=ref_expnum))
                cframe_paths = [cframe_path for cframe_path in cframe_paths if ".gz" in cframe_path]

            eframe_paths = [path.full("lvm_anc", drpver=drpver, tileid=frame.tileid, mjd=mjd, kind="e", imagetype=frame.imagetyp, expnum=expnum, camera=f"{channel}{spec}") for channel in "brz"]
            mask_2d_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, imagetype="mask2d",
                                     expnum=0, camera=f"sp{spec}", kind="")

            if len(rframe_paths) < 3:
                log.warning(f"skipping {rframe_paths = }, less than 3 files found")
                continue
            if len(cframe_paths) < 3:
                log.warning(f"skipping {cframe_paths = }, less than 3 files found")
                continue

            if use_longterm_cals:
                mwave_paths = sorted(glob(os.path.join(masters_path, f"lvm-mwave-?{spec}.fits")))
                mtrace_paths = sorted(glob(os.path.join(masters_path, f"lvm-mtrace-?{spec}.fits")))
            else:
                mwave_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mwave", camera=f"?{spec}"))
                mtrace_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mtrace", camera=f"?{spec}"))

            if skip_done and os.path.exists(mask_2d_path):
                log.info(f"skipping {mask_2d_path}, file already exists")
            else:
                image_tasks.select_lines_2d(in_images=cframe_paths, out_mask=mask_2d_path, lines_list=wave_list,
                                            in_cent_traces=mtrace_paths, in_waves=mwave_paths,
                                            y_widths=y_widths, wave_widths=wave_widths,
                                            display_plots=display_plots)

            image_tasks.fix_pixel_shifts(in_images=rframe_paths, out_images=eframe_paths,
                                         ref_images=cframe_paths, in_mask=mask_2d_path, report=shifts_report.get((spec, expnum), None),
                                         flat_spikes=flat_spikes, threshold_spikes=threshold_spikes,
                                         max_shift=max_shift, shift_rows=shift_rows.get((spec, expnum), None),
                                         interactive=interactive, display_plots=display_plots)


def validate_calibration_epochs(mjd=None, calibrations=CALIBRATION_TYPES, epochs_path=CALIBRATION_EPOCHS_PATH, ring="primary"):
    def _report_standards(sequence, nstandards, label=None):
        label = f" for {label}" if label is not None else ""
        stds = sorted(sequence.calibfib.unique().tolist(), key=lambda s: int(s.split("-")[-1]))
        nstds = len(stds)
        if nstds != nstandards:
            log.error(f"{nstds} exposed standards{label}: {', '.join(stds)}")
        elif nstds == nstandards:
            log.info(f"{nstds} exposed standards{label}: {', '.join(stds)}")


    epochs = load_calibration_epochs(epochs_path=epochs_path, verbose=False)
    if mjd is not None and mjd not in epochs:
        log.error(f"no calibrations epoch found for {mjd} in file {epochs_path}")
        return

    if not isinstance(calibrations, (tuple, list, set)):
        calibrations = [calibrations]

    epoch_mjds = [mjd] if mjd is not None else list(epochs.keys())

    nstandards = 12 if ring in {"primary", "secondary"} else 24

    for mjd in epoch_mjds:
        log.info(f"validating {calibrations = } for epoch = {mjd}")
        epoch = get_calibration_epoch(mjd, **epochs[mjd])
        for calibration in calibrations:
            source_mjds = epoch.get(calibration, [])
            log.info(f"MJDs for {calibration = }: {', '.join(map(str, source_mjds))}")

            frames = md.get_calibrations_metadata(mjds=source_mjds, calibration=calibration)
            if calibration == "trace":
                sequence_ldls, _ = choose_sequence(frames.loc[frames.ldls], calibration=calibration, ring=ring)
                sequence_qrtz, _ = choose_sequence(frames.loc[frames.quartz], calibration=calibration, ring=ring)
                sequence = pd.concat((sequence_ldls, sequence_qrtz), ignore_index=True)
            else:
                sequence, _ = choose_sequence(frames, calibration=calibration, ring=ring)
            _log_dry_run(sequence)

            log.info(f"unique MJDs = {', '.join(sequence.mjd.unique().astype(str))}")
            if calibration == "trace":
                _report_standards(sequence_ldls, nstandards, label="ldls")
                _report_standards(sequence_qrtz, nstandards, label="quartz")
            elif calibration in {"wave", "dome", "twilight"}:
                _report_standards(sequence, nstandards)


def create_bias(mjd, epochs=None, use_longterm_cals=True, skip_done=True, dry_run=False):
    """Reduce a sequence of bias frames to produce master frames for each camera

    Given a set of MJDs and (optionally) exposure numbers, reduce the
    bias frames.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    use_longterm_cals : bool
        Whether to use long-term calibration frames or not, defaults to True
    flavor : str
        The type of frame to reduce
    skip_done : bool
        Skip pipeline steps that have already been done
    dry_run : bool, optional
        Logs useful information abaut the current setup without actually reducing, by default False
    """
    epoch = get_calibration_epoch(mjd=mjd, **(epochs or {}).get(mjd, {}))
    mjds = epoch["bias"]

    frames = md.get_calibrations_metadata(mjds=mjds, calibration="bias")
    if frames.empty:
        log.error("no bias frames found, skipping production of bias frames")
        return

    # define master paths for target frames
    calibs = get_calib_paths(mjd=mjd, version=drpver, longterm_cals=use_longterm_cals, flavors=CALIBRATION_NEEDS["bias"])

    if dry_run:
        _log_dry_run(frames, calibs=calibs, settings=None, caller=create_bias.__name__)
        return

    # preprocess and detrend frames
    reduce_2d(mjds=mjds, calibrations=calibs, expnums=frames.expnum.unique(), assume_imagetyp="bias", reject_cr=False, add_astro=False, sub_straylight=False, skip_done=skip_done)

    # reduce each image type
    frames_analog = frames.groupby(["camera"])

    for camera in frames_analog.groups:
        analogs = frames_analog.get_group((camera,))

        # combine into master frame
        kwargs = get_config_options('reduction_steps.create_master_frame', "bias")
        log.info(f'custom configuration parameters for create_master_frame: {repr(kwargs)}')
        mframe_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mbias", camera=camera)

        os.makedirs(os.path.dirname(mframe_path), exist_ok=True)
        pframe_paths = [path.full("lvm_anc", drpver=drpver, kind="p", imagetype="bias", **frame) for frame in analogs.to_dict("records")]
        image_tasks.create_master_frame(in_images=pframe_paths, out_image=mframe_path, batch_size=200, master_mjd=mjd, **kwargs)


def create_nightly_traces(mjd, use_longterm_cals=False, expnums_ldls=None, expnums_qrtz=None,
                          counts_thresholds=COUNTS_THRESHOLDS, cent_guess_ncolumns=140,
                          trace_full_ncolumns=40,
                          fit_poly=True, poly_deg_amp=5, poly_deg_cent=4, poly_deg_width=5,
                          skip_done=True, dry_run=False):
    """
    Create nightly traces from dome flats.

    Given an MJD and (optionally) exposure numbers, create fiber traces from the nightly dome flats.
    This routine stores the nightly master traces in the corresponding calibration directory for the given MJD.
    If the required dome flats do not exist, they will be created first; otherwise, they will be read from disk.

    Parameters
    ----------
    mjd : int
        MJD to reduce.
    use_longterm_cals : bool, optional
        Whether to use long-term calibration frames, by default False.
    expnums_ldls : list, optional
        List of exposure numbers for LDLS dome flats.
    expnums_qrtz : list, optional
        List of exposure numbers for quartz dome flats.
    counts_thresholds : dict, optional
        Dictionary with count thresholds for each lamp type, by default COUNTS_THRESHOLDS.
    cent_guess_ncolumns : int, optional
        Number of columns to use for centroid tracing, by default 140.
    trace_full_ncolumns : int, optional
        Number of columns to use for full fiber tracing, by default 40.
    fit_poly : bool, optional
        Fit polynomials to traces, by default True.
    poly_deg_amp : int, optional
        Degree of the polynomial to fit to the amplitude, by default 5.
    poly_deg_cent : int, optional
        Degree of the polynomial to fit to the centroids, by default 4.
    poly_deg_width : int, optional
        Degree of the polynomial to fit to the widths, by default 5.
    skip_done : bool, optional
        Skip pipeline steps that have already been done, by default True.
    dry_run : bool, optional
        If True, only logs the steps that would be performed, by default False.
    """
    if expnums_ldls is not None and expnums_qrtz is not None:
        expnums = np.concatenate([expnums_ldls, expnums_qrtz])
    else:
        expnums = None

    frames = md.get_calibrations_metadata(mjd, calibration="trace", expnums=expnums)
    if frames.empty:
        log.error("no dome flat frames found, skipping production of fiber traces")
        return

    if expnums_ldls is None or expnums_qrtz is None:
        frames, expnums = choose_sequence(frames, flavor="trace", kind="nightly")
        expnums_ldls = np.sort(frames.query("ldls").expnum.unique())
        expnums_qrtz = np.sort(frames.query("quartz").expnum.unique())

    # define master paths for target frames
    calibs = get_calib_paths(mjd, version=drpver, longterm_cals=use_longterm_cals, flavors=CALIBRATION_NEEDS["trace"])

    if dry_run:
        _log_dry_run(frames, calibs=calibs, settings=None, caller=create_nightly_traces.__name__)
        return

    # run 2D reduction on flats: preprocessing, detrending
    reduce_2d(mjd, calibrations=calibs, expnums=expnums, reject_cr=False,
              add_astro=False, sub_straylight=False, skip_done=skip_done)

    for channel, lamp in MASTER_CON_LAMPS.items():
        counts_threshold = counts_thresholds[lamp]

        # select dome flats accoding to current channel-lamp combination
        flats_analogs = frames.loc[(frames[lamp])&(frames["camera"].str.startswith(channel))].groupby(["camera",])
        for camera in flats_analogs.groups:
            flats = flats_analogs.get_group((camera,))

            # combine dome flats
            if flats.expnum.min() != flats.expnum.max():
                expnum_str = f"{flats.expnum.min():>08}_{flats.expnum.max():>08}"
            else:
                expnum_str = flats.expnum.min()
            dflat_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="flat", **flat) for flat in flats.to_dict("records")]
            cflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype="flat", camera=camera, expnum=expnum_str)
            os.makedirs(os.path.dirname(cflat_path), exist_ok=True)
            image_tasks.create_master_frame(in_images=dflat_paths, out_image=cflat_path)

            # define paths
            lflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="l", imagetype="flat", camera=camera, expnum=expnum_str)
            dstray_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="stray", camera=camera, expnum=expnum_str)
            dratio_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="ratio", camera=camera, expnum=expnum_str)

            cent_guess_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="ntrace_guess", camera=camera)
            flux_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="namp", camera=camera)
            cent_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="ntrace", camera=camera)
            fwhm_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="nwidth", camera=camera)
            model_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="nmodel", camera=camera)

            # first centroids trace
            if skip_done and os.path.isfile(cent_guess_path):
                log.info(f"skipping {cent_guess_path}, file already exist")
            else:
                log.info(f"going to trace centroids fibers in {camera}")
                centroids, img = image_tasks.trace_centroids(in_image=cflat_path, out_trace_cent=cent_guess_path,
                                                             correct_ref=True, median_box=(1,10), coadd=20, counts_threshold=counts_threshold,
                                                             max_diff=1.5, guess_fwhm=2.5, method="gauss", ncolumns=cent_guess_ncolumns,
                                                             fit_poly=fit_poly, poly_deg=poly_deg_cent,
                                                             interpolate_missing=True)

            # subtract stray light only if imagetyp is flat
            if skip_done and os.path.isfile(lflat_path):
                log.info(f"skipping {lflat_path}, file already exist")
            else:
                image_tasks.subtract_straylight(in_image=cflat_path, out_image=lflat_path, out_stray=dstray_path,
                                                in_cent_trace=cent_guess_path, parallel=1, **STRAYLIGHT_PARS)

            if skip_done and os.path.isfile(flux_path) and os.path.isfile(cent_path) and os.path.isfile(fwhm_path):
                log.info(f"skipping {flux_path}, {cent_path} and {fwhm_path}, files already exist")
            else:
                log.info(f"going to trace fibers in {camera}")
                centroids, trace_cent_fit, trace_flux_fit, trace_fwhm_fit, img_stray, model, mratio = image_tasks.trace_fibers(
                    in_image=lflat_path,
                    out_trace_amp=flux_path, out_trace_cent=cent_path, out_trace_fwhm=fwhm_path, out_model=model_path,
                    in_trace_cent_guess=cent_guess_path,
                    median_box=(1,10), coadd=20,
                    counts_threshold=counts_threshold, max_diff=1.5, fwhms_guess=2.5,
                    ncolumns=trace_full_ncolumns, fwhm_limits=(1.5, 4.5),
                    fit_poly=fit_poly, interpolate_missing=True,
                    poly_deg=(poly_deg_amp, poly_deg_cent, poly_deg_width)
                )

            # eval model continuum and ratio
            if skip_done and os.path.isfile(model_path) and os.path.isfile(dratio_path):
                log.info(f"skipping {model_path}, file already exist")
            else:
                log.info(f"going to create model image and mode/exposure ratio in {camera}")
                if "trace_cent_fit" not in locals():
                    trace_cent_fit = TraceMask.from_file(cent_path)
                    trace_flux_fit = TraceMask.from_file(flux_path)
                    trace_fwhm_fit = TraceMask.from_file(fwhm_path)
                    img_stray = loadImage(lflat_path)
                    img_stray.setData(data=np.nan_to_num(img_stray._data), error=np.nan_to_num(img_stray._error))
                    img_stray = img_stray.replaceMaskMedian(1, 10, replace_error=None)
                    img_stray._data = np.nan_to_num(img_stray._data)
                    img_stray = img_stray.medianImg((1,10), propagate_error=True)
                    img_stray = img_stray.convolveImg(np.ones((1, 20), dtype="uint8"))
                model, ratio = img_stray.eval_fiber_model(trace_cent_fit, trace_fwhm_fit, trace_flux_fit)
                model.writeFitsData(model_path)
                ratio.writeFitsData(dratio_path)


def create_traces(mjd, epochs=None, cameras=CAMERAS, ring="primary",
                  use_longterm_cals=True,
                  cent_guess_ncolumns=140,
                  trace_full_ncolumns=40,
                  skip_done=True, dry_run=False):
    """Create traces from master dome flats

    Given a set of MJDs and (optionally) exposure numbers, create traces from
    the master dome flats. This routine will store the master traces in the
    corresponding calibration directory in the `masters_mjd` or by default in
    the smallest MJD in `mjds`.

    If the corresponding master dome flats do not exist, they will be created
    first. Otherwise they will be read from disk.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    cameras : list or tuple, optional
        List of cameras (e.g., b2, z3) to create traces for
    use_longterm_cals : bool
        Whether to use long-term calibration frames or not, defaults to True
    fit_poly : bool, optional
        Fit polynomials to traces, by default True
    poly_deg_amp : int, optional
        Degree of the polynomial to fit to the amplitude, by default 5
    poly_deg_cent : int, optional
        Degree of the polynomial to fit to the centroids, by default 4
    poly_deg_width : int, optional
        Degree of the polynomial to fit to the widths, by default 5
    skip_done : bool, optional
        Skip pipeline steps that have already been done, by default True
    """

    epoch = get_calibration_epoch(mjd=mjd, **(epochs or {}).get(mjd, {}))
    mjds = epoch["trace"]

    frames = md.get_calibrations_metadata(mjds=mjds, calibration="trace")
    camera_frames = {}
    for camera in cameras:
        lamp = MASTER_CON_LAMPS[camera[0]]
        frames_lamp = frames.loc[frames[lamp]]
        frames_lamp, _ = choose_sequence(frames=frames_lamp, calibration="trace", kind="longterm", ring=ring)
        camera_frames[camera] = get_standards_sequence(frames_lamp, camera=camera, ring=ring)

    frames = pd.concat([sequence for sequence in camera_frames.values()])
    expnums = sorted(frames.expnum.unique())

    if frames.empty:
        log.error("no dome flat frames found, skipping production of fiber traces")
        return

    # define master paths for target frames
    calibs = get_calib_paths(mjd=mjd, version=drpver, longterm_cals=use_longterm_cals, flavors=CALIBRATION_NEEDS["trace"])

    if dry_run:
        _log_dry_run(frames, calibs=calibs, settings=None, caller=create_traces.__name__)
        return

    # run 2D reduction on flats: preprocessing, detrending
    reduce_2d(mjds, calibrations=calibs, expnums=expnums, cameras=cameras, reject_cr=False,
              add_astro=False, sub_straylight=False, skip_done=skip_done)

    # iterate through exposures with std fibers exposed
    for camera in cameras:
        # select fibers in current spectrograph
        fibermap = SLITMAP[SLITMAP["spectrographid"] == int(camera[1])]
        # initialize fiber traces
        columns = np.linspace(5, 4080, trace_full_ncolumns, dtype="int")
        fibers_params = {
            "counts": TraceMask.create_empty(data_dim=(LVM_NFIBERS, LVM_NCOLS), slitmap=SLITMAP, samples_columns=columns),
            "centroids": TraceMask.create_empty(data_dim=(LVM_NFIBERS, LVM_NCOLS), slitmap=SLITMAP, samples_columns=columns),
            "sigmas": TraceMask.create_empty(data_dim=(LVM_NFIBERS, LVM_NCOLS), slitmap=SLITMAP, samples_columns=columns)
        }

        models = []
        exposed_stds, unexposed_stds = get_sequence_iterator(camera_frames[camera], camera)
        for expnum, (std_fiberid, block_idxs) in exposed_stds.items():
            frame = camera_frames[camera].query("expnum == @expnum").squeeze()

            # first fibers fitting (guess using pure Gaussian profiles) using first exposure in sequence
            dflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=frame.mjd, kind="d", imagetype="flat", camera=camera, expnum=expnum)
            counts_guess_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=frame.mjd, kind="d", imagetype="counts_guess", camera=camera, expnum=expnum)
            centroids_guess_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=frame.mjd, kind="d", imagetype="centroids_guess", camera=camera, expnum=expnum)
            sigmas_guess_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=frame.mjd, kind="d", imagetype="sigmas_guess", camera=camera, expnum=expnum)
            guess_paths = {
                "counts": counts_guess_path,
                "centroids": centroids_guess_path,
                "sigmas": sigmas_guess_path
            }
            guess_paths_exist = [os.path.isfile(guess_path) for guess_path in guess_paths.values()]
            if skip_done and all(guess_paths_exist):
                for guess_path in guess_paths.values():
                    log.info(f"skipping {guess_path}, file already exist")
            else:
                log.info(f"going to trace all fibers in {camera}")
                image_tasks.guess_fibers_params(in_image=dflat_path, out_fiber_guess=guess_paths,
                                                coadd=20, counts_range=[0.0, np.inf], centroids_range=[-2.0, +2.0], fwhms_range=[2.0, 3.5],
                                                ncolumns=cent_guess_ncolumns)
            # define paths
            dflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=frame.mjd, kind="d", imagetype="flat", camera=camera, expnum=expnum)
            lflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=frame.mjd, kind="l", imagetype="flat", camera=camera, expnum=expnum)
            dstray_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=frame.mjd, kind="d", imagetype="stray", camera=camera, expnum=expnum)

            # subtract stray light only if imagetyp is flat
            if skip_done and os.path.isfile(lflat_path):
                log.info(f"skipping {lflat_path}, file already exist")
            else:
                image_tasks.subtract_straylight(in_image=dflat_path, out_image=lflat_path, out_stray=dstray_path,
                                                in_cent_trace=guess_paths["centroids"], parallel=1, **STRAYLIGHT_PARS)

            log.info(f"going to trace std fiber {std_fiberid} in {camera} within {block_idxs = }")
            fitted_params, img, model, _ = image_tasks.fit_fibers_params(in_image=lflat_path, in_fiber_guess=guess_paths, coadd=20,
                                                                            ncolumns=trace_full_ncolumns, iblocks=block_idxs,
                                                                            measuring_conf=FIBER_MEASURING_CONFIG, smoothing_conf=FIBER_SMOOTHING_CONFIG)

            # update master traces
            log.info(f"{camera = }, {expnum = }, {std_fiberid = }")
            for name in fitted_params:
                fibers_params[name].setHeader(img._header)

                fibers_param = fibers_params[name]
                fitted_param = fitted_params[name]
                for iblock in block_idxs:
                    fibers_param.set_block(iblock=iblock, from_instance=fitted_param.get_block(iblock))

            # store blocks models
            models.append(model)

        for name, fibers_param in fibers_params.items():
            fibers_param.setHeader(img._header)
            # masking bad fibers
            bad_fibers = fibermap["fibstatus"] == 1
            fibers_param._mask[bad_fibers] = True
            # masking untraced standard fibers (two cases: 1. not set for tracing and 2. flagged during tracing)
            untraced_fibers = np.isin(fibermap["orig_ifulabel"].value, unexposed_stds)
            fibers_param._mask[untraced_fibers] = True
            # interpolate master traces in missing fibers
            fibers_param.interpolate_data(axis="Y", reset_mask=True)

            fibers_param.writeFitsData(path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind=f"m{name}"))

        # store final model and ratio
        model = image_tasks.combineImages(images=models, method="sum", normalize=False, background_subtract=False, replace_with_nan=False)
        model.writeFitsData(path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mmodel"))
        (model / img).writeFitsData(path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mratio"))


def create_dome_fiberflats(mjd, expnums_ldls=None, expnums_qrtz=None, cals_mjd=None, use_longterm_cals=True, kind="longterm", skip_done=True, dry_run=False):
    """Create fiberflats from dome flats

    Parameters
    ----------
    mjd : int
        MJD to reduce
    expnums_ldls : list
        List of exposure numbers for LDLS dome flats
    expnums_qrtz : list
        List of exposure numbers for quartz dome flats
    use_longterm_cals : bool
        Whether to use long-term calibration frames or not, defaults to True
    kind : str, optional
        Kind of calibration frames to produce, by default 'longterm'
    skip_done : bool
        Skip pipeline steps that have already been done
    dry_run : bool, optional
        Logs useful information abaut the current setup without actually reducing, by default False
    """
    log.error("current implementation of dome flats needs updating, skipping dome fiberflat creation")
    return

    if expnums_ldls is not None and expnums_qrtz is not None:
        expnums = np.concatenate([expnums_ldls, expnums_qrtz])
    else:
        expnums = None

    frames = md.get_calibrations_metadata(mjd, calibration="dome", expnums=expnums)
    if frames.empty:
        log.error("no dome flat frames found, skipping production of dome fiberflats")
        return

    if expnums_ldls is None or expnums_qrtz is None:
        frames, expnums = choose_sequence(frames, flavor="dome", kind=kind)

    # define master paths for target frames
    calibs = get_calib_paths(mjd=cals_mjd or mjd, version=drpver, longterm_cals=use_longterm_cals, flavors=CALIBRATION_NEEDS["dome"])
    calibs_grp = calibs.copy()

    if dry_run:
        _log_dry_run(frames, calibs=calibs, settings=None, caller=create_dome_fiberflats.__name__)
        return

    # run 2D reduction on flats: preprocessing, detrending
    reduce_2d(mjd, calibrations=calibs, expnums=expnums, reject_cr=False, add_astro=False, sub_straylight=True, skip_done=skip_done)

    for channel, lamp in MASTER_CON_LAMPS.items():
        # read original combined dome flats and run extraction
        flats = frames.loc[(frames[lamp])&(frames["camera"].str.startswith(channel))]
        if flats.expnum.min() != flats.expnum.max():
            expnum_str = f"{flats.expnum.min():>08}_{flats.expnum.max():>08}"
        else:
            expnum_str = flats.expnum.min()
        xflat_paths = []
        for i, specid in enumerate("123"):
            camera = f"{channel}{specid}"
            cflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype="flat", camera=camera, expnum=expnum_str)
            xflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="flat", camera=camera, expnum=expnum_str)
            if skip_done and os.path.isfile(xflat_path):
                log.info(f"skipping {xflat_path}, file already exists")
            else:
                image_tasks.extract_spectra(in_image=cflat_path, out_rss=xflat_path,
                                            in_trace=calibs["centroids"][camera],
                                            in_sigma=calibs["sigmas"][camera],
                                            in_model=calibs["model"][camera])
            xflat_paths.append(xflat_path)
        xflat = RSS.from_spectrographs(*[RSS.from_file(xflat_path) for xflat_path in xflat_paths])

        if kind == "longterm":
            mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mfiberflat_dome", camera=channel)
        else:
            mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="nfiberflat_dome", camera=channel)

        # group calibrations in channels to build lvmFlat products
        for flavor in {"centroids", "sigmas", "wave", "lsf"}:
            calibs_grp[flavor] = group_calib_paths(calibs[flavor])

        # read calibrations
        mcent = TraceMask.from_spectrographs(*[TraceMask.from_file(mtrace_path) for mtrace_path in calibs_grp["centroids"][channel]])
        mwidth = TraceMask.from_spectrographs(*[TraceMask.from_file(mwidth_path) for mwidth_path in calibs_grp["sigmas"][channel]])
        mwave = TraceMask.from_spectrographs(*[TraceMask.from_file(mwave_path) for mwave_path in calibs_grp["wave"][channel]])
        mlsf = TraceMask.from_spectrographs(*[TraceMask.from_file(mlsf_path) for mlsf_path in calibs_grp["lsf"][channel]])
        # normalize by median fiber
        fflat = RSS(data=xflat._data, error=xflat._error, mask=xflat._mask, wave_trace=mwave, lsf_trace=mlsf, header=xflat._header)
        fflat = fflat.rectify_wave(method="linear", wave_range=SPEC_CHANNELS[channel], wave_disp=0.5)
        median_fiber = bn.nanmedian(fflat._data, axis=0)
        fflat._data = fflat._data / median_fiber
        fflat.set_wave_trace(mwave)
        fflat.set_lsf_trace(mlsf)
        fflat = to_native_wave(fflat)
        fflat.writeFitsData(mflat_path)
        # create lvmFlat object
        lvmflat = lvmFlat(data=xflat._data / fflat._data, error=xflat._error, mask=xflat._mask, header=xflat._header,
                          cent_trace=mcent, width_trace=mwidth,
                          wave_trace=mwave, lsf_trace=mlsf,
                          superflat=fflat._data, slitmap=SLITMAP)
        lvmflat.writeFitsData(path.full("lvm_frame", mjd=mjd, tileid=11111, drpver=drpver, expnum=expnum_str, kind=f'DFlat-{channel}'))


def create_twilight_fiberflats(mjd: int, epochs: dict[int, dict] = None, cals_mjd: int = None, use_longterm_cals: bool = True,
                      ref_kind: Union[int, Callable[[np.ndarray, int], np.ndarray]] = bn.nanmedian,
                      groupby: str = "spec", guess_coeffs: List[int] = [1,0,0,0], fixed_coeffs: List[int] = [0,1,2,3],
                      cnorms: Dict[str, float] = SKYLINES_FIBERFLAT, dwave: float = 20.0,
                      smoothing: float = 0.0,
                      interpolate_invalid: bool = True,
                      skip_done: bool = False,
                      display_plots: bool = False,
                      dry_run: bool = False) -> None:
    """Reduce a sequence of twilight exposures and produce master twilight fiberflats for each channel.

    This function processes a set of twilight flat exposures for the specified MJD and exposure numbers,
    extracting 1D spectra, calibrating wavelength and LSF, rectifying, and fitting fiber throughput.
    The resulting master fiberflats are interpolated to handle masked fibers and saved to disk.
    Optionally, diagnostic plots can be displayed.

    Parameters
    ----------
    mjd : int
        MJD to reduce.
    use_longterm_cals : bool, optional
        Whether to use long-term calibration frames. Defaults to True.
    expnums : list[int], optional
        List of twilight exposure numbers to process. If None, all available are used.
    ref_kind : int or callable, optional
        Reference fiber selection method or index. Defaults to nanmedian.
    groupby : str, optional
        Grouping for normalization (e.g., "spec"). Defaults to "spec".
    guess_coeffs : list[int], optional
        Initial guess for polynomial coefficients in gradient fitting. Defaults to [1,0,0,0].
    fixed_coeffs : list[int], optional
        Indices of coefficients to fix during fitting. Defaults to [1,2,3].
    cnorms : dict, optional
        Dictionary of normalization wavelengths per channel. Defaults to SKYLINES_FIBERFLAT.
    dwave : float, optional
        Width of the wavelength window for normalization. Defaults to 20.0.
    smoothing : float, optional
        Smoothing parameter for fiberflat fitting. Defaults to 0.0.
    interpolate_invalid : bool, optional
        Interpolate over invalid/masked fibers. Defaults to True.
    skip_done : bool, optional
        Skip files that already exist. Defaults to False.
    display_plots : bool, optional
        Display diagnostic plots. Defaults to False.
    dry_run : bool, optional
        Logs useful information abaut the current setup without actually reducing, by default False
    """
    epoch = get_calibration_epoch(mjd=mjd, **(epochs or {}).get(mjd, {}))
    mjds = epoch["twilight"]

    frames = md.get_calibrations_metadata(mjds=mjds, calibration="twilight")
    frames, expnums = choose_sequence(frames, calibration="twilight", kind="longterm")
    if frames.empty:
        log.error("no twilight frames found, skipping production of twilight fiberflats")
        return

    # define master paths for target frames
    calibs = get_calib_paths(mjd=cals_mjd or mjd, version=drpver, longterm_cals=use_longterm_cals, flavors=CALIBRATION_NEEDS["twilight"])

    if dry_run:
        _log_dry_run(frames, calibs=calibs, settings=None, caller=create_dome_fiberflats.__name__)
        return

    # 2D reduction of twilight sequence
    reduce_2d(mjds=mjds, calibrations=calibs, expnums=expnums, reject_cr=True,
              add_astro=False, sub_straylight=True, skip_done=skip_done, **STRAYLIGHT_PARS)

    for flat in frames.to_dict("records"):
        camera = flat["camera"]

        # extract 1D spectra for each frame
        lflat_path = path.full("lvm_anc", drpver=drpver, kind="l", imagetype="flat", **flat)
        xflat_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype="flat", **flat)

        if skip_done and os.path.isfile(xflat_path):
            log.info(f"skipping {xflat_path}, file already exist")
        else:
            image_tasks.extract_spectra(in_image=lflat_path, out_rss=xflat_path,
                                        in_trace=calibs["centroids"][camera],
                                        in_sigma=calibs["sigmas"][camera],
                                        in_model=calibs["model"][camera])

    # group calibs
    for flavor in ["centroids", "sigmas", "wave", "lsf"]:
        calibs[flavor] = group_calib_paths(calibs[flavor])

    # decompose twilight spectra into sun continuum and twilight components
    channels = "brz"
    flat_channels = frames.groupby(frames.camera.str.__getitem__(0))
    for channel in channels:
        flat_expnums = flat_channels.get_group(channel).groupby("expnum")
        xtwi_paths, fflat_paths, lvmflat_paths = [], [], []
        for expnum in flat_expnums.groups:
            flat = flat_expnums.get_group(expnum).iloc[0]

            xflat_paths = sorted(path.expand("lvm_anc", drpver=drpver, kind="x", imagetype=flat.imagetyp, tileid=11111, mjd=flat.mjd, camera=f"{channel}?", expnum=expnum))
            fflat_flatfielded_path = path.full("lvm_anc", drpver=drpver, kind="flatfielded_",
                                   imagetype=flat.imagetyp, tileid=11111, mjd=flat.mjd,
                                   camera=channel, expnum=expnum)
            fflat_path = path.full("lvm_anc", drpver=drpver, kind="f", imagetype="flat", tileid=11111, mjd=flat.mjd, camera=channel, expnum=expnum)
            fflat_paths.append(fflat_path)
            xflat_path = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=flat.imagetyp, tileid=11111, mjd=flat.mjd, camera=channel, expnum=expnum)
            wflat_path = path.full("lvm_anc", drpver=drpver, kind="w", imagetype=flat.imagetyp, tileid=11111, mjd=flat.mjd, camera=channel, expnum=expnum)
            hflat_path = path.full("lvm_anc", drpver=drpver, kind="h", imagetype=flat.imagetyp, tileid=11111, mjd=flat.mjd, camera=channel, expnum=expnum)
            xtwi_paths.append(xflat_path)
            lvmflat_paths.append(path.full("lvm_frame", mjd=mjd, tileid=11111, drpver=drpver, expnum=expnum, kind=f'TFlat-{channel}'))

            if skip_done and os.path.isfile(hflat_path):
                log.info(f"skipping {hflat_path}, file already exist")
            else:
                # spectrograph stack xflats
                rss_tasks.stack_spectrographs(in_rsss=xflat_paths, out_rss=xflat_path)

                # calibrate in wavelength
                rss_tasks.create_pixel_table(in_rss=xflat_path, out_rss=wflat_path, in_waves=calibs["wave"][channel], in_lsfs=calibs["lsf"][channel])

                # match LSF in all fibers
                rss_tasks.match_resolution(in_rss=wflat_path, out_rss=wflat_path, target_fwhm=4.5)

                # rectify in wavelength
                rss_tasks.resample_wavelength(in_rss=wflat_path, out_rss=hflat_path, wave_disp=0.5, wave_range=SPEC_CHANNELS[channel])


            # fit fiber throughput
            fit_fiberflat(in_rss=hflat_path, out_flat=fflat_path, out_rss=fflat_flatfielded_path,
                          ref_kind=ref_kind, groupby=groupby, guess_coeffs=guess_coeffs, fixed_coeffs=fixed_coeffs,
                          norm_cwave=cnorms[channel], norm_dwave=dwave, smoothing=smoothing, interpolate_invalid=interpolate_invalid,
                          display_plots=display_plots)

        # combine individual fiberflats into master fiberflat
        mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mfiberflat_twilight", camera=channel)
        combine_twilight_sequence(
            in_twilights=xtwi_paths,
            in_fflats=fflat_paths, out_mflat=mflat_path, out_lvmflats=lvmflat_paths,
            in_cents=calibs["centroids"][channel], in_widths=calibs["sigmas"][channel],
            in_waves=calibs["wave"][channel], in_lsfs=calibs["lsf"][channel])


def create_fiberflats_corrections(cals_mjd: int, science_mjds: Union[int, List[int]], use_longterm_cals: bool = True, science_expnums: List[int] = None,
                                  sky_cwaves: Dict[str, float] = SKYLINES_FIBERFLAT, cont_cwaves: Dict[str, float] = CONTINUUM_FIBERFLAT,
                                  groupby: str = "spec", quantiles: Tuple[float, float] = (5.0, 97.0), sky_fibers_only: bool = False,
                                  nsigma: float = 2.0, comb_method: str = "median", force_correction: bool = False,
                                  skip_done: bool = False, display_plots: bool = False, dry_run: bool = False) -> None:

    if not all([cals_mjd <= sci_mjd for sci_mjd in science_mjds]):
        log.error(f"some science MJDs are earlier than {cals_mjd = }: {science_mjds = }")
        return

    science_mjds = [science_mjds] if isinstance(science_mjds, int) else science_mjds
    if science_expnums is None:
        frames = pd.concat([md.get_frames_metadata(mjd=mjd).query("tileid != 11111 and qaqual != 'BAD'") for mjd in science_mjds], ignore_index=True)
        science_expnums = frames.sort_values("expnum").drop_duplicates("expnum").expnum

    calibs = get_calib_paths(mjd=cals_mjd, version=drpver, longterm_cals=use_longterm_cals, flavors=CALIBRATION_NEEDS["object"])

    if dry_run:
        _log_dry_run(frames, calibs=calibs, settings=None, caller=create_fiberflats_corrections.__name__)
        return

    # 2D and 1D reduction of science exposures
    for sci_mjd in science_mjds:
        reduce_2d(mjds=sci_mjd, calibrations=calibs, expnums=science_expnums, reject_cr=True, add_astro=True, sub_straylight=True, skip_done=skip_done)
        reduce_1d(mjd=sci_mjd, calibrations=calibs, expnums=science_expnums, sub_straylight=True, skip_done=skip_done)

    for channel in "brz":
        wframe_paths = get_frames_paths(mjds=science_mjds, kind="w", camera_or_channel=channel, expnums=science_expnums)
        if len(wframe_paths) == 0:
            log.error(f"no good quality science frames found for {science_mjds = }, {science_expnums = } in {channel = }")

        fit_skyline_flatfield(
            in_sciences=wframe_paths,
            in_mflat=calibs["fiberflat_twilight"][channel],
            out_mflat=calibs["fiberflat_twilight"][channel],
            groupby=groupby,
            guess_coeffs=[1,0,0,0], fixed_coeffs=[0,1,2,3],
            sky_cwave=sky_cwaves[channel], cont_cwave=cont_cwaves[channel], dwave=20.0,
            quantiles=quantiles, sky_fibers_only=sky_fibers_only,
            nsigma=nsigma, comb_method=comb_method,
            force_correction=force_correction,
            display_plots=display_plots)


def create_illumination_corrections(mjd, use_longterm_cals=True, expnums=None):
    """Create illumination corrections from master dome and twilight flats

    Given a set of MJDs and (optionally) exposure numbers, create illumination
    corrections from the master dome and twilight flats. This routine will
    store the master illumination corrections in the corresponding calibration
    directory in the `masters_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding master dome and twilight flats do not exist, they
    will be created first. Otherwise they will be read from disk.

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    use_longterm_cals : bool
        Whether to use long-term calibration frames or not, defaults to True
    expnums : list
        List of exposure numbers to reduce
    """
    # read master fiber flats (dome and twilight)
    # calculate ratio twilight/dome
    raise NotImplementedError("create_illumination_corrections")


def create_wavelengths(mjd, epochs=None, use_longterm_cals=True, kind="longterm", skip_done=True, dry_run=False):
    """Reduces an arc sequence to create master wavelength solutions

    Given a set of MJDs and (optionally) exposure numbers, create wavelength
    solutions from the master arcs. This routine will store the master
    wavelength solutions in the corresponding calibration directory in the
    `masters_mjd` or by default in the smallest MJD in `mjds`.

    If the corresponding master arcs do not exist, they will be created first.
    Otherwise they will be read from disk.

    Parameters
    ----------
    mjd : int
        MJD to reduce
    use_longterm_cals : bool
        Whether to use long-term calibration frames or not, defaults to True
    expnums : list
        List of exposure numbers to reduce
    kind : str, optional
        Kind of calibration frames to produce, by default 'longterm'
    skip_done : bool
        Skip pipeline steps that have already been done
    dry_run : bool, optional
        Logs useful information abaut the current setup without actually reducing, by default False
    """
    # run wavelength calibration for special MJDs
    if mjd == 60177:
        log.info(f"running dedicated script to create wavelength calibrations for MJD = {mjd}")
        _create_wavelengths_60177(use_longterm_cals=use_longterm_cals, skip_done=skip_done, dry_run=dry_run)
        return

    epoch = get_calibration_epoch(mjd=mjd, **(epochs or {}).get(mjd, {}))
    mjds = epoch["wave"]

    frames = md.get_calibrations_metadata(mjds=mjds, calibration="wave")
    frames, expnums = choose_sequence(frames, calibration="wave", kind=kind)
    if frames.empty:
        log.error("no arc frames found, skipping production of wavelength calibrations")
        return

    # define master paths for target frames
    calibs = get_calib_paths(mjd=mjd, version=drpver, longterm_cals=use_longterm_cals, flavors=CALIBRATION_NEEDS["wave"])

    if dry_run:
        _log_dry_run(frames, calibs=calibs, settings=None, caller=create_wavelengths.__name__)
        return

    reduce_2d(mjds, calibrations=calibs, expnums=expnums, assume_imagetyp="arc", reject_cr=False,
              add_astro=False, sub_straylight=False, skip_done=skip_done)

    if frames.expnum.min() != frames.expnum.max():
        expnum_str = f"{frames.expnum.min():>08}_{frames.expnum.max():>08}"
    else:
        expnum_str = frames.expnum.min()
    arc_analogs = frames.groupby(["camera",])
    for camera in arc_analogs.groups:
        arcs = arc_analogs.get_group((camera,))

        # define product paths
        carc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype="arc", camera=camera, expnum=expnum_str)
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=camera, expnum=expnum_str)
        if kind == "longterm":
            mwave_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mwave")
            mlsf_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="mlsf")
        else:
            mwave_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="nwave")
            mlsf_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="nlsf")
        os.makedirs(os.path.dirname(carc_path), exist_ok=True)

        # combine individual arcs into master arc
        if skip_done and os.path.isfile(carc_path):
            log.info(f"skipping combined arc {carc_path}, file already exists")
        else:
            darc_paths = [path.full("lvm_anc", drpver=drpver, kind="d", imagetype="arc", **arc) for arc in arcs.to_dict("records")]
            image_tasks.create_master_frame(in_images=darc_paths, out_image=carc_path, batch_size=48)

        # TODO: maybe subtract stray light?

        # extract arc
        if skip_done and os.path.isfile(xarc_path):
            log.info(f"skipping extracted arc {xarc_path}, file already exists")
        else:
            image_tasks.extract_spectra(in_image=carc_path, out_rss=xarc_path,
                                        in_trace=calibs["centroids"][camera],
                                        in_sigma=calibs["sigmas"][camera],
                                        in_model=calibs["model"][camera])

        # fit wavelength solution
        ref_lines, _, cent_wave, _, rss, wave_trace, fwhm_trace = rss_tasks.determine_wavelength_solution(
            in_arcs=xarc_path,
            out_wave=mwave_path,
            out_lsf=mlsf_path)

        lvmarc = lvmArc(data=rss._data, error=rss._error, mask=rss._mask, header=rss._header,
                        ref_wave=ref_lines, cent_line=cent_wave,
                        wave_trace=wave_trace, lsf_trace=fwhm_trace)
        lvmarc.writeFitsData(path.full("lvm_frame", mjd=mjd, tileid=11111, drpver=drpver, expnum=expnum_str, kind=f'Arc-{camera}'))

    for channel in "brz":
        if kind == "longterm":
            mwave_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="mwave"))
            mlsf_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="mlsf"))
        else:
            mwave_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="nwave"))
            mlsf_paths = sorted(path.expand("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, camera=f"{channel}?", kind="nlsf"))

        xarc_paths = sorted(path.expand("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=f"{channel}?", expnum=expnum_str))
        xarc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="x", imagetype="arc", camera=channel, expnum=expnum_str)
        harc_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="h", imagetype="arc", camera=channel, expnum=expnum_str)

        # stack spectragraphs
        rss_tasks.stack_spectrographs(in_rsss=xarc_paths, out_rss=xarc_path)
        # apply wavelength solution to arcs and rectify
        rss_tasks.create_pixel_table(in_rss=xarc_path, out_rss=harc_path, in_waves=mwave_paths, in_lsfs=mlsf_paths)
        rss_tasks.resample_wavelength(in_rss=harc_path, out_rss=harc_path, method="linear", wave_range=SPEC_CHANNELS[channel], wave_disp=0.5)


def detrend_calibrations(mjd, calibration, dry_run=False, skip_done=True):
    frames = md.get_calibrations_metadata(mjds=mjd, calibration=calibration)
    frames = frames.loc[~frames.calibfib.isin(STD_FIBER_LABELS)]
    if frames.empty:
        log.error("no bias frames found, skipping production of bias frames")
        return

    # define master paths for target frames
    calibs = get_calib_paths(mjd=mjd, flavors=CALIBRATION_NEEDS[calibration], from_sandbox=True)

    if dry_run:
        _log_dry_run(frames, calibs=calibs, settings=None, caller=detrend_calibrations.__name__)
        return

    # preprocess and detrend frames
    reduce_2d(mjds=mjd, calibrations=calibs, expnums=frames.expnum.unique(), reject_cr=False, add_astro=False, sub_straylight=False, skip_done=skip_done)


def reduce_nightly_sequence(mjd, use_longterm_cals=False, reject_cr=True, only_cals=CALIBRATION_TYPES,
                            counts_thresholds=COUNTS_THRESHOLDS, cent_guess_ncolumns=140, trace_full_ncolumns=40,
                            extract_metadata=False, skip_done=True, keep_ancillary=False,
                            fflats_from=None, link_pixelmasks=True, dry_run=False):
    """Reduces the nightly calibration sequence:

    The nightly calibration sequence consists of the following exposures:
        * 7 - 9 bias
        * 2 dome flat (LDLS and quartz)
        * 2 arc (10s and 50s exposures)
        * ~12 - 24 twilight (~half exposures at dawn and twilight)

    This routine will create *nightly* (not long-term) master calibrations
    at $LVM_SPECTRO_REDUX/{drpver}/0011XX/11111/{mjd}/calib

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    use_longterm_cals : bool
        Whether to use long-term calibration frames or not, defaults to False
    reject_cr : bool
        Reject cosmic rays in 2D reduction, by default True
    only_cals : list, tuple or set
        Only produce this calibrations, by default {'bias', 'trace', 'wave', 'dome', 'twilight'}
    extract_metadata : bool, optional
        Extract or use cached metadata if exist, by default False (use cache)
    skip_done : bool
        Skip pipeline steps that have already been done
    keep_ancillary : bool
        Keep ancillary files, by default False
    fflats_from : int, optional
        Copy twilight fiberflats from given MJD, by default None (no copy)
    link_pixelmasks : bool, optional
        Create a symbolic link of current version of pixel mask and pixel flats to current version, by default True
    dry_run : bool, optional
        Logs useful information abaut the current setup without actually reducing, by default False
    """
    # start logging to file
    start_logging(mjd, tileid=11111)

    if mjd is None:
        log.error(f"nothing to reduce, MJD = {mjd}")
        return

    # create symbolic link to pixel flats and masks
    if link_pixelmasks:
        _link_pixelmasks()

    if not set(only_cals).issubset(CALIBRATION_TYPES):
        raise ValueError(f"some chosen image types in 'only_cals' are not valid: {only_cals.difference(CALIBRATION_TYPES)}")
    log.info(f"going to produce nightly calibrations: {only_cals}")

    if "bias" in only_cals:
        create_bias(mjd=mjd, use_longterm_cals=use_longterm_cals, skip_done=skip_done, dry_run=dry_run)

    if "trace" in only_cals:
        create_nightly_traces(mjd=mjd, use_longterm_cals=use_longterm_cals,
                              counts_thresholds=counts_thresholds,
                              cent_guess_ncolumns=cent_guess_ncolumns,
                              trace_full_ncolumns=trace_full_ncolumns,
                              skip_done=skip_done, dry_run=dry_run)

    if "wave" in only_cals:
        create_wavelengths(mjd=mjd, use_longterm_cals=use_longterm_cals, kind="nightly", skip_done=skip_done, dry_run=dry_run)

    if "dome" in only_cals:
        create_dome_fiberflats(mjd=mjd, use_longterm_cals=use_longterm_cals, kind="nightly", skip_done=skip_done, dry_run=dry_run)

    if "twilight" in only_cals:
        create_twilight_fiberflats(mjd=mjd, use_longterm_cals=use_longterm_cals, skip_done=skip_done, dry_run=dry_run)

    # if not keep_ancillary:
    #     _clean_ancillary(mjd)


def reduce_longterm_sequence(mjd, epochs=None, use_longterm_cals=True,
                             reject_cr=True, only_cals=CALIBRATION_TYPES,
                             counts_thresholds=COUNTS_THRESHOLDS,
                             cent_guess_ncolumns=140, trace_full_ncolumns=40,
                             extract_metadata=False,
                             skip_done=True, keep_ancillary=False,
                             link_pixelmasks=True,
                             dry_run=False):
    """Reduces the long-term calibration sequence:

    The long-term calibration sequence consists of the following exposures:
        * 7 - 9 bias
        * 24 dome flat (12: LDLS and 12: quartz)
        * 24 arc (12: 10s and 12: 50s exposures)
        * ~12 - 24 twilight (~half exposures at dawn and twilight)

    This routine will create *long-term* master calibrations
    at $LVM_SANDBOX/calib/{mjd}

    Parameters:
    ----------
    mjd : int
        MJD to reduce
    epochs : dict[int, dict[str,dict]] | None, optional
        A dictionary with specifications of the calibration epoch, by default None
    use_longterm_cals : bool
        Whether to use long-term calibration frames or not, defaults to True
    reject_cr : bool
        Reject cosmic rays in 2D reduction, by default True
    only_cals : list, tuple or set
        Only produce this calibrations, by default {'bias', 'trace', 'wave', 'dome', 'twilight'}
    extract_metadata : bool, optional
        Extract or use cached metadata if exist, by default False (use cache)
    skip_done : bool
        Skip pipeline steps that have already been done
    keep_ancillary : bool
        Keep ancillary files, by default False
    link_pixelmasks : bool, optional
        Create a symbolic link of current version of pixel mask and pixel flats to current version, by default True
    dry_run : bool, optional
        Logs useful information abaut the current setup without actually reducing, by default False
    """
    # start logging to file
    start_logging(mjd, tileid=11111)

    # create symbolic link to pixel flats and masks
    if link_pixelmasks:
        _link_pixelmasks()

    if not set(only_cals).issubset(CALIBRATION_TYPES):
        raise ValueError(f"some chosen image types in 'only_cals' are not valid: {only_cals.difference(CALIBRATION_TYPES)}")
    log.info(f"going to produce long-term calibrations: {only_cals}")

    if "bias" in only_cals:
        create_bias(mjd=mjd, epochs=epochs, use_longterm_cals=use_longterm_cals, skip_done=skip_done, dry_run=dry_run)

    if "trace" in only_cals:
        create_traces(
            mjd=mjd,
            epochs=epochs,
            use_longterm_cals=use_longterm_cals,
            cent_guess_ncolumns=cent_guess_ncolumns,
            trace_full_ncolumns=trace_full_ncolumns,
            skip_done=skip_done,
            dry_run=dry_run)

    if "wave" in only_cals:
        create_wavelengths(mjd=mjd, epochs=epochs, use_longterm_cals=use_longterm_cals, skip_done=skip_done, dry_run=dry_run)

    if "dome" in only_cals:
        create_dome_fiberflats(mjd=mjd, cals_mjd=mjd, use_longterm_cals=use_longterm_cals, kind="longterm", skip_done=skip_done, dry_run=dry_run)

    if "twilight" in only_cals:
        create_twilight_fiberflats(mjd=mjd, epochs=epochs, cals_mjd=mjd, use_longterm_cals=use_longterm_cals, skip_done=skip_done, dry_run=dry_run)

    # if not keep_ancillary:
    #     _clean_ancillary(mjd)


class lvmArc(lvmFrame):
    """LvmArc class"""

    @classmethod
    def from_hdulist(cls, hdulist):
        header = cls.header_from_hdulist(hdulist)

        data = hdulist["FLUX"].data
        error = np.divide(1, hdulist["IVAR"].data, where=hdulist["IVAR"].data != 0, out=np.zeros_like(hdulist["IVAR"].data))
        error = np.sqrt(error)
        mask = hdulist["MASK"].data.astype("bool")
        lxpeak = Table(hdulist["LXPEAK"].data)
        wave_trace = Table(hdulist["WAVE_TRACE"].data)
        lsf_trace = Table(hdulist["LSF_TRACE"].data)
        return cls(data=data, error=error, mask=mask, header=header, lxpeak=lxpeak,
                   wave_trace=wave_trace, lsf_trace=lsf_trace)

    def __init__(self, data=None, error=None, mask=None, ref_wave=None, cent_line=None, lxpeak=None, wave_trace=None, lsf_trace=None, header=None):
        lvmFrame.__init__(self, data=data, error=error, mask=mask, wave_trace=wave_trace, lsf_trace=lsf_trace, header=header)

        self.set_lxpeak(ref_wave, cent_line, lxpeak=lxpeak)

        self._blueprint = dp.load_blueprint(name="lvmArc")
        self._template = dp.dump_template(dataproduct_bp=self._blueprint, save=False)

    def set_lxpeak(self, ref_wave=None, lin_pixel=None, lxpeak=None):
        """Sets a table with the wavelength of identified lamp lines & the corresponding X position in each fiber"""
        # early return in case incomplete data is given
        if lxpeak is None and (ref_wave is None or lin_pixel is None):
            self._lxpeak = None
            return self._lxpeak

        # set the given lxpeak and return
        if lxpeak is not None:
            self._lxpeak = lxpeak
            return self._lxpeak

        self._lxpeak = Table(dtype=[(f"{wave:.4f}", "f4") for wave in ref_wave])
        for ifiber in range(self._fibers):
            self._lxpeak.add_row(lin_pixel[ifiber])

    def writeFitsData(self, out_file, replace_masked=True):
        # replace masked pixels
        if replace_masked:
            self.apply_pixelmask()

        # update headers
        self.update_header()
        # fill in rest of the template
        self._template["FLUX"].data = self._data
        self._template["IVAR"].data = np.divide(1, self._error**2, where=self._error != 0, out=np.zeros_like(self._error))
        self._template["MASK"].data = self._mask.astype("uint8")
        self._template["LXPEAK"] = fits.BinTableHDU(data=self._lxpeak, name="LXPEAK")
        self._template["WAVE_TRACE"] = fits.BinTableHDU(data=self._wave_trace, name="WAVE_TRACE")
        self._template["LSF_TRACE"] = fits.BinTableHDU(data=self._lsf_trace, name="LSF_TRACE")
        self._template.verify("silentfix")

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self._template[0].header["FILENAME"] = os.path.basename(out_file)
        self._template.writeto(out_file, overwrite=True)

