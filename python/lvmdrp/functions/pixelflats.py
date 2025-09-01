import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from pprint import pformat

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.external.fast_median import fast_median_filter_2d
from lvmdrp.utils import metadata as md
from lvmdrp import main as drp
from scipy import ndimage as ndi


PIXFLAT_EPOCHS_PATH = os.path.join(os.getenv("LVMCORE_DIR"), "etc", "pixflat-epochs.yaml")


def _parse_expnums(expnums):
    parsed_expnums = [[]]
    for idx, expnum in enumerate(expnums):
        if isinstance(expnum, int):
            parsed_expnums.append([expnum])
        elif isinstance(expnum, str) and "," in expnum:
            expnum_range = tuple(int(i) for i in expnum.split(","))
            parsed_expnums.append(np.arange(*expnum_range))
        else:
            raise TypeError(f"Invalid type in `expnums` at {idx}: {expnum}")
    parsed_expnums = np.concatenate(parsed_expnums)
    parsed_expnums.sort()
    return parsed_expnums.astype("int")


def _parse_sequence(sequence):
    parsed_sequence = copy(sequence)
    expnums = _parse_expnums(sequence.get("expnums", []) or [])
    rejects = _parse_expnums(sequence.get("rejects", []) or [])
    parsed_sequence["expnums"] = expnums
    parsed_sequence["rejects"] = rejects

    return parsed_sequence


def _expand_sequence(sequence, repeat=False):
    """Expands a sequence dictionary to extract exposure numbers grouped by type.

    The function interprets the "kind" key in the input dictionary to determine
    the types of exposures (e.g., flat, bias, dark) and their respective counts.
    It then uses the "expnums" key to group the exposure numbers accordingly.

    Parameters
    ----------
    sequence : dict
        A dictionary containing the following keys:
        - "kind" : str
            A string where even-indexed characters represent counts and
            odd-indexed characters represent types ('f' for flat, 'b' for bias,
            'd' for dark).
        - "expnums" : list
            A list of exposure numbers.
    repeat : bool, optional
        Whether to pad bias/dark sequences shorter than flat sequence, by default False

    Returns
    -------
    dict
        A dictionary where keys are exposure types (e.g., "flat_expnums",
    """
    typ_maps = {"f": "flat", "b": "bias", "d": "dark"}

    kind = sequence.get("kind")
    kind_ = list(kind)
    typs = kind_[1::2]
    nums = {typ_maps[typ]: num for typ, num in zip(typs, map(int, kind_[::2]))}
    expnums = sequence.get("expnums")
    rejects = sequence.get("rejects", [])

    expnums = np.asarray(list(set(expnums).difference(rejects)))
    expnums.sort()

    expnums_split = np.split(expnums, expnums.size//sum(nums.values()))
    expnums_dict = {typ_maps[typ]: np.array([], dtype="int") for typ in typs}
    for exps in expnums_split:
        offset = 0
        for key in expnums_dict:
            expnums_dict[key] = np.append(expnums_dict[key], exps[offset:offset+nums[key]])
            offset += nums[key]

    if repeat:
        nflats = len(expnums_dict.get("flat", []))
        for key in {"bias", "dark"}:
            expnums_ = expnums_dict.get(key)
            if expnums_ is None:
                continue
            n = len(expnums_)
            expnums_dict[key] = np.repeat(expnums_, nflats//n)
    return expnums_dict

def rsync_enight(mjds):
    """rsyncs egineering nights from LCO directly

    Parameters
    ----------
    mjds : int|list[int]
        MJDs to pull from LCO computer
    """
    pass


def get_enights_metadata(mjds):
    """Returns metadata table for given MJDs of engineering nights

    Parameters
    ----------
    mjds : list[int]
        List of MJDs for a given engineering night run

    Returns
    -------
    pd.DataFrame
        Dataframe containing metadata for the given MJDs
    """
    mjds = np.atleast_1d(mjds)
    metadata = []
    for mjd in mjds:
        metadata.append(md.get_frames_metadata(mjd, overwrite=False, suffix="fits.gz"))
    return pd.concat(metadata, axis="index", ignore_index=True).sort_values("expnum")


def load_pixflat_epochs(epochs_path=None, filter_by=None):
    epochs_path = epochs_path or PIXFLAT_EPOCHS_PATH
    with open(epochs_path) as f:
        epochs = yaml.safe_load(f)["epochs"]

    log.info(f"loaded {len(epochs)} epochs:")
    for mjd in epochs:
        log.info(f"  {mjd}: {pformat(epochs[mjd])}")

    if filter_by is not None and isinstance(filter_by, (list, tuple)):
        log.info(f"filtering by {filter_by}")
        epochs = {mjd: epochs[mjd] for mjd in filter_by if mjd in epochs}
        if len(epochs) == 0:
            log.error(f"epoch(s) {filter_by} not found in calibration epochs file: '{epochs_path}'")
            return epochs
        log.info(f"after filtering {len(epochs)} epoch(s):")
        for mjd in epochs:
            log.info(f"  {mjd}: {epochs[mjd]}")
    return epochs


def detrend_pixelflats(mjds, camera, flat_expnums, bias_expnums=[], dark_expnums=[], use_pixmask=True, skip_done=True):

    frames = get_enights_metadata(mjds=mjds).query("camera == @camera").sort_values("expnum")

    flats = frames.query("expnum in @flat_expnums")
    biases = pd.DataFrame(data={"expnum": [None]*len(flats)})
    darks = pd.DataFrame(data={"expnum": [None]*len(flats)})
    if len(bias_expnums) != 0:
        biases = frames.query("expnum in @bias_expnums")
    if len(dark_expnums) != 0:
        darks = frames.query("expnum in @dark_expnums")

    if use_pixmask:
        mpixmask_path = path.full("lvm_calib", mjd="pixelmasks", kind="pixmask", camera=camera)
    else:
        mpixmask_path = None

    log.info(f"{len(flat_expnums)} flat exposures: {flat_expnums}")
    log.info(f"{len(dark_expnums)} dark exposures: {dark_expnums}")
    log.info(f"{len(bias_expnums)} bias exposures: {bias_expnums}")

    # NOTE: repeating bias and darks if necessary
    if len(darks) < len(flats):
        n = len(flats) / len(darks)
        darks = darks.loc[darks.index.repeat(n)].reset_index(drop=True)
    if len(biases) < len(flats):
        n = len(flats) / len(biases)
        biases = biases.loc[biases.index.repeat(n)].reset_index(drop=True)

    dflat_paths = []
    for (_, bias), (_, dark), (_, flat) in zip(biases.iterrows(), darks.iterrows(), flats.iterrows()):

        # BIAS -----------------------
        if bias.expnum is not None:
            rbias_path = path.full("lvm_raw", hemi="s", mjd=bias.mjd, camspec=camera, expnum=bias.expnum)
            pbias_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=bias.mjd, kind="p", imagetype="bias", expnum=bias.expnum, camera=camera)
            if skip_done and os.path.isfile(pbias_path):
                pass
            else:
                image_tasks.preproc_raw_frame(in_image=rbias_path, out_image=pbias_path, in_mask=mpixmask_path, assume_imagetyp="bias", replace_with_nan=False)
            pbias_path = pbias_path if os.path.isfile(pbias_path) else None
        else:
            pbias_path = None

        # DARKS ----------------------
        if dark.expnum is not None:
            rdark_path = path.full("lvm_raw", hemi="s", mjd=dark.mjd, camspec=camera, expnum=dark.expnum)
            pdark_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=dark.mjd, kind="p", imagetype="dark", expnum=dark.expnum, camera=camera)
            ddark_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=dark.mjd, kind="d", imagetype="dark", expnum=dark.expnum, camera=camera)
            if skip_done and os.path.isfile(pdark_path):
                pass
            else:
                image_tasks.preproc_raw_frame(in_image=rdark_path, out_image=pdark_path, in_mask=mpixmask_path, assume_imagetyp="dark", replace_with_nan=False)
                image_tasks.detrend_frame(in_image=pdark_path, out_image=ddark_path, in_bias=pbias_path, reject_cr=False, replace_with_nan=False)
            ddark_path = ddark_path if os.path.isfile(ddark_path) else None
        else:
            ddark_path = None

        # FLATS ----------------------
        rflat_path = path.full("lvm_raw", hemi="s", mjd=flat.mjd, camspec=camera, expnum=flat.expnum)
        pflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=flat.mjd, kind="p", imagetype="pixflat", expnum=flat.expnum, camera=camera)
        dflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=flat.mjd, kind="d", imagetype="pixflat", expnum=flat.expnum, camera=camera)

        if skip_done and os.path.isfile(dflat_path):
            pass
        else:
            image_tasks.preproc_raw_frame(in_image=rflat_path, out_image=pflat_path, in_mask=mpixmask_path, assume_imagetyp="pixflat", replace_with_nan=False)
            image_tasks.detrend_frame(in_image=pflat_path, out_image=dflat_path, in_bias=pbias_path, reject_cr=False, normalize_pixelflat=False, replace_with_nan=False)
        if os.path.isfile(dflat_path):
            dflat_paths.append(dflat_path)

    return dflat_paths


def combine_pixelflats(mjds, mjd_epoch, camera, flat_expnums, comb_stat="median", skip_done=True):
    frames = get_enights_metadata(mjds=mjds).query("camera == @camera").sort_values("expnum")

    flats = frames.query("expnum in @flat_expnums")
    dflat_paths = [path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=flat.mjd, kind="d", imagetype="pixflat", expnum=flat.expnum, camera=camera) for _, flat in flats.iterrows()]
    cflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch, kind="cpixflat", camera=camera)

    if skip_done and os.path.isfile(cflat_path):
        cflat = image_tasks.loadImage(cflat_path)
        return cflat, cflat_path
    else:
        cflat = image_tasks.combineImages([image_tasks.loadImage(dflat_path) for dflat_path in dflat_paths], method=comb_stat, replace_with_nan=False)
        cflat.writeFitsData(cflat_path)

    return cflat, cflat_path


def create_pixflats_60171(median_box=(31,31), skip_done=True):
    mjd = 60171
    flat_expnums = np.arange(3098, 3117+1)

    flats = md.get_frames_metadata(mjd=mjd, suffix="fits.gz", overwrite=False).query("expnum in @flat_expnums")

    calibs = drp.get_calib_paths(mjd=60171, version=drpver, longterm_cals=False)

    cameras = flats.camera.unique()
    flat_paths = dict.fromkeys(cameras)
    for camera in cameras:
        for expnum in flat_expnums:
            flat = flats.query("expnum == @expnum and camera == @camera").squeeze()

            rflat_path = path.full("lvm_raw", hemi="s", mjd=mjd, camspec=flat.camera, expnum=flat.expnum)
            pflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="p", imagetype="pixflat", expnum=flat.expnum, camera=flat.camera)
            dflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="pixflat", expnum=flat.expnum, camera=flat.camera)

            if skip_done and os.path.isfile(dflat_path):
                pass
            else:
                image_tasks.preproc_raw_frame(in_image=rflat_path, out_image=pflat_path, assume_imagetyp="pixflat")
                image_tasks.detrend_frame(in_image=pflat_path, out_image=dflat_path, in_bias=calibs["bias"][flat.camera], reject_cr=False, normalize_pixelflat=False)

        cflat, cflat_path = combine_pixelflats(mjds=mjd, mjd_epoch=mjd, camera=camera, flat_expnums=flat_expnums, median_box=median_box, skip_done=skip_done)

        mflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="m", imagetype="pixflat", expnum=f"{flats.expnum.min()}_{flats.expnum.max()}", camera=camera)
        fflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="f", imagetype="pixflat", expnum=f"{flats.expnum.min()}_{flats.expnum.max()}", camera=camera)

        cflat_median = fast_median_filter_2d(cflat._data, median_box)
        mflat = (cflat / cflat_median)
        mflat.writeFitsData(mflat_path)

        fflat = cflat / mflat
        fflat.writeFitsData(fflat_path)

        flat_paths[camera] = (cflat_path, mflat_path, fflat_path)

    return flat_paths


def test_pixflats(mjd, camera, flat_expnums, target_expnum):
    target_mjd = drp.mjd_from_expnum(target_expnum)[0]
    rframe_path = path.full("lvm_raw", hemi="s", mjd=target_mjd, camspec=camera, expnum=target_expnum)
    pframe_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=target_mjd, kind="p", imagetype="pixflat", expnum=target_expnum, camera=camera)
    dframe_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=target_mjd, kind="d", imagetype="pixflat", expnum=target_expnum, camera=camera)

    calibs = drp.get_calib_paths(mjd=mjd, from_sanbox=True)
    mflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="m", imagetype="pixflat", expnum=f"{flat_expnums.min()}_{flat_expnums.max()}", camera=camera)

    # TODO: detrend and extract frame
    # TODO: display CCD artifacts on extracted frame

    return dframe_path


def compare_pixflats(mjd, camera, flat_expnums_a, flat_expnums_b):
    dframe_a_path = test_pixflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums_a)
    dframe_b_path = test_pixflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums_b)

    # TODO: do some plots
    #   - From a selection of features in flats, compare the two


def filtering(image, size=31, min_flat=0.001, min_flat_masking=0.99, max_flat_masking=1.02, return_all=False):

    data = image._data
    error = image._error
    ivar = image.get_ivar()

    # initial model
    smooth_ini = fast_median_filter_2d(np.where(ivar > 0, data, np.NaN), size=(size, size))

    # initial flat by dividing by smooth image, masking only where we have no data
    flat_ini = data / smooth_ini
    flat_ini = np.where((ivar > 0) & (flat_ini > min_flat), flat_ini, 1.0)

    # dilate the mask, increasing sigma until not too large
    flat_error = error / smooth_ini
    for nsig in [3.,3.5,4.,5.,10.,20.]:
        low  = flat_ini < (min_flat_masking - nsig * flat_error)
        high = flat_ini > (max_flat_masking + nsig * flat_error)
        mask = low | high

        mask = ndi.binary_dilation(mask)
        frac = np.sum(mask > 0) / np.sum(ivar > 0)
        if frac < 0.05:
            log.info(f"Used nsig = {nsig}, frac = {frac:4.3f}")
            break

    # https://github.com/desihub/desispec/blob/main/bin/desi_compute_pixel_flatfield#L619

    # now start iterating smoothing and filtering the flat, ignoring newly masked pixels in the smoothing
    mask = mask | (ivar==0)
    smooth = fast_median_filter_2d(np.where(~mask, data, np.NaN), size=(size, size))

    # compute flat
    flat = np.where((ivar > 0) & (smooth > min_flat), data / smooth, 1.0)

    flat_img = copy(image)
    flat_img.setData(data=flat, error=flat_error, mask=mask)

    smooth = image / flat_img

    if return_all:
        return smooth, flat_img
    return smooth


def _desi_pixflat(cflat, size):
    filtered = cflat.apply_per_quadrant(filtering, size=size)
    mflat = cflat / filtered
    return cflat, mflat


def _simple_pixflat(cflat, size):
    cflat_median = fast_median_filter_2d(cflat._data, size)
    mflat = (cflat / cflat_median)
    return cflat, mflat


def get_pixflat(cflat_path, mpixflat_path, fflat_path, size=31, flatfield_threshold=0.001, method="desi"):
    if method not in ["desi", "simple"]:
        raise ValueError(f"Invalid value for `method`: {method}. Expected either 'desi' or 'simple'")

    log.info(f"loading flat frame from {cflat_path}")
    cflat = image_tasks.loadImage(cflat_path)

    log.info(f"filtering input flat using {method = } and box {size = }")
    if method == "desi":
        cflat, mflat = _desi_pixflat(cflat, size=size)
    elif method == "simple":
        cflat, mflat = _simple_pixflat(cflat, size=size)

    log.info(f"replacing invalid values and flatfield values below {flatfield_threshold} with 1.0")
    mflat._data = np.where((mflat._data > flatfield_threshold) & np.isfinite(mflat._data), mflat._data, 1.0)
    log.info(f"writing master pixelflat to {mpixflat_path}")
    mflat.writeFitsData(mpixflat_path)

    log.info(f"writing flatfielded flat to {fflat_path}")
    fflat = cflat / mflat
    fflat.writeFitsData(fflat_path)

    return cflat, mflat, fflat


def create_pixflats(mjds, mjd_epoch, camera, sequence, size=31, flatfield_threshold=0.01, method="desi", skip_done=True):
    """
    Creates pixel flat-field calibration files for a given camera and set of MJDs.

    Parameters
    ----------
    mjds : list or array-like
        List of Modified Julian Dates (MJDs) to process.
    mjd_epoch : int
        MJD for the pixel flat epoch. All master pixel flats will be stored in the corresponding directory.
    camera : str
        Identifier for the camera (e.g., 'r1', 'b2').
    sequence : dict
        Dictionary containing exposure sequences with keys such as "flat", "dark",
        and "bias", and their corresponding exposure numbers.
    size : int, optional
        Size of the smoothing kernel for flat-field correction. Default is 31.
    flatfield_threshold : float, optional
        Threshold for flat-field correction. Default is 0.01.
    method : str, optional
        Method to use for flat-field correction. Default is "desi".
    skip_done : bool, optional
        If True, skip processing for already completed files. Default is True.

    Returns
    -------
    tuple
        Paths to the created calibration files:
        - cflat_path (str): Path to the combined pixel flat file.
        - mflat_path (str): Path to the master pixel flat file.
        - fflat_path (str): Path to the flat-fielded combined pixel flat file.

    Notes
    -----
    - The function first parses the sequence to extract flat, dark, and bias exposure numbers.
    - Metadata for the exposures is retrieved and filtered based on the camera and flat exposures.
    - If no matching frames are found, the function logs an error and exits.
    - The function performs detrending, combines pixel flats, and generates the final flat-field files.
    """
    parsed_sequence = _parse_sequence(sequence=sequence)
    expnums_dict = _expand_sequence(parsed_sequence)
    flat_expnums = expnums_dict.get("flat")
    dark_expnums = expnums_dict.get("dark", [])
    bias_expnums = expnums_dict.get("bias", [])

    if flat_expnums is None:
        raise ValueError(f"No pixel flat exposures found for {camera = } with sequence: {sequence}")

    frames = get_enights_metadata(mjds=mjds)
    frames = frames.query("expnum in @flat_expnums and camera == @camera")

    if frames.empty:
        log.error(f"No pixel flat frames found for {camera = } and {mjds = }")
        return

    detrend_pixelflats(mjds=mjds, camera=camera, flat_expnums=flat_expnums, dark_expnums=dark_expnums, bias_expnums=bias_expnums, skip_done=skip_done)
    _, cflat_path = combine_pixelflats(mjds=mjds, mjd_epoch=mjd_epoch, camera=camera, flat_expnums=flat_expnums, skip_done=skip_done)

    mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch, kind="mpixflat", camera=camera)
    fflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch, kind="fpixflat", camera=camera)

    get_pixflat(cflat_path, mflat_path, fflat_path, size=size, flatfield_threshold=flatfield_threshold, method=method)

    return cflat_path, mflat_path, fflat_path

