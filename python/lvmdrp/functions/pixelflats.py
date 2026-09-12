import os
from shutil import copy2
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy as copy
from pprint import pformat

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.external.fast_median import fast_median_filter_2d
from lvmdrp.utils import metadata as md
from lvmdrp import main as drp
from scipy import ndimage as ndi


PIXFLAT_EPOCHS_PATH = os.path.join(os.getenv("LVMCORE_DIR"), "calibrations", "pixflat-epochs.yaml")


def _parse_expnums(expnums):
    """Parse individual exposure numbers and half-open ranges.

    Parameters
    ----------
    expnums : iterable
        Exposure numbers as integers or comma-separated ``start,stop`` ranges.

    Returns
    -------
    numpy.ndarray
        Sorted exposure numbers with integer dtype.

    Raises
    ------
    TypeError
        If an item is neither an integer nor a comma-separated range.
    """
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
        A dictionary mapping exposure types to grouped exposure numbers.

    Raises
    ------
    KeyError
        If ``kind`` contains an unsupported exposure type.
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


def _parse_sequence(sequence, expand=True):
    """Copy and parse the exposure lists in a pixel-flat sequence.

    Parameters
    ----------
    sequence : dict
        Sequence definition containing ``expnums`` and optionally ``rejects``.

    Returns
    -------
    dict
        A copied sequence with parsed NumPy arrays for both exposure lists.
    """
    parsed_sequence = copy(sequence)
    expnums = _parse_expnums(sequence.get("expnums", []) or [])
    rejects = _parse_expnums(sequence.get("rejects", []) or [])
    parsed_sequence["expnums"] = expnums
    parsed_sequence["rejects"] = rejects

    if expand:
        return _expand_sequence(parsed_sequence)
    return parsed_sequence


def get_shifted_rejects(shifted, mjd_epoch, camera, epochs=None):
    """Return rejects needed for shifted exposures in one pixel-flat sequence.

    Existing rejects are applied before grouping, matching ``_expand_sequence``.
    This preserves intentional exposure-time substitutions while ensuring that
    every newly rejected exposure removes a complete effective sequence group.
    """
    epochs = epochs or load_pixflat_epochs(verbose=False)
    try:
        sequence = epochs[mjd_epoch]["sequences"][camera]
    except KeyError as error:
        raise KeyError(f"No pixelflat sequence found for {mjd_epoch = } and {camera = }") from error

    kind = sequence.get("kind", "")
    if not kind or len(kind) % 2:
        raise ValueError(f"Invalid sequence kind for {mjd_epoch = }, {camera = }: {kind!r}")

    group_size = 0
    for count, exposure_type in zip(kind[::2], kind[1::2]):
        if exposure_type not in {"f", "b", "d"} or not count.isdigit() or int(count) < 1:
            raise ValueError(f"Invalid sequence kind for {mjd_epoch = }, {camera = }: {kind!r}")
        group_size += int(count)

    expnums = _parse_expnums(sequence.get("expnums", []) or [])
    existing_rejects = set(_parse_expnums(sequence.get("rejects", []) or []))
    effective_expnums = np.asarray([expnum for expnum in expnums if expnum not in existing_rejects])
    log.info(
        f"shifted-exposure sequence: {mjd_epoch = }, {camera = }, {kind = }, "
        f"{group_size = }, raw_count = {expnums.size}, "
        f"raw_range = {(int(expnums[0]), int(expnums[-1])) if expnums.size else None}, "
        f"existing_rejects = {sorted(existing_rejects)}, "
        f"effective_count = {effective_expnums.size}"
    )
    if effective_expnums.size % group_size:
        raise ValueError(
            f"Sequence for {mjd_epoch = }, {camera = } has {effective_expnums.size} "
            f"effective exposures, which is not divisible by {group_size}"
        )

    groups = {
        int(expnum): effective_expnums[group_start:group_start + group_size].tolist()
        for group_start in range(0, effective_expnums.size, group_size)
        for expnum in effective_expnums[group_start:group_start + group_size]
    }
    shifted_expnums = [
        int(expnum)
        for epoch, expnum in shifted.get(camera, [])
        if epoch == mjd_epoch
    ]
    log.info(
        f"shifted exposures selected: {mjd_epoch = }, {camera = }, "
        f"shifted_expnums = {sorted(set(shifted_expnums))}, "
        f"shifted_count = {len(shifted_expnums)}"
    )
    missing = sorted(set(shifted_expnums).difference(groups))
    if missing:
        raise ValueError(
            f"Shifted exposures are not in the effective sequence for "
            f"{mjd_epoch = }, {camera = }: {missing}"
        )

    rejects = set(existing_rejects)
    for expnum in shifted_expnums:
        rejects.update(groups[expnum])
        log.info(
            f"rejecting sequence group: {mjd_epoch = }, {camera = }, "
            f"shifted_expnum = {expnum}, group = {groups[expnum]}"
        )
    log.info(
        f"shifted-exposure rejects: {mjd_epoch = }, {camera = }, "
        f"new_rejects = {sorted(rejects.difference(existing_rejects))}, "
        f"all_rejects = {sorted(rejects)}, total = {len(rejects)}"
    )
    return sorted(rejects)


def set_shifted_rejects(rejects, epochs, mjd_epoch, camera):
    try:
        sequence = epochs[mjd_epoch]["sequences"][camera]
    except KeyError as error:
        raise KeyError(f"No pixelflat sequence found for {mjd_epoch = } and {camera = }") from error

    _rejects = sequence.get("rejects", []) or []
    log.info(f"existing rejects: {_rejects}")
    log.info(f"adding new rejects: {rejects}")
    _rejects.extend(rejects)
    _rejects = sorted(set(_rejects))
    log.info(f"final rejects: {_rejects}")

    sequence["rejects"] = _rejects
    return sequence


def validate_sequence_kind(epochs, mjd_epoch, camera):
    try:
        mjds = epochs[mjd_epoch]["sources"]
    except KeyError as error:
        raise KeyError(f"No pixelflat sequence found for {mjd_epoch = }") from error
    try:
        sequence = epochs[mjd_epoch]["sequences"][camera]
    except KeyError as error:
        raise KeyError(f"No pixelflat sequence found for {mjd_epoch = } and {camera = }") from error

    kind = sequence.get("kind", "")
    if not kind or len(kind) % 2:
        raise ValueError(f"Invalid sequence kind for {mjd_epoch = }, {camera = }: {kind!r}")

    group_size = 0
    for count, exposure_type in zip(kind[::2], kind[1::2]):
        if exposure_type not in {"f", "b", "d"} or not count.isdigit() or int(count) < 1:
            raise ValueError(f"Invalid sequence kind for {mjd_epoch = }, {camera = }: {kind!r}")
        group_size += int(count)

    expnums = _parse_sequence(sequence, expand=False)["expnums"]
    existing_rejects = set(_parse_expnums(sequence.get("rejects", []) or []))
    effective_expnums = np.asarray([expnum for expnum in expnums if expnum not in existing_rejects])
    log.info(
        f"shifted-exposure sequence: {mjd_epoch = }, {camera = }, {kind = }, "
        f"{group_size = }, raw_count = {expnums.size}, "
        f"raw_range = {(int(expnums[0]), int(expnums[-1])) if expnums.size else None}, "
        f"existing_rejects = {sorted(existing_rejects)}, "
        f"effective_count = {effective_expnums.size}"
    )

    sequence = _parse_sequence(sequence, expand=True)
    flat_expnums = sequence.get("flat", [])
    bias_expnums = sequence.get("bias", [])
    dark_expnums = sequence.get("dark", [])
    log.info(
        f"{flat_expnums = }, "
        f"{bias_expnums = }, "
        f"{dark_expnums = }, "
    )
    frames = get_enights_metadata(mjds)
    frame_types = {
        "flat": frames.query("expnum in @flat_expnums"),
        "bias": frames.query("expnum in @bias_expnums"),
        "dark": frames.query("expnum in @dark_expnums"),
    }
    TYPE_MAPS = {
        "bias": ["bias"],
        "dark": ["dark"],
        "flat": ["object", "flat"]
    }

    for _type, frame in frame_types.items():
        valid = frame.imagetyp.isin(TYPE_MAPS.get(_type, []) or [])
        if not valid.all():
            invalid = frame.loc[~valid]
            log.warning(f"\n{invalid.to_string()}")
        else:
            log.info(f"all valid '{_type}' exposures: {set(frame.expnum)}")


def rsync_enight(mjds):
    """Placeholder for synchronizing engineering nights from LCO.

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


def load_pixflat_epochs(epochs_path=None, filter_by_mjds=None, filter_by_cameras=None, verbose=True):
    """Load pixel-flat epoch definitions from a YAML file.

    Parameters
    ----------
    epochs_path : str or pathlib.Path, optional
        Path to the pixel-flat epochs file.
    filter_by_mjds : list or tuple, optional
        MJDs to keep from the loaded epoch mapping.
    filter_by_cameras : list or tuple, optional
        Camera sequences to keep within each epoch.
    verbose : bool, optional
        If True, log the loaded and filtered epoch information.
    """
    epochs_path = epochs_path or PIXFLAT_EPOCHS_PATH
    with open(epochs_path) as f:
        epochs = yaml.safe_load(f)["epochs"]

    if verbose:
        log.info(f"loaded {len(epochs)} epochs:")
        for mjd in epochs:
            log.info(f"  {mjd}: {pformat(epochs[mjd])}")

    if filter_by_mjds is not None and isinstance(filter_by_mjds, (list, tuple)):
        if verbose:
            log.info(f"filtering by {filter_by_mjds}")
        epochs = {mjd: epochs[mjd] for mjd in filter_by_mjds if mjd in epochs}
        if len(epochs) == 0:
            log.error(f"epoch(s) {filter_by_mjds} not found in calibration epochs file: '{epochs_path}'")
            return epochs
        if verbose:
            log.info(f"after filtering {len(epochs)} epoch(s):")
            for mjd in epochs:
                log.info(f"  {mjd}: {epochs[mjd]}")

    if filter_by_cameras is not None and isinstance(filter_by_cameras, (list, tuple)):
        if verbose:
            log.info(f"filtering by cameras {filter_by_cameras}")
        epochs = {
            mjd: {
                **epoch,
                "sequences": {
                    camera: sequence
                    for camera, sequence in epoch.get("sequences", {}).items()
                    if camera in filter_by_cameras
                },
            }
            for mjd, epoch in epochs.items()
        }
    return epochs


def detrend_pixelflats(mjds, camera, flat_expnums, bias_expnums=[], dark_expnums=[], use_pixmask=True, skip_done=True):
    """Preprocess and detrend pixel-flat, bias, and dark exposures.

    Parameters
    ----------
    mjds : int or array-like
        Engineering-night MJDs containing the exposures.
    camera : str
        Camera identifier to process.
    flat_expnums : array-like
        Pixel-flat exposure numbers.
    bias_expnums, dark_expnums : array-like, optional
        Bias and dark exposure numbers used for detrending.
    use_pixmask : bool, optional
        Whether to use the current pixel mask during preprocessing.
    skip_done : bool, optional
        Whether to skip products that already exist.

    Returns
    -------
    list[str]
        Paths to the available detrended pixel-flat images.
    """

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
    """Combine detrended pixel flats into an epoch-level flat.

    Parameters
    ----------
    mjds : int or array-like
        Engineering-night MJDs containing the exposures.
    mjd_epoch : int
        MJD used to identify the output calibration epoch.
    camera : str
        Camera identifier to process.
    flat_expnums : array-like
        Pixel-flat exposure numbers to combine.
    comb_stat : str, optional
        Combination statistic passed to ``combineImages``.
    skip_done : bool, optional
        Whether to reuse an existing combined flat.

    Returns
    -------
    tuple
        Combined image object and its output path.
    """
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
    """Create pixel-flat products for the special MJD 60171 sequence.

    Parameters
    ----------
    median_box : tuple, optional
        Two-dimensional smoothing-kernel size for the master flat.
    skip_done : bool, optional
        Whether to skip products that already exist.

    Returns
    -------
    dict
        Mapping of camera identifiers to output product paths.
    """
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
    """Construct the expected detrended path for a test pixel flat.

    Parameters
    ----------
    mjd : int
        MJD associated with the calibration products.
    camera : str
        Camera identifier to test.
    flat_expnums : array-like
        Exposure numbers defining the master flat range.
    target_expnum : int
        Target exposure number.

    Returns
    -------
    str
        Path to the target detrended pixel-flat image.
    """
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
    """Prepare two pixel-flat products for comparison.

    Parameters
    ----------
    mjd : int
        MJD associated with the calibration products.
    camera : str
        Camera identifier to compare.
    flat_expnums_a, flat_expnums_b : array-like
        Exposure-number groups defining the two flats.
    """
    dframe_a_path = test_pixflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums_a)
    dframe_b_path = test_pixflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums_b)

    # TODO: do some plots
    #   - From a selection of features in flats, compare the two


def filtering(image, size=31, min_flat=0.001, min_flat_masking=0.99, max_flat_masking=1.02, return_all=False):
    """Filter a combined flat and identify invalid or deviant pixels.

    Parameters
    ----------
    image : image object
        Image containing data, errors, inverse variance, and mask information.
    size : int, optional
        Median-filter size.
    min_flat : float, optional
        Minimum accepted flat value.
    min_flat_masking, max_flat_masking : float, optional
        Lower and upper limits used when growing the bad-pixel mask.
    return_all : bool, optional
        If True, return both the smooth image and normalized flat image.

    Returns
    -------
    image object or tuple
        Smooth image, or ``(smooth_image, flat_image)`` when ``return_all`` is
        True.
    """

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
    """Create a DESI-style pixel flat using quadrant-wise filtering.

    Parameters
    ----------
    cflat : image object
        Combined pixel-flat image.
    size : int
        Median-filter size.

    Returns
    -------
    tuple
        The input combined flat and its normalized master flat.
    """
    filtered = cflat.apply_per_quadrant(filtering, size=size)
    mflat = cflat / filtered
    return cflat, mflat


def _simple_pixflat(cflat, size):
    """Create a pixel flat by dividing by a global median-filtered image.

    Parameters
    ----------
    cflat : image object
        Combined pixel-flat image.
    size : int
        Median-filter size.

    Returns
    -------
    tuple
        The input combined flat and its normalized master flat.
    """
    cflat_median = fast_median_filter_2d(cflat._data, size)
    mflat = (cflat / cflat_median)
    return cflat, mflat


def get_pixflat(cflat_path, mpixflat_path, fflat_path, size=31, min_flatfield=0.001, method="desi"):
    """Generate and write master and flat-fielded pixel-flat products.

    Parameters
    ----------
    cflat_path : str
        Path to the combined pixel-flat image.
    mpixflat_path : str
        Output path for the normalized master pixel flat.
    fflat_path : str
        Output path for the flat-fielded combined image.
    size : int, optional
        Median-filter size.
    min_flatfield : float, optional
        Minimum valid flat field value.
    method : {"desi", "simple"}, optional
        Pixel-flat construction method.

    Returns
    -------
    tuple
        Combined image, master pixel flat, and flat-fielded image.
    """
    if method not in ["desi", "simple"]:
        raise ValueError(f"Invalid value for `method`: {method}. Expected either 'desi' or 'simple'")

    log.info(f"loading flat frame from {cflat_path}")
    cflat = image_tasks.loadImage(cflat_path)

    log.info(f"filtering input flat using {method = } and box {size = }")
    if method == "desi":
        cflat, mflat = _desi_pixflat(cflat, size=size)
    elif method == "simple":
        cflat, mflat = _simple_pixflat(cflat, size=size)

    log.info(f"replacing invalid values and flatfield values below {min_flatfield} with 1.0")
    mflat._data = np.where((mflat._data > min_flatfield) & np.isfinite(mflat._data), mflat._data, 1.0)
    log.info(f"writing master pixelflat to {mpixflat_path}")
    mflat.writeFitsData(mpixflat_path)

    log.info(f"writing flatfielded flat to {fflat_path}")
    fflat = cflat / mflat
    fflat.writeFitsData(fflat_path)

    return cflat, mflat, fflat


def create_pixflats(mjds, mjd_epoch, camera, sequence, size=31, min_flatfield=0.01, method="desi", skip_done=True, dry_run=False):
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
    min_flatfield : float, optional
        Minimum valid flat field value. Default is 0.01.
    method : str, optional
        Method to use for flat-field correction. Default is "desi".
    skip_done : bool, optional
        If True, skip processing for already completed files. Default is True.
    dry_run : bool, optional
        If True, log the selected inputs and output paths without creating files.

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
    flat_expnums = parsed_sequence.get("flat")
    dark_expnums = parsed_sequence.get("dark", [])
    bias_expnums = parsed_sequence.get("bias", [])

    if flat_expnums is None:
        raise ValueError(f"No pixel flat exposures found for {camera = } with sequence: {sequence}")

    cflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch, kind="cpixflat", camera=camera)
    mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch, kind="mpixflat", camera=camera)
    fflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch, kind="fpixflat", camera=camera)

    if dry_run:
        log.info(f"dry run of '{create_pixflats.__name__}' for {camera = } and {mjd_epoch = }")
        log.info(f"  source MJDs: {mjds}")
        log.info(f"  flat exposures: {flat_expnums}")
        log.info(f"  dark exposures: {dark_expnums}")
        log.info(f"  bias exposures: {bias_expnums}")
        log.info("  output paths:")
        for output_path in (cflat_path, mflat_path, fflat_path):
            log.info(f"    {output_path}")
        return cflat_path, mflat_path, fflat_path

    detrend_pixelflats(mjds=mjds, camera=camera, flat_expnums=flat_expnums, dark_expnums=dark_expnums, bias_expnums=bias_expnums, skip_done=skip_done)
    combine_pixelflats(mjds=mjds, mjd_epoch=mjd_epoch, camera=camera, flat_expnums=flat_expnums, skip_done=skip_done)
    get_pixflat(cflat_path, mflat_path, fflat_path, size=size, min_flatfield=min_flatfield, method=method)

    return cflat_path, mflat_path, fflat_path


def create_super_pixflats(pixflat_epochs, mjd_epoch, size=31, min_flatfield=0.01, method="desi", skip_done=True, dry_run=False):
    """Create super pixel-flat products for all cameras in a set of epochs.

    Parameters
    ----------
    pixflat_epochs : dict
        Epoch mapping returned by :func:`load_pixflat_epochs`. Camera
        sequences are expected to have already been filtered as needed.
    mjd_epoch : int
        MJD used to identify the output super pixel-flat products.
    size : int, optional
        Size of the smoothing kernel used to create the master pixel flat.
        Default is 31.
    min_flatfield : float, optional
        Minimum valid flat-field value. Values below this threshold are set to
        1.0. Default is 0.01.
    method : {"desi", "simple"}, optional
        Method used to create the master pixel flat. Default is "desi".
    skip_done : bool, optional
        Whether to reuse existing detrended and combined products. Default is
        True.
    dry_run : bool, optional
        If True, log the selected inputs and output paths without creating
        files. Default is False.

    Returns
    -------
    dict
        Mapping of camera identifiers to tuples containing the paths of the
        combined, master, and flat-fielded pixel-flat products, respectively.

    Notes
    -----
    Each source epoch is detrended using its own camera sequence. The
    resulting detrended flats are then combined independently for each camera,
    allowing sequence shapes to differ between epochs.
    """
    camera_inputs = {}

    for pixflat_epoch in pixflat_epochs.values():
        mjds = pixflat_epoch.get("sources", [])
        for camera, sequence in pixflat_epoch.get("sequences", {}).items():
            parsed_sequence = _parse_sequence(sequence=sequence)
            flat_expnums = parsed_sequence.get("flat")
            dark_expnums = parsed_sequence.get("dark", [])
            bias_expnums = parsed_sequence.get("bias", [])

            if not dry_run:
                detrend_pixelflats(
                    mjds=mjds,
                    camera=camera,
                    flat_expnums=flat_expnums,
                    dark_expnums=dark_expnums,
                    bias_expnums=bias_expnums,
                    skip_done=skip_done,
                )

            camera_input = camera_inputs.setdefault(camera, {"mjds": [], "flat_expnums": []})
            camera_input["mjds"].extend(np.atleast_1d(mjds).tolist())
            camera_input["flat_expnums"].extend(np.atleast_1d(flat_expnums).tolist())

    output_paths = {}
    for camera, camera_input in camera_inputs.items():
        cflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch, kind="cpixflat", camera=camera)
        mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch, kind="mpixflat", camera=camera)
        fflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch, kind="fpixflat", camera=camera)
        output_paths[camera] = (cflat_path, mflat_path, fflat_path)

        if dry_run:
            log.info(f"dry run of '{create_super_pixflats.__name__}' for {camera = } and {mjd_epoch = }")
            log.info(f"  source MJDs: {camera_input['mjds']}")
            log.info(f"  flat exposures: {camera_input['flat_expnums']}")
            log.info("  output paths:")
            for output_path in output_paths[camera]:
                log.info(f"    {output_path}")
            continue

        combine_pixelflats(
            mjds=camera_input["mjds"],
            mjd_epoch=mjd_epoch,
            camera=camera,
            flat_expnums=camera_input["flat_expnums"],
            skip_done=skip_done,
        )
        get_pixflat(cflat_path, mflat_path, fflat_path, size=size, min_flatfield=min_flatfield, method=method)

    return output_paths


def tag_pixelflats(epoch_mjd, version=drpver, dry_run=False):
    """Copy master pixel flats for an epoch into the sandbox calibration directory.

    Parameters
    ----------
    epoch_mjd : int
        MJD identifying the source master pixel-flat epoch.
    version : str, optional
        Reduction version used to locate the source products. Default is the
        current pipeline version.
    dry_run : bool, optional
        Log source and destination paths without copying files. Default is
        False.

    Returns
    -------
    dict
        Mapping of camera names to source and destination paths.

    Notes
    -----
    Existing sandbox products are replaced by the source product, matching the
    behavior of :func:`tag_longterm_calibrations`.
    """
    source_paths = sorted(
        path.expand(
            "lvm_master",
            drpver=version,
            tileid=11111,
            mjd=epoch_mjd,
            kind="mpixflat",
            camera="*",
        )
    )

    copied_paths = {}
    for source_path in source_paths:
        camera = os.path.basename(source_path).split(".")[0].split("-")[-1]
        destination_path = path.full("lvm_calib", mjd="pixelmasks", kind="pixflat", camera=camera)
        copied_paths[camera] = {"source": source_path, "destination": destination_path}

        destination_exists = os.path.isfile(destination_path)
        if dry_run:
            if not os.path.isfile(source_path):
                log.error(f"source master pixel flat does not exist: {source_path}")
                continue

            source_mtime = datetime.fromtimestamp(os.path.getmtime(source_path))
            destination_mtime = datetime.fromtimestamp(os.path.getmtime(destination_path)) if destination_exists else None
            log.info(f"source/destination for pixel flat, {camera = }:")
            log.info(f"   {source_mtime.strftime('%a %d %b %Y, %I:%M:%S%p')} {source_path}")
            log.info(f"   {destination_mtime.strftime('%a %d %b %Y, %I:%M:%S%p') if destination_mtime else None} {destination_path}")
            if destination_mtime is None:
                log.info("   - source will create a new path on destination")
            elif source_mtime > destination_mtime:
                log.info("   > source is newer than destination")
            elif source_mtime < destination_mtime:
                log.warning("   < source is older than destination")
            else:
                log.info("   = source and destination have the same modification time")
            continue

        if not os.path.isfile(source_path):
            log.error(f"source master pixel flat does not exist: {source_path}")
            continue

        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            copy2(source_path, destination_path)
            log.info(f"copied {source_path} into {destination_path}")
        except PermissionError as error:
            log.error(f"error while copying {source_path}: {error}")

    return copied_paths

