import os
import re
from shutil import copy2
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy as copy
from pprint import pformat

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative

from lvmdrp.core.constants import CAMERAS
from lvmdrp import log, path, __version__ as drpver
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.external.fast_median import fast_median_filter_2d
from lvmdrp.utils import metadata as md
from lvmdrp import main as drp
from scipy import ndimage as ndi


PIXFLAT_EPOCHS_PATH = os.path.join(os.getenv("LVMCORE_DIR"), "calibrations", "pixflat-epochs.yaml")

# Size bins (in pixels, lower-inclusive/upper-exclusive) used to group connected low-value regions
# when looking for pixel-flat artifacts
ARTIFACT_BINS = ((10, 20), (20, 30), (30, 40), (40, 3000))


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
        if isinstance(expnum, (int, np.int64)):
            parsed_expnums.append([expnum])
        elif isinstance(expnum, str) and "," in expnum:
            expnum_range = tuple(int(i) for i in expnum.split(","))
            parsed_expnums.append(np.arange(*expnum_range))
        else:
            raise TypeError(f"Invalid type in {expnums = } at {idx}: {type(expnum)}")
    parsed_expnums = np.concatenate(parsed_expnums)
    parsed_expnums.sort()
    return parsed_expnums.astype("int")


_KIND_PATTERN = re.compile(r"(\d+)([fbd])")


def _parse_kind(kind):
    """Parse a sequence ``kind`` string into ordered (count, type) pairs.

    Unlike naive character-pair slicing (``kind[::2]``/``kind[1::2]``), this
    correctly handles multi-digit counts, e.g. ``'10f2d'`` -> ``[(10, 'f'), (2, 'd')]``.

    Parameters
    ----------
    kind : str
        Sequence kind string, e.g. ``'2f1d'``.

    Returns
    -------
    list[tuple[int, str]]
        Ordered ``(count, type)`` pairs, where ``type`` is one of ``'f'``,
        ``'b'``, ``'d'``.

    Raises
    ------
    ValueError
        If ``kind`` is empty, contains characters other than digit+type
        pairs, or any count is less than 1.
    """
    if not kind:
        raise ValueError(f"Invalid sequence kind: {kind!r}")
    pairs = _KIND_PATTERN.findall(kind)
    reconstructed = "".join(f"{count}{typ}" for count, typ in pairs)
    if reconstructed != kind or any(int(count) < 1 for count, _ in pairs):
        raise ValueError(f"Invalid sequence kind: {kind!r}")
    return [(int(count), typ) for count, typ in pairs]


def _compress_expnums(expnums):
    """Compress exposure numbers into ints and half-open ``'start, end'`` strings.

    This is the inverse of :func:`_parse_expnums`: consecutive runs of two or
    more exposure numbers (step of 1) are collapsed into a single
    ``'start, end'`` string, where ``end`` is exclusive -- i.e. one past the
    last value in the run -- matching how :func:`_parse_expnums` expands such
    strings via ``numpy.arange``. Isolated exposure numbers are left as plain
    integers. This mirrors the compact range notation already used for
    ``expnums`` entries in the pixel-flat epochs file, so that other fields
    (e.g. ``rejects``) can follow the same convention.

    Parameters
    ----------
    expnums : Iterable[int]
        Exposure numbers, in any order and with possible duplicates.

    Returns
    -------
    list[int | str]
        Sorted mix of individual integers and half-open range strings,
        suitable for storing directly in a pixel-flat epochs YAML file.

    Examples
    --------
    >>> _compress_expnums([36284])
    [36284]
    >>> _compress_expnums([36280, 36281, 36282])
    ['36280, 36283']
    >>> _compress_expnums([1, 2, 5, 6, 7, 10])
    ['1, 3', '5, 8', 10]
    """
    expnums = sorted({int(expnum) for expnum in expnums})
    if not expnums:
        return []

    compressed = []
    run_start = run_end = expnums[0]
    for expnum in expnums[1:]:
        if expnum == run_end + 1:
            run_end = expnum
            continue
        compressed.append(run_start if run_start == run_end else f"{run_start}, {run_end + 1}")
        run_start = run_end = expnum
    compressed.append(run_start if run_start == run_end else f"{run_start}, {run_end + 1}")
    return compressed


def _expand_sequence(sequence, repeat=False):
    """Expand a sequence dictionary to extract exposure numbers grouped by type.

    The function interprets the ``"kind"`` key in the input dictionary to
    determine the types of exposures (e.g., flat, bias, dark) and their
    respective counts, via :func:`_parse_kind`. It then uses the
    ``"expnums"`` key to group the exposure numbers accordingly.

    Parameters
    ----------
    sequence : dict
        A dictionary containing the following keys:

        - ``"kind"`` : str
            A string of ``(count, type)`` pairs, e.g. ``'2f1d'`` for two
            flats followed by one dark (``'f'`` for flat, ``'b'`` for bias,
            ``'d'`` for dark). See :func:`_parse_kind`.
        - ``"expnums"`` : list
            A list of exposure numbers.
        - ``"rejects"`` : list, optional
            Exposure numbers to exclude before grouping.
    repeat : bool, optional
        Whether to pad bias/dark sequences shorter than the flat sequence,
        by default False.

    Returns
    -------
    dict
        A dictionary mapping exposure types (``"flat"``, ``"bias"``,
        ``"dark"``) to their grouped exposure numbers.

    Raises
    ------
    ValueError
        If ``sequence["kind"]`` is malformed (see :func:`_parse_kind`), or
        if the number of effective exposures is not evenly divisible by the
        group size implied by ``kind``.
    """
    typ_maps = {"f": "flat", "b": "bias", "d": "dark"}

    kind = sequence.get("kind")
    pairs = _parse_kind(kind)
    typs = [typ for _, typ in pairs]
    nums = {typ_maps[typ]: count for count, typ in pairs}
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
        Sequence definition containing ``expnums`` and optionally ``rejects``,
        plus (when ``expand`` is True) a ``kind`` string as expected by
        :func:`_expand_sequence`.
    expand : bool, optional
        If True (default), group the parsed exposure numbers by type using
        :func:`_expand_sequence`. If False, return the sequence with
        ``expnums``/``rejects`` parsed but otherwise unexpanded.

    Returns
    -------
    dict
        If ``expand`` is True, a dictionary mapping exposure types to grouped
        exposure numbers (see :func:`_expand_sequence`). Otherwise, a copy of
        ``sequence`` with ``expnums`` and ``rejects`` replaced by parsed
        NumPy arrays.

    Raises
    ------
    TypeError
        If ``expnums`` or ``rejects`` contain an item that is neither an
        integer nor a comma-separated range (see :func:`_parse_expnums`).
    ValueError
        If ``expand`` is True and ``sequence["kind"]`` is malformed (see
        :func:`_expand_sequence`).
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

    Parameters
    ----------
    shifted : dict[str, list[tuple[int, int]]]
        Mapping of camera identifier to a list of ``(mjd_epoch, expnum)``
        pairs identifying exposures whose timing was shifted and whose
        containing sequence group should therefore be rejected.
    mjd_epoch : int
        MJD identifying the calibration epoch to inspect.
    camera : str
        Camera identifier whose sequence should be inspected.
    epochs : dict, optional
        Epoch mapping as returned by :func:`load_pixflat_epochs`. If not
        given, it is loaded with ``verbose=False``.

    Returns
    -------
    list[int]
        Sorted union of the sequence's existing rejects and the exposure
        numbers belonging to any group containing a shifted exposure.

    Raises
    ------
    KeyError
        If no sequence exists for ``mjd_epoch`` and ``camera``.
    ValueError
        If the sequence's ``kind`` is malformed (see :func:`_parse_kind`),
        if the number of effective (non-rejected) exposures is not evenly
        divisible by the group size implied by ``kind``, or if any shifted
        exposure in ``shifted`` is not part of the effective sequence.
    """
    epochs = epochs or load_pixflat_epochs(verbose=False)
    try:
        sequence = epochs[mjd_epoch]["sequences"][camera]
    except KeyError as error:
        raise KeyError(f"No pixelflat sequence found for {mjd_epoch = } and {camera = }") from error

    kind = sequence.get("kind", "")
    try:
        kind_pairs = _parse_kind(kind)
    except ValueError as error:
        raise ValueError(f"Invalid sequence kind for {mjd_epoch = }, {camera = }: {kind!r}") from error
    group_size = sum(count for count, _ in kind_pairs)

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
    """Merge new rejects into an epoch's sequence, in place.

    Both the sequence's existing ``"rejects"`` entry and ``rejects`` are
    expanded to individual exposure numbers via :func:`_parse_expnums`
    (so either may freely mix plain integers and half-open range strings),
    unioned, and re-compressed with :func:`_compress_expnums` before being
    stored back -- keeping ``"rejects"`` in the same compact-range
    convention used for ``expnums``.

    Parameters
    ----------
    rejects : Iterable[int or str]
        Exposure numbers to merge into the sequence's existing rejects, e.g.
        as returned by :func:`get_shifted_rejects`. Individual integers
        and/or comma-separated ``"start,stop"`` range strings are accepted,
        as for :func:`_parse_expnums`.
    epochs : dict
        Epoch mapping as returned by :func:`load_pixflat_epochs`. The
        targeted sequence's ``"rejects"`` entry is updated in place.
    mjd_epoch : int
        MJD identifying the calibration epoch to update.
    camera : str
        Camera identifier whose sequence should be updated.

    Returns
    -------
    dict
        The updated sequence dictionary (also mutated in place within
        ``epochs``).

    Raises
    ------
    KeyError
        If no sequence exists for ``mjd_epoch`` and ``camera``.
    TypeError
        If ``rejects`` or the sequence's existing ``"rejects"`` contain an
        item that is neither an integer nor a comma-separated range (see
        :func:`_parse_expnums`).
    """
    try:
        sequence = epochs[mjd_epoch]["sequences"][camera]
    except KeyError as error:
        raise KeyError(f"No pixelflat sequence found for {mjd_epoch = } and {camera = }") from error

    existing_rejects = _parse_expnums(sequence.get("rejects", []) or [])
    new_rejects = _parse_expnums(rejects)
    log.info(f"existing rejects: {sorted(existing_rejects.tolist())}")
    log.info(f"adding new rejects: {sorted(new_rejects.tolist())}")
    _rejects = _compress_expnums(np.union1d(existing_rejects, new_rejects).tolist())
    log.info(f"final rejects: {_rejects}")

    sequence["rejects"] = _rejects or None
    return sequence


def validate_sequence_kind(epochs, mjd_epoch, camera):
    """Validate a sequence's ``kind`` against its exposures' metadata.

    Checks that the sequence's ``kind`` string is well-formed, that the
    effective (non-rejected) exposure count is evenly divisible by the group
    size it implies, and logs a warning for any exposure whose ``imagetyp``
    metadata does not match its expected role (flat, bias, or dark) in the
    sequence.

    Parameters
    ----------
    epochs : dict
        Epoch mapping as returned by :func:`load_pixflat_epochs`.
    mjd_epoch : int
        MJD identifying the calibration epoch to validate.
    camera : str
        Camera identifier whose sequence should be validated.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping of exposure type (``"flat"``, ``"bias"``, ``"dark"``) to the
        metadata rows whose ``imagetyp`` doesn't match that role. Empty
        frames mean every exposure of that type is valid. Mismatches are also
        reported via ``log.warning``.

    Raises
    ------
    KeyError
        If no sequence exists for ``mjd_epoch`` and ``camera``.
    ValueError
        If the sequence's ``kind`` is malformed (see :func:`_parse_kind`), or
        if the number of effective (non-rejected) exposures is not evenly
        divisible by the group size implied by ``kind``.
    """
    try:
        mjds = epochs[mjd_epoch]["sources"]
    except KeyError as error:
        raise KeyError(f"No pixelflat sequence found for {mjd_epoch = }") from error
    try:
        sequence = epochs[mjd_epoch]["sequences"][camera]
    except KeyError as error:
        raise KeyError(f"No pixelflat sequence found for {mjd_epoch = } and {camera = }") from error

    kind = sequence.get("kind", "")
    try:
        kind_pairs = _parse_kind(kind)
    except ValueError as error:
        raise ValueError(f"Invalid sequence kind for {mjd_epoch = }, {camera = }: {kind!r}") from error
    group_size = sum(count for count, _ in kind_pairs)

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
    if effective_expnums.size % group_size:
        raise ValueError(
            f"Sequence for {mjd_epoch = }, {camera = } has {effective_expnums.size} "
            f"effective exposures, which is not divisible by {group_size}"
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

    invalid_frames = {}
    for _type, frame in frame_types.items():
        valid = frame.imagetyp.isin(TYPE_MAPS.get(_type, []) or [])
        invalid_frames[_type] = frame.loc[~valid]
        if not valid.all():
            log.warning(f"\n{invalid_frames[_type].to_string()}")
        else:
            log.info(f"all valid '{_type}' exposures: {set(frame.expnum)}")

    return invalid_frames


def update_epochs(epochs, shifted, cameras=CAMERAS):
    """Apply shifted-exposure rejections across every epoch and camera.

    For each ``(mjd_epoch, camera)`` combination, computes the rejects
    implied by ``shifted`` (see :func:`get_shifted_rejects`), merges and
    stores them back into the sequence (see :func:`set_shifted_rejects`,
    which compresses them via :func:`_compress_expnums`), and validates the
    resulting sequence (see :func:`validate_sequence_kind`). Combinations
    with no matching sequence are skipped with a warning rather than
    raising.

    Parameters
    ----------
    epochs : dict
        Epoch mapping as returned by :func:`load_pixflat_epochs`. Mutated in
        place: each affected sequence's ``"rejects"`` entry is updated. Pass
        the result to :func:`write_pixflat_epochs` to persist the changes.
    shifted : dict[str, list[tuple[int, int]]]
        Mapping of camera identifier to a list of ``(mjd_epoch, expnum)``
        pairs identifying shifted exposures, as expected by
        :func:`get_shifted_rejects`.
    cameras : Iterable[str], optional
        Camera identifiers to check within every epoch. Default is
        :data:`CAMERAS`.

    Returns
    -------
    dict
        The updated epoch mapping (the same object as ``epochs``, mutated in
        place).
    """
    for mjd_epoch in epochs:
        for camera in cameras:
            try:
                rejects = get_shifted_rejects(shifted, mjd_epoch=mjd_epoch, camera=camera, epochs=epochs)
                set_shifted_rejects(rejects=rejects, epochs=epochs, mjd_epoch=mjd_epoch, camera=camera)
                validate_sequence_kind(epochs=epochs, mjd_epoch=mjd_epoch, camera=camera)
            except KeyError:
                log.warning(f"No pixelflat epoch for {mjd_epoch = }, {camera = }, ignoring")
                continue
    return epochs


def write_pixflat_epochs(epochs, epochs_path=None, backup=True):
    """Write an updated pixel-flat epoch mapping back to its YAML file.

    Parameters
    ----------
    epochs : dict
        Epoch mapping to write, in the same format as returned by
        :func:`load_pixflat_epochs` (typically after being mutated by
        :func:`update_epochs`). Written back under the ``"epochs"`` key.
    epochs_path : str or pathlib.Path, optional
        Destination path. Defaults to :data:`PIXFLAT_EPOCHS_PATH`.
    backup : bool, optional
        If True (default) and ``epochs_path`` already exists, copy it to a
        timestamped ``.bak`` file before overwriting.

    Returns
    -------
    str
        The path the epochs were written to.

    Notes
    -----
    If the destination file already exists and has a top-level ``"schemas"``
    key (as in the current pixel-flat epochs file), it is preserved
    unchanged in the rewritten file.
    """
    epochs_path = epochs_path or PIXFLAT_EPOCHS_PATH

    document = {}
    if os.path.isfile(epochs_path):
        with open(epochs_path) as f:
            existing = yaml.safe_load(f) or {}
        if "schemas" in existing:
            document["schemas"] = existing["schemas"]

        if backup:
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            backup_path = f"{epochs_path}.{timestamp}.bak"
            copy2(epochs_path, backup_path)
            log.info(f"backed up existing epochs file to {backup_path}")

    document["epochs"] = epochs

    os.makedirs(os.path.dirname(epochs_path), exist_ok=True)
    with open(epochs_path, "w") as f:
        yaml.safe_dump(document, f, sort_keys=False, default_flow_style=False)

    log.info(f"wrote {len(epochs)} epoch(s) to {epochs_path}")
    return epochs_path


def rsync_enight(mjds):
    """Placeholder for synchronizing engineering nights from LCO.

    Parameters
    ----------
    mjds : int|list[int]
        MJDs to pull from LCO computer
    """
    pass


def get_enights_metadata(mjds):
    """Return the metadata table for the given engineering-night MJDs.

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

    Returns
    -------
    dict
        Mapping of epoch MJD to its calibration metadata (``sources``,
        ``sequences``, ``trigger``, ``comment``), optionally filtered by
        ``filter_by_mjds`` and/or ``filter_by_cameras``. Empty if
        ``filter_by_mjds`` is given and none of the requested MJDs are found.
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
        if len(darks) == 0 or len(flats) % len(darks):
            raise ValueError(
                f"Cannot repeat {len(darks)} dark exposure(s) to match "
                f"{len(flats)} flat exposure(s) for {camera = }"
            )
        n = len(flats) // len(darks)
        darks = darks.loc[darks.index.repeat(n)].reset_index(drop=True)
    if len(biases) < len(flats):
        if len(biases) == 0 or len(flats) % len(biases):
            raise ValueError(
                f"Cannot repeat {len(biases)} bias exposure(s) to match "
                f"{len(flats)} flat exposure(s) for {camera = }"
            )
        n = len(flats) // len(biases)
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
    cflat : image object
        Combined pixel-flat image, either newly combined or loaded from an
        existing product when ``skip_done`` is True.
    cflat_path : str
        Path to the combined pixel-flat image.
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
        Mapping of camera identifiers to ``(cflat_path, mflat_path, fflat_path)``
        tuples giving the paths of the combined, master, and flat-fielded
        pixel-flat products, respectively.
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

        cflat, cflat_path = combine_pixelflats(mjds=mjd, mjd_epoch=mjd, camera=camera, flat_expnums=flat_expnums, skip_done=skip_done)

        mflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="mpixflat", camera=camera)
        fflat_path = path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind="fpixflat", camera=camera)

        cflat_median = fast_median_filter_2d(cflat._data, median_box)
        mflat = (cflat / cflat_median)
        mflat.writeFitsData(mflat_path)

        fflat = cflat / mflat
        fflat.writeFitsData(fflat_path)

        flat_paths[camera] = (cflat_path, mflat_path, fflat_path)

    return flat_paths


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
    smooth_ini = fast_median_filter_2d(np.where(ivar > 0, data, np.nan), size=(size, size))

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
    else:
        log.warning(f"Could not reach masked fraction < 0.05 even at nsig = {nsig}, frac = {frac:4.3f}")

    # https://github.com/desihub/desispec/blob/main/bin/desi_compute_pixel_flatfield#L619

    # now start iterating smoothing and filtering the flat, ignoring newly masked pixels in the smoothing
    mask = mask | (ivar==0)
    smooth = fast_median_filter_2d(np.where(~mask, data, np.nan), size=(size, size))

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
    cflat : image object
        The input combined flat, unchanged.
    mflat : image object
        The normalized master flat.
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
    cflat : image object
        The input combined flat, unchanged.
    mflat : image object
        The normalized master flat.
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
    cflat : image object
        Combined pixel-flat image, as loaded from ``cflat_path``.
    mflat : image object
        Normalized master pixel flat.
    fflat : image object
        Flat-fielded combined image.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"desi"`` or ``"simple"``.
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
    """Create pixel flat-field calibration files for a given camera and set of MJDs.

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
    method : {"desi", "simple"}, optional
        Method to use for flat-field correction. Default is "desi".
    skip_done : bool, optional
        If True, skip processing for already completed files. Default is True.
    dry_run : bool, optional
        If True, log the selected inputs and output paths without creating files.
        Default is False.

    Returns
    -------
    cflat_path : str
        Path to the combined pixel flat file.
    mflat_path : str
        Path to the master pixel flat file.
    fflat_path : str
        Path to the flat-fielded combined pixel flat file.

    Raises
    ------
    ValueError
        If no flat exposures are found in ``sequence`` for ``camera``.

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


def _test_pixflat_paths(mjd_epoch, frame, label=None):
    """Build the raw, preprocessed, and detrended paths of a pixel-flat test frame.

    Parameters
    ----------
    mjd_epoch : int
        MJD identifying the pixel-flat calibration epoch to use.
    frame : pandas.Series
        Frame metadata containing at least ``mjd``, ``camera``, ``expnum``,
        and ``imagetyp``.
    label : str, optional
        Label appended to the intermediate product image type. If omitted,
        ``mjd_epoch`` is used.

    Returns
    -------
    rframe_path : str
        Path to the raw frame.
    pframe_path : str
        Path to the preprocessed frame.
    dframe_path : str
        Path to the detrended frame.
    """
    rframe_path = path.full("lvm_raw", hemi="s", mjd=frame.mjd, camspec=frame.camera, expnum=frame.expnum)
    pframe_path = path.full(
        "lvm_anc", drpver=drpver, tileid=11111, mjd=frame.mjd, kind="p",
        imagetype=f"{frame.imagetyp}_{(label or mjd_epoch)}",
        expnum=frame.expnum, camera=frame.camera,
    )
    dframe_path = path.full(
        "lvm_anc", drpver=drpver, tileid=11111, mjd=frame.mjd, kind="d",
        imagetype=f"{frame.imagetyp}_{(label or mjd_epoch)}",
        expnum=frame.expnum, camera=frame.camera,
    )
    return rframe_path, pframe_path, dframe_path


def test_pixflats(mjd_epoch, frame, label=None, skip_done=True):
    """Reduce a frame using the epoch pixel flat and return the detrended path.

    Parameters
    ----------
    mjd_epoch : int
        MJD identifying the pixel-flat calibration epoch to use.
    frame : pandas.Series
        Frame metadata containing at least ``mjd``, ``camera``, ``expnum``,
        and ``imagetyp``.
    label : str, optional
        Label appended to the intermediate product image type. If omitted,
        ``mjd_epoch`` is used.
    skip_done : bool, optional
        If True, reuse existing preprocessed and detrended products.

    Returns
    -------
    str
        Path to the detrended frame.
    """
    rframe_path, pframe_path, dframe_path = _test_pixflat_paths(mjd_epoch, frame, label=label)

    calibs = drp.get_calib_paths(mjd=frame.mjd, from_sandbox=True)
    mflat_path = path.full(
        "lvm_master", drpver=drpver, tileid=11111, mjd=mjd_epoch,
        kind="mpixflat", camera=frame.camera,
    )

    if skip_done and os.path.isfile(pframe_path):
        log.info(f"{pframe_path} already exists, skipping")
    else:
        image_tasks.preproc_raw_frame(in_image=rframe_path, out_image=pframe_path)

    if skip_done and os.path.isfile(dframe_path):
        log.info(f"{dframe_path} already exists, skipping")
    else:
        image_tasks.detrend_frame(
            in_image=pframe_path,
            out_image=dframe_path,
            in_bias=calibs["bias"][frame.camera],
            in_pixelflat=mflat_path,
            reject_cr=False,
        )

    return dframe_path


def _detect_artifacts(flat_img, bins, threshold=0.95):
    """Detect connected low-value regions in a pixel-flat image.

    Parameters
    ----------
    flat_img : image object
        Pixel-flat image whose ``_data`` array is searched for artifacts.
    bins : iterable[tuple[int, int]]
        Inclusive/exclusive region-size intervals ``(lower, upper)`` used to
        group detected connected components.
    threshold : float, optional
        Maximum flat value considered part of an artifact.

    Returns
    -------
    labels : numpy.ndarray
        Connected-component labels for all pixels below ``threshold``.
    nregions : int
        Number of connected regions.
    sizes : numpy.ndarray
        Number of pixels in each connected region.
    labels_bins : list[tuple[numpy.ndarray, numpy.ndarray, int]]
        Per-size-bin masks, labels, and region counts.
    """
    flat = flat_img._data

    mask = flat <= threshold
    labels, nregions = ndi.label(mask)
    sizes = ndi.sum(mask, labels, range(nregions + 1))

    labels_bins = []
    for bi, bf in bins:
        selection = (sizes >= bi) & (sizes < bf)

        mask_bin = selection[labels]
        labels_bin, n = ndi.label(mask_bin)
        labels_bins.append((mask_bin, labels_bin, n))

    return labels, nregions, sizes, labels_bins


def _calculate_artifact_centroids(labels_bins, max_nregions=10):
    """Calculate centroids of the largest artifact regions in each size bin.

    Parameters
    ----------
    labels_bins : iterable[tuple[numpy.ndarray, numpy.ndarray, int]]
        Per-bin masks, connected-component labels, and region counts returned
        by :func:`_detect_artifacts`.
    max_nregions : int, optional
        Maximum number of regions to report per bin.

    Returns
    -------
    list[list[tuple[int, int]]]
        Integer ``(row, column)`` centroids for up to ``max_nregions`` regions
        in each bin.
    """
    artifacts = []
    for mask, labels, n in labels_bins:
        bin = []
        for ireg in range(1, min(max_nregions + 1, n + 1)):
            i, j = ndi.center_of_mass(mask, labels, index=ireg)
            i = int(i)
            j = int(j)
            bin.append((i, j))
        artifacts.append(bin)

    return artifacts


def _display_artifacts(img, artifacts, bbox_size=15, max_nregions=10, vmin=None, vmax=None, norm=None):
    """Display image cutouts around detected pixel-flat artifacts.

    Parameters
    ----------
    img : image object
        Image containing the artifact regions.
    artifacts : list[list[tuple[int, int]]]
        Artifact centroids grouped by size bin.
    bbox_size : int, optional
        Width of each square cutout in pixels.
    max_nregions : int, optional
        Maximum number of artifact cutouts shown per bin.
    vmin, vmax : float, optional
        Fixed lower and upper limits for the image intensity scale. If either
        is omitted, each cutout is normalized independently, matching the
        previous ``simple_norm`` behavior.
    norm : object, optional
        Astropy normalization object. Used when ``vmin`` or ``vmax`` is omitted
        and it provides an explicit normalization for the cutouts.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure containing the artifact cutouts.
    numpy.ndarray
        Array of Plotly trace references arranged as a ``(nbins, max_nregions)``
        object array.
    """
    use_norm = vmin is None or vmax is None
    nrows = len(artifacts)
    ncols = max_nregions

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        horizontal_spacing=0.01,
        vertical_spacing=0.04,
    )
    traces = np.empty((nrows, ncols), dtype=object)

    hs = bbox_size // 2
    for i in range(nrows):
        for j in range(ncols):
            fig.update_xaxes(visible=False, row=i + 1, col=j + 1)
            fig.update_yaxes(visible=False, row=i + 1, col=j + 1)
            if j >= len(artifacts[i]):
                continue

            ip, jp = artifacts[i][j]
            imin, imax = max(ip - hs, 0), min(ip + hs, 4080)
            jmin, jmax = max(jp - hs, 0), min(jp + hs, 4086)
            data = img._data[imin:imax, jmin:jmax]
            if data.size == 0:
                continue

            if use_norm:
                if norm is not None:
                    # Astropy Normalize-like objects can be used to derive the
                    # equivalent Plotly range from the cutout.
                    z = np.asarray(norm(data), dtype=float)
                    finite = np.isfinite(z)
                    if finite.any():
                        zlo, zhi = np.nanmin(data[finite]), np.nanmax(data[finite])
                    else:
                        zlo, zhi = np.nanmin(data), np.nanmax(data)
                else:
                    finite = data[np.isfinite(data)]
                    if finite.size:
                        zlo, zhi = np.nanpercentile(finite, [10, 90])
                    else:
                        zlo, zhi = None, None
            else:
                zlo, zhi = vmin, vmax

            heatmap = go.Heatmap(
                z=data,
                zmin=zlo,
                zmax=zhi,
                colorscale="Greys",
                zsmooth=False,
                showscale=False,
                hovertemplate="row=%{y}<br>column=%{x}<br>value=%{z}<extra></extra>",
            )
            fig.add_trace(heatmap, row=i + 1, col=j + 1)
            traces[i, j] = len(fig.data) - 1

            fig.add_annotation(
                x=0.1,
                y=0.1,
                text=f"[{ip},{jp}]",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=11, color="greenyellow"),
                row=i + 1,
                col=j + 1,
            )

    fig.update_layout(
        width=1600,
        height=max(350, 150 * max(nrows, 1)),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _display_ratio_hist(
    img,
    labels,
    artifacts,
    bbox_size=15,
    max_nregions=10,
    mu_stat=np.nanmean,
    sigma_stat=np.nanstd,
    bins=10,
    range=(0.95, 1.05)
):
    """Display histograms of artifact-region pixel values.

    Parameters
    ----------
    img : image object
        Image containing the artifact regions.
    labels : numpy.ndarray
        Connected-component labels identifying artifact pixels.
    artifacts : list[list[tuple[int, int]]]
        Artifact centroids grouped by size bin.
    bbox_size : int, optional
        Width of each square cutout in pixels.
    max_nregions : int, optional
        Maximum number of artifact histograms shown per bin.
    mu_stat : callable, optional
        Function used to calculate the location statistic printed on each
        histogram.
    sigma_stat : callable, optional
        Function used to calculate the scale statistic printed on each
        histogram.
    bins : int, optional
        Number of bins in histogram.
    range : tuple, optional
        Range of the histogram.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure containing one histogram panel per artifact region.
    """
    nrows = len(artifacts)
    ncols = max_nregions
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )

    hs = bbox_size // 2
    for i, artifacts_bin in enumerate(artifacts):
        for j, (ip, jp) in enumerate(artifacts_bin):
            imin, imax = max(ip - hs, 0), min(ip + hs, 4080)
            jmin, jmax = max(jp - hs, 0), min(jp + hs, 4086)
            data = img._data[imin:imax, jmin:jmax].ravel()
            mask = labels[imin:imax, jmin:jmax].ravel() != 0
            if data.size == 0:
                continue

            selected = data[mask]
            if selected.size == 0:
                continue

            mu = mu_stat(selected)
            sigma = sigma_stat(selected)

            fig.add_trace(
                go.Histogram(
                    x=selected,
                    histnorm="probability density",
                    xbins={"start": range[0], "end": range[1]},
                    nbinsx=bins,
                ),
                row=i + 1,
                col=j + 1,
            )
            fig.add_annotation(
                x=range[0],
                y=0.9,
                text=f"μ={mu:.3f}<br>σ={sigma:.3f}",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                row=i + 1,
                col=j + 1,
            )

    fig.update_xaxes(range=range)
    fig.update_layout(
        width=1600,
        height=max(350, 150 * max(nrows, 1)),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        barmode="overlay",
    )
    return fig


def _master_pixflat_path(mjd, camera, drpver=drpver, kind="mpixflat"):
    """Build the path of a pixel-flat product for an epoch and camera.

    Parameters
    ----------
    mjd : int
        MJD identifying the pixel-flat epoch.
    camera : str
        Camera identifier, e.g. ``"b1"``.
    drpver : str, optional
        Data-reduction version. Default is the current pipeline version.
    kind : {"mpixflat", "cpixflat", "fpixflat"}, optional
        Product kind. Default is ``"mpixflat"`` (master pixel flat).

    Returns
    -------
    str
        Path to the product.
    """
    return path.full("lvm_master", drpver=drpver, tileid=11111, mjd=mjd, kind=kind, camera=camera)


def _pct_within(x, y, tol=0.01):
    """Return the percentage of pixels where ``y`` is within ``tol`` of ``x``.

    Parameters
    ----------
    x : numpy.ndarray
        Reference values.
    y : numpy.ndarray
        Target values, same shape as ``x``.
    tol : float, optional
        Fractional tolerance. Default is 0.01 (1%).

    Returns
    -------
    float
        Percentage of pixels with ``x * (1 - tol) <= y <= x * (1 + tol)``.
    """
    return float(((y <= x * (1 + tol)) & (y >= x * (1 - tol))).sum() / x.size * 100)


def display_pixflats_comparison(mjd_tar, mjd_ref, drpver, cameras=CAMERAS):
    """Compare target and reference master pixel flats for LVM cameras.

    Each camera is shown as a density heatmap of target versus reference pixel
    values, together with the one-to-one relation and ±1% bounds. Panels keep
    their position in the 3x3 grid even when only a subset of cameras is
    requested, and cameras whose master pixel flat is missing for either epoch
    are left empty (with a warning).

    Parameters
    ----------
    mjd_tar : int
        MJD of the target pixel-flat epoch.
    mjd_ref : int
        MJD of the reference pixel-flat epoch.
    drpver : str
        Data-reduction version used to locate the pixel-flat products.
    cameras : iterable[str], optional
        Cameras to show. Default is all of :data:`CAMERAS`.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure containing one comparison panel per camera.
    """
    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
    )

    for idx, camera in enumerate(CAMERAS):
        if camera not in cameras:
            continue
        row, col = divmod(idx, 3)
        path_ref = _master_pixflat_path(mjd_ref, camera, drpver=drpver)
        path_tar = _master_pixflat_path(mjd_tar, camera, drpver=drpver)
        if not (os.path.isfile(path_ref) and os.path.isfile(path_tar)):
            log.warning(f"missing master pixel flat for {camera = } ({path_ref} or {path_tar}), leaving panel empty")
            continue
        pflat_ref = image_tasks.loadImage(path_ref)
        pflat_tar = image_tasks.loadImage(path_tar)

        x = pflat_ref._data.ravel()
        y = pflat_tar._data.ravel()

        # Match the original 100x100 density estimate over [0.95, 1.05].
        H, xe, ye = np.histogram2d(
            x, y,
            bins=100,
            range=[(0.95, 1.05), (0.95, 1.05)],
            density=False,
        )
        H = H / H.sum() * 100
        H[H == 0] = np.nan

        # The original matplotlib plot used a logarithmic normalization with
        # max_cut=1.5. Transform the positive values to log10 so that the
        # Plotly colorscale has the same logarithmic intensity mapping.
        positive = H[np.isfinite(H) & (H > 0)]
        if positive.size:
            log_h = np.full_like(H, np.nan, dtype=float)
            log_h[H > 0] = np.log10(H[H > 0])
            zmin = float(np.nanmin(log_h))
            zmax = float(np.log10(1.5))
            colorbar = dict(
                title="%",
                len=0.75,
                thickness=12,
                x=1.0,
                xanchor="left",
                tickvals=np.log10([0.01, 0.1, 1.0, 1.5]).tolist(),
                ticktext=["0.01", "0.1", "1.0", "1.5"],
            )
        else:
            log_h = H
            zmin = zmax = None
            colorbar = dict(title="%", len=0.75, thickness=12, x=1.0, xanchor="left")

        fig.add_trace(
            go.Heatmap(
                x=(xe[:-1] + xe[1:]) / 2,
                y=(ye[:-1] + ye[1:]) / 2,
                z=log_h.T,
                colorscale="Greys",
                zmin=zmin,
                zmax=zmax,
                zsmooth=False,
                colorbar=colorbar,
                customdata=H.T,
                hovertemplate="Reference: %{x:.5f}<br>Target: %{y:.5f}<br>Fraction: %{customdata:.3g}%<extra></extra>",
                showscale=True,
            ),
            row=row + 1,
            col=col + 1,
        )

        line = np.asarray([0, 10])
        fig.add_trace(
            go.Scatter(
                x=line,
                y=line,
                mode="lines",
                line=dict(color="rgb(51,51,51)", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row + 1,
            col=col + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=line,
                y=line * 1.01,
                mode="lines",
                line=dict(color="rgb(51,51,51)", dash="dot", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row + 1,
            col=col + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=line,
                y=line * 0.99,
                mode="lines",
                line=dict(color="rgb(51,51,51)", dash="dot", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row + 1,
            col=col + 1,
        )

        percent = _pct_within(x, y)
        fig.add_annotation(
            x=0.01,
            y=0.95,
            # xref=f"x{idx + 1} domain",
            # yref=f"y{idx + 1} domain",
            text=f"{percent:.2f}% pixels within 1% consistency",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=11),
            row=row + 1,
            col=col + 1,
        )

    fig.update_xaxes(range=[0.95, 1.05], constrain="domain")
    fig.update_yaxes(range=[0.95, 1.05], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        width=1400,
        height=1400,
        template="plotly_white",
        margin=dict(l=80, r=80, t=70, b=80),
        title=dict(
            text=f"Pixel-flat comparison: target {mjd_tar} vs reference {mjd_ref}",
            x=0.5,
        ),
    )

    # Preserve the original figure's semantic axis labels: reference on x,
    # target on y.
    fig.update_xaxes(title_text=f"Flatfield {mjd_ref}")
    fig.update_yaxes(title_text=f"Flatfield {mjd_tar}")
    return fig


def display_artifacts_comparison(mjd_tar, mjd_ref, drpver, camera, bins=ARTIFACT_BINS, bbox_size=30, max_regions=10):
    """Compare target and reference master pixel flats around detected artifacts.

    Artifacts are connected low-value regions (see :func:`_detect_artifacts`)
    found in the target master pixel flat. Cutouts around the largest regions
    of each size bin are shown for both epochs, for their ratio, and as a
    histogram of the ratio within the artifact pixels.

    Parameters
    ----------
    mjd_tar : int
        MJD of the target pixel-flat epoch.
    mjd_ref : int
        MJD of the reference pixel-flat epoch.
    drpver : str
        Data-reduction version used to locate the pixel-flat products.
    camera : str
        Camera identifier, e.g. ``"b1"``.
    bins : iterable[tuple[int, int]], optional
        Region-size bins used to group artifacts. Default is :data:`ARTIFACT_BINS`.
    bbox_size : int, optional
        Width of each square cutout in pixels. Default is 30.
    max_regions : int, optional
        Maximum number of regions shown per size bin. Default is 10.

    Returns
    -------
    dict[str, plotly.graph_objects.Figure]
        Figures keyed by ``"flat_ref"``, ``"flat_tar"``, ``"flat_rat"``
        (reference / target), and ``"hist_rat"``.
    """
    pflat_ref = image_tasks.loadImage(_master_pixflat_path(mjd_ref, camera, drpver=drpver))
    pflat_tar = image_tasks.loadImage(_master_pixflat_path(mjd_tar, camera, drpver=drpver))

    labels, _, _, labels_bins = _detect_artifacts(pflat_tar, bins=bins, threshold=0.95)
    artifacts = _calculate_artifact_centroids(labels_bins, max_nregions=max_regions)

    return {
        "flat_ref": _display_artifacts(pflat_ref, artifacts, bbox_size=bbox_size, max_nregions=max_regions, vmin=0.95, vmax=1.05),
        "flat_tar": _display_artifacts(pflat_tar, artifacts, bbox_size=bbox_size, max_nregions=max_regions, vmin=0.95, vmax=1.05),
        "flat_rat": _display_artifacts(pflat_ref/pflat_tar, artifacts, bbox_size=bbox_size, max_nregions=max_regions, vmin=0.99, vmax=1.01),
        "hist_rat": _display_ratio_hist(pflat_ref/pflat_tar, labels, artifacts, bbox_size=bbox_size, max_nregions=max_regions),
    }


def display_flatfielded_comparison(mjd_tar, mjd_ref, drpver, camera, test_flat_paths, bins=ARTIFACT_BINS, bbox_size=30, max_regions=10):
    """Compare a test frame flat-fielded with the target and reference pixel flats.

    The regions shown are the artifacts detected in the target master pixel
    flat, the same ones displayed by :func:`display_artifacts_comparison`.

    Parameters
    ----------
    mjd_tar : int
        MJD of the target pixel-flat epoch.
    mjd_ref : int
        MJD of the reference pixel-flat epoch.
    drpver : str
        Data-reduction version used to locate the pixel-flat products.
    camera : str
        Camera identifier, e.g. ``"b1"``.
    test_flat_paths : dict[int, dict[str, str]]
        Detrended test-frame paths, indexed as ``test_flat_paths[mjd_epoch][camera]``
        (see :func:`test_pixflats`).
    bins : iterable[tuple[int, int]], optional
        Region-size bins used to group artifacts. Default is :data:`ARTIFACT_BINS`.
    bbox_size : int, optional
        Width of each square cutout in pixels. Default is 30.
    max_regions : int, optional
        Maximum number of regions shown per size bin. Default is 10.

    Returns
    -------
    dict[str, plotly.graph_objects.Figure]
        Figures keyed by ``"flat_ref"``, ``"flat_tar"``, ``"flat_rat"``
        (reference / target), and ``"hist_rat"``.
    """
    pflat_tar = image_tasks.loadImage(_master_pixflat_path(mjd_tar, camera, drpver=drpver))
    labels, _, _, labels_bins = _detect_artifacts(pflat_tar, bins=bins, threshold=0.95)
    artifacts = _calculate_artifact_centroids(labels_bins, max_nregions=max_regions)

    fflat_ref = image_tasks.loadImage(test_flat_paths[mjd_ref][camera])
    fflat_tar = image_tasks.loadImage(test_flat_paths[mjd_tar][camera])

    return {
        "flat_ref": _display_artifacts(fflat_ref, artifacts, bbox_size=bbox_size, max_nregions=max_regions),
        "flat_tar": _display_artifacts(fflat_tar, artifacts, bbox_size=bbox_size, max_nregions=max_regions),
        "flat_rat": _display_artifacts(fflat_ref/fflat_tar, artifacts, bbox_size=bbox_size, max_nregions=max_regions, vmin=0.99, vmax=1.01),
        "hist_rat": _display_ratio_hist(fflat_ref/fflat_tar, labels, artifacts, bbox_size=bbox_size, max_nregions=max_regions),
    }


def _raw_quadrant_sections(img):
    """Return the science (TRIMSEC) sections of the quadrants of a raw frame.

    Parameters
    ----------
    img : image object
        Raw frame.

    Returns
    -------
    list[str]
        One ``[x1:x2, y1:y2]`` section per quadrant, in amplifier order. Falls
        back to the default sections if the header has no ``TRIMSEC`` keywords.
    """
    sections = img._header["TRIMSEC?"]
    if not sections:
        log.warning(f"TRIMSEC not found in header, assuming {image_tasks.DEFAULT_TRIMSEC}")
        return image_tasks.DEFAULT_TRIMSEC
    return list(sections.values())


def _measure_contrast(pixels, cut, center, n, window, stat=np.nanmedian):
    """Measure the contrast along a cut, from the end nearest the CCD center to the outer edge.

    ``n`` windows of ``window`` pixels are evenly spaced along the cut, the
    first one touching the end of the cut nearest to ``center`` and the last one
    touching the opposite (outer) end. The contrast of each window is its level
    relative to the level of the innermost window.

    Parameters
    ----------
    pixels : numpy.ndarray
        Pixel coordinates of the cut.
    cut : numpy.ndarray
        Count levels of the cut, same shape as ``pixels``.
    center : float
        Coordinate of the CCD center along the cut, in the same units as ``pixels``.
    n : int
        Number of contrast measurements.
    window : int
        Width in pixels of each window.
    stat : callable, optional
        Statistic used to measure the level of each window. Default is
        ``numpy.nanmedian``.

    Returns
    -------
    positions : numpy.ndarray
        Pixel coordinate of the center of each window, from inner to outer.
    levels : numpy.ndarray
        Level of each window.
    contrasts : numpy.ndarray
        Level of each window relative to the innermost one.
    """
    if abs(pixels[-1] - center) < abs(pixels[0] - center):
        pixels, cut = pixels[::-1], cut[::-1]

    hw = min(window, pixels.size) // 2
    idxs = np.linspace(hw, pixels.size - 1 - hw, n).round().astype(int)
    positions = pixels[idxs]
    levels = np.asarray([stat(cut[max(idx - hw, 0):idx + hw + 1]) for idx in idxs])
    return positions, levels, levels / levels[0]


def _average_raw_frames(raw_paths):
    """Average raw frames pixel by pixel.

    Frames are accumulated one at a time, so only one frame and the running
    sum are held in memory.

    Parameters
    ----------
    raw_paths : list[str]
        Paths of the raw frames, all from the same camera.

    Returns
    -------
    img : image object
        Image with the header of the first frame and the NaN-aware mean of all
        the frames as data.
    expnums : list
        Exposure numbers of the frames, from the ``EXPOSURE`` header keyword.

    Raises
    ------
    ValueError
        If the frames come from different cameras or have different shapes.
    """
    img, total, count, expnums = None, None, None, []
    for raw_path in raw_paths:
        frame = image_tasks.loadImage(raw_path)
        data = frame._data.astype(float)
        if img is None:
            img, total, count = frame, np.zeros_like(data), np.zeros(data.shape, dtype=int)
        elif frame._header.get("CCD") != img._header.get("CCD") or data.shape != total.shape:
            raise ValueError(
                f"cannot average {os.path.basename(raw_path)} ({frame._header.get('CCD')}, {data.shape}) with "
                f"{os.path.basename(raw_paths[0])} ({img._header.get('CCD')}, {total.shape})"
            )
        valid = np.isfinite(data)
        total[valid] += data[valid]
        count += valid
        expnums.append(frame._header.get("EXPOSURE", os.path.basename(raw_path)))

    with np.errstate(invalid="ignore", divide="ignore"):
        img._data = np.where(count > 0, total / count, np.nan)
    return img, expnums


def display_raw_cuts(raw_frames, rows=(1020, 3060), columns=(1021, 3098), cut_width=21, normalize=False,
                     stat=np.nanmedian, n_contrast=5, contrast_window=51, split_amps=True, image_binning=8,
                     combined_flat=False):
    """Plot cuts of count levels along X and Y of a raw flat frame, or of the average of several.

    Each cut collapses a band of ``cut_width`` rows (cuts along X) or columns
    (cuts along Y) centered on the given row or column, and is split into the
    quadrants it crosses. For each quadrant, only the science region given by
    the ``TRIMSEC`` header keywords is used, so the overscan regions are left out.

    Parameters
    ----------
    raw_frames : str, list[str], pandas.Series or pandas.DataFrame
        Raw frame path(s), or frame metadata with at least ``mjd``, ``camera``,
        and ``expnum``, from which the raw paths are built. Unless
        ``combined_flat`` is True, only the first frame is used.
    rows : iterable[int], optional
        1-based rows of the cuts along X. Default is (1020, 3060), the centers
        of the lower and upper quadrants.
    columns : iterable[int], optional
        1-based columns of the cuts along Y. Default is (1021, 3098), the
        centers of the left and right quadrants.
    cut_width : int, optional
        Width in pixels of the band collapsed into each cut. Default is 21.
    normalize : bool, optional
        If True, divide the cuts and the raw image by the median of the science
        regions of the whole image. Default is False.
    stat : callable, optional
        Statistic used to collapse the band and to measure the level of each
        contrast window. Default is ``numpy.nanmedian``.
    n_contrast : int, optional
        Number of contrast measurements per cut and quadrant, evenly spaced from
        the end of the cut nearest the CCD center to the outer edge of the CCD
        (see :func:`_measure_contrast`). The contrast is the level of each
        window relative to the innermost one. Set to 0 to skip. Default is 5.
    contrast_window : int, optional
        Width in pixels of each contrast window. Default is 51.
    split_amps : bool, optional
        If True, show one column of panels per quadrant. If False, show all the
        cuts along X in one axis and all the cuts along Y in another, next to
        the raw image with the cuts and quadrants overlaid. Default is True.
    image_binning : int, optional
        Binning factor of the raw image shown when ``split_amps`` is False,
        to keep the figure size manageable. Default is 8.
    combined_flat : bool, optional
        If True, average all the given frames pixel by pixel (see
        :func:`_average_raw_frames`) and use the average in place of a single
        frame. Default is False.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure with the cuts along X in the top row and the cuts along Y in the
        bottom row, one color per cut, either split in one column per quadrant
        or in single axes next to the raw image (see ``split_amps``). Pixel
        coordinates are 1-based, in the frame of the raw image. Contrast
        windows are marked on the cuts and labeled with their contrast.
    contrasts : pandas.DataFrame
        One row per cut, quadrant, and window, with the columns ``amp``,
        ``axis`` (``"x"`` or ``"y"``), ``cut`` (the row or column of the cut),
        ``window`` (0 is the innermost), ``position``, ``level``, and
        ``contrast``. Empty if ``n_contrast`` is 0.
    """
    if isinstance(raw_frames, pd.Series):
        raw_frames = raw_frames.to_frame().T
    if isinstance(raw_frames, pd.DataFrame):
        raw_paths = [
            path.full("lvm_raw", hemi="s", mjd=frame.mjd, camspec=frame.camera, expnum=frame.expnum)
            for _, frame in raw_frames.iterrows()
        ]
    elif isinstance(raw_frames, str):
        raw_paths = [raw_frames]
    else:
        raw_paths = list(raw_frames)
    if not raw_paths:
        raise ValueError("no raw frame given")

    if combined_flat and len(raw_paths) > 1:
        img, expnums = _average_raw_frames(raw_paths)
        label = f"{img._header.get('CCD', '')}, average of {len(expnums)} frames ({expnums[0]}-{expnums[-1]})"
    else:
        img = image_tasks.loadImage(raw_paths[0])
        label = f"{img._header.get('CCD', '')}-{img._header.get('EXPOSURE', os.path.basename(raw_paths[0]))}"
    quad_sections = _raw_quadrant_sections(img)
    quad_bounds = [image_tasks._parse_ccd_section(section) for section in quad_sections]
    nquads = len(quad_sections)
    ny, nx = img._data.shape
    # 1-based coordinates of the CCD center
    center_y, center_x = ny / 2 + 0.5, nx / 2 + 0.5
    science = np.concatenate([img._data[y0:y1, x0:x1].ravel() for (x0, x1), (y0, y1) in quad_bounds]).astype(float)
    norm = np.nanmedian(science) if normalize else 1.0

    # collapse each cut, one segment per quadrant it crosses
    cuts = []
    hw = cut_width // 2
    for axis, cut_at in [("x", row) for row in rows] + [("y", column) for column in columns]:
        name = f"{'row' if axis == 'x' else 'column'} {cut_at}"
        segments = []
        for i, ((x0, x1), (y0, y1)) in enumerate(quad_bounds):
            lo, hi = (y0, y1) if axis == "x" else (x0, x1)
            if not lo < cut_at <= hi:
                continue
            quad = img._data[y0:y1, x0:x1].astype(float)
            band = slice(max(cut_at - 1 - hw, lo) - lo, min(cut_at + hw, hi) - lo)
            if axis == "x":
                segments.append((i, np.arange(x0, x1) + 1, stat(quad[band, :], axis=0)))
            else:
                segments.append((i, np.arange(y0, y1) + 1, stat(quad[:, band], axis=1)))
        if not segments:
            log.warning(f"{name} doesn't cross the science region of any quadrant, skipping")
            continue
        segments = [(i, pixels, cut / norm) for i, pixels, cut in segments]
        cuts.append((axis, cut_at, name, segments))

    if split_amps:
        fig = make_subplots(
            rows=2,
            cols=nquads,
            shared_yaxes=True,
            horizontal_spacing=0.02,
            vertical_spacing=0.12,
            subplot_titles=[f"AMP{i+1} {sec}" for i, sec in enumerate(quad_sections)] + [""] * nquads,
        )
        cuts_col = 1
    else:
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"rowspan": 2}, {}], [None, {}]],
            column_widths=[0.4, 0.6],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
            subplot_titles=["average raw image" if combined_flat and len(raw_paths) > 1 else "raw image", "cuts along X", "cuts along Y"],
        )
        cuts_col = 2

        # raw image, binned, with a color range set by the science regions
        binned = img._data[:ny - ny % image_binning, :nx - nx % image_binning].astype(float) / norm
        binned = binned.reshape(ny // image_binning, image_binning, nx // image_binning, image_binning).mean(axis=(1, 3))
        zmin, zmax = np.nanpercentile(science / norm, [1, 99])
        fig.add_trace(
            go.Heatmap(
                z=binned,
                x=np.arange(binned.shape[1]) * image_binning + (image_binning + 1) / 2,
                y=np.arange(binned.shape[0]) * image_binning + (image_binning + 1) / 2,
                zmin=zmin,
                zmax=zmax,
                colorscale="gray",
                colorbar=dict(title="normalized" if normalize else "ADU", orientation="h", x=0.2, xanchor="center", y=-0.12, len=0.35, thickness=12),
                hovertemplate="x=%{x:.0f}, y=%{y:.0f}: %{z:.4g}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        for i, ((x0, x1), (y0, y1)) in enumerate(quad_bounds):
            fig.add_shape(
                type="rect", x0=x0 + 0.5, x1=x1 + 0.5, y0=y0 + 0.5, y1=y1 + 0.5,
                line=dict(color="white", width=1, dash="dot"), row=1, col=1,
            )
            fig.add_annotation(
                text=f"AMP{i+1}", x=x0 + 40, y=y1 - 40, xanchor="left", yanchor="top", showarrow=False,
                font=dict(color="white", size=11), row=1, col=1,
            )
        fig.update_xaxes(title_text="X (pixel)", range=[0.5, nx + 0.5], constrain="domain", row=1, col=1)
        fig.update_yaxes(title_text="Y (pixel)", range=[0.5, ny + 0.5], scaleanchor="x", constrain="domain", row=1, col=1)

    colors = qualitative.Plotly
    records = []
    # number of cuts drawn per panel and quadrant span, used to alternate the
    # contrast labels above and below the markers
    ncuts_panel = {}
    for k, (axis, cut_at, name, segments) in enumerate(cuts):
        color = colors[k % len(colors)]
        row = 1 if axis == "x" else 2

        if not split_amps:
            if axis == "x":
                line = dict(x=[0.5, nx + 0.5], y=[cut_at, cut_at])
            else:
                line = dict(x=[cut_at, cut_at], y=[0.5, ny + 0.5])
            fig.add_trace(
                go.Scatter(**line, mode="lines", line=dict(color=color, width=1.5), name=name, legendgroup=name,
                           showlegend=False, hoverinfo="skip"),
                row=1,
                col=1,
            )

        for n, (i, pixels, cut) in enumerate(segments):
            col = i + 1 if split_amps else cuts_col
            (x0, x1), (y0, y1) = quad_bounds[i]
            panel = (row, col, (x0, x1) if axis == "x" else (y0, y1))
            npanel = ncuts_panel.get(panel, 0)
            ncuts_panel[panel] = npanel + 1
            fig.add_trace(
                go.Scattergl(
                    x=pixels,
                    y=cut,
                    mode="lines",
                    line=dict(color=color, width=1),
                    name=name,
                    legendgroup=name,
                    showlegend=n == 0,
                ),
                row=row,
                col=col,
            )
            if n_contrast <= 0:
                continue

            positions, levels, contrasts = _measure_contrast(
                pixels, cut, center_x if axis == "x" else center_y, n=n_contrast, window=contrast_window, stat=stat,
            )
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=levels,
                    mode="markers+text",
                    marker=dict(color=color, size=7, symbol="circle-open", line=dict(width=2)),
                    text=[f"{contrast:.2f}" for contrast in contrasts],
                    textposition="top center" if npanel % 2 == 0 else "bottom center",
                    textfont=dict(color=color, size=10),
                    customdata=contrasts,
                    hovertemplate="%{x}: level=%{y:.4g}, contrast=%{customdata:.3f}",
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            records.extend(
                dict(amp=i + 1, axis=axis, cut=cut_at, window=j, position=position, level=level, contrast=contrast)
                for j, (position, level, contrast) in enumerate(zip(positions, levels, contrasts))
            )

    contrasts = pd.DataFrame(records, columns=["amp", "axis", "cut", "window", "position", "level", "contrast"])

    ylabel = "normalized counts" if normalize else "counts (ADU)"
    fig.update_xaxes(title_text="X (pixel)", row=1, col=None if split_amps else cuts_col)
    fig.update_xaxes(title_text="Y (pixel)", row=2, col=None if split_amps else cuts_col)
    fig.update_yaxes(title_text=f"{ylabel}, cut along X", row=1, col=cuts_col)
    fig.update_yaxes(title_text=f"{ylabel}, cut along Y", row=2, col=cuts_col)
    fig.update_layout(
        title=dict(text=label, x=0.5),
        width=1600,
        height=700 if split_amps else 800,
        margin=dict(l=20, r=20, t=80, b=20 if split_amps else 90),
    )
    return fig, contrasts


def _raw_flat_paths(epoch, camera, nframes=None):
    """Return the raw paths of the pixel flats in an epoch's sequence.

    The raw paths are built from the epoch's source MJDs, without using the
    frames metadata, so they can be found even if nothing has been reduced.

    Parameters
    ----------
    epoch : dict
        Epoch definition as returned by :func:`load_pixflat_epochs`, with the
        ``sources`` MJDs and the per-camera ``sequences``.
    camera : str
        Camera identifier, e.g. ``"b1"``.
    nframes : int, optional
        Maximum number of paths to return. Default is all of them.

    Returns
    -------
    list[str]
        Paths of the raw flats of the sequence found on disk, in sequence order.

    Raises
    ------
    KeyError
        If the epoch has no sequence for ``camera``.
    """
    sequence = _parse_sequence(epoch["sequences"][camera])
    raw_paths = []
    for expnum in np.ravel(sequence.get("flat", [])):
        for mjd in np.atleast_1d(epoch["sources"]):
            raw_path = path.full("lvm_raw", hemi="s", mjd=mjd, camspec=camera, expnum=expnum)
            if os.path.isfile(raw_path):
                raw_paths.append(raw_path)
                break
        if nframes is not None and len(raw_paths) == nframes:
            break
    return raw_paths


def _raw_flat_path(epoch, camera):
    """Return the raw path of the first pixel flat in an epoch's sequence.

    Parameters
    ----------
    epoch : dict
        Epoch definition as returned by :func:`load_pixflat_epochs`.
    camera : str
        Camera identifier, e.g. ``"b1"``.

    Returns
    -------
    str or None
        Path of the first raw flat of the sequence found on disk (see
        :func:`_raw_flat_paths`), or None if none is found.

    Raises
    ------
    KeyError
        If the epoch has no sequence for ``camera``.
    """
    raw_paths = _raw_flat_paths(epoch, camera, nframes=1)
    return raw_paths[0] if raw_paths else None


def _edge_contrast(contrasts):
    """Return the lowest contrast at the CCD edge.

    Parameters
    ----------
    contrasts : pandas.DataFrame
        Contrast measurements as returned by :func:`display_raw_cuts`.

    Returns
    -------
    float
        Minimum over quadrants and cuts of the outermost-window contrast. NaN
        if there are no measurements.
    """
    if contrasts.empty:
        return np.nan
    return float(contrasts[contrasts.window == contrasts.window.max()].contrast.min())
