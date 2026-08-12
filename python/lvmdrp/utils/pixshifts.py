
import os
import yaml
import numpy as np
from copy import deepcopy as copy
from scipy import signal
from pprint import pformat
import warnings

from lvmdrp.core.constants import PIXELSHIFTS_DIR, LVM_NROWS
from lvmdrp.core.plot import plt, create_subplots, save_fig, plot_image_shift
from lvmdrp import log, path, __version__ as drpver

from lvmdrp.core.image import loadImage


SCHEMAS = [
    {
        'name': 'expnum',
        'dtype': 'int',
        'description': 'An exposure number'
        },
    {
        'name': 'camera',
        'dtype': 'str',
        'description': "A camera frame e.g., 'b1', 'r2'"
        },
    {
        'name': 'imagetype',
        'dtype': 'str',
        'description': "Type of the image e.g., 'flat', 'arc'"
        },
    {
        'name': 'rows',
        'dtype': 'dict[int, int]',
        'description': 'A mapping of row and size of pixel shift'
        },
    {
        'name': 'source',
        'dtype': 'str',
        'description': "Origin of the pixel shift detection, either 'drp', 'qc' or 'user'"
        }]


def _remove_spikes(data, width=11, threshold=0.5):
    """Returns a data array with spikes removed

    Parameters
    ----------
    data : array_like
        1-dimensional array of data
    width : int, optional
        width of the window where spikes are located, by default 11
    threshold : float, optional
        threshold to remove spikes, by default 0.5

    Returns
    -------
    array_like
        1-dimensional array with spikes removed
    """
    data_ = copy(data)
    hw = width // 2
    for irow in range(hw, data.size - hw):
        chunk = data[irow-hw:irow+hw+1]
        has_peaks = (chunk[0] == chunk[-1]) and (np.abs(chunk) > chunk[0]).any()
        if has_peaks and (chunk != 0).sum() / width < threshold:
            data_[irow-hw:irow+hw+1] = chunk[0]
    return data_


def _fillin_valleys(data, width=18):
    """fills in valleys in the data array

    Parameters
    ----------
    data : array_like
        1-dimensional array of data
    width : int, optional
        width of the valley to fill, by default 18

    Returns
    -------
    array_like
        1-dimensional array with filled valleys
    """
    data_out = copy(data)
    top = data[0]
    for i in range(data.size):
        if data[i] > top:
            top = data[i]
        if data[i] == top:
            continue
        if data[i] < top:
            j_ini = i
        j_fin = j_ini
        for j in range(j_ini, data.size):
            if data[j] < top:
                continue
            if data[j] == top:
                j_fin = j
                break
        if j_fin - j_ini < width:
            data_out[j_ini:j_fin] = top
    return data_out


def _no_stepdowns(data):
    """Removes stepdowns in the data array

    Parameters
    ----------
    data : array_like
        1-dimensional array of data

    Returns
    -------
    array_like
        1-dimensional array with stepdowns removed
    """
    data_out = copy(data)
    top = data[0]
    for i in range(data.size):
        if data[i] > top:
            top = data[i]
        if data[i] == top:
            continue
        if data[i] < top:
            data_out[i] = top
    return data_out


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
    refs = ref_frames.loc[ref_frames.expnum != frame.expnum]
    if frame.imagetyp == "flat" and frame.ldls|frame.quartz:
        refs = refs.query("imagetyp == 'flat' and (ldls|quartz)")
    elif frame.imagetyp == "flat":
        refs = refs.query("imagetyp == 'flat' and not (ldls|quartz)")
    elif frame.imagetyp == "arc" or (frame.neon|frame.hgne|frame.argon|frame.xenon) and  not (frame.ldls|frame.quartz):
        refs = refs.loc[(refs.neon|refs.hgne|refs.argon|refs.xenon) & (~(refs.ldls|refs.quartz))]
    else:
        refs = refs.query("imagetyp == @frame.imagetyp")

    ref_expnums = refs.expnum.unique()
    if len(ref_expnums) == 0:
        warnings.warn(f"no reference frame found for {frame.imagetyp}, found only {len(ref_expnums)} exposure(s)")
        return None
    idx = np.argmin(np.abs(ref_expnums - frame.expnum))
    return ref_expnums[idx]


def _clean_reference(frames, shifts_qc):
    if not shifts_qc:
        return frames

    # remove QC reported shifts
    specexp, _ = list(zip(*shifts_qc.items()))
    spec, exp = zip(*specexp)
    spec = list(map(lambda s: f"sp{s}", spec))

    r = frames.sort_values("expnum")
    r = r.loc[~(frames.spec.isin(spec)&frames.expnum.isin(exp))]

    return r


def load_qc_shifts(mjd):
    """Reads QC reports with the electronic pixel shifts"""

    shifts_report = {}
    qc_shifts_path = os.path.join(os.environ["LVM_SANDBOX"], "shift_monitor", f"shift_{mjd}.txt")
    if not os.path.isfile(qc_shifts_path):
        return shifts_report

    with open(qc_shifts_path, "r") as f:
        lines = f.readlines()[2:]

    for line in lines:
        cols = line[:-1].split()
        if not cols:
            continue
        _, exp, _, spec = cols[:4]
        exp = int(exp)
        spec = spec[-1]
        shifts = np.array([int(_) for _ in cols[4:]])
        shifts_report[(spec, exp)] = dict(zip(shifts[::2], shifts[1::2][::-1]))

    return shifts_report


def load_shifts(mjd, shifts_dir=PIXELSHIFTS_DIR):
    """Reads validated electronic shifts from LVMCORE"""
    shifts_path = os.path.join(shifts_dir, f"shifts-{mjd}.yaml")
    if not os.path.isfile(shifts_path):
        return []

    log.info(f"loading pixel shifts from {shifts_path}")
    with open(shifts_path, 'r') as f:
        shift_detections = yaml.safe_load(f) or {}
    shifts = shift_detections.get("shifts", []) or []
    return shifts


def write_shifts(shifts, mjd, shifts_dir=PIXELSHIFTS_DIR, verbose=False):
    shifts_path = os.path.join(shifts_dir, f"shifts-{mjd}.yaml")

    shift_detections = dict.fromkeys(["schemas", "shifts"])
    shift_detections["schemas"] = SCHEMAS
    shift_detections["shifts"] = shifts

    if verbose:
        log.info(f"going to write shifts file for {mjd = }:\n {pformat(shift_detections)}")

    with open(shifts_path, "w") as f:
        yaml.safe_dump(shift_detections, f, sort_keys=True)


def locate_shifted(shifts, expnum, camera):
    for idx, shift in enumerate(shifts):
        if shift.get("camera") == camera and shift.get("expnum") == expnum:
            return idx, shift
    return None, {}


def set_shifted(shifts, expnum, camera, imagetype, rows, source):
    if source not in {"drp", "qc", "user"}:
        raise ValueError(f"Invalid value for `source`: {source}. Expected either: 'drp', 'qc' or 'user'")

    idx, shift = locate_shifted(shifts, expnum, camera)
    if idx is None:
        new_shift = dict(expnum=expnum, camera=camera, imagetype=imagetype, rows=rows, source=source)
        shifts.append(new_shift)
    else:
        new_shift = copy(shift)
        new_shift["rows"] = rows
        shifts.pop(idx)
        shifts.insert(idx, new_shift)
    shifts = sorted(shifts, key=lambda i: (i["expnum"], i["camera"]))
    return shifts


def expand_shifts(rows):
    shift_profile = np.zeros(LVM_NROWS)
    if rows is None:
        return shift_profile

    for rows, amount in rows.items():
        shift_profile[rows:] = int(amount)
    return shift_profile


def compress_shifts(shift_profile, as_dict=False):
    rows = np.where(np.gradient(shift_profile) > 0)[0][1::2]
    if as_dict:
       return dict(zip(rows.tolist(), shift_profile[rows].astype("int32").tolist()))
    return rows, shift_profile[rows].astype("int32").tolist()


def apply_shift_correction(image, shifts, display_plots=False):
    image_out = copy(image)
    mjd = image._header.get("SMJD", image._header.get("MJD"))
    expnum = image._header["EXPOSURE"]
    camera = image._header["CCD"]
    imagetyp = image._header["IMAGETYP"]

    y_pixels = np.arange(LVM_NROWS)

    idx, shift = locate_shifted(shifts, expnum=image._header["EXPOSURE"], camera=image._header["CCD"])
    if idx is not None:
        source = shift["source"]
        rows = shift["rows"]
        log.info(f"applying electronic pixel shifts for {expnum = } | {camera = } | {imagetyp = }: {rows}")
        shift_profile = expand_shifts(rows)
        for irow in range(len(shift_profile)):
            if shift_profile[irow] > 0:
                image_out._data[irow, :] = np.roll(image._data[irow, :], int(shift_profile[irow]))
        for i, (row, amount) in enumerate(rows.items()):
            image_out._header[f"HIERARCH {camera.upper()} PIXSHIFT SHIFT{i+1}"] = (f"{row}:{amount}", "electronic pixel shift row:amount")
        image_out._header[f"HIERARCH {camera.upper()} PIXSHIFT SOURCE"] = (source, "electronic pixel shift source")

        cmaps = {"drp": "Blues", "qc": "Greens", "user": "Purples"}
        fig, ax = create_subplots(to_display=display_plots, figsize=(15,7), sharex=True, layout="constrained")
        fig.suptitle(f"{mjd = } | {expnum = } | {camera = } | {imagetyp = }", fontsize="x-large")
        ax.step(y_pixels, shift_profile, where="mid", lw=2, color="tab:purple", label=source)
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlabel("Y (pixel)")
        ax.set_ylabel("Shift (pixel)")
        plot_image_shift(ax, image._data, shift_profile, cmap="Reds")
        axis = plot_image_shift(ax, image_out._data, shift_profile, cmap=cmaps[source], inset_pos=(0.14,1.0-0.32))
        plt.setp(axis, yticklabels=[], ylabel="")
    else:
        fig = None
        log.info(f"no electronic pixel shifts found for {expnum = } | {camera = } | {imagetyp = }")
    return image_out, fig


def compare_shifts(image, drp_shifts=None, qc_shifts=None, user_shifts=None, raw_shifts=None, which_shifts="drp", display_plots=False):
    """Compares the chosen electronic pixel shifts to other alternative solutions

    Parameters
    ----------
    images : list
        list of input images
    out_images : list
        list of output images
    drp_shifts : np.ndarray
        DRP electronic pixel shifts, by default None
    qc_shifts : np.ndarray
        QC electronic pixel shifts, by default None
    user_shifts : np.ndarray
        user-provided electronic pixel shifts, by default None
    raw_shifts : np.ndarray
        raw DRP electronic pixel shifts, by default None
    which_shifts : str
        chosen electronic pixel shifts, by default "drp"
    display_plots : bool
        display plots, by default False

    Returns
    -------
    list
        list of corrected images
    np.ndarray
        the chosen electronic pixel shifts
    str
        name of the chosen electronic pixel shifts ('drp', 'qc' or 'user')
    """
    image_out = copy(image)
    mjd = image._header.get("SMJD", image._header["MJD"])
    expnum, camera = image._header["EXPOSURE"], image._header["CCD"]
    imagetyp = image._header["IMAGETYP"]

    if which_shifts == "drp" and (drp_shifts > 0).any():
        this_shifts = drp_shifts
        image_color = "Blues"
    elif which_shifts == "drp":
        which_shifts = "qc"
    if which_shifts == "qc" and (qc_shifts > 0).any():
        this_shifts = qc_shifts
        image_color = "Greens"
    elif which_shifts == "qc":
        which_shifts = "user"
    if which_shifts == "user" and (user_shifts > 0).any():
        this_shifts = user_shifts
        image_color = "Purples"
    elif which_shifts == "user":
        which_shifts = "drp"
        this_shifts = drp_shifts
        image_color = "Blues"

    if not (this_shifts > 0).any():
        return image_out, this_shifts, which_shifts

    for irow in range(len(this_shifts)):
        if this_shifts[irow] > 0:
            image_out._data[irow, :] = np.roll(image._data[irow, :], int(this_shifts[irow]))

    fig, ax = create_subplots(to_display=display_plots, figsize=(15,7), sharex=True, layout="constrained")
    fig.suptitle(f"{mjd = } | {expnum = } | {camera = } | {imagetyp = }", fontsize="x-large")
    y_pixels = np.arange(this_shifts.size)
    if raw_shifts is not None:
        ax.step(y_pixels, raw_shifts, where="mid", lw=0.5, color="0.9", label="raw DRP")
    ax.step(y_pixels, this_shifts, where="mid", color="k", lw=3)
    if drp_shifts is not None:
        ax.step(y_pixels, drp_shifts, where="mid", lw=1, color="tab:blue", label="DRP")
    if qc_shifts is not None:
        ax.step(y_pixels, qc_shifts, where="mid", lw=2, color="tab:green", label="QC")
    if user_shifts is not None:
        ax.step(y_pixels, user_shifts, where="mid", lw=2, color="tab:purple", label="user")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlabel("Y (pixel)")
    ax.set_ylabel("Shift (pixel)")
    plot_image_shift(ax, image._data, this_shifts, cmap="Reds")
    axis = plot_image_shift(ax, image_out._data, this_shifts, cmap=image_color, inset_pos=(0.14,1.0-0.32))
    plt.setp(axis, yticklabels=[], ylabel="")
    save_fig(
        fig,
        product_path=path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, camera=camera, kind="e", imagetype=imagetyp, expnum=expnum),
        to_display=display_plots,
        figure_path="qa",
        label="pixel_shifts"
    )
    return image_out, this_shifts, which_shifts


def detect_shifts(in_image, ref_image, report=None,
                  max_shift=10, threshold_spikes=0.6, flat_spikes=11,
                  fill_gaps=20, shift_rows=None, interactive=False, display_plots=False):
    """Corrects pixel shifts in raw frames based on reference frames and a selection of spectral regions

    Given a set of raw frames, reference frames and a mask, this function corrects pixel shifts
    based on the reference frames and a selection of spectral regions.

    Parameters
    ----------
    in_images : list
        list of input raw images for the same spectrograph (brz)
    out_images : str
        output pixel shifts file
    ref_images : list
        list of input reference images for the same spectrograph
    report : dict, optional
        input report with keys (spec, expnum) and values (shift_rows, amount), by default None
    max_shift : int, optional
        maximum shift in pixels, by default 10
    threshold_spikes : float, optional
        threshold for spike removal, by default 0.6
    flat_spikes : int, optional
        width of the spike removal, by default 11
    fill_gaps : int, optional
        width of the gap filling, by default 20
    interactive : bool, optional
        interactive mode, by default False
    display_plots : bool, optional
        display plots, by default False

    Returns
    -------
    np.ndarray
        pixel shifts
    np.ndarray
        pixel correlations
    list
        list of corrected images
    """
    log.info(f"loading reference image from {ref_image}")
    cdata = loadImage(ref_image)._data

    # read all three detrended images and channel combine them
    log.info(f"loading input image from {in_image}")
    image = loadImage(in_image)
    rdata = image._data

    # load input images and initialize output images
    image_out = copy(image)
    imagetyp = image._header["IMAGETYP"]
    expnum = image._header["EXPOSURE"]
    camera = image._header["CCD"]

    # initialize shifts
    raw_shifts = None
    dshifts = expand_shifts(None)
    qshifts = expand_shifts(report)
    ushifts = expand_shifts(None)
    which_shifts = "drp"

    # if user provided shifts, use those and return
    if shift_rows is not None:
        log.info("parsing user provided pixel shifts")
        ushifts = np.zeros(LVM_NROWS)
        for irow in shift_rows:
            ushifts[irow:] += 2
        corrs = np.zeros_like(ushifts)
        which_shifts = "user"

        image_out, shifts, _, = compare_shifts(image=image, raw_shifts=raw_shifts,
                                               drp_shifts=dshifts, qc_shifts=qshifts, user_shifts=ushifts,
                                               which_shifts=which_shifts, display_plots=display_plots)
        return shifts, which_shifts, corrs, image_out

    log.info(f"running row-by-row cross-correlation for {imagetyp = } | {expnum = } | {camera = }")
    corrs = np.zeros_like(dshifts)
    for irow in range(LVM_NROWS):
        cimg_row = cdata[irow]
        rimg_row = rdata[irow]
        if np.all(cimg_row == 0) or np.all(rimg_row == 0):
            continue

        shift = signal.correlation_lags(cimg_row.size, rimg_row.size, mode="same")
        corr = signal.correlate(cimg_row, rimg_row, mode="same")

        mask = (np.abs(shift) <= max_shift)
        shift = shift[mask]
        corr = corr[mask]

        max_corr = np.argmax(corr)
        dshifts[irow] = shift[max_corr]
        corrs[irow] = corr[max_corr]

    dshifts = _remove_spikes(dshifts, width=flat_spikes, threshold=threshold_spikes)
    dshifts = _fillin_valleys(dshifts, width=fill_gaps)
    dshifts = _no_stepdowns(dshifts)

    # compare QC reports with the electronic pixel shifts
    shift_detected_or_reported = (qshifts > 0).any() or (dshifts > 0).any()
    if shift_detected_or_reported and interactive:
        log.info("interactive mode enabled")
        qshifted_rows = compress_shifts(qshifts, as_dict=True)
        dshifted_rows = compress_shifts(dshifts, as_dict=True)
        log.info(f"QC shifted rows: {qshifted_rows}")
        log.info(f"DRP shifted rows: {dshifted_rows}")

        compare_shifts(image=image, drp_shifts=dshifts, qc_shifts=qshifts, user_shifts=ushifts, raw_shifts=raw_shifts,
                       which_shifts="drp", display_plots=display_plots)

        if (dshifts != qshifts).any():
            log.warning("QC and DRP shift detections disagree")

        answer = input("apply [q]c, [d]rp, [c]ustom shifts or [n]one: ")
        if answer.lower() == "q":
            log.info("choosing QC shifts")
            shifts = qshifts
            which_shifts = "qc"
        elif answer.lower() == "d":
            log.info("choosing DRP shifts")
            shifts = dshifts
            which_shifts = "drp"
        elif answer.lower() == "c":
            log.info("choosing user-provided shifts")
            answer = input("provide comma-separated shifts and press enter: ")
            shift_rows = np.array([int(_) for _ in answer.split(",")])
            ushifts = np.zeros(LVM_NROWS)
            for irow in shift_rows:
                ushifts[irow:] += 2
            shifts = ushifts
            corrs = np.zeros_like(ushifts)
            which_shifts = "user"
        elif answer.lower() == "n":
            log.info("choosing to apply no shift")
            ushifts = np.zeros(LVM_NROWS)
            shifts = ushifts
            corrs = np.zeros_like(ushifts)
            which_shifts = "user"
    elif shift_detected_or_reported and not interactive:
        log.warning(f"no shift will be applied to the images: {in_image}")
        ushifts = np.zeros(LVM_NROWS)
        shifts = ushifts
        corrs = np.zeros_like(ushifts)
        which_shifts = "user"
    elif not shift_detected_or_reported:
        log.info("no shifts detected or reported")

    # apply pixel shifts to the images
    image_out, shifts, _, = compare_shifts(image=image, raw_shifts=raw_shifts,
                                           drp_shifts=dshifts, qc_shifts=qshifts, user_shifts=ushifts,
                                           which_shifts=which_shifts, display_plots=display_plots)

    return shifts, which_shifts, corrs, image_out