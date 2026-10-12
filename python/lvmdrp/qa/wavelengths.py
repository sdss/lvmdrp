# encoding: utf-8
"""Wavelength and LSF QA of a calibration epoch.

Shows the quality of the master wavelength and LSF solutions of an epoch and
compares two ways of combining its arc exposures (see
:func:`~lvmdrp.functions.rssMethod.determine_wavelength_solution`):

- ``measurements``: the arc lines are measured in each extracted arc exposure
  (:func:`~lvmdrp.functions.rssMethod.measure_arc_lines`), the line centroids
  of each exposure are aligned to remove relative shifts along the dispersion
  axis (e.g., thermal), and the master solutions are fitted to the combined
  measurements (:func:`~lvmdrp.functions.rssMethod.fit_wavelength_solution`).
- ``pixels``: the extracted arc exposures are summed in pixel space and the
  lines are measured once in the sum.

The dashboard shows, for each camera, the residuals of the polynomial
wavelength and LSF fits, their coefficients along the slit, the residuals of
the Gaussian fits to the line profiles in each exposure, and how the LSF of
both combinations compares with the line widths of single exposures. See
:func:`qa_arc_combinations`.
"""

import os
from datetime import datetime, timezone
from html import escape
from multiprocessing import Pool

import numpy as np
import bottleneck as bn
import pandas as pd
import plotly.graph_objects as go
import warnings
from astropy.io import fits
from astropy.table import Table
from plotly.subplots import make_subplots
from scipy.special import erf
from scipy.stats import binned_statistic, binned_statistic_2d

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.core.constants import CAMERAS, CALIBRATION_NEEDS
from lvmdrp.core.rss import lvmArc
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.qa.report import (
    THEME, SERIES, DIVERGING, CALIB_TILEID, base_layout, style_axes, html_table, picker, write_dashboard, epoch_qa_dir,
)

METHODS = ("measurements", "pixels")
METHOD_LABELS = {"measurements": "per-arc measurements", "pixels": "arcs summed in pixels"}
METHOD_COLORS = {"measurements": SERIES[0], "pixels": SERIES[1]}
CHANNEL_COLORS = {"b": SERIES[0], "r": SERIES[1], "z": SERIES[2]}
# ordinal ramp for the exposures of a sequence, sequential blue steps 250-600 (legible on both themes)
EXPOSURE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
FWHM_FACTOR = 8 * np.log(2)


def _qa_dir(mjd_epoch, name):
    """Return the default directory of a wavelength QA dashboard, in the ancillary directory of the epoch"""
    return epoch_qa_dir(drpver, mjd_epoch, "wave", name)


def _measure_task(task):
    """Measure the arc lines of one extracted arc (worker of :func:`compare_camera`)"""
    in_arc, out_arc = task
    try:
        rss_tasks.measure_arc_lines(in_arc=in_arc, out_arc=out_arc)
    except Exception as error:
        log.error(f"failed measuring arc lines in '{in_arc}': {error}")
        return in_arc, str(error)
    return in_arc, None


def master_paths(camera, mjd_epoch, kind="longterm"):
    """Paths of the master wavelength and LSF solutions of a camera in an epoch"""
    kinds = ("mwave", "mlsf") if kind == "longterm" else ("nwave", "nlsf")
    return {name: path.full("lvm_master", drpver=drpver, tileid=CALIB_TILEID, mjd=mjd_epoch, camera=camera, kind=master_kind)
            for name, master_kind in zip(("wave", "lsf"), kinds)}


def master_arcs(wave_path):
    """Arc exposures combined in a master wavelength solution, from its FRAMES extension

    Returns
    -------
    list[dict] | None
        TILEID, MJD, EXPNUM and CAMERA of each exposure, or None if the master or its FRAMES extension is missing
    """
    if not os.path.isfile(wave_path):
        return None
    with fits.open(wave_path) as hdus:
        if "FRAMES" not in hdus or hdus["FRAMES"].data is None or len(hdus["FRAMES"].data) == 0:
            return None
        frames = Table(hdus["FRAMES"].data)
    return [{"tileid": int(row["TILEID"]), "mjd": int(row["MJD"]), "expnum": int(row["EXPNUM"]), "camera": str(row["CAMERA"]).strip()}
            for row in frames]


def arc_products(arc):
    """Paths of the lvmArc and extracted arc of one exposure (a dict with tileid, mjd, expnum and camera)"""
    keys = dict(drpver=drpver, tileid=arc["tileid"], mjd=arc["mjd"], expnum=arc["expnum"])
    return {"lvmarc": path.full("lvm_frame", kind=f"Arc-{arc['camera']}", **keys),
            "xarc": path.full("lvm_anc", kind="x", imagetype="arc", camera=arc["camera"], **keys)}


def compare_camera(camera, arcs, masters, output_dir, with_sum=False, nprocs=1, skip_done=True):
    """Collect the products of a camera, measuring only what the pipeline has not produced

    The lvmArc products and master solutions written by
    :func:`~lvmdrp.functions.run_calseq.create_wavelengths` are used as they are. Arcs without
    an lvmArc are measured from their extracted arc, and a missing master is fitted, into
    `output_dir`. With `with_sum`, the extracted arcs are also summed in pixels and measured
    once, also into `output_dir`. The pipeline products are never written.

    Parameters
    ----------
    camera : str
        Camera name, e.g. 'b1'
    arcs : list[dict]
        Arc exposures, each with tileid, mjd, expnum and camera
    masters : dict
        Paths of the master wavelength ('wave') and LSF ('lsf') solutions, see :func:`master_paths`
    output_dir : str
        Directory of the QA
    with_sum : bool, optional
        Also measure the arcs summed in pixels, by default False
    nprocs : int, optional
        Number of processes measuring arcs, by default 1
    skip_done : bool, optional
        Reuse the products already in `output_dir`, by default True

    Returns
    -------
    dict
        Paths of the products used for each combination ('pixels' is None without the sum) and
        how many lvmArcs were reused from the pipeline or measured
    """
    meas_dir = os.path.join(output_dir, "measurements")
    pix_dir = os.path.join(output_dir, "pixels")

    larcs, tasks, missing, reused, previous = [], [], [], 0, 0
    for arc in arcs:
        products = arc_products(arc)
        if os.path.isfile(products["lvmarc"]):
            larcs.append(products["lvmarc"])
            reused += 1
            continue
        local = os.path.join(meas_dir, os.path.basename(products["lvmarc"]))
        larcs.append(local)
        if skip_done and os.path.isfile(local):
            previous += 1
            continue
        if os.path.isfile(products["xarc"]):
            tasks.append((products["xarc"], local))
        else:
            missing.append(arc["expnum"])
    log.info(f"{camera}: {reused} lvmArc products reused, {previous} measured in an earlier run, {len(tasks)} arcs to measure, "
             f"{len(missing)} without extracted arc")
    if nprocs > 1 and len(tasks) > 1:
        with Pool(min(nprocs, len(tasks))) as pool:
            results = pool.map(_measure_task, tasks)
    else:
        results = [_measure_task(task) for task in tasks]
    failed = [in_arc for in_arc, error in results if error is not None]
    larcs = [larc for larc in larcs if os.path.isfile(larc)]
    if not larcs:
        raise RuntimeError(f"no arc lines measured ({len(missing)} arcs not extracted, {len(failed)} failed)")

    meas = {"arcs": larcs, "wave": masters["wave"], "lsf": masters["lsf"], "master": "pipeline"}
    if not (os.path.isfile(masters["wave"]) and os.path.isfile(masters["lsf"])):
        meas.update(wave=os.path.join(meas_dir, f"lvm-mwave-{camera}.fits"), lsf=os.path.join(meas_dir, f"lvm-mlsf-{camera}.fits"), master="fitted")
        if not (skip_done and os.path.isfile(meas["wave"]) and os.path.isfile(meas["lsf"]) and not tasks):
            log.info(f"{camera}: no master solutions in the pipeline, fitting them from {len(larcs)} lvmArcs")
            rss_tasks.fit_wavelength_solution(in_arcs=larcs, out_wave=meas["wave"], out_lsf=meas["lsf"])

    paths = {"camera": camera, "measurements": meas, "pixels": None, "reused": reused, "previous": previous,
             "measured": len(tasks) - len(failed), "missing": missing + list(failed), "sum_problem": None}
    if with_sum:
        xarcs = [arc_products(arc)["xarc"] for arc in arcs]
        xarcs = [xarc for xarc in xarcs if os.path.isfile(xarc)]
        if len(xarcs) < 2:
            paths["sum_problem"] = f"{len(xarcs)} extracted arcs, need at least two to sum"
            log.warning(f"{camera}: {paths['sum_problem']}")
        else:
            pix = {"arcs": [os.path.join(pix_dir, f"lvmArc-{camera}-sum.fits")],
                   "wave": os.path.join(pix_dir, f"lvm-mwave-{camera}.fits"), "lsf": os.path.join(pix_dir, f"lvm-mlsf-{camera}.fits")}
            if not (skip_done and all(os.path.isfile(p) for p in [pix["wave"], pix["lsf"], *pix["arcs"]])):
                log.info(f"{camera}: measuring arc lines in the sum of {len(xarcs)} arcs")
                try:
                    rss_tasks.determine_wavelength_solution(in_arcs=xarcs, out_wave=pix["wave"], out_lsf=pix["lsf"],
                                                            arcs_combination="pixels", out_arcs=pix["arcs"])
                except Exception as error:
                    # the comparison is optional: keep checking the camera's own solutions without it
                    paths["sum_problem"] = f"measuring the summed arcs failed: {error}"
                    log.error(f"{camera}: {paths['sum_problem']}")
            if paths["sum_problem"] is None:
                paths["pixels"] = pix
    return paths


def _at_lines(values, cent):
    """Evaluate per-fiber arrays (nfibers, npixels) at the line positions (nfibers, nlines) by linear interpolation"""
    pixels = np.arange(values.shape[1])
    out = np.full(cent.shape, np.nan)
    for ifiber in range(cent.shape[0]):
        good = np.isfinite(cent[ifiber])
        if good.any() and np.isfinite(values[ifiber]).all():
            out[ifiber, good] = np.interp(cent[ifiber, good], pixels, values[ifiber])
    return out


def _lines_on(ref_lines, lvmarc, extnames=("LINE_CENT", "LINE_FWHM", "LINE_FLUX", "LINE_BKG")):
    """Return line measurements of an lvmArc on the given reference lines, NaN where missing or masked"""
    waves, _, _, _, mask = lvmarc.get_lines()
    index = {wave: j for j, wave in enumerate(waves)}
    tables = [lvmarc.get_line_table(extname) for extname in extnames]
    out = [np.full((mask.shape[0], ref_lines.size), np.nan) for _ in extnames]
    for j, wave in enumerate(ref_lines):
        if wave in index:
            k = index[wave]
            for arr, values in zip(out, tables):
                arr[:, j] = np.where(mask[:, k], np.nan, values[:, k])
    return out


def _robust_scatter(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def _linear_terms(curves):
    """Least-squares straight line of each fiber's curve along the dispersion axis, as (intercept, slope)"""
    x = np.arange(curves.shape[1], dtype=float)
    xm = x.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ym = np.nanmean(curves, axis=1)
        slope = np.nansum((x - xm)[None] * (curves - ym[:, None]), axis=1) / np.sum((x - xm) ** 2)
    return ym - slope * xm, slope


def _solution_points(traces, cent, fwhm_pix, ref_lines):
    """Data and model of the wavelength and LSF fits of one method at the line positions

    Returns a dict per solution ('wave', 'lsf') with the measured values, the model at the
    lines, the residuals (model - data, in Angstrom), the model curves and their linear terms.
    """
    wave = traces["wave"]._data.astype(float)
    lsf = traces["lsf"]._data.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        disp = _at_lines(np.abs(np.gradient(wave, axis=1)), cent)
        data = {"wave": np.where(np.isfinite(cent), ref_lines[None], np.nan), "lsf": fwhm_pix * disp}
        points = {}
        for name, curves in (("wave", wave), ("lsf", lsf)):
            model = _at_lines(curves, cent)
            intercept, slope = _linear_terms(curves)
            points[name] = {"data": data[name], "model": model, "residual": model - data[name], "curves": curves,
                            "intercept": intercept, "slope": slope, "coeffs": traces[name]._coeffs,
                            "masked": traces[name]._mask.all(axis=1) if traces[name]._mask is not None else np.zeros(curves.shape[0], bool)}
    points["cent"] = cent
    return points


# line profile residuals: default fitting window (aperture) of the arc lines and binning
DEFAULT_APERTURE = 12
PROFILE_STEP = 0.25
FIBER_BIN = 16
CORE_HALF_WIDTH = 1.5


def _half_width(lvmarc):
    """Half width in pixels of the window over which the arc lines of an lvmArc were fitted"""
    camera = lvmarc._header.get("CCD", "")
    return int(lvmarc._header.get(f"{camera} WAVE APERTURE", DEFAULT_APERTURE)) // 2


def _line_model(pix, cent, fwhm, flux, bkg):
    """Fitted line model at pixels `pix`: a pixel-integrated Gaussian of total flux `flux` plus a constant background

    This is the model of :meth:`~lvmdrp.core.spectrum1d.Spectrum1D.fitSepGauss`, whose fiber profile kernel is negligible.
    All arguments broadcast together.
    """
    norm = np.sqrt(2) * fwhm / (2 * np.sqrt(2 * np.log(2)))
    return 0.5 * flux * (erf((pix + 0.5 - cent) / norm) - erf((pix - 0.5 - cent) / norm)) + bkg


def _profile_points(lvmarc, line_index):
    """Residuals of the Gaussian fits to the lines of one arc, relative to each line's peak

    The model is the pixel-integrated Gaussian plus constant background fitted by
    :meth:`~lvmdrp.core.spectrum1d.Spectrum1D.fitSepGauss`, evaluated over the fitting window
    stored in the lvmArc header, centered on the fitted centroid.

    Returns
    -------
    tuple[numpy.ndarray]
        Fiber index, line index (in `line_index` order), offset from the centroid (pix) and
        residual (data - model) / peak, for every valid pixel
    """
    waves, cent, fwhm, flux, mask = lvmarc.get_lines()
    bkg = np.nan_to_num(lvmarc.get_line_table("LINE_BKG"), nan=0.0)
    data = lvmarc._data.astype(float)
    if lvmarc._mask is not None:
        data = np.where(lvmarc._mask, np.nan, data)
    nfibers, npixels = data.shape
    valid = ~mask & np.isfinite(cent) & np.isfinite(fwhm) & np.isfinite(flux) & (fwhm > 0) & (flux > 0)
    cent, fwhm, flux = np.where(valid, cent, 0.0), np.where(valid, fwhm, 1.0), np.where(valid, flux, 0.0)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    peak = flux / (sigma * np.sqrt(2 * np.pi))

    half_width = _half_width(lvmarc)
    offsets = np.arange(-half_width, half_width + 1)
    pix = np.round(cent)[..., None] + offsets
    inside = (pix >= 0) & (pix < npixels)
    pix_c = np.clip(pix, 0, npixels - 1).astype(int)
    values = data[np.arange(nfibers)[:, None, None], pix_c]
    model = _line_model(pix, cent[..., None], fwhm[..., None], flux[..., None], bkg[..., None])
    dx = pix - cent[..., None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        residual = (values - model) / peak[..., None]
    keep = valid[..., None] & inside & np.isfinite(residual) & (np.abs(dx) <= half_width)
    fibers = np.broadcast_to(np.arange(nfibers)[:, None, None], pix.shape)
    lines = np.broadcast_to(np.array([line_index.get(wave, -1) for wave in waves])[None, :, None], pix.shape)
    keep &= lines >= 0
    return fibers[keep], lines[keep], dx[keep], residual[keep]


# arc spectra: the reference fiber of the line measurements and up to four more fibers, spread along the slit by default
SPECTRUM_POSITIONS = (0.05, 0.35, 0.65, 0.95)
MAX_SPECTRUM_FIBERS = len(SPECTRUM_POSITIONS)
DEFAULT_REFERENCE_FIBER = 319
# pixels kept between the line windows when drawing a spectrum (1 in every N)
SPECTRUM_THINNING = 8


def _spread_fibers(fibers):
    """Good non-standard fibers spread along the slit, at :data:`SPECTRUM_POSITIONS`"""
    candidates = fibers[fibers.good & (fibers.telescope != "Spec")]
    if candidates.empty:
        candidates = fibers
    return [int(candidates.fiber.iloc[int(round(quantile * (len(candidates) - 1)))]) for quantile in SPECTRUM_POSITIONS]


def _arc_spectra(lvmarc, fibers):
    """Arc spectrum, its error and the fitted lines of the given fibers of an lvmArc

    Parameters
    ----------
    lvmarc : lvmdrp.core.rss.lvmArc
        Arc exposure with its line measurements
    fibers : dict[str, int]
        Fiber index of each option of the fiber selector

    Returns
    -------
    dict[str, dict]
        Spectrum and lines of each fiber, by selector option
    """
    waves, cent, fwhm, flux, mask = lvmarc.get_lines()
    bkg = np.nan_to_num(lvmarc.get_line_table("LINE_BKG"), nan=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # where each line sits, from the other fibers, to mark it where it was rejected
        expected = np.nanmedian(np.where(mask, np.nan, cent), axis=0)
    spectra = {}
    for key, fiber in fibers.items():
        data = lvmarc._data[fiber].astype(float)
        error = lvmarc._error[fiber].astype(float) if lvmarc._error is not None else np.full(data.shape, np.nan)
        if lvmarc._mask is not None:
            data = np.where(lvmarc._mask[fiber], np.nan, data)
        error = np.where(np.isfinite(error) & (error > 0), error, np.nan)
        spectra[key] = {"fiber": fiber, "data": data, "error": error, "waves": waves, "cent": cent[fiber], "fwhm": fwhm[fiber],
                        "flux": flux[fiber], "bkg": bkg[fiber], "mask": mask[fiber], "expected": expected,
                        "half_width": _half_width(lvmarc)}
    return spectra


def _asinh_ticks(scale, ymin, ymax, min_spacing=0.8):
    """Ticks of an arcsinh(flux / scale) axis at round flux values, spread out from zero in both directions"""
    values = [m * 10.0 ** k for k in range(-2, 8) for m in (1, 2, 5)]
    kept = [0.0]
    for sign in (1, -1):
        last = 0.0
        for value in values:
            flux = sign * value
            if not ymin <= flux <= ymax:
                continue
            t = np.arcsinh(flux / scale)
            if abs(t - last) >= min_spacing:
                kept.append(flux)
                last = t
    kept.sort()
    labels = [f"{v / 1e6:g}M" if abs(v) >= 1e6 else f"{v / 1e3:g}k" if abs(v) >= 1e3 else f"{v:g}" for v in kept]
    return [float(np.arcsinh(v / scale)) for v in kept], labels


def _profile_maps(points, nfibers, nlines, edges):
    """Median profile residuals in bins of offset, across fibers and across lines, and their core RMS"""
    fibers, lines, dx, residual = points
    fiber_edges = np.arange(0, nfibers + FIBER_BIN, FIBER_BIN) - 0.5
    line_edges = np.arange(nlines + 1) - 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        by_fiber = binned_statistic_2d(fibers, dx, residual, statistic="median", bins=[fiber_edges, edges]).statistic
        by_line = binned_statistic_2d(lines, dx, residual, statistic="median", bins=[line_edges, edges]).statistic
        profile = binned_statistic(dx, residual, statistic="median", bins=edges).statistic
    centers = 0.5 * (edges[1:] + edges[:-1])
    core = np.abs(centers) <= CORE_HALF_WIDTH
    return {"by_fiber": by_fiber, "by_line": by_line, "profile": profile,
            "fiber_centers": 0.5 * (fiber_edges[1:] + fiber_edges[:-1]),
            "core_rms": np.sqrt(np.nanmean(profile[core] ** 2)) if np.isfinite(profile[core]).any() else np.nan}


def camera_metrics(paths, fibers=None):
    """Measure the quality of both master solutions of a camera and how the combination methods compare

    Parameters
    ----------
    paths : dict
        Paths of the products, see :func:`compare_camera`
    fibers : list[int], optional
        Up to four fibers whose arc spectra are shown next to the reference fiber. By default, fibers
        spread along the slit

    Returns
    -------
    dict
        Per-fiber and per-line metrics and the arrays used in the charts
    """
    methods = [method for method in METHODS if paths.get(method)]
    lvmarcs = [lvmArc.from_file(larc) for larc in paths["measurements"]["arcs"]]
    sum_arc = lvmArc.from_file(paths["pixels"]["arcs"][0]) if "pixels" in methods else None
    traces = {method: {kind: TraceMask.from_file(paths[method][kind]) for kind in ("wave", "lsf")} for method in methods}

    # lines of each exposure and their combination, as in fit_wavelength_solution
    ref_lines, flux, cent, fwhm, masked = rss_tasks._combine_lvmarcs_lines(lvmarcs)
    flux_c, cent_c, fwhm_c, masked_c, offsets = rss_tasks._combine_arc_lines(flux, cent, fwhm, masked)
    cent_c = np.where(masked_c, np.nan, cent_c)
    fwhm_c = np.where(masked_c, np.nan, fwhm_c)
    flux_k = np.where(masked, np.nan, flux)
    # lines measured in the sum of the exposures
    if sum_arc is not None:
        cent_s, fwhm_s, _, _ = _lines_on(ref_lines, sum_arc)
    else:
        cent_s = fwhm_s = np.full(cent_c.shape, np.nan)
    line_index = {wave: j for j, wave in enumerate(ref_lines)}

    wave_m = traces["measurements"]["wave"]._data.astype(float)
    nfibers, npixels = wave_m.shape
    lines_of = {"measurements": (cent_c, fwhm_c), "pixels": (cent_s, fwhm_s)}
    fits = {method: _solution_points(traces[method], *lines_of[method], ref_lines) for method in methods}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        disp_lines = _at_lines(np.abs(np.gradient(wave_m, axis=1)), cent_c)
        wave_lines = _at_lines(wave_m, cent_c)
        # line widths in Angstrom of single exposures (median over exposures)
        width_single = fwhm_c * disp_lines
        lsf_lines = {method: _at_lines(traces[method]["lsf"]._data.astype(float), cent_c) for method in methods}
        bias = {method: lsf_lines[method] - width_single for method in methods}

        # broadening of the sum predicted from the spread of the line positions among the exposures
        weights = np.nan_to_num(bn.nanmedian(flux_k, axis=2), nan=0.0)
        weights = np.where(np.isfinite(offsets), weights, 0.0)
        # fibers measured in fewer than two exposures (e.g., standard fibers) have no shifts between exposures
        multi = (weights > 0).sum(axis=0) >= 2
        weights[:, ~multi] = 0.0
        wsum = weights.sum(axis=0)
        mean_offset = np.divide((weights * offsets).sum(axis=0), wsum, where=wsum > 0, out=np.full(nfibers, np.nan))
        var_offset = np.divide((weights * (offsets - mean_offset[None]) ** 2).sum(axis=0), wsum, where=wsum > 0, out=np.full(nfibers, np.nan))
        # line width of each exposure relative to the median of the exposures, per fiber (median over lines), where it has offsets
        fwhm_k = np.where(masked, np.nan, fwhm)
        dfwhm = np.where(weights > 0, bn.nanmedian(fwhm_k - bn.nanmedian(fwhm_k, axis=0)[None], axis=2), np.nan)
        predicted = np.where(multi, FWHM_FACTOR * var_offset, np.nan)
        observed = np.where(multi, bn.nanmedian(fwhm_s**2 - fwhm_c**2, axis=1), np.nan)
        lsf = {method: traces[method]["lsf"]._data.astype(float) for method in methods}

    # fibers with a solution in both methods and lines measured
    good = np.ones(nfibers, dtype=bool)
    for method in methods:
        for kind in ("wave", "lsf"):
            good &= ~fits[method][kind]["masked"]
    good &= np.isfinite(width_single).sum(axis=1) > 0

    # flux and width of each line in each exposure relative to all exposures, median over good fibers, to tell brightness from focus
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lit = ((weights > 0) & good[None])[..., None]
        ratio = np.where(lit, flux_k / bn.nanmedian(flux_k, axis=0)[None], np.nan)
        line_dflux = 100 * (bn.nanmedian(ratio, axis=1) - 1)
        line_dfwhm = bn.nanmedian(np.where(lit, fwhm_k - bn.nanmedian(fwhm_k, axis=0)[None], np.nan), axis=1)
        exposure_dflux = 100 * (bn.nanmedian(ratio.reshape(ratio.shape[0], -1), axis=1) - 1)

    # residuals of the line profile fits, per exposure, all exposures together and in the sum
    exposures = [str(larc._header.get("EXPOSURE")) for larc in lvmarcs]
    exposure_info = pd.DataFrame({
        "expnum": exposures, "obstime": [larc._header.get("OBSTIME") for larc in lvmarcs],
        "night": [int(larc._header.get("SMJD", larc._header.get("MJD", -1))) for larc in lvmarcs],
        "labtemp": [float(larc._header.get("LABTEMP", np.nan)) for larc in lvmarcs],
        "ccdtemp": [float(larc._header.get("CCDTEMP1", np.nan)) for larc in lvmarcs],
    })
    half_width = max(_half_width(larc) for larc in lvmarcs + ([sum_arc] if sum_arc is not None else []))
    edges = np.arange(-half_width, half_width + PROFILE_STEP / 2, PROFILE_STEP)
    points = [_profile_points(larc, line_index) for larc in lvmarcs]
    profiles = {expnum: _profile_maps(pts, nfibers, ref_lines.size, edges) for expnum, pts in zip(exposures, points)}
    profiles["all"] = _profile_maps(tuple(np.concatenate(arrs) for arrs in zip(*points)), nfibers, ref_lines.size, edges)
    if sum_arc is not None:
        profiles["sum"] = _profile_maps(_profile_points(sum_arc, line_index), nfibers, ref_lines.size, edges)

    slitmap = lvmarcs[0]._slitmap
    if slitmap is not None and len(slitmap) != nfibers:
        slitmap = slitmap[slitmap["spectrographid"] == int(paths["camera"][1])]
    labels = np.asarray(slitmap["orig_ifulabel"]).astype(str) if slitmap is not None else np.arange(nfibers).astype(str)
    telescopes = np.asarray(slitmap["telescope"]).astype(str) if slitmap is not None else np.full(nfibers, "Sci")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fibers_table = pd.DataFrame({
            "camera": paths["camera"], "fiber": np.arange(nfibers), "label": labels, "telescope": telescopes, "good": good,
            "wave_scatter": [_robust_scatter(row) for row in fits["measurements"]["wave"]["residual"]],
            "lsf_scatter": [_robust_scatter(row) for row in fits["measurements"]["lsf"]["residual"]],
            "lsf_measurements": bn.nanmedian(lsf["measurements"], axis=1),
            "lsf_pixels": bn.nanmedian(lsf["pixels"], axis=1) if "pixels" in methods else np.nan,
            "bias_measurements": bn.nanmedian(bias["measurements"], axis=1),
            "bias_pixels": bn.nanmedian(bias["pixels"], axis=1) if "pixels" in methods else np.nan,
            "shift_range": np.nanmax(np.where(weights > 0, offsets, np.nan), axis=0) - np.nanmin(np.where(weights > 0, offsets, np.nan), axis=0),
            "predicted_dfwhm2": predicted, "observed_dfwhm2": observed,
        })

    # arc spectra of the reference fiber and the selected fibers, in the first, middle and last exposures and in the sum
    camera = paths["camera"]
    reference = int(lvmarcs[0]._header.get(f"{camera} WAVE REFFIB", DEFAULT_REFERENCE_FIBER))
    if fibers is None:
        selected = _spread_fibers(fibers_table)
    else:
        selected = [int(fiber) for fiber in fibers if 0 <= int(fiber) < nfibers]
        if len(selected) < len(fibers):
            log.warning(f"{camera}: fibers outside 0-{nfibers - 1} skipped: {[f for f in fibers if not 0 <= int(f) < nfibers]}")
    spectrum_fibers = {"ref": reference, **{str(i + 1): fiber for i, fiber in enumerate(selected[:MAX_SPECTRUM_FIBERS])}}
    spectra = {exposures[k]: _arc_spectra(lvmarcs[k], spectrum_fibers) for k in sorted({0, len(lvmarcs) // 2, len(lvmarcs) - 1})}
    if sum_arc is not None:
        spectra["sum"] = _arc_spectra(sum_arc, spectrum_fibers)

    return {
        "camera": paths["camera"], "methods": methods, "narcs": len(lvmarcs), "expnums": exposures, "ref_lines": ref_lines,
        "profile_edges": edges, "products": paths, "spectra": spectra,
        "wave_lines": wave_lines, "width_single": width_single, "bias": bias, "offsets": offsets, "weights": weights, "dfwhm": dfwhm,
        "line_dflux": line_dflux, "line_dfwhm": line_dfwhm, "exposure_dflux": exposure_dflux, "exposure_info": exposure_info,
        "fits": fits, "profiles": profiles, "good": good, "fibers": fibers_table, "labels": labels, "telescopes": telescopes,
        "spectrum_fibers": spectrum_fibers, "user_fibers": fibers is not None,
    }


def _summary_row(metrics):
    """Summary statistics of a camera, over its good fibers"""
    good = metrics["good"]
    fibers = metrics["fibers"][good]
    fits = metrics["fits"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ratio = np.divide(fibers.observed_dfwhm2, fibers.predicted_dfwhm2)
        ratio = ratio[fibers.predicted_dfwhm2 > 0.01]
        row = {"camera": metrics["camera"], "narcs": metrics["narcs"], "nfibers": len(fibers),
               "lsf_measurements": np.nanmedian(fibers.lsf_measurements),
               "bias_measurements": np.nanmedian(fibers.bias_measurements), "bias_pixels": np.nanmedian(fibers.bias_pixels),
               "shift_range": np.nanmedian(fibers.shift_range), "explained_ratio": np.nanmedian(ratio) if len(ratio) else np.nan,
               "profile_measurements": metrics["profiles"]["all"]["core_rms"],
               "profile_pixels": metrics["profiles"]["sum"]["core_rms"] if "sum" in metrics["profiles"] else np.nan}
        for method in METHODS:
            for solution in ("wave", "lsf"):
                row[f"{solution}_scatter_{method}"] = _robust_scatter(fits[method][solution]["residual"][good]) if method in fits else np.nan
        products = metrics["products"]
        row["products"] = (f"{products['reused']} from the pipeline, {products['measured'] + products['previous']} measured here"
                           + (f", {len(products['missing'])} missing" if products["missing"] else "")
                           + f"; master {products['measurements']['master']}"
                           + (f"; summed arcs skipped, {products['sum_problem']}" if products.get("sum_problem") else ""))
    return row


def _layout(fig, height, xtitle, ytitle, **kwargs):
    fig.update_layout(**base_layout(height, **kwargs))
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text=ytitle)
    return style_axes(fig)


def _band(x, lo, hi, color, name):
    """Shaded band between two curves"""
    return [go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=name),
            go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=_alpha(color, 0.18),
                       hoverinfo="skip", showlegend=False, legendgroup=name)]


def _alpha(color, alpha):
    color = color.lstrip("#")
    return f"rgba({int(color[0:2], 16)},{int(color[2:4], 16)},{int(color[4:6], 16)},{alpha})"


def _r(values, decimals=4):
    """Round for the page, keeping NaN as gaps, as float32: plotly stores arrays in binary, so this halves their size"""
    return np.round(np.asarray(values, dtype=float), decimals).astype(np.float32)


def _percentiles(values, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (np.nanpercentile(values, 16, axis=axis), np.nanmedian(values, axis=axis), np.nanpercentile(values, 84, axis=axis))


SOLUTION_LABELS = {"wave": "Wavelength", "lsf": "LSF"}


def fig_fit(metrics, method, solution, view):
    """Residuals of a polynomial solution at the lines, or the solution and the lines minus each fiber's linear term

    Medians over the good fibers, with the 16-84% range across fibers as error bars and bands.
    """
    pts, good = metrics["fits"][method], metrics["good"]
    sol = pts[solution]
    color = METHOD_COLORS[method]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        xline = np.nanmedian(pts["cent"][good], axis=0)
        wline = np.nanmedian(metrics["wave_lines"][good], axis=0)
    sort = np.argsort(xline)
    fig = go.Figure()
    if view == "residuals":
        fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
        lo, med, hi = _percentiles(1000 * sol["residual"][good])
        fig.add_trace(go.Scatter(x=_r(xline[sort], 2), y=_r(med[sort], 3), mode="markers", name="median over fibers",
                                 marker=dict(color=color, size=9, line=dict(color=THEME["surface"], width=2)),
                                 error_y=dict(type="data", symmetric=False, array=_r((hi - med)[sort], 3), arrayminus=_r((med - lo)[sort], 3),
                                              color=_alpha(color, 0.5), thickness=1.5, width=0),
                                 customdata=_r(wline[sort], 2),
                                 hovertemplate="X %{x:.1f} pix, line %{customdata:.2f} &#8491;<br>%{y:+.2f} m&#8491;<extra></extra>"))
        return _layout(fig, 400, "X (pix)", f"{SOLUTION_LABELS[solution]} model &#8722; data (m&#8491;)", showlegend=False)

    # minus the linear term of each fiber's solution
    x = np.arange(sol["curves"].shape[1])
    linear = sol["intercept"][:, None] + sol["slope"][:, None] * x[None]
    step = max(1, x.size // 300)
    lo, med, hi = _percentiles((sol["curves"] - linear)[good][:, ::step])
    fig.add_traces(_band(x[::step], _r(lo), _r(hi), color, "model"))
    fig.add_trace(go.Scatter(x=x[::step], y=_r(med), mode="lines", line=dict(color=color, width=2), name="polynomial solution",
                             hovertemplate="X %{x} pix<br>%{y:.4f} &#8491;<extra>solution</extra>"))
    data_ml = sol["data"] - (sol["intercept"][:, None] + sol["slope"][:, None] * pts["cent"])
    dlo, dmed, dhi = _percentiles(data_ml[good])
    fig.add_trace(go.Scatter(x=_r(xline[sort], 2), y=_r(dmed[sort]), mode="markers", name="lines",
                             marker=dict(color=THEME["ink2"], size=8, line=dict(color=THEME["surface"], width=2)),
                             error_y=dict(type="data", symmetric=False, array=_r((dhi - dmed)[sort]), arrayminus=_r((dmed - dlo)[sort]),
                                          color=_alpha(THEME["muted"], 0.7), thickness=1.5, width=0),
                             customdata=_r(wline[sort], 2),
                             hovertemplate="X %{x:.1f} pix, line %{customdata:.2f} &#8491;<br>%{y:.4f} &#8491;<extra>lines</extra>"))
    return _layout(fig, 400, "X (pix)", f"{SOLUTION_LABELS[solution]} &#8722; linear term (&#8491;)")


def fig_residual_map(metrics, method, solution, fiber_bin=8):
    """Map of the fit residuals at each line, in bins of fibers"""
    residual = 1000 * metrics["fits"][method][solution]["residual"]
    residual = np.where(metrics["good"][:, None], residual, np.nan)
    nfibers = residual.shape[0]
    edges = np.arange(0, nfibers + fiber_bin, fiber_bin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z = np.stack([np.nanmedian(residual[lo:hi], axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
        wline = np.nanmedian(metrics["wave_lines"][metrics["good"]], axis=0)
        zmax = np.nanpercentile(np.abs(z), 98) if np.isfinite(z).any() else 1.0
    sort = np.argsort(wline)
    labels = [f"{wave:.1f}" for wave in wline[sort]]
    fig = go.Figure(go.Heatmap(
        x=labels, y=0.5 * (edges[1:] + edges[:-1]) - 0.5, z=_r(z[:, sort], 2), zmid=0, zmin=-zmax, zmax=zmax,
        colorscale=DIVERGING, meta="div",
        colorbar=dict(title=dict(text="m&#8491;", font=dict(color=THEME["ink2"])), tickfont=dict(color=THEME["muted"]), outlinewidth=0),
        hovertemplate=f"fibers %{{y:.0f}} &#177; {fiber_bin // 2}<br>line %{{x}} &#8491;<br>%{{z:+.2f}} m&#8491;<extra></extra>"))
    fig = _layout(fig, 440, "Line wavelength (&#8491;)", "Fiber", margin=dict(l=64, r=16, t=24, b=72))
    fig.update_xaxes(type="category", tickangle=-45, nticks=20, showgrid=False)
    return fig


# telescope colors: science fibers recede, the three calibration telescopes take the validated series colors
TELESCOPE_COLORS = {"Sci": THEME["muted"], "SkyE": SERIES[0], "SkyW": SERIES[1], "Spec": SERIES[2]}


def fig_coeffs(metrics, method, solution, ncols=2):
    """Polynomial coefficients of each fiber's solution, colored by telescope"""
    sol = metrics["fits"][method][solution]
    coeffs = sol["coeffs"]
    if coeffs is None:
        return None
    ncoeffs = coeffs.shape[1]
    nrows = int(np.ceil(ncoeffs / ncols))
    titles = [f"c<sub>{k}</sub> (&#8491;/pix<sup>{k}</sup>)" if k else "c<sub>0</sub> (&#8491;)" for k in range(ncoeffs)]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, horizontal_spacing=0.1, vertical_spacing=0.12 / max(nrows - 1, 1) * 2)
    fibers = np.arange(coeffs.shape[0])
    interpolated = sol["masked"]
    for k in range(ncoeffs):
        row, col = k // ncols + 1, k % ncols + 1
        for telescope, color in TELESCOPE_COLORS.items():
            for interp in (False, True):
                select = (metrics["telescopes"] == telescope) & (interpolated == interp)
                if not select.any():
                    continue
                name = telescope + (" (interpolated)" if interp else "")
                fig.add_trace(go.Scatter(
                    x=fibers[select], y=coeffs[select, k].astype(float), mode="markers", name=name, legendgroup=name, showlegend=k == 0,
                    marker=dict(color=color, size=6 if telescope == "Sci" else 8, symbol="circle-open" if interp else "circle",
                                line=dict(color=THEME["surface"] if not interp else color, width=1 if not interp else 1.5)),
                    customdata=metrics["labels"][select],
                    hovertemplate="fiber %{x} (%{customdata})<br>%{y:.6g}<extra>" + name + "</extra>"), row=row, col=col)
        fig.update_yaxes(exponentformat="power", row=row, col=col)
    fig.update_layout(**base_layout(260 * nrows + 40, margin=dict(l=72, r=16, t=96, b=52)))
    fig.update_layout(legend=dict(y=1.0, yref="container", yanchor="top"))
    fig.update_xaxes(title_text="Fiber", row=nrows)
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(color=THEME["ink2"], size=13))
    return style_axes(fig)


def fig_profile(metrics, exposure, axis):
    """Median residual of the line profile fits, by offset from the centroid, across fibers or across lines"""
    maps = metrics["profiles"].get(exposure)
    if maps is None:
        return None
    edges = metrics["profile_edges"]
    centers = 0.5 * (edges[1:] + edges[:-1])
    z = 100 * maps["by_fiber" if axis == "fiber" else "by_line"]
    if axis == "fiber":
        y, ytitle, yhover = maps["fiber_centers"], "Fiber", f"fibers %{{y:.0f}} &#177; {FIBER_BIN // 2}"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            wline = np.nanmedian(metrics["wave_lines"][metrics["good"]], axis=0)
        sort = np.argsort(wline)
        z, y = z[sort], [f"{wave:.1f}" for wave in wline[sort]]
        ytitle, yhover = "Line wavelength (&#8491;)", "line %{y} &#8491;"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        zmax = np.nanpercentile(np.abs(z), 98) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(
        x=centers, y=y, z=_r(z, 2), zmid=0, zmin=-zmax, zmax=zmax, colorscale=DIVERGING, meta="div",
        colorbar=dict(title=dict(text="% of peak", font=dict(color=THEME["ink2"])), tickfont=dict(color=THEME["muted"]), outlinewidth=0),
        hovertemplate="offset %{x:+.2f} pix<br>" + yhover + "<br>%{z:+.2f}% of peak<extra></extra>"))
    fig = _layout(fig, 460, "Offset from line centroid (pix)", ytitle, margin=dict(l=72, r=16, t=24, b=52))
    if axis == "line":
        fig.update_yaxes(type="category", nticks=20, showgrid=False)
    return fig


def fig_spectrum(metrics, exposure, key):
    """Arc spectrum of one fiber with the fitted line models, and the residuals in units of the error

    The flux axis switches between linear and arcsinh(flux / sigma), with sigma the noise of the
    fiber between the lines: linear near zero and logarithmic for bright lines.
    """
    spec = metrics["spectra"].get(exposure, {}).get(key)
    if spec is None:
        return None
    data, error, fiber, half_width = spec["data"], spec["error"], spec["fiber"], spec["half_width"]
    npixels = data.size
    pix = np.arange(npixels)
    valid = ~spec["mask"] & np.isfinite(spec["cent"]) & np.isfinite(spec["fwhm"]) & np.isfinite(spec["flux"]) & (spec["fwhm"] > 0)

    # model of each line over its fitting window; where windows overlap, the nearest line's model
    model = np.full(npixels, np.nan)
    distance = np.full(npixels, np.inf)
    for c, w, f, b in zip(spec["cent"][valid], spec["fwhm"][valid], spec["flux"][valid], spec["bkg"][valid]):
        window = np.arange(max(0, int(round(c)) - half_width), min(npixels, int(round(c)) + half_width + 1))
        window = window[np.abs(window - c) < distance[window]]
        model[window] = _line_model(window, c, w, f, b)
        distance[window] = np.abs(window - c)
    in_window = np.isfinite(model)
    keep = in_window | (pix % SPECTRUM_THINNING == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        chi = (data - model) / error
    # model pixels only, with a gap where one window ends and the next one starts
    xw = pix[in_window]
    breaks = np.flatnonzero(np.diff(xw) > 1) + 1
    xm = np.insert(xw.astype(float), breaks, np.nan)
    ym = np.insert(model[in_window], breaks, np.nan)
    peaks = _line_model(np.round(spec["cent"][valid]), spec["cent"][valid], spec["fwhm"][valid], spec["flux"][valid], spec["bkg"][valid])
    wave_curve = metrics["fits"]["measurements"]["wave"]["curves"][fiber]
    line_waves = np.interp(spec["cent"][valid], pix, wave_curve)

    # arcsinh stretch with the noise between the lines
    between = data[~in_window & np.isfinite(data)]
    scale = _robust_scatter(between) if between.size > 10 else np.nan
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    stretched = {name: _r(np.arcsinh(values / scale), 4) for name, values in (("model", ym), ("data", data[keep]), ("peaks", peaks))}
    finite = data[np.isfinite(data)]
    tickvals, ticktext = _asinh_ticks(scale, finite.min() if finite.size else 0.0, finite.max() if finite.size else 1.0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    # traces 0-2 carry the flux and are switched by the scale buttons
    fig.add_trace(go.Scatter(x=xm.astype(np.float32), y=_r(ym, 0), mode="lines", name="fitted lines",
                             line=dict(color=_alpha(METHOD_COLORS["measurements"], 0.45), width=4), connectgaps=False,
                             hovertemplate="X %{x} pix<br>%{y:.0f}<extra>model</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pix[keep].astype(np.int16), y=_r(data[keep], 0), mode="lines", name="arc spectrum",
                             line=dict(color=THEME["ink2"], width=1), hovertemplate="X %{x} pix<br>%{y:.0f}<extra>data</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=_r(spec["cent"][valid], 2), y=_r(peaks, 1), mode="markers", name="line centroids",
                             marker=dict(color=METHOD_COLORS["measurements"], size=7, line=dict(color=THEME["surface"], width=1.5)),
                             customdata=np.stack([_r(spec["waves"][valid], 3), _r(line_waves, 3), _r(spec["fwhm"][valid], 3)], axis=1),
                             hovertemplate="line %{customdata[0]} &#8491; at X %{x:.2f} pix<br>solution %{customdata[1]} &#8491;"
                                           "<br>FWHM %{customdata[2]} pix<extra></extra>"), row=1, col=1)
    rejected = spec["mask"] & np.isfinite(spec["expected"])
    if rejected.any():
        fig.add_trace(go.Scatter(x=_r(spec["expected"][rejected], 2), y=np.zeros(rejected.sum(), dtype=np.float32), mode="markers",
                                 name="rejected lines",
                                 marker=dict(color=METHOD_COLORS["pixels"], size=9, symbol="x-thin", line=dict(color=METHOD_COLORS["pixels"], width=2)),
                                 customdata=_r(spec["waves"][rejected], 3),
                                 hovertemplate="line %{customdata} &#8491; rejected<br>expected at X %{x:.1f} pix<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1), row=2, col=1)
    fig.add_trace(go.Scatter(x=pix[in_window].astype(np.int16), y=_r(chi[in_window], 1), mode="markers", name="residuals", showlegend=False,
                             marker=dict(color=METHOD_COLORS["measurements"], size=3),
                             hovertemplate="X %{x} pix<br>%{y:+.1f} &#963;<extra>residual</extra>"), row=2, col=1)

    # the same flux traces stretched by arcsinh, hidden until selected: the buttons only switch visibility and ticks
    nlinear = len(fig.data)
    fig.add_trace(go.Scatter(x=xm.astype(np.float32), y=stretched["model"], mode="lines", name="fitted lines", visible=False,
                             line=dict(color=_alpha(METHOD_COLORS["measurements"], 0.45), width=4), connectgaps=False,
                             hovertemplate="X %{x} pix<extra>model</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pix[keep].astype(np.int16), y=stretched["data"], mode="lines", name="arc spectrum", visible=False,
                             line=dict(color=THEME["ink2"], width=1), hovertemplate="X %{x} pix<extra>data</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=_r(spec["cent"][valid], 2), y=stretched["peaks"], mode="markers", name="line centroids", visible=False,
                             marker=dict(color=METHOD_COLORS["measurements"], size=7, line=dict(color=THEME["surface"], width=1.5)),
                             customdata=fig.data[2].customdata, hovertemplate=fig.data[2].hovertemplate), row=1, col=1)
    flux_linear, flux_asinh = [0, 1, 2], [nlinear, nlinear + 1, nlinear + 2]
    visible_linear = [i not in flux_asinh for i in range(len(fig.data))]
    visible_asinh = [i not in flux_linear for i in range(len(fig.data))]
    linear = [{"visible": visible_linear}, {"yaxis.tickmode": "auto", "yaxis.title.text": "Flux"}]
    asinh = [{"visible": visible_asinh},
             {"yaxis.tickmode": "array", "yaxis.tickvals": tickvals, "yaxis.ticktext": ticktext, "yaxis.title.text": "Flux (arcsinh)"}]
    fig.update_layout(**base_layout(560, margin=dict(l=72, r=16, t=48, b=52)))
    fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=1, xanchor="right", y=1.08, yanchor="bottom",
                                        bgcolor=THEME["surface"], bordercolor=THEME["axis"], font=dict(color=THEME["ink2"], size=12),
                                        buttons=[dict(label="Linear", method="update", args=linear),
                                                 dict(label="arcsinh", method="update", args=asinh)])])
    fig.update_yaxes(title_text="Flux", row=1, col=1)
    fig.update_yaxes(title_text="(data &#8722; model) / &#963;", row=2, col=1)
    fig.update_xaxes(title_text="X (pix)", row=2, col=1)
    name = "Reference fiber" if key == "ref" else "Fiber"
    fig.update_layout(title=dict(text=f"{name} {fiber} ({metrics['labels'][fiber]}, {metrics['telescopes'][fiber]})",
                                 font=dict(color=THEME["ink2"], size=13), x=0.01, xanchor="left", y=0.98))
    return style_axes(fig)


def fig_bias(metrics):
    """Master LSF minus the line widths of single exposures, median over fibers, for each line"""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = bn.nanmedian(metrics["wave_lines"][metrics["good"]], axis=0)
        sort = np.argsort(x)
        for method in metrics["methods"]:
            lo, y, hi = _percentiles(1000 * metrics["bias"][method][metrics["good"]])
            fig.add_trace(go.Scatter(x=_r(x[sort], 2), y=_r(y[sort], 2), mode="markers+lines", name=METHOD_LABELS[method],
                                     line=dict(color=METHOD_COLORS[method], width=2), marker=dict(size=8, line=dict(color=THEME["surface"], width=2)),
                                     error_y=dict(type="data", symmetric=False, array=_r((hi - y)[sort], 2), arrayminus=_r((y - lo)[sort], 2),
                                                  color=_alpha(METHOD_COLORS[method], 0.5), thickness=1.5, width=0),
                                     hovertemplate="%{x:.2f} &#8491;<br>%{y:+.2f} m&#8491;<extra>" + METHOD_LABELS[method] + "</extra>"))
    return _layout(fig, 380, "Line wavelength (&#8491;)", "LSF &#8722; single-exposure FWHM (m&#8491;)")


def _fig_per_exposure(metrics, values, ytitle):
    """One line per exposure across the slit, lighter for earlier exposures"""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    narcs = values.shape[0]
    ramp = [EXPOSURE_RAMP[int(round(i * (len(EXPOSURE_RAMP) - 1) / max(narcs - 1, 1)))] for i in range(narcs)]
    fibers = np.arange(values.shape[1])
    for k in range(narcs):
        fig.add_trace(go.Scatter(x=fibers, y=_r(values[k]), mode="lines", line=dict(color=ramp[k], width=1.5), name=metrics["expnums"][k],
                                 customdata=metrics["labels"],
                                 hovertemplate="fiber %{x} (%{customdata})<br>%{y:+.3f} pix<extra>exposure " + metrics["expnums"][k] + "</extra>"))
    return _layout(fig, 400, "Fiber", ytitle, legend=dict(title=dict(text="Exposure", font=dict(color=THEME["ink2"])),
                                                          orientation="v", x=1.01, y=1, xanchor="left", yanchor="top",
                                                          font=dict(color=THEME["ink2"]), bgcolor="rgba(0,0,0,0)"),
                   margin=dict(l=64, r=120, t=24, b=52))


def fig_shifts(metrics):
    """Line position offset of each exposure relative to the median of the exposures, per fiber"""
    return _fig_per_exposure(metrics, np.where(metrics["weights"] > 0, metrics["offsets"], np.nan), "Offset along dispersion (pix)")


def fig_fwhm_changes(metrics):
    """Line FWHM of each exposure relative to the median of the exposures, per fiber"""
    fig = _fig_per_exposure(metrics, metrics["dfwhm"], "FWHM &#8722; median of exposures (pix)")
    fig.update_yaxes(exponentformat="none")
    return fig


TREND_QUANTITIES = {"fwhm": ("FWHM change", "FWHM &#8722; median of exposures (pix)", "%{y:+.3f} pix"),
                     "flux": ("Line flux change", "Flux / median of exposures &#8722; 1 (%)", "%{y:+.1f}%"),
                     "offset": ("Line offset", "Offset along dispersion (pix)", "%{y:+.3f} pix")}


def _exposure_values(metrics, quantity):
    """Median over good fibers of the FWHM change, flux change (%) or line offset of each exposure"""
    if quantity == "flux":
        return metrics["exposure_dflux"]
    values = metrics["dfwhm"] if quantity == "fwhm" else np.where(metrics["weights"] > 0, metrics["offsets"], np.nan)
    good = metrics["good"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values[:, good], axis=1) if good.any() else np.full(values.shape[0], np.nan)


def fig_trends(all_metrics, quantity, night="all"):
    """Median over fibers of the FWHM, flux or position change of each exposure against time, with the lab temperature

    Parameters
    ----------
    all_metrics : list[dict]
        Metrics of each camera
    quantity : str
        One of :data:`TREND_QUANTITIES`: "fwhm", "flux" or "offset"
    night : int or str
        SJD of the night to show, or "all"

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dashes = {"1": "solid", "2": "dash", "3": "dot"}
    _, ytitle, yhover = TREND_QUANTITIES[quantity]
    info = None
    for metrics in all_metrics:
        camera, info = metrics["camera"], metrics["exposure_info"]
        select = np.ones(len(info), dtype=bool) if night == "all" else (info.night == int(night)).to_numpy()
        info, median = info[select], _exposure_values(metrics, quantity)[select]
        x = info.obstime if info.obstime.notna().all() else info.expnum
        fig.add_trace(go.Scatter(x=x, y=_r(median), mode="lines+markers", name=camera, legendgroup=camera[0],
                                 line=dict(color=CHANNEL_COLORS[camera[0]], width=1.5, dash=dashes.get(camera[1], "solid")), marker=dict(size=6),
                                 customdata=np.stack([info.expnum, info.ccdtemp], axis=1),
                                 hovertemplate="exposure %{customdata[0]}<br>" + yhover + "<br>sensor %{customdata[1]:.1f} &#176;C<extra>" + camera + "</extra>"),
                      secondary_y=False)
    if info is not None and np.isfinite(info.labtemp).any():
        x = info.obstime if info.obstime.notna().all() else info.expnum
        fig.add_trace(go.Scatter(x=x, y=_r(info.labtemp, 2), mode="lines+markers", name="lab temperature",
                                 line=dict(color=THEME["muted"], width=1.5, dash="dash"), marker=dict(size=5, symbol="square"),
                                 hovertemplate="%{y:.2f} &#176;C<extra>lab temperature</extra>"), secondary_y=True)
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1), secondary_y=False)
    fig = _layout(fig, 420, "Exposure time (UTC)", ytitle, margin=dict(l=84, r=64, t=40, b=52))
    fig.update_yaxes(title_text="Lab temperature (&#176;C)", secondary_y=True, showgrid=False)
    fig.update_yaxes(exponentformat="none")
    return fig


def fig_flux_widths(metrics):
    """FWHM change against flux change of each line in each exposure, with a straight line fitted to all of them

    A brightness effect moves the points along a common slope, line by line; a focus change shifts all the lines of an
    exposure together, whatever their flux.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    fig.add_vline(x=0, line=dict(color=THEME["axis"], width=1))
    dflux, dfwhm = metrics["line_dflux"], metrics["line_dfwhm"]
    narcs = dflux.shape[0]
    ramp = [EXPOSURE_RAMP[int(round(i * (len(EXPOSURE_RAMP) - 1) / max(narcs - 1, 1)))] for i in range(narcs)]
    waves = np.round(np.asarray(metrics["ref_lines"], dtype=float), 2)
    for k in range(narcs):
        ok = np.isfinite(dflux[k]) & np.isfinite(dfwhm[k])
        fig.add_trace(go.Scatter(x=_r(dflux[k][ok], 3), y=_r(dfwhm[k][ok]), mode="markers", name=metrics["expnums"][k],
                                 marker=dict(color=ramp[k], size=7, line=dict(color=THEME["surface"], width=1)), customdata=waves[ok],
                                 hovertemplate="%{customdata:.2f} &#8491;<br>flux %{x:+.1f}%<br>FWHM %{y:+.4f} pix<extra>exposure "
                                               + metrics["expnums"][k] + "</extra>"))
    # a slope needs flux changes well above the noise of the fluxes
    ok = np.isfinite(dflux) & np.isfinite(dfwhm)
    if ok.sum() >= 3 and np.ptp(dflux[ok]) >= 1:
        slope, intercept = np.polyfit(dflux[ok], dfwhm[ok], 1)
        x = np.array([dflux[ok].min(), dflux[ok].max()])
        fig.add_trace(go.Scatter(x=_r(x, 2), y=_r(intercept + slope * x), mode="lines", line=dict(color=THEME["muted"], width=1.5, dash="dash"),
                                 name=f"fit: {10 * slope:+.4f} pix per 10% flux", hoverinfo="skip"))
    fig = _layout(fig, 420, "Flux / median of exposures &#8722; 1 (%)", "FWHM &#8722; median of exposures (pix)",
                  legend=dict(title=dict(text="Exposure", font=dict(color=THEME["ink2"])), orientation="v", x=1.01, y=1, xanchor="left",
                              yanchor="top", font=dict(color=THEME["ink2"]), bgcolor="rgba(0,0,0,0)"),
                  margin=dict(l=64, r=200, t=24, b=52))
    fig.update_yaxes(exponentformat="none")
    return fig


def fig_explained(all_metrics):
    """Observed broadening of the lines in the sum against the broadening predicted from the shifts, None without sums"""
    if not any("pixels" in metrics["methods"] for metrics in all_metrics):
        return None
    fig = go.Figure()
    top = 0.0
    for channel in "brz":
        frames = [m["fibers"][m["fibers"].good] for m in all_metrics if m["camera"][0] == channel]
        if not frames:
            continue
        fibers = pd.concat(frames)
        fibers = fibers[np.isfinite(fibers.predicted_dfwhm2) & np.isfinite(fibers.observed_dfwhm2)]
        if fibers.empty:
            continue
        top = max(top, np.nanpercentile(np.concatenate([fibers.predicted_dfwhm2, fibers.observed_dfwhm2]), 99.5))
        fig.add_trace(go.Scatter(x=_r(fibers.predicted_dfwhm2), y=_r(fibers.observed_dfwhm2), mode="markers", name=f"{channel} cameras",
                                 marker=dict(color=CHANNEL_COLORS[channel], size=8, opacity=0.6, line=dict(color=THEME["surface"], width=1)),
                                 customdata=np.stack([fibers.camera, fibers.fiber, fibers.label], axis=1),
                                 hovertemplate="%{customdata[0]} fiber %{customdata[1]} (%{customdata[2]})<br>predicted %{x:.3f} pix&#178;"
                                               "<br>observed %{y:.3f} pix&#178;<extra></extra>"))
    top = top if np.isfinite(top) and top > 0 else 1.0
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=THEME["muted"], width=1.5, dash="dash"),
                             name="observed = predicted", hoverinfo="skip"))
    return _layout(fig, 460, "Predicted from the shifts, 8 ln2 &#963;&#178;<sub>shift</sub> (pix&#178;)",
                   "Observed FWHM&#178;<sub>sum</sub> &#8722; FWHM&#178;<sub>single</sub> (pix&#178;)")


EXPLAINED_HTML = """<p>For each fiber, how much wider the lines are in the sum than in single exposures, against the broadening
    expected from summing Gaussians shifted by those offsets. Points along the dashed line mean the broader LSF of the sum is
    entirely due to the shifts between exposures.</p>
    <div class="chart" data-fig="fig-explained" role="img" aria-label="Observed against predicted line broadening per fiber"></div>"""


def _fmt(value, fmt):
    return "&ndash;" if value is None or not np.isfinite(value) else format(value, fmt)


def write_report(all_metrics, report_path, mjd_epoch, kind, mjds, problems=None):
    """Write the dashboard

    Parameters
    ----------
    all_metrics : list[dict]
        Metrics of each camera, see :func:`camera_metrics`
    report_path : str
        Path of the HTML dashboard
    mjd_epoch : int
        MJD of the calibration epoch
    kind : str
        Calibration sequence kind
    mjds : list[int]
        MJDs from which the arcs are sourced
    problems : dict[str, str], optional
        Cameras that could not be compared and why

    Returns
    -------
    pd.DataFrame
        Summary of each camera
    """
    problems = problems or {}
    summary = pd.DataFrame([_summary_row(metrics) for metrics in all_metrics])
    cameras = [metrics["camera"] for metrics in all_metrics]

    has_sum = any("pixels" in metrics["methods"] for metrics in all_metrics)
    figures = {"fig-explained": fig_explained(all_metrics)} if has_sum else {}
    nights = sorted({int(night) for metrics in all_metrics for night in metrics["exposure_info"].night})
    for quantity in TREND_QUANTITIES if all_metrics else []:
        for night in nights + ["all"]:
            figures[f"trends|{quantity}|{night}"] = fig_trends(all_metrics, quantity, night)
    for metrics in all_metrics:
        camera = metrics["camera"]
        figures[f"bias|{camera}"] = fig_bias(metrics)
        figures[f"shifts|{camera}"] = fig_shifts(metrics)
        figures[f"fwhms|{camera}"] = fig_fwhm_changes(metrics)
        figures[f"fluxwidths|{camera}"] = fig_flux_widths(metrics)
        for solution in ("wave", "lsf"):
            for method in metrics["methods"]:
                for view in ("residuals", "linear"):
                    figures[f"fit|{camera}|{solution}|{method}|{view}"] = fig_fit(metrics, method, solution, view)
                figures[f"rmap|{camera}|{solution}|{method}"] = fig_residual_map(metrics, method, solution)
                coeffs = fig_coeffs(metrics, method, solution)
                if coeffs is not None:
                    figures[f"coeffs|{camera}|{solution}|{method}"] = coeffs
        for exposure in metrics["spectra"]:
            for key in metrics["spectrum_fibers"]:
                spectrum = fig_spectrum(metrics, exposure, key)
                if spectrum is not None:
                    figures[f"spectrum|{camera}|{exposure}|{key}"] = spectrum
        for exposure in metrics["profiles"]:
            for axis in ("fiber", "line"):
                profile = fig_profile(metrics, exposure, axis)
                if profile is not None:
                    figures[f"profile|{camera}|{exposure}|{axis}"] = profile

    camera_dim = ("Camera", [(camera, camera) for camera in cameras])
    solution_dim = ("Solution", [("wave", "Wavelength"), ("lsf", "LSF")])
    method_dim = ("Combination", [(method, METHOD_LABELS[method]) for method in METHODS if method == "measurements" or has_sum])
    exposures = all_metrics[0]["expnums"] if all_metrics else []
    exposure_dim = ("Exposure", [("all", "All exposures")] + [(expnum, expnum) for expnum in exposures]
                    + ([("sum", "Sum of exposures")] if has_sum else []))
    # arc spectra: the reference fiber first, then the selected fibers (the same numbers in every camera if given)
    first = all_metrics[0] if all_metrics else None
    spectrum_dim = ("Exposure", [(exposure, "Sum of exposures" if exposure == "sum" else exposure) for exposure in (first["spectra"] if first else [])])
    fiber_options = [("ref", "Reference fiber")]
    if first is not None:
        for key, fiber in first["spectrum_fibers"].items():
            if key != "ref":
                position = SPECTRUM_POSITIONS[int(key) - 1]
                fiber_options.append((key, f"Fiber {fiber}" if first["user_fibers"] else f"{100 * position:g}% along the slit"))
    fiber_dim = ("Fiber", fiber_options)

    # tiles: quality of the per-arc master solutions first, then the combination
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tiles = [
            ("Wavelength fit scatter", f"{1000 * np.nanmedian(summary.wave_scatter_measurements):.2f} mÅ",
             "robust scatter of the residuals at the lines, median over cameras"),
            ("LSF fit scatter", f"{1000 * np.nanmedian(summary.lsf_scatter_measurements):.2f} mÅ",
             "robust scatter of the residuals at the lines, median over cameras"),
            ("Line profile residual", f"{100 * np.nanmedian(summary.profile_measurements):.2f}%",
             f"RMS within ±{CORE_HALF_WIDTH:g} pix of the centroid, of the peak"),
            ("LSF bias vs single exposures",
             f"{1000 * np.nanmedian(summary.bias_measurements):+.1f} / {1000 * np.nanmedian(summary.bias_pixels):+.1f} mÅ" if has_sum
             else f"{1000 * np.nanmedian(summary.bias_measurements):+.1f} mÅ",
             "per-arc / summed arcs, median over cameras" if has_sum else "per-arc, median over cameras"),
            ("Shift range between exposures", f"{np.nanmedian(summary.shift_range):.3f} pix", "median over fibers of max - min offset"),
        ]
        if has_sum:
            tiles.append(("Broadening explained by the shifts", f"{np.nanmedian(summary.explained_ratio):.2f}", "observed / predicted, 1 = fully explained"))

    columns = [
        ("Camera", ""), ("Arcs", "Number of arc exposures"), ("Fibers", "Fibers with solutions in both methods"),
        ("Wave scatter (mÅ)", "Robust scatter of the wavelength fit residuals, per-arc / summed arcs"),
        ("LSF scatter (mÅ)", "Robust scatter of the LSF fit residuals, per-arc / summed arcs"),
        ("Profile residual", "RMS of the median line profile residual within the core, per-arc / summed arcs"),
        ("LSF (Å)", "Median master LSF FWHM, per-arc measurements"),
        ("LSF bias (mÅ)", "Median of master LSF - single-exposure line width, per-arc / summed arcs"),
        ("Shift range (pix)", "Median over fibers of the range of the exposure offsets"),
        ("Obs./pred.", "Median ratio of observed to predicted broadening of the sum"),
        ("Products", "lvmArc products reused from the pipeline or measured for this dashboard, and where the master solutions come from"),
    ]
    rows = [[escape(r.camera), f"{r.narcs}", f"{r.nfibers}",
             f"{_fmt(1000 * r.wave_scatter_measurements, '.2f')} / {_fmt(1000 * r.wave_scatter_pixels, '.2f')}",
             f"{_fmt(1000 * r.lsf_scatter_measurements, '.2f')} / {_fmt(1000 * r.lsf_scatter_pixels, '.2f')}",
             f"{_fmt(100 * r.profile_measurements, '.2f')}% / {_fmt(100 * r.profile_pixels, '.2f')}%",
             _fmt(r.lsf_measurements, ".4f"),
             f"{_fmt(1000 * r.bias_measurements, '+.1f')} / {_fmt(1000 * r.bias_pixels, '+.1f')}",
             _fmt(r.shift_range, ".3f"), _fmt(r.explained_ratio, ".2f"), f'<span class="range">{escape(r.products)}</span>']
            for r in summary.itertuples()]
    rows += [[escape(camera), "&ndash;", "&ndash;"] + [""] * (len(columns) - 4) + [f'<span class="range">{escape(problem)}</span>']
             for camera, problem in problems.items()]
    summary_table = html_table(columns, rows, numeric=[False] + [True] * (len(columns) - 2) + [False])

    fibers_table = pd.concat([metrics["fibers"] for metrics in all_metrics]) if all_metrics else pd.DataFrame()
    fiber_rows = [[escape(r.camera), f"{r.fiber}", escape(str(r.label)), escape(str(r.telescope)),
                   _fmt(1000 * r.wave_scatter, ".2f"), _fmt(1000 * r.lsf_scatter, ".2f"), _fmt(r.lsf_measurements, ".4f"),
                   _fmt(1000 * r.bias_measurements, "+.1f"), _fmt(1000 * r.bias_pixels, "+.1f"), _fmt(r.shift_range, ".3f")]
                  for r in fibers_table[fibers_table.good].itertuples()] if len(fibers_table) else []
    fiber_table_html = html_table([("Camera", ""), ("Fiber", ""), ("Label", ""), ("Telescope", ""), ("Wave scatter (mÅ)", ""),
                                   ("LSF scatter (mÅ)", ""), ("LSF (Å)", ""), ("Bias per-arc (mÅ)", ""),
                                   ("Bias summed (mÅ)", ""), ("Shift range (pix)", "")],
                                  fiber_rows, numeric=[False, True, False, False] + [True] * 6)

    sections = [
        f"""<section>
    <h2>Summary by camera</h2>
    <p>Values are per-arc / summed arcs where both are shown. Scatters are robust (1.4826 MAD) over all good fibers and
    lines. A positive LSF bias means the master LSF is broader than the lines of a single exposure. <em>Products</em> says
    how many lvmArc products came from the pipeline and how many were measured for this dashboard.</p>
    {summary_table}
  </section>""",
        f"""<section>
    <h2>Arc spectra and fitted lines</h2>
    <p>Continuum-subtracted arc spectrum of one fiber with the fitted model of each line over its fitting window, and the
    residuals in units of the error below, for the first, middle and last exposures. The reference fiber of the line
    measurements is shown first; the other fibers are spread along the slit unless chosen with <span class="mono">--fibers</span>.
    The chart names the fiber. Rejected lines are marked with a cross where the other fibers place them. Between lines, one
    pixel in {SPECTRUM_THINNING} is drawn. <em>arcsinh</em> stretches the flux as asinh(flux / &sigma;), with &sigma; the noise of the
    fiber between lines: linear near zero and logarithmic for bright lines.</p>
    {picker("fig-spectrum", "spectrum", [camera_dim, spectrum_dim, fiber_dim], 560)}
  </section>""",
        f"""<section>
    <h2>Line profile residuals</h2>
    <p>Data minus the fitted Gaussian around each line, as a fraction of the line peak, against the offset from the
    centroid. Across fibers, the residuals of all lines are combined in groups of {FIBER_BIN} fibers; across lines, those of all
    fibers, over the window each line was fitted in. A symmetric pattern means the line profile is not Gaussian; an
    antisymmetric one means it is skewed.{" The sum of exposures shows the profile of the arcs summed in pixels." if has_sum else ""}</p>
    {picker("fig-profile", "profile", [camera_dim, exposure_dim, ("Across", [("fiber", "Fibers"), ("line", "Lines")])], 460)}
  </section>""",
        f"""<section>
    <h2>Polynomial fits</h2>
    <p>Each fiber's wavelength or LSF polynomial at the measured lines. <em>Residuals</em> shows model minus data;
    <em>minus linear term</em> shows the solution and the lines after removing a straight line fitted to each fiber's
    solution, which leaves the curvature the polynomial has to describe. Medians over fibers, with the 16&ndash;84% range
    across fibers as bands and error bars.</p>
    {picker("fig-fit", "fit", [camera_dim, solution_dim, method_dim, ("View", [("residuals", "Residuals"), ("linear", "Minus linear term")])], 400)}
  </section>""",
        f"""<section>
    <h2>Fit residuals across the slit</h2>
    <p>Model minus data at each line, median over groups of 8 fibers. Structure along the fibers, or a line that is off
    in every fiber, points to a polynomial that is too stiff or a misidentified line.</p>
    {picker("fig-rmap", "rmap", [camera_dim, solution_dim, method_dim], 440)}
  </section>""",
        f"""<section>
    <h2>Polynomial coefficients</h2>
    <p>Coefficients of each fiber's polynomial, in powers of the pixel coordinate. They should vary smoothly along the slit.
    Open circles are fibers whose solution was interpolated from their neighbors.</p>
    {picker("fig-coeffs", "coeffs", [camera_dim, solution_dim, method_dim], 860)}
  </section>""",
        f"""<section>
    <h2>Combining exposures</h2>
    <p>Master LSF {"of both combinations " if has_sum else ""}minus the FWHM of each line in single exposures (median over
    exposures), median over fibers with the 16&ndash;84% range as error bars. Single exposures carry the LSF a science exposure
    has, so a good LSF stays close to zero.</p>
    {picker("fig-bias", "bias", [camera_dim], 380)}
    <p>Offset of each exposure's line positions from the median of all exposures along the dispersion axis, per fiber
    (median over lines). The per-arc combination removes these before combining the line positions{"; the sum is blurred by them" if has_sum else ""}.
    Lighter lines are earlier exposures.</p>
    {picker("fig-shifts", "shifts", [camera_dim], 400)}
    <p>Line FWHM of each exposure minus the median of all exposures, per fiber (median over lines). Changes common to the whole
    slit follow the spectrograph (e.g. focus with temperature); changes along the slit follow the fibers. Lighter lines are
    earlier exposures.</p>
    {picker("fig-fwhms", "fwhms", [camera_dim], 400)}
    <p>The lamp brightness changes between exposures too, and brighter lines can be wider. Each point is one line in one
    exposure: its FWHM change against its flux change, both relative to the median of the exposures and median over fibers.
    A brightness effect lines the points up along the fitted slope, across lines and exposures; a focus change moves all the
    lines of an exposure up or down together, whatever their flux. The slope is fitted when the fluxes change by at least 1%.</p>
    {picker("fig-fluxwidths", "fluxwidths", [camera_dim], 420)}
    <p>Median over fibers of the FWHM change, flux change or line offset of each exposure, against the time of the exposure,
    with the lab temperature (right axis), one night at a time. Line styles tell the spectrographs apart: solid, dashed and
    dotted for 1, 2 and 3; hover for the sensor temperature.</p>
    {picker("fig-trends", "trends", [("Quantity", [(key, label) for key, (label, _, _) in TREND_QUANTITIES.items()]),
                                     ("Night", [(str(night), str(night)) for night in nights] + [("all", "All nights")])], 420)}
    {EXPLAINED_HTML if has_sum else ""}
  </section>""",
        f"""<section>
    <details><summary>Table view: all fibers</summary>{fiber_table_html}</details>
  </section>""",
        DEFINITIONS,
    ]

    meta = [("Epoch", mjd_epoch), ("Sequence", kind), ("Arc MJDs", ", ".join(map(str, mjds))), ("Cameras", ", ".join(cameras)),
            ("Pipeline", drpver), ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]
    # six summary tiles read best as two rows of three
    intro = ("<style>.tiles { grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); }</style>"
             "<p>Quality of the master wavelength and LSF solutions of the epoch, fitted from lines measured in each arc "
             "exposure" + (", and how they compare with fitting the arcs summed in pixels" if has_sum else "") + ". The reference "
             "for the LSF is the width of the lines in single exposures, which is the LSF a science exposure has. The equations "
             'are listed under <a href="#definitions">How the quantities are computed</a>.</p>')
    write_dashboard(report_path, f"Wavelength and LSF, epoch {mjd_epoch}", f"LVM DRP · wavelength · {kind} arcs · {drpver}",
                    intro, meta, tiles, "\n\n  ".join(sections), figures, "lvmdrp.qa.wavelengths.qa_arc_combinations")
    return summary


def qa_arc_combinations(mjd_epoch, cameras=CAMERAS, kind="longterm", epochs=None, output_dir=None, with_sum=False,
                        fibers=None, extract=False, nprocs=1, skip_done=True, dry_run=False):
    """Write the wavelength and LSF QA dashboard of a calibration epoch

    For each camera:

    1. Find the arc exposures of the master solution from its FRAMES extension, or, without a
       master, from the epoch's arc sequence (as in :func:`~lvmdrp.functions.run_calseq.create_wavelengths`).
    2. Use the lvmArc products and master solutions of the pipeline, measuring only the arcs
       without an lvmArc and fitting a missing master (:func:`compare_camera`). With `with_sum`,
       the arcs are also summed in pixels and measured once. Everything measured goes to
       `output_dir`; the pipeline products are never written.
    3. Measure the quality of the solutions and how the combinations compare (:func:`camera_metrics`).

    Then write the dashboard and a per-fiber table (:func:`write_report`).

    Parameters
    ----------
    mjd_epoch : int
        MJD of the calibration epoch
    cameras : iterable[str], optional
        Cameras to check, by default all
    kind : str, optional
        Arc sequence, either 'longterm' or 'nightly', by default 'longterm'
    epochs : dict, optional
        Calibration epochs as returned by :func:`~lvmdrp.functions.run_calseq.load_epochs_file`, used only for
        cameras without a master solution. If omitted, they are loaded from the default calibration epochs file
    output_dir : str, optional
        Directory of the dashboard and of anything measured for it. Default is ``wave_qa/{mjd_epoch}_{kind}_arcs``
        in the epoch's ancillary directory
    with_sum : bool, optional
        Also compare with the arcs summed in pixels, which takes one measurement per camera, by default False
    fibers : list[int], optional
        Up to four fibers whose arc spectra are shown next to the reference fiber. By default, fibers spread along the slit
    extract : bool, optional
        Detrend and extract the arcs needed and not extracted yet, by default False
    nprocs : int, optional
        Number of processes measuring arcs, by default 1
    skip_done : bool, optional
        Reuse the products already in `output_dir`, by default True
    dry_run : bool, optional
        Only log the products found and the output paths, by default False

    Returns
    -------
    dict
        ``"summary"`` (pd.DataFrame), ``"report_path"`` and ``"table_path"``
    """
    from lvmdrp.functions.run_calseq import load_epochs_file, get_arc_sequence, extract_arcs, get_calib_paths
    from lvmdrp.main import reduce_2d

    if fibers is not None and len(fibers) > MAX_SPECTRUM_FIBERS:
        raise ValueError(f"at most {MAX_SPECTRUM_FIBERS} fibers can be shown besides the reference fiber, got {len(fibers)}")
    cameras = [camera for camera in CAMERAS if camera in cameras]
    output_dir = output_dir or _qa_dir(mjd_epoch, f"{mjd_epoch}_{kind}_arcs")
    report_path = os.path.join(output_dir, f"wave-qa_{mjd_epoch}_{kind}.html")
    table_path = os.path.join(output_dir, f"wave-qa_{mjd_epoch}_{kind}.csv")

    sequence = {}

    def arc_sequence():
        # metadata of the epoch's arc sequence, loaded only when needed
        if not sequence:
            epochs_ = epochs if epochs is not None else load_epochs_file(epochs_kind="calibration", verbose=False)
            sequence["mjds"], sequence["frames"], sequence["expnums"] = get_arc_sequence(mjd=mjd_epoch, epochs=epochs_, kind=kind)
        return sequence

    arcs_by_camera, sources, problems = {}, {}, {}
    for camera in cameras:
        masters = master_paths(camera, mjd_epoch, kind)
        arcs = master_arcs(masters["wave"])
        sources[camera] = "master FRAMES"
        if arcs is None:
            frames = arc_sequence()["frames"]
            frames = frames[frames.camera == camera] if not frames.empty else frames
            arcs = [{"tileid": int(row.tileid), "mjd": int(row.mjd), "expnum": int(row.expnum), "camera": camera} for row in frames.itertuples()]
            sources[camera] = "epoch arc sequence"
        if not arcs:
            problems[camera] = "no arcs found in the master solution or the epoch's arc sequence"
            continue
        arcs_by_camera[camera] = (arcs, masters)
    mjds = sorted({arc["mjd"] for arcs, _ in arcs_by_camera.values() for arc in arcs})

    def needs_extraction(arc):
        products = arc_products(arc)
        return not os.path.isfile(products["xarc"]) and (with_sum or not os.path.isfile(products["lvmarc"]))

    if dry_run:
        log.info(f"dry run of '{qa_arc_combinations.__name__}' for epoch {mjd_epoch} ({kind}), arcs from MJDs {mjds}")
        for camera, (arcs, masters) in arcs_by_camera.items():
            products = [arc_products(arc) for arc in arcs]
            nlvmarc = sum(os.path.isfile(p["lvmarc"]) for p in products)
            nxarc = sum(os.path.isfile(p["xarc"]) for p in products)
            master = "found" if all(os.path.isfile(p) for p in masters.values()) else "missing, will be fitted"
            log.info(f"  {camera}: {len(arcs)} arcs from the {sources[camera]} (exposures {', '.join(str(a['expnum']) for a in arcs)}), "
                     f"{nlvmarc} lvmArc products, {nxarc} extracted arcs, master {master}"
                     + (f", sum of {nxarc} arcs" if with_sum else ""))
        for camera, problem in problems.items():
            log.info(f"  {camera}: {problem}")
        log.info(f"  dashboard: {report_path}")
        log.info(f"  table: {table_path}")
        return {"summary": None, "report_path": report_path, "table_path": table_path}

    to_extract = {camera: [arc["expnum"] for arc in arcs if needs_extraction(arc)] for camera, (arcs, _) in arcs_by_camera.items()}
    if extract and any(to_extract.values()):
        seq = arc_sequence()
        calibs = get_calib_paths(mjd=mjd_epoch, version=drpver, flavors=CALIBRATION_NEEDS["wave"], from_sandbox=False)
        expnums = sorted({expnum for expnums in to_extract.values() for expnum in expnums})
        reduce_2d(seq["mjds"], calibrations=calibs, expnums=expnums, cameras=[camera for camera in to_extract if to_extract[camera]],
                  assume_imagetyp="arc", reject_cr=False, add_astro=False, sub_straylight=False, skip_done=True)
        for camera, expnums in to_extract.items():
            if not expnums:
                continue
            rows = seq["frames"][(seq["frames"].camera == camera) & seq["frames"].expnum.isin(expnums)]
            if len(rows) < len(expnums):
                log.warning(f"{camera}: {len(expnums) - len(rows)} arcs to extract are not in the epoch's arc sequence")
            extract_arcs(rows, calibs=calibs, skip_done=True)

    all_metrics, caveats = [], {}
    for camera, (arcs, masters) in arcs_by_camera.items():
        try:
            paths = compare_camera(camera, arcs, masters, output_dir, with_sum=with_sum, nprocs=nprocs, skip_done=skip_done)
            if paths["missing"]:
                log.warning(f"{camera}: {len(paths['missing'])} arcs without lvmArc or extracted arc" + ("" if extract else " (try --extract)"))
            if paths["sum_problem"]:
                caveats[camera] = f"summed arcs skipped: {paths['sum_problem']}"
            all_metrics.append(camera_metrics(paths, fibers=fibers))
        except Exception as error:
            problems[camera] = f"failed: {error}"
            log.error(f"{camera}: {problems[camera]}")

    if not all_metrics:
        log.error(f"no camera could be checked for epoch {mjd_epoch}: {problems}")
        return {"summary": None, "report_path": None, "table_path": None}

    for camera, caveat in caveats.items():
        log.warning(f"{camera}: {caveat}")
    summary = write_report(all_metrics, report_path, mjd_epoch, kind, mjds, problems=problems)
    pd.concat([metrics["fibers"] for metrics in all_metrics]).to_csv(table_path, index=False)
    log.info(f"wrote dashboard to '{report_path}' and per-fiber table to '{table_path}'")
    return {"summary": summary, "report_path": report_path, "table_path": table_path}


DEFINITIONS = r"""<section id="definitions">
    <h2>How the quantities are computed</h2>
    <p>Exposures are indexed by \(k\), fibers by \(i\) and lines by \(j\). \(x_{kij}\) and \(w_{kij}\) are the centroid
    and FWHM in pixels of a line measured in exposure \(k\), \(\lambda_j\) its reference wavelength and \(D_i(x)\) the dispersion
    in &#8491;/pixel of a master wavelength solution \(\lambda_i(x)\).</p>
    <div class="defs">
      <div class="def">
        <h3>Fit residuals</h3>
        <p>Polynomial solution at the combined line position \(\bar x_{ij}\) minus the data: the reference wavelength, or
        the line width in &#8491;. Each combination is evaluated against its own lines.</p>
        <div class="eq">\[ r^{\lambda}_{ij} = \lambda_i(\bar x_{ij}) - \lambda_j, \qquad r^{\mathrm{LSF}}_{ij} = \mathrm{LSF}_i(\bar x_{ij}) - D_i(\bar x_{ij})\,\bar w_{ij} \]</div>
      </div>
      <div class="def">
        <h3>Fit scatter</h3>
        <p>Robust scatter of the residuals over all good fibers and lines of a camera.</p>
        <div class="eq">\[ s = 1.4826\,\mathrm{MAD}_{ij}\left(r_{ij}\right) \]</div>
      </div>
      <div class="def">
        <h3>Minus linear term</h3>
        <p>A straight line \(a_i + b_i x\) is fitted by least squares to each fiber's solution over the detector and
        subtracted from the solution and from the data at the lines.</p>
        <div class="eq">\[ \lambda_i(x) - (a_i + b_i x), \qquad \lambda_j - (a_i + b_i \bar x_{ij}) \]</div>
      </div>
      <div class="def">
        <h3>Line profile residual</h3>
        <p>Data minus the fitted model (pixel-integrated Gaussian of flux \(F\), centroid \(x_0\), width \(\sigma\), plus
        a constant background \(B\)) over the fitting window, relative to the line peak. The tile reports the RMS of the
        median residual profile within \(\pm\)1.5 pixels of the centroid.</p>
        <div class="eq">\[ \rho(\delta) = \frac{d(x_0 + \delta) - m(x_0 + \delta)}{F / (\sqrt{2\pi}\,\sigma)} \]</div>
      </div>
      <div class="def">
        <h3>Exposure offsets</h3>
        <p>Offset of each exposure from the median line positions of all exposures, median over lines. The per-arc
        combination removes these before combining the centroids.</p>
        <div class="eq">\[ o_{ki} = \mathrm{med}_j\left(x_{kij} - \mathrm{med}_k\, x_{kij}\right) \]</div>
      </div>
      <div class="def">
        <h3>LSF bias</h3>
        <p>Master LSF of each combination at the line position, minus the line width in single exposures (median over
        exposures).</p>
        <div class="eq">\[ b_{ij} = \mathrm{LSF}_i(\bar x_{ij}) - D_i(\bar x_{ij})\, \mathrm{med}_k\, w_{kij} \]</div>
      </div>
      <div class="def">
        <h3>FWHM change</h3>
        <p>FWHM of each exposure minus the median FWHM of all exposures, median over lines. The trend is its median over
        good fibers, per exposure.</p>
        <div class="eq">\[ \Delta w_{ki} = \mathrm{med}_j\left(w_{kij} - \mathrm{med}_k\, w_{kij}\right) \]</div>
      </div>
      <div class="def">
        <h3>Broadening of the sum</h3>
        <p>Summing Gaussians shifted by \(o_{ki}\) adds the flux-weighted variance of the shifts to the squared width. The
        observed broadening is the median over lines of the squared width in the sum minus that of single exposures.</p>
        <div class="eq">\[ \Delta^{\mathrm{pred}}_i = 8\ln 2\, \frac{\sum_k f_{ki}\,(o_{ki} - \bar o_i)^2}{\sum_k f_{ki}}, \qquad \Delta^{\mathrm{obs}}_i = \mathrm{med}_j\left(w^{\mathrm{sum}\,2}_{ij} - (\mathrm{med}_k\, w_{kij})^2\right) \]</div>
      </div>
    </div>
    <p class="note">Pixels below zero after continuum subtraction are masked before the lines are fitted, so the far wings
    of faint lines carry a small positive residual. Single-exposure widths come from lines with less signal than the sum;
    above a peak S/N of about 5 the fitted FWHM of a Gaussian is unbiased, so this mainly adds scatter.</p>
  </section>"""
