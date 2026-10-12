# encoding: utf-8
"""Arc LSF against the widths of sky lines in a science exposure.

For a calibration epoch, one science exposure and one or two pipeline versions (e.g. the previous
arc combination scheme and the current one):

1. For each version, take the master wavelength and LSF solutions of the epoch and its arc lines:
   the lvmArc products of each arc exposure (located through the FRAMES table of the master), or,
   for versions without them, the lines measured here in the version's combined extracted arc.
2. Detect isolated sky lines in the median spectrum of the sky telescope fibers of the science
   exposure (extracted, native pixels) and fit them in the fibers of every telescope with the same
   line model, window, bounds and continuum subtraction as the arc lines.
3. Write a dashboard comparing the sky line widths with each version's arc LSF, per telescope, each
   sky line with the arc lines sitting close to it along X in the same fibers, and the line profiles
   of single fibers and exposures.

With `reduce`, products missing for the current pipeline version are made here: the arcs and the
science exposure are detrended, extracted and measured, and the master solutions fitted.

Run it with ``drp calibrations qa-arc-sky``.
"""

import glob
import os
import warnings
from copy import deepcopy as copy
from datetime import datetime, timezone
from html import escape
from multiprocessing import Pool

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.io import fits
from scipy.signal import find_peaks

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.core.constants import CAMERAS, CALIBRATION_NEEDS
from lvmdrp.core.rss import RSS, lvmArc
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rm
from lvmdrp.qa import wavelengths as qw
from lvmdrp.qa.report import THEME, SERIES, base_layout, style_axes, html_table, picker, write_dashboard, epoch_qa_dir

# the line fitting parameters of measure_arc_lines, so that sky and arc lines are measured alike
FIT = dict(aperture=12, fwhm_guess=3.0, bg_guess=0.0, cent_range=[-2.0, 2.0], fwhm_range=[1.5, 4.5], bg_range=[-1e3, 1e4])
CONTINUUM = dict(cont_niter=3, cont_thresh=0.999, cont_box_range=(50, 300))
# sky lines are detected in the sky telescope fibers and fitted in the fibers of every telescope
SKY_TELESCOPES = ("SkyE", "SkyW")
TELESCOPES = ("SkyE", "SkyW", "Sci")
TELESCOPE_SYMBOLS = {"SkyE": "circle", "SkyW": "square", "Sci": "diamond"}
CALIB_TILEID = 11111
# sky lines take the orange series; up to two versions take the blue and aqua ones
SKY_COLOR = SERIES[1]
VERSION_COLORS = (SERIES[0], SERIES[2])
GAUSS_FWHM = 2 * np.sqrt(2 * np.log(2))
# line profiles: wavelength - centroid range and bin size, Angstrom
PROFILE_RANGE = 4.0
ANGSTROM = "Å"


# ---------------------------------------------------------------------------------------------------------------------
# products of a version

def master_paths(version, epoch, camera, kind="longterm"):
    """Paths of the master wavelength and LSF solutions of a camera in a version's reductions"""
    kinds = ("mwave", "mlsf") if kind == "longterm" else ("nwave", "nlsf")
    return {name: path.full("lvm_master", drpver=version, tileid=CALIB_TILEID, mjd=epoch, camera=camera, kind=master_kind)
            for name, master_kind in zip(("wave", "lsf"), kinds)}


def arc_paths(version, arc):
    """Paths of the lvmArc and extracted arc of one exposure in a version's reductions"""
    keys = dict(drpver=version, tileid=arc["tileid"], mjd=arc["mjd"], expnum=arc["expnum"])
    return {"lvmarc": path.full("lvm_frame", kind=f"Arc-{arc['camera']}", **keys),
            "xarc": path.full("lvm_anc", kind="x", imagetype="arc", camera=arc["camera"], **keys)}


def combined_xarcs(version, epoch, camera):
    """Combined extracted arcs (lvm-xarc-{camera}-{first}_{last}) of an epoch in a version's reductions"""
    folder = os.path.dirname(path.full("lvm_anc", drpver=version, kind="x", imagetype="arc", tileid=CALIB_TILEID,
                                       mjd=epoch, camera=camera, expnum=0))
    return sorted(glob.glob(os.path.join(folder, f"lvm-xarc-{camera}-*_*.fits")))


def science_xframe(version, tileid, mjd, camera, expnum):
    return path.full("lvm_anc", drpver=version, kind="x", imagetype="object", tileid=tileid, mjd=mjd, camera=camera, expnum=expnum)


def version_products(version, epoch, camera, output_dir):
    """Master solutions and arc line measurements of a version

    The arc lines are the version's lvmArc products when its master lists them in FRAMES. Otherwise
    the version's combined extracted arc is listed in ``to_measure``, to be measured into `output_dir`.
    """
    masters = master_paths(version, epoch, camera)
    products = {"version": version, "camera": camera, "masters": masters, "larcs": [], "to_measure": [], "source": None, "problem": None}
    if not all(os.path.isfile(p) for p in masters.values()):
        products["problem"] = f"no master solutions in {version}"
        return products
    larcs = [arc_paths(version, arc)["lvmarc"] for arc in qw.master_arcs(masters["wave"]) or []]
    larcs = [larc for larc in larcs if os.path.isfile(larc)]
    if larcs:
        products.update(larcs=larcs, source=f"{len(larcs)} lvmArc products")
        return products
    combined = combined_xarcs(version, epoch, camera)
    if not combined:
        products["problem"] = f"no lvmArc products nor combined extracted arc in {version}"
        return products
    out = os.path.join(output_dir, version, os.path.basename(combined[-1]).replace("lvm-xarc-", "lvmArc-"))
    products.update(larcs=[out], to_measure=[] if os.path.isfile(out) else [(combined[-1], out)],
                    source="combined arc, lines measured here")
    return products


# ---------------------------------------------------------------------------------------------------------------------
# reduction with the current version (--reduce)

def _extract_task(task):
    """Extract one detrended frame (worker)"""
    in_image, out_rss, trace, sigma, model = task
    if os.path.isfile(out_rss):
        return out_rss, None
    try:
        image_tasks.extract_spectra(in_image=in_image, out_rss=out_rss, in_trace=trace, in_sigma=sigma, in_model=model)
    except Exception as error:
        return out_rss, str(error)
    return out_rss, None


def _pool_map(func, tasks, nprocs):
    if nprocs > 1 and len(tasks) > 1:
        with Pool(min(nprocs, len(tasks))) as pool:
            return pool.map(func, tasks)
    return [func(task) for task in tasks]


def epoch_calibrations(epoch):
    from lvmdrp.functions.run_calseq import get_calib_paths
    return get_calib_paths(mjd=epoch, version=drpver, flavors=CALIBRATION_NEEDS["wave"], from_sandbox=True)


def reduce_arcs(epoch, cameras, nprocs, output_dir):
    """Detrend, extract and measure the long-term arcs of an epoch with the current version; fit the master solutions

    Returns
    -------
    dict
        Products of each camera, as :func:`version_products`
    """
    from lvmdrp.functions.run_calseq import load_epochs_file, get_arc_sequence
    from lvmdrp.main import reduce_2d

    epochs = load_epochs_file(epochs_kind="calibration", verbose=False)
    mjds, frames, _ = get_arc_sequence(mjd=epoch, epochs=epochs, kind="longterm")
    frames = frames[frames.camera.isin(cameras)]
    calibs = epoch_calibrations(epoch)
    log.info(f"epoch {epoch}: {frames.expnum.nunique()} arc exposures from MJDs {mjds}, cameras {cameras}")
    reduce_2d(mjds, calibrations=calibs, expnums=sorted(frames.expnum.unique()), cameras=cameras, assume_imagetyp="arc",
              reject_cr=False, add_astro=False, sub_straylight=False, skip_done=True)

    arcs = {camera: [{"tileid": int(row.tileid), "mjd": int(row.mjd), "expnum": int(row.expnum), "camera": camera}
                     for row in frames[frames.camera == camera].itertuples()] for camera in cameras}
    tasks = [(path.full("lvm_anc", drpver=drpver, kind="d", imagetype="arc", tileid=arc["tileid"], mjd=arc["mjd"], camera=camera,
                        expnum=arc["expnum"]), arc_paths(drpver, arc)["xarc"],
              calibs["centroids"][camera], calibs["sigmas"][camera], calibs["model"][camera])
             for camera, camera_arcs in arcs.items() for arc in camera_arcs]
    log.info(f"extracting {len(tasks)} arcs")
    for out_rss, error in _pool_map(_extract_task, tasks, nprocs):
        if error:
            log.error(f"failed extracting {out_rss}: {error}")

    # measure every arc in one pool, across cameras, then fit each camera's master solutions
    folder = os.path.join(output_dir, drpver)
    measure = [(arc_paths(drpver, arc)["xarc"], os.path.join(folder, os.path.basename(arc_paths(drpver, arc)["lvmarc"])))
               for camera_arcs in arcs.values() for arc in camera_arcs]
    measure = [(xarc, larc) for xarc, larc in measure if os.path.isfile(xarc) and not os.path.isfile(larc)]
    log.info(f"measuring arc lines in {len(measure)} arcs")
    _pool_map(qw._measure_task, measure, nprocs)

    products = {}
    for camera in arcs:
        larcs = sorted(glob.glob(os.path.join(folder, f"lvmArc-{camera}-*.fits")))
        masters = {"wave": os.path.join(folder, f"lvm-mwave-{camera}.fits"), "lsf": os.path.join(folder, f"lvm-mlsf-{camera}.fits")}
        if larcs and not all(os.path.isfile(p) for p in masters.values()):
            rm.fit_wavelength_solution(in_arcs=larcs, out_wave=masters["wave"], out_lsf=masters["lsf"])
        products[camera] = {"version": drpver, "camera": camera, "masters": masters, "larcs": larcs, "to_measure": [],
                            "source": f"{len(larcs)} lvmArc products measured here", "problem": None if larcs else "no arcs measured"}
    return products


def reduce_science(mjd, expnum, cameras, epoch, nprocs):
    """Detrend and extract a science exposure with the current version, as the arcs: native pixels, no stray light subtraction"""
    from lvmdrp.main import reduce_2d
    from lvmdrp.utils import metadata as md

    calibs = epoch_calibrations(epoch)
    reduce_2d(mjd, calibrations=calibs, expnums=[expnum], cameras=cameras, reject_cr=True, add_astro=False,
              sub_straylight=False, skip_done=True)
    frames = md.get_frames_metadata(mjd)
    frames = frames[(frames.expnum == expnum) & frames.camera.isin(cameras)]
    tasks, xframes = [], {}
    for row in frames.to_dict("records"):
        camera = row["camera"]
        dframe = path.full("lvm_anc", drpver=drpver, kind="d", imagetype=row["imagetyp"], **row)
        xframes[camera] = path.full("lvm_anc", drpver=drpver, kind="x", imagetype=row["imagetyp"], **row)
        tasks.append((dframe, xframes[camera], calibs["centroids"][camera], calibs["sigmas"][camera], calibs["model"][camera]))
    for out_rss, error in _pool_map(_extract_task, tasks, nprocs):
        if error:
            log.error(f"failed extracting {out_rss}: {error}")
    return {camera: frame for camera, frame in xframes.items() if os.path.isfile(frame)}


# ---------------------------------------------------------------------------------------------------------------------
# sky lines

def _robust_noise(values):
    values = values[np.isfinite(values)]
    return 1.4826 * np.median(np.abs(values - np.median(values))) if values.size else np.nan


def _pixel_of(wave_curve, waves):
    """Pixel positions of wavelengths along one fiber's wavelength solution"""
    pixels = np.arange(wave_curve.size, dtype=float)
    order = np.argsort(wave_curve)
    return np.interp(waves, wave_curve[order], pixels[order], left=np.nan, right=np.nan)


def measure_sky(xframe, camera, wave_curves, fiber_radius=0.01, snr_min=15.0, isolation=10, nlines_max=60, flux_min=20.0):
    """Detect isolated sky lines in the sky telescope fibers and fit them in the fibers of every telescope with the arc line model

    Parameters
    ----------
    xframe : str
        Extracted science frame
    camera : str
        Camera name
    wave_curves : numpy.ndarray
        Master wavelength solution of each fiber, used to place the lines in each fiber
    fiber_radius : float
        Radius in pixels of the fiber profile in the line model; use that of the arc line fits
    snr_min : float
        Minimum peak S/N of a line in the median sky spectrum
    isolation : int
        Half width in pixels within which no other line above 10% of a line's peak may sit
    nlines_max : int
        Maximum number of lines, the brightest
    flux_min : float
        Minimum integrated flux of a fitted line

    Returns
    -------
    dict
        Fitted fibers and their telescopes, their continuum subtracted spectra, approximate line
        wavelengths and the fitted centroids, FWHMs (pixels), fluxes and backgrounds
    """
    rss = RSS.from_file(xframe)
    slitmap = rss._slitmap
    if len(slitmap) != rss._fibers:
        slitmap = slitmap[slitmap["spectrographid"] == int(camera[1])]
    status = np.asarray(slitmap["fibstatus"]) if "fibstatus" in slitmap.colnames else np.zeros(len(slitmap), int)
    telescopes = np.asarray(slitmap["telescope"]).astype(str)
    sky = np.flatnonzero(np.isin(telescopes, SKY_TELESCOPES) & (status == 0))
    fitted = np.flatnonzero(np.isin(telescopes, TELESCOPES) & (status == 0))
    parc = rm._prepare_arc(copy(rss), **CONTINUUM)
    data = np.where(parc._mask, np.nan, parc._data).astype(float)
    npix = data.shape[1]

    # median sky spectrum on the wavelength grid of the median sky fiber
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grid = np.nanmedian(wave_curves[sky], axis=0)
        stack = np.array([np.interp(grid, wave_curves[i], data[i], left=np.nan, right=np.nan) for i in sky])
        median = np.nanmedian(stack, axis=0)
    noise = _robust_noise(median)
    filled = np.nan_to_num(median, nan=0.0)
    candidates, _ = find_peaks(filled, height=3 * noise)
    strong = [p for p in candidates if filled[p] >= snr_min * noise and FIT["aperture"] <= p < npix - FIT["aperture"]]
    isolated = [p for p in strong if not any(abs(q - p) <= isolation and q != p and filled[q] > 0.1 * filled[p] for q in candidates)]
    isolated = sorted(sorted(isolated, key=lambda p: -filled[p])[:nlines_max])
    waves = grid[isolated]
    log.info(f"{camera}: {len(isolated)} isolated sky lines with peak S/N > {snr_min:g} in {len(sky)} sky fibers, "
             f"fitted in {fitted.size} fibers of {', '.join(TELESCOPES)}")

    out = {key: np.full((fitted.size, waves.size), np.nan) for key in ("cent", "fwhm", "flux", "bkg")}
    for k, fiber in enumerate(fitted):
        guess = _pixel_of(wave_curves[fiber], waves)
        ok = np.isfinite(guess)
        if not ok.any():
            continue
        flux, cent, fwhm, bkg = parc.getSpec(fiber).fitSepGauss(guess[ok], flux_range=[flux_min, np.inf], fiber_radius=fiber_radius, **FIT)
        for key, values in zip(("flux", "cent", "fwhm", "bkg"), (flux, cent, fwhm, bkg)):
            out[key][k, ok] = values
    out.update(fibers=fitted, telescopes=telescopes[fitted], labels=np.asarray(slitmap["orig_ifulabel"]).astype(str)[fitted],
               waves=waves, data=data[fitted])
    return out


def _select(sky, telescope):
    """Sky line measurements in the fibers of one telescope"""
    sel = sky["telescopes"] == telescope
    return {**sky, **{key: sky[key][sel] for key in ("cent", "fwhm", "flux", "bkg", "fibers", "telescopes", "labels", "data")}}


# ---------------------------------------------------------------------------------------------------------------------
# comparison

def _peak(flux, fwhm):
    return flux / (fwhm / GAUSS_FWHM * np.sqrt(2 * np.pi))


def _at(curves, fibers, cent):
    """Per-fiber curves evaluated at positions cent (nfibers_sel, nlines)"""
    out = np.full(cent.shape, np.nan)
    pixels = np.arange(curves.shape[1])
    for k, fiber in enumerate(fibers):
        ok = np.isfinite(cent[k])
        out[k, ok] = np.interp(cent[k, ok], pixels, curves[fiber])
    return out


def _profile_points(data, cent, fwhm, flux, bkg, disp, half_width=None):
    """Background subtracted, peak normalized pixels around one line in several spectra, against wavelength - centroid

    data : (nspectra, npix); cent, fwhm, flux, bkg : (nspectra,) fitted in pixels; disp : (nspectra,) Angstrom/pixel
    """
    half_width = half_width or FIT["aperture"] // 2
    pixels = np.arange(data.shape[1])
    dwave, values = [], []
    for spectrum, c, w, f, b, d in zip(data, cent, fwhm, flux, bkg, disp):
        if not np.isfinite([c, w, f, d]).all() or f <= 0:
            continue
        window = (np.abs(pixels - c) <= half_width) & np.isfinite(spectrum)
        dwave.append((pixels[window] - c) * d)
        values.append((spectrum[window] - (b if np.isfinite(b) else 0.0)) / _peak(f, w))
    if not dwave:
        return np.array([]), np.array([])
    return np.concatenate(dwave), np.concatenate(values)


def _arc_profiles(lvmarcs, ref_line, fiber, disp):
    """Pixels of one arc line in one fiber, one profile per arc exposure, none combined

    Returns
    -------
    list[tuple]
        Exposure number, wavelength - centroid, peak normalized flux and fitted sigma (Angstrom) of each exposure
    """
    profiles = []
    for i, larc in enumerate(lvmarcs):
        waves, cent, fwhm, flux, masked = larc.get_lines()
        col = np.flatnonzero(np.isclose(waves, ref_line))
        if col.size == 0 or masked[fiber, col[0]]:
            continue
        col = col[0]
        data = larc._data[fiber].astype(float)
        if larc._mask is not None:
            data = np.where(larc._mask[fiber], np.nan, data)
        d = np.interp(cent[fiber, col], np.arange(disp.shape[1]), disp[fiber])
        dwave, values = _profile_points(data[None], [cent[fiber, col]], [fwhm[fiber, col]], [flux[fiber, col]],
                                        [larc.get_line_table("LINE_BKG")[fiber, col]], [d])
        profiles.append((larc._header.get("EXPOSURE", i), dwave, values, fwhm[fiber, col] * d / GAUSS_FWHM))
    return profiles


def select_fibers(sky, requested=None):
    """Fibers whose line profiles are shown: the requested ones, or 4 spread along the slit, in the sky and science telescopes

    Parameters
    ----------
    sky : dict
        Sky line measurements (see :func:`measure_sky`)
    requested : dict[str, list[int]]
        Fibers (rows of the camera's frame) asked for each group, "sky" and "sci"

    Returns
    -------
    dict[str, int]
        Fiber of each selector slot ("sky1".."sky4", "sci1".."sci4")
    """
    slots = {}
    for group, telescopes in (("sky", SKY_TELESCOPES), ("sci", ("Sci",))):
        candidates = sky["fibers"][np.isin(sky["telescopes"], telescopes)]
        asked = (requested or {}).get(group) or []
        chosen = [fiber for fiber in asked if fiber in candidates]
        if len(chosen) < len(asked):
            log.warning(f"fibers {sorted(set(asked) - set(chosen))} are not working fibers of {'/'.join(telescopes)}, skipped")
        if not chosen and candidates.size:
            chosen = [int(candidates[int(round(q * (candidates.size - 1)))]) for q in qw.SPECTRUM_POSITIONS]
        slots.update({f"{group}{i + 1}": int(fiber) for i, fiber in enumerate(chosen)})
    return slots


def compare_version(products, sky, telescope, show=(), max_dx=40):
    """Arc and sky line widths in the fibers of one telescope, with one version's arcs and master solutions

    The line profiles of the pairs are kept for the fibers in `show`, each fiber and arc exposure on its own.
    """
    lvmarcs = [lvmArc.from_file(larc) for larc in products["larcs"]]
    wave = TraceMask.from_file(products["masters"]["wave"])._data.astype(float)
    lsf = TraceMask.from_file(products["masters"]["lsf"])._data.astype(float)
    disp = np.abs(np.gradient(wave, axis=1))
    fibers = sky["fibers"]

    # arc lines: single-exposure values, median over exposures
    ref_lines, flux, cent, fwhm, masked = rm._combine_lvmarcs_lines(lvmarcs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cent_a = np.nanmedian(np.where(masked, np.nan, cent), axis=0)[fibers]
        fwhm_a = np.nanmedian(np.where(masked, np.nan, fwhm), axis=0)[fibers]
        flux_a = np.nanmedian(np.where(masked, np.nan, flux), axis=0)[fibers]
        xa = np.nanmedian(cent_a, axis=0)
        disp_a = _at(disp, fibers, cent_a)

        # sky lines, in pixels and Angstrom with this version's solution
        valid = np.isfinite(sky["cent"]) & np.isfinite(sky["fwhm"]) & np.isfinite(sky["flux"])
        cent_s, fwhm_s = np.where(valid, sky["cent"], np.nan), np.where(valid, sky["fwhm"], np.nan)
        xs = np.nanmedian(cent_s, axis=0)
        disp_s = _at(disp, fibers, cent_s)
        lsf_s = _at(lsf, fibers, cent_s) / disp_s
        wave_s = np.nanmedian(_at(wave, fibers, cent_s), axis=0)

        # each sky line against the nearest arc line along X, fiber by fiber, with both line profiles in the shown fibers
        shown = [(row, int(fiber)) for row, fiber in enumerate(fibers) if fiber in set(show)]
        pairs = []
        for k in range(xs.size):
            if not np.isfinite(xs[k]) or not np.isfinite(xa).any():
                continue
            j = int(np.nanargmin(np.abs(xa - xs[k])))
            dx = xa[j] - xs[k]
            if abs(dx) > max_dx:
                continue
            diff = fwhm_s[:, k] - fwhm_a[:, j]
            pairs.append({
                "k": k, "sky_wave": wave_s[k], "arc_wave": ref_lines[j], "x": xs[k], "dx": dx, "nfibers": int(np.isfinite(diff).sum()),
                "diff": diff, "ratio": fwhm_s[:, k] / fwhm_a[:, j],
                "sky_sigma": np.nanmedian(fwhm_s[:, k] * disp_s[:, k]) / GAUSS_FWHM,
                "arc_sigma": np.nanmedian(fwhm_a[:, j] * disp_a[:, j]) / GAUSS_FWHM,
                "sky_peak": np.nanmedian(_peak(sky["flux"][:, k], fwhm_s[:, k])), "arc_peak": np.nanmedian(_peak(flux_a[:, j], fwhm_a[:, j])),
                "profiles": {fiber: {
                    "sky": _profile_points(sky["data"][[row]], cent_s[[row], k], fwhm_s[[row], k], sky["flux"][[row], k],
                                           sky["bkg"][[row], k], disp_s[[row], k]),
                    "sky_sigma": fwhm_s[row, k] * disp_s[row, k] / GAUSS_FWHM,
                    "arcs": _arc_profiles(lvmarcs, ref_lines[j], fiber, disp)} for row, fiber in shown},
            })

        summary = {
            "camera": products["camera"], "version": products["version"], "telescope": telescope, "source": products["source"],
            "narcs": len(lvmarcs), "nfibers": int(fibers.size), "nsky": int(np.isfinite(xs).sum()), "npairs": len(pairs), "sky_fwhm": np.nanmedian(fwhm_s),
            "sky_over_lsf": np.nanmedian(fwhm_s / lsf_s),
            "pair_diff": np.nanmedian([np.nanmedian(p["diff"]) for p in pairs]) if pairs else np.nan,
            "pair_ratio": np.nanmedian([np.nanmedian(p["ratio"]) for p in pairs]) if pairs else np.nan,
        }
        return {
            "version": products["version"], "telescope": telescope, "summary": summary, "pairs": pairs, "sky_over_lsf": fwhm_s / lsf_s,
            "lsf_curve": np.nanmedian(lsf[fibers], axis=0), "wave_curve": np.nanmedian(wave[fibers], axis=0),
            "sky_A": fwhm_s * disp_s, "sky_wave": wave_s, "arc_A": fwhm_a * disp_a, "arc_wave": np.nanmedian(_at(wave, fibers, cent_a), axis=0),
            "sky_minus_lsf": fwhm_s - lsf_s, "xs": xs, "fwhm_a": fwhm_a, "flux_a": flux_a, "fwhm_s": fwhm_s, "flux_s": sky["flux"],
        }


# ---------------------------------------------------------------------------------------------------------------------
# dashboard

def _r(values, decimals=4):
    return np.round(np.asarray(values, dtype=float), decimals).astype(np.float32)


def _layout(fig, height, xtitle, ytitle, **kwargs):
    fig.update_layout(**base_layout(height, **kwargs))
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text=ytitle)
    return style_axes(fig)


def _alpha(color, alpha):
    color = color.lstrip("#")
    return f"rgba({int(color[0:2], 16)},{int(color[2:4], 16)},{int(color[4:6], 16)},{alpha})"


def _stats(values, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(values, 16, axis=axis), np.nanmedian(values, axis=axis), np.nanpercentile(values, 84, axis=axis)


def _error_bars(lo, med, hi, color):
    return dict(type="data", symmetric=False, array=_r(hi - med), arrayminus=_r(med - lo), color=_alpha(color, 0.5), thickness=1.5, width=0)


def fig_widths(camera_results):
    """Each version's arc LSF along the detector, with arc and sky line widths, in Angstrom"""
    fig = go.Figure()
    step = 8
    for result, color in zip(camera_results, VERSION_COLORS):
        version = result["version"]
        fig.add_trace(go.Scatter(x=_r(result["wave_curve"][::step], 2), y=_r(result["lsf_curve"][::step]), mode="lines",
                                 name=f"arc LSF, {version}", line=dict(color=color, width=2),
                                 hovertemplate="%{x:.1f} &#8491;<br>%{y:.4f} &#8491;<extra>arc LSF " + version + "</extra>"))
        lo, med, hi = _stats(result["arc_A"])
        ok = np.isfinite(result["arc_wave"]) & np.isfinite(med)
        fig.add_trace(go.Scatter(x=_r(result["arc_wave"][ok], 2), y=_r(med[ok]), mode="markers", name=f"arc lines, {version}",
                                 marker=dict(color=color, size=6, symbol="circle-open", line=dict(color=color, width=1.5)),
                                 error_y=_error_bars(lo[ok], med[ok], hi[ok], color),
                                 hovertemplate="%{x:.2f} &#8491;<br>FWHM %{y:.4f} &#8491;<extra>arc line " + version + "</extra>"))
    # sky lines in Angstrom with the first version's solution
    first = camera_results[0]
    lo, med, hi = _stats(first["sky_A"])
    ok = np.isfinite(first["sky_wave"]) & np.isfinite(med)
    fig.add_trace(go.Scatter(x=_r(first["sky_wave"][ok], 2), y=_r(med[ok]), mode="markers", name="sky lines",
                             marker=dict(color=SKY_COLOR, size=8, line=dict(color=THEME["surface"], width=1.5)),
                             error_y=_error_bars(lo[ok], med[ok], hi[ok], SKY_COLOR),
                             hovertemplate="%{x:.2f} &#8491;<br>FWHM %{y:.4f} &#8491;<extra>sky line</extra>"))
    return _layout(fig, 440, "Wavelength (&#8491;)", "FWHM (&#8491;)")


def fig_pairs(camera_results):
    """Sky line width minus the width of the nearest arc line along X in the same fibers, and minus the arc LSF"""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    for result, color in zip(camera_results, VERSION_COLORS):
        version, pairs = result["version"], result["pairs"]
        if pairs:
            lo, med, hi = (np.array(v) for v in zip(*[_stats(p["diff"]) for p in pairs]))
            custom = np.array([[p["sky_wave"], p["arc_wave"], p["dx"], p["nfibers"]] for p in pairs], dtype=float)
            fig.add_trace(go.Scatter(x=_r([p["x"] for p in pairs], 1), y=_r(med), mode="markers", name=f"sky &#8722; nearest arc line, {version}",
                                     marker=dict(color=color, size=9, line=dict(color=THEME["surface"], width=1.5)),
                                     error_y=_error_bars(lo, med, hi, color), customdata=_r(custom, 2),
                                     hovertemplate="X %{x:.0f} pix<br>sky %{customdata[0]:.2f} &#8491;, arc %{customdata[1]:.2f} &#8491;"
                                                   "<br>&#916;X %{customdata[2]:+.0f} pix, %{customdata[3]:.0f} fibers"
                                                   "<br>&#916;FWHM %{y:+.3f} pix<extra>" + version + "</extra>"))
        med = _stats(result["sky_minus_lsf"])[1]
        ok = np.isfinite(result["xs"]) & np.isfinite(med)
        fig.add_trace(go.Scatter(x=_r(result["xs"][ok], 1), y=_r(med[ok]), mode="markers", name=f"sky &#8722; arc LSF, {version}",
                                 marker=dict(color=_alpha(color, 0.15), size=7, symbol="diamond", line=dict(color=color, width=1.5)),
                                 hovertemplate="X %{x:.0f} pix<br>&#916;FWHM %{y:+.3f} pix<extra>sky &#8722; LSF " + version + "</extra>"))
    return _layout(fig, 440, "X (pix)", "FWHM difference (pix)")


def fig_peaks(camera_results):
    """FWHM against line peak for each version's arc lines and the sky lines, in the sky fibers"""
    fig = go.Figure()
    series = [(f"arc lines, {r['version']}", r["fwhm_a"], r["flux_a"], color) for r, color in zip(camera_results, VERSION_COLORS)]
    series.append(("sky lines", camera_results[0]["fwhm_s"], camera_results[0]["flux_s"], SKY_COLOR))
    for name, fwhm, flux, color in series:
        peak = _peak(flux, fwhm)
        ok = np.isfinite(peak) & np.isfinite(fwhm) & (peak > 0)
        fig.add_trace(go.Scatter(x=_r(peak[ok], 1), y=_r(fwhm[ok], 3), mode="markers", name=name, marker=dict(color=_alpha(color, 0.45), size=5),
                                 hovertemplate="peak %{x:.0f}<br>FWHM %{y:.3f} pix<extra>" + name + "</extra>"))
    fig = _layout(fig, 420, "Line peak (e&#8315;/pix)", "FWHM (pix)")
    fig.update_xaxes(type="log")
    return fig


def fig_telescopes(results, versions):
    """Sky line FWHM over the arc LSF in the fibers of each telescope, per camera and version"""
    fig = go.Figure()
    fig.add_hline(y=1, line=dict(color=THEME["axis"], width=1))
    cameras = list(results)
    offsets = dict(zip(TELESCOPES, np.linspace(-0.24, 0.24, len(TELESCOPES))))
    for version, color in zip(versions, VERSION_COLORS):
        shift = 0.04 * (versions.index(version) - (len(versions) - 1) / 2)
        for telescope in TELESCOPES:
            x, stats, nfibers = [], [], []
            for i, camera in enumerate(cameras):
                result = next((r for r in results[camera].get(telescope, []) if r["version"] == version), None)
                if result is not None:
                    x.append(i + offsets[telescope] + shift)
                    stats.append(_stats(result["sky_over_lsf"].ravel()))
                    nfibers.append(result["summary"]["nfibers"])
            if not x:
                continue
            lo, med, hi = (np.array(v) for v in zip(*stats))
            fig.add_trace(go.Scatter(x=_r(x, 3), y=_r(med), mode="markers", name=f"{telescope}, {version}",
                                     marker=dict(color=color, size=9, symbol=TELESCOPE_SYMBOLS[telescope], line=dict(color=THEME["surface"], width=1.5)),
                                     error_y=_error_bars(lo, med, hi, color), customdata=np.array(nfibers),
                                     hovertemplate="%{y:.3f} in %{customdata} fibers<extra>" + f"{telescope}, {version}" + "</extra>"))
    fig = _layout(fig, 420, "Camera", "Sky FWHM / arc LSF")
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(cameras))), ticktext=cameras, range=[-0.5, len(cameras) - 0.5])
    return fig


def fig_profile(camera_results, sky_index, fiber, title):
    """One sky line and its nearest arc line in each version in one fiber, peak normalized, against wavelength - centroid

    Pixels are drawn as measured: the sky line of the science exposure and the arc line of each arc exposure, none combined.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    sky = None
    for result, color in zip(camera_results, VERSION_COLORS):
        pair = next((p for p in result["pairs"] if p["k"] == sky_index), None)
        profiles = pair["profiles"].get(fiber) if pair is not None else None
        if profiles is None:
            continue
        sky = sky or (pair, profiles)
        arcs, version = profiles["arcs"], result["version"]
        sigma = np.nanmedian([s for *_, s in arcs]) if arcs else np.nan
        exposures = f" ({len(arcs)} exposures)" if len(arcs) > 1 else ""
        for i, (expnum, dwave, values, s) in enumerate(arcs):
            order = np.argsort(dwave)
            fig.add_trace(go.Scatter(x=_r(dwave[order], 3), y=_r(values[order]), mode="lines+markers", legendgroup=version, showlegend=i == 0,
                                     name=f"arc {pair['arc_wave']:.2f} &#8491;, {version}{exposures}, &#963; = {_fmt(sigma, '.3f')} &#8491;",
                                     line=dict(color=color, width=1.5), marker=dict(size=4), opacity=0.75 if len(arcs) > 1 else 1,
                                     hovertemplate="%{x:+.2f} &#8491;<br>%{y:.3f}<extra>" + f"{version}, exposure {expnum}, &#963; = {s:.3f} &#8491;" + "</extra>"))
    if sky is None:
        return None
    pair, profiles = sky
    dwave, values = profiles["sky"]
    order = np.argsort(dwave)
    fig.add_trace(go.Scatter(x=_r(dwave[order], 3), y=_r(values[order]), mode="lines+markers", line=dict(color=SKY_COLOR, width=2.5),
                             name=f"sky {pair['sky_wave']:.2f} &#8491;, &#963; = {_fmt(profiles['sky_sigma'], '.3f')} &#8491;",
                             marker=dict(size=6, symbol="square"), hovertemplate="%{x:+.2f} &#8491;<br>%{y:.3f}<extra>sky</extra>"))
    fig = _layout(fig, 420, "Wavelength &#8722; centroid (&#8491;)", "Peak-normalized flux")
    fig.update_xaxes(range=[-PROFILE_RANGE, PROFILE_RANGE])
    fig.add_annotation(text=title, xref="paper", yref="paper", x=1, y=0.98, xanchor="right", yanchor="top", showarrow=False,
                       font=dict(color=THEME["muted"], size=12))
    return fig


def _fmt(value, fmt):
    return "&ndash;" if value is None or not np.isfinite(value) else format(value, fmt)


def write_report(results, report_path, epoch, expnum, mjd, versions, science_version, fiber_radius, max_dx, fiber_slots=None, problems=None):
    """Write the dashboard

    Parameters
    ----------
    results : dict[str, dict[str, list[dict]]]
        Comparison of each camera and telescope, one result per version (see :func:`compare_version`)
    fiber_slots : dict[str, dict[str, tuple]]
        Fiber, telescope and IFU label of each fiber selector slot, per camera (see :func:`select_fibers`)
    problems : dict[str, str]
        Products missing in each camera
    """
    problems, fiber_slots = problems or {}, fiber_slots or {}
    cameras = list(results)
    summary = pd.DataFrame([r["summary"] for by_telescope in results.values() for camera_results in by_telescope.values() for r in camera_results])
    figures, pair_options = {"telescopes": fig_telescopes(results, versions)}, {}
    for camera, by_telescope in results.items():
        for telescope, camera_results in by_telescope.items():
            figures[f"widths|{camera}|{telescope}"] = fig_widths(camera_results)
            figures[f"pairs|{camera}|{telescope}"] = fig_pairs(camera_results)
            figures[f"peaks|{camera}|{telescope}"] = fig_peaks(camera_results)
        # line profiles in the shown fibers: one option per sky line paired with an arc line, across cameras (the select lists them all)
        for slot, (fiber, telescope, label) in fiber_slots.get(camera, {}).items():
            camera_results = by_telescope.get(telescope, [])
            sky_waves = {p["k"]: p["sky_wave"] for r in camera_results for p in r["pairs"]}
            for k in sky_waves:
                profile = fig_profile(camera_results, k, fiber, f"{camera} · fiber {fiber} · {telescope} {label}")
                if profile is not None:
                    figures[f"profile|{camera}:{k}|{slot}"] = profile
                    pair_options.setdefault((camera, k), (sky_waves[k], f"{camera} · sky {sky_waves[k]:.2f} {ANGSTROM}"))
    pair_options = [(f"{camera}:{k}", label) for (camera, k), (_, label) in
                    sorted(pair_options.items(), key=lambda item: (cameras.index(item[0][0]), item[1][0]))]
    slots = sorted({slot for camera_slots in fiber_slots.values() for slot in camera_slots}, key=lambda slot: (slot[:3] != "sky", int(slot[3:])))
    slot_dim = [("Fiber", [(slot, f"{slot[:3].capitalize()} fiber {slot[3:]}") for slot in slots])]
    telescope_dim = [("Telescope", [(telescope, telescope) for telescope in TELESCOPES])]
    camera_dim = [("Camera", [(camera, camera) for camera in cameras])] + telescope_dim
    profile_picker = (picker("fig-profile", "profile", [("Line pair", pair_options)] + slot_dim, 420) if pair_options
                      else '<p class="note">No sky line has an arc line within reach.</p>')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        nsky = int(summary.drop_duplicates("camera").nsky.sum()) if len(summary) else 0
        tiles = [("Sky lines measured", f"{nsky}", f"in {len(cameras)} cameras")]
        for telescope in TELESCOPES:
            values = [_fmt(np.nanmedian(summary[(summary.version == version) & (summary.telescope == telescope)].sky_over_lsf)
                           if len(summary) else np.nan, ".3f") for version in versions]
            tiles.append((f"Sky FWHM / arc LSF, {telescope}", " / ".join(values), f"{' / '.join(versions)}, median over cameras"))
    columns = [("Camera", ""), ("Version", ""), ("Telescope", ""), ("Arc lines from", ""), ("Fibers", "Fibers with sky line fits"),
               ("Sky lines", "Isolated sky lines measured"), ("Pairs", "Sky lines with an arc line nearby along X"),
               ("Sky FWHM (pix)", "Median sky line FWHM"), ("Sky / arc LSF", "Median of sky FWHM over the arc LSF at the line"),
               ("Sky - arc line (pix)", "Median FWHM difference of the pairs, same fibers"), ("Sky / arc line", "Median FWHM ratio of the pairs")]
    rows = [[escape(r.camera), escape(r.version), escape(r.telescope), f'<span class="range">{escape(str(r.source))}</span>', f"{r.nfibers}",
             f"{r.nsky}", f"{r.npairs}", _fmt(r.sky_fwhm, ".3f"), _fmt(r.sky_over_lsf, ".3f"), _fmt(r.pair_diff, "+.3f"), _fmt(r.pair_ratio, ".3f")]
            for r in summary.itertuples()]
    rows += [[escape(camera), "&ndash;", "&ndash;", f'<span class="range">{escape(problem)}</span>'] + [""] * (len(columns) - 4)
             for camera, problem in problems.items()]
    pair_rows = [[escape(camera), escape(r["version"]), escape(telescope), _fmt(p["sky_wave"], ".2f"), _fmt(p["arc_wave"], ".2f"),
                  _fmt(p["x"], ".0f"), _fmt(p["dx"], "+.0f"), f"{p['nfibers']}", _fmt(p["sky_sigma"], ".3f"), _fmt(p["arc_sigma"], ".3f"),
                  _fmt(np.nanmedian(p["diff"]), "+.3f"), _fmt(p["sky_peak"], ".0f"), _fmt(p["arc_peak"], ".0f")]
                 for camera, by_telescope in results.items() for telescope, camera_results in by_telescope.items()
                 for r in camera_results for p in r["pairs"]]
    pair_table = html_table([("Camera", ""), ("Version", ""), ("Telescope", ""), (f"Sky line ({ANGSTROM})", ""), (f"Arc line ({ANGSTROM})", ""),
                             ("X (pix)", ""), ("ΔX (pix)", ""), ("Fibers", ""), (f"Sky σ ({ANGSTROM})", ""), (f"Arc σ ({ANGSTROM})", ""),
                             ("ΔFWHM (pix)", ""), ("Sky peak", ""), ("Arc peak", "")], pair_rows, numeric=[False] * 3 + [True] * 10)

    sections = [
        f"""<section>
    <h2>Summary by camera</h2>
    <p>Sky lines are detected in the sky telescope fibers of the science exposure and fitted in the fibers of every telescope
    with the arc line model: a pixel-integrated Gaussian convolved with the fiber profile of the arc line fits
    ({fiber_radius:g} pix radius), plus a constant background, over the same window and bounds, after the same continuum
    subtraction, in native pixels. Arc widths are those of the same fibers, single-exposure values (median over the arc
    exposures) where a version has lvmArc products, otherwise those of its combined arc.</p>
    {html_table(columns, rows, numeric=[False] * 4 + [True] * (len(columns) - 4))}
  </section>""",
        """<section>
    <h2>By telescope</h2>
    <p>Sky line FWHM over the arc LSF at the line, in the fibers of each telescope: median over fibers and sky lines with the
    16&ndash;84% range. The arc LSF is measured in the same fibers, so a telescope offset points to the light entering the
    fibers (e.g. focal ratio, illumination) rather than to the spectrograph. Science fibers also carry the target, whose
    emission lines may blend with sky lines.</p>
    <div class="chart" data-fig="telescopes" style="min-height: 420px"></div>
  </section>""",
        f"""<section>
    <h2>Line profiles</h2>
    <p>One sky line and its nearest arc line along X in each version, in a single fiber: the pixels as measured, background
    subtracted and divided by the fitted peak, against the wavelength minus the fitted centroid. Nothing is combined: each
    arc exposure is drawn on its own (hover for its number and &sigma;), and &sigma; in the legend is the fitted Gaussian width
    of the sky line, or the median over the arc exposures. A version without lvmArc products has only its combined arc. The
    fibers are four of the sky telescopes and four of the science telescope, spread along the slit unless chosen with
    <code>--sky-fibers</code> and <code>--sci-fibers</code>; the label in the plot names the fiber of the selected camera.</p>
    {profile_picker}
  </section>""",
        f"""<section>
    <h2>Arc LSF against sky line widths</h2>
    <p>Master LSF of each version, median over the fibers of the selected telescope, with the FWHM of each arc and sky line,
    median over those fibers with the 16&ndash;84% range as error bars.</p>
    {picker("fig-widths", "widths", camera_dim, 440)}
  </section>""",
        f"""<section>
    <h2>Sky lines next to arc lines</h2>
    <p>For each sky line, its FWHM minus that of the nearest arc line along X, fiber by fiber, median over the fibers of the
    selected telescope with the 16&ndash;84% range. Only arc lines within {max_dx:g} pixels are paired, so both lines sample the same part of the detector.
    Diamonds compare each sky line with the version's arc LSF at its position. Positive means the sky line is wider.</p>
    {picker("fig-pairs", "pairs", camera_dim, 440)}
    <details><summary>Table view: line pairs</summary>{pair_table}</details>
  </section>""",
        f"""<section>
    <h2>Width against line peak</h2>
    <p>FWHM of each line in each fiber of the selected telescope against its peak. A trend between arc lines of different brightness (e.g. the
    brighter-fatter effect) would bias the arc LSF with respect to the much fainter sky lines.</p>
    {picker("fig-peaks", "peaks", camera_dim, 420)}
  </section>""",
    ]
    meta = [("Epoch", epoch), ("Science exposure", expnum), ("Science MJD", mjd), ("Versions", ", ".join(versions)),
            ("Sky frame from", science_version), ("Cameras", ", ".join(cameras)),
            ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]
    intro = (f"<p>Are arc lines wider than sky lines? The LSF and arc lines of epoch {epoch} in {' and '.join(versions)}, against "
             f"isolated sky lines in science exposure {expnum}, all measured with the same line model in native pixels.</p>")
    write_dashboard(report_path, f"Arc LSF and sky lines, {epoch} / {expnum}", f"LVM DRP · wavelength · arc vs sky LSF · {drpver}",
                    intro, meta, tiles, "\n\n  ".join(sections), figures, "lvmdrp.qa.arc_sky.qa_arc_sky")
    return summary


# ---------------------------------------------------------------------------------------------------------------------

def science_mjd(expnum):
    """MJD of a raw exposure, from the directory holding its raw frames"""
    raws = path.expand("lvm_raw", hemi="s", camspec="b1", mjd="*", expnum=expnum)
    if not raws:
        raise FileNotFoundError(f"no raw frames found for exposure {expnum}, give --mjd")
    return int(os.path.basename(os.path.dirname(raws[0])))


def science_tileid(mjd, expnum):
    """Tile of a science exposure, from the frames metadata"""
    from lvmdrp.utils import metadata as md
    frames = md.get_frames_metadata(mjd)
    frames = frames[frames.expnum == expnum]
    if frames.empty:
        raise ValueError(f"exposure {expnum} not found in the metadata of MJD {mjd}, give --tileid")
    return int(frames.tileid.iloc[0])


def qa_arc_sky(epoch, expnum, cameras=CAMERAS, versions=None, science_version=None, mjd=None, tileid=None, output_dir=None,
               sky_fibers=None, sci_fibers=None, max_dx=40, reduce=False, nprocs=1):
    """Compare the arc LSF of a calibration epoch with the widths of sky lines in a science exposure and write a dashboard

    Parameters
    ----------
    epoch : int
        MJD of the calibration epoch
    expnum : int
        Science exposure whose sky lines are measured
    cameras : list[str]
        Cameras to compare
    versions : list[str]
        One or two pipeline versions whose arcs and master solutions are compared, by default the current one
    science_version : str
        Version holding the extracted science frame, by default the first of `versions`
    mjd, tileid : int
        MJD and tile of the science exposure, found from its raw frames and the metadata if not given
    output_dir : str
        Directory of the dashboard and of anything measured for it, by default in the epoch's QA directory
    sky_fibers, sci_fibers : list[int]
        Up to four fibers (rows of each camera's frame) of the sky and science telescopes whose line profiles are
        shown, by default spread along the slit
    max_dx : float
        Largest separation along X in pixels of a sky line and the arc line it is compared with
    reduce : bool
        Make the products missing for the current pipeline version
    nprocs : int
        Processes measuring and extracting arcs

    Returns
    -------
    pandas.DataFrame
        Summary of each camera, version and telescope
    str
        Path of the dashboard
    """
    versions = list(versions or [drpver])
    if not 1 <= len(versions) <= 2:
        raise ValueError(f"one or two versions can be compared, got {versions}")
    cameras = [camera for camera in CAMERAS if camera in cameras]
    science_version = science_version or versions[0]
    mjd = mjd or science_mjd(expnum)
    tileid = tileid or science_tileid(mjd, expnum)
    output_dir = output_dir or epoch_qa_dir(drpver, epoch, "wave", f"{epoch}_sky_{expnum}")
    fibers = {"sky": sky_fibers, "sci": sci_fibers}
    log.info(f"arc vs sky LSF: epoch {epoch}, science exposure {expnum} (MJD {mjd}, tile {tileid}), versions {', '.join(versions)}")
    os.makedirs(output_dir, exist_ok=True)

    products = {camera: {version: version_products(version, epoch, camera, output_dir) for version in versions} for camera in cameras}
    if reduce and drpver in versions:
        missing = [camera for camera in cameras if products[camera][drpver]["problem"]]
        if missing:
            for camera, camera_products in reduce_arcs(epoch, missing, nprocs, output_dir).items():
                products[camera][drpver] = camera_products
    # measure the combined arcs of versions without lvmArc products, in one pool
    measure = [task for camera in cameras for p in products[camera].values() for task in p["to_measure"]]
    if measure:
        log.info(f"measuring arc lines in {len(measure)} combined arcs")
        _pool_map(qw._measure_task, measure, nprocs)

    xframes = {camera: science_xframe(science_version, tileid, mjd, camera, expnum) for camera in cameras}
    missing = [camera for camera, xframe in xframes.items() if not os.path.isfile(xframe)]
    if reduce and missing and science_version == drpver:
        xframes.update(reduce_science(mjd, expnum, missing, epoch, nprocs))

    results, problems, fiber_slots, fiber_radius = {}, {}, {}, None
    for camera in cameras:
        usable, failed = [], []
        for p in products[camera].values():
            if p["problem"] or not all(os.path.isfile(larc) for larc in p["larcs"]):
                failed.append(f"{p['version']}: {p['problem'] or 'arc lines not measured'}")
            else:
                usable.append(p)
        if not os.path.isfile(xframes[camera]):
            failed.append(f"no science frame in {science_version}")
        if failed:
            problems[camera] = "; ".join(failed)
            log.warning(f"{camera}: {problems[camera]}")
        if not usable or not os.path.isfile(xframes[camera]):
            continue
        # sky lines placed with the science version's solution (or the first usable), fitted with its arcs' fiber radius
        reference = next((p for p in usable if p["version"] == science_version), usable[0])
        wave = TraceMask.from_file(reference["masters"]["wave"])._data.astype(float)
        fiber_radius = float(fits.getheader(reference["larcs"][0]).get(f"{camera} WAVE FIBRAD", 0.01))
        log.info(f"{camera}: fitting sky lines with the arc lines fiber radius {fiber_radius:g} pix")
        sky = measure_sky(xframes[camera], camera, wave, fiber_radius=fiber_radius)
        slots = select_fibers(sky, fibers)
        rows = {fiber: row for row, fiber in enumerate(sky["fibers"])}
        fiber_slots[camera] = {slot: (fiber, sky["telescopes"][rows[fiber]], sky["labels"][rows[fiber]]) for slot, fiber in slots.items()}
        log.info(f"{camera}: line profiles of fibers {', '.join(f'{f} ({t} {lb})' for f, t, lb in fiber_slots[camera].values())}")
        results[camera] = {telescope: [compare_version(p, _select(sky, telescope), telescope, show=list(slots.values()), max_dx=max_dx)
                                       for p in usable]
                           for telescope in TELESCOPES if (sky["telescopes"] == telescope).any()}

    report = os.path.join(output_dir, f"arc-sky-lsf_{epoch}_{expnum}.html")
    summary = write_report(results, report, epoch, expnum, mjd, versions, science_version, fiber_radius if fiber_radius is not None else 0.01,
                           max_dx, fiber_slots=fiber_slots, problems=problems)
    log.info(f"dashboard: {report}")
    return summary, report
