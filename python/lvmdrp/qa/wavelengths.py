# encoding: utf-8
"""Wavelength and LSF QA: combination of arc exposures.

Compares two ways of combining the arc exposures of a calibration epoch into
the master wavelength and LSF solutions (see
:func:`~lvmdrp.functions.rssMethod.determine_wavelength_solution`):

- ``measurements``: the arc lines are measured in each extracted arc exposure
  (:func:`~lvmdrp.functions.rssMethod.measure_arc_lines`), the line centroids
  of each exposure are aligned to remove relative shifts along the dispersion
  axis (e.g., thermal), and the master solutions are fitted to the combined
  measurements (:func:`~lvmdrp.functions.rssMethod.fit_wavelength_solution`).
- ``pixels``: the extracted arc exposures are summed in pixel space and the
  lines are measured once in the sum.

A line in the sum of shifted exposures is wider than in any of them, so the
``pixels`` LSF is expected to be broader by an amount set by the spread of the
line positions among the exposures. The dashboard shows both LSFs, how they
compare with the line widths measured in single exposures (the LSF a science
exposure actually has) and whether the difference is explained by the measured
shifts. See :func:`qa_arc_combinations`.
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

from lvmdrp import log, __version__ as drpver
from lvmdrp.core.constants import CAMERAS, CALIBRATION_NEEDS
from lvmdrp.core.rss import lvmArc
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.qa.report import (
    THEME, SERIES, DIVERGING, base_layout, style_axes, html_table, picker, write_dashboard, epoch_qa_dir,
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


def camera_paths(camera, xarc_paths, output_dir):
    """Return the paths of the products of both combination methods of a camera

    Parameters
    ----------
    camera : str
        Camera name, e.g. 'b1'
    xarc_paths : list[str]
        Paths to the extracted arc exposures of the camera
    output_dir : str
        Directory of the QA

    Returns
    -------
    dict
        Paths of the lvmArc products and master traces of each method
    """
    meas_dir = os.path.join(output_dir, "measurements")
    pix_dir = os.path.join(output_dir, "pixels")
    return {
        "camera": camera,
        "xarcs": list(xarc_paths),
        "measurements": {
            "arcs": [os.path.join(meas_dir, os.path.basename(xarc).replace("lvm-xarc-", "lvmArc-")) for xarc in xarc_paths],
            "wave": os.path.join(meas_dir, f"lvm-mwave-{camera}.fits"),
            "lsf": os.path.join(meas_dir, f"lvm-mlsf-{camera}.fits"),
        },
        "pixels": {
            "arcs": [os.path.join(pix_dir, f"lvmArc-{camera}-sum.fits")],
            "wave": os.path.join(pix_dir, f"lvm-mwave-{camera}.fits"),
            "lsf": os.path.join(pix_dir, f"lvm-mlsf-{camera}.fits"),
        },
    }


def compare_camera(camera, xarc_paths, output_dir, nprocs=1, skip_done=True):
    """Fit the master wavelength and LSF solutions of a camera with both combination methods

    All products are written to `output_dir`, never to the master calibrations.

    Parameters
    ----------
    camera : str
        Camera name, e.g. 'b1'
    xarc_paths : list[str]
        Paths to the extracted arc exposures of the camera
    output_dir : str
        Directory of the QA
    nprocs : int, optional
        Number of processes measuring the arcs, by default 1
    skip_done : bool, optional
        Reuse the products already in `output_dir`, by default True

    Returns
    -------
    dict
        Paths of the products, see :func:`camera_paths`
    """
    paths = camera_paths(camera, xarc_paths, output_dir)

    # measurements: lines measured in each arc, then combined
    meas = paths["measurements"]
    tasks = [(xarc, larc) for xarc, larc in zip(xarc_paths, meas["arcs"]) if not (skip_done and os.path.isfile(larc))]
    log.info(f"{camera}: measuring arc lines in {len(tasks)} arcs ({len(xarc_paths) - len(tasks)} already measured)")
    if nprocs > 1 and len(tasks) > 1:
        with Pool(min(nprocs, len(tasks))) as pool:
            results = pool.map(_measure_task, tasks)
    else:
        results = [_measure_task(task) for task in tasks]
    failed = [in_arc for in_arc, error in results if error is not None]
    meas["arcs"] = [larc for larc in meas["arcs"] if os.path.isfile(larc)]
    if len(meas["arcs"]) < 2:
        raise RuntimeError(f"{camera}: fewer than two arcs measured ({len(failed)} failed), nothing to combine")
    if not (skip_done and os.path.isfile(meas["wave"]) and os.path.isfile(meas["lsf"]) and not tasks):
        rss_tasks.fit_wavelength_solution(in_arcs=meas["arcs"], out_wave=meas["wave"], out_lsf=meas["lsf"])

    # pixels: arcs summed in pixel space, then measured
    pix = paths["pixels"]
    used_xarcs = [xarc for xarc in xarc_paths if xarc not in failed]
    if not (skip_done and all(os.path.isfile(p) for p in [pix["wave"], pix["lsf"], *pix["arcs"]]) and not tasks):
        log.info(f"{camera}: measuring arc lines in the sum of {len(used_xarcs)} arcs")
        rss_tasks.determine_wavelength_solution(in_arcs=used_xarcs, out_wave=pix["wave"], out_lsf=pix["lsf"],
                                                arcs_combination="pixels", out_arcs=pix["arcs"])
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


def _lines_on(ref_lines, lvmarc):
    """Return centroids, FWHMs and fluxes of an lvmArc on the given reference lines, NaN where missing or masked"""
    waves, cent, fwhm, flux, mask = lvmarc.get_lines()
    index = {wave: j for j, wave in enumerate(waves)}
    out = [np.full((cent.shape[0], ref_lines.size), np.nan) for _ in range(3)]
    for j, wave in enumerate(ref_lines):
        if wave in index:
            k = index[wave]
            for arr, values in zip(out, (cent, fwhm, flux)):
                arr[:, j] = np.where(mask[:, k], np.nan, values[:, k])
    return out


def camera_metrics(paths):
    """Measure how the two combination methods compare for a camera

    Parameters
    ----------
    paths : dict
        Paths of the products, see :func:`compare_camera`

    Returns
    -------
    dict
        Per-fiber and per-line metrics and the arrays used in the charts
    """
    lvmarcs = [lvmArc.from_file(larc) for larc in paths["measurements"]["arcs"]]
    sum_arc = lvmArc.from_file(paths["pixels"]["arcs"][0])
    traces = {method: {kind: TraceMask.from_file(paths[method][kind]) for kind in ("wave", "lsf")} for method in METHODS}
    headers = {method: fits.getheader(paths[method]["wave"]) for method in METHODS}

    # lines of each exposure and their combination, as in fit_wavelength_solution
    ref_lines, flux, cent, fwhm, masked = rss_tasks._combine_lvmarcs_lines(lvmarcs)
    flux_c, cent_c, fwhm_c, masked_c, offsets = rss_tasks._combine_arc_lines(flux, cent, fwhm, masked)
    cent_c = np.where(masked_c, np.nan, cent_c)
    fwhm_c = np.where(masked_c, np.nan, fwhm_c)
    fwhm_k = np.where(masked, np.nan, fwhm)
    flux_k = np.where(masked, np.nan, flux)
    # lines measured in the sum of the exposures
    cent_s, fwhm_s, _ = _lines_on(ref_lines, sum_arc)

    wave_m = traces["measurements"]["wave"]._data.astype(float)
    nfibers, npixels = wave_m.shape
    disp = np.abs(np.gradient(wave_m, axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        disp_lines = _at_lines(disp, cent_c)
        wave_lines = _at_lines(wave_m, cent_c)
        # line widths in Angstrom: single exposures (median over exposures) and the sum
        width_single = fwhm_c * disp_lines
        width_k = fwhm_k * disp_lines[None]
        width_sum = fwhm_s * disp_lines
        lsf_lines = {method: _at_lines(traces[method]["lsf"]._data.astype(float), cent_c) for method in METHODS}
        bias = {method: lsf_lines[method] - width_single for method in METHODS}

        # broadening of the sum predicted from the spread of the line positions among the exposures
        weights = np.nan_to_num(bn.nanmedian(flux_k, axis=2), nan=0.0)
        weights = np.where(np.isfinite(offsets), weights, 0.0)
        # fibers measured in fewer than two exposures (e.g., standard fibers) have no shifts between exposures
        multi = (weights > 0).sum(axis=0) >= 2
        weights[:, ~multi] = 0.0
        wsum = weights.sum(axis=0)
        mean_offset = np.divide((weights * offsets).sum(axis=0), wsum, where=wsum > 0, out=np.full(nfibers, np.nan))
        var_offset = np.divide((weights * (offsets - mean_offset[None]) ** 2).sum(axis=0), wsum, where=wsum > 0, out=np.full(nfibers, np.nan))
        predicted = np.where(multi, FWHM_FACTOR * var_offset, np.nan)
        observed = np.where(multi, bn.nanmedian(fwhm_s**2 - fwhm_c**2, axis=1), np.nan)

        lsf = {method: traces[method]["lsf"]._data.astype(float) for method in METHODS}
        lsf_ratio = lsf["pixels"] / lsf["measurements"] - 1
        wave_diff = bn.nanmedian(traces["pixels"]["wave"]._data - wave_m, axis=1)

    # fibers with a solution in both methods and lines measured
    good = np.ones(nfibers, dtype=bool)
    for method in METHODS:
        for kind in ("wave", "lsf"):
            mask = traces[method][kind]._mask
            if mask is not None:
                good &= ~mask.all(axis=1)
    good &= np.isfinite(width_single).sum(axis=1) > 0

    slitmap = lvmarcs[0]._slitmap
    if slitmap is not None and len(slitmap) != nfibers:
        slitmap = slitmap[slitmap["spectrographid"] == int(paths["camera"][1])]
    labels = np.asarray(slitmap["orig_ifulabel"]).astype(str) if slitmap is not None else np.arange(nfibers).astype(str)
    is_std = np.asarray(slitmap["telescope"] == "Spec") if slitmap is not None else np.zeros(nfibers, dtype=bool)

    fibers = pd.DataFrame({
        "camera": paths["camera"], "fiber": np.arange(nfibers), "label": labels, "good": good, "std": is_std,
        "lsf_measurements": bn.nanmedian(lsf["measurements"], axis=1), "lsf_pixels": bn.nanmedian(lsf["pixels"], axis=1),
        "lsf_change": bn.nanmedian(lsf_ratio, axis=1),
        "bias_measurements": bn.nanmedian(bias["measurements"], axis=1), "bias_pixels": bn.nanmedian(bias["pixels"], axis=1),
        "shift_range": np.nanmax(np.where(weights > 0, offsets, np.nan), axis=0) - np.nanmin(np.where(weights > 0, offsets, np.nan), axis=0),
        "predicted_dfwhm2": predicted, "observed_dfwhm2": observed, "wave_difference": wave_diff,
    })

    expnums = [larc._header.get("EXPOSURE") for larc in lvmarcs]
    return {
        "camera": paths["camera"], "narcs": len(lvmarcs), "expnums": expnums, "ref_lines": ref_lines,
        "wave_lines": wave_lines, "width_single": width_single, "width_k": width_k, "width_sum": width_sum,
        "bias": bias, "offsets": offsets, "weights": weights, "lsf": lsf, "lsf_ratio": lsf_ratio, "wave": wave_m,
        "good": good, "fibers": fibers, "headers": headers, "labels": labels,
    }


def _fit_rms(header, camera, kind):
    """Median RMS of the wavelength ('DISP') or LSF ('FWHM') fits stored in a master trace header"""
    # products written before the keywords were named after the camera use the PIPE prefix
    for key in (f"HIERARCH {camera} WAVE {kind} RMS MEDIAN", f"HIERARCH PIPE {kind} RMS MEDIAN"):
        if key in header:
            return header[key]
    return np.nan


def _summary_row(metrics):
    """Summary statistics of a camera, over its good fibers"""
    fibers = metrics["fibers"][metrics["fibers"].good]
    headers = metrics["headers"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ratio = np.divide(fibers.observed_dfwhm2, fibers.predicted_dfwhm2)
        ratio = ratio[fibers.predicted_dfwhm2 > 0.01]
        return {
            "camera": metrics["camera"], "narcs": metrics["narcs"], "nfibers": len(fibers),
            "lsf_measurements": np.nanmedian(fibers.lsf_measurements), "lsf_pixels": np.nanmedian(fibers.lsf_pixels),
            "lsf_change": np.nanmedian(fibers.lsf_change), "broader_fraction": np.nanmean(fibers.lsf_change > 0),
            "bias_measurements": np.nanmedian(fibers.bias_measurements), "bias_pixels": np.nanmedian(fibers.bias_pixels),
            "shift_range": np.nanmedian(fibers.shift_range),
            "predicted_dfwhm2": np.nanmedian(fibers.predicted_dfwhm2), "observed_dfwhm2": np.nanmedian(fibers.observed_dfwhm2),
            "explained_ratio": np.nanmedian(ratio) if len(ratio) else np.nan,
            "wave_difference": np.nanmedian(fibers.wave_difference),
            "wave_rms_measurements": _fit_rms(headers["measurements"], metrics["camera"], "DISP"),
            "wave_rms_pixels": _fit_rms(headers["pixels"], metrics["camera"], "DISP"),
            "lsf_rms_measurements": _fit_rms(headers["measurements"], metrics["camera"], "FWHM"),
            "lsf_rms_pixels": _fit_rms(headers["pixels"], metrics["camera"], "FWHM"),
        }


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


def _binned(wave, values, good, nbins=150):
    """Median wavelength and 16/50/84 percentiles over good fibers of `values` in bins of pixels"""
    edges = np.linspace(0, wave.shape[1], nbins + 1).astype(int)
    x, p16, p50, p84 = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lo, hi in zip(edges[:-1], edges[1:]):
            block = values[good, lo:hi]
            x.append(np.nanmedian(wave[good, lo:hi]))
            q = np.nanpercentile(block, [16, 50, 84])
            p16.append(q[0]); p50.append(q[1]); p84.append(q[2])
    return np.asarray(x), np.asarray(p16), np.asarray(p50), np.asarray(p84)


def fig_lsf(metrics):
    """Master LSF of both methods along the wavelength range: median over fibers and 16-84% range"""
    fig = go.Figure()
    for method in METHODS:
        x, p16, p50, p84 = _binned(metrics["wave"], metrics["lsf"][method], metrics["good"])
        color = METHOD_COLORS[method]
        fig.add_traces(_band(x, p16, p84, color, method))
        fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", line=dict(color=color, width=2), name=METHOD_LABELS[method],
                                 legendgroup=method, hovertemplate="%{x:.1f} &#8491;<br>FWHM %{y:.4f} &#8491;<extra>" + METHOD_LABELS[method] + "</extra>"))
    return _layout(fig, 380, "Wavelength (&#8491;)", "LSF FWHM (&#8491;)", hovermode="x unified")


def fig_ratio(metrics, nbins=120):
    """Map of the relative LSF difference across fibers and wavelength"""
    wave, ratio = metrics["wave"], 100 * metrics["lsf_ratio"]
    edges = np.linspace(0, wave.shape[1], nbins + 1).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z = np.stack([bn.nanmedian(ratio[:, lo:hi], axis=1) for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
        x = [np.nanmedian(wave[:, lo:hi]) for lo, hi in zip(edges[:-1], edges[1:])]
        z[~metrics["good"]] = np.nan
        zmax = np.nanpercentile(np.abs(z), 98) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(
        x=x, y=np.arange(z.shape[0]), z=z, zmid=0, zmin=-zmax, zmax=zmax, colorscale=DIVERGING, meta="div",
        customdata=np.repeat(metrics["labels"][:, None], z.shape[1], axis=1),
        colorbar=dict(title=dict(text="%", font=dict(color=THEME["ink2"])), tickfont=dict(color=THEME["muted"]), outlinewidth=0),
        hovertemplate="fiber %{y} (%{customdata})<br>%{x:.1f} &#8491;<br>%{z:+.2f}%<extra></extra>"))
    return _layout(fig, 420, "Wavelength (&#8491;)", "Fiber", margin=dict(l=64, r=16, t=24, b=52))


def fig_fiber(metrics, ifiber):
    """Line widths of one fiber: single exposures, their median, the sum, and both master LSFs"""
    fig = go.Figure()
    wave_lines = metrics["wave_lines"][ifiber]
    width_k = metrics["width_k"][:, ifiber]
    xk = np.broadcast_to(wave_lines, width_k.shape)
    good_k = np.isfinite(width_k)
    fig.add_trace(go.Scatter(x=xk[good_k], y=width_k[good_k], mode="markers", name="single exposures",
                             marker=dict(color=_alpha(THEME["muted"], 0.45), size=6, symbol="circle-open"),
                             hovertemplate="%{x:.2f} &#8491;<br>%{y:.4f} &#8491;<extra>single exposure</extra>"))
    for name, values, method in (("median of single exposures", metrics["width_single"][ifiber], "measurements"),
                                 ("measured in the sum", metrics["width_sum"][ifiber], "pixels")):
        good = np.isfinite(values) & np.isfinite(wave_lines)
        fig.add_trace(go.Scatter(x=wave_lines[good], y=values[good], mode="markers", name=name,
                                 marker=dict(color=METHOD_COLORS[method], size=9, line=dict(color=THEME["surface"], width=2)),
                                 hovertemplate="%{x:.2f} &#8491;<br>%{y:.4f} &#8491;<extra>" + name + "</extra>"))
    wave = metrics["wave"][ifiber]
    for method in METHODS:
        fig.add_trace(go.Scatter(x=wave[::8], y=metrics["lsf"][method][ifiber][::8], mode="lines", name=f"LSF, {METHOD_LABELS[method]}",
                                 line=dict(color=METHOD_COLORS[method], width=2, dash="solid" if method == "measurements" else "dash"),
                                 hovertemplate="%{x:.1f} &#8491;<br>%{y:.4f} &#8491;<extra>LSF " + METHOD_LABELS[method] + "</extra>"))
    return _layout(fig, 420, "Wavelength (&#8491;)", "FWHM (&#8491;)")


def fig_bias(metrics):
    """Master LSF minus the line widths of single exposures, median over fibers, for each line"""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = bn.nanmedian(metrics["wave_lines"][metrics["good"]], axis=0)
        for method in METHODS:
            values = 1000 * metrics["bias"][method][metrics["good"]]
            y = bn.nanmedian(values, axis=0)
            lo, hi = np.nanpercentile(values, 16, axis=0), np.nanpercentile(values, 84, axis=0)
            sort = np.argsort(x)
            fig.add_trace(go.Scatter(x=x[sort], y=y[sort], mode="markers+lines", name=METHOD_LABELS[method],
                                     line=dict(color=METHOD_COLORS[method], width=2), marker=dict(size=8, line=dict(color=THEME["surface"], width=2)),
                                     error_y=dict(type="data", symmetric=False, array=(hi - y)[sort], arrayminus=(y - lo)[sort],
                                                  color=_alpha(METHOD_COLORS[method], 0.5), thickness=1.5, width=0),
                                     hovertemplate="%{x:.2f} &#8491;<br>%{y:+.2f} m&#8491;<extra>" + METHOD_LABELS[method] + "</extra>"))
    return _layout(fig, 380, "Line wavelength (&#8491;)", "LSF &#8722; single-exposure FWHM (m&#8491;)")


def fig_shifts(metrics):
    """Line position offset of each exposure relative to the median of the exposures, per fiber"""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    offsets = np.where(metrics["weights"] > 0, metrics["offsets"], np.nan)
    narcs = offsets.shape[0]
    ramp = [EXPOSURE_RAMP[int(round(i * (len(EXPOSURE_RAMP) - 1) / max(narcs - 1, 1)))] for i in range(narcs)]
    fibers = np.arange(offsets.shape[1])
    for k in range(narcs):
        fig.add_trace(go.Scatter(x=fibers, y=offsets[k], mode="lines", line=dict(color=ramp[k], width=1.5), name=str(metrics["expnums"][k]),
                                 customdata=metrics["labels"],
                                 hovertemplate="fiber %{x} (%{customdata})<br>%{y:+.3f} pix<extra>exposure " + str(metrics["expnums"][k]) + "</extra>"))
    return _layout(fig, 400, "Fiber", "Offset along dispersion (pix)", legend=dict(title=dict(text="Exposure", font=dict(color=THEME["ink2"])),
                                                                                   orientation="v", x=1.01, y=1, xanchor="left", yanchor="top",
                                                                                   font=dict(color=THEME["ink2"]), bgcolor="rgba(0,0,0,0)"),
                   margin=dict(l=64, r=120, t=24, b=52))


def fig_explained(all_metrics):
    """Observed broadening of the lines in the sum against the broadening predicted from the shifts"""
    fig = go.Figure()
    top = 0.0
    for channel in "brz":
        frames = [m["fibers"][m["fibers"].good] for m in all_metrics if m["camera"][0] == channel]
        if not frames:
            continue
        fibers = pd.concat(frames)
        top = max(top, np.nanpercentile(np.concatenate([fibers.predicted_dfwhm2, fibers.observed_dfwhm2]), 99.5))
        fig.add_trace(go.Scatter(x=fibers.predicted_dfwhm2, y=fibers.observed_dfwhm2, mode="markers", name=f"{channel} cameras",
                                 marker=dict(color=CHANNEL_COLORS[channel], size=8, opacity=0.6, line=dict(color=THEME["surface"], width=1)),
                                 customdata=np.stack([fibers.camera, fibers.fiber, fibers.label], axis=1),
                                 hovertemplate="%{customdata[0]} fiber %{customdata[1]} (%{customdata[2]})<br>predicted %{x:.3f} pix&#178;"
                                               "<br>observed %{y:.3f} pix&#178;<extra></extra>"))
    top = top if np.isfinite(top) and top > 0 else 1.0
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=THEME["muted"], width=1.5, dash="dash"),
                             name="observed = predicted", hoverinfo="skip"))
    return _layout(fig, 460, "Predicted from the shifts, 8 ln2 &#963;&#178;<sub>shift</sub> (pix&#178;)",
                   "Observed FWHM&#178;<sub>sum</sub> &#8722; FWHM&#178;<sub>single</sub> (pix&#178;)")


def _fmt(value, fmt):
    return "&ndash;" if value is None or not np.isfinite(value) else format(value, fmt)


def _sample_fibers(metrics, nsample=8):
    """Evenly spaced good, non-standard fibers of a camera"""
    fibers = metrics["fibers"]
    candidates = fibers[fibers.good & ~fibers["std"]].fiber.to_numpy()
    if candidates.size == 0:
        return []
    return list(candidates[np.unique(np.linspace(0, candidates.size - 1, nsample).round().astype(int))])


def write_report(all_metrics, report_path, mjd_epoch, kind, mjds, problems=None):
    """Write the dashboard of the comparison

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

    figures = {"fig-explained": fig_explained(all_metrics)}
    fiber_options = {}
    for metrics in all_metrics:
        camera = metrics["camera"]
        figures[f"lsf|{camera}"] = fig_lsf(metrics)
        figures[f"ratio|{camera}"] = fig_ratio(metrics)
        figures[f"bias|{camera}"] = fig_bias(metrics)
        figures[f"shifts|{camera}"] = fig_shifts(metrics)
        fiber_options[camera] = _sample_fibers(metrics)
        for ifiber in fiber_options[camera]:
            figures[f"fiber|{camera}|{ifiber}"] = fig_fiber(metrics, ifiber)
    camera_options = [(camera, camera) for camera in cameras]
    # the fiber select lists the fibers of the first camera; other cameras use the same fiber indices when possible
    first_fibers = fiber_options[cameras[0]] if cameras else []
    for metrics in all_metrics[1:]:
        for ifiber in first_fibers:
            key = f"fiber|{metrics['camera']}|{ifiber}"
            if key not in figures and metrics["good"][ifiber]:
                figures[key] = fig_fiber(metrics, ifiber)
    fiber_select = [(str(ifiber), f"{ifiber} ({all_metrics[0]['labels'][ifiber]})") for ifiber in first_fibers]

    # tiles
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tiles = [
            ("Cameras compared", f"{len(cameras)}", f"of {len(cameras) + len(problems)} requested"),
            ("LSF change, sum vs per-arc", f"{100 * np.nanmedian(summary.lsf_change):+.2f}%", "median over cameras of the median over fibers"),
            ("Fibers with a broader LSF in the sum", f"{100 * np.nanmedian(summary.broader_fraction):.0f}%", "median over cameras"),
            ("Shift range between exposures", f"{np.nanmedian(summary.shift_range):.3f} pix", "median over fibers of max - min offset"),
            ("Broadening explained by the shifts", f"{np.nanmedian(summary.explained_ratio):.2f}", "observed / predicted, 1 = fully explained"),
            ("LSF bias vs single exposures", f"{1000 * np.nanmedian(summary.bias_measurements):+.1f} / {1000 * np.nanmedian(summary.bias_pixels):+.1f} mÅ",
             "per-arc / sum, median over cameras"),
        ]

    columns = [
        ("Camera", ""), ("Arcs", "Number of arc exposures"), ("Fibers", "Fibers with solutions in both methods"),
        ("LSF per-arc (Å)", "Median master LSF FWHM, per-arc measurements"), ("LSF sum (Å)", "Median master LSF FWHM, arcs summed in pixels"),
        ("Change", "Median over fibers of LSF(sum) / LSF(per-arc) - 1"), ("Broader", "Fraction of fibers with a broader LSF in the sum"),
        ("Bias per-arc (mÅ)", "Median of master LSF - single-exposure line width, per-arc method"),
        ("Bias sum (mÅ)", "Median of master LSF - single-exposure line width, sum method"),
        ("Shift range (pix)", "Median over fibers of the range of the exposure offsets"),
        ("Predicted (pix²)", "Median of 8 ln2 var(offsets)"), ("Observed (pix²)", "Median of FWHM²(sum) - FWHM²(single)"),
        ("Obs./pred.", "Median ratio over fibers with predicted > 0.01 pix²"),
        ("Δwave (mÅ)", "Median of wave(sum) - wave(per-arc)"),
        ("Wave RMS per-arc / sum (Å)", "Median RMS of the wavelength fits"), ("LSF RMS per-arc / sum (Å)", "Median RMS of the LSF fits"),
    ]
    rows = [[escape(row.camera), f"{row.narcs}", f"{row.nfibers}", _fmt(row.lsf_measurements, ".4f"), _fmt(row.lsf_pixels, ".4f"),
             _fmt(100 * row.lsf_change, "+.2f") + "%", _fmt(100 * row.broader_fraction, ".0f") + "%",
             _fmt(1000 * row.bias_measurements, "+.1f"), _fmt(1000 * row.bias_pixels, "+.1f"), _fmt(row.shift_range, ".3f"),
             _fmt(row.predicted_dfwhm2, ".3f"), _fmt(row.observed_dfwhm2, ".3f"), _fmt(row.explained_ratio, ".2f"),
             _fmt(1000 * row.wave_difference, "+.2f"),
             f"{_fmt(row.wave_rms_measurements, '.4f')} / {_fmt(row.wave_rms_pixels, '.4f')}",
             f"{_fmt(row.lsf_rms_measurements, '.4f')} / {_fmt(row.lsf_rms_pixels, '.4f')}"]
            for row in summary.itertuples()]
    rows += [[escape(camera), "&ndash;", "&ndash;", f'<span class="range">{escape(problem)}</span>'] + [""] * (len(columns) - 4)
             for camera, problem in problems.items()]
    summary_table = html_table(columns, rows,
                               numeric=[False] + [True] * (len(columns) - 1))

    fibers_table = pd.concat([metrics["fibers"] for metrics in all_metrics]) if all_metrics else pd.DataFrame()
    fiber_rows = [[escape(r.camera), f"{r.fiber}", escape(str(r.label)), _fmt(r.lsf_measurements, ".4f"), _fmt(r.lsf_pixels, ".4f"),
                   _fmt(100 * r.lsf_change, "+.2f") + "%", _fmt(r.shift_range, ".3f"), _fmt(r.predicted_dfwhm2, ".3f"), _fmt(r.observed_dfwhm2, ".3f")]
                  for r in fibers_table[fibers_table.good].itertuples()] if len(fibers_table) else []
    fiber_table_html = html_table([("Camera", ""), ("Fiber", ""), ("Label", ""), ("LSF per-arc (Å)", ""), ("LSF sum (Å)", ""),
                                   ("Change", ""), ("Shift range (pix)", ""), ("Predicted (pix²)", ""), ("Observed (pix²)", "")],
                                  fiber_rows, numeric=[False, True, False] + [True] * 6)

    cam_dim = [("Camera", camera_options)]
    sections = [
        f"""<section>
    <h2>Summary by camera</h2>
    <p>Both master solutions of each camera, fitted from the same extracted arcs. A negative bias means the master LSF is
    narrower than the lines in a single exposure, a positive one that it is broader.</p>
    {summary_table}
  </section>""",
        f"""<section>
    <h2>Master LSF along the wavelength range</h2>
    <p>Median over fibers of the master LSF of each method, with the 16&ndash;84% range across fibers shaded.</p>
    {picker("fig-lsf", "lsf", cam_dim, 380)}
  </section>""",
        f"""<section>
    <h2>Where the two LSFs differ</h2>
    <p>Relative difference LSF<sub>sum</sub> / LSF<sub>per-arc</sub> &minus; 1 for each fiber, in bins of wavelength.
    Red is broader in the sum. Bands across fibers follow how the exposure offsets vary along the slit.</p>
    {picker("fig-ratio", "ratio", cam_dim, 420)}
  </section>""",
        f"""<section>
    <h2>Against the line widths of single exposures</h2>
    <p>Master LSF minus the FWHM of each line measured in single exposures (median over exposures), for each line,
    median over fibers with the 16&ndash;84% range as error bars. This is the LSF a science exposure has, so the better
    method stays closer to zero.</p>
    {picker("fig-bias", "bias", cam_dim, 380)}
  </section>""",
        f"""<section>
    <h2>Line widths in one fiber</h2>
    <p>Each line's FWHM in every single exposure (open circles), their median, the FWHM measured in the sum, and both
    master LSFs. Fibers are evenly spaced along the slit, skipping standard fibers.</p>
    {picker("fig-fiber", "fiber", [("Camera", camera_options), ("Fiber", fiber_select)], 420)}
  </section>""",
        f"""<section>
    <h2>Shifts between exposures</h2>
    <p>Offset of the line positions of each exposure from the median of all exposures along the dispersion axis, per
    fiber (median over lines). These are removed before combining the per-arc measurements, but blur the lines of the sum.
    Lighter lines are earlier exposures.</p>
    {picker("fig-shifts", "shifts", cam_dim, 400)}
  </section>""",
        """<section>
    <h2>Is the broadening explained by the shifts?</h2>
    <p>For each fiber, how much wider the lines are in the sum than in single exposures, against the broadening expected
    from summing Gaussians shifted by the measured offsets. Points along the dashed line mean the broader LSF of the sum is
    entirely due to the shifts between exposures.</p>
    <div class="chart" data-fig="fig-explained" role="img" aria-label="Observed against predicted line broadening per fiber"></div>
  </section>""",
        f"""<section>
    <details><summary>Table view: all fibers</summary>{fiber_table_html}</details>
  </section>""",
        DEFINITIONS,
    ]

    meta = [("Epoch", mjd_epoch), ("Sequence", kind), ("Arc MJDs", ", ".join(map(str, mjds))), ("Cameras", ", ".join(cameras)),
            ("Pipeline", drpver), ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]
    intro = ("<p>Did fitting the LSF from lines measured in each arc improve on summing the arcs first? Both master "
             "solutions are fitted from the same extracted arcs of the epoch. The reference is the width of the lines in "
             "single exposures, which is the LSF a science exposure has. The equations are listed under "
             '<a href="#definitions">How the quantities are computed</a>.</p>')
    write_dashboard(report_path, f"Arc combination, epoch {mjd_epoch}", f"LVM DRP · wavelength · arc combination · {drpver}",
                    intro, meta, tiles, "\n\n  ".join(sections), figures, "lvmdrp.qa.wavelengths.qa_arc_combinations")
    return summary


def qa_arc_combinations(mjd_epoch, cameras=CAMERAS, kind="longterm", epochs=None, output_dir=None, extract=False,
                        nprocs=1, skip_done=True, dry_run=False):
    """Compare the master wavelength/LSF solutions from per-arc measurements and from arcs summed in pixels

    For each camera:

    1. Find the extracted arcs of the epoch's arc sequence (as in
       :func:`~lvmdrp.functions.run_calseq.create_wavelengths`). With `extract`,
       missing arcs are detrended and extracted first.
    2. Fit the master solutions with both methods (:func:`compare_camera`). All
       products go to `output_dir`; the master calibrations are not touched.
    3. Measure how they compare (:func:`camera_metrics`).

    Then write the dashboard and a per-fiber table (:func:`write_report`).

    Parameters
    ----------
    mjd_epoch : int
        MJD of the calibration epoch
    cameras : iterable[str], optional
        Cameras to compare, by default all
    kind : str, optional
        Arc sequence, either 'longterm' or 'nightly', by default 'longterm'
    epochs : dict, optional
        Calibration epochs as returned by :func:`~lvmdrp.functions.run_calseq.load_epochs_file`.
        If omitted, they are loaded from the default calibration epochs file
    output_dir : str, optional
        Directory of the dashboard and products. Default is ``wave_qa/{mjd_epoch}_{kind}_arcs`` in the
        epoch's ancillary directory
    extract : bool, optional
        Detrend and extract the arcs that are not extracted yet, by default False
    nprocs : int, optional
        Number of processes measuring the arcs, by default 1
    skip_done : bool, optional
        Reuse the products already in `output_dir`, by default True
    dry_run : bool, optional
        Only log the arcs found and the output paths, by default False

    Returns
    -------
    dict
        ``"summary"`` (pd.DataFrame), ``"report_path"`` and ``"table_path"``
    """
    from lvmdrp.functions.run_calseq import load_epochs_file, get_arc_sequence, extract_arcs, get_calib_paths
    from lvmdrp.main import reduce_2d
    from lvmdrp import path

    epochs = epochs if epochs is not None else load_epochs_file(epochs_kind="calibration", verbose=False)
    mjds, frames, expnums = get_arc_sequence(mjd=mjd_epoch, epochs=epochs, kind=kind)
    cameras = [camera for camera in CAMERAS if camera in cameras]
    output_dir = output_dir or _qa_dir(mjd_epoch, f"{mjd_epoch}_{kind}_arcs")
    report_path = os.path.join(output_dir, f"wave-arcs-qa_{mjd_epoch}_{kind}.html")
    table_path = os.path.join(output_dir, f"wave-arcs-qa_{mjd_epoch}_{kind}.csv")

    arcs_by_camera, problems = {}, {}
    for camera in cameras:
        arcs = frames[frames.camera == camera] if not frames.empty else frames
        if arcs.empty:
            problems[camera] = "no arcs in the sequence"
            continue
        xarcs = [path.full("lvm_anc", drpver=drpver, kind="x", imagetype="arc", **arc) for arc in arcs.to_dict("records")]
        arcs_by_camera[camera] = (arcs, xarcs)

    if dry_run:
        log.info(f"dry run of '{qa_arc_combinations.__name__}' for epoch {mjd_epoch} ({kind}), arcs from MJDs {mjds}")
        for camera, (arcs, xarcs) in arcs_by_camera.items():
            found = sum(os.path.isfile(xarc) for xarc in xarcs)
            log.info(f"  {camera}: {len(xarcs)} arcs (exposures {', '.join(map(str, arcs.expnum))}), {found} extracted")
        for camera, problem in problems.items():
            log.info(f"  {camera}: {problem}")
        log.info(f"  dashboard: {report_path}")
        log.info(f"  table: {table_path}")
        return {"summary": None, "report_path": report_path, "table_path": table_path}

    if extract and any(not os.path.isfile(xarc) for _, xarcs in arcs_by_camera.values() for xarc in xarcs):
        calibs = get_calib_paths(mjd=mjd_epoch, version=drpver, flavors=CALIBRATION_NEEDS["wave"], from_sandbox=False)
        reduce_2d(mjds, calibrations=calibs, expnums=expnums, cameras=list(arcs_by_camera), assume_imagetyp="arc",
                  reject_cr=False, add_astro=False, sub_straylight=False, skip_done=True)
        for camera, (arcs, _) in arcs_by_camera.items():
            extract_arcs(arcs, calibs=calibs, skip_done=True)

    all_metrics = []
    for camera, (arcs, xarcs) in arcs_by_camera.items():
        found = [xarc for xarc in xarcs if os.path.isfile(xarc)]
        if len(found) < 2:
            problems[camera] = f"{len(found)} of {len(xarcs)} arcs extracted, need at least two" + ("" if extract else " (try --extract)")
            log.error(f"{camera}: {problems[camera]}")
            continue
        try:
            paths = compare_camera(camera, found, output_dir, nprocs=nprocs, skip_done=skip_done)
            all_metrics.append(camera_metrics(paths))
        except Exception as error:
            problems[camera] = f"failed: {error}"
            log.error(f"{camera}: {problems[camera]}")

    if not all_metrics:
        log.error(f"no camera could be compared for epoch {mjd_epoch}: {problems}")
        return {"summary": None, "report_path": None, "table_path": None}

    summary = write_report(all_metrics, report_path, mjd_epoch, kind, mjds, problems=problems)
    pd.concat([metrics["fibers"] for metrics in all_metrics]).to_csv(table_path, index=False)
    log.info(f"wrote dashboard to '{report_path}' and per-fiber table to '{table_path}'")
    return {"summary": summary, "report_path": report_path, "table_path": table_path}


DEFINITIONS = r"""<section id="definitions">
    <h2>How the quantities are computed</h2>
    <p>Exposures are indexed by \(k\), fibers by \(i\) and lines by \(j\). \(x_{kij}\) and \(w_{kij}\) are the centroid
    and FWHM in pixels of a line measured in exposure \(k\), and \(D_i(x)\) the dispersion in &#8491;/pixel of the per-arc
    master wavelength solution.</p>
    <div class="defs">
      <div class="def">
        <h3>Exposure offsets</h3>
        <p>Offset of each exposure from the median line positions of all exposures, median over lines. The per-arc
        method removes these before combining the centroids.</p>
        <div class="eq">\[ o_{ki} = \mathrm{med}_j\left(x_{kij} - \mathrm{med}_k\, x_{kij}\right) \]</div>
      </div>
      <div class="def">
        <h3>Single-exposure line width</h3>
        <p>Median over exposures of each line's FWHM, in &#8491;. This is what the per-arc method fits, and the LSF a
        single science exposure has.</p>
        <div class="eq">\[ W_{ij} = D_i(\bar x_{ij})\, \mathrm{med}_k\, w_{kij} \]</div>
      </div>
      <div class="def">
        <h3>LSF bias</h3>
        <p>Master LSF of each method evaluated at the line position, minus the single-exposure line width.</p>
        <div class="eq">\[ b_{ij} = \mathrm{LSF}_i(\bar x_{ij}) - W_{ij} \]</div>
      </div>
      <div class="def">
        <h3>Predicted broadening of the sum</h3>
        <p>Summing Gaussians shifted by \(o_{ki}\) adds the variance of the shifts to that of the line. The exposures
        are weighted by their median line flux \(f_{ki}\), as in the sum.</p>
        <div class="eq">\[ \Delta^{\mathrm{pred}}_i = 8\ln 2\, \frac{\sum_k f_{ki}\,(o_{ki} - \bar o_i)^2}{\sum_k f_{ki}} \]</div>
      </div>
      <div class="def">
        <h3>Observed broadening of the sum</h3>
        <p>Squared FWHM of the lines measured in the sum, minus that of single exposures, median over lines, in
        pixels&#178;.</p>
        <div class="eq">\[ \Delta^{\mathrm{obs}}_i = \mathrm{med}_j\left(w^{\mathrm{sum}\,2}_{ij} - (\mathrm{med}_k\, w_{kij})^2\right) \]</div>
      </div>
      <div class="def">
        <h3>LSF change</h3>
        <p>Relative difference of the master LSFs, median over the pixels of each fiber.</p>
        <div class="eq">\[ c_i = \mathrm{med}_x\left(\frac{\mathrm{LSF}^{\mathrm{sum}}_i(x)}{\mathrm{LSF}^{\mathrm{per\text{-}arc}}_i(x)} - 1\right) \]</div>
      </div>
    </div>
    <p class="note">Single-exposure widths come from lines with less signal than the sum. Above a peak S/N of about 5 the
    fitted FWHM of a Gaussian is unbiased, so this mainly adds scatter; faint lines are rejected by the flux limit of
    the line fitting.</p>
  </section>"""
