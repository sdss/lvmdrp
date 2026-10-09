# encoding: utf-8
"""Pixel-flat QA dashboards.

Two self-contained interactive HTML dashboards, built with the shared QA report
theme and page (:mod:`lvmdrp.qa.report`):

- :func:`qa_raw_pixelflats`: the raw pixel-flat sequence of one epoch, as
  defined in the epochs file. For each camera it describes the sequence and
  its completeness, the signal and photons reached using the corrected header gains, the
  lamp stability, the dark and bias levels, the read noise, the saturation and
  the illumination pattern.
- :func:`qa_pixelflats`: the master pixel flats produced for a target epoch,
  compared against a reference epoch: 1:1 comparison, artifacts in the master
  pixel flats and artifacts in a flat-fielded test exposure.
"""

import os
from datetime import datetime, timezone
from html import escape
from multiprocessing import Pool

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.io import fits
from plotly.subplots import make_subplots
from tqdm import tqdm

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.core.constants import CAMERAS
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import pixelflats as pf
from lvmdrp.qa.report import (
    MONO_FONT, SERIES, STATUS_COLORS, THEME,
    badge, base_layout, epoch_qa_dir, html_table, issues_html, picker, restyle, style_axes, write_dashboard,
)


# Parts of the pixel-flat products QA, in the order they run (see qa_pixelflats)
QA_PARTS = ("comparison", "artifacts", "flatfielded")

# exposure types in a sequence kind, e.g. '2f1d'
ROLE_TYPES = {"f": "flat", "d": "dark", "b": "bias"}
ROLE_PLURALS = {"flat": "flats", "dark": "darks", "bias": "biases"}
# IMAGETYP header values expected for each exposure type
EXPECTED_IMAGETYP = {"flat": ("object", "flat"), "dark": ("dark",), "bias": ("bias",)}
# lamp header keywords and the values meaning the lamp is on (as in lvmdrp.utils.metadata)
LAMPS = ("LDLS", "QUARTZ", "NEON", "HGNE", "ARGON", "XENON", "KRYPTON")
ONLAMP = ("ON", True, "T", 1)
NQUADS = 4
# line dash of each spectrograph in the charts
SPEC_DASHES = {"1": "solid", "2": "dash", "3": "dot"}


def _camera_color(camera):
    """Color of a camera's channel: blue, red and near-infrared in the series order."""
    return SERIES["brz".index(camera[0])] if camera and camera[0] in "brz" else THEME["muted"]


def _plural(count, role):
    return f"{count} {role if count == 1 else ROLE_PLURALS.get(role, role + 's')}"


def _format_expnums(items):
    """Format exposure numbers as written in the epochs file: half-open 'start, stop' ranges become 'start-last'."""
    parts = []
    for item in items or []:
        if isinstance(item, str) and "," in item:
            start, stop = (int(value) for value in item.split(","))
            parts.append(f"{start}–{stop - 1}" if stop - 1 > start else str(start))
        else:
            parts.append(str(item))
    return ", ".join(parts) or "–"


def _num(value, fmt=".2f", missing="–"):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return missing
    return format(value, fmt) if np.isfinite(value) else missing


def _qa_dir(mjd_epoch, name):
    """Return the default directory of a pixel-flat QA dashboard.

    The directory sits in the ancillary directory of the given epoch, in the
    reductions of the current pipeline version (see
    :func:`~lvmdrp.qa.report.epoch_qa_dir`).

    Parameters
    ----------
    mjd_epoch : int
        MJD of the pixel-flat epoch.
    name : str
        Name of the subdirectory, e.g. ``"60741_raw"``.

    Returns
    -------
    str
        Path to the QA directory.
    """
    return epoch_qa_dir(drpver, mjd_epoch, "pixflat", name)


# ---------------------------------------------------------------------------
# raw pixel-flat sequences
# ---------------------------------------------------------------------------

def _find_raw(sources, camera, expnum):
    """Return the raw path of an exposure in any of the source MJDs, or None."""
    for mjd in np.atleast_1d(sources):
        raw_path = path.full("lvm_raw", hemi="s", mjd=int(mjd), camspec=camera, expnum=int(expnum))
        if os.path.isfile(raw_path):
            return raw_path
    return None


def describe_sequence(epoch, camera):
    """Describe a camera's pixel-flat sequence as defined in the epochs file.

    Every exposure gets the role implied by its position in the sequence
    ``kind`` (e.g. ``'2f1d'``: two flats then one dark, repeated), after
    removing the rejects. Exposures of an incomplete last group are kept as
    ``"unassigned"``, and the rejects as ``"rejected"``. For ``kind: auto``,
    roles come from the ``imagetyp`` metadata as in the reduction (see
    :func:`~lvmdrp.functions.pixelflats._classify_sequence`), and exposures
    with no flat, dark or bias ``imagetyp``, or missing from the metadata,
    are kept as ``"unclassified"``. The raw file of each exposure is searched
    for in the epoch's source MJDs.

    Parameters
    ----------
    epoch : dict
        Epoch definition as returned by :func:`~lvmdrp.functions.pixelflats.load_pixflat_epochs`.
    camera : str
        Camera identifier, e.g. ``"b1"``.

    Returns
    -------
    dict
        ``camera``, ``kind``, ``kind_text`` (e.g. "2 flats + 1 dark"),
        ``group_size``, ``ngroups`` (complete groups), ``nleftover``
        (exposures in an incomplete last group), the ``expnums`` and
        ``rejects`` as written in the file, and ``exposures``, a DataFrame with
        one row per exposure: ``expnum``, ``role`` and ``path`` (None if the
        raw file is missing). For ``kind: auto``, ``kind_text`` gives the
        counts per type, ``group_size`` is None, ``ngroups`` is the number of
        runs of consecutive flats, and ``nleftover`` is 0.

    Raises
    ------
    KeyError
        If the epoch has no sequence for ``camera``.
    ValueError
        If the sequence ``kind`` is malformed.
    """
    sequence = epoch["sequences"][camera]
    kind = sequence.get("kind", "")
    expnums = pf._parse_expnums(sequence.get("expnums") or [])
    rejects = set(int(expnum) for expnum in pf._parse_expnums(sequence.get("rejects") or []))
    effective = [int(expnum) for expnum in expnums if expnum not in rejects]

    if kind == pf.AUTO_KIND:
        groups, _ = pf._classify_sequence(pf._parse_sequence(sequence, expand=False), mjds=epoch["sources"], camera=camera)
        roles = {int(expnum): role for role, role_expnums in groups.items() for expnum in role_expnums}
        exposures = [{"expnum": expnum, "role": roles.get(expnum, "unclassified")} for expnum in effective]
        counts = " + ".join(_plural(len(role_expnums), role) for role, role_expnums in groups.items() if len(role_expnums))
        kind_text = f"{counts}, by IMAGETYP" if counts else "by IMAGETYP, none classified"
        # without a pattern, a group is a run of consecutive flats with the exposures that follow it
        flags = [exposure["role"] == "flat" for exposure in exposures]
        ngroups = sum(flag and (i == 0 or not flags[i - 1]) for i, flag in enumerate(flags))
        group_size, nleftover = None, 0
    else:
        pairs = pf._parse_kind(kind)
        pattern = [ROLE_TYPES[typ] for count, typ in pairs for _ in range(count)]
        ngroups, nleftover = divmod(len(effective), len(pattern))
        exposures = [
            {"expnum": expnum, "role": pattern[i % len(pattern)] if i < ngroups * len(pattern) else "unassigned"}
            for i, expnum in enumerate(effective)
        ]
        kind_text = " + ".join(_plural(count, ROLE_TYPES[typ]) for count, typ in pairs)
        group_size = len(pattern)

    exposures += [{"expnum": expnum, "role": "rejected"} for expnum in sorted(rejects)]
    for exposure in exposures:
        exposure["path"] = _find_raw(epoch["sources"], camera, exposure["expnum"])

    return {
        "camera": camera, "kind": kind, "kind_text": kind_text,
        "group_size": group_size, "ngroups": ngroups, "nleftover": nleftover,
        "expnums": sequence.get("expnums") or [], "rejects": sequence.get("rejects") or [],
        # object dtype keeps missing paths as None, where newer pandas would infer a string column with NaN
        "exposures": pd.DataFrame(exposures, columns=["expnum", "role", "path"], dtype=object)
                       .astype({"expnum": int}).sort_values("expnum", ignore_index=True),
    }


def _quadrant_sections(header, keyword, defaults):
    sections = [header.get(f"{keyword}{i}") for i in range(1, NQUADS + 1)]
    return list(defaults) if any(section is None for section in sections) else sections


def measure_raw_frame(raw_path, saturation=65000.0):
    """Measure the count levels of a raw frame, per quadrant.

    For each quadrant, the overscan level and noise are measured in the
    ``BIASSEC`` region and the illumination in the ``TRIMSEC`` region, and
    converted to electrons with the ``GAIN`` header values divided by the
    pipeline's gain corrections, as in the preprocessing (see
    :func:`~lvmdrp.functions.imageMethod.correct_gains`), or with the default
    gains if the header has none.

    Parameters
    ----------
    raw_path : str
        Path of the raw frame.
    saturation : float, optional
        Raw count level (ADU) at and above which a pixel counts as saturated.
        Default is 65000.

    Returns
    -------
    dict
        Header values (``camera``, ``expnum``, ``mjd``, ``imagetyp``,
        ``exptime``, ``obstime``, ``lamps``, ``gain_source``) and, for each
        quadrant ``q``: ``gain_q`` (e-/ADU), ``overscan_q`` (ADU),
        ``rdnoise_q`` (e-, from the overscan), ``signal_q``, ``signal_p05_q``
        and ``signal_p95_q`` (median, 5th and 95th percentile of the
        overscan-subtracted science region, e-/pixel), and ``saturated_q``
        (fraction of science pixels at or above ``saturation``).
    """
    with fits.open(raw_path) as hdul:
        header = hdul[0].header
        data = hdul[0].data.astype(np.float32)

    camera = header.get("CCD", "")
    gains = [header.get(f"GAIN{i}") for i in range(1, NQUADS + 1)]
    gain_source = "header"
    if any(gain is None for gain in gains):
        gains, gain_source = image_tasks.DEFAULT_GAIN.get(camera, [np.nan] * NQUADS), "default"
    else:
        gains = image_tasks.correct_gains(camera, gains)

    row = {
        "path": raw_path, "camera": camera, "expnum": int(header.get("EXPOSURE", -1)), "mjd": int(header.get("MJD", -1)),
        "imagetyp": str(header.get("IMAGETYP", "")).lower(), "exptime": float(header.get("EXPTIME", np.nan)),
        "obstime": str(header.get("OBSTIME", "")),
        "lamps": ",".join(lamp for lamp in LAMPS if header.get(lamp, "OFF") in ONLAMP) or "none",
        "gain_source": gain_source,
    }
    trimsecs = _quadrant_sections(header, "TRIMSEC", image_tasks.DEFAULT_TRIMSEC)
    biassecs = _quadrant_sections(header, "BIASSEC", image_tasks.DEFAULT_BIASSEC)
    for q, (trimsec, biassec, gain) in enumerate(zip(trimsecs, biassecs, gains), start=1):
        (x0, x1), (y0, y1) = image_tasks._parse_ccd_section(trimsec)
        (bx0, bx1), (by0, by1) = image_tasks._parse_ccd_section(biassec)
        overscan = data[by0:by1, bx0:bx1]
        science = data[y0:y1, x0:x1]
        os_level = float(np.median(overscan))
        p05, p50, p95 = np.percentile(science, [5, 50, 95])
        row.update({
            f"gain_{q}": float(gain), f"overscan_{q}": os_level,
            f"rdnoise_{q}": float(1.4826 * np.median(np.abs(overscan - os_level)) * gain),
            f"signal_{q}": float((p50 - os_level) * gain), f"signal_p05_{q}": float((p05 - os_level) * gain),
            f"signal_p95_{q}": float((p95 - os_level) * gain),
            f"saturated_{q}": float(np.count_nonzero(science >= saturation) / science.size),
        })
    return row


def _measure_task(task):
    """Measure one raw frame for the pool, returning (path, row, error)."""
    raw_path, saturation = task
    try:
        return raw_path, measure_raw_frame(raw_path, saturation=saturation), None
    except Exception as error:
        return raw_path, None, f"{type(error).__name__}: {error}"


class _SerialPool:
    """Minimal stand-in for multiprocessing.Pool running in the current process"""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def imap_unordered(self, func, iterable):
        return map(func, iterable)


def _quad_columns(name):
    return [f"{name}_{q}" for q in range(1, NQUADS + 1)]


def _raw_camera_stats(description, frames, target_precision, max_level_deviation, max_saturated, max_dark_signal):
    """Summarize the measured frames of a camera's sequence and list its issues.

    Parameters
    ----------
    description : dict
        Sequence description as returned by :func:`describe_sequence`.
    frames : pandas.DataFrame
        The camera's exposures merged with their measurements
        (see :func:`measure_raw_frame`), with ``role`` and ``error`` columns.
    target_precision : float
        Poisson-limited precision the accumulated flats should reach.
    max_level_deviation : float
        Largest deviation of a flat's level from the sequence median before it
        counts as an outlier.
    max_saturated : float
        Largest fraction of saturated pixels allowed in a flat quadrant.
    max_dark_signal : float
        Largest signal (e-/pixel) allowed in a dark.

    Returns
    -------
    dict
        Counts per exposure type, per-quadrant accumulated signal and
        precision, level stability, saturation, dark and bias levels, read
        noise, the ``issues`` found and the resulting ``status`` (``"ok"``,
        ``"warn"`` or ``"bad"``).
    """
    exposures = description["exposures"]
    stats = {"camera": description["camera"], "issues": []}
    issues = stats["issues"]

    for role in ("flat", "dark", "bias"):
        selected = exposures[exposures.role == role]
        stats[f"n{role}_expected"] = len(selected)
        stats[f"n{role}_found"] = int(selected.path.notna().sum())
    missing = exposures[(exposures.role != "rejected") & exposures.path.isna()]
    stats["missing"] = missing.expnum.tolist()
    if len(missing):
        issues.append(f"{len(missing)} raw file(s) missing: {pf._compress_expnums(missing.expnum)}")
    if description["nleftover"]:
        issues.append(f"incomplete last group: {description['nleftover']} of {description['group_size']} exposures")
    unclassified = exposures[exposures.role == "unclassified"]
    if len(unclassified):
        issues.append(f"{len(unclassified)} exposure(s) not used, unexpected IMAGETYP or missing from metadata: {pf._compress_expnums(unclassified.expnum)}")

    measured = frames[frames.error.fillna("") == ""] if "error" in frames else frames.iloc[0:0]
    failed = frames[frames.error.fillna("") != ""] if "error" in frames else frames.iloc[0:0]
    if len(failed):
        issues.append(f"{len(failed)} frame(s) could not be measured")

    # IMAGETYP of each exposure against its role in the sequence
    mismatched = [row.expnum for row in measured.itertuples() if row.role in EXPECTED_IMAGETYP and row.imagetyp not in EXPECTED_IMAGETYP[row.role]]
    stats["imagetyp_mismatches"] = mismatched
    if mismatched:
        issues.append(f"{len(mismatched)} IMAGETYP mismatch(es): {pf._compress_expnums(mismatched)}")
    if (measured.get("gain_source", pd.Series(dtype=str)) == "default").any():
        issues.append("gains missing in some headers, used the default gains")

    flats = measured[measured.role == "flat"]
    stats["nflat_measured"] = len(flats)
    stats["exptimes"] = sorted(set(flats.exptime.round(2))) if len(flats) else []
    stats["lamps"] = sorted(set(flats.lamps)) if len(flats) else []
    if len(stats["exptimes"]) > 1:
        issues.append(f"flats with different exposure times: {stats['exptimes']} s")
    if len(stats["lamps"]) > 1:
        issues.append(f"flats with different lamps on: {stats['lamps']}")

    if len(flats):
        signal = flats[_quad_columns("signal")].to_numpy()
        signal_dim = flats[_quad_columns("signal_p05")].to_numpy()
        rdnoise = np.median(flats[_quad_columns("rdnoise")].to_numpy(), axis=0)
        nflats = len(flats)
        total = signal.sum(axis=0)
        total_dim = signal_dim.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.sqrt(total + nflats * rdnoise ** 2) / total
            precision_dim = np.sqrt(total_dim + nflats * rdnoise ** 2) / total_dim
        stats.update({
            "signal_per_flat": float(np.median(signal.mean(axis=1))),
            "signal_per_flat_q": np.median(signal, axis=0), "signal_dim_per_flat_q": np.median(signal_dim, axis=0),
            "total_q": total, "total_dim_q": total_dim, "precision_q": precision, "precision_dim_q": precision_dim,
            "rdnoise_q": rdnoise, "gain_q": np.median(flats[_quad_columns("gain")].to_numpy(), axis=0),
            "precision": float(np.nanmax(precision)), "precision_dim": float(np.nanmax(precision_dim)),
            "total_min": float(np.nanmin(total)), "worst_quad": int(np.nanargmax(precision)) + 1,
        })
        level = signal.mean(axis=1)
        deviation = level / np.median(level) - 1
        outliers = flats.expnum[np.abs(deviation) > max_level_deviation].tolist()
        stats.update({
            "level_scatter": float(1.4826 * np.median(np.abs(deviation - np.median(deviation)))),
            "level_max_deviation": float(np.max(np.abs(deviation))), "level_outliers": outliers,
            "saturated_max": float(flats[_quad_columns("saturated")].to_numpy().max()),
        })
        if outliers:
            issues.append(f"{len(outliers)} flat(s) off the median level by > {100 * max_level_deviation:g}%: {pf._compress_expnums(outliers)}")
        if stats["saturated_max"] > max_saturated:
            issues.append(f"saturated pixels in flats: up to {100 * stats['saturated_max']:.3g}% of a quadrant")
        if stats["precision"] > target_precision:
            issues.append(f"Poisson precision {100 * stats['precision']:.3f}% (AMP{stats['worst_quad']}) "
                          f"worse than the {100 * target_precision:g}% target")
    else:
        issues.append("no flats measured")

    for role in ("dark", "bias"):
        selected = measured[measured.role == role]
        stats[f"{role}_signal"] = float(np.median(selected[_quad_columns("signal")].to_numpy())) if len(selected) else np.nan
        stats[f"{role}_lamps_on"] = int((selected.lamps != "none").sum()) if len(selected) else 0
        if stats[f"{role}_lamps_on"]:
            issues.append(f"{_plural(stats[f'{role}_lamps_on'], role)} taken with a lamp on")
    if np.isfinite(stats["dark_signal"]) and stats["dark_signal"] > max_dark_signal:
        issues.append(f"dark signal {stats['dark_signal']:.1f} e-/pixel above {max_dark_signal:g}")

    stats["status"] = "bad" if not len(flats) else ("warn" if issues else "ok")
    return stats


def figure_sequence_strip(descriptions, frames, outliers):
    """Exposures of each camera's sequence along the exposure number.

    Parameters
    ----------
    descriptions : dict[str, dict]
        Sequence description per camera (see :func:`describe_sequence`).
    frames : pandas.DataFrame
        Measured frames, with ``camera``, ``expnum`` and ``signal`` columns.
    outliers : dict[str, list[int]]
        Flats off the median level, per camera.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    styles = {
        "flat": ("Flat", SERIES[0], "circle"), "dark": ("Dark", SERIES[1], "square"), "bias": ("Bias", SERIES[2], "diamond"),
        "unassigned": ("Incomplete group", THEME["muted"], "triangle-up"), "unclassified": ("Unclassified IMAGETYP", THEME["muted"], "triangle-down"),
        "rejected": ("Rejected", THEME["muted"], "x-thin-open"),
    }
    rows = []
    for camera, description in descriptions.items():
        for exposure in description["exposures"].itertuples():
            rows.append((camera, exposure.expnum, exposure.role, pd.notna(exposure.path)))
    table = pd.DataFrame(rows, columns=["camera", "expnum", "role", "found"])
    if len(frames):
        table = table.merge(frames[["camera", "expnum", "imagetyp", "lamps", "signal"]], on=["camera", "expnum"], how="left")
    else:
        table = table.assign(imagetyp="", lamps="", signal=np.nan)

    fig = go.Figure()
    for role, (label, color, symbol) in styles.items():
        selected = table[(table.role == role) & (table.found | (role == "rejected"))]
        if len(selected) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=selected.expnum, y=selected.camera, mode="markers", name=label,
            marker=dict(size=9, color=color, symbol=symbol, line=dict(width=1.5 if symbol.endswith("open") else 1, color=color if symbol.endswith("open") else THEME["surface"])),
            customdata=np.stack([selected.imagetyp.fillna("").astype(str), selected.lamps.fillna("").astype(str), _fmt_col(selected.signal, ",.0f")], axis=-1),
            hovertemplate=f"<b>%{{y}}</b> %{{x}}<br>{label.lower()}, IMAGETYP %{{customdata[0]}}<br>lamps %{{customdata[1]}}<br>"
                          "signal %{customdata[2]} e-/pixel<extra></extra>"))
    missing = table[~table.found & (table.role != "rejected")]
    if len(missing):
        fig.add_trace(go.Scatter(
            x=missing.expnum, y=missing.camera, mode="markers", name="Missing file",
            marker=dict(size=11, color=STATUS_COLORS["bad"], symbol="circle-open", line=dict(width=2, color=STATUS_COLORS["bad"])),
            customdata=missing.role, hovertemplate="<b>%{y}</b> %{x}<br>%{customdata}, raw file missing<extra></extra>"))
    flagged = [(camera, expnum) for camera, expnums in outliers.items() for expnum in expnums]
    if flagged:
        fig.add_trace(go.Scatter(
            x=[expnum for _, expnum in flagged], y=[camera for camera, _ in flagged], mode="markers", name="Flat off level",
            marker=dict(size=16, color=STATUS_COLORS["warn"], symbol="circle-open", line=dict(width=2, color=STATUS_COLORS["warn"])),
            hovertemplate="<b>%{y}</b> %{x}<br>flat level off the median<extra></extra>"))

    cameras = list(descriptions)
    fig.update_layout(base_layout(110 + 36 * max(len(cameras), 1), xaxis_title="Exposure number", margin=dict(l=56, r=16, t=40, b=52)))
    style_axes(fig)
    fig.update_xaxes(tickformat="d")
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=cameras[::-1],
                     tickfont=dict(family=MONO_FONT, color=THEME["ink2"]))
    return fig


def _fmt_col(values, fmt):
    return np.array([_num(value, fmt) for value in values])


def figure_photons(stats, target_precision):
    """Signal accumulated over each camera's flats, per quadrant.

    Parameters
    ----------
    stats : dict[str, dict]
        Camera statistics as returned by :func:`_raw_camera_stats`.
    target_precision : float
        Target Poisson precision, drawn as the signal needed to reach it.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cameras = [camera for camera, stat in stats.items() if "total_q" in stat]
    fig = go.Figure()
    for kind, label, color, symbol in (("total_q", "Median illumination", SERIES[0], "circle"),
                                       ("total_dim_q", "Dimmest 5% of pixels", SERIES[1], "circle-open")):
        x, y, customdata = [], [], []
        for i, camera in enumerate(cameras):
            precision = stats[camera]["precision_q" if kind == "total_q" else "precision_dim_q"]
            for q in range(NQUADS):
                x.append(i + (q - 1.5) * 0.14)
                y.append(stats[camera][kind][q])
                customdata.append([camera, q + 1, _num(100 * precision[q], ".3f"), stats[camera]["nflat_measured"]])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=label, customdata=customdata,
            marker=dict(size=10, color=color, symbol=symbol, line=dict(width=2, color=color if symbol.endswith("open") else THEME["surface"])),
            hovertemplate="<b>%{customdata[0]}</b> AMP%{customdata[1]}<br>%{y:,.0f} e-/pixel over %{customdata[3]} flats"
                          "<br>Poisson precision %{customdata[2]}%<extra></extra>"))
    needed = 1 / target_precision ** 2
    fig.add_hline(y=needed, line=dict(color=THEME["muted"], width=1.5, dash="dash"),
                  annotation=dict(text=f"{100 * target_precision:g}% precision", font=dict(color=THEME["muted"], size=12)),
                  annotation_position="top left")
    fig.update_layout(base_layout(400, yaxis_title="Accumulated signal (e-/pixel)", yaxis_type="log"))
    style_axes(fig)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(cameras))), ticktext=cameras, showgrid=False, zeroline=False,
                     range=[-0.5, len(cameras) - 0.5], tickfont=dict(family=MONO_FONT, color=THEME["ink2"]))
    fig.update_yaxes(tickvals=[m * 10 ** e for e in range(3, 9) for m in (1, 2, 5)], tickformat="~s")
    return fig


def figure_stability(frames, max_level_deviation):
    """Level of each flat relative to its sequence median, per camera.

    Parameters
    ----------
    frames : pandas.DataFrame
        Measured flats, with ``camera``, ``expnum``, ``signal`` and
        ``deviation`` columns.
    max_level_deviation : float
        Deviation drawn as the tolerance band.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    band = 100 * max_level_deviation
    fig.add_hrect(y0=-band, y1=band, fillcolor=THEME["grid"], opacity=0.5, line_width=0, layer="below")
    for camera in CAMERAS:
        flats = frames[frames.camera == camera].sort_values("expnum")
        if len(flats) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(flats) + 1), y=100 * flats.deviation, mode="lines+markers", name=camera,
            line=dict(width=1.5, color=_camera_color(camera), dash=SPEC_DASHES.get(camera[1:], "solid")),
            marker=dict(size=5, color=_camera_color(camera)),
            customdata=np.stack([flats.expnum, _fmt_col(flats.signal, ",.0f")], axis=-1),
            hovertemplate=f"<b>{camera}</b> flat %{{x}} (exposure %{{customdata[0]}})<br>%{{y:+.2f}}% from the median"
                          "<br>%{customdata[1]} e-/pixel<extra></extra>"))
    fig.update_layout(base_layout(380, xaxis_title="Flat in sequence", yaxis_title="Level from median (%)"))
    style_axes(fig)
    fig.update_xaxes(tickformat="d")
    return fig


def figure_darks(frames):
    """Signal of the darks and biases of each camera's sequence.

    Parameters
    ----------
    frames : pandas.DataFrame
        Measured darks and biases, with ``camera``, ``role``, ``expnum``,
        ``exptime`` and ``signal`` columns.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    for role, symbol in (("dark", "square"), ("bias", "diamond")):
        for camera in CAMERAS:
            selected = frames[(frames.camera == camera) & (frames.role == role)].sort_values("expnum")
            if len(selected) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=np.arange(1, len(selected) + 1), y=selected.signal, mode="lines+markers", name=f"{camera} {role}",
                line=dict(width=1, color=_camera_color(camera), dash=SPEC_DASHES.get(camera[1:], "solid")),
                marker=dict(size=7, symbol=symbol, color=_camera_color(camera)),
                customdata=np.stack([selected.expnum, selected.exptime, selected.lamps], axis=-1),
                hovertemplate=f"<b>{camera}</b> {role} %{{x}} (exposure %{{customdata[0]}})<br>%{{y:.1f}} e-/pixel"
                              "<br>%{customdata[1]} s, lamps %{customdata[2]}<extra></extra>"))
    fig.update_layout(base_layout(340, xaxis_title="Exposure of its type in sequence", yaxis_title="Signal over overscan (e-/pixel)"))
    style_axes(fig)
    fig.update_xaxes(tickformat="d")
    return fig


def qa_raw_pixelflats(mjd_epoch, cameras=CAMERAS, epochs=None, output_dir=None, nprocs=1,
                      cut_rows=(1020, 3060), cut_columns=(1021, 3098), cut_width=21, combined=False,
                      image_binning=16, n_contrast=5, contrast_window=51, saturation=65000.0,
                      target_precision=0.002, max_level_deviation=0.02, max_saturated=1e-4,
                      max_dark_signal=20.0, skip_done=True, dry_run=False):
    """Run the QA of the raw pixel-flat sequences of an epoch and write a dashboard.

    For each camera with a sequence in the epochs file:

    1. Describe the sequence (:func:`describe_sequence`): its kind, the role of
       each exposure, the rejects, and which raw files are on disk.
    2. Measure every non-rejected raw frame (:func:`measure_raw_frame`): the
       overscan level and read noise, and the illumination in electrons using
       the corrected header gains, per quadrant.
    3. Summarize the sequence (:func:`_raw_camera_stats`): completeness,
       IMAGETYP of each exposure against its role, signal accumulated over the
       flats and the Poisson-limited precision it allows, level stability,
       saturation, lamps, dark and bias levels; and list the issues found.
    4. Plot cuts along X and Y of the first flat (or the average of the flats)
       with the contrast from the CCD center towards the edge
       (:func:`~lvmdrp.functions.pixelflats.display_raw_cuts`).

    The measurements are written to a CSV table next to the dashboard. With
    ``skip_done``, frames already in that table are not measured again.

    Parameters
    ----------
    mjd_epoch : int
        MJD of the pixel-flat epoch.
    cameras : iterable[str], optional
        Cameras to check. Default is all of :data:`CAMERAS`.
    epochs : dict, optional
        Epoch mapping as returned by :func:`~lvmdrp.functions.pixelflats.load_pixflat_epochs`.
        If omitted, it is loaded from the default epochs file.
    output_dir : str, optional
        Directory of the dashboard and the measurements table. Default is
        ``pixflat_qa/{mjd_epoch}_raw`` in the epoch's ancillary directory.
    nprocs : int, optional
        Number of processes measuring the raw frames. Default is 1.
    cut_rows, cut_columns : iterable[int], optional
        1-based rows and columns of the illumination cuts. Default is
        (1020, 3060) and (1021, 3098), the quadrant centers.
    cut_width : int, optional
        Width in pixels of the band collapsed into each cut. Default is 21.
    combined : bool, optional
        If True, the cuts use the average of all the flats of the sequence
        instead of the first one. Default is False.
    image_binning : int, optional
        Binning factor of the raw image shown next to the cuts. Default is 16.
    n_contrast, contrast_window : int, optional
        Number and width in pixels of the contrast windows along each cut.
        Default is 5 and 51.
    saturation : float, optional
        Raw count level (ADU) at and above which a pixel counts as saturated.
        Default is 65000.
    target_precision : float, optional
        Poisson-limited precision the accumulated flats should reach in every
        quadrant at the median illumination. Default is 0.002 (0.2%).
    max_level_deviation : float, optional
        Largest deviation of a flat's level from the sequence median before it
        is flagged. Default is 0.02 (2%).
    max_saturated : float, optional
        Largest fraction of saturated pixels in a flat quadrant before it is
        flagged. Default is 1e-4.
    max_dark_signal : float, optional
        Largest signal in a dark (e-/pixel above the overscan) before it is
        flagged. Default is 20.
    skip_done : bool, optional
        If True, reuse the measurements already in the table. Default is True.
    dry_run : bool, optional
        If True, log the sequences, the raw files found and the output paths
        without measuring anything. Default is False.

    Returns
    -------
    dict
        - ``"summary"`` : pandas.DataFrame, one row per camera with its status,
          completeness, signal, precision, stability and issues. None in a dry run.
        - ``"frames"`` : pandas.DataFrame, one row per measured frame.
        - ``"descriptions"`` : dict, sequence description per camera.
        - ``"report_path"`` : str, path of the dashboard.
        - ``"table_path"`` : str, path of the measurements table.
    """
    epochs = epochs if epochs is not None else pf.load_pixflat_epochs(verbose=False)
    if mjd_epoch not in epochs:
        raise ValueError(f"epoch {mjd_epoch} not found in the epochs file, available: {sorted(epochs)}")
    epoch = epochs[mjd_epoch]
    cameras = [camera for camera in CAMERAS if camera in cameras]
    output_dir = output_dir or _qa_dir(mjd_epoch, f"{mjd_epoch}_raw")
    report_path = os.path.join(output_dir, f"pixflat-raw-qa_{mjd_epoch}.html")
    table_path = os.path.join(output_dir, f"pixflat-raw-qa_{mjd_epoch}.csv")

    descriptions, problems = {}, {}
    for camera in cameras:
        try:
            descriptions[camera] = describe_sequence(epoch, camera)
        except KeyError:
            problems[camera] = "no sequence in the epochs file"
        except (TypeError, ValueError) as error:
            problems[camera] = f"invalid sequence: {error}"
        if camera in problems:
            log.warning(f"{camera = }: {problems[camera]}")

    to_measure = [
        exposure.path for description in descriptions.values()
        for exposure in description["exposures"].itertuples() if exposure.role != "rejected" and pd.notna(exposure.path)
    ]

    if dry_run:
        log.info(f"dry run of '{qa_raw_pixelflats.__name__}' for epoch {mjd_epoch}, sources {epoch.get('sources')}")
        for camera, description in descriptions.items():
            exposures = description["exposures"]
            counts = ", ".join(f"{role} {int(group.path.notna().sum())}/{len(group)}" for role, group in exposures.groupby("role"))
            log.info(f"  {camera}: kind {description['kind']} ({description['kind_text']}), {description['ngroups']} groups, files found {counts}")
        for camera, problem in problems.items():
            log.info(f"  {camera}: {problem}")
        log.info(f"  {len(to_measure)} raw frames to measure")
        log.info(f"  dashboard: {report_path}")
        log.info(f"  table: {table_path}")
        return {"summary": None, "frames": None, "descriptions": descriptions, "report_path": report_path, "table_path": table_path}

    # measure the raw frames, reusing the existing table
    measured = []
    if skip_done and os.path.isfile(table_path):
        previous = pd.read_csv(table_path)
        previous = previous[previous.path.isin(to_measure)]
        measured.append(previous)
        to_measure = [raw_path for raw_path in to_measure if raw_path not in set(previous.path)]
        log.info(f"reusing {len(previous)} measured frames from {table_path}")
    rows, failed = [], []
    with Pool(nprocs) if nprocs > 1 else _SerialPool() as pool:
        tasks = [(raw_path, saturation) for raw_path in to_measure]
        for raw_path, row, error in tqdm(pool.imap_unordered(_measure_task, tasks), total=len(tasks),
                                         desc="raw pixel flats", unit="frame", ascii=True):
            if row is None:
                log.error(f"failed to measure {os.path.basename(raw_path)}: {error}")
                failed.append({"path": raw_path, "error": error})
            else:
                rows.append(row)
    measured.append(pd.DataFrame(rows))
    measurements = pd.concat([table for table in measured if len(table)], ignore_index=True) if any(len(t) for t in measured) else pd.DataFrame(columns=["path"])
    os.makedirs(output_dir, exist_ok=True)
    measurements.to_csv(table_path, index=False)

    # merge the measurements into each camera's exposures
    frames = []
    for camera, description in descriptions.items():
        exposures = description["exposures"]
        exposures = exposures[exposures.path.notna() & (exposures.role != "rejected")].drop(columns=[])
        merged = exposures.merge(measurements.drop(columns=["expnum"], errors="ignore"), on="path", how="left")
        merged["camera"] = camera
        merged["error"] = ""
        for fail in failed:
            merged.loc[merged.path == fail["path"], "error"] = fail["error"]
        frames.append(merged)
    frames = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["camera", "expnum", "role", "path", "error"])
    # frames that couldn't be read keep NaN measurements
    for column in ["imagetyp", "lamps", "exptime", "gain_source"] + [
            _quad_columns(name)[q] for name in ("gain", "rdnoise", "signal", "signal_p05", "signal_p95", "saturated") for q in range(NQUADS)]:
        if column not in frames:
            frames[column] = np.nan
    frames["signal"] = frames[_quad_columns("signal")].mean(axis=1)

    params = dict(target_precision=target_precision, max_level_deviation=max_level_deviation,
                  max_saturated=max_saturated, max_dark_signal=max_dark_signal)
    stats = {camera: _raw_camera_stats(description, frames[frames.camera == camera], **params)
             for camera, description in descriptions.items()}

    # level of each flat relative to its sequence median
    frames["deviation"] = np.nan
    for camera in descriptions:
        selected = (frames.camera == camera) & (frames.role == "flat") & (frames.error == "") & frames.signal.notna()
        if selected.any():
            frames.loc[selected, "deviation"] = frames.loc[selected, "signal"] / frames.loc[selected, "signal"].median() - 1

    # illumination cuts and contrast
    figures, contrasts = {}, {}
    for camera in descriptions:
        flat_paths = frames[(frames.camera == camera) & (frames.role == "flat") & (frames.error == "")].sort_values("expnum").path.tolist()
        if not flat_paths:
            continue
        try:
            fig, contrast = pf.display_raw_cuts(
                flat_paths if combined else flat_paths[:1], rows=cut_rows, columns=cut_columns, cut_width=cut_width,
                normalize=True, split_amps=False, image_binning=image_binning, combined_flat=combined,
                n_contrast=n_contrast, contrast_window=contrast_window,
            )
        except Exception as error:
            log.error(f"illumination cuts failed for {camera = }: {type(error).__name__}: {error}")
            stats[camera]["issues"].append(f"illumination cuts failed: {type(error).__name__}: {error}")
            continue
        stats[camera]["cut_frame"] = fig.layout.title.text
        stats[camera]["edge_contrast"] = pf._edge_contrast(contrast)
        contrasts[camera] = contrast
        fig = restyle(fig, height=860)
        fig.update_layout(margin=dict(l=72, r=16, t=80, b=90))
        figures[f"cut|{camera}"] = fig

    summary = _raw_summary(cameras, descriptions, stats, problems)
    log.info(f"raw pixel-flat QA summary, epoch {mjd_epoch}:\n{summary.drop(columns=['issues']).to_string()}")

    _write_raw_dashboard(report_path, mjd_epoch, epoch, cameras, descriptions, problems, stats, frames, contrasts,
                         figures, summary, table_path, params, saturation, combined)
    log.info(f"wrote raw pixel-flat QA dashboard to {report_path}")
    return {"summary": summary, "frames": frames, "descriptions": descriptions, "report_path": report_path, "table_path": table_path}


def _raw_summary(cameras, descriptions, stats, problems):
    """One row per camera with the main numbers of the raw QA."""
    rows = []
    for camera in cameras:
        if camera in problems:
            rows.append({"camera": camera, "status": "bad", "issues": problems[camera]})
            continue
        stat, description = stats[camera], descriptions[camera]
        rows.append({
            "camera": camera, "status": stat["status"], "kind": description["kind"], "groups": description["ngroups"],
            "flats": f"{stat['nflat_found']}/{stat['nflat_expected']}", "darks": f"{stat['ndark_found']}/{stat['ndark_expected']}",
            "biases": f"{stat['nbias_found']}/{stat['nbias_expected']}",
            "signal_per_flat": stat.get("signal_per_flat", np.nan), "total_min": stat.get("total_min", np.nan),
            "precision": stat.get("precision", np.nan), "precision_dim": stat.get("precision_dim", np.nan),
            "level_scatter": stat.get("level_scatter", np.nan), "saturated_max": stat.get("saturated_max", np.nan),
            "dark_signal": stat.get("dark_signal", np.nan), "edge_contrast": stat.get("edge_contrast", np.nan),
            "issues": "; ".join(stat["issues"]),
        })
    return pd.DataFrame(rows).set_index("camera")


def _write_raw_dashboard(report_path, mjd_epoch, epoch, cameras, descriptions, problems, stats, frames, contrasts,
                         figures, summary, table_path, params, saturation, combined):
    """Build and write the raw pixel-flat dashboard (see :func:`qa_raw_pixelflats`)."""
    target_precision = params["target_precision"]
    max_level_deviation = params["max_level_deviation"]
    measured = frames[frames.error == ""] if len(frames) else frames
    flats = measured[measured.role == "flat"] if len(measured) else measured
    with_flats = [camera for camera in descriptions if "total_q" in stats[camera]]
    outliers = {camera: stats[camera].get("level_outliers", []) for camera in descriptions}

    figures = dict(figures)
    figures["fig-sequence"] = figure_sequence_strip(descriptions, measured, outliers)
    if with_flats:
        figures["fig-photons"] = figure_photons({camera: stats[camera] for camera in with_flats}, target_precision)
        figures["fig-stability"] = figure_stability(flats, max_level_deviation)
    others = measured[measured.role.isin(["dark", "bias"])] if len(measured) else measured
    if len(others):
        figures["fig-darks"] = figure_darks(others)

    # tiles
    nstatus = summary.status.value_counts()
    expected = sum(stats[c]["nflat_expected"] for c in descriptions)
    found = sum(stats[c]["nflat_found"] for c in descriptions)
    ndarks = sum(stats[c]["ndark_found"] for c in descriptions)
    nbiases = sum(stats[c]["nbias_found"] for c in descriptions)
    tiles = [
        ("Cameras", f"{nstatus.get('ok', 0)} / {len(cameras)} ok",
         f"{nstatus.get('warn', 0)} with warnings, {nstatus.get('bad', 0)} failing"),
        ("Flats on disk", f"{found:,} / {expected:,}", f"plus {ndarks:,} darks and {nbiases:,} biases"),
    ]
    if with_flats:
        per_flat = {camera: stats[camera]["signal_per_flat"] for camera in with_flats}
        lo, hi = min(per_flat, key=per_flat.__getitem__), max(per_flat, key=per_flat.__getitem__)
        tiles.append(("Signal per flat", f"{np.median(list(per_flat.values())):,.0f} e-",
                      f"median per pixel; {lo} {per_flat[lo]:,.0f} to {hi} {per_flat[hi]:,.0f}"))
        worst = max(with_flats, key=lambda camera: stats[camera]["precision"])
        tiles.append(("Worst Poisson precision", f"{100 * stats[worst]['precision']:.3f}%",
                      (f"{worst} AMP{stats[worst]['worst_quad']}, {stats[worst]['total_min']:,.0f} e- accumulated; "
                       f"target {100 * target_precision:g}%")))
        dim = max(with_flats, key=lambda camera: stats[camera]["precision_dim"])
        tiles.append(("Precision in dim regions", f"{100 * stats[dim]['precision_dim']:.3f}%", f"{dim}, dimmest 5% of pixels"))
        unstable = max(with_flats, key=lambda camera: stats[camera]["level_scatter"])
        noutliers = sum(len(outliers[camera]) for camera in with_flats)
        tiles.append(("Lamp stability", f"{100 * stats[unstable]['level_scatter']:.2f}%",
                      f"largest flat-to-flat scatter ({unstable}); {noutliers} flats off by > {100 * max_level_deviation:g}%"))
    edge = {camera: stats[camera]["edge_contrast"] for camera in descriptions if np.isfinite(stats[camera].get("edge_contrast", np.nan))}
    if edge:
        low = min(edge, key=edge.__getitem__)
        tiles.append(("Lowest edge contrast", f"{edge[low]:.2f}", f"{low}, CCD edge relative to center"))

    # sequence table
    sequence_rows = []
    for camera in cameras:
        if camera in problems:
            sequence_rows.append([f'<span class="mono">{camera}</span>', badge("bad")] + ["–"] * 8 + [issues_html([problems[camera]])])
            continue
        description, stat = descriptions[camera], stats[camera]
        sequence_rows.append([
            f'<span class="mono">{camera}</span>', badge(stat["status"]),
            f'<span class="mono">{escape(description["kind"])}</span> <span class="range">{escape(description["kind_text"])}</span>',
            f'<span class="mono">{escape(_format_expnums(description["expnums"]))}</span>',
            f'<span class="mono">{escape(_format_expnums(description["rejects"]))}</span>',
            f'{description["ngroups"]}' + (f' <span class="range">+{description["nleftover"]}</span>' if description["nleftover"] else ""),
            f'{stat["nflat_found"]} / {stat["nflat_expected"]}', f'{stat["ndark_found"]} / {stat["ndark_expected"]}',
            f'{stat["nbias_found"]} / {stat["nbias_expected"]}',
            escape(", ".join(f"{t:g}" for t in stat.get("exptimes", [])) or "–") + " · " + escape(", ".join(stat.get("lamps", [])) or "–"),
            issues_html(stat["issues"]),
        ])
    sequence_table = html_table(
        [("Camera", ""), ("Status", "ok: no issues; warn: issues listed; bad: no flats measured or no valid sequence"),
         ("Kind", "exposure pattern repeated along the sequence, or auto for types taken from IMAGETYP"), ("Exposures", "ranges in the epochs file, inclusive"),
         ("Rejects", "excluded in the epochs file"), ("Groups", "complete groups (+ exposures of an incomplete last group); for auto, runs of consecutive flats"),
         ("Flats", "raw files found / expected"), ("Darks", "raw files found / expected"), ("Biases", "raw files found / expected"),
         ("Flats: exptime (s) · lamps", "exposure times and lamps on in the flats"), ("Issues", "")],
        sequence_rows, numeric=[False, False, False, False, False, True, True, True, True, False, False])

    # photons table
    photon_rows = []
    for camera in with_flats:
        stat = stats[camera]
        for q in range(NQUADS):
            photon_rows.append([
                f'<span class="mono">{camera}</span>', f"AMP{q + 1}", _num(stat["gain_q"][q], ".3f"), str(stat["nflat_measured"]),
                _num(stat["signal_per_flat_q"][q], ",.0f"), _num(stat["signal_dim_per_flat_q"][q], ",.0f"),
                _num(stat["total_q"][q], ",.0f"), _num(stat["total_dim_q"][q], ",.0f"),
                _num(100 * stat["precision_q"][q], ".3f"), _num(100 * stat["precision_dim_q"][q], ".3f"), _num(stat["rdnoise_q"][q], ".2f"),
            ])
    photon_table = html_table(
        [("Camera", ""), ("Amp", ""), ("Gain (e-/ADU)", "median of the corrected GAIN header values"), ("Flats", "measured"),
         ("Per flat (e-)", "median over the flats of the signal per pixel"), ("Per flat, dim (e-)", "5th percentile of the pixels"),
         ("Accumulated (e-)", "summed over the flats"), ("Accumulated, dim (e-)", "5th percentile, summed over the flats"),
         ("Precision (%)", "Poisson-limited, median illumination"), ("Precision, dim (%)", "Poisson-limited, dimmest 5% of pixels"),
         ("Read noise (e-)", "from the overscan")],
        photon_rows, numeric=[False, False] + [True] * 9)

    # stability and frames tables
    stability_rows = []
    for camera in with_flats:
        stat = stats[camera]
        stability_rows.append([f'<span class="mono">{camera}</span>', str(stat["nflat_measured"]), _num(100 * stat["level_scatter"], ".2f"),
                               _num(100 * stat["level_max_deviation"], ".2f"), _num(100 * stat["saturated_max"], ".4f"),
                               _num(stat["dark_signal"], ".1f"), _num(stat["bias_signal"], ".1f"),
                               f'<span class="mono">{escape(_format_expnums(pf._compress_expnums(stat["level_outliers"])))}</span>'])
    stability_table = html_table(
        [("Camera", ""), ("Flats", ""), ("Scatter (%)", "robust flat-to-flat scatter of the level"),
         ("Max deviation (%)", "largest deviation from the median level"), ("Max saturated (%)", "largest fraction of a quadrant"),
         ("Dark (e-)", "median signal of the darks"), ("Bias (e-)", "median signal of the biases"), ("Flats off level", "")],
        stability_rows, numeric=[False, True, True, True, True, True, True, False])

    frame_rows = [[f'<span class="mono">{r.camera}</span>', str(r.expnum), escape(r.role), escape(str(r.imagetyp)), _num(r.exptime, "g"),
                   escape(str(r.lamps)), _num(r.signal, ",.1f"), _num(100 * r.deviation, "+.2f"),
                   _num(100 * max(getattr(r, f"saturated_{q}") for q in range(1, NQUADS + 1)), ".4f"),
                   f'<span class="mono">{escape(os.path.basename(r.path))}</span>']
                  for r in measured.sort_values(["camera", "expnum"]).itertuples()] if len(measured) else []
    frame_table = html_table(
        [("Camera", ""), ("Exposure", ""), ("Role", "in the sequence"), ("IMAGETYP", "header"), ("Exptime (s)", ""), ("Lamps", "on"),
         ("Signal (e-)", "median per pixel, mean of the quadrants"), ("Level (%)", "flats: from the sequence median"),
         ("Saturated (%)", "largest fraction of a quadrant"), ("File", "")],
        frame_rows, numeric=[False, True, False, False, True, False, True, True, True, False])

    contrast_rows = []
    for camera, contrast in contrasts.items():
        edge_rows = contrast[contrast.window == contrast.window.max()]
        contrast_rows.append([f'<span class="mono">{camera}</span>', escape(stats[camera].get("cut_frame", "")),
                              _num(edge_rows[edge_rows.axis == "x"].contrast.min(), ".3f"),
                              _num(edge_rows[edge_rows.axis == "y"].contrast.min(), ".3f"), _num(stats[camera]["edge_contrast"], ".3f")])
    contrast_table = html_table(
        [("Camera", ""), ("Frame", ""), ("Edge, cuts along X", "lowest outermost-window contrast"),
         ("Edge, cuts along Y", "lowest outermost-window contrast"), ("Edge, lowest", "")],
        contrast_rows, numeric=[False, False, True, True, True])

    # page
    sources = ", ".join(str(mjd) for mjd in np.atleast_1d(epoch.get("sources", [])))
    meta = [("Epoch", mjd_epoch), ("Source MJDs", sources), ("Trigger", epoch.get("trigger") or "–"),
            ("DRP version", drpver), ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
            ("Target precision", f"{100 * target_precision:g}%"), ("Saturation", f"{saturation:g} ADU"),
            ("Measurements", os.path.basename(table_path))]
    comment = epoch.get("comment")
    intro = (
        "<p>Pixel flats are built by combining a sequence of exposures of a smooth, featureless illumination, interleaved "
        "with darks or biases. This dashboard checks the raw sequence of one epoch as defined in the pixel-flat epochs "
        "file: whether every exposure is there and of the expected type, how many photons the flats collect, and whether "
        "the illumination is stable along the sequence. The equations are listed under "
        '<a href="#definitions">How the quantities are computed</a>.</p>'
        + (f"<p><strong>Epoch comment:</strong> {escape(str(comment))}</p>" if comment else "")
    )

    cuts_chart = ""
    if contrasts:
        cuts_chart = picker("fig-cuts", "cut", [("Camera", [(camera, camera) for camera in contrasts])], 860)
    sections = [
        f"""<section>
    <h2>Sequences</h2>
    <p>Each camera's sequence as defined in the epochs file. The <em>kind</em> gives the exposure pattern that repeats
    along the sequence, e.g. <span class="mono">2f1d</span> is two flats followed by one dark. Rejected exposures are
    excluded before assigning the pattern; exposures left over at the end form an incomplete group. Sequences with
    kind <span class="mono">auto</span> don't follow a pattern: each exposure's type comes from its IMAGETYP, as in the
    reduction, and each flat is detrended with the first bias taken after it, or the last one before it.</p>
    {sequence_table}
  </section>""",
        f"""<section>
    <h2>Exposures along the sequence</h2>
    <p>Every exposure of each camera's sequence at its exposure number, by type. Missing raw files and flats whose level
    is off the sequence median are circled.</p>
    <div class="chart" data-fig="fig-sequence" role="img" aria-label="Exposures of each sequence by type"></div>
    <details><summary>Table view: all measured frames</summary>{frame_table}</details>
  </section>""",
    ]
    if with_flats:
        sections += [
            f"""<section>
    <h2>Photons reached</h2>
    <p>Signal accumulated over all the flats of each camera, per quadrant, in electrons per pixel using the header gains with the pipeline's gain corrections.
    Filled markers are the median illumination, open markers the dimmest 5% of pixels. The dashed line is the signal
    needed for a {100 * target_precision:g}% Poisson-limited pixel flat; the precision also includes the read noise of
    every flat.</p>
    <div class="chart" data-fig="fig-photons" role="img" aria-label="Accumulated signal per camera and quadrant"></div>
    <details><summary>Table view: signal and precision per quadrant</summary>{photon_table}</details>
  </section>""",
            f"""<section>
    <h2>Lamp stability</h2>
    <p>Level of each flat relative to the median of its sequence. A drifting lamp shows up as a trend; a flat with the
    shutter or lamp misbehaving as a single outlier. The shaded band is &plusmn;{100 * max_level_deviation:g}%.</p>
    <div class="chart" data-fig="fig-stability" role="img" aria-label="Flat level relative to the sequence median"></div>
    <details><summary>Table view: stability, saturation and dark levels</summary>{stability_table}</details>
  </section>""",
        ]
    if "fig-darks" in figures:
        sections.append("""<section>
    <h2>Darks and biases</h2>
    <p>Signal of the darks and biases above the overscan level. It should be flat and close to zero; a step or a high
    level points to a light leak, a lamp left on, or a change in the bias structure.</p>
    <div class="chart" data-fig="fig-darks" role="img" aria-label="Signal of the darks and biases"></div>
  </section>""")
    if cuts_chart:
        sections.append(f"""<section>
    <h2>Illumination pattern</h2>
    <p>Cuts along rows and columns of the {'average of the flats' if combined else 'first flat'} of each camera,
    normalized by the median of the image, next to the raw image with the cuts drawn on it. The numbers are the
    contrast: the level of each window relative to the window nearest the CCD center. Low contrast at the edges means
    few photons there and a noisier pixel flat.</p>
    {cuts_chart}
    <details><summary>Table view: edge contrast per camera</summary>{contrast_table}</details>
  </section>""")
    sections.append(RAW_DEFINITIONS)

    write_dashboard(report_path, f"Raw pixel flats, epoch {mjd_epoch}", f"LVM DRP · pixel flats · raw sequences · {drpver}",
                     intro, meta, tiles, "\n\n  ".join(sections), figures, "lvmdrp.qa.pixelflats.qa_raw_pixelflats")


RAW_DEFINITIONS = r"""<section id="definitions">
    <h2>How the quantities are computed</h2>
    <p>Quadrants (amplifiers) are indexed by \(q\), flats by \(k\). \(g_q\) is the header gain divided by the pipeline's gain correction, \(T_q\) the
    science region (TRIMSEC) and \(O_q\) the overscan region (BIASSEC). MAD is the median absolute deviation.</p>
    <div class="defs">
      <div class="def">
        <h3>Signal</h3>
        <p>Median counts of the science region above the median of the overscan, in electrons per pixel. The dim-region
        signal uses the 5th percentile \(\mathrm{P5}\) instead of the median.</p>
        <div class="eq">\[ \begin{gathered} S_{q,k} = g_q\left[\mathrm{med}_{T_q}(c_k) - \mathrm{med}_{O_q}(c_k)\right] \\ S^{\mathrm{dim}}_{q,k} = g_q\left[\mathrm{P5}_{T_q}(c_k) - \mathrm{med}_{O_q}(c_k)\right] \end{gathered} \]</div>
      </div>
      <div class="def">
        <h3>Read noise</h3>
        <p>Robust scatter of the overscan of each flat, in electrons; the median over the flats is used.</p>
        <div class="eq">\[ \sigma_q = 1.4826\, g_q\, \mathrm{MAD}_{O_q}(c_k) \]</div>
      </div>
      <div class="def">
        <h3>Accumulated signal and precision</h3>
        <p>Signal summed over the \(n\) flats, and the relative noise of a pixel of the combined flat from photon and
        read noise alone. The reported precision of a camera is that of its worst quadrant.</p>
        <div class="eq">\[ N_q = \sum_k S_{q,k}, \qquad \epsilon_q = \frac{\sqrt{N_q + n\,\sigma_q^2}}{N_q} \]</div>
      </div>
      <div class="def">
        <h3>Level stability</h3>
        <p>Deviation of the mean signal of each flat over the quadrants from the median of the sequence, and its robust
        scatter.</p>
        <div class="eq">\[ d_k = \frac{\bar S_k}{\mathrm{med}_k\left(\bar S_k\right)} - 1, \qquad s = 1.4826\,\mathrm{MAD}_k\left(d_k\right) \]</div>
      </div>
      <div class="def">
        <h3>Saturated fraction</h3>
        <p>Fraction of the science pixels of a quadrant at or above the saturation level \(c_{\mathrm{sat}}\), in raw
        counts.</p>
        <div class="eq">\[ f_{q,k} = \frac{1}{|T_q|}\sum_{p \in T_q} \mathbf{1}\left[c_{k,p} \ge c_{\mathrm{sat}}\right] \]</div>
      </div>
      <div class="def">
        <h3>Contrast</h3>
        <p>Along each cut, windows \(j = 0 \ldots J\) run from the end nearest the CCD center (\(j = 0\)) to the edge.
        The edge contrast is the lowest \(C_J\) over the cuts and quadrants.</p>
        <div class="eq">\[ C_j = \frac{\mathrm{med}\left(L_j\right)}{\mathrm{med}\left(L_0\right)} \]</div>
      </div>
    </div>
  </section>"""


# ---------------------------------------------------------------------------
# produced pixel flats
# ---------------------------------------------------------------------------

def _qa_comparison(camera, record, products, mjd_tar, mjd_ref):
    """Run the 1:1 master pixel-flat comparison part of the QA for one camera.

    Parameters
    ----------
    camera : str
        Camera identifier, e.g. ``"b1"``.
    record : dict
        Summary record of the camera, updated with the percentage of pixels of
        the target master pixel flat within 1% of the reference, and the median
        and robust scatter of their ratio.
    products : dict
        Master pixel-flat paths indexed as ``[camera][mjd_epoch]``.
    mjd_tar, mjd_ref : int
        MJDs of the target and reference pixel-flat epochs.

    Returns
    -------
    dict
        Empty, the figure is made for all cameras at once.
    """
    pflat_ref = image_tasks.loadImage(products[camera][mjd_ref])
    pflat_tar = image_tasks.loadImage(products[camera][mjd_tar])
    record["pct_within_1pct"] = pf._pct_within(pflat_ref._data.ravel(), pflat_tar._data.ravel())
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (pflat_tar._data / pflat_ref._data).ravel()
    ratio = ratio[np.isfinite(ratio)]
    median = np.median(ratio)
    record["ratio_median"] = float(median)
    record["ratio_scatter"] = float(1.4826 * np.median(np.abs(ratio - median)))
    return {}


def _count_artifacts(pflat, bins):
    _, _, _, labels_bins = pf._detect_artifacts(pflat, bins=bins, threshold=0.95)
    return [int(n) for _, _, n in labels_bins]


def _qa_artifacts(camera, record, products, mjd_tar, mjd_ref, bins, bbox_size, max_regions):
    """Run the master pixel-flat artifacts part of the QA for one camera.

    Parameters
    ----------
    camera : str
        Camera identifier, e.g. ``"b1"``.
    record : dict
        Summary record of the camera, updated with the number of artifacts
        detected in the target and reference master pixel flats, in total and
        per size bin.
    products : dict
        Master pixel-flat paths indexed as ``[camera][mjd_epoch]``.
    mjd_tar, mjd_ref : int
        MJDs of the target and reference pixel-flat epochs.
    bins, bbox_size, max_regions
        Passed to :func:`~lvmdrp.functions.pixelflats.display_artifacts_comparison`.

    Returns
    -------
    dict
        Figures as returned by :func:`~lvmdrp.functions.pixelflats.display_artifacts_comparison`.
    """
    counts_tar = _count_artifacts(image_tasks.loadImage(products[camera][mjd_tar]), bins)
    counts_ref = _count_artifacts(image_tasks.loadImage(products[camera][mjd_ref]), bins)
    record.update({"n_artifacts_tar": sum(counts_tar), "n_artifacts_ref": sum(counts_ref),
                   "artifacts_tar_bins": counts_tar, "artifacts_ref_bins": counts_ref})
    return pf.display_artifacts_comparison(
        mjd_tar, mjd_ref, drpver=drpver, camera=camera, bins=bins, bbox_size=bbox_size, max_regions=max_regions,
    )


def _qa_flatfielded(camera, record, frame, test_flat_paths, mjd_tar, mjd_ref, bins, bbox_size, max_regions, skip_done):
    """Run the flat-fielded test exposure part of the QA for one camera.

    Parameters
    ----------
    camera : str
        Camera identifier, e.g. ``"b1"``.
    record : dict
        Summary record of the camera, updated with the test exposure used.
    frame : pandas.Series
        Test frame metadata (see :func:`~lvmdrp.functions.pixelflats.test_pixflats`).
    test_flat_paths : dict
        Detrended test-frame paths indexed as ``[mjd_epoch][camera]``.
    mjd_tar, mjd_ref : int
        MJDs of the target and reference pixel-flat epochs.
    bins, bbox_size, max_regions
        Passed to :func:`~lvmdrp.functions.pixelflats.display_flatfielded_comparison`.
    skip_done : bool
        Passed to :func:`~lvmdrp.functions.pixelflats.test_pixflats`.

    Returns
    -------
    dict
        Figures as returned by :func:`~lvmdrp.functions.pixelflats.display_flatfielded_comparison`.
    """
    for mjd_epoch in (mjd_tar, mjd_ref):
        pf.test_pixflats(mjd_epoch, frame, skip_done=skip_done)
    figures = pf.display_flatfielded_comparison(
        mjd_tar, mjd_ref, drpver=drpver, camera=camera, test_flat_paths=test_flat_paths,
        bins=bins, bbox_size=bbox_size, max_regions=max_regions,
    )
    record["test_frame"] = f"{camera}-{frame.expnum}"
    return figures


def figure_consistency(summary):
    """Percentage of consistent pixels and ratio scatter per camera.

    Parameters
    ----------
    summary : pandas.DataFrame
        Products QA summary, indexed by camera, with ``pct_within_1pct`` and
        ``ratio_scatter`` columns.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    selected = summary[summary.pct_within_1pct.notna()]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=["Pixels within 1% of the reference", "Scatter of the target / reference ratio"])
    colors = [_camera_color(camera) for camera in selected.index]
    fig.add_trace(go.Bar(x=selected.index, y=selected.pct_within_1pct, marker_color=colors, showlegend=False,
                         hovertemplate="<b>%{x}</b><br>%{y:.3f}% of pixels within 1%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=selected.index, y=100 * selected.ratio_scatter, marker_color=colors, showlegend=False,
                         hovertemplate="<b>%{x}</b><br>robust scatter %{y:.3f}%<extra></extra>"), row=1, col=2)
    low = float(np.nanmin(selected.pct_within_1pct)) if len(selected) else 99.0
    fig.update_layout(base_layout(340, bargap=0.35))
    style_axes(fig)
    fig.update_yaxes(title_text="Pixels (%)", range=[min(low - 0.5, 99.0), 100], row=1, col=1)
    fig.update_yaxes(title_text="Scatter (%)", rangemode="tozero", row=1, col=2)
    fig.update_xaxes(showgrid=False, tickfont=dict(family=MONO_FONT, color=THEME["ink2"]))
    fig.update_annotations(font=dict(color=THEME["ink2"], size=13))
    return fig


def figure_artifact_counts(summary, bins):
    """Number of artifacts in the target and reference master pixel flats.

    Parameters
    ----------
    summary : pandas.DataFrame
        Products QA summary, indexed by camera, with the artifact counts.
    bins : iterable[tuple[int, int]]
        Region-size bins of the counts.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    selected = summary[summary.n_artifacts_tar.notna()]
    fig = go.Figure()
    for column, label, color in (("artifacts_tar_bins", "Target", SERIES[0]), ("artifacts_ref_bins", "Reference", THEME["muted"])):
        breakdown = ["<br>".join(f"{lo}–{hi} px: {n}" for (lo, hi), n in zip(bins, counts)) for counts in selected[column]]
        fig.add_trace(go.Bar(x=selected.index, y=[sum(counts) for counts in selected[column]], name=label, marker_color=color,
                             customdata=breakdown, hovertemplate=f"<b>%{{x}}</b> {label.lower()}<br>%{{y}} artifacts<br>%{{customdata}}<extra></extra>"))
    fig.update_layout(base_layout(320, barmode="group", bargap=0.3, yaxis_title="Artifacts (regions ≤ 0.95)"))
    style_axes(fig)
    fig.update_xaxes(showgrid=False, tickfont=dict(family=MONO_FONT, color=THEME["ink2"]))
    return fig


def qa_pixelflats(mjd_tar, mjd_ref, cameras=CAMERAS, parts=QA_PARTS,
                  test_frames=None, epochs=None, output_dir=None,
                  bins=pf.ARTIFACT_BINS, bbox_size=30, max_regions=10, min_consistency=99.0,
                  skip_done=True, dry_run=False):
    """Run the QA of the master pixel flats of a target epoch against a reference epoch.

    The target epoch's sequences are always validated
    (:func:`~lvmdrp.functions.pixelflats.validate_sequence_kind`). Then, for
    each camera with the master pixel flats of both epochs, the selected
    ``parts`` run in this order:

    - ``"comparison"``: compare the master pixel flats of both epochs 1:1
      (:func:`~lvmdrp.functions.pixelflats.display_pixflats_comparison`), and
      record the percentage of pixels consistent within 1% and the scatter of
      their ratio.
    - ``"artifacts"``: count the artifacts of both master pixel flats and
      compare both epochs around those of the target
      (:func:`~lvmdrp.functions.pixelflats.display_artifacts_comparison`).
    - ``"flatfielded"``: reduce the ``test_frames`` with both epochs' pixel
      flats (:func:`~lvmdrp.functions.pixelflats.test_pixflats`) and compare the
      flat-fielded results around the artifacts
      (:func:`~lvmdrp.functions.pixelflats.display_flatfielded_comparison`).

    Everything is written to a single dashboard. A failure in one part is
    logged and recorded in the summary, and doesn't stop the remaining parts
    or cameras. The raw sequences are checked by :func:`qa_raw_pixelflats`.

    Parameters
    ----------
    mjd_tar : int
        MJD of the target pixel-flat epoch, the one being checked.
    mjd_ref : int
        MJD of the reference pixel-flat epoch, the one to compare against.
    cameras : iterable[str], optional
        Cameras to check. Default is all of :data:`CAMERAS`.
    parts : iterable[str], optional
        Parts of the QA to run, any of :data:`QA_PARTS`. Default is all of them.
    test_frames : pandas.DataFrame, optional
        Frames used in the ``"flatfielded"`` part, with at least the columns
        ``mjd``, ``camera``, ``expnum``, and ``imagetyp``. The first row of
        each camera is used, and cameras without a row are skipped. If omitted,
        the ``"flatfielded"`` part is skipped.
    epochs : dict, optional
        Epoch mapping as returned by :func:`~lvmdrp.functions.pixelflats.load_pixflat_epochs`.
        If omitted, it is loaded from the default epochs file.
    output_dir : str, optional
        Directory of the dashboard. Default is ``pixflat_qa/{mjd_tar}_vs_{mjd_ref}``
        in the target epoch's ancillary directory.
    bins : iterable[tuple[int, int]], optional
        Region-size bins used to group artifacts. Default is
        :data:`~lvmdrp.functions.pixelflats.ARTIFACT_BINS`.
    bbox_size : int, optional
        Width of each artifact cutout in pixels. Default is 30.
    max_regions : int, optional
        Maximum number of artifact regions shown per size bin. Default is 10.
    min_consistency : float, optional
        Lowest percentage of pixels within 1% of the reference before a camera
        is flagged. Default is 99.
    skip_done : bool, optional
        If True, reuse existing test-frame reductions. Figures and the
        dashboard are always regenerated. Default is True.
    dry_run : bool, optional
        If True, log the cameras, test-frame paths, and dashboard path without
        running any check or writing files. Default is False.

    Returns
    -------
    dict
        - ``"summary"`` : pandas.DataFrame, one row per camera, with the sequence
          validation result, its status, whether both master pixel flats
          exist, any error, and the columns of the parts that ran. None in a
          dry run.
        - ``"figures"`` : dict, the Plotly figures of the dashboard. Empty in a
          dry run.
        - ``"test_flat_paths"`` : dict, detrended test-frame paths indexed as
          ``[mjd_epoch][camera]``.
        - ``"report_path"`` : str, path to the dashboard.
    """
    unknown = set(parts).difference(QA_PARTS)
    if unknown:
        raise ValueError(f"unknown QA parts {sorted(unknown)}, expected any of {QA_PARTS}")
    parts = [part for part in QA_PARTS if part in parts]

    cameras = [camera for camera in CAMERAS if camera in cameras]
    output_dir = output_dir or _qa_dir(mjd_tar, f"{mjd_tar}_vs_{mjd_ref}")
    report_path = os.path.join(output_dir, f"pixflat-qa_{mjd_tar}_vs_{mjd_ref}.html")

    products = {camera: {mjd: pf._master_pixflat_path(mjd, camera) for mjd in (mjd_tar, mjd_ref)} for camera in cameras}
    available = [camera for camera in cameras if all(os.path.isfile(product) for product in products[camera].values())]
    for camera in cameras:
        if camera not in available:
            log.warning(f"master pixel flat missing for {camera = } in {mjd_tar = } or {mjd_ref = }, skipping it")

    frames = {}
    if "flatfielded" not in parts:
        pass
    elif test_frames is None:
        log.info("no test frames given, skipping the flat-fielded comparison")
    else:
        for camera in available:
            selected = test_frames.query("camera == @camera")
            if selected.empty:
                log.warning(f"no test frame given for {camera = }, skipping its flat-fielded comparison")
                continue
            frames[camera] = selected.iloc[0]

    test_flat_paths = {mjd_tar: {}, mjd_ref: {}}
    for camera, frame in frames.items():
        for mjd_epoch in (mjd_tar, mjd_ref):
            test_flat_paths[mjd_epoch][camera] = pf._test_pixflat_paths(mjd_epoch, frame)[-1]

    if dry_run:
        log.info(f"dry run of '{qa_pixelflats.__name__}' for {mjd_tar = } against {mjd_ref = }")
        log.info(f"  parts: {parts}")
        log.info(f"  cameras with both master pixel flats: {available}")
        for mjd_epoch, epoch_paths in test_flat_paths.items():
            for camera, dframe_path in epoch_paths.items():
                status = "exists" if os.path.isfile(dframe_path) else "would be created"
                log.info(f"  test frame, {mjd_epoch = }, {camera = }: {dframe_path} [{status}]")
        log.info(f"  dashboard: {report_path}")
        return {"summary": None, "figures": {}, "test_flat_paths": test_flat_paths, "report_path": report_path}

    epochs = epochs if epochs is not None else pf.load_pixflat_epochs(verbose=False)
    validation = {}
    for camera in cameras:
        try:
            invalid = pf.validate_sequence_kind(epochs, mjd_epoch=mjd_tar, camera=camera)
        except KeyError:
            validation[camera] = "no sequence"
        except Exception as error:
            validation[camera] = f"invalid sequence: {error}"
        else:
            mismatches = {_type: len(frame) for _type, frame in invalid.items() if len(frame)}
            validation[camera] = "ok" if not mismatches else "imagetyp mismatch: " + ", ".join(f"{n} {_type}" for _type, n in mismatches.items())

    runners = {
        "comparison": lambda camera, record: _qa_comparison(camera, record, products, mjd_tar, mjd_ref),
        "artifacts": lambda camera, record: _qa_artifacts(
            camera, record, products, mjd_tar, mjd_ref, bins=bins, bbox_size=bbox_size, max_regions=max_regions,
        ),
        "flatfielded": lambda camera, record: _qa_flatfielded(
            camera, record, frames[camera], test_flat_paths, mjd_tar, mjd_ref,
            bins=bins, bbox_size=bbox_size, max_regions=max_regions, skip_done=skip_done,
        ),
    }

    camera_figures = {}
    compared = []
    records = []
    for camera in cameras:
        record = {"camera": camera, "sequence": validation[camera], "products": camera in available,
                  "pct_within_1pct": np.nan, "ratio_median": np.nan, "ratio_scatter": np.nan,
                  "n_artifacts_tar": np.nan, "n_artifacts_ref": np.nan, "artifacts_tar_bins": None,
                  "artifacts_ref_bins": None, "test_frame": None, "error": ""}
        records.append(record)
        camera_figures[camera] = {}
        if camera not in available:
            continue
        errors = []
        for part in parts:
            if part == "flatfielded" and camera not in frames:
                continue
            try:
                part_figures = runners[part](camera, record)
            except Exception as error:
                log.error(f"{part} QA failed for {camera = }: {type(error).__name__}: {error}")
                errors.append(f"{part}: {type(error).__name__}: {error}")
                continue
            if part == "comparison":
                compared.append(camera)
            if part_figures:
                camera_figures[camera][part] = part_figures
        record["error"] = "; ".join(errors)

    summary = pd.DataFrame(records).set_index("camera")
    summary["status"] = [
        "bad" if not r.products or r.error else
        ("warn" if (np.isfinite(r.pct_within_1pct) and r.pct_within_1pct < min_consistency) or r.sequence != "ok" else "ok")
        for r in summary.itertuples()
    ]
    log.info(f"pixel-flat QA summary, {mjd_tar = } against {mjd_ref = }:\n"
             f"{summary.drop(columns=['artifacts_tar_bins', 'artifacts_ref_bins']).to_string()}")

    figures = {}
    if compared:
        try:
            figures["fig-master"] = restyle(pf.display_pixflats_comparison(mjd_tar, mjd_ref, drpver=drpver, cameras=compared), height=1000)
        except Exception as error:
            log.error(f"master pixel-flat comparison failed: {type(error).__name__}: {error}")
        figures["fig-consistency"] = figure_consistency(summary)
    if summary.n_artifacts_tar.notna().any():
        figures["fig-artifact-counts"] = figure_artifact_counts(summary, bins)
    for camera, groups in camera_figures.items():
        for part, part_figures in groups.items():
            for name, fig in part_figures.items():
                figures[f"{part}|{camera}|{name}"] = restyle(fig)

    _write_products_dashboard(report_path, mjd_tar, mjd_ref, cameras, parts, summary, figures, camera_figures,
                              bins, min_consistency, epochs)
    log.info(f"wrote pixel-flat QA dashboard to {report_path}")
    return {"summary": summary, "figures": figures, "test_flat_paths": test_flat_paths, "report_path": report_path}


PRODUCT_VIEWS = (("flat_tar", "target"), ("flat_ref", "reference"), ("flat_rat", "ratio reference / target"),
                 ("hist_rat", "ratio histogram"))


def _write_products_dashboard(report_path, mjd_tar, mjd_ref, cameras, parts, summary, figures, camera_figures,
                              bins, min_consistency, epochs):
    """Build and write the pixel-flat products dashboard (see :func:`qa_pixelflats`)."""
    compared = summary[summary.pct_within_1pct.notna()]
    tiles = [
        ("Cameras", f"{(summary.status == 'ok').sum()} / {len(cameras)} ok",
         f"{int(summary.products.sum())} with both master pixel flats"),
    ]
    if len(compared):
        worst = compared.pct_within_1pct.idxmin()
        tiles.append(("Lowest consistency", f"{compared.pct_within_1pct.min():.2f}%", f"{worst}, pixels within 1% of {mjd_ref}"))
        noisy = compared.ratio_scatter.idxmax()
        tiles.append(("Largest ratio scatter", f"{100 * compared.ratio_scatter.max():.3f}%", f"{noisy}, target / reference"))
    counted = summary[summary.n_artifacts_tar.notna()]
    if len(counted):
        tiles.append(("Artifacts", f"{int(counted.n_artifacts_tar.sum()):,}",
                      f"in the target flats; {int(counted.n_artifacts_ref.sum()):,} in the reference"))
    if summary.test_frame.notna().any():
        tiles.append(("Flat-fielded tests", f"{int(summary.test_frame.notna().sum())}", "cameras with a test exposure"))

    rows = []
    for camera in cameras:
        r = summary.loc[camera]
        rows.append([
            f'<span class="mono">{camera}</span>', badge(r.status), escape(str(r.sequence)), "yes" if r.products else "no",
            _num(r.pct_within_1pct, ".3f"), _num(r.ratio_median, ".5f"), _num(100 * r.ratio_scatter, ".3f"),
            _num(r.n_artifacts_tar, ".0f"), _num(r.n_artifacts_ref, ".0f"), escape(str(r.test_frame or "–")),
            issues_html([e for e in str(r.error).split("; ") if e]),
        ])
    summary_table = html_table(
        [("Camera", ""), ("Status", f"bad: missing products or errors; warn: < {min_consistency:g}% consistent or sequence issues"),
         ("Sequence", "validation of the target sequence"), ("Products", "both master pixel flats exist"),
         ("Within 1% (%)", "pixels of the target within 1% of the reference"), ("Ratio median", "target / reference"),
         ("Ratio scatter (%)", "robust scatter of target / reference"), ("Artifacts target", ""), ("Artifacts reference", ""),
         ("Test exposure", ""), ("Errors", "")],
        rows, numeric=[False, False, False, False, True, True, True, True, True, False, False])

    sections = [f"""<section>
    <h2>Cameras</h2>
    <p>Status of each camera. The sequence column is the validation of the target epoch's sequence in the epochs file;
    the raw sequences are checked in detail by the raw pixel-flat dashboard.</p>
    {summary_table}
  </section>"""]
    if "fig-consistency" in figures:
        sections.append(f"""<section>
    <h2>1:1 comparison</h2>
    <p>A new pixel flat should agree with the previous one except where the detector changed. Left: fraction of pixels
    whose target value is within 1% of the reference; right: robust scatter of the target over reference ratio, which
    includes the noise of both flats.</p>
    <div class="chart" data-fig="fig-consistency" role="img" aria-label="Consistency of the target and reference master pixel flats"></div>
    {'<details><summary>Pixel-by-pixel density per camera</summary><div class="chart" data-fig="fig-master" role="img" aria-label="Density of target versus reference pixel values"></div></details>' if "fig-master" in figures else ""}
  </section>""")

    views = [(name, label) for name, label in PRODUCT_VIEWS]
    for part, heading, text in (
        ("artifacts", "Artifacts in the master pixel flats",
         ("Artifacts are connected regions of the target master pixel flat at or below 0.95, such as dust shadows, grouped by "
          "size. The cutouts show the largest ones of each size bin in the target, the reference, and their ratio; "
          "a ratio close to 1 means the artifact was already there.")),
        ("flatfielded", "Artifacts in a flat-fielded test exposure",
         ("The test exposure reduced with each epoch's pixel flat, around the target artifacts. An artifact that the pixel "
          "flat corrects disappears in the flat-fielded frame; the histograms show the ratio of both reductions inside "
          "the artifacts.")),
    ):
        with_part = [camera for camera in cameras if part in camera_figures.get(camera, {})]
        if not with_part:
            continue
        chart = picker(f"fig-{part}", part, [("Camera", [(camera, camera) for camera in with_part]),
                                               ("View", [(name, label) for name, label in views])], 360)
        extra = ('<div class="chart" data-fig="fig-artifact-counts" role="img" aria-label="Number of artifacts per camera"></div>'
                 if part == "artifacts" and "fig-artifact-counts" in figures else "")
        sections.append(f"""<section>
    <h2>{heading}</h2>
    <p>{text}</p>
    {extra}
    {chart}
  </section>""")
    sections.append(PRODUCT_DEFINITIONS.replace("@BINS@", ", ".join(f"{lo}&ndash;{hi}" for lo, hi in bins)))

    meta = [("Target epoch", mjd_tar), ("Reference epoch", mjd_ref), ("Parts", ", ".join(parts)),
            ("Trigger", (epochs.get(mjd_tar) or {}).get("trigger") or "–"), ("DRP version", drpver),
            ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]
    intro = ("<p>The master pixel flats of the target epoch, compared against those of the reference epoch: how well "
             "they agree pixel by pixel, which artifacts they contain, and whether flat-fielding a test exposure with "
             "them removes the artifacts. The equations are listed under "
             '<a href="#definitions">How the quantities are computed</a>.</p>')
    write_dashboard(report_path, f"Pixel flats {mjd_tar} vs {mjd_ref}", f"LVM DRP · pixel flats · products · {drpver}",
                     intro, meta, tiles, "\n\n  ".join(sections), figures, "lvmdrp.qa.pixelflats.qa_pixelflats")


PRODUCT_DEFINITIONS = r"""<section id="definitions">
    <h2>How the quantities are computed</h2>
    <p>\(t_p\) and \(r_p\) are the target and reference master pixel flats at pixel \(p\), over the \(P\) pixels of
    the detector.</p>
    <div class="defs">
      <div class="def">
        <h3>Pixels within 1%</h3>
        <div class="eq">\[ \frac{100}{P}\sum_p \mathbf{1}\left[\,(1 - 0.01)\,r_p \le t_p \le (1 + 0.01)\,r_p\,\right] \]</div>
      </div>
      <div class="def">
        <h3>Ratio median and scatter</h3>
        <div class="eq">\[ \rho_p = \frac{t_p}{r_p}, \qquad \tilde\rho = \mathrm{med}_p\left(\rho_p\right), \qquad s = 1.4826\,\mathrm{MAD}_p\left(\rho_p\right) \]</div>
      </div>
      <div class="def">
        <h3>Artifacts</h3>
        <p>Connected regions of pixels with a pixel-flat value at or below 0.95, grouped by their size in pixels into the
        bins @BINS@.</p>
        <div class="eq">\[ A = \left\{\, p : t_p \le 0.95 \,\right\}, \quad \text{split into connected regions} \]</div>
      </div>
    </div>
  </section>"""


