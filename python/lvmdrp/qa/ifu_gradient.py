# encoding: utf-8
"""Instrument gradient across the science IFU.

The gradient that the pipeline fits across the science IFU (the plane of
:class:`lvmdrp.core.fit_profile.IFUGradient`) mixes two components:

- an instrument illumination gradient I, fixed in some instrument frame;
- a sky gradient S, fixed on the sky: towards the Sun in twilights, towards the
  horizon (airglow) and the Moon at night.

The twilight fiber flats remove the whole plane, so I never reaches the master
flats and stays in the flat-fielded science frames. This module tests whether I
exists and characterizes it before any pipeline change: which frame holds it
fixed, whether it jumps at instrument interventions, and whether it agrees
between twilights, dome flats and science frames.

For one channel, the gradient vector g_e of exposure e (at the IFU edge, in IFU
coordinates, %) is modeled as

    g_e = R(s phi_e) I_k + sum_j a_j(e) u_j(e) + noise

with I_k the instrument gradient of the segment k (between instrument
interventions) the exposure belongs to, R(s phi_e) a rotation by the K-mirror
angle for an instrument gradient fixed before the K-mirror (identity for one
fixed in the IFU), u_j(e) the unit vectors of the sky directions (Sun, horizon,
Moon) from the geometry of the exposure, and a_j(e) their amplitudes: constant,
linear in the depression of the Sun, or free per night or per exposure. The
model is linear and is fitted by least squares; competing models are compared
with the BIC and with the rms of leave-one-night-out predictions.

The inputs are the tables of the trends dashboards
(:func:`lvmdrp.qa.fiberflats.qa_twilight_trends` and
:func:`~lvmdrp.qa.fiberflats.qa_dome_trends`) and, for science frames, the
summary table of the flat field QA (:func:`lvmdrp.qa.fiberflats.run_flatfield_qa_batch`)
with the geometry of each frame (:func:`science_gradients`). The dashboard is
written by :func:`qa_instrument_gradient` (``drp calibrations qa-instrument-gradient``).
"""

import os
from contextlib import nullcontext
from datetime import datetime, timezone
from html import escape
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from plotly.subplots import make_subplots

from lvmdrp import log
from lvmdrp.qa.fiberflats import (
    _night_sjd, _rayleigh_pvalue, _resultant_length, _wrap_angle, frame_gradients, load_calibration_epochs,
    load_kmirror_relation, siderostat_field_angle, twilight_geometry,
)
from lvmdrp.qa.report import SERIES, THEME, base_layout, html_table, sandbox_qa_dir, style_axes, write_dashboard
from lvmdrp.utils.convert import tileid_grp


# frames that can hold the instrument gradient fixed: (label, rotation sign applied to the K-mirror angle)
INSTRUMENT_FRAMES = {
    "none": ("no instrument gradient", None),
    "ifu": ("fixed in the IFU", 0),
    "kmirror": ("fixed before the K-mirror, rotating with +phi", 1),
    "kmirror-": ("fixed before the K-mirror, rotating with -phi", -1),
}

# amplitude of a sky gradient along its direction
SKY_AMPLITUDES = ("constant", "depth", "night", "exposure")

# default sky gradients of each source: {direction column: amplitude}
DEFAULT_SKY = {
    "twilight": {"dir_sun": "depth"},
    "dome": {"dir_horizon": "constant"},
    "science": {"dir_horizon": "exposure"},
}

# triggers of the calibration epochs that may change the instrument
INTERVENTION_TRIGGERS = ("instrument intervention", "warm-up", "warm up")

# Sun altitude (degrees) where the twilight sky amplitude of the 'depth' model is referred to
SUN_ALT_REF = -3.0


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def intervention_breaks(epochs: Dict[int, Dict] = None, triggers: Sequence[str] = INTERVENTION_TRIGGERS) -> List[int]:
    """MJDs of the calibration epochs whose trigger may have changed the instrument

    Parameters
    ----------
    epochs : dict, optional
        Calibration epochs, as from :func:`lvmdrp.qa.fiberflats.load_calibration_epochs`.
        Default is the epochs file in lvmcore.
    triggers : list[str], optional
        Triggers, case insensitive, that start a new segment. Default is
        :data:`INTERVENTION_TRIGGERS`.

    Returns
    -------
    list[int]
        Sorted MJDs.
    """
    epochs = epochs if epochs is not None else load_calibration_epochs()
    triggers = [t.lower() for t in triggers]
    return sorted(int(mjd) for mjd, epoch in epochs.items() if str(epoch.get("trigger", "")).lower() in triggers)


def _kmirror_rotation(data: pd.DataFrame, field_sign: int) -> np.ndarray:
    """Rotation of the image by the K-mirror up to a constant: PA - field_sign * field angle, in degrees"""
    if "field" not in data.columns:
        data["field"] = np.nan
        good = data[["ra", "dec"]].notna().all(axis=1) & data["time"].notna() if {"ra", "dec", "time"} <= set(data.columns) else []
        if np.any(good):
            data.loc[good, "field"] = siderostat_field_angle(data.loc[good, "ra"].to_numpy(float), data.loc[good, "dec"].to_numpy(float),
                                                             Time(list(data.loc[good, "time"]), scale="utc"))
    pa = data["pa"].astype(float) if "pa" in data.columns else np.nan
    return _wrap_angle(pa - field_sign * data["field"].astype(float))


def prepare_gradients(data: Union[str, pd.DataFrame], source: str, kmirror=None, channels: str = "brz") -> pd.DataFrame:
    """Gradients of one source with the columns the fits need

    Parameters
    ----------
    data : str or pandas.DataFrame
        Table of a trends dashboard (twilight or dome) or from
        :func:`science_gradients`, or the path of its CSV file. Needs ``expnum``,
        ``mjd`` (night), ``channel``, ``grad_x`` and ``grad_y`` (fractions), and
        the direction columns of the sky model (e.g. ``dir_sun``).
    source : str
        'twilight', 'dome' or 'science'.
    kmirror : dict or str, optional
        K-mirror relation (see :func:`lvmdrp.qa.fiberflats.load_kmirror_relation`),
        for the sign of the field angle in the K-mirror rotation. Default is the
        sign found at LCO (-1).
    channels : str, optional
        Channels to keep. Default is 'brz'.

    Returns
    -------
    pandas.DataFrame
        The rows with a gradient, plus ``source``, ``gx`` and ``gy`` (%),
        ``depth`` (degrees of the Sun below -3 deg), ``field`` (siderostat
        field angle) and ``phi`` (rotation of the K-mirror up to a constant,
        PA - field_sign x field angle).
    """
    data = pd.read_csv(data) if isinstance(data, (str, os.PathLike)) else data.copy()
    data = data[data["channel"].astype(str).isin(list(channels))]
    data = data[np.isfinite(data["grad_x"].astype(float)) & np.isfinite(data["grad_y"].astype(float))].reset_index(drop=True)
    data["source"] = source
    data["gx"], data["gy"] = 100 * data["grad_x"].astype(float), 100 * data["grad_y"].astype(float)
    data["depth"] = SUN_ALT_REF - data["sun_alt"].astype(float) if "sun_alt" in data.columns else np.nan
    relation = load_kmirror_relation(kmirror)
    data["phi"] = _kmirror_rotation(data, relation["field_sign"] if relation else -1)
    return data


def science_gradients(summary: Union[str, Table], drpver: str = None, redux_dir: str = None, kmirror=None) -> pd.DataFrame:
    """Gradient across the science IFU of each science frame, with its sky geometry

    The gradients are the per frame medians of the sky line and continuum
    measurements of the flat field QA (:func:`lvmdrp.qa.fiberflats.frame_gradients`);
    the geometry (directions of the horizon, the Moon and the Sun across the
    IFU, position angle and siderostat field angle) comes from the header of
    the lvmFrame with :func:`lvmdrp.qa.fiberflats.twilight_geometry`.

    Parameters
    ----------
    summary : str or astropy.table.Table
        Summary table of the flat field QA, or the path of its FITS file.
    drpver : str, optional
        Pipeline version of the lvmFrames. Default is the one in the summary.
    redux_dir : str, optional
        Reductions root directory. Default is $LVM_SPECTRO_REDUX.
    kmirror : dict or str, optional
        K-mirror relation for the frames without ``POSCIPA``.

    Returns
    -------
    pandas.DataFrame
        One row per frame and channel, with the columns of a trends table.
    """
    from lvmdrp.main import read_fibermap

    summary = Table.read(summary, hdu=1) if isinstance(summary, (str, os.PathLike)) else Table(summary, copy=True)
    # the string columns come back from FITS as bytes
    summary.convert_bytestring_to_unicode()
    gradients = frame_gradients(summary).to_pandas()
    if gradients.empty:
        raise ValueError("no reliable gradient measurements in the flat field QA summary")
    redux_dir = redux_dir or os.getenv("LVM_SPECTRO_REDUX", ".")
    frames = summary.to_pandas()[["filename", "tileid", "mjd"] + (["drpver"] if "drpver" in summary.colnames else [])].drop_duplicates("filename")
    gradients = gradients.merge(frames.drop(columns="mjd"), on="filename", how="left")

    relation = load_kmirror_relation(kmirror)
    fibermap = read_fibermap(as_table=True)
    geometry = {}
    for row in gradients.drop_duplicates("expnum").itertuples():
        version = drpver or getattr(row, "drpver", None)
        path = os.path.join(redux_dir, str(version), tileid_grp(int(row.tileid)), str(int(row.tileid)), str(int(row.mjd)), row.filename)
        if not os.path.isfile(path):
            log.warning(f"missing {path}, skipping the geometry of exposure {row.expnum}")
            continue
        header = fits.getheader(path, 0)
        geometry[int(row.expnum)] = {"expnum": int(row.expnum), "mjd_night": _night_sjd(header, path),
                                     **twilight_geometry(header, fibermap, kmirror=relation)}
    if not geometry:
        raise FileNotFoundError(f"none of the lvmFrames of the summary are in {redux_dir}")
    data = gradients.merge(pd.DataFrame(list(geometry.values())), on="expnum", how="inner")
    data["mjd"] = data.pop("mjd_night")
    data["strength"] = np.hypot(data["grad_x"], data["grad_y"])
    data["direction"] = np.degrees(np.arctan2(data["grad_y"], data["grad_x"]))
    return data


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def _segments(mjd: np.ndarray, breaks) -> np.ndarray:
    """Segment of each night: 0 before the first break, k from the k-th break on; 'nights' gives a segment per night"""
    if isinstance(breaks, str) and breaks == "nights":
        return np.unique(mjd, return_inverse=True)[1]
    return np.searchsorted(np.sort(np.asarray(breaks if breaks is not None else [], dtype=float)), mjd, side="right")


def _segment_edges(mjd: np.ndarray, breaks) -> Dict[int, Tuple[float, float]]:
    """First and last night of the data in each segment"""
    segment = _segments(mjd, breaks)
    return {int(k): (float(mjd[segment == k].min()), float(mjd[segment == k].max())) for k in np.unique(segment)}


def gradient_design(data: pd.DataFrame, frame: str = "ifu", sky: Dict[str, Dict[str, str]] = None, breaks=None,
                    drift: bool = False) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """Design matrix of the gradient model, see the module docstring

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel, from :func:`prepare_gradients` (one or more sources).
    frame : str, optional
        Frame of the instrument gradient, a key of :data:`INSTRUMENT_FRAMES`. Default is 'ifu'.
    sky : dict, optional
        Sky gradients of each source, {source: {direction column: amplitude}},
        with the amplitudes in :data:`SKY_AMPLITUDES`. Default is :data:`DEFAULT_SKY`.
    breaks : list[int] or 'nights', optional
        MJDs where the instrument gradient may jump, or 'nights' for one per
        night. Default is none.
    drift : bool, optional
        Add a linear drift of the instrument gradient within each segment. Default is False.

    Returns
    -------
    tuple
        Design matrix (2n x k: the x components of the n exposures, then the
        y components), target vector (2n) and the label of each parameter:
        ('inst', segment, 'x'|'y'), ('drift', segment, 'x'|'y') or
        ('sky', source, column, amplitude, group).
    """
    sky = sky or DEFAULT_SKY
    n = len(data)
    mjd = data["mjd"].to_numpy(float)
    columns, labels = [], []
    rotation = INSTRUMENT_FRAMES[frame][1]
    if rotation is not None:
        phi = np.radians(rotation * data["phi"].to_numpy(float)) if rotation else np.zeros(n)
        cos, sin = np.cos(phi), np.sin(phi)
        segment = _segments(mjd, breaks)
        for k in np.unique(segment):
            m = (segment == k).astype(float)
            # R(phi) (Ix, Iy) = (cos Ix - sin Iy, sin Ix + cos Iy)
            basis = [("x", np.r_[m * cos, m * sin]), ("y", np.r_[-m * sin, m * cos])]
            columns += [b for _, b in basis]
            labels += [("inst", int(k), c) for c, _ in basis]
            if drift:
                years = (mjd - mjd[segment == k].mean()) / 365.25 * m
                columns += [np.r_[years * cos, years * sin], np.r_[-years * sin, years * cos]]
                labels += [("drift", int(k), "x"), ("drift", int(k), "y")]

    for source, terms in sky.items():
        in_source = (data["source"] == source).to_numpy()
        if not in_source.any():
            continue
        for column, amplitude in terms.items():
            if amplitude not in SKY_AMPLITUDES:
                raise ValueError(f"unknown sky amplitude {amplitude!r}, use one of {SKY_AMPLITUDES}")
            angle = np.radians(data[column].to_numpy(float))
            ux, uy = np.where(in_source, np.cos(angle), 0.0), np.where(in_source, np.sin(angle), 0.0)
            if amplitude in ("constant", "depth"):
                groups = {"all": in_source.astype(float)}
            else:
                key = data["mjd" if amplitude == "night" else "expnum"].to_numpy()
                groups = {value: (in_source & (key == value)).astype(float) for value in np.unique(key[in_source])}
            for group, m in groups.items():
                columns.append(np.r_[m * ux, m * uy])
                labels.append(("sky", source, column, amplitude, group))
            if amplitude == "depth":
                depth = np.nan_to_num(data["depth"].to_numpy(float))
                columns.append(np.r_[depth * ux, depth * uy])
                labels.append(("sky", source, column, "slope", "all"))
    target = np.r_[data["gx"].to_numpy(float), data["gy"].to_numpy(float)]
    design = np.column_stack(columns) if columns else np.zeros((2 * n, 0))
    return design, target, labels


def _usable(data: pd.DataFrame, frame: str, sky: Dict[str, Dict[str, str]]) -> np.ndarray:
    """Rows with all the values the model needs"""
    good = np.isfinite(data["gx"].to_numpy(float)) & np.isfinite(data["gy"].to_numpy(float))
    if INSTRUMENT_FRAMES[frame][1]:
        good &= np.isfinite(data["phi"].to_numpy(float))
    for source, terms in sky.items():
        in_source = (data["source"] == source).to_numpy()
        for column, amplitude in terms.items():
            values = data[column].to_numpy(float) if column in data.columns else np.full(len(data), np.nan)
            ok = np.isfinite(values)
            if amplitude == "depth":
                ok &= np.isfinite(data["depth"].to_numpy(float))
            good &= ~in_source | ok
    return good


def _single_thread():
    """Limit BLAS to one thread: the many small least squares fits are much slower with multithreaded OpenBLAS"""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def _solve(design, target):
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    return params, target - design @ params


def _holdout_residuals(design, target, nights, n):
    """Leave-one-night-out residuals and the parameters fitted without each night

    The parameters only supported by the held-out night (its own sky
    amplitudes, or the instrument gradient of a segment with that night only)
    are fitted on the held-out night with the others fixed, so the residuals
    measure how well the other nights predict the shared parameters.
    """
    residuals = np.full(target.shape, np.nan)
    jackknife = []
    for night in np.unique(nights):
        test = np.r_[nights == night, nights == night]
        train = ~test
        supported = np.any(design[train] != 0, axis=0)
        params = np.full(design.shape[1], np.nan)
        params[supported], _ = _solve(design[train][:, supported], target[train])
        predicted = design[test][:, supported] @ params[supported]
        if (~supported).any():
            local, _ = _solve(design[test][:, ~supported], target[test] - predicted)
            predicted = predicted + design[test][:, ~supported] @ local
        residuals[test] = target[test] - predicted
        jackknife.append(params)
    return residuals, np.array(jackknife)


def fit_gradient_model(data: pd.DataFrame, frame: str = "ifu", sky: Dict[str, Dict[str, str]] = None, breaks=None,
                       drift: bool = False, crossval: bool = True) -> Dict:
    """Fits the gradient model of the module docstring to the gradients of one channel

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel, from :func:`prepare_gradients` (one or more
        sources; the instrument gradient is shared by all of them).
    frame, sky, breaks, drift
        See :func:`gradient_design`.
    crossval : bool, optional
        Compute the leave-one-night-out rms and the jackknife errors of the
        instrument gradient. Default is True.

    Returns
    -------
    dict
        n (exposures), nights, k (parameters), rank, rms (% per exposure, of
        the 2D residual), cv_rms (leave-one-night-out), bic, frame, breaks,
        instrument (pandas.DataFrame per segment: first and last night, n,
        gx, gy, their jackknife errors over nights (NaN with fewer than 3
        nights) and formal errors, strength, direction, spread of the K-mirror
        angle), drift (likewise, %/yr), sky (pandas.DataFrame of the
        sky amplitudes), and data (the rows used, with the instrument and sky
        model and the residuals: inst_x, inst_y, sky_x, sky_y, res_x, res_y).
    """
    with _single_thread():
        return _fit_gradient_model(data, frame, sky or DEFAULT_SKY, breaks, drift, crossval)


def _fit_gradient_model(data, frame, sky, breaks, drift, crossval):
    good = _usable(data, frame, sky)
    data = data[good].reset_index(drop=True)
    n = len(data)
    result = {"n": n, "nights": int(data["mjd"].nunique()), "frame": frame,
              "breaks": breaks if isinstance(breaks, str) else list(breaks or [])}
    if n < 3:
        for column in ("inst_x", "inst_y", "sky_x", "sky_y", "res_x", "res_y"):
            data[column] = np.nan
        return {**result, "k": 0, "rank": 0, "rms": np.nan, "cv_rms": np.nan, "bic": np.nan,
                "instrument": pd.DataFrame(), "drift": pd.DataFrame(), "sky": pd.DataFrame(), "data": data}

    design, target, labels = gradient_design(data, frame=frame, sky=sky, breaks=breaks, drift=drift)
    params, residual = _solve(design, target)
    rss = float(residual @ residual)
    size, k = target.size, design.shape[1]
    result.update({"k": k, "rank": int(np.linalg.matrix_rank(design)) if k else 0,
                   "rms": float(np.sqrt(rss / n)), "bic": float(size * np.log(max(rss, 1e-12) / size) + k * np.log(size))})

    jackknife = None
    if crossval and result["nights"] > 2:
        cv_residual, jackknife = _holdout_residuals(design, target, data["mjd"].to_numpy(), n)
        result["cv_rms"] = float(np.sqrt(np.nansum(cv_residual ** 2) / n))
    else:
        result["cv_rms"] = np.nan

    # formal errors, which ignore that the exposures of a night share their errors; they show how well each parameter is constrained
    with np.errstate(invalid="ignore"):
        formal = np.sqrt(np.diag(rss / max(size - result["rank"], 1) * np.linalg.pinv(design.T @ design)))

    def errors(index):
        if jackknife is None:
            return np.nan
        values = jackknife[:, index]
        values = values[np.isfinite(values)]
        m = values.size
        return float(np.sqrt((m - 1) / m * np.sum((values - values.mean()) ** 2))) if m > 2 else np.nan

    mjd = data["mjd"].to_numpy(float)
    segment = _segments(mjd, breaks)
    edges = _segment_edges(mjd, breaks)
    phi = data["phi"].to_numpy(float)

    def per_segment(kind):
        rows = []
        for k_seg in sorted(edges):
            ix = [i for i, label in enumerate(labels) if label[:2] == (kind, k_seg)]
            if len(ix) != 2:
                continue
            gx, gy = params[ix]
            in_segment = segment == k_seg
            nights = int(np.unique(mjd[in_segment]).size)
            # the jackknife over nights needs a few nights in the segment to see the scatter of its own nights
            gx_err, gy_err = (errors(ix[0]), errors(ix[1])) if nights >= 3 else (np.nan, np.nan)
            rows.append({"segment": k_seg, "mjd_first": edges[k_seg][0], "mjd_last": edges[k_seg][1],
                         "n": int(in_segment.sum()), "nights": nights, "gx": float(gx), "gy": float(gy),
                         "gx_err": gx_err, "gy_err": gy_err, "gx_sig": float(formal[ix[0]]), "gy_sig": float(formal[ix[1]]),
                         "strength": float(np.hypot(gx, gy)), "direction": float(np.degrees(np.arctan2(gy, gx))),
                         "phi_spread": float(np.nanstd(phi[in_segment])) if np.isfinite(phi[in_segment]).any() else np.nan})
        return pd.DataFrame(rows)

    result["instrument"] = per_segment("inst")
    result["drift"] = per_segment("drift")
    result["sky"] = pd.DataFrame([{"source": label[1], "direction": label[2], "amplitude": label[3], "group": label[4],
                                   "value": float(params[i]), "error": errors(i)}
                                  for i, label in enumerate(labels) if label[0] == "sky"])

    is_inst = np.array([label[0] in ("inst", "drift") for label in labels], dtype=bool)
    inst = design[:, is_inst] @ params[is_inst] if is_inst.any() else np.zeros(target.size)
    sky_model = design[:, ~is_inst] @ params[~is_inst] if (~is_inst).any() else np.zeros(target.size)
    data["inst_x"], data["inst_y"] = inst[:n], inst[n:]
    data["sky_x"], data["sky_y"] = sky_model[:n], sky_model[n:]
    data["res_x"], data["res_y"] = residual[:n], residual[n:]
    result["data"] = data
    return result


def _label_breaks(breaks) -> str:
    if isinstance(breaks, str):
        return breaks
    return ", ".join(str(int(b)) for b in breaks) if breaks else "none"


def _label_sky(sky: Dict[str, Dict[str, str]], sources) -> str:
    return "; ".join(f"{source}: " + " + ".join(f"{column.replace('dir_', '')} ({amplitude})" for column, amplitude in terms.items())
                     for source, terms in sky.items() if source in sources)


def _instrument_label(instrument: pd.DataFrame) -> str:
    if instrument.empty:
        return "-"
    return " | ".join(f"{row.strength:.2f}% @ {row.direction:.0f}°" for row in instrument.itertuples())


def compare_gradient_models(data: pd.DataFrame, frames: Sequence[str] = tuple(INSTRUMENT_FRAMES), skies: Dict[str, Dict] = None,
                            breaks: Dict[str, object] = None, drift: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Fits every combination of instrument frame, sky model and breaks to the gradients of one channel

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel (one or more sources), from :func:`prepare_gradients`.
    frames : list[str], optional
        Instrument frames, keys of :data:`INSTRUMENT_FRAMES`. Default is all.
    skies : dict, optional
        Sky models to try, {label: {source: {direction column: amplitude}}}.
        Default is :data:`DEFAULT_SKY` only.
    breaks : dict, optional
        Breaks to try, {label: list of MJDs or 'nights'}. Default is no breaks.
    drift : bool, optional
        Also try a linear drift within the segments. Default is False.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        One row per model (frame, sky, breaks, drift, n, k, rank, rms, cv_rms,
        bic, dbic from the lowest BIC, degenerate: rank deficient, an instrument
        gradient with a formal error above half the rms, or a K-mirror frame
        with a fixed K-mirror; instrument), sorted by BIC,
        and the fits by model key (frame, sky label, breaks label, drift).
    """
    skies = skies or {"default": DEFAULT_SKY}
    breaks = breaks or {"none": []}
    sources = set(data["source"])
    rows, fits = [], {}
    for frame in frames:
        for sky_label, sky in skies.items():
            for breaks_label, breaks_value in breaks.items():
                for with_drift in ((False, True) if drift else (False,)):
                    if INSTRUMENT_FRAMES[frame][1] is None and (breaks_label != list(breaks)[0] or with_drift):
                        continue  # without an instrument gradient, the breaks and the drift change nothing
                    fit = fit_gradient_model(data, frame=frame, sky=sky, breaks=breaks_value, drift=with_drift)
                    key = (frame, sky_label, breaks_label, with_drift)
                    fits[key] = fit
                    instrument = fit["instrument"]
                    phi_spread = instrument["phi_spread"].max() if not instrument.empty else np.nan
                    # an instrument gradient less constrained than a single exposure, e.g. one night with a free sky amplitude per night
                    loose = not instrument.empty and bool((instrument[["gx_sig", "gy_sig"]].max(axis=1) > 0.5 * fit["rms"]).any())
                    rows.append({"frame": frame, "sky": sky_label, "sky_model": _label_sky(sky, sources), "breaks": breaks_label,
                                 "drift": with_drift, "n": fit["n"], "nights": fit["nights"], "k": fit["k"], "rank": fit["rank"],
                                 "rms": fit["rms"], "cv_rms": fit["cv_rms"], "bic": fit["bic"],
                                 "degenerate": bool(fit["rank"] < fit["k"] or loose
                                                    or (INSTRUMENT_FRAMES[frame][1] and np.isfinite(phi_spread) and phi_spread < 1.0)),
                                 "instrument": _instrument_label(fit["instrument"])})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["dbic"] = table["bic"] - table["bic"].min()
        table = table.sort_values("bic").reset_index(drop=True)
    return table, fits


def residual_alignment(fit: Dict, references: Sequence[str] = ("dir_sun", "dir_horizon", "dir_moon")) -> pd.DataFrame:
    """Direction of the residual gradients, after the instrument gradient is removed, in the IFU and relative to the sky directions

    After a correct instrument gradient is removed, what is left of the
    measured gradient is the sky gradient, so its direction should stay fixed
    relative to its sky direction (mean resultant length R near 1) and change
    in IFU coordinates.

    Parameters
    ----------
    fit : dict
        Fit from :func:`fit_gradient_model`.
    references : list[str], optional
        Direction columns to compare with. Default is the Sun, the horizon and the Moon.

    Returns
    -------
    pandas.DataFrame
        For the measured gradient (g) and the gradient minus the instrument
        model (g - I): frame, R, circular mean, Rayleigh p-value and n.
    """
    data = fit["data"]
    rows = []
    for label, gx, gy in (("g", data["gx"], data["gy"]), ("g - I", data["gx"] - data["inst_x"], data["gy"] - data["inst_y"])):
        direction = np.degrees(np.arctan2(np.asarray(gy, dtype=float), np.asarray(gx, dtype=float)))
        for frame, reference in [("IFU", None)] + [(ref.replace("dir_", "").capitalize(), ref) for ref in references if ref in data.columns]:
            angles = direction if reference is None else _wrap_angle(direction - data[reference].to_numpy(float))
            mean, length = _resultant_length(angles)
            n = int(np.isfinite(angles).sum())
            rows.append({"gradient": label, "frame": frame, "R": length, "mean": mean, "p": _rayleigh_pvalue(n, length), "n": n})
    return pd.DataFrame(rows)


def sky_direction_scan(data: pd.DataFrame, column: str = "dir_sun", amplitude: str = "depth", breaks=None,
                       offsets: Sequence[float] = tuple(range(-180, 180, 5))) -> pd.DataFrame:
    """Fits with the sky direction rotated and mirrored, to test the geometry of the sky gradient

    An error in the position angle or in the orientation of the IFU would
    shift (or mirror) the computed direction of the sky gradient. With a right
    geometry, the fit is best with no offset; an instrument gradient that only
    makes up for a wrong geometry would be needed less at the best offset.

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel and one source, from :func:`prepare_gradients`.
    column : str, optional
        Direction of the sky gradient. Default is 'dir_sun'.
    amplitude : str, optional
        Its amplitude, see :data:`SKY_AMPLITUDES`. Default is 'depth'.
    breaks : list[int], optional
        Breaks of the instrument gradient. Default is none.
    offsets : list[float], optional
        Offsets added to the direction, in degrees.

    Returns
    -------
    pandas.DataFrame
        offset, mirrored, rms and bic with an IFU-fixed instrument gradient
        (rms_inst, bic_inst) and without one (rms_sky, bic_sky).
    """
    source = data["source"].iloc[0]
    rows = []
    for mirrored in (False, True):
        for offset in offsets:
            shifted = data.copy()
            shifted[column] = (-1 if mirrored else 1) * data[column].astype(float) + offset
            row = {"offset": float(offset), "mirrored": mirrored}
            for frame, suffix in (("ifu", "inst"), ("none", "sky")):
                fit = fit_gradient_model(shifted, frame=frame, sky={source: {column: amplitude}}, breaks=breaks, crossval=False)
                row.update({f"rms_{suffix}": fit["rms"], f"bic_{suffix}": fit["bic"]})
            rows.append(row)
    return pd.DataFrame(rows)


def predict_instrument(data: pd.DataFrame, fit: Dict) -> pd.DataFrame:
    """Instrument gradient of a fit at other exposures, e.g. twilight I at science frames

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of the same channel, from :func:`prepare_gradients`.
    fit : dict
        Fit from :func:`fit_gradient_model`. Exposures out of the nights of a
        segment take the instrument gradient of the segment they fall in.

    Returns
    -------
    pandas.DataFrame
        `data` with inst_x and inst_y (%), NaN out of the segments of the fit.
    """
    data = data.copy()
    data["inst_x"], data["inst_y"] = np.nan, np.nan
    instrument = fit["instrument"]
    if instrument.empty:
        data["inst_x"], data["inst_y"] = 0.0, 0.0
        return data
    segment = _segments(data["mjd"].to_numpy(float), fit["breaks"]) if not isinstance(fit["breaks"], str) else None
    if segment is None:
        raise ValueError("cannot predict an instrument gradient fitted per night at other exposures")
    rotation = INSTRUMENT_FRAMES[fit["frame"]][1]
    phi = np.radians(rotation * data["phi"].to_numpy(float)) if rotation else np.zeros(len(data))
    for row in instrument.itertuples():
        m = segment == row.segment
        data.loc[m, "inst_x"] = np.cos(phi[m]) * row.gx - np.sin(phi[m]) * row.gy
        data.loc[m, "inst_y"] = np.sin(phi[m]) * row.gx + np.cos(phi[m]) * row.gy
    return data


def cross_check(data: pd.DataFrame, fit: Dict, sky: Dict[str, str]) -> Dict:
    """How well the instrument gradient of one source explains the gradients of another

    Removes the predicted instrument gradient from the gradients of `data`
    and fits only their sky gradient, without and with the correction.

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel and source, from :func:`prepare_gradients`.
    fit : dict
        Fit of the instrument gradient on another source, from :func:`fit_gradient_model`.
    sky : dict
        Sky gradients of `data`, {direction column: amplitude}.

    Returns
    -------
    dict
        n, rms of the gradients around zero (raw) and after the correction
        (corrected), rms of the sky-only fit without (sky_raw) and with the
        correction (sky_corrected), and the alignment of the corrected
        gradients (see :func:`residual_alignment`).
    """
    source = data["source"].iloc[0]
    predicted = predict_instrument(data, fit)
    predicted = predicted[np.isfinite(predicted["inst_x"])].reset_index(drop=True)
    if predicted.empty:
        return {"n": 0}
    corrected = predicted.copy()
    corrected["gx"], corrected["gy"] = predicted["gx"] - predicted["inst_x"], predicted["gy"] - predicted["inst_y"]
    raw_fit = fit_gradient_model(predicted, frame="none", sky={source: sky}, crossval=False)
    corrected_fit = fit_gradient_model(corrected, frame="none", sky={source: sky}, crossval=False)
    corrected_fit["data"]["inst_x"], corrected_fit["data"]["inst_y"] = 0.0, 0.0
    return {"n": len(predicted),
            "raw": float(np.sqrt(np.mean(predicted["gx"] ** 2 + predicted["gy"] ** 2))),
            "corrected": float(np.sqrt(np.mean(corrected["gx"] ** 2 + corrected["gy"] ** 2))),
            "sky_raw": raw_fit["rms"], "sky_corrected": corrected_fit["rms"],
            "alignment": residual_alignment(corrected_fit)}


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def _num(value, fmt):
    return "-" if value is None or not np.isfinite(value) else format(value, fmt)


def figure_vectors(fit: Dict, title: str = None) -> go.Figure:
    """Measured gradients in IFU coordinates by segment, with the instrument gradient of each segment"""
    data, instrument = fit["data"], fit["instrument"]
    segment = _segments(data["mjd"].to_numpy(float), fit["breaks"]) if not isinstance(fit["breaks"], str) else np.zeros(len(data), int)
    fig = go.Figure()
    for i, k in enumerate(np.unique(segment)):
        sel = data[segment == k]
        color = SERIES[i % len(SERIES)]
        name = f"{int(sel['mjd'].min())}-{int(sel['mjd'].max())}"
        fig.add_trace(go.Scatter(x=sel["gx"], y=sel["gy"], mode="markers", name=f"g, {name}", legendgroup=name,
                                 marker=dict(size=6, color=color, opacity=0.45),
                                 customdata=np.stack([sel["mjd"], sel["expnum"]], axis=-1),
                                 hovertemplate="MJD %{customdata[0]} · exposure %{customdata[1]}<br>(%{x:.2f}, %{y:.2f})%<extra></extra>"))
        row = instrument[instrument["segment"] == k] if not instrument.empty else instrument
        if len(row):
            row = row.iloc[0]
            fig.add_trace(go.Scatter(x=[0, row.gx], y=[0, row.gy], mode="lines+markers", name=f"I, {name}", legendgroup=name,
                                     line=dict(color=color, width=3), marker=dict(size=[0, 14], symbol="diamond", color=color,
                                                                                    line=dict(width=2, color=THEME["ink"])),
                                     error_x=dict(type="data", array=[0, row.gx_err], visible=True, color=THEME["ink2"]),
                                     error_y=dict(type="data", array=[0, row.gy_err], visible=True, color=THEME["ink2"]),
                                     hovertemplate=f"<b>I {name}</b><br>{row.strength:.2f}% @ {row.direction:.0f}°<extra></extra>"))
    limit = float(np.nanpercentile(np.abs(np.r_[data["gx"], data["gy"]]), 98)) * 1.15 if len(data) else 1.0
    fig.update_layout(base_layout(520, xaxis_title="g<sub>x</sub> (%)", yaxis_title="g<sub>y</sub> (%)", title=title))
    style_axes(fig)
    fig.update_xaxes(range=[-limit, limit], zeroline=True, constrain="domain")
    fig.update_yaxes(range=[-limit, limit], zeroline=True, scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def figure_nightly_instrument(fits: Dict[str, Dict], breaks: Sequence[int] = ()) -> go.Figure:
    """Instrument gradient fitted per night (sky model shared by all nights), strength and direction against the night"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for i, (label, fit) in enumerate(fits.items()):
        instrument = fit["instrument"]
        if instrument.empty:
            continue
        color = SERIES[i % len(SERIES)]
        err = np.hypot(instrument["gx_err"], instrument["gy_err"])
        direction_err = np.degrees(err / np.maximum(instrument["strength"], 1e-6))
        common = dict(mode="markers", legendgroup=label, marker=dict(size=7, color=color),
                      customdata=np.stack([instrument["mjd_first"], instrument["n"]], axis=-1))
        fig.add_trace(go.Scatter(x=instrument["mjd_first"], y=instrument["strength"], name=label,
                                 error_y=dict(type="data", array=err, visible=True, thickness=1),
                                 hovertemplate="MJD %{customdata[0]:.0f} · %{customdata[1]} exposures<br>%{y:.2f}%<extra></extra>", **common), row=1, col=1)
        fig.add_trace(go.Scatter(x=instrument["mjd_first"], y=instrument["direction"], name=label, showlegend=False,
                                 error_y=dict(type="data", array=np.minimum(direction_err, 180), visible=True, thickness=1),
                                 hovertemplate="MJD %{customdata[0]:.0f} · %{customdata[1]} exposures<br>%{y:.0f}°<extra></extra>", **common), row=2, col=1)
    for mjd in breaks:
        fig.add_vline(x=mjd, line=dict(color=THEME["muted"], width=1, dash="dot"))
    fig.update_layout(base_layout(620))
    style_axes(fig)
    fig.update_yaxes(title_text="Strength (%)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Direction (°)", range=[-180, 180], dtick=90, row=2, col=1)
    fig.update_xaxes(title_text="MJD", row=2, col=1)
    return fig


def figure_direction_scan(scan: pd.DataFrame) -> go.Figure:
    """rms of the fits against the offset added to the sky direction, with and without an instrument gradient"""
    fig = go.Figure()
    for mirrored, dash in ((False, "solid"), (True, "dot")):
        sel = scan[scan["mirrored"] == mirrored]
        suffix = " (mirrored)" if mirrored else ""
        fig.add_trace(go.Scatter(x=sel["offset"], y=sel["rms_inst"], mode="lines", name=f"sky + I{suffix}", line=dict(color=SERIES[0], dash=dash)))
        fig.add_trace(go.Scatter(x=sel["offset"], y=sel["rms_sky"], mode="lines", name=f"sky only{suffix}", line=dict(color=SERIES[1], dash=dash)))
    fig.add_vline(x=0, line=dict(color=THEME["muted"], width=1, dash="dot"))
    fig.update_layout(base_layout(420, xaxis_title="Offset added to the sky direction (°)", yaxis_title="Residual rms (%)"))
    style_axes(fig)
    fig.update_xaxes(dtick=45)
    fig.update_yaxes(rangemode="tozero")
    return fig


def figure_residuals(fit: Dict) -> go.Figure:
    """Residual of the best fit against the night"""
    data = fit["data"]
    fig = go.Figure(go.Scatter(x=data["mjd"], y=np.hypot(data["res_x"], data["res_y"]), mode="markers",
                               marker=dict(size=6, color=SERIES[0], opacity=0.6),
                               customdata=data["expnum"], hovertemplate="exposure %{customdata}<br>%{y:.2f}%<extra></extra>"))
    fig.update_layout(base_layout(320, xaxis_title="MJD", yaxis_title="|residual| (%)"))
    style_axes(fig)
    fig.update_yaxes(rangemode="tozero")
    return fig


def _comparison_table(table: pd.DataFrame) -> str:
    rows = [[escape(INSTRUMENT_FRAMES[r.frame][0]), escape(r.sky_model), escape(str(r.breaks)), "yes" if r.drift else "",
             str(r.n), str(r.k), _num(r.rms, ".2f"), _num(r.cv_rms, ".2f"), _num(r.dbic, ".0f"),
             "degenerate" if r.degenerate else "", escape(r.instrument)] for r in table.itertuples()]
    return html_table([("Instrument", "frame of the instrument gradient"), ("Sky", "direction (amplitude) per source"),
                       ("Breaks", "MJDs where the instrument gradient may jump"), ("Drift", "linear drift within the segments"),
                       ("n", "exposures"), ("k", "parameters"), ("rms", "residual per exposure (%)"),
                       ("CV rms", "leave-one-night-out residual (%)"), ("ΔBIC", "from the best model"),
                       ("", "rank deficient, a poorly constrained segment, or a K-mirror frame with a fixed K-mirror"),
                       ("I per segment", "strength @ direction")],
                      rows, numeric=[False, False, False, False, True, True, True, True, True, False, False])


def _instrument_table(instrument: pd.DataFrame) -> str:
    def error(jackknife, formal):
        return _num(jackknife, ".2f") if np.isfinite(jackknife) else f"{_num(formal, '.2f')} (formal)"

    rows = [[f"{int(r.mjd_first)}-{int(r.mjd_last)}", str(r.nights), str(r.n), f"{r.gx:+.2f} ± {error(r.gx_err, r.gx_sig)}",
             f"{r.gy:+.2f} ± {error(r.gy_err, r.gy_sig)}", f"{r.strength:.2f}", f"{r.direction:.0f}", _num(r.phi_spread, ".1f")]
            for r in instrument.itertuples()]
    error_note = "jackknife error over nights, or the formal error with fewer than 3 nights"
    return html_table([("Nights", "first and last night"), ("n nights", ""), ("n", "exposures"), ("I_x (%)", error_note),
                       ("I_y (%)", error_note), ("Strength (%)", ""), ("Direction (°)", "from IFU +x towards +y"),
                       ("K-mirror spread (°)", "std of the K-mirror rotation; near 0 the K-mirror frames equal the IFU")],
                      rows, numeric=[False, True, True, True, True, True, True, True])


def _alignment_table(alignment: pd.DataFrame) -> str:
    rows = [[escape(r.gradient), escape(r.frame), _num(r.R, ".2f"), _num(r.mean, ".0f"), _num(r.p, ".1e"), str(r.n)] for r in alignment.itertuples()]
    return html_table([("Gradient", "measured, or minus the instrument model"), ("Frame", "direction measured in the IFU or relative to"),
                       ("R", "mean resultant length"), ("Mean (°)", "circular mean"), ("p", "Rayleigh test"), ("n", "")],
                      rows, numeric=[False, False, True, True, True, True])


def qa_instrument_gradient(twilight=None, dome=None, science=None, breaks: Sequence[int] = None, channels: str = "brz",
                           kmirror=None, skies: Dict[str, Dict] = None, output_dir: str = None, name: str = "instrument_gradient") -> Dict:
    """Tests for an instrument gradient across the science IFU and writes a dashboard

    For each channel and source (twilight, dome, science, and all of them
    together), fits the gradient model of the module docstring with every
    instrument frame (none, IFU, before the K-mirror) and with and without the
    breaks, and ranks the models by BIC. For the best model with an
    instrument gradient it shows the instrument gradient per segment, the
    instrument gradient fitted night by night, the alignment of the residual
    gradients with the sky directions, a scan of the sky direction (twilights)
    and how well the twilight instrument gradient explains the dome and
    science gradients.

    Parameters
    ----------
    twilight, dome : str or pandas.DataFrame, optional
        Tables of :func:`lvmdrp.qa.fiberflats.qa_twilight_trends` and
        :func:`~lvmdrp.qa.fiberflats.qa_dome_trends`, or their CSV paths.
    science : str or pandas.DataFrame, optional
        Table of :func:`science_gradients`, or its CSV path.
    breaks : list[int], optional
        MJDs where the instrument gradient may jump, e.g. from
        :func:`intervention_breaks`. Default is none.
    channels : str, optional
        Channels to fit. Default is 'brz'.
    kmirror : dict or str, optional
        K-mirror relation, for the sign of the field angle.
    skies : dict, optional
        Sky models to compare, {label: {source: {direction column: amplitude}}}.
        Default is :data:`DEFAULT_SKY` and, as an alternative, a free amplitude
        per night along the same directions.
    output_dir : str, optional
        Directory of the dashboard and tables. Default is
        ``qa/fiberflat/{name}`` in the master calibrations directory.
    name : str, optional
        Name of the dashboard. Default is 'instrument_gradient'.

    Returns
    -------
    dict
        comparison (pandas.DataFrame of all the models), instrument
        (pandas.DataFrame of the instrument gradient of the best model per
        channel and source), cross_checks, figures and report (path).
    """
    inputs = {"twilight": twilight, "dome": dome, "science": science}
    sources = {source: prepare_gradients(table, source, kmirror=kmirror, channels=channels)
               for source, table in inputs.items() if table is not None}
    if not sources:
        raise ValueError("give at least one table of gradients (twilight, dome or science)")
    breaks = sorted(int(b) for b in (breaks or []))
    skies = skies or {"default": DEFAULT_SKY,
                      "per night": {s: {column: "night" for column in terms} for s, terms in DEFAULT_SKY.items()}}
    breaks_options = {"none": []} if not breaks else {"none": [], "interventions": breaks}
    output_dir = output_dir or sandbox_qa_dir("fiberflat", name)
    os.makedirs(output_dir, exist_ok=True)

    groups = dict(sources)
    if len(sources) > 1:
        groups["joint"] = pd.concat(list(sources.values()), ignore_index=True)

    comparisons, instruments, figures, sections, cross_checks = [], [], {}, [], {}
    for group, data in groups.items():
        for channel in [c for c in channels if c in set(data["channel"])]:
            subset = data[data["channel"] == channel].reset_index(drop=True)
            table, fits = compare_gradient_models(subset, skies=skies, breaks=breaks_options)
            table.insert(0, "channel", channel)
            table.insert(0, "group", group)
            comparisons.append(table)
            best = table[(table["frame"] != "none") & ~table["degenerate"] & np.isfinite(table["bic"])].head(1)
            if best.empty:
                log.warning(f"no constrained model with an instrument gradient for the {group} gradients of channel {channel} ({len(subset)} exposures)")
                continue
            best = best.iloc[0]
            fit = fits[(best.frame, best.sky, best.breaks, best.drift)]
            instrument = fit["instrument"].copy()
            instrument.insert(0, "frame", best.frame)
            instrument.insert(0, "channel", channel)
            instrument.insert(0, "group", group)
            instruments.append(instrument)
            sky_model = skies[best.sky]
            nightly = {"IFU": fit_gradient_model(subset, frame="ifu", sky=sky_model, breaks="nights")}
            figures[f"vectors|{group}|{channel}"] = figure_vectors(fit)
            figures[f"nightly|{group}|{channel}"] = figure_nightly_instrument(nightly, breaks)
            figures[f"residuals|{group}|{channel}"] = figure_residuals(fit)
            alignment = residual_alignment(fit)

            scan_html = ""
            if group == "twilight":
                column, amplitude = next(iter(DEFAULT_SKY["twilight"].items()))
                scan = sky_direction_scan(subset, column=column, amplitude=amplitude, breaks=fit["breaks"])
                figures[f"scan|{group}|{channel}"] = figure_direction_scan(scan)
                best_scan = scan.loc[scan["bic_inst"].idxmin()]
                scan_html = (f"<h3>Geometry of the sky gradient</h3><p>The fit with the direction of the Sun rotated by an offset, "
                             f"and mirrored. The best offset is {best_scan.offset:+.0f}°{' mirrored' if best_scan.mirrored else ''} "
                             f"(rms {best_scan.rms_inst:.2f}% against {scan[(scan.offset == 0) & ~scan.mirrored].rms_inst.iloc[0]:.2f}% "
                             f"with no offset); without an instrument gradient the best rms is {scan.rms_sky.min():.2f}%.</p>"
                             f'<div class="chart" data-fig="scan|{group}|{channel}"></div>')

            check_html = ""
            if group == "twilight":
                for other in ("dome", "science"):
                    if other not in sources:
                        continue
                    other_data = sources[other][sources[other]["channel"] == channel].reset_index(drop=True)
                    if other_data.empty:
                        continue
                    check = cross_check(other_data, fit, DEFAULT_SKY[other])
                    cross_checks[(other, channel)] = check
                    if check["n"] >= 3:
                        check_html += (f"<h3>Twilight I applied to the {other} gradients</h3><p>{check['n']} exposures: rms of the "
                                       f"gradients {check['raw']:.2f}% before and {check['corrected']:.2f}% after removing the twilight "
                                       f"instrument gradient; rms of their sky-only fit {check['sky_raw']:.2f}% before and "
                                       f"{check['sky_corrected']:.2f}% after.</p>" + _alignment_table(check["alignment"]))

            sections.append(
                f'<details{" open" if not sections else ""}><summary>{escape(group.capitalize())} · {channel}: '
                f'I = {escape(best.instrument)}</summary>'
                f"<h3>Model comparison</h3>{_comparison_table(table)}"
                f"<h3>Instrument gradient of the best model ({escape(INSTRUMENT_FRAMES[best.frame][0])}, sky {escape(best.sky)}, "
                f"breaks {escape(str(best.breaks))})</h3>{_instrument_table(fit['instrument'])}"
                f'<div class="chart" data-fig="vectors|{group}|{channel}"></div>'
                f"<h3>Instrument gradient night by night</h3><p>One IFU-fixed instrument gradient per night with the sky model "
                f"shared by all the nights; a jump marks a change of the instrument, the dotted lines are the breaks.</p>"
                f'<div class="chart" data-fig="nightly|{group}|{channel}"></div>'
                f"<h3>Direction of the residual gradients</h3><p>After a correct instrument gradient is removed, the rest keeps its "
                f"direction relative to the sky (R near 1) instead of the IFU.</p>{_alignment_table(alignment)}"
                f'<div class="chart" data-fig="residuals|{group}|{channel}"></div>{scan_html}{check_html}</details>')

    comparison = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    instrument = pd.concat(instruments, ignore_index=True) if instruments else pd.DataFrame()
    comparison.to_csv(os.path.join(output_dir, f"{name}_models.csv"), index=False)
    instrument.to_csv(os.path.join(output_dir, f"{name}_instrument.csv"), index=False)

    tiles = []
    for (group, channel), row in (instrument.groupby(["group", "channel"]).last().iterrows() if not instrument.empty else []):
        rows = comparison[(comparison["group"] == group) & (comparison["channel"] == channel)]
        none = rows[rows["frame"] == "none"]["bic"].min()
        best_bic = rows[(rows["frame"] != "none") & ~rows["degenerate"]]["bic"].min()
        tiles.append((f"{group} {channel}", f"{row.strength:.2f}% @ {row.direction:.0f}°",
                      f"last segment, {INSTRUMENT_FRAMES[row.frame][0]}; ΔBIC {none - best_bic:.0f} over no instrument gradient"))
    meta = [("Sources", ", ".join(f"{s}: {d['expnum'].nunique()} exposures, {d['mjd'].nunique()} nights" for s, d in sources.items())),
            ("Channels", channels), ("Breaks", _label_breaks(breaks)),
            ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]
    intro = ("<p>Tests whether the gradient that the pipeline fits across the science IFU holds an instrument gradient, fixed in "
             "the IFU or before the K-mirror, besides the sky gradient, fixed on the sky. Each model is fitted to the gradient vectors "
             "of all the exposures of a channel; the instrument gradient is shared by all the exposures of a segment between breaks, "
             "the sky gradient follows the direction of the Sun (twilights) or of the horizon (dome flats, science frames) across the "
             "IFU. Lower BIC is better; the leave-one-night-out rms checks that the models predict nights they did not see. In "
             "twilights and dome flats the K-mirror does not move, so only science frames tell an IFU-fixed gradient from one fixed "
             "before the K-mirror (flagged as degenerate otherwise).</p>")
    body = "".join(sections) or "<p>No model with an instrument gradient could be fitted.</p>"
    report = write_dashboard(os.path.join(output_dir, f"{name}.html"), "LVM Instrument Gradient", "LVM DRP · fiber flats · IFU gradient",
                             intro, meta, tiles, body, figures, "lvmdrp.qa.ifu_gradient.qa_instrument_gradient")
    log.info(f"written instrument gradient dashboard to {report}")
    return {"comparison": comparison, "instrument": instrument, "cross_checks": cross_checks, "figures": figures, "report": report}

