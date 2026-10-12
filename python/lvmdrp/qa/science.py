# encoding: utf-8
"""Per-exposure QA dashboard of a science reduction.

A self-contained interactive HTML dashboard, built with the shared QA report
theme and page (:mod:`lvmdrp.qa.report`), that follows the steps of
:func:`lvmdrp.main.science_reduction` in order:

1. detrended 2D frames of each camera, with the fiber positions at the
   reference column before and after the thermal shift correction, and the
   reference sky lines before and after the wavelength correction;
2. stray light residuals at the inter-block strips where it was measured;
3. thermal fiber shifts measured at extraction;
4. wavelength offsets measured from the reference sky lines;
5. IFU view of a sky line and a continuum window before and after fiber flat
   fielding, and after the wavelength rectification;
6. standard stars calibrated with each flux calibration method (MOD, STD,
   SCI), with their Gaia XP spectra and synthetic photometry;
7. sky subtraction residual maps for a selection of sky lines.

Measurements stored in the reduced products are reused instead of repeated:
the thermal fiber shifts come from the lvmFrame ``SLITMAP`` extension, the
corrected wavelengths from its ``WAVE_TRACE`` extension, the extracted spectra before
flat fielding from the lvmFrame ``SUPERFLAT`` extension, the standard stars
and the sensitivity curves of every flux calibration method from the
lvmCFrame, and the sky-subtracted spectra from the lvmSFrame. The detrended
frames, the stray light subtracted frames and the rectified frames are
ancillary products, which are removed when a reduction cleans its ancillary
directory. Missing ancillary products are logged and their parts of the
dashboard are skipped, unless ``remeasure=True``, which produces them again in
the dashboard directory with the routines and parameters of the reduction.
See :func:`qa_science_exposure`.
"""

import base64
import inspect
import io
import os
import warnings
from datetime import datetime, timezone
from html import escape
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
from astropy.io import fits
from astropy.table import Table
from plotly.subplots import make_subplots
from PIL import Image as PILImage
from scipy.stats import binned_statistic

from lvmdrp import log, path, __version__ as drpver
from lvmdrp.core import fluxcal
from lvmdrp.core.constants import (CAMERAS, LVM_BLOCKSIZE, LVM_REFERENCE_COLUMN, MASTERS_DIR, REF_SKYLINES,
                                   SKYLINES_FIBERFLAT, CONTINUUM_FIBERFLAT, SPEC_CHANNELS)
from lvmdrp.core.rss import RSS, lvmCFrame
from lvmdrp.core.tracemask import TraceMask
from lvmdrp.functions import imageMethod as image_tasks
from lvmdrp.functions import rssMethod as rss_tasks
from lvmdrp.functions import skyMethod as sky_tasks
from lvmdrp.qa.fiberflats import (SKYLINES_FLATFIELD_QA, _ifu_geometry, _integrate_line, measure_continuum,
                                  measure_skylines)
from lvmdrp.qa.report import (DIVERGING, SERIES, THEME, base_layout, exposure_qa_dir, html_table, picker,
                              style_axes, write_dashboard)
from lvmdrp.utils.metadata import get_frames_metadata
from lvmdrp.utils.paths import get_calib_paths, get_master_mjd, mjd_from_expnum


# parts of the dashboard, in the order of the reduction steps
QA_PARTS = ("detrending", "straylight", "fibershifts", "skyshifts", "ifu", "fluxcal", "skysub")
PART_TITLES = {"detrending": "Detrended frames", "straylight": "Stray light", "fibershifts": "Fiber thermal shifts",
               "skyshifts": "Sky line wavelength correction", "ifu": "Fiber flat fielding and rectification",
               "fluxcal": "Flux calibration", "skysub": "Sky subtraction"}

CHANNEL_COLORS = {"b": SERIES[0], "r": SERIES[1], "z": SERIES[2]}
SPEC_DASHES = {1: "solid", 2: "dash", 3: "dot"}
# calibrations used by science_reduction
CALIB_FLAVORS = ["pixmask", "pixflat", "bias", "centroids", "sigmas", "model", "wave", "lsf", "fiberflat_twilight"]
# the full frame is shown binned, the cutouts at full resolution
FULL_FRAME_BIN = 4
CUTOUT_HALF_WIDTH = 25
MAX_LINE_VIEWS = 4
# columns binned together along the stray light strips
STRAY_COLUMN_BIN = 64
# fibers binned together in the sky residual maps
FIBER_BIN = 8
# wavelength bin of the standard star spectra shown, in Angstroms
STD_BIN = 2.0
# flux calibration methods compared, in the order of the charts
FLUXCAL_METHODS = ("MOD", "STD", "SCI")
METHOD_COLORS = {"MOD": SERIES[0], "STD": SERIES[1], "SCI": SERIES[2]}
# stages of the IFU views
IFU_STAGES = (("extracted", "before flat fielding"), ("flatfielded", "after flat fielding"), ("rectified", "after rectification"))


def _defaults(func, *names):
    """Default values of the given parameters of a reduction routine, to measure again with the same parameters"""
    parameters = inspect.signature(func).parameters
    return {name: parameters[name].default for name in names}


def _r(values, decimals=3):
    """Round for the page as float32, keeping NaN as gaps"""
    return np.round(np.asarray(values, dtype=float), decimals).astype(np.float32)


def _num(value, fmt=".2f", missing="–"):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return missing
    return format(value, fmt) if np.isfinite(value) else missing


def _decode(values):
    values = np.asarray(values)
    return np.char.decode(values) if values.dtype.kind == "S" else values.astype(str)


def _robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------

class ExposureProducts:
    """Paths of the products of a science exposure needed by the dashboard

    Final products are always taken from the reductions. Ancillary products
    are taken from the ancillary directory, or from the dashboard work
    directory when produced there by a previous run. With ``remeasure=True``
    they are produced again in the work directory with the routines and
    parameters of :func:`lvmdrp.main.science_reduction`.

    Parameters
    ----------
    expnum : int
        exposure number
    work_dir : str
        directory where the ancillary products are produced again
    remeasure : bool, optional
        produce the ancillary products again, by default False
    use_nightly_cals, use_untagged_cals : bool, optional
        calibrations used by the reduction, as in ``drp run``
    drpver : str, optional
        pipeline version of the reductions, by default the current one
    """

    def __init__(self, expnum, work_dir, remeasure=False, use_nightly_cals=False, use_untagged_cals=False, drpver=drpver):
        self.expnum = int(expnum)
        self.work_dir = work_dir
        self.remeasure = remeasure
        self.drpver = drpver
        self.missing = []
        self._produced = {}

        self.mjd = mjd_from_expnum(self.expnum)[0]
        frames = get_frames_metadata(mjd=self.mjd)
        frames = frames.query("expnum == @self.expnum").sort_values("camera")
        if frames.empty:
            raise ValueError(f"exposure {self.expnum} not found in the metadata of MJD {self.mjd}")
        self.frames = {frame["camera"]: frame for frame in frames.to_dict("records")}
        self.tileid = int(frames["tileid"].iloc[0])
        self.imagetyp = frames["imagetyp"].iloc[0]

        use_untagged_cals = use_untagged_cals or use_nightly_cals
        self.calibs, self.cals_mjd = get_calib_paths(mjd=get_master_mjd(self.mjd), version=self.drpver, nightly=use_nightly_cals,
                                                     from_sandbox=not use_untagged_cals, flavors=CALIB_FLAVORS, return_mjd=True)

    def anc(self, kind, camera, imagetype=None):
        return path.full("lvm_anc", drpver=self.drpver, kind=kind, imagetype=imagetype or self.imagetyp, tileid=self.tileid,
                         mjd=self.mjd, camera=camera, expnum=self.expnum)

    def final(self, kind):
        """Path of a final product (e.g., kind='Frame-b'), or None with a warning if it does not exist"""
        product = path.full("lvm_frame", drpver=self.drpver, tileid=self.tileid, mjd=self.mjd, expnum=self.expnum, kind=kind)
        if os.path.isfile(product):
            return product
        log.warning(f"missing product {product}")
        self.missing.append(f"lvm{kind}: {os.path.basename(product)} not found")
        return None

    def calib(self, flavor, camera):
        calib = self.calibs.get(flavor, {}).get(camera)
        if calib is None or not os.path.isfile(calib):
            log.warning(f"missing {flavor} calibration for {camera}: {calib}")
            self.missing.append(f"{flavor} calibration of {camera} not found")
            return None
        return calib

    def _work(self, product):
        return os.path.join(self.work_dir, os.path.basename(product))

    def _get(self, kind, camera, produce, imagetype=None):
        """Path of an ancillary product, produced again in the work directory when remeasuring"""
        product = self.anc(kind, camera, imagetype=imagetype)
        work = self._work(product)
        if self.remeasure:
            if work not in self._produced:
                os.makedirs(self.work_dir, exist_ok=True)
                try:
                    produce()
                    self._produced[work] = os.path.isfile(work)
                except Exception as error:
                    log.error(f"failed producing {os.path.basename(work)} again: {type(error).__name__}: {error}")
                    self.missing.append(f"{os.path.basename(work)} could not be produced again: {type(error).__name__}: {error}")
                    self._produced[work] = False
            return work if self._produced[work] else None
        for candidate in (product, work):
            if os.path.isfile(candidate):
                return candidate
        log.warning(f"missing ancillary product {product}, skipping the parts that need it (measure it again with remeasure=True)")
        self.missing.append(f"{os.path.basename(product)} not found in the ancillary directory")
        return None

    def detrended(self, camera):
        """Detrended frame of a camera, as produced by :func:`lvmdrp.main.reduce_2d`"""
        def produce():
            from lvmdrp.main import fibermap
            frame = self.frames[camera]
            raw = path.full("lvm_raw", camspec=camera, **frame)
            pframe = self._work(self.anc("p", camera))
            dframe = self._work(self.anc("d", camera))
            # same as reduce_2d in science_reduction, without the astrometry, which only adds the WCS to the header
            image_tasks.preproc_raw_frame(in_image=raw, out_image=pframe, in_mask=self.calibs.get("pixmask", {}).get(camera),
                                          replace_with_nan=True, assume_imagetyp=None)
            image_tasks.detrend_frame(in_image=pframe, out_image=dframe, in_bias=self.calibs.get("bias", {}).get(camera),
                                      in_pixelflat=self.calibs.get("pixflat", {}).get(camera), replace_with_nan=True,
                                      reject_cr=True, in_slitmap=fibermap if self.imagetyp in {"flat", "arc", "object"} else None)
        return self._get("d", camera, produce)

    def straylight(self, camera):
        """Stray light subtracted frame and stray light model of a camera, as produced by :func:`lvmdrp.main.reduce_2d`"""
        def produce():
            dframe = self.detrended(camera)
            if dframe is None:
                raise FileNotFoundError(f"no detrended frame of {camera}")
            image_tasks.subtract_straylight(in_image=dframe, out_image=self._work(self.anc("l", camera)),
                                            out_stray=self._work(self.anc("d", camera, imagetype="stray")),
                                            in_cent_trace=self.calibs["centroids"][camera])
        lframe = self._get("l", camera, produce)
        stray = self.anc("d", camera, imagetype="stray")
        for candidate in ((self._work(stray),) if self.remeasure else (stray, self._work(stray))):
            if os.path.isfile(candidate):
                return lframe, candidate
        return lframe, None

    def rectified(self, channel):
        """Sky-interpolated and resampled frame of a channel, as produced by :func:`lvmdrp.main.science_reduction`"""
        def produce():
            frame = self.final(f"Frame-{channel}")
            if frame is None:
                raise FileNotFoundError(f"no lvmFrame of channel {channel}")
            sframe = self._work(self.anc("s", channel))
            hframe = self._work(self.anc("h", channel))
            sky_tasks.interpolate_sky(in_frame=frame, out_rss=sframe)
            sky_tasks.combine_skies(in_rss=sframe, out_rss=sframe, sky_weights=None)
            rss_tasks.resample_wavelength(in_rss=sframe, out_rss=hframe, wave_range=SPEC_CHANNELS[channel], wave_disp=0.5,
                                          convert_to_density=True)
        return self._get("h", channel, produce)


def _coeff_table_eval(table):
    """Evaluate a trace coefficients table (e.g., lvmFrame WAVE_TRACE) on the pixel grid"""
    return TraceMask.from_coeff_table(Table(table))._data


def read_frame(frame_path):
    """Read the parts of an lvmFrame used by the dashboard

    Returns
    -------
    dict
        header, flux, error, mask, superflat, slitmap, and the evaluated
        wavelength traces
    """
    with fits.open(frame_path, memmap=False) as hdus:
        ivar = hdus["IVAR"].data.astype(np.float32)
        with np.errstate(divide="ignore"):
            error = np.where(ivar > 0, 1 / np.sqrt(ivar), np.nan).astype(np.float32)
        return {"header": hdus["PRIMARY"].header.copy(), "data": hdus["FLUX"].data.astype(np.float32), "error": error,
                "mask": hdus["MASK"].data.astype(bool), "superflat": hdus["SUPERFLAT"].data.astype(np.float32),
                "slitmap": Table.read(hdus["SLITMAP"]), "wave": _coeff_table_eval(hdus["WAVE_TRACE"].data)}


def _spec_rows(slitmap, specid):
    return np.where(np.asarray(slitmap["spectrographid"]) == specid)[0]


def thermal_shifts(frame_slitmap, master_slitmap, camera):
    """Thermal shift of each fiber of a camera applied at extraction

    The extraction adds the shift of each fiber block to the master centroids
    and to the ``ypix_{channel}`` column of the slitmap (see
    :func:`~lvmdrp.functions.imageMethod.extract_spectra`). The centroids
    are stored as coefficients only when the master trace has them, while the
    slitmap always carries the shift, so the shift is the slitmap of the
    reduced frame minus the slitmap of the master centroids.

    Parameters
    ----------
    frame_slitmap, master_slitmap : astropy.table.Table
        slitmaps of the reduced frame and of the master centroids
    camera : str
        camera, e.g. 'b1'

    Returns
    -------
    numpy.ndarray
        shift in pixels of each fiber of the camera's spectrograph
    """
    channel, specid = camera[0], int(camera[1])
    frame_ypix = np.asarray(frame_slitmap[f"ypix_{channel}"], dtype=float)[_spec_rows(frame_slitmap, specid)]
    master_ypix = np.asarray(master_slitmap[f"ypix_{channel}"], dtype=float)[_spec_rows(master_slitmap, specid)]
    return frame_ypix - master_ypix


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

def reference_skylines(channel):
    """Reference sky lines of a channel used by the reduction, and what for"""
    lines = {cwave: ["wavelength correction"] for cwave in REF_SKYLINES[channel]}
    lines.setdefault(SKYLINES_FIBERFLAT[channel], []).append("fiber flat check")
    return [(cwave, ", ".join(uses)) for cwave, uses in sorted(lines.items())]


def line_crossings(wave, cent, cwave):
    """Column where each fiber reaches a wavelength, and the fiber centroid row there

    Parameters
    ----------
    wave : numpy.ndarray
        wavelength of each fiber and column (fibers x columns)
    cent : numpy.ndarray
        centroid row of each fiber and column (fibers x columns)
    cwave : float
        wavelength in Angstroms

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        column and row of each fiber, NaN where the fiber does not reach the wavelength
    """
    pixels = np.arange(wave.shape[1], dtype=float)
    x, y = np.full(wave.shape[0], np.nan), np.full(wave.shape[0], np.nan)
    for ifiber, (w, c) in enumerate(zip(wave, cent)):
        good = np.isfinite(w)
        if good.sum() < 2:
            continue
        w, p = w[good], pixels[good]
        if w[-1] < w[0]:
            w, p = w[::-1], p[::-1]
        if not w[0] <= cwave <= w[-1]:
            continue
        x[ifiber] = np.interp(cwave, w, p)
        if np.isfinite(c).any():
            y[ifiber] = np.interp(x[ifiber], pixels[np.isfinite(c)], c[np.isfinite(c)])
    return x, y


def measure_camera_traces(camera, master_cent, master_wave, frame_cent=None, frame_wave=None, ref_column=LVM_REFERENCE_COLUMN):
    """Fiber positions and reference sky lines of a camera before and after their corrections

    Parameters
    ----------
    camera : str
        camera, e.g. 'b1'
    master_cent, master_wave : numpy.ndarray
        master centroids and wavelengths used by the reduction (fibers x columns)
    frame_cent, frame_wave : numpy.ndarray, optional
        corrected centroids and wavelengths of the same fibers, from the reduced products
    ref_column : int, optional
        reference column of the fiber positions, by default 2000

    Returns
    -------
    dict
        fiber positions at the reference column ('y_master', 'y_frame', 'shift'),
        the applied wavelength offsets ('wave_offset', Angstroms) and, for each
        reference sky line, the column and row of each fiber before and after
        the corrections ('lines')
    """
    result = {"camera": camera, "ref_column": ref_column, "y_master": master_cent[:, ref_column], "lines": []}
    if frame_cent is not None:
        result["y_frame"] = frame_cent[:, ref_column]
        result["shift"] = result["y_frame"] - result["y_master"]
    if frame_wave is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result["wave_offset"] = np.nanmedian(master_wave - frame_wave, axis=1)
    for cwave, use in reference_skylines(camera[0]):
        line = {"wave": cwave, "use": use}
        line["x_master"], line["y_master"] = line_crossings(master_wave, master_cent, cwave)
        if frame_wave is not None:
            line["x_frame"], line["y_frame"] = line_crossings(frame_wave, frame_cent if frame_cent is not None else master_cent, cwave)
        result["lines"].append(line)
    return result


def _stretch_limits(data):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99.5])
    return float(lo), float(hi if hi > lo else lo + 1)


def encode_image(data, lo, hi, softening=10.0, quality=85):
    """JPEG data URI of an image with an asinh stretch between lo and hi, first row at the bottom

    JPEG keeps the noise of the frames at a fraction of the size of a PNG.
    """
    z = np.clip((data - lo) / (hi - lo), 0, 1)
    z = np.arcsinh(softening * z) / np.arcsinh(softening)
    z = np.nan_to_num(z, nan=0.0)
    buffer = io.BytesIO()
    PILImage.fromarray((255 * z[::-1]).astype(np.uint8), mode="L").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _bin_image(data, factor):
    ny, nx = (data.shape[0] // factor) * factor, (data.shape[1] // factor) * factor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(data[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor), axis=(1, 3))


def _read_image(image_path):
    with fits.open(image_path, memmap=False) as hdus:
        data = hdus["PRIMARY"].data.astype(np.float32)
        if "BADPIX" in hdus:
            data[hdus["BADPIX"].data.astype(bool)] = np.nan
        return data, hdus["PRIMARY"].header.get("BUNIT", "electron")


def measure_straylight(lframe, centroids, dframe=None, stray=None, column_bin=STRAY_COLUMN_BIN):
    """Stray light residuals at the pixels where the stray light was measured

    The stray light is fitted to the pixels between the fiber blocks and
    beyond the first and last fibers (see
    :func:`~lvmdrp.functions.imageMethod.subtract_straylight`). The same
    strips are selected here with the same routine and parameters, and the
    stray light subtracted frame there is the residual of the fit.

    Parameters
    ----------
    lframe : str or None
        stray light subtracted frame; if None, ``dframe - stray`` is used
    centroids : str
        master centroids used by the stray light subtraction
    dframe, stray : str, optional
        detrended frame and stray light model, used for the pixel mask and
        the stray light level
    column_bin : int, optional
        number of columns binned along the strips, by default 64

    Returns
    -------
    dict
        median residual ('residual'), stray light level ('stray') and number
        of pixels ('npix') per strip and column bin, the mean row of each strip
        ('rows'), the column bin centers ('columns') and the residuals of all
        strip pixels ('pixels')
    """
    pars = _defaults(image_tasks.subtract_straylight, "margin", "select_nrows")
    source = dframe if dframe is not None else lframe
    img = image_tasks.loadImage(source)
    # mask as in subtract_straylight
    mask = img._mask if img._mask is not None else np.zeros(img._data.shape, dtype=bool)
    mask = mask | ~np.isfinite(img._data) | (img._data == 0)
    if lframe is not None:
        residual = image_tasks.loadImage(lframe)._data.astype(float) if source != lframe else img._data.astype(float)
    else:
        residual = img._data - fits.getdata(stray, ext=0)
    stray_data = fits.getdata(stray, ext=0).astype(float) if stray is not None else None
    trace = TraceMask.from_file(centroids)
    strips, _, Y = img.straylight_binning(trace, y_margin=pars["margin"], nrows=pars["select_nrows"], return_bins=False)

    ncols = residual.shape[1]
    edges = np.arange(0, ncols + column_bin, column_bin)
    edges[-1] = ncols
    shape = (len(strips), edges.size - 1)
    result = {"residual": np.full(shape, np.nan), "stray": np.full(shape, np.nan), "npix": np.zeros(shape, dtype=int),
              "rows": np.full(len(strips), np.nan), "columns": 0.5 * (edges[1:] + edges[:-1]) - 0.5, "pixels": []}
    for istrip, strip in enumerate(strips):
        rows, cols = np.where(strip & ~mask & np.isfinite(residual))
        if rows.size == 0:
            continue
        values = residual[rows, cols]
        result["rows"][istrip] = Y[rows, cols].mean()
        result["pixels"].append(values)
        result["residual"][istrip] = binned_statistic(cols, values, statistic="median", bins=edges).statistic
        result["npix"][istrip] = np.histogram(cols, bins=edges)[0]
        if stray_data is not None:
            result["stray"][istrip] = binned_statistic(cols, stray_data[rows, cols], statistic="median", bins=edges).statistic
    result["pixels"] = np.concatenate(result["pixels"]) if result["pixels"] else np.array([])
    return result


def _frame_like(data, error, mask, wave, slitmap, header):
    """Minimal RSS-like object for the flat-field QA measurements"""
    return SimpleNamespace(_data=data, _error=error, _mask=mask, _wave=wave, _slitmap=slitmap, _header=header)


def measure_ifu(frame, cwave, window):
    """Normalized sky line flux and continuum level of each fiber

    The sky line is integrated with sideband continuum subtraction and the
    continuum averaged in a window, as in the flat-field QA (see
    :func:`~lvmdrp.qa.fiberflats.measure_skylines` and
    :func:`~lvmdrp.qa.fiberflats.measure_continuum`). Each value is
    normalized by the robust mean of the fibers of the same telescope.

    Returns
    -------
    dict
        normalized values of each fiber, keyed by 'line' and 'cont'
    """
    line = measure_skylines(frame, [cwave], method="integrate")
    cont = measure_continuum(frame, [window])
    return {"line": np.asarray(line[f"L{cwave:.2f}_norm"]), "cont": np.asarray(cont[f"C{window[0]:.0f}-{window[1]:.0f}_norm"])}


def standard_stars(header, slitmap):
    """Standard stars acquired in the exposure: index, fiber, Gaia ID, exposure time, row of the fiber"""
    labels = _decode(slitmap["orig_ifulabel"])
    stars = []
    for i in range(1, 16):
        if not header.get(f"STD{i}ACQ", False):
            continue
        fiber, gaia_id = header.get(f"STD{i}FIB"), header.get(f"STD{i}ID")
        rows = np.where(labels == fiber)[0]
        if gaia_id in (None, "") or rows.size == 0:
            continue
        stars.append({"index": i, "fiber": fiber, "gaia_id": int(gaia_id), "exptime": header.get(f"STD{i}EXP"), "row": int(rows[0])})
    return stars


def load_xp_spectrum(gaia_id, cache_dir, fetch=False):
    """Gaia XP spectrum of a star in Angstroms and erg/s/cm2/A, from the Gaia cache of the reduction

    Returns None if it is not cached, unless ``fetch=True``, which downloads
    it with the routine used by the flux calibration.
    """
    gaia = fluxcal.GaiaXPSpectra(cache_dir=cache_dir)
    if not gaia._xp_spectrum_exists(gaia_id):
        if not fetch:
            return None
        wave, flux = fluxcal.get_xp_spectra_from_ids([gaia_id], cache_dir=cache_dir)
        return np.asarray(wave, dtype=float), np.asarray(flux, dtype=float)[0]
    wave, flux = gaia.load_xp_spectra(gaia_id, convert_to_cgs=True)
    return np.asarray(wave, dtype=float), np.asarray(flux, dtype=float)[0]


def synthetic_mags(wave, flux):
    """AB magnitudes in the LVM photometric system within each channel, as in the flux calibration (see :func:`~lvmdrp.core.fluxcal.spec_to_LVM_mAB`)"""
    mags = {}
    for channel in "brz":
        good = np.isfinite(flux) & (wave >= SPEC_CHANNELS[channel][0]) & (wave <= SPEC_CHANNELS[channel][1])
        mag = fluxcal.spec_to_LVM_mAB(channel, wave[good], flux[good]) if good.sum() > 10 else np.nan
        mags[channel] = np.nan if (mag is None or mag < -999) else mag
    return mags


def read_cframe(cframe_path, rows):
    """lvmCFrame restricted to the given fibers, with all its sensitivity curves"""
    with fits.open(cframe_path, memmap=True) as hdus:
        header = lvmCFrame.header_from_hdulist(hdus)
        ivar = np.asarray(hdus["IVAR"].data[rows], dtype=np.float32)
        with np.errstate(divide="ignore"):
            error = np.where(ivar > 0, 1 / np.sqrt(ivar), np.nan).astype(np.float32)
        return lvmCFrame(data=np.asarray(hdus["FLUX"].data[rows], dtype=np.float32), error=error,
                         mask=np.asarray(hdus["MASK"].data[rows], dtype=bool), header=header.copy(),
                         wave=np.asarray(hdus["WAVE"].data), lsf=np.asarray(hdus["LSF"].data[rows]),
                         fluxcal_std=Table.read(hdus["FLUXCAL_STD"]), fluxcal_sci=Table.read(hdus["FLUXCAL_SCI"]),
                         fluxcal_mod=Table.read(hdus["FLUXCAL_MOD"]), slitmap=Table.read(hdus["SLITMAP"])[rows])


def calibrate_with_methods(cframe, methods=None):
    """Spectra flux calibrated with each method, swapping the applied calibration (see :meth:`~lvmdrp.core.rss.RSS.swap_fluxcal`)

    Parameters
    ----------
    cframe : lvmdrp.core.rss.lvmCFrame
        flux-calibrated frame
    methods : list[str], optional
        flux calibration methods, by default 'MOD', 'STD' and 'SCI'

    Returns
    -------
    tuple[dict, dict]
        spectra calibrated with each method, and the reason each missing method could not be applied
    """
    methods = methods or FLUXCAL_METHODS
    applied = cframe._header.get("FLUXCAL")
    spectra, unavailable = {}, {}
    for method in methods:
        sens = cframe.get_fluxcal(source=method.lower())
        if sens is None or not np.isfinite(np.asarray(sens["mean"], dtype=float)).any():
            unavailable[method] = "the lvmCFrame has no sensitivity curve for it"
            continue
        if method == applied:
            spectra[method] = np.asarray(cframe._data, dtype=float)
            continue
        if applied is None:
            unavailable[method] = "the flux calibration method applied is not in the header"
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spectra[method] = np.asarray(cframe.swap_fluxcal(method, inplace=False)._data, dtype=float)
    return spectra, unavailable


def _bin_spectrum(wave, flux, width):
    edges = np.arange(wave.min(), wave.max() + width, width)
    good = np.isfinite(flux)
    if good.sum() == 0:
        return 0.5 * (edges[1:] + edges[:-1]), np.full(edges.size - 1, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        binned = binned_statistic(wave[good], flux[good], statistic="mean", bins=edges).statistic
    return 0.5 * (edges[1:] + edges[:-1]), binned


def measure_sky_residuals(wave, flux, sky, error, select, cwaves):
    """Sky line residuals after sky subtraction, relative to the sky line flux

    Each line is integrated in the sky-subtracted spectra and in the sky
    model with the sideband continuum subtraction of the flat-field QA (see
    :func:`~lvmdrp.qa.fiberflats.measure_skylines` with ``method='integrate'``).

    Parameters
    ----------
    wave : numpy.ndarray
        common wavelength grid (columns)
    flux, sky, error : numpy.ndarray
        sky-subtracted spectra, sky model and errors (fibers x columns)
    select : numpy.ndarray
        fibers measured
    cwaves : list[float]
        sky lines in Angstroms

    Returns
    -------
    dict
        residual over sky line flux of each fiber and line ('fraction'), and
        for each line the residual profile relative to the sky line peak
        ('profiles', fibers x offsets) and the offsets ('offsets', Angstroms)
    """
    pars = _defaults(measure_skylines, "line_hw", "cont_gap", "cont_width")
    fraction = np.full((flux.shape[0], len(cwaves)), np.nan)
    profiles, offsets = [], []
    wave2d = np.broadcast_to(wave, flux.shape)
    for iline, cwave in enumerate(cwaves):
        i0, i1 = np.searchsorted(wave, [cwave - pars["cont_gap"] - pars["cont_width"] - 1, cwave + pars["cont_gap"] + pars["cont_width"] + 1])
        if i1 - i0 < 3:
            profiles.append(None)
            offsets.append(None)
            continue
        w, f, s, e = wave2d[:, i0:i1], flux[:, i0:i1], sky[:, i0:i1], error[:, i0:i1]
        mask = ~np.isfinite(f) | ~np.isfinite(s)
        res_flux, _, bad_res = _integrate_line(w, f, e, mask, cwave, pars["line_hw"], pars["cont_gap"], pars["cont_width"])
        sky_flux, _, bad_sky = _integrate_line(w, s, e, mask, cwave, pars["line_hw"], pars["cont_gap"], pars["cont_width"])
        good = select & ~bad_res & ~bad_sky & (sky_flux > 0)
        fraction[good, iline] = res_flux[good] / sky_flux[good]

        # profiles: continuum subtracted residual relative to the continuum subtracted sky peak
        core = np.abs(wave[i0:i1] - cwave) <= pars["line_hw"]
        side = ~core & (np.abs(wave[i0:i1] - cwave) >= pars["cont_gap"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            res = f[:, core] - np.nanmedian(np.where(side, f, np.nan), axis=1)[:, None]
            sky_line = s[:, core] - np.nanmedian(np.where(side, s, np.nan), axis=1)[:, None]
            peak = np.nanmax(sky_line, axis=1)
            profile = np.where((good & (peak > 0))[:, None], res / peak[:, None], np.nan)
        profiles.append(profile)
        offsets.append(wave[i0:i1][core] - cwave)
    return {"fraction": fraction, "profiles": profiles, "offsets": offsets}


def _fiber_bins(values, fiber_bin=FIBER_BIN):
    edges = np.arange(0, values.shape[0] + fiber_bin, fiber_bin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        binned = np.stack([np.nanmedian(values[lo:hi], axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    return 0.5 * (edges[1:] + edges[:-1]) - 0.5, binned


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _layout(fig, height, xtitle, ytitle, **kwargs):
    fig.update_layout(**base_layout(height, **kwargs))
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text=ytitle)
    return style_axes(fig)


def _overlay_traces(traces, x_range, y_range):
    """Fiber positions and sky lines drawn on the frame images, clipped to the view"""
    out = []
    fiberid = np.arange(1, traces["y_master"].size + 1)

    def clip(x, y):
        inside = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
        return np.where(inside, x, np.nan), np.where(inside, y, np.nan)

    ref = traces["ref_column"]
    if x_range[0] <= ref <= x_range[1]:
        for key, name, color, symbol in (("y_master", "fibers, master trace", SERIES[0], "circle-open"),
                                         ("y_frame", "fibers, thermal shift corrected", SERIES[1], "x-thin-open")):
            if key not in traces:
                continue
            x, y = clip(np.full(traces[key].size, float(ref)), traces[key])
            out.append(go.Scatter(x=_r(x, 1), y=_r(y, 3), mode="markers", name=name, legendgroup=key,
                                  marker=dict(color=color, symbol=symbol, size=7, line=dict(width=1.5, color=color)),
                                  customdata=fiberid, hovertemplate=f"fiber %{{customdata}}<br>row %{{y:.3f}} @ column {ref}<extra>{name}</extra>"))
    for iline, line in enumerate(traces["lines"]):
        for key, name, color, dash in (("master", "master wavelengths", SERIES[0], "dash"), ("frame", "sky line corrected", SERIES[1], "solid")):
            if f"x_{key}" not in line:
                continue
            x, y = clip(line[f"x_{key}"], line[f"y_{key}"])
            out.append(go.Scatter(x=_r(x, 2), y=_r(y, 2), mode="lines", name=f"sky lines, {name}", legendgroup=f"line_{key}",
                                  showlegend=iline == 0, line=dict(color=color, width=1.5, dash=dash), customdata=fiberid,
                                  hovertemplate=(f"{line['wave']:.2f} &#8491; ({escape(line['use'])})<br>fiber %{{customdata}}<br>"
                                                 "column %{x:.2f}, row %{y:.1f}<extra>" + name + "</extra>")))
    return out


def fig_frame(data, lo, hi, traces, view, unit="electron"):
    """Detrended frame of a camera with the fiber positions and the reference sky lines

    Parameters
    ----------
    data : numpy.ndarray
        detrended frame
    lo, hi : float
        limits of the asinh stretch
    traces : dict
        output of :func:`measure_camera_traces`
    view : str
        'full' for the binned full frame, 'ref' for the reference column at
        full resolution, or 'lineN' for the N-th reference sky line at full resolution

    Returns
    -------
    plotly.graph_objects.Figure or None
        None if the view does not exist for the camera
    """
    nrows, ncols = data.shape
    if view == "full":
        binning, c0, c1, title = FULL_FRAME_BIN, 0, ncols, f"{traces['camera']} full frame, binned {FULL_FRAME_BIN}&#215;{FULL_FRAME_BIN}"
    else:
        binning = 1
        if view == "ref":
            center, title = traces["ref_column"], f"{traces['camera']} around the reference column {traces['ref_column']}"
        else:
            iline = int(view[4:]) - 1
            if iline >= len(traces["lines"]):
                return None
            line = traces["lines"][iline]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                center = np.nanmedian(np.concatenate([line.get("x_frame", line["x_master"]), line["x_master"]]))
            if not np.isfinite(center):
                return None
            title = f"{traces['camera']} around the sky line {line['wave']:.2f} &#8491; ({escape(line['use'])})"
        center = int(round(center))
        c0, c1 = max(center - CUTOUT_HALF_WIDTH, 0), min(center + CUTOUT_HALF_WIDTH + 1, ncols)
    cut = data[:, c0:c1]
    image = _bin_image(cut, binning) if binning > 1 else cut
    x_range = (c0 - 0.5, c0 + image.shape[1] * binning - 0.5)
    y_range = (-0.5, image.shape[0] * binning - 0.5)

    fig = go.Figure(_overlay_traces(traces, x_range, y_range))
    fig.add_layout_image(dict(source=encode_image(image, lo, hi), xref="x", yref="y", x=x_range[0], y=y_range[1],
                              sizex=x_range[1] - x_range[0], sizey=y_range[1] - y_range[0], sizing="stretch",
                              xanchor="left", yanchor="top", layer="below"))
    fig = _layout(fig, 760, "Column (pix)", "Row (pix)", margin=dict(l=64, r=16, t=48, b=96),
                  legend=dict(orientation="h", x=0, y=-0.1, xanchor="left", yanchor="top", font=dict(color=THEME["ink2"]), bgcolor="rgba(0,0,0,0)"),
                  title=dict(text=title, font=dict(color=THEME["ink"], size=14), x=0, xanchor="left"))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", annotations=[dict(
        text=f"asinh stretch {lo:.0f} to {hi:.0f} {escape(unit)}", xref="paper", yref="paper", x=1, y=1.0, xanchor="right",
        yanchor="bottom", showarrow=False, font=dict(color=THEME["muted"], size=12))])
    fig.update_xaxes(range=list(x_range), showgrid=False, zeroline=False)
    fig.update_yaxes(range=list(y_range), showgrid=False, zeroline=False)
    if view == "full":
        fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
        fig.update_xaxes(constrain="domain")
    return fig


def fig_straylight_map(stray):
    """Median stray light residual per strip and column bin"""
    z = stray["residual"]
    finite = np.isfinite(z)
    zmax = float(np.nanpercentile(np.abs(z[finite]), 98)) if finite.any() else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = 100 * z / stray["stray"]
    fig = go.Figure(go.Heatmap(
        x=_r(stray["columns"], 1), y=_r(stray["rows"], 1), z=_r(z, 2), zmid=0, zmin=-zmax, zmax=zmax, colorscale=DIVERGING, meta="div",
        customdata=np.dstack([_r(stray["stray"], 2), _r(relative, 2), stray["npix"]]),
        colorbar=dict(title=dict(text="e&#8315;", font=dict(color=THEME["ink2"])), tickfont=dict(color=THEME["muted"]), outlinewidth=0),
        hovertemplate=("column %{x:.0f}, strip at row %{y:.0f}<br>residual %{z:+.2f} e&#8315; (%{customdata[1]:+.1f}% of the stray light)"
                       "<br>stray light %{customdata[0]:.1f} e&#8315;, %{customdata[2]} pixels<extra></extra>")))
    fig = _layout(fig, 420, "Column (pix)", "Strip mean row (pix)", margin=dict(l=64, r=16, t=24, b=52))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def fig_straylight_profiles(stray):
    """Median stray light residual along each strip, colored from the bottom to the top of the frame"""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    nstrips = len(stray["rows"])
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    for istrip in range(nstrips):
        if not np.isfinite(stray["rows"][istrip]):
            continue
        color = ramp[min(int(4 * istrip / max(nstrips, 1)), 3)]
        fig.add_trace(go.Scatter(x=_r(stray["columns"], 1), y=_r(stray["residual"][istrip], 2), mode="lines",
                                 line=dict(color=color, width=1.5), name=f"row {stray['rows'][istrip]:.0f}",
                                 hovertemplate=f"strip at row {stray['rows'][istrip]:.0f}<br>column %{{x:.0f}}<br>%{{y:+.2f}} e&#8315;<extra></extra>"))
    fig = _layout(fig, 360, "Column (pix)", "Median residual (e&#8315;)", showlegend=False, margin=dict(l=64, r=16, t=24, b=52))
    return fig


def fig_fiber_shifts(traces):
    """Thermal fiber shifts at the reference column, per camera"""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    for camera, camera_traces in traces.items():
        if "shift" not in camera_traces:
            continue
        specid = int(camera[1])
        fiber = np.arange(camera_traces["shift"].size) + 1 + (specid - 1) * camera_traces["shift"].size
        fig.add_trace(go.Scatter(x=fiber, y=_r(camera_traces["shift"], 4), mode="lines", name=camera, legendgroup=camera[0],
                                 line=dict(color=CHANNEL_COLORS[camera[0]], width=1.5, dash=SPEC_DASHES[specid]),
                                 hovertemplate=f"{camera}<br>fiber %{{x}}<br>%{{y:+.3f}} pix<extra></extra>"))
    for iblock in range(1, 3 * 18):
        fig.add_vline(x=iblock * LVM_BLOCKSIZE + 0.5, line=dict(color=THEME["grid"], width=1))
    return _layout(fig, 380, "Fiber ID", "Corrected &#8722; master row (pix)")


def fig_sky_offsets(traces):
    """Wavelength offsets applied from the reference sky lines, per camera"""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1))
    for camera, camera_traces in traces.items():
        if "wave_offset" not in camera_traces:
            continue
        specid = int(camera[1])
        offset = 1000 * camera_traces["wave_offset"]
        fiber = np.arange(offset.size) + 1 + (specid - 1) * offset.size
        fig.add_trace(go.Scatter(x=fiber, y=_r(offset, 2), mode="lines", name=camera, legendgroup=camera[0],
                                 line=dict(color=CHANNEL_COLORS[camera[0]], width=1.5, dash=SPEC_DASHES[specid]),
                                 hovertemplate=f"{camera}<br>fiber %{{x}}<br>%{{y:+.1f}} m&#8491;<extra></extra>"))
    return _layout(fig, 380, "Fiber ID", "Master &#8722; corrected wavelength (m&#8491;)")


def fig_ifu(geometry, stages, scale, title):
    """IFU view of normalized fiber values at each stage, side by side"""
    names = [label for key, label in IFU_STAGES if key in stages]
    fig = make_subplots(rows=1, cols=max(len(names), 1), subplot_titles=names, horizontal_spacing=0.03)
    size = max(3.0, 2 * geometry["hex_radius"] * 300 / 2.1)
    icol = 0
    for key, label in IFU_STAGES:
        if key not in stages:
            continue
        icol += 1
        values = stages[key][geometry["select"]]
        fig.add_trace(go.Scatter(
            x=geometry["x"], y=geometry["y"], mode="markers", name=label, showlegend=False,
            marker=dict(symbol="hexagon", size=size, color=_r(values, 4), cmin=1 - scale, cmax=1 + scale, colorscale=DIVERGING,
                        line=dict(width=0), showscale=icol == 1,
                        colorbar=dict(title=dict(text="normalized", font=dict(color=THEME["ink2"])), tickfont=dict(color=THEME["muted"]),
                                      outlinewidth=0, len=0.9)),
            customdata=np.column_stack([geometry["fiberid"], geometry["label"]]),
            hovertemplate=f"{label}<br>fiber %{{customdata[0]}} (%{{customdata[1]}})<br>%{{marker.color:.4f}}<extra></extra>"),
            row=1, col=icol)
        fig.update_xaxes(range=[-1.08, 1.08], visible=False, row=1, col=icol)
        fig.update_yaxes(range=[-1.08, 1.08], visible=False, scaleanchor=f"x{icol if icol > 1 else ''}", scaleratio=1, row=1, col=icol)
    fig.update_layout(**base_layout(400, margin=dict(l=16, r=16, t=56, b=16)),
                      title=dict(text=title, font=dict(color=THEME["ink"], size=14), x=0, xanchor="left"))
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(color=THEME["ink2"], size=13))
    return fig


def fig_std_spectrum(star, wave, spectra, applied, xp):
    """Standard star spectrum calibrated with each flux calibration method, with its Gaia XP spectrum"""
    fig = go.Figure()
    for method, flux in spectra.items():
        bwave, bflux = _bin_spectrum(wave, flux, STD_BIN)
        name = f"{method} (applied)" if method == applied else method
        fig.add_trace(go.Scatter(x=_r(bwave, 1), y=_r(1e15 * bflux, 4), mode="lines", name=name,
                                 line=dict(color=METHOD_COLORS[method], width=1),
                                 hovertemplate=f"%{{x:.1f}} &#8491;<br>%{{y:.3f}}<extra>{name}</extra>"))
    if xp is not None:
        xwave, xflux = xp
        inside = (xwave >= wave.min() - 100) & (xwave <= wave.max() + 100)
        fig.add_trace(go.Scatter(x=_r(xwave[inside], 1), y=_r(1e15 * xflux[inside], 4), mode="lines+markers", name="Gaia XP",
                                 line=dict(color=THEME["ink"], width=2), marker=dict(size=4),
                                 hovertemplate="%{x:.0f} &#8491;<br>%{y:.3f}<extra>Gaia XP</extra>"))
    fig = _layout(fig, 460, "Wavelength (&#8491;)", "Flux (10&#8315;&#185;&#8309; erg/s/cm&#178;/&#8491;)",
                  margin=dict(l=72, r=16, t=72, b=52),
                  title=dict(text=f"STD{star['index']} &#183; fiber {escape(str(star['fiber']))} &#183; Gaia DR3 {star['gaia_id']}",
                             font=dict(color=THEME["ink"], size=14), x=0, xanchor="left", y=0.98, yanchor="top"))
    return fig


def fig_std_photometry(rows, methods, applied):
    """Synthetic LVM-band magnitudes of the standard stars minus those of their Gaia XP spectra, per calibration method"""
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=[f"{channel} band" for channel in "brz"],
                        horizontal_spacing=0.03)
    x = [f"STD{row['index']}" for row in rows]
    for icol, channel in enumerate("brz", start=1):
        fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1), row=1, col=icol)
        for method in methods:
            name = f"{method} (applied)" if method == applied else method
            y = [row["lvm"][method][channel] - row["gaia"][channel] for row in rows]
            fig.add_trace(go.Scatter(x=x, y=_r(y, 3), mode="markers", name=name, legendgroup=method, showlegend=icol == 1,
                                     marker=dict(color=METHOD_COLORS[method], size=9, line=dict(color=THEME["surface"], width=1.5)),
                                     customdata=[[row["gaia"][channel], row["lvm"][method][channel], row["fiber"]] for row in rows],
                                     hovertemplate=(f"%{{x}} (%{{customdata[2]}}), {channel} band, {name}<br>Gaia %{{customdata[0]:.3f}}, "
                                                    "LVM %{customdata[1]:.3f}<br>LVM &#8722; Gaia %{y:+.3f} mag<extra></extra>")),
                          row=1, col=icol)
        fig.update_xaxes(type="category", tickangle=-45, row=1, col=icol)
    fig = _layout(fig, 420, None, None, margin=dict(l=64, r=16, t=88, b=64),
                  legend=dict(orientation="h", x=0, y=1.14, xanchor="left", yanchor="bottom", font=dict(color=THEME["ink2"]),
                              bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(title_text="LVM &#8722; Gaia (mag)", row=1, col=1)
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(color=THEME["ink2"], size=13))
    return fig


def fig_sensitivity(wave, curves, applied, reference="MOD"):
    """Mean sensitivity curve of each flux calibration method and, when the reference method has one, the other curves divided by it

    Parameters
    ----------
    wave : numpy.ndarray
        wavelengths in Angstroms
    curves : dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        mean and rms sensitivity curve of each method
    applied : str
        method applied by the reduction
    reference : str, optional
        method the others are divided by, by default 'MOD'

    Returns
    -------
    plotly.graph_objects.Figure
    """
    compared = [method for method in curves if method != reference] if reference in curves else []
    nrows = 2 if compared else 1
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4] if compared else [1.0], vertical_spacing=0.05)
    step = max(1, wave.size // 3000)
    x = _r(wave[::step], 1)
    for method, (mean, rms) in curves.items():
        name = f"{method} (applied)" if method == applied else method
        color = METHOD_COLORS[method]
        fig.add_traces([go.Scatter(x=x, y=(mean + rms)[::step].astype(np.float32), mode="lines", line=dict(width=0),
                                   hoverinfo="skip", showlegend=False, legendgroup=method),
                        go.Scatter(x=x, y=np.clip(mean - rms, 0, None)[::step].astype(np.float32), mode="lines",
                                   line=dict(width=0), fill="tonexty", fillcolor=_alpha(color, 0.18), hoverinfo="skip",
                                   showlegend=False, legendgroup=method)], rows=[1, 1], cols=[1, 1])
        fig.add_trace(go.Scatter(x=x, y=mean[::step].astype(np.float32), mode="lines", name=name, legendgroup=method,
                                 line=dict(color=color, width=1.5),
                                 hovertemplate=f"%{{x:.1f}} &#8491;<br>%{{y:.3e}}<extra>{name}</extra>"), row=1, col=1)
    if compared:
        fig.add_hline(y=1, line=dict(color=THEME["axis"], width=1), row=2, col=1)
        for method in compared:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = curves[method][0] / curves[reference][0]
            fig.add_trace(go.Scatter(x=x, y=_r(ratio[::step], 4), mode="lines", name=f"{method} / {reference}", legendgroup=method,
                                     showlegend=False, line=dict(color=METHOD_COLORS[method], width=1.5),
                                     hovertemplate=f"%{{x:.1f}} &#8491;<br>%{{y:.4f}}<extra>{method} / {reference}</extra>"), row=2, col=1)
    fig = _layout(fig, 560 if compared else 380, None, None, margin=dict(l=80, r=16, t=40, b=52))
    if compared:
        fig.update_yaxes(title_text=f"Divided by {reference}", row=2, col=1)
    fig.update_yaxes(type="log", title_text="Sensitivity (erg/cm&#178;/e&#8315;)", row=1, col=1)
    fig.update_xaxes(title_text="Wavelength (&#8491;)", row=nrows, col=1)
    return fig


def _alpha(color, alpha):
    color = color.lstrip("#")
    return f"rgba({int(color[0:2], 16)},{int(color[2:4], 16)},{int(color[4:6], 16)},{alpha})"


def fig_sky_map(fiber_centers, binned, cwaves):
    """Integrated sky line residuals over the sky line flux, per line and fiber bin"""
    z = 100 * binned
    finite = np.isfinite(z)
    zmax = max(float(np.nanpercentile(np.abs(z[finite]), 98)), 0.5) if finite.any() else 1.0
    labels = [f"{cwave:.2f}" for cwave in cwaves]
    fig = go.Figure(go.Heatmap(
        x=labels, y=_r(fiber_centers + 1, 1), z=_r(z, 2), zmid=0, zmin=-zmax, zmax=zmax, colorscale=DIVERGING, meta="div",
        colorbar=dict(title=dict(text="%", font=dict(color=THEME["ink2"])), tickfont=dict(color=THEME["muted"]), outlinewidth=0),
        hovertemplate=f"fibers %{{y:.0f}} &#177; {FIBER_BIN // 2}<br>line %{{x}} &#8491;<br>%{{z:+.2f}}% of the sky line<extra></extra>"))
    fig = _layout(fig, 480, "Sky line (&#8491;)", "Fiber ID", margin=dict(l=64, r=16, t=24, b=72))
    fig.update_xaxes(type="category", tickangle=-45, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def fig_sky_profile(fiber_centers, binned, offsets, cwave):
    """Sky subtraction residual profile of a line relative to the sky line peak, per fiber bin"""
    z = 100 * binned
    finite = np.isfinite(z)
    zmax = max(float(np.nanpercentile(np.abs(z[finite]), 98)), 0.5) if finite.any() else 1.0
    fig = go.Figure(go.Heatmap(
        x=_r(offsets, 2), y=_r(fiber_centers + 1, 1), z=_r(z, 2), zmid=0, zmin=-zmax, zmax=zmax, colorscale=DIVERGING, meta="div",
        colorbar=dict(title=dict(text="% of peak", font=dict(color=THEME["ink2"])), tickfont=dict(color=THEME["muted"]), outlinewidth=0),
        hovertemplate=f"fibers %{{y:.0f}} &#177; {FIBER_BIN // 2}<br>&#916;&#955; %{{x:+.2f}} &#8491;<br>%{{z:+.2f}}% of the sky line peak<extra></extra>"))
    fig = _layout(fig, 480, f"Wavelength &#8722; {cwave:.2f} &#8491;", "Fiber ID", margin=dict(l=64, r=16, t=24, b=52))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def _header_float(header, key):
    try:
        value = float(header[key])
    except (KeyError, TypeError, ValueError):
        return np.nan
    return value if value > -999 else np.nan


def qa_science_exposure(expnum, cameras=CAMERAS, parts=QA_PARTS, sky_lines=None, output_dir=None, remeasure=False,
                        fetch_xp=False, use_nightly_cals=False, use_untagged_cals=False, drpver=drpver, dry_run=False):
    """Write the QA dashboard of the reduction of a science exposure

    The dashboard follows the steps of :func:`lvmdrp.main.science_reduction`,
    see :mod:`lvmdrp.qa.science`. Parts whose products are missing are logged
    and skipped, and listed in the dashboard.

    Parameters
    ----------
    expnum : int
        exposure number
    cameras : list[str], optional
        cameras of the detrended frames and stray light parts, by default all
    parts : list[str], optional
        parts of the dashboard, any of :data:`QA_PARTS`, by default all
    sky_lines : list[float], optional
        sky lines of the sky subtraction residual maps, by default the isolated
        sky lines of the flat-field QA
    output_dir : str, optional
        directory of the dashboard and of the ancillary products produced
        again, by default ``qa/science_qa/{expnum}`` in the night directory of
        the tile (see :func:`~lvmdrp.qa.report.exposure_qa_dir`)
    remeasure : bool, optional
        produce the ancillary products again (detrended frames, stray light
        subtracted frames and rectified frames) with the routines and
        parameters of the reduction, instead of reusing them, by default False
    fetch_xp : bool, optional
        download the Gaia XP spectra of standard stars not in the Gaia cache, by default False
    use_nightly_cals, use_untagged_cals : bool, optional
        calibrations used by the reduction, as in ``drp run``
    drpver : str, optional
        pipeline version of the reductions, by default the current one
    dry_run : bool, optional
        only report the products found and the output paths

    Returns
    -------
    dict
        'report' path (None in a dry run) and 'missing' products
    """
    parts = [part for part in QA_PARTS if part in parts]
    unknown_cameras = set(cameras).difference(CAMERAS)
    if unknown_cameras:
        raise ValueError(f"unknown cameras {sorted(unknown_cameras)}")
    cameras = [camera for camera in CAMERAS if camera in cameras]
    channels = "".join(channel for channel in "brz" if any(camera[0] == channel for camera in cameras))

    mjd = mjd_from_expnum(int(expnum))[0]
    frames = get_frames_metadata(mjd=mjd).query("expnum == @expnum")
    if frames.empty:
        raise ValueError(f"exposure {expnum} not found in the metadata of MJD {mjd}")
    tileid = int(frames["tileid"].iloc[0])
    output_dir = output_dir or exposure_qa_dir(drpver, tileid, mjd, "science", f"{int(expnum):08d}")
    report_path = os.path.join(output_dir, f"science-qa_{int(expnum):08d}.html")
    products = ExposureProducts(expnum, work_dir=os.path.join(output_dir, "products"), remeasure=remeasure,
                                use_nightly_cals=use_nightly_cals, use_untagged_cals=use_untagged_cals, drpver=drpver)

    if dry_run:
        found = {}
        for camera in cameras:
            found[f"d {camera}"] = products.anc("d", camera)
            found[f"l {camera}"] = products.anc("l", camera)
        for channel in channels:
            found[f"Frame-{channel}"] = path.full("lvm_frame", drpver=products.drpver, tileid=tileid, mjd=mjd, expnum=expnum, kind=f"Frame-{channel}")
            found[f"h {channel}"] = products.anc("h", channel)
        found["SFrame"] = path.full("lvm_frame", drpver=products.drpver, tileid=tileid, mjd=mjd, expnum=expnum, kind="SFrame")
        for label, product in found.items():
            log.info(f"{label:>10}: {'found  ' if os.path.isfile(product) else 'missing'} {product}")
        log.info(f"dashboard: {report_path}")
        log.info(f"ancillary products measured again in: {products.work_dir} ({remeasure = })")
        return {"report": None, "missing": [label for label, product in found.items() if not os.path.isfile(product)]}

    figures, sections, tiles, summary = {}, [], [], {}
    frame_cache = {}

    def frame(channel):
        if channel not in frame_cache:
            frame_path = products.final(f"Frame-{channel}")
            frame_cache[channel] = read_frame(frame_path) if frame_path else None
        return frame_cache[channel]

    # fiber positions and reference sky lines, measured by the reduction
    traces = {}
    if {"detrending", "fibershifts", "skyshifts"}.intersection(parts):
        for camera in cameras:
            master_cent, master_wave = products.calib("centroids", camera), products.calib("wave", camera)
            if master_cent is None or master_wave is None:
                continue
            fr = frame(camera[0])
            master_cent_data = fits.getdata(master_cent, ext=0).astype(float)
            frame_cent = frame_wave = None
            if fr is not None:
                shifts = thermal_shifts(fr["slitmap"], Table.read(master_cent, hdu="SLITMAP"), camera)
                frame_cent = master_cent_data + shifts[:, None]
                frame_wave = fr["wave"][_spec_rows(fr["slitmap"], int(camera[1]))]
            traces[camera] = measure_camera_traces(camera, master_cent_data, fits.getdata(master_wave, ext=0).astype(float),
                                                   frame_cent, frame_wave)
            if fr is not None:
                traces[camera]["header_shift"] = _header_float(fr["header"], f"{camera.upper()} FIBER SHIFT")
                traces[camera]["header_skyoff"] = _header_float(fr["header"], f"{camera.upper()} WAVE SKYOFF")

    # 1. detrended frames
    if "detrending" in parts:
        with_frames = []
        for camera in cameras:
            dframe = products.detrended(camera)
            if dframe is None or camera not in traces:
                continue
            data, unit = _read_image(dframe)
            lo, hi = _stretch_limits(data)
            for view in ["full", "ref"] + [f"line{i}" for i in range(1, MAX_LINE_VIEWS + 1)]:
                fig = fig_frame(data, lo, hi, traces[camera], view, unit=unit)
                if fig is not None:
                    figures[f"frame|{camera}|{view}"] = fig
            with_frames.append(camera)
            del data
        if with_frames:
            chart = picker("fig-frames", "frame", [
                ("Camera", [(camera, camera) for camera in with_frames]),
                ("View", [("full", "full frame"), ("ref", f"reference column {LVM_REFERENCE_COLUMN}")]
                 + [(f"line{i}", f"reference sky line {i}") for i in range(1, MAX_LINE_VIEWS + 1)])], 760)
            lines_text = "; ".join(f"{channel}: " + ", ".join(f"{cwave:.2f} &#8491; ({use})" for cwave, use in reference_skylines(channel))
                                   for channel in channels)
            sections.append(f"""<section>
    <h2>1. {PART_TITLES['detrending']}</h2>
    <p>The detrended frame of each camera (bias subtracted, pixel flat fielded, in electrons). Circles mark the fiber
    positions at the reference column in the master trace and crosses the positions after the thermal shift measured at
    extraction. Dashed curves follow the reference sky lines through the fibers with the master wavelength solution and
    solid curves with the wavelengths corrected from the sky lines. The full frame is binned; the other views are at full
    resolution, zoom in to see the fibers. Reference sky lines: {lines_text}.</p>
    {chart}
  </section>""")
            summary["frames"] = len(with_frames)
        tiles.append(("Detrended frames", f"{len(with_frames)} / {len(cameras)}", "cameras shown"))

    # 2. stray light
    if "straylight" in parts:
        rows_html, with_stray = [], []
        for camera in cameras:
            lframe, stray = products.straylight(camera)
            centroids = products.calib("centroids", camera)
            if centroids is None or (lframe is None and stray is None):
                continue
            dframe = products.detrended(camera)
            if lframe is None and dframe is None:
                continue
            try:
                result = measure_straylight(lframe, centroids, dframe=dframe, stray=stray)
            except Exception as error:
                log.error(f"stray light residuals of {camera} failed: {type(error).__name__}: {error}")
                products.missing.append(f"stray light residuals of {camera} failed: {type(error).__name__}: {error}")
                continue
            figures[f"stray|{camera}|map"] = fig_straylight_map(result)
            figures[f"stray|{camera}|profiles"] = fig_straylight_profiles(result)
            level = float(np.nanmedian(result["stray"])) if np.isfinite(result["stray"]).any() else np.nan
            median, sigma = np.nanmedian(result["pixels"]) if result["pixels"].size else np.nan, _robust_sigma(result["pixels"])
            binned_sigma = _robust_sigma(result["residual"])
            summary.setdefault("stray", {})[camera] = {"median": median, "sigma": sigma, "binned": binned_sigma, "level": level}
            rows_html.append([f'<span class="mono">{camera}</span>', _num(level, ".1f"), _num(median, "+.2f"), _num(sigma, ".2f"),
                              _num(binned_sigma, ".2f"), _num(100 * binned_sigma / level if level else np.nan, ".1f")])
            with_stray.append(camera)
        if with_stray:
            table = html_table([("Camera", ""), ("Stray light (e⁻)", "median stray light model at the strips"),
                                ("Residual median (e⁻)", "median of the strip pixels"),
                                ("Pixel scatter (e⁻)", "robust scatter of the strip pixels, includes the noise"),
                                ("Binned scatter (e⁻)", f"robust scatter of the medians in {STRAY_COLUMN_BIN}-column bins"),
                                ("Binned / stray light (%)", "")],
                               rows_html, numeric=[False, True, True, True, True, True])
            chart = picker("fig-stray", "stray", [("Camera", [(camera, camera) for camera in with_stray]),
                                                  ("View", [("map", "map"), ("profiles", "profiles along the strips")])], 420)
            pars = _defaults(image_tasks.subtract_straylight, "margin", "select_nrows")
            sections.append(f"""<section>
    <h2>2. {PART_TITLES['straylight']}</h2>
    <p>The stray light is fitted to the pixels between the fiber blocks and to {pars['select_nrows']} rows beyond the first
    and last fibers, {pars['margin']} pixels away from the fiber centroids. These are the residuals of the stray light subtracted
    frame at those pixels, as medians in {STRAY_COLUMN_BIN}-column bins. A good fit leaves residuals scattered around zero
    without structure along the strips or across the frame.</p>
    {table}
    {chart}
  </section>""")
            worst = max(with_stray, key=lambda camera: abs(summary["stray"][camera]["binned"]))
            tiles.append(("Stray light residuals", f"{summary['stray'][worst]['binned']:.2f} e⁻",
                          f"largest binned scatter, {worst}"))

    # 3. fiber thermal shifts
    if "fibershifts" in parts:
        with_shifts = {camera: t for camera, t in traces.items() if "shift" in t}
        if with_shifts:
            figures["fig-shifts"] = fig_fiber_shifts(with_shifts)
            rows_html = []
            for camera, t in with_shifts.items():
                shift = t["shift"]
                blocks = shift[: (shift.size // LVM_BLOCKSIZE) * LVM_BLOCKSIZE].reshape(-1, LVM_BLOCKSIZE)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    per_block = np.nanmedian(blocks, axis=1)
                rows_html.append([f'<span class="mono">{camera}</span>', _num(np.nanmedian(shift), "+.3f"), _num(t.get("header_shift"), "+.3f"),
                                  _num(np.nanmin(per_block), "+.3f"), _num(np.nanmax(per_block), "+.3f")])
            table = html_table([("Camera", ""), ("Median shift (pix)", "corrected minus master row at the reference column"),
                                ("Header (pix)", "FIBER SHIFT keyword"), ("Min block (pix)", ""), ("Max block (pix)", "")],
                               rows_html, numeric=[False, True, True, True, True])
            sections.append(f"""<section>
    <h2>3. {PART_TITLES['fibershifts']}</h2>
    <p>The fiber traces move along the rows with the temperature of the spectrographs. At extraction the shift of each
    fiber block is measured by cross-correlating the frame with the fiber model at several columns, and applied to the master
    centroids. This is the corrected minus the master centroid at the reference column {LVM_REFERENCE_COLUMN}; the vertical lines
    separate the fiber blocks.</p>
    <div class="chart" data-fig="fig-shifts" role="img" aria-label="Thermal fiber shifts along the slit"></div>
    {table}
  </section>""")
            values = np.concatenate([t["shift"] for t in with_shifts.values()])
            tiles.append(("Fiber shifts", f"{np.nanmedian(values):+.3f} pix", f"median, {np.nanmin(values):+.2f} to {np.nanmax(values):+.2f}"))

    # 4. sky line wavelength correction
    if "skyshifts" in parts:
        with_offsets = {camera: t for camera, t in traces.items() if "wave_offset" in t}
        if with_offsets:
            figures["fig-skyoffsets"] = fig_sky_offsets(with_offsets)
            rows_html = []
            for camera, t in with_offsets.items():
                dx = [np.nanmedian(line["x_frame"] - line["x_master"]) for line in t["lines"] if "x_frame" in line]
                rows_html.append([f'<span class="mono">{camera}</span>', _num(1000 * np.nanmedian(t["wave_offset"]), "+.1f"),
                                  _num(1000 * t.get("header_skyoff", np.nan), "+.1f"), _num(1000 * _robust_sigma(t["wave_offset"]), ".1f"),
                                  ", ".join(_num(value, "+.3f") for value in dx)])
            table = html_table([("Camera", ""), ("Median offset (mÅ)", "master minus corrected wavelength"),
                                ("Header (mÅ)", "WAVE SKYOFF keyword, mean sky line offset"),
                                ("Scatter (mÅ)", "robust scatter along the slit"),
                                ("Sky line shifts (pix)", "median column shift of each reference sky line")],
                               rows_html, numeric=[False, True, True, True, True])
            lines_text = "; ".join(f"{channel}: " + ", ".join(f"{cwave:.2f}" for cwave in REF_SKYLINES[channel]) + " &#8491;" for channel in channels)
            sections.append(f"""<section>
    <h2>4. {PART_TITLES['skyshifts']}</h2>
    <p>After flat fielding, the reference sky lines ({lines_text}) are fitted in each fiber and a smooth curve along
    each spectrograph of their offsets is subtracted from the wavelength solution. This is the master minus the corrected
    wavelength of each fiber.</p>
    <div class="chart" data-fig="fig-skyoffsets" role="img" aria-label="Wavelength offsets from the sky lines along the slit"></div>
    {table}
  </section>""")

    # 5. IFU views
    if "ifu" in parts:
        ifu_rows, ifu_options = [], []
        geometry = None
        for channel in channels:
            fr = frame(channel)
            if fr is None:
                continue
            geometry = geometry or _ifu_geometry(fr["slitmap"])
            cwave, window = SKYLINES_FIBERFLAT[channel], (CONTINUUM_FIBERFLAT[channel] - 25, CONTINUUM_FIBERFLAT[channel] + 25)
            stages = {}
            with np.errstate(invalid="ignore"):
                extracted = _frame_like(fr["data"] * fr["superflat"], fr["error"] * fr["superflat"], fr["mask"], fr["wave"], fr["slitmap"], fr["header"])
            stages["extracted"] = measure_ifu(extracted, cwave, window)
            stages["flatfielded"] = measure_ifu(_frame_like(fr["data"], fr["error"], fr["mask"], fr["wave"], fr["slitmap"], fr["header"]), cwave, window)
            hframe = products.rectified(channel)
            if hframe is not None:
                rect = RSS.from_file(hframe)
                stages["rectified"] = measure_ifu(rect, cwave, window)
                del rect
            for feature, label in (("line", f"sky line {cwave:.2f} &#8491;"), ("cont", f"continuum {window[0]:.0f}&#8211;{window[1]:.0f} &#8491;")):
                values = {stage: v[feature] for stage, v in stages.items()}
                sci = {stage: v[geometry["select"]] for stage, v in values.items()}
                spread = np.concatenate([np.abs(v[np.isfinite(v)] - 1) for v in sci.values()])
                scale = max(round(float(np.percentile(spread, 98)), 3), 0.01) if spread.size else 0.05
                figures[f"ifu|{channel}|{feature}"] = fig_ifu(geometry, values, scale, f"{channel} channel, {label}, science IFU")
                ifu_rows.append([escape(channel), label] + [_num(100 * _robust_sigma(sci[key]), ".2f") if key in sci else "–" for key, _ in IFU_STAGES])
                summary.setdefault("ifu", {})[(channel, feature)] = {key: _robust_sigma(sci[key]) for key in sci}
            ifu_options.append((channel, channel))
        if ifu_options:
            table = html_table([("Channel", ""), ("Feature", "")] + [(f"Scatter {label} (%)", "robust scatter of the normalized science fibers")
                                                                      for _, label in IFU_STAGES],
                               ifu_rows, numeric=[False, False, True, True, True])
            chart = picker("fig-ifu", "ifu", [("Channel", ifu_options), ("Feature", [("line", "sky line"), ("cont", "continuum")])], 400)
            sections.append(f"""<section>
    <h2>5. {PART_TITLES['ifu']}</h2>
    <p>The science IFU in a bright sky line, integrated after subtracting the continuum from sidebands, and in a continuum
    window, normalized by the robust mean of the science fibers. Before flat fielding is the lvmFrame times its fiber flat
    (the <span class="mono">SUPERFLAT</span> extension), the extracted and wavelength calibrated spectra; after flat fielding
    is the lvmFrame; after rectification is the frame resampled to the common wavelength grid, with the sky fibers
    interpolated. The sky is nearly uniform across the IFU, so after flat fielding the sky line should show no fiber,
    block or spectrograph pattern; the continuum includes the target.</p>
    {chart}
    {table}
  </section>""")
            line_ff = [summary["ifu"][(c, "line")].get("flatfielded") for c, _ in ifu_options]
            tiles.append(("Flat-fielded sky line", f"{100 * np.nanmax(line_ff):.2f}%", "largest robust scatter across the science IFU"))

    # 6. flux calibration
    if "fluxcal" in parts:
        cframe_path = products.final("CFrame")
        if cframe_path is not None:
            cache_dir = os.path.join(MASTERS_DIR, "gaia_cache")
            with fits.open(cframe_path, memmap=True) as hdus:
                header = hdus["PRIMARY"].header
                stars = standard_stars(header, Table.read(hdus["SLITMAP"]))
            if stars:
                cframe = read_cframe(cframe_path, [star["row"] for star in stars])
                wave = np.asarray(cframe._wave, dtype=float)
                applied = cframe._header.get("FLUXCAL")
                spectra, unavailable = calibrate_with_methods(cframe)
                for method, reason in unavailable.items():
                    log.warning(f"flux calibration method {method} not shown: {reason}")
                methods = [method for method in FLUXCAL_METHODS if method in spectra]
                curves = {method: (np.asarray(cframe.get_fluxcal(method.lower())["mean"], dtype=float),
                                   np.asarray(cframe.get_fluxcal(method.lower())["rms"], dtype=float)) for method in methods}
                figures["fig-sens"] = fig_sensitivity(wave, curves, applied)

                phot_rows, star_options, rows_html = [], [], []
                for istar, star in enumerate(stars):
                    xp = None
                    try:
                        xp = load_xp_spectrum(star["gaia_id"], cache_dir, fetch=fetch_xp)
                    except Exception as error:
                        log.warning(f"Gaia XP spectrum of {star['gaia_id']} could not be loaded: {type(error).__name__}: {error}")
                    if xp is None:
                        log.warning(f"Gaia XP spectrum of {star['gaia_id']} (STD{star['index']}) not in {cache_dir}, using the Gaia magnitudes in the header")
                        products.missing.append(f"Gaia XP spectrum of {star['gaia_id']} (STD{star['index']}) not cached")
                    star_spectra = {method: spectra[method][istar] for method in methods}
                    figures[f"std|{star['index']}"] = fig_std_spectrum(star, wave, star_spectra, applied, xp)
                    star_options.append((str(star["index"]), f"STD{star['index']} ({star['fiber']})"))
                    lvm = {method: synthetic_mags(wave, flux) for method, flux in star_spectra.items()}
                    gaia = synthetic_mags(*xp) if xp is not None else {c: _header_float(header, f"STD{star['index']}{c.upper()}AB") for c in "brz"}
                    phot_rows.append({"index": star["index"], "fiber": star["fiber"], "lvm": lvm, "gaia": gaia})
                    rows_html.append([f"STD{star['index']}", escape(str(star["fiber"])), str(star["gaia_id"]), _num(star["exptime"], ".1f")]
                                     + [_num(gaia[c], ".3f") for c in "brz"]
                                     + [" / ".join(_num(lvm[method][c] - gaia[c], "+.3f") for c in "brz") for method in methods])
                figures["fig-stdphot"] = fig_std_photometry(phot_rows, methods, applied)
                table = html_table([("Star", ""), ("Fiber", ""), ("Gaia DR3", ""), ("Exp. time (s)", "")]
                                   + [(f"Gaia {c} (mag)", f"AB magnitude of the Gaia XP spectrum in the LVM {c} band") for c in "brz"]
                                   + [(f"{method}{' (applied)' if method == applied else ''} Δb / Δr / Δz", "LVM minus Gaia, in magnitudes")
                                      for method in methods],
                                   rows_html, numeric=[False, False, True, True] + [True] * (3 + len(methods)))
                chart = picker("fig-std", "std", [("Standard star", star_options)], 460)
                missing_methods = "".join(f" {method} is not shown: {escape(reason)}." for method, reason in unavailable.items())
                sections.append(f"""<section>
    <h2>6. {PART_TITLES['fluxcal']}</h2>
    <p>The standard stars observed through the spectrophotometric telescope, from the lvmCFrame, calibrated with each flux
    calibration method: the applied one, <span class="mono">{escape(str(applied))}</span>, and the others swapped in with
    their sensitivity curves in the lvmCFrame. The telluric correction of the MOD method is not swapped. The sky is not subtracted, which only shows at the sky
    lines for these bright stars.{missing_methods} The magnitudes are AB magnitudes in the Gaussian LVM photometric bands,
    synthesized from the LVM and the Gaia XP spectra with the routine of the flux calibration. The XP spectra are shown for
    reference only: their resolution and smoothing are not those of LVM, so they are not divided into the LVM spectra.</p>
    <div class="chart" data-fig="fig-stdphot" role="img" aria-label="Synthetic magnitudes of the standard stars minus Gaia, per calibration method"></div>
    {chart}
    <p>The mean sensitivity curve of each method with its scatter among the standard stars, and the STD and SCI curves
    divided by the MOD curve where there is one.</p>
    <div class="chart" data-fig="fig-sens" role="img" aria-label="Mean sensitivity curves of each flux calibration method, and their ratio to MOD"></div>
    {table}
  </section>""")
                notes = []
                for method in methods:
                    deltas = np.array([[row["lvm"][method][c] - row["gaia"][c] for c in "brz"] for row in phot_rows], dtype=float)
                    summary.setdefault("fluxcal", {})[method] = float(np.nanmedian(deltas)) if np.isfinite(deltas).any() else np.nan
                    notes.append(f"{method} {_num(summary['fluxcal'][method], '+.3f')}")
                if applied in summary.get("fluxcal", {}):
                    tiles.append(("Standard stars", f"{summary['fluxcal'][applied]:+.3f} mag",
                                  f"median LVM − Gaia, {applied} applied; " + ", ".join(notes)))
            else:
                log.warning("no standard stars acquired in this exposure, skipping the flux calibration part")
                products.missing.append("no standard stars acquired in this exposure")

    # 7. sky subtraction
    if "skysub" in parts:
        sframe_path = products.final("SFrame")
        if sframe_path is not None:
            cwaves = sorted(sky_lines) if sky_lines else sorted(cwave for channel in channels for cwave in SKYLINES_FLATFIELD_QA[channel])
            with fits.open(sframe_path, memmap=True) as hdus:
                wave = np.asarray(hdus["WAVE"].data, dtype=float)
                slitmap = Table.read(hdus["SLITMAP"])
                lo, hi = np.searchsorted(wave, [min(cwaves) - 15, max(cwaves) + 15])
                flux = np.asarray(hdus["FLUX"].data[:, lo:hi], dtype=float)
                sky = np.asarray(hdus["SKY"].data[:, lo:hi], dtype=float)
                ivar = np.asarray(hdus["IVAR"].data[:, lo:hi], dtype=float)
                mask = np.asarray(hdus["MASK"].data[:, lo:hi], dtype=bool)
            flux[mask] = np.nan
            with np.errstate(divide="ignore"):
                error = np.where(ivar > 0, 1 / np.sqrt(ivar), np.nan)
            telescope = _decode(slitmap["telescope"])
            select = (telescope != "Spec") & (np.asarray(slitmap["fibstatus"]) == 0)
            result = measure_sky_residuals(wave[lo:hi], flux, sky, error, select, cwaves)
            centers, binned = _fiber_bins(result["fraction"])
            figures["fig-skymap"] = fig_sky_map(centers, binned, cwaves)
            line_options, rows_html = [], []
            sci = select & (telescope == "Sci")
            for iline, cwave in enumerate(cwaves):
                if result["profiles"][iline] is None:
                    continue
                pcenters, pbinned = _fiber_bins(result["profiles"][iline])
                figures[f"skyprof|{cwave:.2f}"] = fig_sky_profile(pcenters, pbinned, result["offsets"][iline], cwave)
                line_options.append((f"{cwave:.2f}", f"{cwave:.2f} Å"))
                fraction = result["fraction"][:, iline]
                rows_html.append([f"{cwave:.2f}"] + [_num(100 * np.nanmedian(fraction[select & (telescope == tel)]), "+.2f") for tel in ("Sci", "SkyE", "SkyW")]
                                 + [_num(100 * _robust_sigma(fraction[sci]), ".2f")])
            table = html_table([("Sky line (Å)", "")] + [(f"{tel} median (%)", "residual over sky line flux") for tel in ("Sci", "SkyE", "SkyW")]
                               + [("Sci scatter (%)", "robust scatter of the science fibers")],
                               rows_html, numeric=[True] * 5)
            chart = picker("fig-skyprof", "skyprof", [("Sky line", line_options)], 480)
            pars = _defaults(measure_skylines, "line_hw", "cont_gap", "cont_width")
            sections.append(f"""<section>
    <h2>7. {PART_TITLES['skysub']}</h2>
    <p>The residuals of the sky subtraction (lvmSFrame) at isolated sky lines, as a fraction of the sky line flux in the sky
    model. Both are integrated within &#177;{pars['line_hw']:g} &#8491; of the line after subtracting the continuum from
    sidebands {pars['cont_gap']:g}&#8211;{pars['cont_gap'] + pars['cont_width']:g} &#8491; away, in all fibers but those of the
    spectrophotometric telescope, and shown as medians in bins of {FIBER_BIN} fibers. The profiles show the residuals across
    each line relative to the sky line peak: an S shape is a wavelength mismatch, a symmetric residual a resolution
    or flux mismatch. The fibers of the sky telescopes see their own patch of sky, so their residuals measure how much that
    patch differs from the sky model of the science field.</p>
    <div class="chart" data-fig="fig-skymap" role="img" aria-label="Sky subtraction residuals per sky line and fiber"></div>
    {chart}
    {table}
  </section>""")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                worst = np.nanmax(np.abs(np.nanmedian(result["fraction"][sci], axis=0))) if sci.any() else np.nan
            tiles.append(("Sky residuals", f"{100 * worst:.2f}%", "largest median over the science fibers, of the sky line flux"))

    _write(report_path, products, frames, cameras, parts, sections, tiles, figures, remeasure)
    log.info(f"science QA dashboard written to {report_path}")
    return {"report": report_path, "missing": products.missing}


def _write(report_path, products, frames, cameras, parts, sections, tiles, figures, remeasure):
    header = {}
    raw = frames.iloc[0].to_dict()
    for key in ("object", "exptime", "ra", "dec"):
        if key in raw:
            header[key] = raw[key]
    if products.missing:
        missing = "".join(f"<li>{escape(item)}</li>" for item in dict.fromkeys(products.missing))
        sections.insert(0, f"""<section>
    <h2>Skipped or incomplete parts</h2>
    <p>Products not found or not produced, so the parts that need them are missing or incomplete. Ancillary products
    removed after the reduction can be measured again with <span class="mono">drp qa-science --remeasure</span>.</p>
    <ul class="issues">{missing}</ul>
  </section>""")
    if not sections:
        sections.append('<section><p class="note">No part of the dashboard could be made, see the log.</p></section>')
    meta = [("Exposure", products.expnum), ("MJD", products.mjd), ("Tile", products.tileid)]
    meta += [(key.capitalize() if key != "exptime" else "Exp. time (s)", value) for key, value in header.items()]
    meta += [("Calibrations", products.cals_mjd), ("Ancillary products", "measured again" if remeasure else "from the reduction"),
             ("Cameras", ",".join(cameras)), ("DRP version", products.drpver),
             ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]
    intro = ("<p>Quality of the reduction of a science exposure, step by step in the order of the reduction: detrending, "
             "stray light subtraction, extraction, wavelength calibration, fiber flat fielding, rectification, flux calibration "
             "and sky subtraction. The measurements stored in the reduced products are reused; parts whose products are "
             "missing are listed below.</p>")
    write_dashboard(report_path, f"Science exposure {products.expnum}", f"LVM DRP · science QA · {products.drpver}",
                    intro, meta, tiles, "\n\n  ".join(sections), figures, "lvmdrp.qa.science.qa_science_exposure")
