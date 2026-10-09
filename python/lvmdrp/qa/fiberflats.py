# encoding: utf-8
"""Fiber flat-field QA.

Two parts, each with its own dashboard and CLI (``drp qa-flatfield`` and
``drp calibrations qa-fiberflats``), styled like the other QA reports
(:mod:`lvmdrp.qa.report`):

- Flat-fielded science frames (lvmFrame): isolated sky lines and sky
  line-free continuum windows are measured in every fiber. The sky is (nearly)
  uniform across the fibers of each telescope, so after a perfect flat fielding
  the sky line fluxes and continuum levels should be the same in all fibers up
  to photon noise, once the sky gradient across each IFU is removed. The
  fiber-to-fiber scatter in excess of the noise measured at different
  wavelengths quantifies the residuals of the flat fielding across the
  wavelength range of each channel. See :func:`measure_flatfield_qa` for one
  frame and :func:`run_flatfield_qa_batch` for many.
- Master fiber flats across calibration epochs: :func:`qa_fiberflat_epochs`
  compares the master fiber flats of all the epochs in the calibration epochs
  file (``$LVMCORE_DIR/calibrations/calibration-epochs.yaml``) against the flat
  of a reference epoch, on a common wavelength grid, and splits the ratios
  into spectrograph offsets, a gradient across each IFU and the fiber-level
  change.
"""

import os
import re
import json
from datetime import datetime, timezone
from glob import glob
from html import escape
from itertools import product
from multiprocessing import Pool
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import numpy as np
import bottleneck as bn
import pandas as pd
import plotly.graph_objects as go
import yaml
from astropy.io import fits
from astropy.stats import biweight_location, biweight_scale
from astropy.table import Table, hstack, vstack
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree
from scipy.special import erf
from tqdm import tqdm

from lvmdrp import log
from lvmdrp.core.plot import plt, save_fig
from lvmdrp.core.rss import RSS, lvmFrame
from lvmdrp.qa.report import (
    THEME, SERIES, DARK_COLORS, SEQUENTIAL, SEQUENTIAL_DARK, MONO_FONT,
    base_layout, style_axes, html_table, tiles_html, meta_html, figures_json, page_head, picker, write_dashboard,
)
from lvmdrp.utils.convert import tileid_grp


# ---------------------------------------------------------------------------
# flat-fielded science frames
# ---------------------------------------------------------------------------

TELESCOPES = ("Sci", "SkyE", "SkyW")

# isolated sky lines (UVES atlas, air) spanning each channel, used to assess the
# quality of the fiber flat fielding as a function of wavelength. Components
# closer than 0.8 Angstroms (e.g., OH lambda-doublets) are unresolved at LVM
# resolution and are listed as their flux-weighted centroid. Other atlas lines
# within +/-8 Angstroms (the fitting window) add up to less than 8% of the
# feature flux. Telluric absorption scales the line flux equally in all fibers,
# so lines within absorption bands are still useful. There are no strong,
# isolated sky lines in the b channel other than [OI] 5577
SKYLINES_FLATFIELD_QA = {
    "b": [5577.35],
    "r": [6300.31, 6363.78, 6533.05, 7316.29, 7401.86],
    "z": [8399.18, 8465.37, 8885.86, 8958.10, 9439.67, 9519.38, 9719.84]
}

# sky line-free continuum windows (Angstroms) spanning each channel, used to assess
# the quality of the fiber flat fielding as a function of wavelength
CONTINUUM_FLATFIELD_QA = {
    "b": [(3890, 3910), (4590, 4610), (5005, 5025), (5355, 5375)],
    "r": [(6005, 6025), (6390, 6410), (6670, 6690), (7015, 7035)],
    "z": [(8155, 8175), (8550, 8570), (9180, 9200)]
}


def _pixel_window_weights(wave: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    """Fraction of each pixel overlapping the wavelength window [wmin, wmax]

    Using fractional pixel weights avoids the fiber-to-fiber flux jitter that
    a hard pixel selection would introduce given that each fiber has a
    slightly different wavelength solution.

    Parameters
    ----------
    wave : np.ndarray
        2D (fibers x pixels) array of pixel central wavelengths
    wmin, wmax : float
        window limits in Angstroms

    Returns
    -------
    np.ndarray
        2D array of weights between 0 and 1
    """
    mid = 0.5 * (wave[:, 1:] + wave[:, :-1])
    left = np.concatenate([wave[:, :1] - (mid[:, :1] - wave[:, :1]), mid], axis=1)
    right = np.concatenate([mid, wave[:, -1:] + (wave[:, -1:] - mid[:, -1:])], axis=1)
    overlap = np.minimum(right, wmax) - np.maximum(left, wmin)
    return np.clip(overlap, 0, None) / (right - left)


def _sideband_level(data, error, mask, weights, min_coverage=0.5):
    """Median level and its error in a sideband, per fiber"""
    select = (weights > min_coverage) & ~mask
    npix = select.sum(axis=1)
    level = bn.nanmedian(np.where(select, data, np.nan), axis=1)
    # error of the median ~ sqrt(pi/2) times the error of the mean
    level_error = np.sqrt(np.pi / 2 * bn.nansum(np.where(select, error**2, 0), axis=1)) / np.where(npix > 0, npix, np.nan)
    level[npix == 0] = np.nan
    return level, level_error


def _get_arrays(frame: RSS):
    data = frame._data.astype(float)
    error = frame._error.astype(float) if frame._error is not None else np.sqrt(np.abs(data))
    mask = frame._mask if frame._mask is not None else np.zeros_like(data, dtype=bool)
    mask = mask | ~np.isfinite(data) | ~np.isfinite(error)
    wave = np.broadcast_to(frame._wave, data.shape)
    return wave, data, error, mask


def _telescopes(frame: RSS) -> np.ndarray:
    """Telescope name of each fiber as unicode strings"""
    telescope = np.asarray(frame._slitmap["telescope"])
    return np.char.decode(telescope) if telescope.dtype.kind == "S" else telescope.astype(str)


def _select_fibers(frame: RSS, telescopes=TELESCOPES) -> np.ndarray:
    slitmap = frame._slitmap
    select = np.isin(_telescopes(frame), telescopes)
    if "fibstatus" in slitmap.colnames:
        select &= slitmap["fibstatus"].data == 0
    return select


def _normalize_by_telescope(frame: RSS, flux: np.ndarray, select: np.ndarray) -> np.ndarray:
    """Normalize fluxes by the robust mean of the fibers in each telescope

    Each telescope points to a different patch of sky, with possibly different
    sky brightness, so the normalization is done per telescope.
    """
    telescope = _telescopes(frame)
    flux_norm = np.full_like(flux, np.nan)
    for tel in np.unique(telescope[select]):
        in_tel = select & (telescope == tel)
        norm = biweight_location(flux[in_tel], ignore_nan=True)
        if np.isfinite(norm) and norm != 0:
            flux_norm[in_tel] = flux[in_tel] / norm
    return flux_norm


def _extract_windows(wave, data, error, mask, cwave, half_width):
    """Extracts a fixed number of pixels around `cwave` in each fiber

    Returns the cutouts of wave, data, error and mask, together with the
    pixel edges, and the index of the pixel closest to `cwave` in each fiber.
    Pixels outside of the detector are masked.
    """
    nfibers, npix = wave.shape
    disp = np.nanmedian(np.abs(np.diff(wave, axis=1)))
    n = int(np.ceil(half_width / disp))
    icen = np.nanargmin(np.abs(wave - cwave), axis=1)
    idx = icen[:, None] + np.arange(-n, n + 1)[None, :]
    inside = (idx >= 0) & (idx < npix)
    idx = np.clip(idx, 0, npix - 1)
    rows = np.arange(nfibers)[:, None]

    w, d, e = wave[rows, idx], data[rows, idx], error[rows, idx]
    m = mask[rows, idx] | ~inside | (np.abs(w - cwave) > half_width)
    mid = 0.5 * (w[:, 1:] + w[:, :-1])
    left = np.concatenate([w[:, :1] - (mid[:, :1] - w[:, :1]), mid], axis=1)
    right = np.concatenate([mid, w[:, -1:] + (w[:, -1:] - mid[:, -1:])], axis=1)
    return w, d, e, m, left, right, icen


def _gaussian_floor_model(params, left, right, wave, cwave, cont_deg):
    """Pixel-integrated Gaussian on top of a polynomial continuum floor and its Jacobian

    params is (nfibers, 3 + cont_deg + 1): flux, center, sigma, c0[, c1]. The
    flux is the total line counts and the floor is in counts per pixel.
    """
    flux, mu, sigma = params[:, 0, None], params[:, 1, None], params[:, 2, None]
    zl, zr = (left - mu) / sigma, (right - mu) / sigma
    cdf_l, cdf_r = 0.5 * (1 + erf(zl / np.sqrt(2))), 0.5 * (1 + erf(zr / np.sqrt(2)))
    pdf_l, pdf_r = np.exp(-0.5 * zl**2) / np.sqrt(2 * np.pi), np.exp(-0.5 * zr**2) / np.sqrt(2 * np.pi)
    profile = cdf_r - cdf_l

    x = wave - cwave
    floor = sum(params[:, 3 + k, None] * x**k for k in range(cont_deg + 1))
    model = flux * profile + floor

    jacobian = [profile, -flux / sigma * (pdf_r - pdf_l), -flux / sigma * (zr * pdf_r - zl * pdf_l)]
    jacobian += [x**k for k in range(cont_deg + 1)]
    return model, np.stack(jacobian, axis=-1)


def _solve_batched(matrix, vector):
    try:
        return np.linalg.solve(matrix, vector[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.einsum("fkl,fl->fk", np.linalg.pinv(matrix), vector)


def fit_gaussian_floor(wave, data, error, mask, cwave, sigma_guess, fit_hw=8.0, cont_deg=1,
                       max_shift=1.5, sigma_bounds=(0.3, 3.0), niter=50, tol=1e-6) -> Dict[str, np.ndarray]:
    """Fits a Gaussian line plus a polynomial continuum floor to all fibers at once

    The fit is a weighted Levenberg-Marquardt least squares, vectorized over
    fibers. The Gaussian is integrated over the pixel edges, so the fitted flux
    is the total line counts regardless of the sampling.

    Parameters
    ----------
    wave, data, error, mask : np.ndarray
        2D (fibers x pixels) arrays
    cwave : float
        expected line center in Angstroms
    sigma_guess : np.ndarray
        per fiber initial guess of the line sigma in Angstroms (e.g., from the LSF)
    fit_hw : float, optional
        half width of the fitting window, by default 8.0 Angstroms
    cont_deg : int, optional
        polynomial degree of the continuum floor (0 or 1), by default 1
    max_shift : float, optional
        maximum offset of the line center from `cwave`, by default 1.5 Angstroms
    sigma_bounds : tuple[float, float], optional
        allowed range of sigma relative to `sigma_guess`, by default (0.3, 3.0)
    niter : int, optional
        maximum number of iterations, by default 50
    tol : float, optional
        relative chi-square change for convergence, by default 1e-6

    Returns
    -------
    dict[str, np.ndarray]
        per fiber arrays: flux, flux_error, center, sigma, cont (floor at the
        line center in counts per pixel), chi2 (reduced), npix (pixels used),
        converged, and disp (pixel size at the line center in Angstroms)
    """
    w, d, e, m, left, right, icen = _extract_windows(wave, data, error, mask, cwave, fit_hw)
    nfibers = w.shape[0]
    weights = np.where(m | ~(e > 0), 0.0, 1.0 / np.where(e > 0, e, 1.0)**2)
    d = np.where(weights > 0, d, 0.0)
    npix = (weights > 0).sum(axis=1)
    nparams = 4 + cont_deg
    sigma_guess = np.broadcast_to(np.asarray(sigma_guess, dtype=float), (nfibers,)).copy()

    # initial guesses: floor from the window wings, center from the peak, flux from the residual sum
    wings = (np.abs(w - cwave) > 3 * sigma_guess[:, None]) & (weights > 0)
    floor0 = bn.nanmedian(np.where(wings, d, np.nan), axis=1)
    floor0 = np.where(np.isfinite(floor0), floor0, 0.0)
    resid = np.where(weights > 0, d - floor0[:, None], np.nan)
    core = np.abs(w - cwave) <= max_shift
    ipeak = np.nanargmax(np.where(core & np.isfinite(resid), resid, -np.inf), axis=1)
    mu0 = w[np.arange(nfibers), ipeak]
    flux0 = bn.nansum(np.where(np.abs(w - mu0[:, None]) <= 3 * sigma_guess[:, None], resid, 0.0), axis=1)

    params = np.zeros((nfibers, nparams))
    params[:, 0], params[:, 1], params[:, 2], params[:, 3] = flux0, mu0, sigma_guess, floor0
    lower = np.full((nfibers, nparams), -np.inf)
    upper = np.full((nfibers, nparams), np.inf)
    lower[:, 1], upper[:, 1] = cwave - max_shift, cwave + max_shift
    lower[:, 2], upper[:, 2] = sigma_bounds[0] * sigma_guess, sigma_bounds[1] * sigma_guess

    fit = npix > nparams
    weights[~fit] = 1.0  # dummy weights to keep the batched algebra well defined

    model, jac = _gaussian_floor_model(params, left, right, w, cwave, cont_deg)
    chi2 = np.sum(weights * (d - model)**2, axis=1)
    lam = np.full(nfibers, 1e-3)
    converged = np.zeros(nfibers, dtype=bool)
    eye = np.eye(nparams)[None]
    for _ in range(niter):
        jw = jac * weights[..., None]
        hess = np.einsum("fpk,fpl->fkl", jw, jac)
        grad = np.einsum("fpk,fp->fk", jw, d - model)
        diag = np.einsum("fkk->fk", hess)
        damped = hess + eye * (lam[:, None] * (diag + 1e-12 * diag.max(axis=1, keepdims=True)))[:, :, None]
        trial = np.clip(params + _solve_batched(damped, grad), lower, upper)

        trial_model, trial_jac = _gaussian_floor_model(trial, left, right, w, cwave, cont_deg)
        trial_chi2 = np.sum(weights * (d - trial_model)**2, axis=1)
        better = (trial_chi2 < chi2) & ~converged
        dchi2 = np.where(better, (chi2 - trial_chi2) / np.maximum(chi2, 1e-30), 0.0)

        params[better], model[better], jac[better], chi2[better] = trial[better], trial_model[better], trial_jac[better], trial_chi2[better]
        lam = np.where(better, lam / 10, lam * 10)
        converged |= (better & (dchi2 < tol)) | (lam > 1e10)
        if converged[fit].all():
            break

    # parameter errors from the undamped curvature matrix at the solution
    jw = jac * weights[..., None]
    hess = np.einsum("fpk,fpl->fkl", jw, jac)
    cov = np.linalg.pinv(hess)
    perr = np.sqrt(np.clip(np.einsum("fkk->fk", cov), 0, None))

    disp = (right - left)[np.arange(nfibers), np.argmin(np.abs(w - params[:, 1, None]), axis=1)]
    result = {
        "flux": params[:, 0], "flux_error": perr[:, 0], "center": params[:, 1], "sigma": params[:, 2],
        "cont": params[:, 3], "chi2": chi2 / np.maximum(npix - nparams, 1), "npix": npix,
        "converged": converged, "disp": disp,
    }
    for key in result:
        if result[key].dtype.kind == "f":
            result[key] = np.where(fit, result[key], np.nan)
    result["converged"] &= fit
    return result


def _lsf_sigma(frame: RSS, cwave: float, default_fwhm: float) -> np.ndarray:
    """Per fiber LSF sigma at `cwave` in Angstroms, from the frame LSF if available"""
    lsf = getattr(frame, "_lsf", None)
    nfibers = frame._data.shape[0]
    if lsf is None:
        return np.full(nfibers, default_fwhm / 2.355)
    wave = np.broadcast_to(frame._wave, frame._data.shape)
    lsf = np.broadcast_to(lsf, frame._data.shape)
    fwhm = lsf[np.arange(nfibers), np.nanargmin(np.abs(wave - cwave), axis=1)].astype(float)
    fwhm = np.where(np.isfinite(fwhm) & (fwhm > 0), fwhm, default_fwhm)
    return fwhm / 2.355


def measure_skylines(frame: RSS, cwaves: List[float], method: str = "fit",
                     fit_hw: float = 8.0, cont_deg: int = 1, max_shift: float = 1.5,
                     default_fwhm: float = 2.0, fwhm_range: Tuple[float, float] = (0.5, 2.0),
                     min_fiber_snr: float = 5.0, min_coverage: float = 0.9,
                     line_hw: float = 4.0, cont_gap: float = 6.0, cont_width: float = 4.0,
                     telescopes: Tuple[str] = TELESCOPES) -> Table:
    """Measures continuum subtracted sky line fluxes in each fiber

    With `method='fit'` (default), a pixel-integrated Gaussian on top of a
    polynomial continuum floor is fitted within `cwave +/- fit_hw` in each
    fiber, with the initial line width taken from the frame LSF. A fiber
    measurement is flagged as good when the fit converged, the line S/N is at
    least `min_fiber_snr`, the fitted FWHM is within `fwhm_range` times the LSF
    FWHM, the center is not at the `max_shift` bounds, and at least
    `min_coverage` of the window pixels are unmasked. Only good fibers are
    normalized.

    With `method='integrate'`, the line flux is integrated within
    `cwave +/- line_hw` using fractional pixel weights, after subtracting a
    linear continuum interpolated between the median levels in two sidebands
    at `cont_gap` to `cont_gap+cont_width` Angstroms on each side of the line.

    Parameters
    ----------
    frame : RSS
        flat fielded frame (e.g., lvmFrame), not sky subtracted
    cwaves : list[float]
        line central wavelengths in Angstroms
    method : str, optional
        'fit' or 'integrate', by default 'fit'
    fit_hw : float, optional
        half width of the fitting window, by default 8.0 Angstroms
    cont_deg : int, optional
        polynomial degree of the continuum floor (0 or 1), by default 1
    max_shift : float, optional
        maximum offset of the line center from `cwave`, by default 1.5 Angstroms
    default_fwhm : float, optional
        line FWHM guess when the frame has no LSF, by default 2.0 Angstroms
    fwhm_range : tuple[float, float], optional
        allowed fitted FWHM relative to the LSF for a good fiber, by default (0.5, 2.0)
    min_fiber_snr : float, optional
        minimum line S/N for a good fiber, by default 5.0
    min_coverage : float, optional
        minimum fraction of unmasked pixels in the fitting window, by default 0.9
    line_hw, cont_gap, cont_width : float, optional
        integration window half width, sidebands gap and width for `method='integrate'`
    telescopes : tuple[str], optional
        telescopes whose fibers are measured, by default ('Sci', 'SkyE', 'SkyW')

    Returns
    -------
    Table
        per fiber table with columns {name}_flux, {name}_error and {name}_norm
        for each line, where name is 'L{cwave:.2f}'. The normalized fluxes are
        relative to the robust mean of the good fibers in the same telescope.
        With `method='fit'`, also {name}_cont (floor at the line center, counts
        per pixel), {name}_center, {name}_fwhm, {name}_snr, {name}_contrast
        (line peak over floor), {name}_chi2 (reduced) and {name}_good
    """
    if method not in ("fit", "integrate"):
        raise ValueError(f"Invalid value for `method`: {method}. Expected either 'fit' or 'integrate'")

    wave, data, error, mask = _get_arrays(frame)
    select = _select_fibers(frame, telescopes)

    table = Table()
    for cwave in np.atleast_1d(cwaves):
        name = f"L{cwave:.2f}"
        if method == "integrate":
            flux, flux_error, bad = _integrate_line(wave, data, error, mask, cwave, line_hw, cont_gap, cont_width)
            bad |= ~select
            flux[bad] = np.nan
            flux_error[bad] = np.nan
            table[f"{name}_flux"] = flux
            table[f"{name}_error"] = flux_error
            table[f"{name}_norm"] = _normalize_by_telescope(frame, flux, ~bad)
            table[f"{name}_flux"].meta = {"kind": "line", "method": method, "wave": cwave, "wmin": cwave - line_hw, "wmax": cwave + line_hw}
            continue

        sigma_lsf = _lsf_sigma(frame, cwave, default_fwhm)
        res = fit_gaussian_floor(wave, data, error, mask, cwave, sigma_lsf, fit_hw=fit_hw, cont_deg=cont_deg, max_shift=max_shift)
        npix_window = int(np.round(2 * fit_hw / np.nanmedian(res["disp"]))) if np.isfinite(res["disp"]).any() else 1

        with np.errstate(invalid="ignore", divide="ignore"):
            snr = res["flux"] / res["flux_error"]
            peak = res["flux"] * res["disp"] / (np.sqrt(2 * np.pi) * res["sigma"])
            contrast = np.where(res["cont"] > 0, peak / res["cont"], np.inf)
            fwhm_ratio = res["sigma"] / sigma_lsf
            at_bounds = np.isclose(np.abs(res["center"] - cwave), max_shift, atol=1e-3)
            good = (select & res["converged"] & (snr >= min_fiber_snr) & ~at_bounds
                    & (fwhm_ratio >= fwhm_range[0]) & (fwhm_ratio <= fwhm_range[1])
                    & (res["npix"] >= min_coverage * npix_window))

        attempted = select & np.isfinite(res["flux"])
        table[f"{name}_flux"] = np.where(attempted, res["flux"], np.nan)
        table[f"{name}_error"] = np.where(attempted, res["flux_error"], np.nan)
        table[f"{name}_norm"] = _normalize_by_telescope(frame, res["flux"], good)
        table[f"{name}_cont"] = np.where(attempted, res["cont"], np.nan)
        table[f"{name}_center"] = np.where(attempted, res["center"], np.nan)
        table[f"{name}_fwhm"] = np.where(attempted, 2.355 * res["sigma"], np.nan)
        table[f"{name}_snr"] = np.where(attempted, snr, np.nan)
        table[f"{name}_contrast"] = np.where(attempted, contrast, np.nan)
        table[f"{name}_chi2"] = np.where(attempted, res["chi2"], np.nan)
        table[f"{name}_good"] = good
        table[f"{name}_flux"].meta = {"kind": "line", "method": method, "wave": cwave, "wmin": cwave - fit_hw, "wmax": cwave + fit_hw}

    return table


def _integrate_line(wave, data, error, mask, cwave, line_hw, cont_gap, cont_width):
    """Sideband continuum subtracted line flux integrated with fractional pixel weights"""
    w_line = _pixel_window_weights(wave, cwave - line_hw, cwave + line_hw)
    w_blue = _pixel_window_weights(wave, cwave - cont_gap - cont_width, cwave - cont_gap)
    w_red = _pixel_window_weights(wave, cwave + cont_gap, cwave + cont_gap + cont_width)

    blue, blue_error = _sideband_level(data, error, mask, w_blue)
    red, red_error = _sideband_level(data, error, mask, w_red)

    # linear continuum between the sideband centers evaluated at each pixel
    wblue, wred = cwave - cont_gap - cont_width / 2, cwave + cont_gap + cont_width / 2
    slope = (red - blue) / (wred - wblue)
    cont = blue[:, None] + slope[:, None] * (wave - wblue)
    npix_line = w_line.sum(axis=1)

    flux = bn.nansum(w_line * (data - cont), axis=1)
    flux_error = np.sqrt(bn.nansum(w_line**2 * error**2, axis=1) + (0.5 * npix_line)**2 * (blue_error**2 + red_error**2))
    bad = (npix_line == 0) | ((w_line > 0) & mask).any(axis=1) | ~np.isfinite(blue) | ~np.isfinite(red)
    return flux, flux_error, bad


def measure_continuum(frame: RSS, windows: List[Tuple[float, float]], min_coverage: float = 0.8,
                      telescopes: Tuple[str] = TELESCOPES) -> Table:
    """Measures the mean continuum level in wavelength windows in each fiber

    Masked pixels are ignored and the mean is computed with fractional pixel
    weights. Fibers with less than `min_coverage` of the window unmasked are
    set to NaN.

    Parameters
    ----------
    frame : RSS
        flat fielded frame (e.g., lvmFrame), not sky subtracted
    windows : list[tuple[float, float]]
        list of (wmin, wmax) windows in Angstroms
    min_coverage : float, optional
        minimum fraction of unmasked pixels in a window, by default 0.8
    telescopes : tuple[str], optional
        telescopes whose fibers are measured, by default ('Sci', 'SkyE', 'SkyW')

    Returns
    -------
    Table
        per fiber table with columns {name}_flux, {name}_error and {name}_norm
        for each window, where name is 'C{wmin:.0f}-{wmax:.0f}'. The flux is
        the mean counts per pixel. The normalized fluxes are relative to the
        robust mean of the fibers in the same telescope
    """
    wave, data, error, mask = _get_arrays(frame)
    select = _select_fibers(frame, telescopes)

    table = Table()
    for wmin, wmax in windows:
        name = f"C{wmin:.0f}-{wmax:.0f}"
        weights = _pixel_window_weights(wave, wmin, wmax)
        npix = weights.sum(axis=1)
        weights = np.where(mask, 0, weights)
        npix_good = weights.sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            flux = bn.nansum(weights * np.where(mask, 0, data), axis=1) / npix_good
            flux_error = np.sqrt(bn.nansum(weights**2 * np.where(mask, 0, error)**2, axis=1)) / npix_good
            coverage = npix_good / npix

        bad = ~select | ~(coverage >= min_coverage)
        flux[bad] = np.nan
        flux_error[bad] = np.nan

        table[f"{name}_flux"] = flux
        table[f"{name}_error"] = flux_error
        table[f"{name}_norm"] = _normalize_by_telescope(frame, flux, ~bad)
        table[f"{name}_flux"].meta = {"kind": "cont", "wave": (wmin + wmax) / 2, "wmin": float(wmin), "wmax": float(wmax)}

    return table


def _ifu_coordinates(frame: RSS) -> Tuple[np.ndarray, np.ndarray]:
    """Fiber positions relative to the center of their IFU, in units of the IFU radius"""
    slitmap = frame._slitmap
    telescope = _telescopes(frame)
    if "xpmm" not in slitmap.colnames or "ypmm" not in slitmap.colnames:
        return np.zeros(len(slitmap)), np.zeros(len(slitmap))
    x, y = np.asarray(slitmap["xpmm"], dtype=float), np.asarray(slitmap["ypmm"], dtype=float)
    xn, yn = np.full_like(x, np.nan), np.full_like(y, np.nan)
    for tel in np.unique(telescope):
        in_tel = telescope == tel
        dx, dy = x[in_tel] - np.nanmean(x[in_tel]), y[in_tel] - np.nanmean(y[in_tel])
        radius = np.nanmax(np.hypot(dx, dy))
        if radius > 0:
            xn[in_tel], yn[in_tel] = dx / radius, dy / radius
    return xn, yn


def fit_sky_gradient(frame: RSS, norm: np.ndarray, deg: int = 1, telescopes: Tuple[str] = TELESCOPES,
                     min_fibers: int = 20, clip: float = 4.0, niter: int = 10) -> Dict:
    """Jointly fits a sky gradient across each IFU and the spectrograph offsets

    The normalized fluxes of the fibers are modeled as

        norm - 1 = a_tel + P_tel(x, y) + o_spec

    where a_tel is an intercept per telescope, P_tel a polynomial of degree
    `deg` (no constant term) in the fiber positions relative to the IFU center
    in units of the IFU radius, and o_spec an offset per spectrograph shared by
    all telescopes. The spectrograph offsets capture flat field errors at the
    spectrograph level, while the polynomial captures the sky gradient. Fitting
    them jointly avoids that a sky gradient across the science IFU, whose fibers
    are split between the spectrographs in three wedges, is mistaken for
    spectrograph offsets. The sky IFUs, which have fibers from all
    spectrographs on a small patch of sky, help to break this degeneracy.

    The fit is a linear least squares with iterative clipping of outliers
    beyond `clip` times the robust scatter (e.g., stars or nebular emission).

    Parameters
    ----------
    frame : RSS
        frame from which the measurements were taken
    norm : np.ndarray
        per fiber normalized fluxes, NaN for fibers not to use
    deg : int, optional
        polynomial degree of the sky gradient: 0 (none), 1 (plane) or 2
        (quadratic), by default 1
    telescopes : tuple[str], optional
        telescopes to fit, by default ('Sci', 'SkyE', 'SkyW')
    min_fibers : int, optional
        minimum number of valid fibers to fit a gradient in a telescope, by default 20
    clip : float, optional
        clipping threshold in units of the robust scatter, by default 4.0
    niter : int, optional
        maximum number of clipping iterations, by default 10

    Returns
    -------
    dict
        sky: per fiber multiplicative sky model, 1 + a_tel + P_tel (NaN for
        fibers outside the fitted telescopes); offsets: dict of spectrograph
        offsets with zero mean over the used fibers; gradients: dict per
        telescope with the linear gradient components at the IFU edge (gx, gy)
        and the half peak-to-valley amplitude of the sky model (amp), all
        relative to the telescope mean; used: boolean array of fibers kept in
        the final fit
    """
    slitmap = frame._slitmap
    telescope = _telescopes(frame)
    specid = np.asarray(slitmap["spectrographid"].data)
    xn, yn = _ifu_coordinates(frame)
    if deg > 0 and ("xpmm" not in slitmap.colnames or "ypmm" not in slitmap.colnames):
        log.warning("no fiber positions (xpmm, ypmm) in the slitmap, fitting spectrograph offsets only")
        deg = 0

    valid = np.isfinite(norm) & np.isfinite(xn) & np.isfinite(yn)
    fit_tels = [tel for tel in telescopes if (valid & (telescope == tel)).sum() >= max(min_fibers, 3)]
    result = {"sky": np.full(norm.shape, np.nan), "offsets": {i: np.nan for i in (1, 2, 3)},
              "gradients": {tel: {"gx": np.nan, "gy": np.nan, "amp": np.nan} for tel in telescopes},
              "used": np.zeros(norm.shape, dtype=bool)}
    if not fit_tels:
        return result
    valid &= np.isin(telescope, fit_tels)

    terms = [(1, 0), (0, 1)] if deg >= 1 else []
    terms += [(2, 0), (1, 1), (0, 2)] if deg >= 2 else []
    columns, labels = [], []
    for tel in fit_tels:
        in_tel = (telescope == tel).astype(float)
        columns.append(in_tel)
        labels.append((tel, None))
        for px, py in terms:
            columns.append(in_tel * np.nan_to_num(xn)**px * np.nan_to_num(yn)**py)
            labels.append((tel, (px, py)))
    specs = [i for i in (1, 2, 3) if (valid & (specid == i)).any()]
    for i in specs[1:]:
        columns.append((specid == i).astype(float))
        labels.append(("spec", i))
    design = np.column_stack(columns)

    target = np.nan_to_num(norm - 1)
    used = valid.copy()
    for _ in range(niter):
        coeffs, *_ = np.linalg.lstsq(design[used], target[used], rcond=None)
        resid = target - design @ coeffs
        scale = 1.4826 * np.nanmedian(np.abs(resid[used] - np.nanmedian(resid[used])))
        new_used = valid & (np.abs(resid) <= clip * scale) if scale > 0 else used
        if np.array_equal(new_used, used):
            break
        used = new_used

    # spectrograph offsets with zero mean over the used fibers
    offsets = {i: 0.0 for i in specs}
    for (kind, value), coeff in zip(labels, coeffs):
        if kind == "spec":
            offsets[value] = coeff
    mean_offset = np.mean([offsets[i] for i in specid[used]]) if used.any() else 0.0
    offsets = {i: offsets[i] - mean_offset for i in specs}

    sky = np.full(norm.shape, np.nan)
    for tel in fit_tels:
        in_tel = telescope == tel
        model = np.full(in_tel.sum(), mean_offset)
        gx = gy = 0.0
        for (kind, value), coeff in zip(labels, coeffs):
            if kind != tel:
                continue
            if value is None:
                model = model + coeff
            else:
                px, py = value
                model = model + coeff * xn[in_tel]**px * yn[in_tel]**py
                if value == (1, 0):
                    gx = coeff
                elif value == (0, 1):
                    gy = coeff
        sky[in_tel] = 1 + model
        level = np.nanmean(sky[in_tel & used]) if (in_tel & used).any() else np.nanmean(sky[in_tel])
        result["gradients"][tel] = {"gx": gx / level, "gy": gy / level,
                                    "amp": 0.5 * (np.nanmax(sky[in_tel]) - np.nanmin(sky[in_tel])) / level}

    result["sky"] = sky
    result["offsets"].update(offsets)
    result["used"] = used
    return result


def correct_sky_gradient(frame: RSS, measurements: Table, deg: int = 1, telescopes: Tuple[str] = TELESCOPES, **fit_kwargs) -> Table:
    """Removes the sky gradient from the normalized fluxes of each feature

    For each feature, `fit_sky_gradient` is run on the normalized fluxes and
    the columns {name}_sky (sky model) and {name}_corr (normalized flux divided
    by the sky model) are added. The corrected fluxes keep the spectrograph
    offsets and the fiber-to-fiber flat field errors. The fitted spectrograph
    offsets and gradients are stored in the metadata of the {name}_flux column.

    Parameters
    ----------
    frame : RSS
        frame from which the measurements were taken
    measurements : Table
        per fiber table as returned by `measure_skylines` or `measure_continuum`
    deg : int, optional
        polynomial degree of the sky gradient, by default 1
    telescopes : tuple[str], optional
        telescopes to fit, by default ('Sci', 'SkyE', 'SkyW')
    **fit_kwargs
        additional keyword arguments passed to `fit_sky_gradient`

    Returns
    -------
    Table
        the input table with the new columns
    """
    names = [c[:-len("_flux")] for c in measurements.colnames if c.endswith("_flux")]
    for name in names:
        norm = np.asarray(measurements[f"{name}_norm"], dtype=float)
        fit = fit_sky_gradient(frame, norm, deg=deg, telescopes=telescopes, **fit_kwargs)
        with np.errstate(invalid="ignore", divide="ignore"):
            measurements[f"{name}_sky"] = fit["sky"]
            measurements[f"{name}_corr"] = norm / fit["sky"]
        measurements[f"{name}_flux"].meta.update({"gradient_deg": deg, "offsets": fit["offsets"], "gradients": fit["gradients"]})
    return measurements


def summarize_flatfield_qa(frame: RSS, measurements: Table, max_deviation: float = 0.02, min_snr: float = 10.0,
                           min_good_fraction: float = 0.8, min_contrast: float = 1.0, verbose: bool = True) -> Table:
    """Summarizes the fiber-to-fiber scatter of each feature

    A feature is reliable, i.e. useful to assess the flat fielding, when its
    median S/N is at least `min_snr`. For sky lines fitted with a Gaussian plus
    continuum floor (see `measure_skylines`), the median S/N is computed over
    all fitted fibers, and in addition at least `min_good_fraction` of them
    must have good fits and the median line peak must be at least
    `min_contrast` times the underlying continuum.

    Parameters
    ----------
    frame : RSS
        frame from which the measurements were taken
    measurements : Table
        per fiber table as returned by `measure_skylines` or `measure_continuum`
    max_deviation : float, optional
        fractional deviation from unity above which a fiber is counted as an
        outlier, by default 0.02
    min_snr : float, optional
        minimum median S/N for a feature to be considered reliable, by default 10.0
    min_good_fraction : float, optional
        minimum fraction of fitted fibers with good line fits, by default 0.8
    min_contrast : float, optional
        minimum median line peak over continuum floor, by default 1.0
    verbose : bool, optional
        whether to log warnings for unreliable features, by default True

    Returns
    -------
    Table
        one row per feature with columns:
          - name, kind ('line' or 'cont'), wave, wmin, wmax
          - nfibers: number of fibers used in the statistics (good fits for lines)
          - snr: median S/N of the feature
          - scatter: biweight scale of the normalized fluxes, after removing
            the sky gradient if `correct_sky_gradient` was run
          - noise: median expected fractional scatter from the errors
          - excess: scatter in excess of the noise, sqrt(scatter**2 - noise**2)
          - scatter_{telescope}: biweight scale per telescope
          - offset_sp{1,2,3}: spectrograph offsets, from the joint fit with the
            sky gradient if available, otherwise robust mean normalized flux
            per spectrograph minus 1
          - grad_x_{telescope}, grad_y_{telescope}, grad_amp_{telescope}: sky
            gradient components at the IFU edge and half peak-to-valley
            amplitude, relative to the telescope mean (NaN if not fitted)
          - frac_outliers: fraction of fibers deviating more than `max_deviation`
          - frac_good, contrast, fwhm, shift, chi2: for fitted lines, fraction
            of good fits and medians of the line peak over continuum, FWHM
            (Angstroms), center offset from the nominal wavelength (Angstroms)
            and reduced chi-square over good fits; NaN otherwise
          - reliable: whether the feature passes the criteria above
          - reason: why the feature is not reliable (empty if reliable)
    """
    slitmap = frame._slitmap
    telescope = _telescopes(frame)
    specid = slitmap["spectrographid"].data

    names = [c[:-len("_flux")] for c in measurements.colnames if c.endswith("_flux")]
    rows = []
    for name in names:
        meta = measurements[f"{name}_flux"].meta
        flux = measurements[f"{name}_flux"].data
        error = measurements[f"{name}_error"].data
        corrected = f"{name}_corr" in measurements.colnames
        norm = measurements[f"{name}_corr" if corrected else f"{name}_norm"].data
        good = np.isfinite(norm)
        fitted = f"{name}_good" in measurements.colnames

        row = {"name": name, "kind": meta.get("kind", ""), "wave": meta.get("wave", np.nan),
               "wmin": meta.get("wmin", np.nan), "wmax": meta.get("wmax", np.nan),
               "nfibers": int(good.sum()), "snr": np.nan, "scatter": np.nan, "noise": np.nan, "excess": np.nan}
        row.update({f"scatter_{tel}": np.nan for tel in TELESCOPES})
        row.update({f"offset_sp{i}": np.nan for i in (1, 2, 3)})
        row.update({"frac_outliers": np.nan, "frac_good": np.nan, "contrast": np.nan, "fwhm": np.nan, "shift": np.nan, "chi2": np.nan})
        gradients = meta.get("gradients", {})
        for tel in TELESCOPES:
            for column, key in (("x", "gx"), ("y", "gy"), ("amp", "amp")):
                row[f"grad_{column}_{tel}"] = float(gradients.get(tel, {}).get(key, np.nan))

        with np.errstate(invalid="ignore", divide="ignore"):
            # S/N over all attempted fibers for fitted lines, over valid fibers otherwise
            attempted = np.isfinite(flux) if fitted else good
            if attempted.any():
                row["snr"] = float(np.nanmedian(flux[attempted] / error[attempted]))
            if fitted and attempted.any():
                row["frac_good"] = float(good.sum() / attempted.sum())
                if good.any():
                    row["contrast"] = float(np.nanmedian(measurements[f"{name}_contrast"][good]))
                    row["fwhm"] = float(np.nanmedian(measurements[f"{name}_fwhm"][good]))
                    row["shift"] = float(np.nanmedian(measurements[f"{name}_center"][good]) - row["wave"])
                    row["chi2"] = float(np.nanmedian(measurements[f"{name}_chi2"][good]))

            if good.sum() >= 3:
                scatter = biweight_scale(norm[good])
                noise = np.nanmedian(np.abs(error[good] / flux[good]))
                row["scatter"], row["noise"] = scatter, noise
                row["excess"] = np.sqrt(max(scatter**2 - noise**2, 0.0))
                for tel in TELESCOPES:
                    in_tel = good & (telescope == tel)
                    row[f"scatter_{tel}"] = biweight_scale(norm[in_tel]) if in_tel.sum() >= 3 else np.nan
                for i in (1, 2, 3):
                    in_spec = good & (specid == i)
                    if corrected:
                        row[f"offset_sp{i}"] = float(meta.get("offsets", {}).get(i, np.nan))
                    else:
                        row[f"offset_sp{i}"] = biweight_location(norm[in_spec]) - 1 if in_spec.sum() >= 3 else np.nan
                row["frac_outliers"] = np.mean(np.abs(norm[good] - 1) > max_deviation)

        reasons = []
        if good.sum() < 3:
            reasons.append("too few fibers")
        if not row["snr"] >= min_snr:
            reasons.append(f"S/N {row['snr']:.1f} < {min_snr:g}")
        if fitted and not row["frac_good"] >= min_good_fraction:
            reasons.append(f"good fits {100 * np.nan_to_num(row['frac_good']):.0f}% < {100 * min_good_fraction:.0f}%")
        if fitted and not row["contrast"] >= min_contrast:
            reasons.append(f"line/continuum {row['contrast']:.2f} < {min_contrast:g}")
        row["reliable"] = not reasons
        row["reason"] = "; ".join(reasons)
        if reasons and verbose:
            log.warning(f"feature {name} is not reliable: {row['reason']}")
        rows.append(row)

    summary = Table(rows=rows)
    summary.meta["MAXDEV"] = max_deviation
    summary.meta["MINSNR"] = min_snr
    summary.meta["MINGOOD"] = min_good_fraction
    summary.meta["MINCONTR"] = min_contrast
    return summary


def plot_flatfield_qa(frame: RSS, measurements: Table, summary: Table, ylim: float = 0.06, title: str = None):
    """Plots the normalized feature fluxes along the slit and the summary vs wavelength

    Parameters
    ----------
    frame : RSS
        frame from which the measurements were taken
    measurements : Table
        per fiber measurements
    summary : Table
        summary as returned by `summarize_flatfield_qa`
    ylim : float, optional
        fractional deviation range shown in the slit panels, by default 0.06
    title : str, optional
        figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    fiberid = frame._slitmap["fiberid"].data
    telescope = _telescopes(frame)
    colors = {"Sci": "tab:blue", "SkyE": "tab:red", "SkyW": "tab:green"}
    nfeatures = len(summary)
    ncols = 3
    nrows = int(np.ceil(nfeatures / ncols))

    fig = plt.figure(figsize=(15, 4 + 2.2 * nrows), layout="constrained")
    if title is not None:
        fig.suptitle(title, fontsize="xx-large")
    gs = fig.add_gridspec(2 + nrows, ncols, height_ratios=[1.6, 1.2] + nrows * [1])

    # summary vs wavelength
    ax_sct = fig.add_subplot(gs[0, :])
    ax_off = fig.add_subplot(gs[1, :], sharex=ax_sct)
    reliable = summary["reliable"] if "reliable" in summary.colnames else np.ones(len(summary), dtype=bool)
    for kind, marker, label in [("line", "o", "sky lines"), ("cont", "s", "continuum")]:
        sel = (summary["kind"] == kind) & reliable
        if not sel.any():
            continue
        ax_sct.plot(summary["wave"][sel], 100 * summary["scatter"][sel], marker, color="0.2", mfc="none", mew=1.5, ms=9, label=f"observed scatter ({label})")
        ax_sct.plot(summary["wave"][sel], 100 * summary["noise"][sel], marker, color="0.6", ms=5, label=f"expected noise ({label})")
        ax_sct.plot(summary["wave"][sel], 100 * summary["excess"][sel], marker, color="tab:orange", ms=6, label=f"excess ({label})")
        for i, c in zip((1, 2, 3), ("tab:blue", "tab:red", "tab:green")):
            ax_off.plot(summary["wave"][sel], 100 * summary[f"offset_sp{i}"][sel], marker, color=c, ms=7, label=f"sp{i}" if kind == "line" else None)
    for row, ok in zip(summary, reliable):
        ax_sct.axvspan(row["wmin"], row["wmax"], lw=0, fc="0.85" if ok else "mistyrose", zorder=-1)
    if not reliable.all():
        ax_sct.set_title("Shaded red: features not reliable (not shown)", loc="right", fontsize="small", color="tab:red")
    ax_sct.set_ylim(0, None)
    ax_sct.set_ylabel("Fiber-to-fiber scatter (%)", fontsize="large")
    ax_sct.legend(loc="upper left", frameon=False, fontsize="small", ncols=3)
    ax_sct.tick_params(labelbottom=False)
    ax_off.axhline(0, ls="--", lw=1, color="0.2")
    ax_off.set_ylabel("Spec. offset (%)", fontsize="large")
    ax_off.set_xlabel("Wavelength (Angstroms)", fontsize="large")
    ax_off.legend(loc="upper left", frameon=False, fontsize="small", ncols=3)

    # normalized fluxes along the slit
    for i, row in enumerate(summary):
        ax = fig.add_subplot(gs[2 + i // ncols, i % ncols])
        norm = measurements[f"{row['name']}_norm"].data
        if f"{row['name']}_corr" in measurements.colnames:
            ax.plot(fiberid, norm, ".", ms=2, color="0.75", label="before sky gradient" if i == 0 else None)
            norm = measurements[f"{row['name']}_corr"].data
        for tel, c in colors.items():
            sel = telescope == tel
            ax.plot(fiberid[sel], norm[sel], ".", ms=2, color=c, label=tel if i == 0 else None)
        for m in (0.01, 0.02):
            ax.axhspan(1 - m, 1 + m, lw=0, fc="0.5", alpha=0.2)
        ax.axhline(1, ls="--", lw=1, color="0.2")
        ax.vlines([648.5, 2 * 648.5], 1 - ylim, 1 + ylim, ls=":", lw=1, color="0.2")
        ax.set_ylim(1 - ylim, 1 + ylim)
        label = f"{row['name']}  scatter={100*row['scatter']:.2f}%  noise={100*row['noise']:.2f}%  S/N={row['snr']:.0f}"
        if "reason" in summary.colnames and row["reason"]:
            label += f"\n{row['reason']}"
        ax.set_title(label, loc="left", fontsize="small", color="0.0" if reliable[i] else "tab:red")
        if i % ncols == 0:
            ax.set_ylabel("Normalized flux")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Fiber ID")
        if i == 0:
            ax.legend(loc="lower left", frameon=False, fontsize="x-small", ncols=3, markerscale=4)

    return fig


def measure_flatfield_qa(in_frame: Union[str, RSS],
                         skylines: List[float] = None,
                         cont_windows: List[Tuple[float, float]] = None,
                         line_method: str = "fit", line_kwargs: Dict = None,
                         telescopes: Tuple[str] = TELESCOPES,
                         max_deviation: float = 0.02, min_snr: float = 10.0,
                         min_good_fraction: float = 0.8, min_contrast: float = 1.0,
                         gradient_deg: int = 1,
                         out_table: str = None,
                         plot: bool = True, plot_path: str = None, display_plots: bool = False,
                         verbose: bool = True) -> Tuple[Table, Table, Table, Table]:
    """Measures the quality of the fiber flat fielding across the wavelength range

    Uses isolated sky lines and sky line-free continuum windows in a flat
    fielded, non sky subtracted science frame (lvmFrame). After flat fielding,
    the sky should be (nearly) the same in all the fibers of a telescope, so the
    fiber-to-fiber scatter in excess of the noise at each wavelength is a
    measure of the flat field residuals.

    Sky lines are measured by fitting a Gaussian on top of a continuum floor in
    each fiber (see `measure_skylines`), and a line is only used when its flux
    can be reliably measured above the underlying continuum (see
    `summarize_flatfield_qa`). Sky lines are insensitive to astrophysical
    continuum sources in the science IFU, but may be contaminated by nebular emission (e.g., [OI] 6300,6363 in
    HII regions). Continuum windows sample the full wavelength range, but are
    sensitive to stars and other continuum sources in the science IFU. Robust
    statistics are used throughout to mitigate both. Restrict `telescopes` to
    ('SkyE', 'SkyW') to avoid science field contamination altogether.

    Parameters
    ----------
    in_frame : str or RSS
        path to an lvmFrame or a flat fielded RSS object
    skylines : list[float], optional
        sky lines central wavelengths, by default SKYLINES_FLATFIELD_QA[channel]
    cont_windows : list[tuple[float, float]], optional
        continuum windows, by default CONTINUUM_FLATFIELD_QA[channel]
    line_method : str, optional
        sky line measurement method, 'fit' (Gaussian plus continuum floor) or
        'integrate' (sideband subtracted window sum), by default 'fit'
    line_kwargs : dict, optional
        additional keyword arguments passed to `measure_skylines` (e.g., fit_hw,
        cont_deg, max_shift, fwhm_range, min_fiber_snr), by default None
    telescopes : tuple[str], optional
        telescopes whose fibers are measured, by default ('Sci', 'SkyE', 'SkyW')
    max_deviation : float, optional
        fractional deviation counted as outlier, by default 0.02
    min_snr : float, optional
        minimum median S/N for a feature to be considered reliable, by default 10.0
    min_good_fraction : float, optional
        minimum fraction of good line fits for a sky line to be reliable, by default 0.8
    min_contrast : float, optional
        minimum median line peak over continuum for a sky line to be reliable, by default 1.0
    gradient_deg : int, optional
        polynomial degree of the sky gradient fitted across each IFU jointly
        with the spectrograph offsets (see `fit_sky_gradient`): 0 (offsets
        only), 1 (plane) or 2 (quadratic); None skips the correction, by default 1
    out_table : str, optional
        if given, path to a FITS file where the per fiber measurements and
        summaries are written, by default None
    plot : bool, optional
        whether to produce the QA plot, by default True
    plot_path : str, optional
        product path used to name the QA plot, which is saved in a 'qa'
        subdirectory next to it, by default the frame path (or `out_table` if
        `in_frame` is an RSS object)
    display_plots : bool, optional
        whether to display the QA plot instead of only saving it, by default False
    verbose : bool, optional
        whether to log the measurements of each feature, by default True

    Returns
    -------
    lines : Table
        per fiber sky line measurements
    lines_summary : Table
        sky lines summary statistics
    cont : Table
        per fiber continuum measurements
    cont_summary : Table
        continuum summary statistics
    """
    if isinstance(in_frame, str):
        log.info(f"loading frame from {os.path.basename(in_frame)}")
        frame = lvmFrame.from_file(in_frame)
        product_path = in_frame
    else:
        frame = in_frame
        product_path = out_table
    product_path = plot_path or product_path

    channel = frame._header["CCD"][0]
    expnum = frame._header.get("EXPOSURE")
    skylines = SKYLINES_FLATFIELD_QA[channel] if skylines is None else skylines
    cont_windows = CONTINUUM_FLATFIELD_QA[channel] if cont_windows is None else cont_windows

    # keep only features within the wavelength range of the frame
    line_kwargs = dict(line_kwargs or {})
    if line_method == "fit":
        margin = line_kwargs.get("fit_hw", 8.0)
    else:
        margin = line_kwargs.get("line_hw", 4.0) + line_kwargs.get("cont_gap", 6.0) + line_kwargs.get("cont_width", 4.0)
    wmin, wmax = np.nanmin(frame._wave), np.nanmax(frame._wave)
    skylines = [w for w in skylines if wmin + margin < w < wmax - margin]
    cont_windows = [(a, b) for a, b in cont_windows if wmin < a and b < wmax]

    if verbose:
        log.info(f"measuring {len(skylines)} sky lines and {len(cont_windows)} continuum windows in {expnum = }, {channel = }")
    lines = measure_skylines(frame, skylines, method=line_method, telescopes=telescopes, **line_kwargs)
    cont = measure_continuum(frame, cont_windows, telescopes=telescopes)
    if gradient_deg is not None:
        correct_sky_gradient(frame, lines, deg=gradient_deg, telescopes=telescopes)
        correct_sky_gradient(frame, cont, deg=gradient_deg, telescopes=telescopes)
    criteria = dict(max_deviation=max_deviation, min_snr=min_snr, min_good_fraction=min_good_fraction,
                    min_contrast=min_contrast, verbose=verbose)
    lines_summary = summarize_flatfield_qa(frame, lines, **criteria)
    cont_summary = summarize_flatfield_qa(frame, cont, **criteria)

    if verbose:
        for row in list(lines_summary) + list(cont_summary):
            fit_info = f", good fits = {100*row['frac_good']:.0f}%, line/cont = {row['contrast']:.2f}" if np.isfinite(row["frac_good"]) else ""
            log.info(f"  {row['name']:>12s}: scatter = {100*row['scatter']:.2f}%, noise = {100*row['noise']:.2f}%, "
                     f"excess = {100*row['excess']:.2f}%, S/N = {row['snr']:.1f}, nfibers = {row['nfibers']}{fit_info}")

    if out_table is not None:
        fibers = frame._slitmap["fiberid", "spectrographid", "telescope"]
        hdus = fits.HDUList([
            fits.PrimaryHDU(header=fits.Header({"EXPOSURE": expnum, "CCD": channel})),
            fits.table_to_hdu(hstack([fibers, lines])) if len(lines.colnames) else fits.BinTableHDU(name="LINES"),
            fits.table_to_hdu(lines_summary) if len(lines_summary) else fits.BinTableHDU(name="LINES_SUMMARY"),
            fits.table_to_hdu(hstack([fibers, cont])) if len(cont.colnames) else fits.BinTableHDU(name="CONT"),
            fits.table_to_hdu(cont_summary) if len(cont_summary) else fits.BinTableHDU(name="CONT_SUMMARY"),
        ])
        for hdu, name in zip(hdus[1:], ["LINES", "LINES_SUMMARY", "CONT", "CONT_SUMMARY"]):
            hdu.name = name
        os.makedirs(os.path.dirname(os.path.abspath(out_table)), exist_ok=True)
        hdus.writeto(out_table, overwrite=True)
        log.info(f"written flat field QA measurements to {out_table}")

    if plot:
        summary = vstack([t for t in (lines_summary, cont_summary) if len(t)])
        measurements = hstack([t for t in (lines, cont) if len(t.colnames)])
        fig = plot_flatfield_qa(frame, measurements, summary, title=f"Flat field QA for {expnum = }, {channel = }")
        if product_path is not None:
            save_fig(fig, product_path=product_path, to_display=display_plots, figure_path="qa", label="flatfield_qa")
        elif display_plots:
            plt.show()
        else:
            plt.close(fig)

    return lines, lines_summary, cont, cont_summary


# header keywords stored with the per frame summaries: (column, keyword, default)
FRAME_METADATA = [
    ("expnum", "EXPOSURE", -999),
    ("mjd", "MJD", -999),
    ("tileid", "TILE_ID", -999),
    ("exptime", "EXPTIME", np.nan),
    ("obstime", "OBSTIME", ""),
    ("sciam", "SCIAM", np.nan),
    ("skyeam", "SKYEAM", np.nan),
    ("skywam", "SKYWAM", np.nan),
    ("drpqual", "DRPQUAL", -999),
    ("drpver", "DRPVER", ""),
]

# marker color, symbol and legend label of each feature kind in the report
KIND_STYLE = {"line": (SERIES[0], "circle", "Sky lines"), "cont": (SERIES[1], "square", "Continuum")}


def find_frames(drpver: str, channels: str = "brz", mjds: List[int] = None, mjd_range: Tuple[int, int] = None,
                tileids: List[int] = None, redux_dir: str = None) -> List[str]:
    """Finds lvmFrame files in the reductions directory of a given DRP version

    Files are searched following the SAS layout:
    {redux_dir}/{drpver}/{tilegrp}/{tileid}/{mjd}/lvmFrame-{channel}-{expnum}.fits*

    Parameters
    ----------
    drpver : str
        DRP version (tag) of the reductions
    channels : str, optional
        channels to search for, by default 'brz'
    mjds : list[int], optional
        MJDs to search for, by default all
    mjd_range : tuple[int, int], optional
        inclusive (min, max) MJD range to keep, by default no limits
    tileids : list[int], optional
        tile IDs to search for, by default all
    redux_dir : str, optional
        reductions root directory, by default $LVM_SPECTRO_REDUX

    Returns
    -------
    list[str]
        sorted list of lvmFrame paths
    """
    redux_dir = redux_dir or os.getenv("LVM_SPECTRO_REDUX")
    if redux_dir is None:
        raise ValueError("either `redux_dir` or $LVM_SPECTRO_REDUX have to be given")

    mjd_patterns = ["*"] if mjds is None else [str(mjd) for mjd in mjds]
    tile_patterns = ["*"] if tileids is None else [str(tileid) for tileid in tileids]
    paths = set()
    for tileid, mjd, channel in product(tile_patterns, mjd_patterns, channels):
        paths.update(glob(os.path.join(redux_dir, drpver, "*", tileid, mjd, f"lvmFrame-{channel}-*.fits*")))

    if mjd_range is not None:
        mjd_min, mjd_max = mjd_range
        paths = {p for p in paths if mjd_min <= int(os.path.basename(os.path.dirname(p))) <= mjd_max}

    def _sort_key(path):
        match = re.match(r"lvmFrame-([brz])-(\d+)\.fits", os.path.basename(path))
        return (int(match.group(2)), match.group(1)) if match else (np.inf, path)

    paths = sorted(paths, key=_sort_key)
    log.info(f"found {len(paths)} lvmFrame files for {drpver = } in {redux_dir}")
    return paths


def _frame_metadata(header) -> Dict:
    metadata = {}
    for column, keyword, default in FRAME_METADATA:
        value = header.get(keyword, default)
        try:
            metadata[column] = type(default)(default if value is None else value)
        except (TypeError, ValueError):
            metadata[column] = default
    metadata["channel"] = header.get("CCD", " ")[0]
    return metadata


def _process_frame(args) -> Tuple[str, Union[Table, None], Union[str, None]]:
    """Runs the flat field QA in a single frame, returns (path, summary, error)"""
    frame_path, out_dir, frame_plots, min_exptime, qa_kwargs = args
    try:
        frame = lvmFrame.from_file(frame_path)
        metadata = _frame_metadata(frame._header)
        if frame._header.get("IMAGETYP", "object") != "object":
            return frame_path, None, f"skipped IMAGETYP = {frame._header.get('IMAGETYP')}"
        if not metadata["exptime"] >= min_exptime:
            return frame_path, None, f"skipped exptime = {metadata['exptime']} < {min_exptime}"

        _, lines_summary, _, cont_summary = measure_flatfield_qa(
            frame, plot=frame_plots, plot_path=os.path.join(out_dir, os.path.basename(frame_path)),
            verbose=False, **qa_kwargs)
        summary = vstack([t for t in (lines_summary, cont_summary) if len(t)])
        for i, (column, value) in enumerate(metadata.items()):
            summary.add_column(value, name=column, index=i)
        summary.add_column(os.path.basename(frame_path), name="filename", index=0)
        return frame_path, summary, None
    except Exception as e:
        return frame_path, None, f"{type(e).__name__}: {e}"
    finally:
        plt.close("all")


def _count_reason(reasons, key: str) -> int:
    return int(sum(key in str(reason) for reason in reasons))


def aggregate_flatfield_qa(summary: Table, percentiles: Tuple[float, float] = (16, 84)) -> Table:
    """Aggregates the per frame summaries into per feature statistics

    Only features flagged as reliable in each frame are used.

    Parameters
    ----------
    summary : Table
        per frame and feature summary, as returned by `run_flatfield_qa_batch`
    percentiles : tuple[float, float], optional
        lower and upper percentiles reported, by default (16, 84)

    Returns
    -------
    Table
        one row per channel and feature with the number of frames, and the
        median and percentiles of the excess, scatter, noise, spectrograph
        offsets, fraction of outliers, fraction of good line fits and line peak
        over continuum across frames, and the number of frames in which the
        feature was rejected for low S/N (nlow_snr), few good line fits
        (nfew_good) or a line too weak relative to the continuum (nlow_contrast)
    """
    reliable = summary[summary["reliable"]]
    reasons = np.asarray(summary["reason"]) if "reason" in summary.colnames else np.full(len(summary), "")
    rows = []
    for channel, name in sorted(set(zip(summary["channel"], summary["name"])), key=lambda k: summary["wave"][summary["name"] == k[1]][0]):
        feature = summary[(summary["channel"] == channel) & (summary["name"] == name)]
        sel = reliable[(reliable["channel"] == channel) & (reliable["name"] == name)]
        in_feature = (summary["channel"] == channel) & (summary["name"] == name)
        row = {"channel": channel, "name": name, "kind": feature["kind"][0], "wave": feature["wave"][0],
               "nframes": len(feature), "nreliable": len(sel),
               "nlow_snr": _count_reason(reasons[in_feature], "S/N"),
               "nfew_good": _count_reason(reasons[in_feature], "good fits"),
               "nlow_contrast": _count_reason(reasons[in_feature], "line/continuum")}
        for column in ["excess", "scatter", "noise", "offset_sp1", "offset_sp2", "offset_sp3", "frac_outliers", "frac_good", "contrast",
                       "grad_x_Sci", "grad_y_Sci", "grad_amp_Sci"]:
            if column not in sel.colnames:
                continue
            values = np.asarray(sel[column], dtype=float) if len(sel) else np.array([np.nan])
            row[f"{column}_median"] = np.nanmedian(values) if np.isfinite(values).any() else np.nan
            for q in percentiles:
                row[f"{column}_p{q:g}"] = np.nanpercentile(values, q) if np.isfinite(values).any() else np.nan
        rows.append(row)
    return Table(rows=rows)


def _robust_range(values, percentiles=(16, 84)):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return np.nan, np.nan, np.nan
    lo, med, hi = np.nanpercentile(values, [percentiles[0], 50, percentiles[1]])
    return med, lo, hi


def _fmt_array(values, fmt):
    return np.array([format(v, fmt) if np.isfinite(v) else "–" for v in np.asarray(values, dtype=float)])


def _channel_annotations(aggregate: Table) -> List[Dict]:
    annotations = []
    for channel in "brz":
        waves = aggregate["wave"][aggregate["channel"] == channel]
        if len(waves):
            annotations.append(dict(x=float(np.mean([np.min(waves), np.max(waves)])), y=1.0, xref="x", yref="paper",
                                    text=f"<b>{channel}</b>", showarrow=False, yanchor="top",
                                    font=dict(color=THEME["muted"], size=13)))
    return annotations


def _errorbar_trace(x, med, lo, hi, name, color, symbol, customdata, hovertemplate, showlegend=True):
    return go.Scatter(
        x=x, y=med, name=name, mode="markers", showlegend=showlegend,
        marker=dict(size=10, color=color, symbol=symbol, line=dict(width=2, color=THEME["surface"])),
        error_y=dict(type="data", symmetric=False, array=hi - med, arrayminus=med - lo, color=color, thickness=1.5, width=0),
        customdata=customdata, hovertemplate=hovertemplate,
    )


def figure_excess_wavelength(summary: Table, aggregate: Table, max_frame_points: int = 20000, seed: int = 0):
    """Excess fiber-to-fiber scatter vs wavelength: median and 16-84th percentiles across frames"""
    fig = go.Figure()

    # individual frames, hidden by default
    reliable = summary[summary["reliable"]]
    if len(reliable):
        idx = np.arange(len(reliable))
        if idx.size > max_frame_points:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_frame_points, replace=False))
        fig.add_trace(go.Scattergl(
            x=np.asarray(reliable["wave"][idx]) + np.random.default_rng(seed).uniform(-10, 10, idx.size),
            y=100 * np.asarray(reliable["excess"][idx]), mode="markers", name="Individual frames",
            visible="legendonly", marker=dict(size=4, color=THEME["muted"], opacity=0.35),
            customdata=np.stack([reliable["filename"][idx], reliable["name"][idx]], axis=-1),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>excess %{y:.2f}%<extra></extra>"))

    for kind, (color, symbol, label) in KIND_STYLE.items():
        agg = aggregate[aggregate["kind"] == kind]
        if len(agg) == 0:
            continue
        med, lo, hi = (100 * np.asarray(agg[f"excess_{s}"], dtype=float) for s in ("median", "p16", "p84"))
        customdata = np.stack([agg["channel"], agg["name"], agg["nreliable"], agg["nframes"], _fmt_array(lo, ".2f"), _fmt_array(hi, ".2f")], axis=-1)
        fig.add_trace(_errorbar_trace(
            agg["wave"], med, lo, hi, label, color, symbol, customdata,
            "<b>%{customdata[1]}</b> (%{customdata[0]})<br>excess %{y:.2f}% "
            "[%{customdata[4]}, %{customdata[5]}]<br>%{customdata[2]} of %{customdata[3]} frames reliable<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=aggregate["wave"], y=100 * np.asarray(aggregate["noise_median"], dtype=float), mode="markers",
        name="Expected noise", marker=dict(size=12, symbol="line-ew-open", color=THEME["muted"], line=dict(width=2, color=THEME["muted"])),
        customdata=aggregate["name"], hovertemplate="%{customdata}<br>noise %{y:.2f}%<extra></extra>"))

    fig.update_layout(base_layout(420, annotations=_channel_annotations(aggregate),
                                   xaxis_title="Wavelength (Å)", yaxis_title="Excess scatter (%)",
                                   yaxis_rangemode="tozero"))
    return style_axes(fig)


def figure_offsets_wavelength(aggregate: Table):
    """Spectrograph flux offsets vs wavelength: median and 16-84th percentiles across frames"""
    fig = go.Figure()
    for i, specid in enumerate((1, 2, 3)):
        med, lo, hi = (100 * np.asarray(aggregate[f"offset_sp{specid}_{s}"], dtype=float) for s in ("median", "p16", "p84"))
        customdata = np.stack([aggregate["channel"], aggregate["name"], _fmt_array(lo, "+.2f"), _fmt_array(hi, "+.2f")], axis=-1)
        fig.add_trace(_errorbar_trace(
            np.asarray(aggregate["wave"]) + 12 * (specid - 2), med, lo, hi, f"sp{specid}", SERIES[i], "circle", customdata,
            f"<b>sp{specid}</b> %{{customdata[1]}} (%{{customdata[0]}})<br>offset %{{y:+.2f}}% "
            "[%{customdata[2]}, %{customdata[3]}]<extra></extra>"))
    fig.update_layout(base_layout(340, annotations=_channel_annotations(aggregate),
                                   xaxis_title="Wavelength (Å)", yaxis_title="Spectrograph offset (%)"))
    return style_axes(fig)


def figure_outliers_wavelength(aggregate: Table, max_deviation: float):
    """Fraction of fibers deviating more than `max_deviation` vs wavelength"""
    fig = go.Figure()
    for kind, (color, symbol, label) in KIND_STYLE.items():
        agg = aggregate[aggregate["kind"] == kind]
        if len(agg) == 0:
            continue
        med, lo, hi = (100 * np.asarray(agg[f"frac_outliers_{s}"], dtype=float) for s in ("median", "p16", "p84"))
        customdata = np.stack([agg["channel"], agg["name"], _fmt_array(lo, ".1f"), _fmt_array(hi, ".1f")], axis=-1)
        fig.add_trace(_errorbar_trace(
            agg["wave"], med, lo, hi, label, color, symbol, customdata,
            "<b>%{customdata[1]}</b> (%{customdata[0]})<br>%{y:.1f}% of fibers "
            "[%{customdata[2]}, %{customdata[3]}]<extra></extra>"))
    fig.update_layout(base_layout(300, annotations=_channel_annotations(aggregate),
                                   xaxis_title="Wavelength (Å)", yaxis_title=f"Fibers off by >{100 * max_deviation:g}% (%)",
                                   yaxis_rangemode="tozero"))
    return style_axes(fig)


def nightly_excess(summary: Table) -> Table:
    """Median excess per night (MJD) and feature, using reliable measurements only"""
    reliable = summary[summary["reliable"]]
    rows = []
    for channel, name, wave in sorted(set(zip(reliable["channel"], reliable["name"], reliable["wave"])), key=lambda k: k[2]):
        feature = reliable[(reliable["channel"] == channel) & (reliable["name"] == name)]
        for mjd in np.unique(feature["mjd"]):
            night = feature[feature["mjd"] == mjd]
            rows.append({"channel": channel, "name": name, "wave": wave, "mjd": int(mjd),
                         "nframes": len(night), "excess": float(np.nanmedian(night["excess"]))})
    return Table(rows=rows) if rows else Table(names=["channel", "name", "wave", "mjd", "nframes", "excess"])


def figure_excess_timeline(nightly: Table):
    """Heatmap of the nightly median excess per feature"""
    features = sorted(set(zip(nightly["wave"], nightly["channel"], nightly["name"])))
    mjds = np.unique(nightly["mjd"])
    z = np.full((len(features), mjds.size), np.nan)
    n = np.zeros_like(z)
    for i, (_, channel, name) in enumerate(features):
        sel = (nightly["channel"] == channel) & (nightly["name"] == name)
        j = np.searchsorted(mjds, nightly["mjd"][sel])
        z[i, j] = 100 * np.asarray(nightly["excess"][sel])
        n[i, j] = nightly["nframes"][sel]
    labels = [f"{channel} · {name}" for _, channel, name in features]
    zmax = max(0.5, float(np.nanpercentile(z, 98))) if np.isfinite(z).any() else 1.0

    fig = go.Figure(go.Heatmap(
        x=mjds, y=labels, z=z, customdata=n, colorscale=SEQUENTIAL, zmin=0, zmax=zmax, xgap=1, ygap=2,
        colorbar=dict(title=dict(text="Excess (%)", side="right"), thickness=12, outlinewidth=0, tickfont=dict(color=THEME["muted"])),
        hovertemplate="MJD %{x}<br>%{y}<br>excess %{z:.2f}%<br>%{customdata} frames<extra></extra>"))
    fig.update_layout(base_layout(max(260, 26 * len(labels) + 110), xaxis_title="MJD", margin=dict(l=150, r=16, t=16, b=52)))
    style_axes(fig)
    fig.update_xaxes(showgrid=False, tickformat="d")
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(family=MONO_FONT, color=THEME["ink2"], size=11))
    return fig


def nightly_offsets(summary: Table) -> Table:
    """Median spectrograph offsets per night and channel across reliable features"""
    reliable = summary[summary["reliable"]]
    rows = []
    for channel in "brz":
        in_channel = reliable[reliable["channel"] == channel]
        for mjd in np.unique(in_channel["mjd"]):
            night = in_channel[in_channel["mjd"] == mjd]
            row = {"channel": channel, "mjd": int(mjd), "nframes": len(set(night["filename"]))}
            for specid in (1, 2, 3):
                row[f"offset_sp{specid}"] = float(np.nanmedian(night[f"offset_sp{specid}"]))
            rows.append(row)
    return Table(rows=rows) if rows else Table(names=["channel", "mjd", "nframes", "offset_sp1", "offset_sp2", "offset_sp3"])


def figure_offsets_timeline(nightly: Table):
    """Nightly median spectrograph offsets, one panel per channel"""
    channels = [c for c in "brz" if c in nightly["channel"]] or ["b"]
    fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[f"{c} channel" for c in channels])
    for row, channel in enumerate(channels, start=1):
        night = nightly[nightly["channel"] == channel]
        for i, specid in enumerate((1, 2, 3)):
            fig.add_trace(go.Scatter(
                x=night["mjd"], y=100 * np.asarray(night[f"offset_sp{specid}"]), name=f"sp{specid}",
                mode="lines+markers", legendgroup=f"sp{specid}", showlegend=row == 1,
                line=dict(width=2, color=SERIES[i]), marker=dict(size=8, color=SERIES[i], line=dict(width=2, color=THEME["surface"])),
                customdata=night["nframes"],
                hovertemplate=f"<b>sp{specid}</b> {channel}<br>MJD %{{x}}<br>offset %{{y:+.2f}}%<br>%{{customdata}} frames<extra></extra>"),
                row=row, col=1)
        fig.update_yaxes(title_text="Offset (%)", row=row, col=1)
    fig.update_xaxes(title_text="MJD", row=len(channels), col=1)
    fig.update_layout(base_layout(200 * len(channels) + 100, hovermode="x unified"))
    fig.update_layout(legend=dict(x=1, xanchor="right"))
    fig.update_annotations(font=dict(color=THEME["ink2"], size=13), x=0, xanchor="left")
    style_axes(fig)
    fig.update_xaxes(tickformat="d")
    return fig


def frame_gradients(summary: Table) -> Table:
    """Per frame median sky gradient across the science IFU over reliable features"""
    names = ["filename", "channel", "mjd", "expnum", "nfeatures", "grad_x", "grad_y", "grad_amp"]
    if "grad_x_Sci" not in summary.colnames:
        return Table(names=names)
    reliable = summary[summary["reliable"] & np.isfinite(summary["grad_x_Sci"])]
    rows = []
    for filename in np.unique(reliable["filename"]):
        frame = reliable[reliable["filename"] == filename]
        rows.append({"filename": filename, "channel": frame["channel"][0], "mjd": int(frame["mjd"][0]),
                     "expnum": int(frame["expnum"][0]), "nfeatures": len(frame),
                     "grad_x": float(np.nanmedian(frame["grad_x_Sci"])), "grad_y": float(np.nanmedian(frame["grad_y_Sci"])),
                     "grad_amp": float(np.nanmedian(frame["grad_amp_Sci"]))})
    return Table(rows=rows) if rows else Table(names=names)


def figure_gradient_vectors(gradients: Table):
    """Sky gradient across the science IFU per frame, in IFU coordinates"""
    fig = go.Figure()
    limit = 1.0
    for i, channel in enumerate("brz"):
        sel = gradients[gradients["channel"] == channel] if len(gradients) else gradients
        if len(sel) == 0:
            continue
        gx, gy = 100 * np.asarray(sel["grad_x"]), 100 * np.asarray(sel["grad_y"])
        limit = max(limit, float(np.nanpercentile(np.abs(np.concatenate([gx, gy])), 98)) * 1.15)
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="markers", name=f"{channel} frames", legendgroup=channel,
            marker=dict(size=8, color=SERIES[i], opacity=0.55, line=dict(width=1, color=THEME["surface"])),
            customdata=np.stack([sel["filename"], sel["mjd"], _fmt_array(100 * np.asarray(sel["grad_amp"]), ".2f")], axis=-1),
            hovertemplate="%{customdata[0]}<br>MJD %{customdata[1]}<br>gradient (%{x:.2f}, %{y:.2f})%<br>"
                          "amplitude %{customdata[2]}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[np.nanmedian(gx)], y=[np.nanmedian(gy)], mode="markers", name=f"{channel} median", legendgroup=channel,
            marker=dict(size=16, color=SERIES[i], symbol="diamond", line=dict(width=2, color=THEME["ink"])),
            hovertemplate=f"<b>{channel} median</b><br>(%{{x:.2f}}, %{{y:.2f}})%<extra></extra>"))
    fig.update_layout(base_layout(520, xaxis_title="Gradient along IFU x at the edge (%)", yaxis_title="Gradient along IFU y at the edge (%)"))
    style_axes(fig)
    fig.update_xaxes(range=[-limit, limit], zeroline=True, zerolinewidth=1.5, constrain="domain")
    fig.update_yaxes(range=[-limit, limit], zeroline=True, zerolinewidth=1.5, scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def frame_ranking(summary: Table) -> Table:
    """Per frame median excess across reliable features, worst first"""
    reliable = summary[summary["reliable"]]
    rows = []
    for filename in np.unique(reliable["filename"]):
        frame = reliable[reliable["filename"] == filename]
        rows.append({"filename": filename, "expnum": int(frame["expnum"][0]), "mjd": int(frame["mjd"][0]),
                     "tileid": int(frame["tileid"][0]), "channel": frame["channel"][0], "nfeatures": len(frame),
                     "excess": float(np.nanmedian(frame["excess"])), "excess_max": float(np.nanmax(frame["excess"])),
                     "worst": frame["name"][int(np.nanargmax(frame["excess"]))]})
    if not rows:
        return Table(names=["filename", "expnum", "mjd", "tileid", "channel", "nfeatures", "excess", "excess_max", "worst"])
    ranking = Table(rows=rows)
    ranking.sort("excess", reverse=True)
    return ranking


def _fmt(value, fmt=".2f", scale=100.0, sign=False):
    try:
        value = float(value) * scale
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(value):
        return "–"
    return format(value, ("+" if sign else "") + fmt)


def write_flatfield_qa_report(summary: Table, aggregate: Table, out_html: str, run_info: Dict = None,
                              title: str = "LVM Flat-Field QA", nworst: int = 25) -> str:
    """Writes the batch flat field QA as a self-contained interactive HTML page

    The page shows summary tiles, interactive plotly charts of the flat field
    quality metrics as a function of wavelength and time (each with a table
    view), and a ranking of the frames with the largest flat field residuals.
    plotly.js is loaded from cdn.jsdelivr.net. The page follows the viewer's
    light/dark theme.

    Parameters
    ----------
    summary : Table
        per frame and feature summary, as returned by `run_flatfield_qa_batch`
    aggregate : Table
        per feature statistics, as returned by `aggregate_flatfield_qa`
    out_html : str
        path to the output HTML file
    run_info : dict, optional
        run metadata shown in the page header (e.g., nskipped, nfailed, telescopes)
    title : str, optional
        page title, by default 'LVM Flat-Field QA'
    nworst : int, optional
        number of frames listed in the ranking of worst frames, by default 25

    Returns
    -------
    str
        path to the written HTML file
    """
    run_info = dict(run_info or {})
    max_deviation = summary.meta.get("MAXDEV", 0.02)
    min_snr = summary.meta.get("MINSNR", 10.0)
    min_good = summary.meta.get("MINGOOD", 0.8)
    min_contrast = summary.meta.get("MINCONTR", 1.0)
    reliable = summary[summary["reliable"]]
    drpvers = ", ".join(sorted(set(summary["drpver"]))) or "unknown"
    mjds = np.asarray(summary["mjd"])
    nframes = len(set(summary["filename"]))
    nights = len(set(mjds))
    nightly = nightly_excess(summary)
    nightly_off = nightly_offsets(summary)
    ranking = frame_ranking(summary)
    gradients = frame_gradients(summary)

    figures = {
        "fig-excess": figure_excess_wavelength(summary, aggregate),
        "fig-offsets": figure_offsets_wavelength(aggregate),
        "fig-outliers": figure_outliers_wavelength(aggregate, max_deviation),
        "fig-timeline": figure_excess_timeline(nightly),
        "fig-offsets-time": figure_offsets_timeline(nightly_off),
        "fig-gradients": figure_gradient_vectors(gradients),
    }

    # summary tiles
    tiles = [("Frames measured", f"{nframes:,}", f"{len(set(reliable['filename'])):,} with reliable features"),
             ("Nights", f"{nights:,}", f"MJD {mjds.min()}–{mjds.max()}" if len(mjds) else "")]
    for channel in "brz":
        values = reliable["excess"][reliable["channel"] == channel]
        if len(values):
            med, lo, hi = _robust_range(values)
            tiles.append((f"{channel} median excess", f"{100 * med:.2f}%", f"16–84th: {100 * lo:.2f}–{100 * hi:.2f}%"))

    # run metadata
    meta = [("DRP version", drpvers), ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
            ("Telescopes", ", ".join(run_info.pop("telescopes", TELESCOPES))),
            ("Reliable feature", f"median S/N ≥ {min_snr:g}"),
            ("Reliable sky line", f"≥ {100 * min_good:g}% good fits, peak/continuum ≥ {min_contrast:g}"),
            ("Outlier fiber", f"off by > {100 * max_deviation:g}%")]
    for key in ("nskipped", "nfailed"):
        if key in run_info:
            meta.append(({"nskipped": "Frames skipped", "nfailed": "Frames failed"}[key], f"{run_info.pop(key):,}"))
    meta += [(str(k), str(v)) for k, v in run_info.items()]

    # tables
    feature_rows = [[escape(r["channel"]), f'<span class="mono">{escape(r["name"])}</span>',
                     "sky line" if r["kind"] == "line" else "continuum", f'{r["wave"]:.1f}',
                     f'{r["nreliable"]} / {r["nframes"]}',
                     f'{_fmt(r["excess_median"])} <span class="range">[{_fmt(r["excess_p16"])}, {_fmt(r["excess_p84"])}]</span>',
                     _fmt(r["noise_median"]), _fmt(r["offset_sp1_median"], sign=True), _fmt(r["offset_sp2_median"], sign=True),
                     _fmt(r["offset_sp3_median"], sign=True), _fmt(r["frac_outliers_median"], ".1f"),
                     _fmt(r["frac_good_median"], ".0f") if "frac_good_median" in aggregate.colnames else "–",
                     _fmt(r["contrast_median"], ".2f", scale=1) if "contrast_median" in aggregate.colnames else "–",
                     (f'{r["nlow_snr"]} · {r["nfew_good"]} · {r["nlow_contrast"]}' if r["kind"] == "line" else str(r["nlow_snr"])),
                     _fmt(r["grad_amp_Sci_median"]) if "grad_amp_Sci_median" in aggregate.colnames else "–"]
                    for r in aggregate]
    feature_table = html_table(
        [("Ch", "channel"), ("Feature", "L: sky line center; C: continuum window"), ("Type", ""), ("λ (Å)", "central wavelength"),
         ("Reliable / frames", "frames with median S/N above threshold"), ("Excess (%) [16–84th]", "scatter in excess of noise"),
         ("Noise (%)", "median expected scatter from errors"), ("sp1 (%)", "median offset"), ("sp2 (%)", "median offset"),
         ("sp3 (%)", "median offset"), ("Outliers (%)", "median fraction of outlier fibers"),
         ("Good fits (%)", "median fraction of fibers with good line fits"), ("Peak/cont", "median line peak over continuum floor"),
         ("Rejected: S/N · fits · contrast", "frames where the feature was not reliable, by failed criterion"),
         ("Sci gradient (%)", "median sky gradient amplitude across the science IFU")],
        feature_rows, numeric=[False, False, False, True, True, True, True, True, True, True, True, True, True, True, True])

    gradient_rows = []
    for channel in "brz":
        sel = gradients[gradients["channel"] == channel] if len(gradients) else gradients
        if len(sel) == 0:
            continue
        gx, gy = np.nanmedian(sel["grad_x"]), np.nanmedian(sel["grad_y"])
        gradient_rows.append([escape(channel), str(len(sel)), _fmt(gx, sign=True), _fmt(gy, sign=True),
                              _fmt(np.hypot(gx, gy)), _fmt(np.nanmedian(sel["grad_amp"]))])
    gradient_table = html_table(
        [("Ch", "channel"), ("Frames", ""), ("Median x (%)", "median gradient along IFU x at the edge"),
         ("Median y (%)", "median gradient along IFU y at the edge"), ("Persistent (%)", "length of the median gradient vector"),
         ("Typical amplitude (%)", "median gradient amplitude per frame")],
        gradient_rows, numeric=[False, True, True, True, True, True])

    worst_rows = [[f'<span class="mono">{escape(r["filename"])}</span>', str(r["expnum"]), str(r["mjd"]), str(r["tileid"]),
                   escape(r["channel"]), str(r["nfeatures"]), _fmt(r["excess"]), _fmt(r["excess_max"]),
                   f'<span class="mono">{escape(r["worst"])}</span>'] for r in ranking[:nworst]]
    worst_table = html_table(
        [("File", ""), ("Exposure", ""), ("MJD", ""), ("Tile", ""), ("Ch", "channel"), ("Features", "reliable features"),
         ("Median excess (%)", "median across reliable features"), ("Max excess (%)", "largest across reliable features"),
         ("Worst feature", "")],
        worst_rows, numeric=[False, True, True, True, False, True, True, True, False])

    nightly_rows = [[str(r["mjd"]), escape(r["channel"]), f'<span class="mono">{escape(r["name"])}</span>',
                     str(r["nframes"]), _fmt(r["excess"])] for r in nightly]
    nightly_table = html_table([("MJD", ""), ("Ch", ""), ("Feature", ""), ("Frames", ""), ("Excess (%)", "nightly median")],
                                nightly_rows, numeric=[True, False, False, True, True])
    offsets_rows = [[str(r["mjd"]), escape(r["channel"]), str(r["nframes"]), _fmt(r["offset_sp1"], sign=True),
                     _fmt(r["offset_sp2"], sign=True), _fmt(r["offset_sp3"], sign=True)] for r in nightly_off]
    offsets_table = html_table([("MJD", ""), ("Ch", ""), ("Frames", ""), ("sp1 (%)", ""), ("sp2 (%)", ""), ("sp3 (%)", "")],
                                offsets_rows, numeric=[True, False, True, True, True, True])

    definitions = DEFINITIONS_TEMPLATE
    for token, value in {"MAXDEV": f"{max_deviation:g}", "MAXDEVPCT": f"{100 * max_deviation:g}", "MINSNR": f"{min_snr:g}",
                         "MINGOOD": f"{min_good:g}", "MINCONTR": f"{min_contrast:g}"}.items():
        definitions = definitions.replace(f"@{token}@", value)

    html = REPORT_TEMPLATE.format(
        head=page_head(title), title=escape(title), drpvers=escape(drpvers), meta=meta_html(meta), tiles=tiles_html(tiles), definitions=definitions,
        feature_table=feature_table, worst_table=worst_table, gradient_table=gradient_table, nightly_table=nightly_table, offsets_table=offsets_table,
        nworst=min(nworst, len(ranking)), maxdev=f"{100 * max_deviation:g}", plotly_version=get_plotlyjs_version(),
        figures=figures_json(figures), dark_map=json.dumps(DARK_COLORS), seq_dark=json.dumps(SEQUENTIAL_DARK))

    # keep the page pure ASCII (HTML character references for the markup, the chart JSON is
    # already escaped) so it renders correctly regardless of how the file is served
    html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"written flat field QA report to {out_html}")
    return out_html


DEFINITIONS_TEMPLATE = r"""<section id="definitions">
    <h2>How the quantities are computed</h2>
    <p>Fibers are indexed by \(i\), pixels by \(p\). \(T(i)\) and \(s(i)\) are the telescope (Sci, SkyE, SkyW) and the
    spectrograph of fiber \(i\). BL and BS are the biweight location and scale, robust versions of the mean and standard
    deviation.</p>
    <div class="defs">
      <div class="def">
        <h3>Sky line flux</h3>
        <p>The counts \(f_{i,p}\) within &plusmn;8 &Aring; of the line are fitted with a Gaussian integrated over the pixel
        edges \(\lambda^{-}_{p}, \lambda^{+}_{p}\), on top of a linear continuum floor. \(\Phi\) is the standard normal
        cumulative distribution and \(F_i\) the line flux; the fit is weighted by the pixel errors and the flux error
        \(\sigma_{F_i}\) comes from its covariance matrix.</p>
        <div class="eq">\[ f_{i,p} \simeq F_i\left[\Phi\!\left(\frac{\lambda^{+}_{p}-\mu_i}{\sigma_i}\right)-\Phi\!\left(\frac{\lambda^{-}_{p}-\mu_i}{\sigma_i}\right)\right] + c_{0,i} + c_{1,i}\,(\lambda_p-\lambda_0) \]</div>
      </div>
      <div class="def">
        <h3>Line peak over continuum</h3>
        <p>Height of the fitted line peak relative to the continuum floor under it, with \(\Delta\lambda_i\) the pixel size
        at the line center.</p>
        <div class="eq">\[ \kappa_i = \frac{F_i\,\Delta\lambda_i}{\sqrt{2\pi}\,\sigma_i\,c_{0,i}} \]</div>
      </div>
      <div class="def">
        <h3>Continuum level</h3>
        <p>Mean counts per pixel in a line-free window, with \(w_{i,p}\) the fraction of pixel \(p\) inside the window
        (zero for masked pixels).</p>
        <div class="eq">\[ C_i = \frac{\sum_p w_{i,p}\, f_{i,p}}{\sum_p w_{i,p}} \]</div>
      </div>
      <div class="def">
        <h3>Normalized flux</h3>
        <p>Each line flux (or continuum level) is divided by the robust mean of the good fibers of the same telescope,
        since each telescope looks at a different patch of sky.</p>
        <div class="eq">\[ \tilde F_i = \frac{F_i}{\mathrm{BL}_{\,j \in T(i)}\left(F_j\right)} \]</div>
      </div>
      <div class="def">
        <h3>Sky gradient and spectrograph offsets</h3>
        <p>Fitted jointly over all fibers by least squares with iterative 4&sigma; clipping. \(\hat x_i, \hat y_i\) are the
        fiber positions from the IFU center in units of the IFU radius. The offsets \(o_s\) are shared by all telescopes and
        add up to zero over the fibers.</p>
        <div class="eq">\[ \tilde F_i - 1 = a_{T(i)} + g^{x}_{T(i)}\,\hat x_i + g^{y}_{T(i)}\,\hat y_i + o_{s(i)}, \qquad \sum_i o_{s(i)} = 0 \]</div>
      </div>
      <div class="def">
        <h3>Gradient-corrected flux</h3>
        <p>The sky model \(S_i\) is divided out. What is left contains the spectrograph offsets and the fiber-to-fiber
        flat-field errors.</p>
        <div class="eq">\[ r_i = \frac{\tilde F_i}{S_i}, \qquad S_i = 1 + a_{T(i)} + g^{x}_{T(i)}\,\hat x_i + g^{y}_{T(i)}\,\hat y_i \]</div>
      </div>
      <div class="def">
        <h3>Scatter, noise and excess scatter</h3>
        <p>The observed scatter \(s\), the scatter expected from the errors alone \(n\), and the excess \(e\) plotted
        against wavelength.</p>
        <div class="eq">\[ s = \mathrm{BS}_i\left(r_i\right), \qquad n = \mathrm{med}_i\!\left(\frac{\sigma_{F_i}}{F_i}\right), \qquad e = \sqrt{\max\left(s^2 - n^2,\ 0\right)} \]</div>
      </div>
      <div class="def">
        <h3>Sky gradient vector and amplitude</h3>
        <p>Gradient at the IFU edge and half peak-to-valley of the sky model across the IFU, relative to the telescope mean
        \(\langle S \rangle_T\). Across frames, the persistent gradient is the length of the median vector.</p>
        <div class="eq">\[ \vec g_T = \frac{\left(g^{x}_{T},\ g^{y}_{T}\right)}{\langle S \rangle_T}, \qquad A_T = \frac{\max_{i \in T} S_i - \min_{i \in T} S_i}{2\,\langle S \rangle_T} \]</div>
      </div>
      <div class="def">
        <h3>Outlier fibers</h3>
        <p>Fraction of the \(N\) good fibers whose corrected flux deviates more than \(\delta = @MAXDEV@\)
        (@MAXDEVPCT@%).</p>
        <div class="eq">\[ q = \frac{1}{N}\sum_i \mathbf{1}\left[\,\left|r_i - 1\right| > \delta\,\right] \]</div>
      </div>
      <div class="def">
        <h3>Reliable feature in a frame</h3>
        <p>A feature is used only if its median S/N passes; a sky line also needs most fibers with good fits and a peak that
        stands above the continuum. A fiber fit is good when it converged, \(F_i/\sigma_{F_i} \ge 5\), its FWHM is within
        0.5&ndash;2 times the LSF and the center is not at the &plusmn;1.5 &Aring; limit (defaults).</p>
        <div class="eq">\[ \mathrm{med}_i\!\left(\frac{F_i}{\sigma_{F_i}}\right) \ge @MINSNR@, \qquad \frac{N_{\mathrm{good}}}{N_{\mathrm{fit}}} \ge @MINGOOD@, \qquad \mathrm{med}_i\left(\kappa_i\right) \ge @MINCONTR@ \]</div>
      </div>
      <div class="def">
        <h3>Across frames</h3>
        <p>Markers show the median over the frames where the feature is reliable, error bars the 16th&ndash;84th
        percentile range. Nightly values are medians over the frames of each MJD.</p>
        <div class="eq">\[ \hat e = \mathrm{med}_{k}\left(e_k\right), \qquad \left[\,P_{16}\left(e_k\right),\ P_{84}\left(e_k\right)\right] \]</div>
      </div>
    </div>
  </section>"""


REPORT_TEMPLATE = """{head}
<div class="page">
  <header>
    <div class="eyebrow">LVM DRP · fiber flat field · {drpvers}</div>
    <h1>{title}</h1>
    <p>Each flat-fielded science frame (lvmFrame) is checked against the sky. Within one telescope the sky is nearly uniform,
    so after a good flat field the fluxes of isolated sky lines and the continuum in line-free windows agree across fibers
    up to photon noise. The <strong>excess scatter</strong> \\(e\\) is the fiber-to-fiber scatter that the noise does
    not explain, measured after removing the sky gradient across each IFU. It measures the flat-field residuals at each
    wavelength. The equations behind every plotted quantity are listed under <a href="#definitions">How the quantities
    are computed</a>.</p>
    <p>Sky line fluxes come from fitting a Gaussian on a continuum floor in every fiber. A line counts in a frame only when
    its flux is measured reliably above the continuum: high enough S/N, good fits in most fibers, and a peak that stands
    above the continuum under it. Lines in bright continuum, such as twilight or moonlit sky, drop out.</p>
    <dl class="meta">{meta}</dl>
  </header>

  <div class="tiles">{tiles}</div>

  {definitions}

  <section>
    <h2>Excess scatter across wavelength</h2>
    <p>Markers show the median over frames and bars the 16th–84th percentile range. Only features with a reliable
    measurement in a frame are used. Click <em>Individual frames</em> in the legend to show every measurement.</p>
    <div class="chart" id="fig-excess" role="img" aria-label="Excess fiber-to-fiber scatter versus wavelength"></div>
    <details><summary>Table view: all features</summary>{feature_table}</details>
  </section>

  <section>
    <h2>Spectrograph offsets</h2>
    <p>Flux offset of each spectrograph, fitted jointly with the sky gradient. The science IFU is split between the
    spectrographs in three wedges, so a sky gradient would otherwise show up as spectrograph offsets; the sky IFUs, with
    fibers from all three spectrographs on a small patch of sky, help separate the two. A spectrograph that sits away from
    zero across many frames points to an error in the spectrograph-to-spectrograph flat-field factors.</p>
    <div class="chart" id="fig-offsets" role="img" aria-label="Spectrograph offsets versus wavelength"></div>
  </section>

  <section>
    <h2>Sky gradient across the science IFU</h2>
    <p>Each point is the gradient fitted in one frame, as the flux change from the IFU center to its edge, in IFU
    coordinates. A real sky gradient points in a different direction from frame to frame, so the points scatter around
    zero. A gradient that stays put in IFU coordinates is not sky: it is a smooth illumination error in the flat field that
    the gradient fit removes from the excess scatter, so check it here.</p>
    <div class="chart" id="fig-gradients" role="img" aria-label="Sky gradient vectors across the science IFU per frame"></div>
    <details><summary>Table view: gradients per channel</summary>{gradient_table}</details>
  </section>

  <section>
    <h2>Outlier fibers</h2>
    <p>Fraction of fibers whose normalized flux is off by more than {maxdev}%. This includes the noise, so compare it
    against features with similar S/N.</p>
    <div class="chart" id="fig-outliers" role="img" aria-label="Fraction of outlier fibers versus wavelength"></div>
  </section>

  <section>
    <h2>Excess scatter over time</h2>
    <p>Nightly median excess for each feature. Changes that line up with a new set of fiber flats show up as vertical edges.</p>
    <div class="chart" id="fig-timeline" role="img" aria-label="Heatmap of nightly excess scatter per feature"></div>
    <details><summary>Table view: nightly excess</summary>{nightly_table}</details>
  </section>

  <section>
    <h2>Spectrograph offsets over time</h2>
    <p>Nightly median offset of each spectrograph across reliable features.</p>
    <div class="chart" id="fig-offsets-time" role="img" aria-label="Nightly spectrograph offsets per channel"></div>
    <details><summary>Table view: nightly offsets</summary>{offsets_table}</details>
  </section>

  <section>
    <h2>Frames with the largest residuals</h2>
    <p>The {nworst} frames with the highest median excess across their reliable features.</p>
    {worst_table}
  </section>

  <p class="note">Generated by lvmdrp.qa.fiberflats. Charts use plotly.js {plotly_version}.</p>
</div>

<script>
window.MathJax = {{ tex: {{ inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] }}, svg: {{ fontCache: "global" }} }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js" async></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{plotly_version}/plotly.min.js"></script>
<script>
(function () {{
  const FIGURES = {figures};
  const DARK = {dark_map};
  const SEQ_DARK = {seq_dark};
  const pattern = new RegExp(Object.keys(DARK).join("|"), "gi");
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  function isDark() {{
    const theme = document.documentElement.getAttribute("data-theme");
    return theme ? theme === "dark" : media.matches;
  }}

  function render() {{
    if (!window.Plotly) {{
      for (const id of Object.keys(FIGURES)) {{
        const el = document.getElementById(id);
        el.innerHTML = '<div class="plotly-missing">This chart needs plotly.js from cdn.jsdelivr.net, which did not load. The table views below each chart hold the same numbers.</div>';
      }}
      return;
    }}
    const dark = isDark();
    for (const [id, figure] of Object.entries(FIGURES)) {{
      let text = JSON.stringify(figure);
      if (dark) text = text.replace(pattern, (m) => DARK[m.toLowerCase()]);
      const fig = JSON.parse(text);
      if (dark) fig.data.forEach((trace) => {{ if (trace.type === "heatmap") trace.colorscale = SEQ_DARK; }});
      Plotly.react(id, fig.data, fig.layout, {{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] }});
    }}
  }}

  render();
  media.addEventListener("change", render);
  new MutationObserver(render).observe(document.documentElement, {{ attributes: true, attributeFilter: ["data-theme"] }});
}})();
</script>
"""


def run_flatfield_qa_batch(frame_paths: List[str], out_dir: str, label: str = "flatfield_qa",
                           nprocs: int = 1, frame_plots: bool = False, min_exptime: float = 0.0,
                           overwrite: bool = False, checkpoint_every: int = 200,
                           report_title: str = "LVM Flat-Field QA", **qa_kwargs) -> Tuple[Table, Table]:
    """Runs the flat field QA on a list of lvmFrames and summarizes the results

    For each frame, the sky lines and continuum windows are measured with
    `measure_flatfield_qa`. The per frame and feature summaries are stacked in
    a single table together with relevant header metadata, and aggregated per
    feature across frames. The results are written to
    '{out_dir}/{label}.fits' (HDUs 'FRAMES' and 'FEATURES') and an interactive
    report with the quality metrics as a function of wavelength and time is
    written to '{out_dir}/{label}.html' (see `write_flatfield_qa_report`).

    The run can be resumed: unless `overwrite` is True, frames already present
    in an existing output table are not measured again. The output table is
    written every `checkpoint_every` frames.

    Parameters
    ----------
    frame_paths : list[str]
        list of lvmFrame paths, see `find_frames`
    out_dir : str
        output directory
    label : str, optional
        name of the output table and report, by default 'flatfield_qa'
    nprocs : int, optional
        number of parallel processes, by default 1
    frame_plots : bool, optional
        whether to save the QA plot of each frame in '{out_dir}/qa', by default False
    min_exptime : float, optional
        frames with shorter exposure times are skipped, by default 0.0 seconds
    overwrite : bool, optional
        whether to measure all frames again, by default False
    checkpoint_every : int, optional
        number of frames after which the output table is written, by default 200
    report_title : str, optional
        title of the HTML report, by default 'LVM Flat-Field QA'
    **qa_kwargs
        additional keyword arguments passed to `measure_flatfield_qa` (e.g.,
        skylines, cont_windows, telescopes, min_snr, max_deviation)

    Returns
    -------
    summary : Table
        one row per frame and feature, with the frame metadata and the
        columns returned by `summarize_flatfield_qa`
    aggregate : Table
        one row per channel and feature, as returned by `aggregate_flatfield_qa`
    """
    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, f"{label}.fits")

    summaries = []
    if not overwrite and os.path.isfile(table_path):
        previous = Table.read(table_path, hdu="FRAMES")
        previous.convert_bytestring_to_unicode()
        summaries.append(previous)
        done = set(previous["filename"])
        frame_paths = [p for p in frame_paths if os.path.basename(p) not in done]
        log.info(f"resuming from {table_path}: {len(done)} frames already measured")

    def _write(summaries, final=False):
        if not summaries:
            return None, None
        summary = vstack(summaries, metadata_conflicts="silent")
        summary.meta["MAXDEV"] = qa_kwargs.get("max_deviation", 0.02)
        summary.meta["MINSNR"] = qa_kwargs.get("min_snr", 10.0)
        summary.meta["MINGOOD"] = qa_kwargs.get("min_good_fraction", 0.8)
        summary.meta["MINCONTR"] = qa_kwargs.get("min_contrast", 1.0)
        aggregate = aggregate_flatfield_qa(summary)
        hdus = fits.HDUList([fits.PrimaryHDU(), fits.table_to_hdu(summary), fits.table_to_hdu(aggregate)])
        hdus[1].name, hdus[2].name = "FRAMES", "FEATURES"
        hdus.writeto(table_path, overwrite=True)
        if final:
            log.info(f"written {len(set(summary['filename']))} frames summary to {table_path}")
        return summary, aggregate

    log.info(f"running flat field QA on {len(frame_paths)} frames using {nprocs} process(es)")
    tasks = [(path, out_dir, frame_plots, min_exptime, qa_kwargs) for path in frame_paths]
    nskipped, nfailed = 0, 0
    with Pool(nprocs) if nprocs > 1 else _SerialPool() as pool:
        iterator = tqdm(pool.imap_unordered(_process_frame, tasks), total=len(tasks), desc="flat field QA", unit="frame", ascii=True)
        for i, (path, summary, message) in enumerate(iterator):
            if summary is not None:
                summaries.append(summary)
            elif message.startswith("skipped"):
                nskipped += 1
            else:
                nfailed += 1
                log.error(f"failed {os.path.basename(path)}: {message}")
            if checkpoint_every and (i + 1) % checkpoint_every == 0:
                _write(summaries)

    log.info(f"skipped {nskipped} frames, failed {nfailed} frames")
    summary, aggregate = _write(summaries, final=True)
    if summary is None:
        log.warning("no frames were measured, nothing to report")
        return None, None

    run_info = {"telescopes": qa_kwargs.get("telescopes", TELESCOPES), "nskipped": nskipped, "nfailed": nfailed}
    write_flatfield_qa_report(summary, aggregate, out_html=os.path.join(out_dir, f"{label}.html"),
                              run_info=run_info, title=report_title)

    return summary, aggregate


def report_flatfield_qa(table_path: str, out_html: str = None, **report_kwargs) -> str:
    """Writes the HTML report from an existing batch QA table without measuring again

    Parameters
    ----------
    table_path : str
        path to the FITS table written by `run_flatfield_qa_batch`
    out_html : str, optional
        output HTML path, by default the table path with '.html' extension
    **report_kwargs
        additional keyword arguments passed to `write_flatfield_qa_report`

    Returns
    -------
    str
        path to the written HTML file
    """
    summary = Table.read(table_path, hdu="FRAMES")
    summary.convert_bytestring_to_unicode()
    aggregate = aggregate_flatfield_qa(summary)
    out_html = out_html or os.path.splitext(table_path)[0] + ".html"
    return write_flatfield_qa_report(summary, aggregate, out_html=out_html, **report_kwargs)


class _SerialPool:
    """Minimal stand-in for multiprocessing.Pool running in the current process"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap_unordered(self, func, iterable):
        return map(func, iterable)


# ---------------------------------------------------------------------------
# master fiber flats across calibration epochs
# ---------------------------------------------------------------------------

# epochs with this trigger are routine, any other trigger is highlighted
ROUTINE_TRIGGER = "Normal operations"
# diverging color scale of the IFU views (blue <-> red around a neutral gray), light and dark themes
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2aaa6", "#e34948", "#a3302e"]
DIVERGING_DARK = ["#9ec5f4", "#3987e5", "#1c4f8f", "#383835", "#8a3533", "#e66767", "#f4a9a8"]


def default_epochs_path() -> str:
    """Path of the calibration epochs file in lvmcore."""
    return os.path.join(os.getenv("LVMCORE_DIR", "."), "calibrations", "calibration-epochs.yaml")


def load_calibration_epochs(epochs_path: str = None) -> Dict[int, Dict]:
    """Load the calibration epochs as a dictionary {epoch MJD: epoch definition}.

    Parameters
    ----------
    epochs_path : str, optional
        Path of the calibration epochs file. Default is :func:`default_epochs_path`.

    Returns
    -------
    dict[int, dict]
        Epoch definitions with their ``flavors``, ``trigger`` and ``comment``.
    """
    epochs_path = epochs_path or default_epochs_path()
    with open(epochs_path) as f:
        epochs = yaml.safe_load(f)["epochs"]
    return {int(mjd): epoch or {} for mjd, epoch in epochs.items()}


def fiberflat_path(mjd_epoch: int, channel: str, kind: str = "twilight", flats_dir: str = None,
                   drpver: str = None, redux_dir: str = None) -> str:
    """Path of the master fiber flat of an epoch and channel.

    The flat is taken from a pipeline version (``drpver``) when given, else from
    the master calibrations directory (``flats_dir``, by default ``$LVM_MASTER_DIR``).

    Parameters
    ----------
    mjd_epoch : int
        MJD of the calibration epoch.
    channel : str
        Channel, one of 'b', 'r' or 'z'.
    kind : str, optional
        Kind of fiber flat, 'twilight' or 'dome'. Default is 'twilight'.
    flats_dir : str, optional
        Master calibrations directory. Default is ``$LVM_MASTER_DIR``.
    drpver : str, optional
        Pipeline version whose master calibrations are used instead.
    redux_dir : str, optional
        Reductions root directory for ``drpver``. Default is ``$LVM_SPECTRO_REDUX``.

    Returns
    -------
    str
        Path of the fiber flat (it may not exist).
    """
    name = f"lvm-mfiberflat_{kind}-{channel}.fits"
    if drpver is not None:
        redux_dir = redux_dir or os.getenv("LVM_SPECTRO_REDUX", ".")
        return os.path.join(redux_dir, drpver, tileid_grp(11111), "11111", str(mjd_epoch), "calib", name)
    flats_dir = flats_dir or os.getenv("LVM_MASTER_DIR", ".")
    return os.path.join(flats_dir, str(mjd_epoch), name)


def wavelength_edges(flat: RSS, bin_width: float = 50.0) -> np.ndarray:
    """Edges of the wavelength bins covered by all the fibers of a flat.

    Parameters
    ----------
    flat : RSS
        Fiber flat.
    bin_width : float, optional
        Width of the bins in Angstroms. Default is 50.

    Returns
    -------
    np.ndarray
        Uniformly spaced bin edges.
    """
    wave = np.broadcast_to(flat._wave, flat._data.shape)
    valid = np.isfinite(flat._data) & (flat._data > 0)
    if flat._mask is not None:
        valid &= ~flat._mask.astype(bool)
    wmin = np.nanmin(np.where(valid, wave, np.nan), axis=1)
    wmax = np.nanmax(np.where(valid, wave, np.nan), axis=1)
    start, stop = np.nanmedian(wmin), np.nanmedian(wmax)
    nbins = int((stop - start) // bin_width)
    if nbins < 1:
        raise ValueError(f"the flat covers less than one {bin_width} Angstrom bin")
    return start + bin_width * np.arange(nbins + 1)


def bin_fiberflat(flat: RSS, edges: np.ndarray, step: float = 0.5) -> np.ndarray:
    """Median fiber flat in wavelength bins.

    Each fiber is resampled with linear interpolation on a grid of ``step``
    Angstroms, so the fibers of flats with different wavelength solutions are
    compared at the same wavelengths. Masked, non-finite and non-positive
    pixels are excluded.

    Parameters
    ----------
    flat : RSS
        Fiber flat.
    edges : np.ndarray
        Uniformly spaced edges of the wavelength bins, see :func:`wavelength_edges`.
    step : float, optional
        Resampling step in Angstroms. Default is 0.5.

    Returns
    -------
    np.ndarray
        Binned flat, fibers x bins, NaN where a bin has no valid pixels.
    """
    data = np.asarray(flat._data, dtype=float)
    bad = ~np.isfinite(data) | (data <= 0)
    if flat._mask is not None:
        bad |= flat._mask.astype(bool)
    data = np.where(bad, np.nan, data)
    wave = np.broadcast_to(flat._wave, data.shape)

    nbins = len(edges) - 1
    nsub = max(int(round((edges[1] - edges[0]) / step)), 1)
    grid = edges[0] + (edges[1] - edges[0]) / nsub * (np.arange(nbins * nsub) + 0.5)
    resampled = np.full((data.shape[0], grid.size), np.nan)
    for ifiber in range(data.shape[0]):
        ok = np.isfinite(wave[ifiber])
        if ok.sum() >= 2:
            # NaNs propagate to the neighboring grid points, so masked pixels stay excluded
            resampled[ifiber] = np.interp(grid, wave[ifiber][ok], data[ifiber][ok], left=np.nan, right=np.nan)
    return bn.nanmedian(resampled.reshape(data.shape[0], nbins, nsub), axis=2)


def ratio_stats(ratio: np.ndarray, max_deviation: float = 0.01) -> Dict[str, float]:
    """Distribution statistics of flat-field ratios.

    Parameters
    ----------
    ratio : np.ndarray
        Ratio values; non-finite values are ignored.
    max_deviation : float, optional
        Deviation from 1 counted as off. Default is 0.01.

    Returns
    -------
    dict[str, float]
        Number of values, 2.5, 25, 50, 75 and 97.5 percentiles, robust sigma
        (1.4826 times the median absolute deviation) and fraction off by more
        than ``max_deviation``.
    """
    values = ratio[np.isfinite(ratio)]
    if values.size == 0:
        return {"nvalues": 0, "p2.5": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "p97.5": np.nan,
                "sigma": np.nan, "frac_off": np.nan}
    p025, p25, p50, p75, p975 = np.percentile(values, [2.5, 25, 50, 75, 97.5])
    return {"nvalues": int(values.size), "p2.5": p025, "p25": p25, "median": p50, "p75": p75, "p97.5": p975,
            "sigma": 1.4826 * np.median(np.abs(values - p50)), "frac_off": float(np.mean(np.abs(values - 1) > max_deviation))}


def decompose_ratio(slitmap, fiber_ratio: np.ndarray, good: np.ndarray, deg: int = 1,
                    telescopes: Tuple[str] = TELESCOPES) -> Dict:
    """Split the per-fiber ratio of two flats into large-scale components and a fiber-level residual.

    The per-fiber ratio, normalized by its median, is fitted with
    :func:`fit_sky_gradient`: an offset per spectrograph
    (shared by all telescopes) plus an intercept and a polynomial gradient of
    degree ``deg`` across each IFU. These capture changes in the spectrograph
    normalization factors and in the illumination pattern across the IFUs;
    what remains is the change of the fiber-to-fiber flat field.

    Parameters
    ----------
    slitmap : astropy.table.Table
        Slitmap of the flats.
    fiber_ratio : np.ndarray
        Ratio of the two flats per fiber (e.g., median over wavelength).
    good : np.ndarray
        Fibers to use in the fit.
    deg : int, optional
        Polynomial degree of the gradient across each IFU. Default is 1.
    telescopes : tuple[str], optional
        Telescopes to fit. Default is ('Sci', 'SkyE', 'SkyW').

    Returns
    -------
    dict
        level: median ratio; model: per fiber large-scale model relative to
        ``level`` (NaN outside the fitted fibers); offsets: spectrograph
        offsets; gradients: per telescope gradient (gx, gy, amp), see
        :func:`fit_sky_gradient`.
    """
    level = float(np.nanmedian(fiber_ratio[good])) if np.isfinite(fiber_ratio[good]).any() else np.nan
    norm = np.where(good, fiber_ratio / level, np.nan)
    fit = fit_sky_gradient(SimpleNamespace(_slitmap=slitmap), norm, deg=deg, telescopes=telescopes)
    specid = np.asarray(slitmap["spectrographid"]).astype(int)
    offsets = np.array([fit["offsets"].get(i, 0.0) for i in specid])
    return {"level": level, "model": fit["sky"] + np.nan_to_num(offsets), "offsets": fit["offsets"], "gradients": fit["gradients"]}


def _header_factors(header, channel) -> Dict[str, float]:
    """Spectrograph factors applied to a fiber flat, from its header (NaN if missing)."""
    factors = {}
    for i in (1, 2, 3):
        value = None
        for key in (f"{channel.upper()} FIBERFLAT FACTOR{i}", f"{channel} FIBERFLAT FACTOR{i}"):
            if header is not None and key in header:
                value = header[key]
                break
        factors[f"hdr_factor{i}"] = float(value) if value is not None else np.nan
    return factors


def _decode(column) -> np.ndarray:
    values = np.asarray(column)
    return np.char.decode(values) if values.dtype.kind == "S" else values.astype(str)


def _ifu_geometry(slitmap) -> Dict:
    """Science IFU fiber positions in units of the IFU radius and the hexagon shape of the fibers."""
    sci = _decode(slitmap["telescope"]) == "Sci"
    x, y = np.asarray(slitmap["xpmm"], dtype=float)[sci], np.asarray(slitmap["ypmm"], dtype=float)[sci]
    x, y = x - x.mean(), y - y.mean()
    radius = np.max(np.hypot(x, y))
    x, y = x / radius, y / radius

    # hexagonal packing: cell circumradius from the fiber pitch, vertices 30 deg off the neighbor directions
    distance, index = cKDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=2)
    pitch = float(np.median(distance[:, 1]))
    angles = np.degrees(np.arctan2(y[index[:, 1]] - y, x[index[:, 1]] - x)) % 60
    rotation = float(np.median(angles)) + 30

    labels = _decode(slitmap["orig_ifulabel"] if "orig_ifulabel" in slitmap.colnames else slitmap["fiberid"])[sci]
    return {"select": sci, "x": np.round(x, 5).tolist(), "y": np.round(y, 5).tolist(),
            "fiberid": np.asarray(slitmap["fiberid"])[sci].astype(int).tolist(), "label": labels.tolist(),
            "hex_radius": pitch / np.sqrt(3), "hex_rotation": rotation}


def _encode(values: np.ndarray) -> list:
    """Values scaled by 1e4 as integers, None for non-finite values, to keep the page small."""
    return [int(round(v * 1e4)) if np.isfinite(v) else None for v in values]


def _short_trigger(trigger):
    return str(trigger or "unknown").replace("Warm-up", "Warm up")


def figure_ratio_boxes(summary: pd.DataFrame, mjd_ref: int, channels: str, missing: Dict[str, list],
                       max_deviation: float = 0.01, prefix: str = "") -> go.Figure:
    """Box plots of the flat-field ratios against the reference flat vs epoch MJD, one row per channel.

    ``prefix`` selects the statistics, '' for the ratios as measured or 'corr_'
    for the ratios after removing the large-scale components.
    """
    if prefix:
        keys = ("nvalues", "p2.5", "p25", "median", "p75", "p97.5", "sigma", "frac_off")
        summary = summary.drop(columns=list(keys)).rename(columns={f"{prefix}{key}": key for key in keys})
    fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[f"{channel} channel" for channel in channels])
    mjds = np.sort(summary["epoch"].unique())
    width = float(np.clip(0.6 * np.min(np.diff(mjds)), 2, 12)) if mjds.size > 1 else 8.0

    for row, channel in enumerate(channels, start=1):
        data = summary[(summary["channel"] == channel) & (summary["epoch"] != mjd_ref) & (summary["nvalues"] > 0)]
        for routine, color, name in [(True, SERIES[0], "Normal operations"), (False, SERIES[1], "Other trigger")]:
            sel = data[(data["trigger"] == ROUTINE_TRIGGER) == routine]
            if len(sel) == 0:
                continue
            fig.add_trace(go.Box(
                x=sel["epoch"], q1=sel["p25"], median=sel["median"], q3=sel["p75"],
                lowerfence=sel["p2.5"], upperfence=sel["p97.5"], width=width, name=name, legendgroup=name,
                showlegend=row == 1, marker_color=color, line=dict(color=color, width=1.5), fillcolor=_translucent(color),
                hoverinfo="skip"), row=row, col=1)
            # invisible hover targets with the epoch metadata
            fig.add_trace(go.Scatter(
                x=sel["epoch"], y=sel["median"], mode="markers", showlegend=False, legendgroup=name,
                marker=dict(size=22, color=color, opacity=0),
                customdata=np.stack([_fmt_list(sel["p25"], ".4f"), _fmt_list(sel["p75"], ".4f"), _fmt_list(sel["p2.5"], ".4f"),
                                     _fmt_list(sel["p97.5"], ".4f"), _fmt_list(100 * sel["sigma"], ".2f"),
                                     _fmt_list(100 * sel["frac_off"], ".1f"), sel["trigger"].map(_short_trigger),
                                     sel["comment"].fillna("").map(lambda c: _wrap(escape(c)))], axis=-1),
                hovertemplate=(f"<b>MJD %{{x}}</b> \u00b7 {channel}<br>%{{customdata[6]}}<br>"
                               "median %{y:.4f}, IQR [%{customdata[0]}, %{customdata[1]}]<br>"
                               "2.5\u201397.5%: [%{customdata[2]}, %{customdata[3]}]<br>"
                               f"robust \u03c3 %{{customdata[4]}}%, off by \u2265{100 * max_deviation:g}%: %{{customdata[5]}}%<br>"
                               "<i>%{customdata[7]}</i><extra></extra>")), row=row, col=1)

        if missing.get(channel):
            fig.add_trace(go.Scatter(
                x=missing[channel], y=[1.0] * len(missing[channel]), mode="markers", name="No flat", legendgroup="missing",
                showlegend=row == 1, marker=dict(size=9, symbol="x-thin", color=THEME["muted"], line=dict(width=2, color=THEME["muted"])),
                hovertemplate="<b>MJD %{x}</b><br>no fiber flat available<extra></extra>"), row=row, col=1)

        fig.add_hline(y=1.0, line=dict(color=THEME["axis"], width=1), row=row, col=1)
        fig.add_vline(x=mjd_ref, line=dict(color=THEME["ink2"], width=1.5), row=row, col=1)
        if len(data):
            lo, hi = np.nanmin(data["p2.5"]), np.nanmax(data["p97.5"])
            pad = 0.1 * max(hi - lo, 0.01)
            fig.update_yaxes(range=[min(lo, 1) - pad, max(hi, 1) + pad], row=row, col=1)
        fig.update_yaxes(title_text="Flat / reference", tickformat=".3f", row=row, col=1)

    fig.add_annotation(x=mjd_ref, y=1.0, xref="x", yref="paper", text=f"reference {mjd_ref}", showarrow=False,
                       xanchor="left", yanchor="bottom", xshift=4, font=dict(color=THEME["ink2"], size=12))
    fig.update_xaxes(title_text="Epoch MJD", tickformat="d", row=len(channels), col=1)
    fig.update_layout(base_layout(260 * len(channels) + 110, margin=dict(l=72, r=16, t=56, b=52)))
    fig.update_layout(legend=dict(x=1, xanchor="right"))
    fig.update_annotations(selector=lambda annotation: str(annotation.text).endswith(" channel"), x=0, xanchor="left")
    fig.update_annotations(font=dict(color=THEME["ink2"], size=13))
    return style_axes(fig)


def figure_offsets(summary: pd.DataFrame, mjd_ref: int, channels: str) -> go.Figure:
    """Spectrograph offsets of each epoch relative to the reference, one row per channel."""
    fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{channel} channel" for channel in channels])
    for row, channel in enumerate(channels, start=1):
        data = summary[(summary["channel"] == channel) & (summary["epoch"] != mjd_ref) & (summary["nvalues"] > 0)].sort_values("epoch")
        for i, specid in enumerate((1, 2, 3)):
            fig.add_trace(go.Scatter(
                x=data["epoch"], y=100 * data[f"offset_sp{specid}"], mode="lines+markers", name=f"sp{specid}", legendgroup=f"sp{specid}",
                showlegend=row == 1, line=dict(width=2, color=SERIES[i]), marker=dict(size=8, color=SERIES[i], line=dict(width=2, color=THEME["surface"])),
                hovertemplate=f"<b>MJD %{{x}}</b> \u00b7 {channel}<br>sp{specid} offset %{{y:+.2f}}%<extra></extra>"), row=row, col=1)
        fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1), row=row, col=1)
        fig.add_vline(x=mjd_ref, line=dict(color=THEME["ink2"], width=1.5), row=row, col=1)
        fig.update_yaxes(title_text="Offset (%)", row=row, col=1)
    return _finish_epoch_figure(fig, channels)


def figure_gradients(summary: pd.DataFrame, mjd_ref: int, channels: str) -> go.Figure:
    """Gradient across the science IFU of each epoch relative to the reference, one row per channel."""
    fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{channel} channel" for channel in channels])
    for row, channel in enumerate(channels, start=1):
        data = summary[(summary["channel"] == channel) & (summary["epoch"] != mjd_ref) & (summary["nvalues"] > 0)].sort_values("epoch")
        for i, (column, name) in enumerate((("grad_x", "along IFU x"), ("grad_y", "along IFU y"))):
            fig.add_trace(go.Scatter(
                x=data["epoch"], y=100 * data[column], mode="lines+markers", name=name, legendgroup=name, showlegend=row == 1,
                line=dict(width=2, color=SERIES[i]), marker=dict(size=8, color=SERIES[i], line=dict(width=2, color=THEME["surface"])),
                hovertemplate=f"<b>MJD %{{x}}</b> \u00b7 {channel}<br>gradient {name} %{{y:+.2f}}% at the edge<extra></extra>"), row=row, col=1)
        fig.add_hline(y=0, line=dict(color=THEME["axis"], width=1), row=row, col=1)
        fig.add_vline(x=mjd_ref, line=dict(color=THEME["ink2"], width=1.5), row=row, col=1)
        fig.update_yaxes(title_text="Gradient (%)", row=row, col=1)
    return _finish_epoch_figure(fig, channels)


def _finish_epoch_figure(fig, channels):
    fig.update_xaxes(title_text="Epoch MJD", tickformat="d", row=len(channels), col=1)
    fig.update_layout(base_layout(200 * len(channels) + 110, margin=dict(l=72, r=16, t=56, b=52)))
    fig.update_layout(legend=dict(x=1, xanchor="right"))
    fig.update_annotations(font=dict(color=THEME["ink2"], size=13), x=0, xanchor="left")
    return style_axes(fig)


def _translucent(hex_color, alpha=0.3):
    value = int(hex_color.lstrip("#"), 16)
    return f"rgba({(value >> 16) & 255},{(value >> 8) & 255},{value & 255},{alpha})"


def _fmt_list(values, fmt):
    return np.array([format(v, fmt) if np.isfinite(v) else "-" for v in np.asarray(values, dtype=float)])


def _wrap(text, width=60):
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return "<br>".join(lines)


def qa_fiberflat_epochs(mjd_ref: int = None, channels: str = "brz", kind: str = "twilight",
                        epochs: Dict[int, Dict] = None, epochs_path: str = None,
                        flats_dir: str = None, drpver: str = None, redux_dir: str = None,
                        output_dir: str = None, bin_width: float = 50.0,
                        telescopes: Tuple[str] = TELESCOPES, max_deviation: float = 0.01,
                        gradient_deg: int = 1, dry_run: bool = False) -> Dict:
    """Compare the fiber flats of all calibration epochs against a reference epoch and write a dashboard.

    For each epoch of the calibration epochs file and each channel with a
    master fiber flat, the flat is median-binned in wavelength
    (:func:`bin_fiberflat`) on the wavelength bins of the reference flat, and
    divided by the binned reference flat. The ratios of the good fibers
    (``fibstatus == 0`` in both flats) of the given ``telescopes`` are
    summarized with :func:`ratio_stats`.

    Each epoch's ratio is also split (:func:`decompose_ratio`) into changes of
    the spectrograph normalization factors, changes of the illumination
    gradient across the IFUs, and the change of the fiber-to-fiber flat field,
    which is summarized in the ``corr_*`` columns.

    The dashboard shows box plots of the ratios (as measured, or after
    removing the spectrograph factors and gradients) against the epoch MJD, the
    spectrograph offsets and science IFU gradient of each epoch, and the flats
    (or their ratios to the reference) of all epochs on the science IFU, using
    the median over wavelength of each fiber.

    Parameters
    ----------
    mjd_ref : int, optional
        MJD of the reference epoch. Default is the latest epoch with flats in
        all ``channels``.
    channels : str, optional
        Channels to compare. Default is 'brz'.
    kind : str, optional
        Kind of fiber flat, 'twilight' or 'dome'. Default is 'twilight'.
    epochs : dict, optional
        Epoch definitions {MJD: definition}. Default is loaded from ``epochs_path``.
    epochs_path : str, optional
        Path of the calibration epochs file. Default is :func:`default_epochs_path`.
    flats_dir : str, optional
        Master calibrations directory. Default is ``$LVM_MASTER_DIR``.
    drpver : str, optional
        Pipeline version whose master fiber flats are compared instead of the
        master calibrations directory.
    redux_dir : str, optional
        Reductions root directory for ``drpver``. Default is ``$LVM_SPECTRO_REDUX``.
    output_dir : str, optional
        Directory of the dashboard and summary table. Default is
        ``fiberflat_qa/{kind}_vs_{mjd_ref}`` in the current directory.
    bin_width : float, optional
        Width of the wavelength bins in Angstroms. Default is 50.
    telescopes : tuple[str], optional
        Telescopes whose fibers enter the ratio statistics. Default is ('Sci', 'SkyE', 'SkyW').
    max_deviation : float, optional
        Deviation of the ratio from 1 counted as off. Default is 0.01.
    gradient_deg : int, optional
        Polynomial degree of the gradient across each IFU in the decomposition
        of the ratios (see :func:`decompose_ratio`). Default is 1.
    dry_run : bool, optional
        If True, log the flats found and the output paths without reading
        flats or writing files. Default is False.

    Returns
    -------
    dict
        - ``"summary"`` : pandas.DataFrame, one row per epoch and channel, with
          the epoch trigger and comment, the flat path, whether it exists, the
          ratio statistics, the decomposition (median level, spectrograph
          offsets, science IFU gradient), the ratio statistics after removing
          them (``corr_*``) and the spectrograph factors from the flat header
          (``hdr_factor*``). None in a dry run.
        - ``"figures"`` : dict, the Plotly figures of the dashboard.
        - ``"report"`` : str, path of the dashboard, None in a dry run.
    """
    epochs = epochs if epochs is not None else load_calibration_epochs(epochs_path)
    paths = {mjd: {channel: fiberflat_path(mjd, channel, kind=kind, flats_dir=flats_dir, drpver=drpver, redux_dir=redux_dir)
                   for channel in channels} for mjd in sorted(epochs)}
    exists = {mjd: {channel: os.path.isfile(p) for channel, p in paths[mjd].items()} for mjd in paths}
    complete = [mjd for mjd in paths if all(exists[mjd].values())]

    if mjd_ref is None:
        if not complete:
            raise FileNotFoundError(f"no epoch has {kind} fiber flats for all channels '{channels}'")
        mjd_ref = complete[-1]
    if mjd_ref not in epochs:
        raise ValueError(f"reference epoch {mjd_ref} not found in the calibration epochs, available: {sorted(epochs)}")
    if mjd_ref not in complete:
        raise FileNotFoundError(f"reference epoch {mjd_ref} is missing {kind} fiber flats: "
                                f"{[paths[mjd_ref][c] for c in channels if not exists[mjd_ref][c]]}")

    output_dir = output_dir or os.path.join(os.getcwd(), "fiberflat_qa", f"{kind}_vs_{mjd_ref}")
    report_path = os.path.join(output_dir, f"fiberflat-epochs-qa_{kind}_vs_{mjd_ref}.html")
    table_path = os.path.join(output_dir, f"fiberflat-epochs-qa_{kind}_vs_{mjd_ref}.csv")
    missing = {channel: [mjd for mjd in paths if not exists[mjd][channel]] for channel in channels}
    log.info(f"comparing {kind} fiber flats of {len(epochs)} calibration epochs against reference epoch {mjd_ref}")
    for channel in channels:
        log.info(f"  {channel}: {len(epochs) - len(missing[channel])} flats, missing for epochs {missing[channel]}")
    if dry_run:
        log.info(f"dry run: dashboard would be written to {report_path}")
        return {"summary": None, "figures": {}, "report": None}

    rows, ifu_values, geometry = [], {}, None
    for channel in channels:
        reference = RSS.from_file(paths[mjd_ref][channel])
        edges = wavelength_edges(reference, bin_width=bin_width)
        ref_binned = bin_fiberflat(reference, edges)
        ref_fibstatus = np.asarray(reference._slitmap["fibstatus"]) if "fibstatus" in reference._slitmap.colnames else np.zeros(ref_binned.shape[0])
        select = np.isin(_decode(reference._slitmap["telescope"]), telescopes)
        if geometry is None:
            geometry = _ifu_geometry(reference._slitmap)
        ifu_values[channel] = {}

        for mjd in tqdm(sorted(epochs), desc=f"{channel} fiber flats", unit="epoch", ascii=True):
            epoch = epochs[mjd]
            row = {"epoch": mjd, "channel": channel, "reference": mjd == mjd_ref, "trigger": epoch.get("trigger"),
                   "comment": epoch.get("comment"), "twilight_mjds": str((epoch.get("flavors") or {}).get(kind, "")),
                   "path": paths[mjd][channel], "exists": exists[mjd][channel]}
            stats = ratio_stats(np.array([]))
            if exists[mjd][channel]:
                try:
                    flat = reference if mjd == mjd_ref else RSS.from_file(paths[mjd][channel])
                    binned = ref_binned if mjd == mjd_ref else bin_fiberflat(flat, edges)
                    fibstatus = np.asarray(flat._slitmap["fibstatus"]) if "fibstatus" in flat._slitmap.colnames else np.zeros(binned.shape[0])
                    good = select & (fibstatus == 0) & (ref_fibstatus == 0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ratio = binned / ref_binned
                    level = bn.nanmedian(binned, axis=1)
                    fiber_ratio = bn.nanmedian(ratio, axis=1)
                    level[fibstatus != 0], fiber_ratio[(fibstatus != 0) | (ref_fibstatus != 0)] = np.nan, np.nan
                    # the reference ratio to itself is 1 by definition (NaN for flagged fibers)
                    fiber_corr = np.where(np.isfinite(fiber_ratio), 1.0, np.nan) if mjd == mjd_ref else np.full_like(fiber_ratio, np.nan)
                    row.update(_header_factors(getattr(flat, "_header", None), channel))
                    if mjd != mjd_ref:
                        stats = ratio_stats(ratio[good], max_deviation=max_deviation)
                        parts = decompose_ratio(flat._slitmap, fiber_ratio, good, deg=gradient_deg, telescopes=telescopes)
                        scale = parts["level"] * parts["model"]
                        with np.errstate(invalid="ignore", divide="ignore"):
                            fiber_corr = fiber_ratio / scale
                            corr_stats = ratio_stats((ratio / scale[:, None])[good], max_deviation=max_deviation)
                        row.update({f"corr_{key}": value for key, value in corr_stats.items()})
                        row["level"] = parts["level"]
                        row.update({f"offset_sp{i}": parts["offsets"].get(i, np.nan) for i in (1, 2, 3)})
                        sci = parts["gradients"].get("Sci", {})
                        row.update({"grad_x": sci.get("gx", np.nan), "grad_y": sci.get("gy", np.nan), "grad_amp": sci.get("amp", np.nan)})
                    ifu_values[channel][mjd] = {"flat": level[geometry["select"]], "ratio": fiber_ratio[geometry["select"]],
                                                "corr": fiber_corr[geometry["select"]]}
                except Exception as e:
                    log.error(f"failed to compare {paths[mjd][channel]}: {type(e).__name__}: {e}")
                    row["error"] = f"{type(e).__name__}: {e}"
            row.update(stats)
            rows.append(row)

    summary = pd.DataFrame(rows)
    for column in ["error", "level", "offset_sp1", "offset_sp2", "offset_sp3", "grad_x", "grad_y", "grad_amp",
                   "hdr_factor1", "hdr_factor2", "hdr_factor3"] + [f"corr_{key}" for key in ratio_stats(np.array([]))]:
        if column not in summary.columns:
            summary[column] = None if column == "error" else np.nan
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(table_path, index=False)

    figures = {"ratios|raw": figure_ratio_boxes(summary, mjd_ref, channels, missing, max_deviation=max_deviation),
               "ratios|corr": figure_ratio_boxes(summary, mjd_ref, channels, missing, max_deviation=max_deviation, prefix="corr_"),
               "offsets": figure_offsets(summary, mjd_ref, channels), "gradients": figure_gradients(summary, mjd_ref, channels)}
    _write_fiberflat_dashboard(report_path, summary, figures, epochs, geometry, ifu_values, mjd_ref, channels, kind,
                               missing, bin_width, telescopes, max_deviation, flats_dir, drpver)
    log.info(f"written fiber flat epochs dashboard to {report_path}")
    return {"summary": summary, "figures": figures, "report": report_path}


def _ifu_payload(epochs, geometry, ifu_values, mjd_ref, channels, kind):
    """Data of the IFU small multiples, as a JSON-serializable dictionary."""
    scales = {}
    for channel in channels:
        for mode, floor in (("flat", 0.02), ("ratio", 0.005), ("corr", 0.002)):
            values = np.concatenate([v[mode] for mjd, v in ifu_values[channel].items() if not (mode != "flat" and mjd == mjd_ref)] or [np.array([np.nan])])
            values = values[np.isfinite(values)]
            half = float(np.percentile(np.abs(values - 1), 98)) if values.size else floor
            scales[f"{channel}|{mode}"] = max(round(half, 4), floor)
    return {
        "x": geometry["x"], "y": geometry["y"], "fiberid": geometry["fiberid"], "label": geometry["label"],
        "hexRadius": geometry["hex_radius"], "hexRotation": geometry["hex_rotation"],
        "reference": mjd_ref, "kind": kind,
        "epochs": [{"mjd": mjd, "trigger": _short_trigger(epochs[mjd].get("trigger")),
                    "routine": epochs[mjd].get("trigger") == ROUTINE_TRIGGER} for mjd in sorted(epochs)],
        "values": {channel: {str(mjd): {mode: _encode(v[mode]) for mode in ("flat", "ratio", "corr")} for mjd, v in ifu_values[channel].items()}
                   for channel in channels},
        "scales": scales, "diverging": DIVERGING, "divergingDark": DIVERGING_DARK,
    }


def _write_fiberflat_dashboard(report_path, summary, figures, epochs, geometry, ifu_values, mjd_ref, channels, kind,
                               missing, bin_width, telescopes, max_deviation, flats_dir, drpver):
    compared = summary[(summary["nvalues"] > 0)]
    nflats = len(set(summary.loc[summary["exists"], "epoch"]))
    ref_trigger = _short_trigger(epochs[mjd_ref].get("trigger"))

    tiles = [("Epochs with flats", f"{nflats}", f"of {len(epochs)} in the epochs file"),
             ("Reference epoch", f"{mjd_ref}", ref_trigger)]
    for channel in channels:
        sel = compared[compared["channel"] == channel]
        if len(sel) and np.isfinite(sel["corr_sigma"].astype(float)).any():
            worst = sel.loc[sel["corr_sigma"].astype(float).idxmax()]
            tiles.append((f"{channel} fiber-level change", f"{100 * np.nanmedian(sel['corr_sigma']):.2f}%",
                          (f"median robust σ ({100 * np.nanmedian(sel['sigma']):.1f}% as measured); "
                           f"largest {100 * worst['corr_sigma']:.2f}% at {int(worst['epoch'])}")))

    source = f"pipeline version {drpver}" if drpver else (flats_dir or os.getenv("LVM_MASTER_DIR", "."))
    meta = [("Fiber flats", f"lvm-mfiberflat_{kind}"), ("Source", source), ("Epochs file", os.path.basename(default_epochs_path())),
            ("Reference", f"{mjd_ref} ({ref_trigger})"), ("Wavelength bins", f"{bin_width:g} Å"),
            ("Telescopes", ", ".join(telescopes)), ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]
    missing_all = sorted(set(mjd for channel in channels for mjd in missing[channel]))
    if missing_all:
        meta.append(("Epochs without flats", ", ".join(map(str, missing_all))))

    intro = (f"<p>The master {kind} fiber flat of every calibration epoch is compared with the flat of the reference "
             f"epoch {mjd_ref}. Each fiber is resampled to a common wavelength grid and median-binned in {bin_width:g} &Aring; "
             "bins, because the wavelength solution changes between epochs. A stable fiber-to-fiber throughput gives ratios "
             "close to 1 with a small spread. Epochs, triggers and comments come from the calibration epochs file.</p>")

    table_rows = []
    for _, r in summary.sort_values(["epoch", "channel"]).iterrows():
        status = "reference" if r["reference"] else ("no flat" if not r["exists"] else ("error" if r.get("error") else ""))
        table_rows.append([
            str(int(r["epoch"])), escape(_short_trigger(r["trigger"])), escape(r["channel"]), escape(status),
            f"{int(r['nvalues']):,}" if r["nvalues"] else "-", _num(r["median"], ".4f"),
            f'{_num(r["p25"], ".4f")} &ndash; {_num(r["p75"], ".4f")}', f'{_num(r["p2.5"], ".4f")} &ndash; {_num(r["p97.5"], ".4f")}',
            _num(100 * r["sigma"], ".2f"), _num(100 * r["frac_off"], ".1f"), _num(100 * r["corr_sigma"], ".2f"),
            _num(100 * r["corr_frac_off"], ".1f"), " / ".join(_num(100 * r[f"offset_sp{i}"], "+.2f") for i in (1, 2, 3)),
            _num(100 * r["grad_amp"], ".2f"), " / ".join(_num(r[f"hdr_factor{i}"], ".4f") for i in (1, 2, 3)),
            f'<span class="range">{escape(str(r["comment"])) if isinstance(r["comment"], str) else ""}</span>'])
    table = html_table(
        [("Epoch", "MJD of the calibration epoch"), ("Trigger", ""), ("Ch", "channel"), ("Status", ""),
         ("Values", "fibers x wavelength bins compared"), ("Median", "median ratio"), ("IQR", "25th to 75th percentile"),
         ("2.5–97.5%", "box whiskers"), ("Robust σ (%)", "1.4826 times the median absolute deviation"),
         (f"Off by >{100 * max_deviation:g}% (%)", ""),
         ("Fiber-level σ (%)", "robust sigma after removing spectrograph offsets and IFU gradients"),
         (f"Fiber-level off >{100 * max_deviation:g}% (%)", ""),
         ("sp1 / sp2 / sp3 (%)", "spectrograph offsets relative to the reference"),
         ("Sci gradient (%)", "half peak-to-valley of the gradient across the science IFU"),
         ("Header factors sp1 / sp2 / sp3", "spectrograph factors applied to this flat (FIBERFLAT FACTOR keywords)"),
         ("Comment", "from the calibration epochs file")],
        table_rows, numeric=[True, False, False, False, True, True, True, True, True, True, True, True, True, True, True, False])

    payload = json.dumps(_ifu_payload(epochs, geometry, ifu_values, mjd_ref, channels, kind)).replace("</", "<\\/")
    channel_options = "".join(f'<option value="{c}">{c}</option>' for c in channels)
    body = (BODY_TEMPLATE
            .replace("@@REF@@", str(mjd_ref)).replace("@@KIND@@", escape(kind)).replace("@@TABLE@@", table)
            .replace("@@CHANNEL_OPTIONS@@", channel_options).replace("@@MAXDEV@@", f"{max_deviation:g}")
            .replace("@@BINWIDTH@@", f"{bin_width:g}").replace("@@PAYLOAD@@", payload)
            .replace("@@RATIOS_PICKER@@", picker("ratios-chart", "ratios", [("Ratios", [
                ("corr", "fiber level: spectrograph factors and IFU gradients removed"), ("raw", "as measured")])], 260 * len(channels) + 110)))
    write_dashboard(report_path, title="LVM Fiber Flat Epochs", eyebrow=f"LVM DRP · fiber flats · {kind}",
                    intro=intro, meta=meta, tiles=tiles, body=body, figures=figures, generator="lvmdrp.qa.fiberflats.qa_fiberflat_epochs")
    return report_path


def _num(value, fmt):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return format(value, fmt) if np.isfinite(value) else "-"


# sections of the dashboard; @@TOKENS@@ are filled in by _write_fiberflat_dashboard. The
# script must stay ASCII: the page writer turns other characters into HTML entities
BODY_TEMPLATE = r"""
<section>
  <h2>Flat-field ratios across epochs</h2>
  <p>Each box summarizes the ratio of an epoch's flat to the reference flat (MJD @@REF@@, vertical line) over all good
  fibers and wavelength bins: the box spans the 25th to 75th percentiles, the line inside is the median and the whiskers
  reach the 2.5th and 97.5th percentiles. Orange boxes are epochs triggered by something other than normal operations;
  hover a box for its trigger and comment.</p>
  <p>As measured, the ratios are dominated by large-scale changes: the spectrograph normalization factors and the
  illumination gradient across the IFUs, shown in the next chart. The fiber-level view divides those out, leaving the
  change of the fiber-to-fiber flat field.</p>
  @@RATIOS_PICKER@@
  <details><summary>Table view: ratio statistics per epoch and channel</summary>@@TABLE@@</details>
</section>

<section>
  <h2>Spectrograph offsets</h2>
  <p>Large-scale part of each epoch's ratio to the reference: the flux offset of each spectrograph's fibers. A flat whose
  spectrograph normalization factors were changed (FIBERFLAT FACTOR keywords, listed in the table view above) shifts all
  three offsets together.</p>
  <div class="chart" data-fig="offsets" role="img" aria-label="Spectrograph offsets versus epoch MJD"></div>
</section>

<section>
  <h2>Gradient across the science IFU</h2>
  <p>The other large-scale part: the change of the ratio from the center of the science IFU to its edge, along the IFU
  axes. A gradient that appears in every epoch belongs to the reference flat.</p>
  <div class="chart" data-fig="gradients" role="img" aria-label="Gradient across the science IFU versus epoch MJD"></div>
</section>

<section>
  <h2>Fiber flats on the science IFU</h2>
  <p>Each map is one epoch's fiber flat on the science IFU, as the median over wavelength of every fiber, or its ratio to
  the reference flat. The color scale is centered on 1. Gray fibers are flagged in the slitmap or have no valid
  pixels. Hover a fiber for its value.</p>
  <div class="ifu-controls">
    <label>Channel <select id="ifu-channel">@@CHANNEL_OPTIONS@@</select></label>
    <label>Show <select id="ifu-mode"><option value="corr">Fiber-level ratio to @@REF@@</option><option value="ratio">Ratio to @@REF@@ as measured</option><option value="flat">Fiber flat</option></select></label>
    <div class="ifu-legend" aria-hidden="true"><span id="ifu-lo"></span><canvas id="ifu-bar" width="160" height="10"></canvas><span id="ifu-hi"></span></div>
  </div>
  <div class="ifu-wrap">
    <div class="ifu-grid" id="ifu-grid"></div>
    <div class="ifu-tip" id="ifu-tip" hidden></div>
  </div>
</section>

<section id="definitions">
  <h2>How the quantities are computed</h2>
  <div class="defs">
    <div class="def">
      <h3>Binned fiber flat</h3>
      <p>The flat \(f_i(\lambda)\) of fiber \(i\) is resampled on a common grid and median-binned in bins \(B_k\) of
      @@BINWIDTH@@ &Aring;, ignoring masked pixels. The bins are those of the reference flat.</p>
      <div class="eq">\[ F_{i,k} = \mathrm{med}_{\lambda \in B_k}\, f_i(\lambda) \]</div>
    </div>
    <div class="def">
      <h3>Ratio to the reference</h3>
      <p>For epoch \(e\) and the reference epoch \(e_0\), using the good fibers of both flats.</p>
      <div class="eq">\[ R^{e}_{i,k} = \frac{F^{e}_{i,k}}{F^{e_0}_{i,k}} \]</div>
    </div>
    <div class="def">
      <h3>Box plot and scatter</h3>
      <p>Percentiles \(P_q\) of all \(R^{e}_{i,k}\) of an epoch; the robust scatter is 1.4826 times the median absolute
      deviation, and \(q\) is the fraction off by more than \(\delta = @@MAXDEV@@\).</p>
      <div class="eq">\[ \sigma_e = 1.4826\,\mathrm{med}\left|R^{e}_{i,k} - \mathrm{med}\,R^{e}\right|, \qquad q_e = \frac{1}{N}\sum_{i,k} \mathbf{1}\left[\left|R^{e}_{i,k} - 1\right| > \delta\right] \]</div>
    </div>
    <div class="def">
      <h3>Large-scale components</h3>
      <p>The per-fiber ratio \(\bar R^{e}_{i}\), divided by its median \(m_e\), is fitted with an offset \(o_s\) per
      spectrograph and an intercept and gradient per telescope \(T\), with \(\hat x_i, \hat y_i\) the fiber positions in
      units of the IFU radius (iterative 4&sigma; clipping).</p>
      <div class="eq">\[ \frac{\bar R^{e}_{i}}{m_e} - 1 = a_{T(i)} + g^{x}_{T(i)}\,\hat x_i + g^{y}_{T(i)}\,\hat y_i + o_{s(i)} \]</div>
    </div>
    <div class="def">
      <h3>Fiber-level ratio</h3>
      <p>The ratio with the large-scale model \(M_i = 1 + a_{T(i)} + g^{x}_{T(i)}\hat x_i + g^{y}_{T(i)}\hat y_i + o_{s(i)}\)
      divided out: what changed in the fiber-to-fiber flat field.</p>
      <div class="eq">\[ \tilde R^{e}_{i,k} = \frac{R^{e}_{i,k}}{m_e\, M_i} \]</div>
    </div>
    <div class="def">
      <h3>IFU maps</h3>
      <p>One value per fiber: the median over wavelength bins of the flat, or of its ratio to the reference.</p>
      <div class="eq">\[ \bar F^{e}_{i} = \mathrm{med}_k\, F^{e}_{i,k}, \qquad \bar R^{e}_{i} = \mathrm{med}_k\, R^{e}_{i,k} \]</div>
    </div>
  </div>
</section>

<style>
.ifu-controls { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; font-size: 13px; color: var(--ink-2); }
.ifu-controls label { display: inline-flex; gap: 8px; align-items: center; }
.ifu-controls select { font: 13px var(--sans); color: var(--ink); background: var(--surface); border: 1px solid var(--axis); border-radius: 6px; padding: 4px 8px; }
.ifu-controls select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.ifu-legend { display: inline-flex; gap: 8px; align-items: center; font: 12px var(--mono); color: var(--ink-2); font-variant-numeric: tabular-nums; }
.ifu-legend canvas { width: 160px; height: 10px; border-radius: 2px; }
.ifu-wrap { position: relative; }
.ifu-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.ifu-tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px; display: grid; gap: 6px; align-content: start; min-width: 0; }
.ifu-tile.ref { border-color: var(--ink-2); }
.ifu-tile canvas { width: 100%; aspect-ratio: 1; display: block; cursor: crosshair; }
.ifu-cap { display: grid; gap: 0; font-size: 12px; line-height: 1.35; }
.ifu-cap b { font: 500 13px var(--mono); color: var(--ink); }
.ifu-cap span { color: var(--muted); }
.ifu-tip { position: absolute; z-index: 5; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--axis);
  border-radius: 6px; padding: 6px 8px; font: 12px/1.4 var(--mono); white-space: nowrap; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
</style>

<script>
(function () {
  const D = @@PAYLOAD@@;
  const grid = document.getElementById("ifu-grid");
  const tip = document.getElementById("ifu-tip");
  const channelSelect = document.getElementById("ifu-channel");
  const modeSelect = document.getElementById("ifu-mode");
  const bar = document.getElementById("ifu-bar");
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const n = D.x.length;

  function isDark() {
    const theme = document.documentElement.getAttribute("data-theme");
    return theme ? theme === "dark" : media.matches;
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function hexToRgb(h) { const v = parseInt(h.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }

  // diverging scale centered on 1, linear in RGB between the stops
  function colorScale(half) {
    const stops = (isDark() ? D.divergingDark : D.diverging).map(hexToRgb);
    return function (value) {
      let t = (value - 1) / half;
      t = Math.max(-1, Math.min(1, t));
      const pos = (t + 1) / 2 * (stops.length - 1);
      const i = Math.min(Math.floor(pos), stops.length - 2), f = pos - i;
      const a = stops[i], b = stops[i + 1];
      return "rgb(" + [0, 1, 2].map((j) => Math.round(a[j] + f * (b[j] - a[j]))).join(",") + ")";
    };
  }

  function state() {
    const channel = channelSelect.value, mode = modeSelect.value;
    return { channel: channel, mode: mode, half: D.scales[channel + "|" + mode], values: D.values[channel] || {} };
  }

  function drawLegend(s) {
    const ctx = bar.getContext("2d"), color = colorScale(s.half);
    for (let px = 0; px < bar.width; px++) {
      ctx.fillStyle = color(1 - s.half + 2 * s.half * px / (bar.width - 1));
      ctx.fillRect(px, 0, 1, bar.height);
    }
    document.getElementById("ifu-lo").textContent = (1 - s.half).toFixed(3);
    document.getElementById("ifu-hi").textContent = (1 + s.half).toFixed(3);
  }

  function hexPath(ctx, cx, cy, r) {
    ctx.beginPath();
    for (let k = 0; k < 6; k++) {
      const a = (D.hexRotation + 60 * k) * Math.PI / 180;
      const px = cx + r * Math.cos(a), py = cy - r * Math.sin(a);
      if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
  }

  function drawTile(canvas, encoded, s) {
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.clientWidth;
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    const scale = (size / 2 - 2) / (1 + D.hexRadius);
    canvas._scale = scale;
    const color = colorScale(s.half), missing = css("--grid");
    for (let i = 0; i < n; i++) {
      const v = encoded[i];
      hexPath(ctx, size / 2 + D.x[i] * scale, size / 2 - D.y[i] * scale, D.hexRadius * scale);
      ctx.fillStyle = v === null ? missing : color(v / 1e4);
      ctx.fill();
    }
  }

  function render() {
    const s = state();
    drawLegend(s);
    grid.innerHTML = "";
    D.epochs.forEach((epoch) => {
      const data = s.values[String(epoch.mjd)];
      if (!data) return;
      const tile = document.createElement("div");
      const isRef = epoch.mjd === D.reference;
      tile.className = "ifu-tile" + (isRef ? " ref" : "");
      const canvas = document.createElement("canvas");
      canvas.setAttribute("role", "img");
      const encoded = data[s.mode];
      const finite = encoded.filter((v) => v !== null).map((v) => v / 1e4).sort((a, b) => a - b);
      const median = finite.length ? finite[Math.floor(finite.length / 2)] : NaN;
      canvas.setAttribute("aria-label", "MJD " + epoch.mjd + ", " + s.channel + " channel, median " + median.toFixed(4));
      canvas._encoded = encoded;
      canvas._mjd = epoch.mjd;
      const cap = document.createElement("div");
      cap.className = "ifu-cap";
      const title = document.createElement("b");
      title.textContent = epoch.mjd + (isRef ? " \u00b7 reference" : "");
      const note = document.createElement("span");
      note.textContent = epoch.trigger + " \u00b7 med " + (isFinite(median) ? median.toFixed(3) : "-");
      cap.appendChild(title);
      cap.appendChild(note);
      tile.appendChild(canvas);
      tile.appendChild(cap);
      grid.appendChild(tile);
      drawTile(canvas, encoded, s);
    });
  }

  grid.addEventListener("pointermove", (event) => {
    const canvas = event.target;
    if (canvas.tagName !== "CANVAS" || !canvas._encoded) { tip.hidden = true; return; }
    const rect = canvas.getBoundingClientRect(), size = rect.width, scale = canvas._scale;
    const x = (event.clientX - rect.left - size / 2) / scale, y = -(event.clientY - rect.top - size / 2) / scale;
    let best = -1, bestDist = Infinity;
    for (let i = 0; i < n; i++) {
      const d = (D.x[i] - x) * (D.x[i] - x) + (D.y[i] - y) * (D.y[i] - y);
      if (d < bestDist) { bestDist = d; best = i; }
    }
    if (best < 0 || Math.sqrt(bestDist) > 1.2 * D.hexRadius) { tip.hidden = true; return; }
    const v = canvas._encoded[best];
    tip.textContent = "";
    const line1 = document.createElement("div"), line2 = document.createElement("div");
    line1.textContent = "MJD " + canvas._mjd + " \u00b7 " + D.label[best] + " (fiber " + D.fiberid[best] + ")";
    line2.textContent = ({ corr: "fiber-level ratio ", ratio: "ratio ", flat: "flat " })[modeSelect.value] + (v === null ? "no data" : (v / 1e4).toFixed(4));
    tip.appendChild(line1);
    tip.appendChild(line2);
    tip.hidden = false;
    const wrap = grid.parentElement.getBoundingClientRect();
    let left = event.clientX - wrap.left + 12;
    if (left + tip.offsetWidth > wrap.width) left = event.clientX - wrap.left - tip.offsetWidth - 12;
    tip.style.left = left + "px";
    tip.style.top = (event.clientY - wrap.top + 12) + "px";
  });
  grid.addEventListener("pointerleave", () => { tip.hidden = true; });

  channelSelect.addEventListener("change", render);
  modeSelect.addEventListener("change", render);
  media.addEventListener("change", render);
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  let width = grid.clientWidth;
  new ResizeObserver(() => { if (grid.clientWidth !== width) { width = grid.clientWidth; render(); } }).observe(grid);
  render();
})();
</script>
"""
