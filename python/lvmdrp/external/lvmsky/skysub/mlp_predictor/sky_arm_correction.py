"""Correct a predicted science sky by the sky arms' own decomposition residual.

Why this works
--------------
The decomposition leaves a residual in every arm that is mostly NOT noise. In
the blue it is a fixed pattern of solar absorption lines (Balmer series, Ca H&K,
the G band) that the moon and zodi basis does not reproduce: on bright-continuum
rows the science arm's fractional residual correlates at 0.98 with one fixed
template, with an RMS of ~10% of the continuum. The science prediction inherits
that residual, because the ML reproduces the science decomposition's
coefficients and not the science spectrum. Both sky arms carry the same pattern
at the same time, so their residual is a direct, simultaneous measurement of
what the prediction is missing.

The estimator
-------------
For each sky arm ``a`` (near, far):

    e_a(lambda) = [obs_a - model_a](lambda) * k_a(lambda)
    k_a         = M_sci(lambda) / M_a(lambda)       (scale='model', the default)

``M_sci`` is the PREDICTED science sky model and ``M_a`` the arm's own model,
both total (every family), taken pixel by pixel and clipped to [0.2, 5]. The
residual at a pixel scales with whatever light makes up that pixel: sunlight in
the blue continuum, airglow in the diffuse, OH in a line core. The total-model
ratio follows each of them, so it needs no choice of family. Measured on 457
uncrowded held-out rows (full-band correction, median chi2):

    k = 1            0.287      moon+zodi ratio          0.276
    model ratio      0.269      on OH-line pixels: model 0.848, moon+zodi 0.934

In the blue the model ratio and the moon+zodi ratio are equivalent (0.148 both).
``scale='solar'`` keeps the moon+zodi ratio, smoothed over ``ratio_smoothing``
Angstrom. The arms are then combined with inverse-variance weights, using each
arm's photon-noise variance times ``k_a**2``.

When ``max_wavelength`` cuts the band short, the correction is applied in full
up to it and then tapered to zero with a cosine over the next ``taper_width``
Angstrom, so that no step is left in the corrected spectrum. The ramp sits
OUTSIDE the requested range on purpose: a ramp ending at the boundary halves
the correction inside it (4900-5000 A: chi2 0.079 -> 0.251), while the
correction just redward is as good as anywhere (5000-5100 A: 0.354 -> 0.078
uncorrected -> corrected), so the ramp costs nothing there.

Two safeguards, both measured as necessary on held-out rows:

* An arm whose decomposition failed (blue fractional residual RMS above
  ``gate``; the median is 0.064, failures sit near 1.0) is dropped. Without
  this gate one bad far-arm fit raises the mean chi2 by x40.
* On faint rows the arms' noise competes with the pattern. A 3-pixel running
  median helps there, but costs on bright rows (x0.007 -> x0.06). So
  ``smoothing='auto'`` smooths only rows whose solar-continuum S/N is below
  ``auto_snr``.

Measured on the held-out split of the palacecorr corpus (uncrowded fields, test
rows, blue < 5000 A, median single-fibre photon chi2): uncorrected ML 1.140;
near arm only and unsmoothed 0.236; near arm with a 10 A running median 0.915;
this estimator 0.146 (with moon+zodi scaling; the model ratio gives the same
in the blue). The decomposition's own floor is 0.966.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

__all__ = ["SkyArm", "arm_photon_variance", "sky_arm_residual_correction"]


@dataclass
class SkyArm:
    """One sky arm's inputs; every array is (n_wave,) or (n_rows, n_wave).

    observed  the arm's observed spectrum.
    model     the arm's reconstruction from its OWN decomposition coefficients.
    solar     the solar-continuum part of that model: moon + zodi.
    variance  the photon-noise variance of ``observed`` (see
              :func:`arm_photon_variance`).
    """
    observed: np.ndarray
    model: np.ndarray
    solar: np.ndarray
    variance: np.ndarray


def arm_photon_variance(flux, wave, n_fibres=None, exptime=900.0, sens=None):
    """Photon variance of a sky-arm median stack, with the shared floor.

    ``n_fibres`` is META's ``fibers_sky_{near,far}_used``. Without it the
    variance is the single-fibre one, which over-states the arm's noise by the
    ~50 stacked fibres and so under-weights the arm.  ``sens`` is the
    absolute sensitivity on ``wave``; pass it when calling per row to avoid
    re-reading the sensitivity tables.
    """
    from mlp_predictor.noise import (floor_variance, load_absolute_sensitivity,
                                     photon_variance_absolute)
    wave = np.asarray(wave, dtype=np.float64)
    flux2 = np.atleast_2d(np.asarray(flux, dtype=np.float64))
    sens = np.asarray(load_absolute_sensitivity(wave) if sens is None else sens,
                      dtype=np.float64)
    nf = None if n_fibres is None else np.atleast_1d(np.asarray(n_fibres, dtype=np.float64))
    var = floor_variance(photon_variance_absolute(
        np.clip(np.nan_to_num(flux2), 0.0, None), sens, exptime=float(exptime),
        dwave=float(np.median(np.diff(wave))), n_fibres=nf))
    return var if np.ndim(flux) > 1 else var[0]


def _width_px(wave, width_a):
    return max(3, int(round(float(width_a) / float(np.median(np.diff(wave))))) | 1)


def _running_median(x, wave, width_a):
    return median_filter(x, size=(1, _width_px(wave, width_a)), mode="nearest")


def _running_mean(x, wave, width_a):
    return uniform_filter1d(x, _width_px(wave, width_a), axis=-1, mode="nearest")


def _taper(wave, max_wavelength, taper_width):
    """1 up to ``max``, cosine down to 0 at ``max + width``, 0 beyond.

    All ones when ``max_wavelength`` covers the whole grid.
    """
    lmax = float(max_wavelength)
    if lmax >= wave.max():
        return np.ones_like(wave)
    w = max(float(taper_width), 0.0)
    if w == 0.0:
        return (wave < lmax).astype(np.float64)
    x = np.clip((lmax + w - wave) / w, 0.0, 1.0)    # 1 at the boundary, 0 a width out
    return np.sin(0.5 * np.pi * x) ** 2


def sky_arm_residual_correction(
    wave,
    sci_model,
    sci_solar,
    arms: Sequence[SkyArm],
    *,
    scale: Optional[str] = "model",
    max_wavelength: float = 5000.0,
    taper_width: float = 100.0,
    smoothing: Union[str, float, None] = "auto",
    auto_snr: float = 100.0,
    auto_width: float = 1.5,
    gate: float = 0.25,
    gate_max_wavelength: float = 5000.0,
    ratio_smoothing: float = 100.0,
    return_info: bool = False,
):
    """Correction to SUBTRACT from (science observed - predicted science sky).

    Equivalently, ADD it to the predicted sky. The returned array has the shape
    of ``sci_model``.

    Parameters
    ----------
    wave : (n_wave,) wavelength grid in Angstrom, shared by every array.
    sci_model : the PREDICTED science sky, total (every family). It sets
        ``k_a`` when ``scale='model'``.
    sci_solar : the PREDICTED science moon + zodi. It sets the row's S/N for
        ``smoothing='auto'``, and ``k_a`` when ``scale='solar'``. Using the
        decomposition's science instead of the prediction changes the result by
        under 1%, so the prediction's errors do not matter here.
    arms : the sky arms to combine, usually ``(near, far)``. One arm works too.
    scale : ``'model'`` (total-model ratio per pixel), ``'solar'`` (smoothed
        moon + zodi ratio), or None (each arm's residual as it stands).
    max_wavelength : the correction is applied in full up to here. The full
        band also helps (median chi2 1.27 -> 0.27 on held-out rows); pass
        ``np.inf`` for it.
    taper_width : width in Angstrom of the cosine taper that starts at
        ``max_wavelength`` and reaches zero ``taper_width`` redward of it.
        Unused when ``max_wavelength`` covers the grid.
    smoothing : ``'auto'`` (3-pixel median below ``auto_snr``), a width in
        Angstrom (running median on every row), or 0 / None (unsmoothed).
    gate : drop an arm whose blue fractional residual RMS,
        ``|obs - model| / solar`` below ``gate_max_wavelength``, exceeds this.
    return_info : also return a dict with per-arm usage, row S/N, smoothed rows.
    """
    wave = np.asarray(wave, dtype=np.float64)
    one_row = np.ndim(sci_model) == 1
    m_sci = np.atleast_2d(np.asarray(sci_model, dtype=np.float64))
    n_rows, n_wave = m_sci.shape
    s_sci = np.broadcast_to(np.atleast_2d(np.asarray(sci_solar, dtype=np.float64)),
                            (n_rows, n_wave))
    if wave.shape != (n_wave,):
        raise ValueError(f"wave {wave.shape} does not match sci_model {m_sci.shape}")
    if not arms:
        raise ValueError("need at least one sky arm")
    if scale not in ("model", "solar", None):
        raise ValueError(f"scale must be 'model', 'solar' or None; got {scale!r}")
    gate_band = wave < float(gate_max_wavelength)
    if scale == "solar":
        s_sci_smooth = _running_mean(np.nan_to_num(s_sci), wave, ratio_smoothing)

    num = np.zeros((n_rows, n_wave))
    den = np.zeros((n_rows, n_wave))
    used, frac_rms = [], []
    for arm in arms:
        obs, mod, sol, var = (np.broadcast_to(np.atleast_2d(np.asarray(x, dtype=np.float64)),
                                              (n_rows, n_wave))
                              for x in (arm.observed, arm.model, arm.solar, arm.variance))
        resid = obs - mod
        with np.errstate(divide="ignore", invalid="ignore"):
            fr = np.sqrt(np.nanmedian(np.where(gate_band, (resid / sol) ** 2, np.nan), axis=1))
            if scale == "model":
                k = m_sci / mod
                k = np.where(np.isfinite(k) & (m_sci > 0) & (mod > 0),
                             np.clip(k, 0.2, 5.0), 1.0)
            elif scale == "solar":
                sol_smooth = _running_mean(np.nan_to_num(sol), wave, ratio_smoothing)
                k = np.where(sol_smooth > 0, s_sci_smooth / sol_smooth, np.nan)
            else:
                k = np.ones_like(resid)
            ok_row = np.isfinite(fr) & (fr < float(gate))
            ok = ok_row[:, None] & np.isfinite(resid) & np.isfinite(k) & np.isfinite(var) & (var > 0)
            w = np.where(ok, 1.0 / np.where(ok, var * k ** 2, 1.0), 0.0)
        num += w * np.where(ok, resid * k, 0.0)
        den += w
        used.append(ok_row)
        frac_rms.append(fr)

    corr = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    with np.errstate(invalid="ignore"):
        snr = np.nanmedian(np.where(gate_band, s_sci * np.sqrt(den), np.nan), axis=1)
    smoothed = np.zeros(n_rows, dtype=bool)
    if isinstance(smoothing, str):
        if smoothing != "auto":
            raise ValueError(f"smoothing must be 'auto', a width in A, or None; got {smoothing!r}")
        smoothed = ~(snr >= float(auto_snr))       # NaN S/N (no usable arm) counts as faint
        if smoothed.any():
            corr[smoothed] = _running_median(corr[smoothed], wave, auto_width)
    elif smoothing:
        smoothed[:] = True
        corr = _running_median(corr, wave, float(smoothing))
    corr = corr * _taper(wave, max_wavelength, taper_width)[None, :]

    out = corr[0] if one_row else corr
    if not return_info:
        return out
    info = dict(arm_used=np.stack(used, axis=0), arm_frac_rms=np.stack(frac_rms, axis=0),
                snr=snr, smoothed=smoothed)
    if one_row:
        info = {k: (v[:, 0] if v.ndim == 2 else v[0]) for k, v in info.items()}
    return out, info
