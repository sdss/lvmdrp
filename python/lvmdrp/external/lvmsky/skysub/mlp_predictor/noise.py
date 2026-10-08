"""Photon-noise model for the per-pixel flux-space loss.

The spectra carry no per-pixel error array, so until now the flux-space term in
``trainer.compressed_loss`` weighted every wavelength equally.  That is wrong by
a large factor: the instrument throughput varies by 4.3x across the b channel
alone, so a fixed flux error is a very different number of photons at 3600 A
than at 5000 A, and the unweighted loss was fitting the poorly-measured blue
end with the same authority as the well-measured middle.

The model
---------
``mean-sens-{b,r,z}-v1.1.csv`` in ``$LVMCORE_DIR/sensitivity`` give the mean
standard-star sensitivity per channel: flux per count, i.e. the inverse of the
total throughput (atmosphere x telescope x instrument).  Then

    counts(lambda)  = flux(lambda) / sens(lambda)
    sigma_counts    = sqrt(counts)                      (Poisson)
    sigma_flux      = sigma_counts * sens = sqrt(flux(lambda) * sens(lambda))

so the inverse-variance pixel weight is ``1 / (flux * sens)``.  ``flux`` is the
TOTAL observed flux in the fibre -- the whole photon budget sets the noise, not
the component being fitted -- and a global constant on ``sens`` (exposure time,
fibre count in the median stack, gain) rescales every weight together and drops
out once the weights are normalised.

The one thing that does NOT drop out: the per-arm normalisation
--------------------------------------------------------------
``avgsens-1-1.py`` normalises each channel's curve to unit mean inside a
Gaussian band at 4500 / 6500 / 8500 A BEFORE taking the median over exposures,
so the three published curves are shapes on three independent scales.  Joining
them takes two numbers and they matter: naive concatenation puts factor-2 and
factor-6 steps at the arm joins straight into the noise model.

``SENS_ARM_SCALE`` comes from the total-throughput curve (including atmosphere)
at the three band centres, ~0.375 / 0.50 / 0.545, because a published
throughput is a direct measurement.  Three other routes were tried; they
disagree, and the disagreement is the honest uncertainty on this model:

* Matching the arms in their OVERLAPS (5775-5800, 7520-7570 A) gives r = 0.520,
  z = 0.158.  REJECTED: the overlaps sit at the extreme detector edges where
  the two arms have opposite slopes and disagree in SHAPE -- the b/r ratio
  swings 0.44-0.63 across 51 pixels -- and on a balanced set of line-free
  windows this leaves a 3.22x arm-to-arm residual against 1.36x for the values
  adopted here.
* A low envelope of measured ``sigma^2/flux`` over 20 A windows: r = 0.539,
  z = 0.442.
* The fit RESIDUALS: r = 0.679, z = 4.474, and unusable -- in the NIR they are
  dominated by model misfit on the bright OH lines rather than photon noise
  (log-scatter 88-182% against 23% for the direct estimator).

No noise-based route can settle it, because they all rest on a
second-difference estimator that assumes WHITE noise while the DRP resamples
onto the common 0.5 A grid -- and the LSF is 3.13 / 2.62 / 3.50 pixels FWHM in
b / r / z, so the pixel-to-pixel correlation, and hence the estimator's bias,
is arm-dependent in precisely the comparison being made.  So treat the
arm-to-arm weighting as good to ~1.5x, and the WITHIN-arm shape -- which
carries the 4.3x dynamic range and comes from hundreds of standard-star
exposures -- as good.  Exact band-centre throughputs would go straight into
SENS_ARM_SCALE and nothing else would change.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# (name, blue edge, red edge) of each published curve, in Angstrom.
SENS_ARMS = (("b", 3600.0, 5800.0), ("r", 5775.0, 7570.0), ("z", 7520.0, 9800.0))

# Per-arm normalisation relative to b, for the SUPERSEDED relative loader.
# Measured against the absolute curves it is wrong by 1.60x in r and 2.25x in
# z; kept only so an old artifact can be reproduced.
SENS_ARM_SCALE = {"b": 1.0, "r": 0.74, "z": 0.69}

SENS_FILENAME = "mean-sens-{arm}-v1.1.csv"

# --- Absolute sensitivity ---------------------------------------------------
# `mean-sens-{arm}-v1.1.csv` cannot carry an absolute scale BY CONSTRUCTION:
# `$LVMCORE_DIR/sensitivity/avgsens-1-1.py` divides every per-exposure curve by
# its own Gaussian-weighted integral about the arm centre (4500 / 6500 / 8500 A,
# sigma 250) before taking the median, so each arm comes out with unit weighted
# mean and both the absolute AND the cross-arm scale are gone.  That is the
# whole reason SENS_ARM_SCALE existed, and why it had to be eyeballed off a
# throughput figure.
#
# `sens_percentiles-{arm}.csv` is the same quantity WITHOUT that step, in
# absolute [erg/s/cm^2/A] per [e-/s/A].  Column 4 of 7 is the median, verified:
# renormalising it exactly the way avgsens does reproduces the deployed curve
# to a maximum relative error of 1.4e-2 (b), 5.6e-3 (r), 3.3e-2 (z), while
# every other percentile column is 2-20x worse.  The scales that had been
# divided out are 6.9309e-14 (b), 3.2117e-14 (r), 2.1217e-14 (z), i.e. arm
# ratios 1 : 0.4634 : 0.3061 against the guessed 1 : 0.74 : 0.69.
SENS_PERCENTILE_FILENAME = "sens_percentiles-{arm}.csv"
SENS_ABS_COLUMN = 4

# A median of N samples has variance (pi/2) sigma^2 / N asymptotically, so an
# N-fibre median stack behaves like (2/pi) N independent samples.
MEDIAN_STACK_EFFICIENCY = 2.0 / np.pi


def sensitivity_dir(sens_dir=None):
    """Directory holding the mean sensitivity CSVs."""
    if sens_dir is not None:
        return Path(sens_dir).expanduser()
    root = os.environ.get("LVMCORE_DIR")
    if not root:
        raise RuntimeError(
            "LVMCORE_DIR is not set and sens_dir was not given; cannot locate "
            "mean-sens-{b,r,z}-v1.1.csv")
    return Path(root).expanduser() / "sensitivity"


def load_relative_sensitivity(wave, sens_dir=None, arm_scale=None, verbose=False):
    """Joined relative sensitivity (flux per count, arbitrary global scale).

    Each published curve is interpolated onto ``wave``, scaled by its
    ``SENS_ARM_SCALE`` factor, and CROSSFADED linearly across the overlap so the
    join is continuous.  A plain average over the overlap leaves a visible step
    (0.10 and 0.056, against a typical adjacent-pixel change of 0.00015)
    because the two arms cross there with opposite slopes; the crossfade also
    puts the weight on whichever arm is further from its own edge, which is the
    one worth trusting.

    Raises if any pixel ends up uncovered: a silent NaN here becomes a NaN
    pixel weight and a NaN loss.
    """
    wave = np.asarray(wave, dtype=np.float64)
    scale = dict(SENS_ARM_SCALE if arm_scale is None else arm_scale)
    directory = sensitivity_dir(sens_dir)
    num = np.zeros_like(wave)
    den = np.zeros_like(wave)
    for arm, _lo, _hi in SENS_ARMS:
        table = np.loadtxt(directory / SENS_FILENAME.format(arm=arm), delimiter=",")
        w_a, s_a = table[:, 0], table[:, 1]
        inside = (wave >= w_a[0]) & (wave <= w_a[-1])
        if not np.any(inside):
            continue
        lam = wave[inside]
        vals = np.interp(lam, w_a, s_a) * float(scale[arm])
        weight = np.ones(lam.size, dtype=np.float64)
        for other, olo, ohi in SENS_ARMS:
            if other == arm:
                continue
            a, b = max(w_a[0], olo), min(w_a[-1], ohi)
            if b <= a:
                continue
            frac = np.clip((lam - a) / (b - a), 0.0, 1.0)
            # Ramp up away from this arm's own edge: if the overlap starts at
            # this arm's blue end the arm is untrustworthy at `a`, else at `b`.
            ramp = frac if a <= w_a[0] + 1e-9 else 1.0 - frac
            weight = np.where((lam >= a) & (lam <= b), ramp * weight, weight)
        num[inside] += vals * weight
        den[inside] += weight
    bad = ~(den > 0) | ~np.isfinite(num)
    if np.any(bad):
        raise RuntimeError(
            f"relative sensitivity is undefined on {int(bad.sum())} of "
            f"{wave.size} pixels (first at {wave[np.flatnonzero(bad)[0]]:.1f} A); "
            f"the published curves do not cover this wavelength grid")
    sens = num / den
    if verbose:
        print(f"  [noise] relative sensitivity {sens.min():.4g}-{sens.max():.4g} "
              f"over {wave[0]:.0f}-{wave[-1]:.0f} A; arm scales "
              + ", ".join(f"{k}={v:g}" for k, v in scale.items()))
    return sens


def load_absolute_sensitivity(wave, sens_dir=None, verbose=False):
    """Joined ABSOLUTE sensitivity, [erg/s/cm^2/A] per [e-/s/A].

    Same crossfade as :func:`load_relative_sensitivity` -- the join would
    otherwise show a step, because the two arms cross the overlap with opposite
    slopes -- but reading the un-normalised percentile table, so no per-arm
    scale factor is applied or needed.

    Raises if any pixel is uncovered: a silent NaN here becomes a NaN weight
    and a NaN loss.
    """
    wave = np.asarray(wave, dtype=np.float64)
    directory = sensitivity_dir(sens_dir)
    num = np.zeros_like(wave)
    den = np.zeros_like(wave)
    for arm, _lo, _hi in SENS_ARMS:
        path = directory / SENS_PERCENTILE_FILENAME.format(arm=arm)
        table = np.loadtxt(path, delimiter=",")
        w_a, s_a = table[:, 0], table[:, SENS_ABS_COLUMN]
        inside = (wave >= w_a[0]) & (wave <= w_a[-1])
        if not np.any(inside):
            continue
        lam = wave[inside]
        vals = np.interp(lam, w_a, s_a)
        weight = np.ones(lam.size, dtype=np.float64)
        for other, olo, ohi in SENS_ARMS:
            if other == arm:
                continue
            a, b = max(w_a[0], olo), min(w_a[-1], ohi)
            if b <= a:
                continue
            frac = np.clip((lam - a) / (b - a), 0.0, 1.0)
            ramp = frac if a <= w_a[0] + 1e-9 else 1.0 - frac
            weight = np.where((lam >= a) & (lam <= b), ramp * weight, weight)
        num[inside] += vals * weight
        den[inside] += weight
    bad = ~(den > 0) | ~np.isfinite(num)
    if np.any(bad):
        raise RuntimeError(
            f"absolute sensitivity is undefined on {int(bad.sum())} of "
            f"{wave.size} pixels (first at {wave[np.flatnonzero(bad)[0]]:.1f} A)")
    sens = num / den
    if verbose:
        print(f"  [noise] absolute sensitivity {sens.min():.4g}-{sens.max():.4g} "
              f"erg/s/cm2/A per e-/s/A over {wave[0]:.0f}-{wave[-1]:.0f} A")
    return sens


def photon_variance_absolute(flux, sens_abs, exptime, dwave, n_fibres=None,
                             median_stack=True):
    """Poisson variance of the observed flux, in (flux unit)^2.  ABSOLUTE.

    With ``sens`` in [erg/s/cm^2/A] per [e-/s/A] the electron count in one
    pixel is ``N = flux / sens * exptime * dwave``; Poisson gives ``var(N) = N``
    and converting back divides by ``(exptime * dwave / sens)^2``, so

        var(flux) = flux * sens / (exptime * dwave * N_eff)

    with ``N_eff`` the effective number of independent fibre samples behind the
    stack.  Nothing here is free to scale -- unlike the relative model, whose
    overall normalisation was undefined and whose per-arm ratios were guessed.

    ``n_fibres`` is the per-row fibre count (``fibers_*_used`` in META).  It
    matters a lot and is NOT a detail: the science arm stacks a median of 536
    fibres against ~50 in the sky arms, a ~10x difference in effective exposure
    and ~3.3x in sigma, and it ranges from 4 to 1615 across rows.  With
    ``median_stack`` the count is reduced by ``MEDIAN_STACK_EFFICIENCY``,
    because these corpora are median stacks, not means.

    Read noise, dark current and calibration error are NOT included; the
    variance floor in :func:`photon_pixel_variance` stands in for them.
    """
    flux = np.asarray(flux, dtype=np.float64)
    sens_abs = np.asarray(sens_abs, dtype=np.float64)
    if flux.ndim != 2 or sens_abs.shape != (flux.shape[1],):
        raise ValueError(
            f"shape mismatch: flux {flux.shape} against sens {sens_abs.shape}")
    n_eff = 1.0
    if n_fibres is not None:
        n_eff = np.asarray(n_fibres, dtype=np.float64).reshape(-1, 1)
        if median_stack:
            n_eff = n_eff * MEDIAN_STACK_EFFICIENCY
        n_eff = np.where(np.isfinite(n_eff) & (n_eff > 0.0), n_eff, 1.0)
    denom = float(exptime) * float(dwave)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"exptime*dwave must be positive, got {denom!r}")
    return flux * sens_abs[None, :] / (denom * n_eff)


def photon_pixel_weight(flux, sens, floor_frac=0.05):
    """Inverse-variance pixel weights ``1/(flux*sens)``, row-normalised to mean 1.

    ``flux`` is (n_row, n_pix) TOTAL observed flux; ``sens`` is (n_pix,).  The
    variance is floored at ``floor_frac`` times the row's median variance before
    inverting, which does two jobs: it stops pixels whose observed flux went
    non-positive on noise from taking infinite weight, and it stands in for the
    read-noise term this model does not have (between the OH bands in the NIR
    the sky is faint enough for read noise to matter).

    Normalising each ROW to mean weight 1 is deliberate.  It keeps the relative
    weighting across wavelength, which is the point, while leaving the relative
    weighting BETWEEN rows exactly as it was: row-to-row noise differences are
    real, but they also track sky brightness and exposure depth, and letting
    them in here would silently re-weight the training set on top of the
    existing per-row weights.
    """
    var = photon_pixel_variance(flux, sens, floor_frac=floor_frac)
    return weights_from_variance(var, floor_frac=None)


def floor_variance(var, floor_frac=0.05):
    """Apply the one variance floor every consumer shares.

    Non-finite and non-positive entries take the row's median variance, and
    everything is clipped up to ``floor_frac`` of that median.  Two jobs: a
    pixel whose observed flux went non-positive on noise cannot take a
    near-zero sigma (10.9% of observed sci pixels are non-positive, and
    unfloored they give a reduced chi2 of order 1e6), and the floor stands in
    for the read noise and dark current this Poisson model omits.

    Defined ONCE here because the training loss and the chi2 diagnostic must
    floor identically or the diagnostic measures a different noise model from
    the one being tested.
    """
    var = np.asarray(var, dtype=np.float64)
    var = np.where(np.isfinite(var) & (var > 0.0), var, np.nan)
    med = np.nanmedian(var, axis=1, keepdims=True)
    med = np.where(np.isfinite(med) & (med > 0.0), med, 1.0)
    var = np.where(np.isfinite(var), var, med)
    return np.maximum(var, float(floor_frac) * med)


def weights_from_variance(var, floor_frac=0.05):
    """Row-normalised inverse-variance weights from an ALREADY-BUILT variance.

    Split out so the absolute model can supply its own variance.  Pass
    ``floor_frac=None`` when the variance is already floored, otherwise the
    same floor as :func:`photon_pixel_variance` is applied here: non-finite or
    non-positive entries take the row median and everything is clipped up to
    ``floor_frac`` of it, which is what keeps a near-zero flux pixel from
    taking unbounded weight and stands in for the read noise this model omits.

    Row normalisation to mean weight 1 is deliberate and unchanged: it fixes
    the weighting ACROSS wavelength, which is the point, while leaving the
    weighting BETWEEN rows exactly as it was.
    """
    var = np.asarray(var, dtype=np.float64)
    if floor_frac is not None:
        var = floor_variance(var, floor_frac=floor_frac)
    w = 1.0 / var
    w /= np.mean(w, axis=1, keepdims=True)
    return w.astype(np.float32)


def photon_pixel_variance(flux, sens, floor_frac=0.05):
    """Floored photon variance ``flux*sens`` per pixel, in RELATIVE units.

    Split out of ``photon_pixel_weight`` so a diagnostic can measure the noise
    model the loss actually uses instead of re-deriving ``sqrt(flux*sens)`` and
    silently dropping the floor -- which is what makes faint pixels take a
    near-zero sigma and a reduced chi2 of order 1e6.

    Unlike the weights this is NOT row-normalised, so the absolute scale
    survives and a caller can calibrate it against an independent sigma.  The
    scale is arbitrary regardless (``sens`` is relative), so it is only ever
    meaningful up to one multiplicative constant.
    """
    flux = np.asarray(flux, dtype=np.float64)
    sens = np.asarray(sens, dtype=np.float64)
    if flux.ndim != 2 or sens.shape != (flux.shape[1],):
        raise ValueError(
            f"shape mismatch: flux {flux.shape} against sens {sens.shape}")
    return floor_variance(flux * sens[None, :], floor_frac=floor_frac)


__all__ = [
    "SENS_ARMS",
    "load_absolute_sensitivity",
    "photon_variance_absolute",
    "weights_from_variance",
    "MEDIAN_STACK_EFFICIENCY",
    "photon_pixel_variance",
    "SENS_ARM_SCALE",
    "load_relative_sensitivity",
    "photon_pixel_weight",
    "sensitivity_dir",
]
