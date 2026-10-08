"""Per-pixel inverse-variance weights for the decomposition fit.

WHY THIS EXISTS
---------------
The decomposition has always been fitted with ``ivar = isfinite(flux)`` -- an
unweighted MASK, not a variance (see ``decompose_parallel._fit_worker_row``).
Every pixel therefore counted equally, from the OH band heads to the faint
inter-band continuum, even though their photon noise differs by more than an
order of magnitude.  The ML loss has used the absolute photon model since
2026-09-09; this brings the FIT onto the same footing.

THE MODEL is the same one the loss uses, and deliberately shares its
definitions so the two cannot drift:

    N          = flux / sens * exptime * dwave        (electrons in a pixel)
    var(N)     = N                                    (Poisson)
    var(flux)  = flux * sens / (exptime * dwave * N_eff)

with ``sens`` the ABSOLUTE sensitivity in [erg/s/cm^2/A] per [e-/s/A] and
``N_eff`` the effective number of independent fibre samples behind a stacked
row -- ``MEDIAN_STACK_EFFICIENCY * n_fibres`` for a median stack.

WHY THE CURVES ARE VENDORED
---------------------------
``data/sensitivity/sens_abs-{b,r,z}.csv`` are copies of column 4 of
``lvmcore/sensitivity/sens_percentiles-{arm}.csv`` (provenance and checksums in
that directory's README), so a decomposition does not depend on ``$LVMCORE_DIR``
resolving.  The percentile table is the only one carrying an ABSOLUTE scale:
``mean-sens-{arm}-v1.1.csv`` divides each arm by its own weighted mean, which
fixes the shape within an arm but leaves the normalisation and the arm-to-arm
ratios undetermined.

ROW NORMALISATION IS THE IMPORTANT DESIGN CHOICE
------------------------------------------------
``pixel_ivar`` returns weights normalised to mean 1 over the good pixels.  That
keeps the RELATIVE weighting across wavelength -- the entire point -- while
leaving the overall scale exactly where the unweighted mask had it.  It matters
because every regularisation constant in the fit was tuned against ivar = 1:
``moon_smooth_lambda`` (0.1), ``zodi_smooth_lambda``, the LSF
``roughness_fraction`` (1e-4), ``line_weight`` (5e-4) and
``huber_transition_sigma`` (3.0).  Feeding raw inverse variances, whose
magnitudes are of order 1e26 in fit units, would silently rescale every one of
them.  With the normalisation those constants keep their meaning and the change
is purely a re-weighting across wavelength.

Note this makes ``reduced_chi2`` in the products a properly weighted chi2 rather
than the unweighted residual-per-pixel it has been, so its VALUES are not
comparable across the change even though its definition is unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Arm coverage and crossfade windows, identical to mlp_predictor.noise.SENS_ARMS.
# The overlaps are real: b/r cross near 5775-5800 A and r/z near 7520-7570 A, and
# the two arms traverse an overlap with opposite slopes, so a hard join leaves a
# visible step.  Hence the linear crossfade in `absolute_sensitivity`.
SENS_ARMS = (("b", 3600.0, 5800.0), ("r", 5775.0, 7570.0), ("z", 7520.0, 9800.0))
SENS_FILENAME = "sens_abs-{arm}.csv"
# Var(median of n) / Var(mean of n) -> pi/2 asymptotically, so a median stack of
# n fibres is worth (2/pi)*n independent samples.
MEDIAN_STACK_EFFICIENCY = 2.0 / np.pi
DEFAULT_FLOOR_FRAC = 0.05
DEFAULT_EXPTIME_S = 900.0

_CACHE: dict[tuple, np.ndarray] = {}


def sensitivity_dir(data_dir=None) -> Path:
    """Directory holding the vendored absolute-sensitivity CSVs.

    Order: ``data_dir`` if given, else ``$LVMSKY_DATA_ROOT/sensitivity`` if that
    variable is set, else ``data/sensitivity`` next to this file.
    """
    if data_dir is not None:
        return Path(data_dir).expanduser()
    # 2026-10-08: LVMSKY_DATA_ROOT, when set, replaces the bundled data root, the
    # same variable as moon_zodi_model.DEFAULT_DATA_ROOT (read here directly to
    # keep this module numpy-only).  Unset, the behaviour is unchanged.
    root = os.environ.get("LVMSKY_DATA_ROOT")
    if root:
        return Path(root).expanduser().resolve() / "sensitivity"
    return Path(__file__).resolve().parent / "data" / "sensitivity"


def absolute_sensitivity(wave, data_dir=None, verbose=False) -> np.ndarray:
    """Joined ABSOLUTE sensitivity on ``wave``, [erg/s/cm^2/A] per [e-/s/A].

    Linear crossfade over each arm overlap, matching
    ``mlp_predictor.noise.load_absolute_sensitivity`` exactly -- verified
    bit-identical, because the fit and the loss must not use different noise.

    Raises if any pixel is uncovered: a silent NaN here becomes a NaN weight and
    a silently unweighted fit.
    """
    wave = np.asarray(wave, dtype=np.float64)
    key = (wave.shape, float(wave[0]), float(wave[-1]),
           str(sensitivity_dir(data_dir)))
    hit = _CACHE.get(key)
    if hit is not None and hit.shape == wave.shape:
        return hit
    directory = sensitivity_dir(data_dir)
    num = np.zeros_like(wave)
    den = np.zeros_like(wave)
    for arm, _lo, _hi in SENS_ARMS:
        path = directory / SENS_FILENAME.format(arm=arm)
        table = np.loadtxt(path, delimiter=",", comments="#")
        w_a, s_a = table[:, 0], table[:, 1]
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
            f"{wave.size} pixels (first at {wave[np.flatnonzero(bad)[0]]:.1f} A); "
            f"checked {directory}")
    sens = num / den
    if verbose:
        print(f"  [pixel-weights] absolute sensitivity "
              f"{sens.min():.4g}-{sens.max():.4g} over "
              f"{wave[0]:.0f}-{wave[-1]:.0f} A from {directory}")
    _CACHE[key] = sens
    return sens


def floor_variance(var, floor_frac=DEFAULT_FLOOR_FRAC) -> np.ndarray:
    """Floor a 1-D variance at ``floor_frac`` of its own median.

    Same definition as ``mlp_predictor.noise.floor_variance`` for one row.  Two
    jobs: a pixel whose observed flux went non-positive on noise cannot take a
    near-zero sigma (about 11% of observed sci pixels are non-positive), and the
    floor stands in for the read noise and dark current this Poisson model
    omits, which matter between the OH bands in the NIR.
    """
    var = np.asarray(var, dtype=np.float64)
    good = np.isfinite(var) & (var > 0.0)
    if not np.any(good):
        return np.ones_like(var)
    med = float(np.median(var[good]))
    if not np.isfinite(med) or med <= 0.0:
        med = 1.0
    var = np.where(good, var, med)
    return np.maximum(var, float(floor_frac) * med)


def photon_variance(flux, sens, exptime=DEFAULT_EXPTIME_S, dwave=None,
                    n_fibres=None, median_stack=True) -> np.ndarray:
    """``var(flux) = flux * sens / (exptime * dwave * N_eff)`` for one row."""
    flux = np.asarray(flux, dtype=np.float64)
    sens = np.asarray(sens, dtype=np.float64)
    if flux.shape != sens.shape:
        raise ValueError(f"flux {flux.shape} and sens {sens.shape} must match")
    if dwave is None or not np.isfinite(dwave) or float(dwave) <= 0.0:
        raise ValueError(f"dwave must be positive, got {dwave!r}")
    n_eff = 1.0
    if n_fibres is not None and np.isfinite(n_fibres) and float(n_fibres) > 0.0:
        n_eff = float(n_fibres) * (MEDIAN_STACK_EFFICIENCY if median_stack else 1.0)
    denom = float(exptime) * float(dwave) * n_eff
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(f"exptime*dwave*N_eff must be positive, got {denom!r}")
    return flux * sens / denom


def pixel_ivar(flux, wave, *, exptime=DEFAULT_EXPTIME_S, dwave=None,
               n_fibres=None, median_stack=True, mask=None,
               floor_frac=DEFAULT_FLOOR_FRAC, normalise=True, clip=None,
               flux_scale=1.0, data_dir=None, sens=None) -> np.ndarray:
    """Per-pixel ivar for ONE spectrum, 0 on masked or unusable pixels.

    ``flux`` is the observed flux the fit is given, already multiplied by the
    decomposition's ``FACTOR``; pass ``flux_scale=FACTOR`` so the variance is
    computed from PHYSICAL flux and then returned on the fit's scale.  Getting
    that wrong only rescales the weights, which the row normalisation removes --
    but it would matter if ``normalise=False``.

    ``mask`` is True where the pixel must be EXCLUDED (the science-line windows,
    for instance); those pixels come back as ivar 0, exactly as the mask-only
    behaviour did.

    With ``normalise`` the good pixels have mean weight 1, so the fit's tuned
    regularisation constants keep their meaning -- see the module docstring.

    ``clip`` bounds the DYNAMIC RANGE of the weights to ``[mean/clip,
    mean*clip]`` (so ``clip=3`` spans a factor 9) and renormalises.  The raw
    weights span ~600x within a row, which is enough for the fit to abandon a
    faint blue component rather than fit it; see the measurement in
    ``decompose_parallel.FIT_PIXEL_WEIGHT_CLIP``.  ``None`` leaves them raw.
    """
    flux = np.asarray(flux, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)
    if flux.shape != wave.shape:
        raise ValueError(f"flux {flux.shape} and wave {wave.shape} must match")
    if dwave is None:
        dwave = float(np.median(np.diff(wave)))
    if sens is None:
        sens = absolute_sensitivity(wave, data_dir=data_dir)
    usable = np.isfinite(flux)
    if mask is not None:
        usable &= ~np.asarray(mask, dtype=bool)
    if not np.any(usable):
        return np.zeros_like(flux)
    # Physical flux for the Poisson term; a non-positive pixel is not a
    # negative variance, it is a pixel the floor has to rescue, so take |flux|
    # and let `floor_variance` do the clipping.
    #
    # Taking max(flux, 0) instead -- arguably the truer Poisson statement,
    # since a pixel measured below zero has a signal near zero rather than a
    # large one -- was measured on 2026-09-20 and is a null: it reaches only
    # the 2.2% of rows with >1% negative pixels (all moon-down, all blue) and
    # moved their blue chi2 by -0.1%.
    phys = np.abs(flux) / float(flux_scale)
    var = photon_variance(phys, sens, exptime=exptime, dwave=dwave,
                          n_fibres=n_fibres, median_stack=median_stack)
    var = floor_variance(np.where(usable, var, np.nan), floor_frac=floor_frac)
    var = var * float(flux_scale) ** 2          # back onto the fit's flux scale
    ivar = np.where(usable, 1.0 / var, 0.0)

    def _renorm(w):
        m = float(np.mean(w[usable]))
        if np.isfinite(m) and m > 0.0:
            w = w / m
        return np.where(usable, w, 0.0)

    if clip is not None:
        c = float(clip)
        if not (c > 1.0):
            raise ValueError(f"clip must be > 1, got {clip!r}")
        # Clip about the row mean, which is what the bound is expressed against,
        # then renormalise so the mean is 1 again (clipping moves it).
        ivar = _renorm(ivar)
        ivar = np.where(usable, np.clip(ivar, 1.0 / c, c), 0.0)
    if normalise:
        ivar = _renorm(ivar)
    return ivar


__all__ = [
    "SENS_ARMS", "MEDIAN_STACK_EFFICIENCY", "DEFAULT_FLOOR_FRAC",
    "DEFAULT_EXPTIME_S", "sensitivity_dir", "absolute_sensitivity",
    "floor_variance", "photon_variance", "pixel_ivar",
]
