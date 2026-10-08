"""Per-row reliability flags for a sky decomposition, and the reversal test.

WHY THIS EXISTS
---------------
Until now every reliability judgement lived in the CORPUS BUILD as a gate that
drops rows (`mlp_predictor.data`: the moon/zodi reversal gate, the
science-continuum colour gate, the diffuse-collapse gate, kappa-sigma/OH-MAD).
Nothing was recorded in the products, so a consumer of a single fitted row --
production sky subtraction, where dropping the row is not an option -- could not
tell whether its decomposition was role-reversed or had no diffuse continuum at
all.  `fit_status` only says the QP converged.

This module holds the definitions ONCE so the fitter that writes the flags and
any analysis that reads them cannot drift apart.  It is numpy-only on purpose.

WHAT A REVERSAL IS
------------------
A row where the fitted moon continuum is REDDER than the fitted zodi continuum,
``moon_slope > zodi_slope`` in log-log flux against wavelength: the two families
have swapped roles, so the row's moon coefficients describe zodiacal light and
vice versa.  Physics puts the moon near -3.7 (scattered sunlight,
Rayleigh-dominated) and the zodi near -0.3 (Leinert reddening), so the ordering
is unambiguous WHERE BOTH FAMILIES CARRY FLUX -- and only there.  Rows with one
family essentially switched off carry no ordering information and are reported
untestable, not reversed; dark-time rows sit at a moon share of ~0.02 under the
current priors and are exempt by construction.

The slope is fitted on the refined ``components`` planes (the same arrays that
become the ``COMP_MOON`` / ``COMP_ZODI`` HDUs), so no basis is rebuilt and no
reconstruction convention has to be reproduced.  `loglog_slope` is a
single-row transcription of `mlp_predictor.data._loglog_slopes` and is checked
against it numerically by the tests; keep the two in step.
"""

from __future__ import annotations

import numpy as np

# Bit values for the `reliability` column, split into two severities.
#
# ERRORS (low 16 bits) mean the row's fitted coefficients do not describe what
# their names say: the fit failed, the moon/zodi labels are swapped, a whole
# family is missing, or the science fibre's continuum is not sky.  These are the
# rows the corpus build drops, and a consumer should not score or train on them.
#
# WARNINGS (bit 16 up) mean the fit is usable but a CONSTRAINT shaped it, so the
# value is partly the prior rather than the data.  The zodi anchor is the reason
# this distinction has to exist: it binds on 93% of moon-up rows, which makes
# those zodi targets partly synthetic, and until now nothing in the products
# said so.  A warning is information, not a reason to drop a row.
#
# Bit values are PERSISTED in the products.  They were renumbered once, on
# 2026-09-18, to open the error/warning split -- at that point no corpus had
# been written with them, only same-day local smoke runs.  Append only from
# here: never reuse or renumber a bit again.
RELIABILITY_FIT_FAILED = 1 << 0           # fit_status is not 'Solved'
RELIABILITY_REVERSED = 1 << 1             # moon/zodi roles swapped, AS WRITTEN
RELIABILITY_DIFFUSE_COLLAPSED = 1 << 2    # diffuse block zero, not merely faint
RELIABILITY_SCI_COLOUR_EXCESS = 1 << 3    # sci continuum is redder than the sky

RELIABILITY_REVERSAL_UNTESTABLE = 1 << 16  # one family off: no colour ordering
RELIABILITY_REVERSAL_RETRIED = 1 << 17     # a tighter-bound refit was run
RELIABILITY_REVERSAL_RECOVERED = 1 << 18   # ... and it un-reversed the row
RELIABILITY_ZODI_ANCHOR_PINNED = 1 << 19   # int(zodi) on the Leinert bracket
RELIABILITY_MOON_SHARE_PINNED = 1 << 20    # moon share on its geometry bracket
RELIABILITY_DIFFUSE_OH_CAP_BINDING = 1 << 21   # diffuse/OH block cap active
RELIABILITY_DIFFUSE_RATIO_PINNED = 1 << 22     # a species ratio on its bracket
RELIABILITY_SHAPE_BOUND_ACTIVE = 1 << 23   # moon/zodi adjacent-knot bound active

# MEASURED OCCUPANCY, so a near-universal warning is not misread as a rare
# event (24 every10 telluric rows, sci arm, verified against the solver's own
# functionals to 1.0000):
#
#   zodi_anchor_pinned       every moon-up row tested sat EXACTLY on the
#                            kappa_z ceiling (v / (kappa_z * Z) = 1.0000);
#                            interior rows read 0.36-0.91.  Consistent with the
#                            93%-of-moon-up-rows figure measured separately.
#   moon_share_pinned        dark-time rows pin at f_hi = amp_prior_floor =
#                            0.02 (the "ghost" moon block); moon-up rows pin at
#                            f_lo.  9 of 24.
#   diffuse_oh_cap_binding   binds on essentially every GATED row
#                            (block/cap = 1.0000); 13 of 24, all of them gated.
#   diffuse_ratio_pinned     ~80% of rows: the three species are individually
#                            unidentifiable, so the QP parks them on the +/-0.2
#                            dex bracket.  Expected, not a defect.
#   shape_bound_active       ~100% of rows, so the BOOLEAN carries almost no
#                            information -- `shape_bound_pairs` (0 to 16 of 18
#                            adjacent pairs) is the quantity to read.  Note a
#                            reversal retry at a tighter bound typically moves
#                            the fit OFF these bounds entirely.

RELIABILITY_ERROR_MASK = 0x0000FFFF
RELIABILITY_WARNING_MASK = 0x7FFF0000

RELIABILITY_BITS = (
    (RELIABILITY_FIT_FAILED, "fit_failed"),
    (RELIABILITY_REVERSED, "reversed"),
    (RELIABILITY_DIFFUSE_COLLAPSED, "diffuse_collapsed"),
    (RELIABILITY_SCI_COLOUR_EXCESS, "sci_colour_excess"),
    (RELIABILITY_REVERSAL_UNTESTABLE, "reversal_untestable"),
    (RELIABILITY_REVERSAL_RETRIED, "reversal_retried"),
    (RELIABILITY_REVERSAL_RECOVERED, "reversal_recovered"),
    (RELIABILITY_ZODI_ANCHOR_PINNED, "zodi_anchor_pinned"),
    (RELIABILITY_MOON_SHARE_PINNED, "moon_share_pinned"),
    (RELIABILITY_DIFFUSE_OH_CAP_BINDING, "diffuse_oh_cap_binding"),
    (RELIABILITY_DIFFUSE_RATIO_PINNED, "diffuse_ratio_pinned"),
    (RELIABILITY_SHAPE_BOUND_ACTIVE, "shape_bound_active"),
)


def has_error(bits):
    """True when any ERROR bit is set (the row is not describing what it says)."""
    return bool(int(bits) > 0 and int(bits) & RELIABILITY_ERROR_MASK)


def has_warning(bits):
    """True when any WARNING bit is set (usable, but a constraint shaped it)."""
    return bool(int(bits) > 0 and int(bits) & RELIABILITY_WARNING_MASK)

# Matches mlp_predictor.data.split_zodi_reversal_diagnostics.
REVERSAL_MIN_COMPONENT_FRAC = 0.05
REVERSAL_MIN_SEPARATION = 0.0
REVERSAL_MIN_PIXELS = 200

# Row-local diffuse-collapse threshold: the block's share of the row's own
# fitted flux.  The corpus gate (`mlp_predictor.data.diffuse_zeroed_mask`) is
# relative to the CORPUS median instead, which a single row cannot know.  Both
# work because the population is bimodal -- the QP either fits a diffuse
# continuum or switches the whole block off exactly -- so any threshold in the
# empty middle selects the same rows.
DIFFUSE_COLLAPSED_SHARE = 1.0e-6

DIFFUSE_COMPONENT_KEYS = ("ho2", "feo", "o2ac")


def loglog_slope(component, log_wave, min_pixels=REVERSAL_MIN_PIXELS):
    """Log-log slope ``b`` in ``log f = a + b log(lambda)`` for ONE row.

    Fitted over finite, strictly POSITIVE samples only: the QP leaves exact
    zeros wherever a family is switched off and those carry no colour
    information.  Returns NaN below ``min_pixels`` usable samples or on a
    degenerate lever arm.
    """
    flux = np.asarray(component, dtype=np.float64).ravel()
    x = np.asarray(log_wave, dtype=np.float64).ravel()
    if flux.size != x.size:
        raise ValueError(f"component {flux.size} and log_wave {x.size} must match")
    good = np.isfinite(flux) & (flux > 0.0)
    n = float(good.sum())
    if n < float(min_pixels):
        return float("nan")
    y = np.log(flux[good])
    xg = x[good]
    sx = float(xg.sum())
    sxx = float((xg * xg).sum())
    den = n * sxx - sx * sx
    if not (den > 0.0):
        return float("nan")
    sy = float(y.sum())
    sxy = float((xg * y).sum())
    return (n * sxy - sx * sy) / den


def _positive_total(component):
    a = np.asarray(component, dtype=np.float64)
    return float(np.nansum(np.where(a > 0.0, a, 0.0)))


def reversal_state(components, log_wave,
                   min_component_frac=REVERSAL_MIN_COMPONENT_FRAC,
                   min_separation=REVERSAL_MIN_SEPARATION,
                   min_pixels=REVERSAL_MIN_PIXELS):
    """Moon/zodi colour ordering for one fitted row.

    ``components`` is the result's component mapping; ``log_wave`` is
    ``np.log(wave)``, passed in because the caller fits many rows on one grid.

    Returns ``(is_reversed, testable, info)`` where ``info`` carries
    ``moon_slope``, ``zodi_slope``, ``separation`` (``zodi_slope -
    moon_slope``, so NEGATIVE means reversed) and ``moon_frac``.  A row that is
    not testable is never reported reversed.
    """
    info = {"moon_slope": float("nan"), "zodi_slope": float("nan"),
            "separation": float("nan"), "moon_frac": float("nan")}
    if not components or "moon" not in components or "zodi" not in components:
        return False, False, info
    moon = components["moon"]
    zodi = components["zodi"]
    moon_slope = loglog_slope(moon, log_wave, min_pixels=min_pixels)
    zodi_slope = loglog_slope(zodi, log_wave, min_pixels=min_pixels)
    m_tot = _positive_total(moon)
    z_tot = _positive_total(zodi)
    total = m_tot + z_tot
    moon_frac = (m_tot / total) if total > 0.0 else float("nan")
    separation = zodi_slope - moon_slope
    info.update(moon_slope=moon_slope, zodi_slope=zodi_slope,
                separation=separation, moon_frac=moon_frac)
    frac_lo = float(min_component_frac)
    testable = bool(
        np.isfinite(moon_slope) and np.isfinite(zodi_slope)
        and np.isfinite(moon_frac)
        and moon_frac >= frac_lo and moon_frac <= 1.0 - frac_lo
    )
    is_reversed = bool(testable and separation < float(min_separation))
    return is_reversed, testable, info


def diffuse_collapsed(components, bestfit,
                      share_threshold=DIFFUSE_COLLAPSED_SHARE):
    """True when the whole diffuse block is switched off, not merely faint.

    Judged on the block's share of the row's own fitted flux, so it needs no
    corpus context.  All three species go together in practice (among
    corpus-gated rows HO2 is non-zero on 0.5%, FeO on 0.0%, O2Ac on 0.5%), and
    the summed ``diffuse`` plane is used when present so the test does not
    depend on which species names a variant carries.
    """
    if not components:
        return False
    if "diffuse" in components:
        block = _positive_total(components["diffuse"])
    else:
        keys = [k for k in DIFFUSE_COMPONENT_KEYS if k in components]
        if not keys:
            return False
        block = sum(_positive_total(components[k]) for k in keys)
    total = _positive_total(bestfit)
    if not (total > 0.0):
        return False
    return bool(block / total < float(share_threshold))


# Science-continuum colour test, mirroring
# mlp_predictor.data.sci_continuum_colour_excess: two line-free continuum
# windows, and dC = colour(sci) - mean(colour(near), colour(far)).  A pure
# ratio, so throughput and exposure time cancel and only SHAPE survives.
SCI_COLOUR_BLUE_BAND = (4150.0, 4400.0)
SCI_COLOUR_RED_BAND = (6050.0, 6250.0)
SCI_COLOUR_EXCESS_MAX = 0.05

# How close to a bound counts as ON it.  The constraints are linear in the
# band-integrated component fluxes, which this module recomputes by integrating
# the fitted component planes over the SAME good pixels the solve used -- so the
# agreement is exact up to the QP's own feasibility tolerance, not up to a
# modelling approximation.  1e-3 is ~100x that tolerance and still far inside
# the gap between a binding row and a free one.
BINDING_RTOL = 1.0e-3


def _band_median(flux, wave, band):
    flux = np.asarray(flux, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)
    use = (wave >= float(band[0])) & (wave <= float(band[1])) & np.isfinite(flux)
    if not np.any(use):
        return float("nan")
    return float(np.median(flux[use]))


def _colour(flux, wave):
    # A colour that cannot be measured returns NaN rather than raising: this is
    # a diagnostic, and an exception here would fail the whole ROW (and, before
    # `_fit_worker_row` catches it, potentially the chunk) over a flag.  A
    # length mismatch means the caller handed us a grid that is not this
    # spectrum's, which is exactly such a case.
    if np.asarray(flux).shape != np.asarray(wave).shape:
        return float("nan")
    blue = _band_median(flux, wave, SCI_COLOUR_BLUE_BAND)
    red = _band_median(flux, wave, SCI_COLOUR_RED_BAND)
    if not (np.isfinite(blue) and np.isfinite(red) and blue > 0.0 and red > 0.0):
        return float("nan")
    return float(np.log10(red / blue))


def sci_colour_excess(flux_sci, flux_near, flux_far, wave):
    """``dC``: how much redder the science fibre's continuum is than the sky's.

    A POSITIVE excess means the science fibre carries continuum the sky model
    cannot represent (a field star, nebular continuum), and the 15-knot moon
    spline -- the only flexible continuum in the basis -- absorbs it, so the
    row's moon coefficients stop describing scattered moonlight.  Measured
    rho(dC, moon colour distortion) = +0.405 against +0.034 for the
    far-minus-near control.

    Returns NaN when any band median is non-positive or missing.  This needs
    all three arms of the same exposure, so it is a property of the ROW, not of
    the arm being fitted.
    """
    c_sci = _colour(flux_sci, wave)
    c_near = _colour(flux_near, wave)
    c_far = _colour(flux_far, wave)
    if not np.isfinite(c_sci):
        return float("nan")
    sky = [c for c in (c_near, c_far) if np.isfinite(c)]
    if not sky:
        return float("nan")
    return float(c_sci - float(np.mean(sky)))


def _integral(component, good):
    a = np.asarray(component, dtype=np.float64)
    if good is None:
        return float(np.nansum(a))
    return float(np.nansum(a[np.asarray(good, dtype=bool)]))


def _on_bound(value, bound, rtol=BINDING_RTOL):
    if not (np.isfinite(value) and np.isfinite(bound)) or bound == 0.0:
        return False
    return bool(abs(value / bound - 1.0) <= float(rtol))


def constraint_bits(components, prior, good=None, coef_blocks=None,
                    rtol=BINDING_RTOL):
    """Which identifiability constraints SHAPED this row's fit.

    Every one of these is a hard linear inequality on band-integrated component
    fluxes (see the ``ratio_rows`` block in ``sky_decomp/fit.py``), so the test
    is simply whether the fitted quantity sits on its bound.  The integrals are
    taken over ``good`` -- the pixels the solve actually used -- because the
    constraint rows are built from ``a_mat.sum(axis=0)``, which excludes masked
    pixels; integrating the full row instead would shift every quantity by the
    masked fraction and break the comparison on near-boundary rows.

    ``prior`` carries the scalars that were INSTALLED for this row (the
    geometry-predicted moon fraction and zodi total, the bracket widths, the
    diffuse/OH cap parameters).  ``coef_blocks`` maps 'moon'/'zodi' to that
    family's spline coefficients, for the adjacent-knot shape bound.

    Returns ``(bits, info)``; ``info`` carries the measured quantities so a
    caller can record them without recomputing, including
    ``shape_bound_pairs`` -- read that rather than the near-universal
    ``shape_bound_active`` bit.
    """
    bits = 0
    info = {}
    components = components or {}
    prior = prior or {}
    u = _integral(components.get("moon", 0.0), good)
    v = _integral(components.get("zodi", 0.0), good)
    info["moon_int"] = u
    info["zodi_int"] = v
    # NB the two ABSOLUTE tests below integrate over `good`, matching the
    # solve. Their right-hand sides are full-grid quantities, so both the
    # solve and this test sit ~0.76% loose against the physical target -- but
    # they agree WITH EACH OTHER, which is what makes the reported bits
    # meaningful. Change one and you must change the other: moving only the
    # solve to a full-grid LHS sends `frac_zodi_anchor_pinned` from 50% to 0%
    # while the fit is unchanged. The ratio tests are immune either way.

    # --- moon share against the geometry bracket -------------------------
    frac = prior.get("moon_fraction")
    kappa_f = float(prior.get("amp_prior_tol", 0.0) or 0.0)
    eps = float(prior.get("amp_prior_floor", 0.02) or 0.02)
    total = u + v
    share = (u / total) if total > 0.0 else float("nan")
    info["moon_share"] = share
    if (frac is not None and np.isfinite(frac) and kappa_f > 1.0
            and np.isfinite(share)):
        f = float(np.clip(frac, 0.0, 1.0))
        f_hi = min(max(kappa_f * f, eps), 1.0 - eps)
        f_lo = min(max(f / kappa_f, 0.0), f_hi)
        if _on_bound(share, f_hi, rtol) or _on_bound(share, f_lo, rtol):
            bits |= RELIABILITY_MOON_SHARE_PINNED

    # --- absolute zodi against the Leinert bracket ------------------------
    zodi_total = prior.get("zodi_total")
    kappa_z = float(prior.get("zodi_amp_bound", 0.0) or 0.0)
    if (zodi_total is not None and np.isfinite(zodi_total) and zodi_total > 0.0
            and kappa_z > 1.0 and np.isfinite(v)):
        if (_on_bound(v, kappa_z * float(zodi_total), rtol)
                or _on_bound(v, float(zodi_total) / kappa_z, rtol)):
            bits |= RELIABILITY_ZODI_ANCHOR_PINNED

    # --- moon-gated diffuse/OH block cap ----------------------------------
    species = [_integral(components.get(k, 0.0), good)
               for k in DIFFUSE_COMPONENT_KEYS]
    block = (_integral(components["diffuse"], good)
             if "diffuse" in components else float(np.sum(species)))
    info["diffuse_int"] = block
    oh_amp = prior.get("diffuse_oh_amp")
    centre = prior.get("diffuse_oh_centre_log10")
    width = float(prior.get("diffuse_oh_bound_dex", 0.0) or 0.0)
    gate = float(prior.get("diffuse_oh_gate_frac", 0.6) or 0.6)
    relax = float(prior.get("diffuse_oh_relax_dex", 0.0) or 0.0)
    if (oh_amp is not None and centre is not None and width > 0.0
            and frac is not None and np.isfinite(frac)):
        f = float(np.clip(frac, 0.0, 1.0))
        if f > gate:
            s = float(np.clip((f - gate) / max(1.0 - gate, 1e-6), 0.0, 1.0))
            cap = 10.0 ** (float(centre) + width + relax * (1.0 - s)) * float(oh_amp)
            info["diffuse_oh_cap"] = cap
            if np.isfinite(block) and cap > 0.0 and block >= cap * (1.0 - rtol):
                bits |= RELIABILITY_DIFFUSE_OH_CAP_BINDING

    # --- diffuse species ratio bracket ------------------------------------
    nominal = prior.get("diffuse_ratio_nominal")
    ratio_w = float(prior.get("diffuse_ratio_bound_dex", 0.0) or 0.0)
    if nominal is not None and ratio_w > 0.0:
        nom = np.asarray(nominal, dtype=np.float64).ravel()
        if nom.size == len(species) and np.all(nom > 0.0) and species[0] > 0.0:
            w = 10.0 ** ratio_w
            for k in range(1, nom.size):
                if not species[k] > 0.0:
                    continue
                measured = species[k] / species[0]
                nominal_ratio = float(nom[k] / nom[0])
                if (_on_bound(measured, nominal_ratio * w, rtol)
                        or _on_bound(measured, nominal_ratio / w, rtol)):
                    bits |= RELIABILITY_DIFFUSE_RATIO_PINNED
                    break

    # --- moon/zodi adjacent-knot shape bound ------------------------------
    pairs_on_bound = 0
    for family, key in (("moon", "moon_ratio_bound"), ("zodi", "zodi_ratio_bound")):
        beta = float(prior.get(key, 0.0) or 0.0)
        block_coef = (coef_blocks or {}).get(family)
        if block_coef is None or not (0.0 < beta < 1.0):
            continue
        c = np.asarray(block_coef, dtype=np.float64).ravel()
        for k in range(c.size - 1):
            # A zero coefficient forces its neighbour to zero through the same
            # bound, which is the family being switched OFF rather than shaped.
            if not (c[k] > 0.0 and c[k + 1] > 0.0):
                continue
            ratio = c[k + 1] / c[k]
            if _on_bound(ratio, beta, rtol) or _on_bound(ratio, 1.0 / beta, rtol):
                pairs_on_bound += 1
    info["shape_bound_pairs"] = pairs_on_bound
    if pairs_on_bound:
        bits |= RELIABILITY_SHAPE_BOUND_ACTIVE
    return bits, info


def describe(bits):
    """Human-readable flag list, for logs and error messages."""
    bits = int(bits)
    if bits == 0:
        return "ok"
    names = [name for value, name in RELIABILITY_BITS if bits & value]
    unknown = bits & ~sum(value for value, _ in RELIABILITY_BITS)
    if unknown:
        names.append(f"unknown(0x{unknown:x})")
    return "|".join(names)


__all__ = [
    "RELIABILITY_FIT_FAILED", "RELIABILITY_REVERSED",
    "RELIABILITY_DIFFUSE_COLLAPSED", "RELIABILITY_SCI_COLOUR_EXCESS",
    "RELIABILITY_REVERSAL_UNTESTABLE", "RELIABILITY_REVERSAL_RETRIED",
    "RELIABILITY_REVERSAL_RECOVERED", "RELIABILITY_ZODI_ANCHOR_PINNED",
    "RELIABILITY_MOON_SHARE_PINNED", "RELIABILITY_DIFFUSE_OH_CAP_BINDING",
    "RELIABILITY_DIFFUSE_RATIO_PINNED", "RELIABILITY_SHAPE_BOUND_ACTIVE",
    "RELIABILITY_ERROR_MASK", "RELIABILITY_WARNING_MASK",
    "has_error", "has_warning", "constraint_bits", "sci_colour_excess",
    "SCI_COLOUR_BLUE_BAND", "SCI_COLOUR_RED_BAND", "SCI_COLOUR_EXCESS_MAX",
    "BINDING_RTOL",
    "RELIABILITY_BITS", "REVERSAL_MIN_COMPONENT_FRAC",
    "REVERSAL_MIN_SEPARATION", "REVERSAL_MIN_PIXELS",
    "DIFFUSE_COLLAPSED_SHARE", "DIFFUSE_COMPONENT_KEYS",
    "loglog_slope", "reversal_state", "diffuse_collapsed", "describe",
]
