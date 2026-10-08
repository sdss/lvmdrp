"""Fixed-weight scoring of a decomposition run, for A/B comparison.

Why this exists
---------------
``reduced_chi2`` in a run's META is computed with *that run's own* fit
weights. Every arm of the weighting / regularisation A/B changes those
weights, so their ``reduced_chi2`` values are not on the same scale and
comparing them measures the weighting change rather than the fit quality.
This module re-scores a finished run against ONE reference weighting that
depends only on the observed data, so any two arms over the same rows are
directly comparable.

The reference weighting is the absolute photon-noise ivar and nothing else:
no Huber down-weighting, no ``line_weight`` suppression of skyline pixels,
no per-LSF-channel renormalisation. Those three are fitting devices, and a
score built from them would reward an arm for the weights it chose rather
than for the spectrum it reproduced.

What it reports
---------------
``chi2flat_*``  the same residuals under UNIFORM weights. Needed whenever
            the arms differ in what THEY weighted by: an arm fitted with flat
            weights is at a disadvantage under ``chi2_*``, which is aligned
            with the photon-weighted arm's own objective, and vice versa. If
            an arm wins under both it has genuinely won; if each wins its own,
            the choice has to be made on physics, not on chi2.
``chi2_*``  band-resolved weighted mean of ``ivar * resid**2`` per pixel,
            over the three sensitivity arms and the full range. Row-
            normalised ivar (mean 1 over the row), so a row's score is a
            pure goodness-of-fit number, free of exposure time, fibre count
            and FACTOR. Summarised by MEDIAN and p90: a handful of failed
            rows put the mean three orders of magnitude above the median,
            where it measures the tail's depth rather than the fit.
``frac_*``  occupancy of each reliability bit: the error bits carry the
            diffuse-collapse and reversal rates, the warning bits the four
            constraint-activation rates.
``moon_share`` / ``zodi_int``  the moon/zodi partition, split two ways:
            by whether the zodi anchor was pinned (the anchor reproduces its
            own target, so pinned rows inflate any zodi agreement statistic)
            and by moon altitude (on moon-down rows the moon amplitude sits
            on its 0.02 floor and the block carries no information, so a
            median over all rows just reports the floor).

Arms are compared row-matched with :func:`compare_paired`: the per-row log
ratio, not the difference of two independently-taken medians. The row set
must be held fixed across arms -- dropping each arm's own failures moves the
score on its own and makes the comparison unattributable.

Scoring is read-only: it takes the finished FITS products and never refits.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

from .pixel_weights import SENS_ARMS, absolute_sensitivity, pixel_ivar
from . import reliability as _rel


# Reference weighting. Frozen deliberately: changing any of these invalidates
# comparison against previously recorded scores, so bump SCORER_VERSION too.
SCORER_VERSION = 1
REF_EXPTIME_S = 900.0
REF_FLOOR_FRAC = 0.05
REF_CLIP = None

# Column holding the number of co-added fibres, per flux extension. Needed
# only for the absolute score; the row-normalised score cancels it.
FIBRE_COLUMN = {"sci": "fibers_sci_used",
                "sky1": "fibers_sky_near_used",
                "sky2": "fibers_sky_far_used"}

# Non-overlapping bands from the sensitivity arms, split at the midpoint of
# each overlap so every pixel is counted exactly once.
def _disjoint_bands():
    edges = []
    for i, (name, lo, hi) in enumerate(SENS_ARMS):
        lo_eff = lo if i == 0 else 0.5 * (SENS_ARMS[i - 1][2] + lo)
        hi_eff = hi if i == len(SENS_ARMS) - 1 else 0.5 * (hi + SENS_ARMS[i + 1][1])
        edges.append((name, lo_eff, hi_eff))
    return tuple(edges)


BANDS = _disjoint_bands()

# The bits worth reporting as rates, in the order a report should read them.
_ERROR_BITS = (
    ("fit_failed", _rel.RELIABILITY_FIT_FAILED),
    ("reversed", _rel.RELIABILITY_REVERSED),
    ("diffuse_collapsed", _rel.RELIABILITY_DIFFUSE_COLLAPSED),
    ("sci_colour_excess", _rel.RELIABILITY_SCI_COLOUR_EXCESS),
)
_WARNING_BITS = (
    ("reversal_untestable", _rel.RELIABILITY_REVERSAL_UNTESTABLE),
    ("reversal_retried", _rel.RELIABILITY_REVERSAL_RETRIED),
    ("reversal_recovered", _rel.RELIABILITY_REVERSAL_RECOVERED),
    ("zodi_anchor_pinned", _rel.RELIABILITY_ZODI_ANCHOR_PINNED),
    ("moon_share_pinned", _rel.RELIABILITY_MOON_SHARE_PINNED),
    ("diffuse_oh_cap_binding", _rel.RELIABILITY_DIFFUSE_OH_CAP_BINDING),
    ("diffuse_ratio_pinned", _rel.RELIABILITY_DIFFUSE_RATIO_PINNED),
    ("shape_bound_active", _rel.RELIABILITY_SHAPE_BOUND_ACTIVE),
)


def reference_ivar(flux_row_fit_units, wave, *, sens, flux_scale, mask=None,
                   normalise=True, n_fibres=None):
    """The one weighting every arm is scored against.

    ``flux_row_fit_units`` is the flux as the fit saw it (physical * FACTOR);
    ``flux_scale`` undoes that so the photon model sees physical units.
    ``normalise=True`` puts the row's mean weight at 1, which cancels
    exposure time, fibre count and FACTOR exactly and is what makes two
    arms comparable. ``normalise=False`` keeps the absolute photon scale,
    where chi2 = 1 means the model is consistent with the noise of the
    actual co-added stack -- interpretable, but then the score also carries
    the row's exposure depth.
    """
    return pixel_ivar(
        flux_row_fit_units,
        wave,
        exptime=REF_EXPTIME_S,
        flux_scale=flux_scale,
        floor_frac=REF_FLOOR_FRAC,
        clip=REF_CLIP,
        sens=sens,
        normalise=normalise,
        n_fibres=n_fibres,
        mask=mask,
    )


def _band_masks(wave):
    return {name: (wave >= lo) & (wave < hi) for name, lo, hi in BANDS}


def score_decomposition(
    base_fits,
    decomp_fits,
    *,
    kind="sci",
    factor=1e14,
    rows=None,
    science_line_mask=None,
    data_dir=None,
    model_hdu="BESTFIT_LSF",
    moon_alt=None,
    moon_up_deg=10.0,
    n_fibres=None,
):
    """Re-score one finished decomposition against the reference weighting.

    Parameters
    ----------
    base_fits, decomp_fits
        The input cube and the ``*_decomp_{kind}_*.fits`` it produced.
    kind
        Which flux extension was fitted: ``sci``, ``sky1`` or ``sky2``.
    factor
        The ``FACTOR`` the run used, so the scorer reconstructs the flux in
        fit units exactly as the worker did.
    rows
        Optional row indices; default is every row in the product.
    science_line_mask
        Boolean array, True where a science emission line was masked out of
        the fit. Those pixels carry no fit information and are excluded.
        Pass the STATIC mask (zero velocity): it must be identical across
        arms, and the per-row Halpha refinement is not stored in the product.
    model_hdu
        ``BESTFIT_LSF`` is the LSF-refined model and is what the components
        sum to; ``BESTFIT`` is the pre-refinement seed and will score worse
        for reasons that have nothing to do with the arm.
    moon_alt
        Optional per-row moon altitude in degrees, from the INPUT cube's
        META (``moon_alt``), indexed over the full cube. Adds the moon-up /
        moon-down split; without it the moon statistics are dominated by the
        moon-down rows sitting on the amplitude floor.
    n_fibres
        Optional per-row co-added fibre count, from the INPUT cube's META
        (see ``FIBRE_COLUMN``). Adds ``chi2abs_*``, the same residuals under
        the UN-normalised photon weighting, where 1.0 means "consistent with
        the stack's own noise". Diagnostic only -- compare arms on the
        normalised score.

    Returns
    -------
    dict of scalars plus ``per_row`` arrays, ready to tabulate.
    """
    base_fits = Path(base_fits)
    decomp_fits = Path(decomp_fits)

    wave = np.asarray(fits.getdata(base_fits, "WAVE"), dtype=np.float64)
    sens = absolute_sensitivity(wave, data_dir=data_dir)
    bands = _band_masks(wave)

    flux_ext = {"sci": "FLUX_SCI", "sky1": "FLUX_SKY_NEAR", "sky2": "FLUX_SKY_FAR"}[kind]

    with fits.open(base_fits, memmap=True) as hb, fits.open(decomp_fits, memmap=True) as hd:
        flux_all = hb[flux_ext].data
        model_all = hd[model_hdu].data
        meta = hd["META"].data
        n_rows = int(model_all.shape[0])
        idx = np.arange(n_rows) if rows is None else np.asarray(rows, dtype=int)

        all_bands = list(bands.items()) + [("full", np.ones(wave.size, dtype=bool))]
        chi2 = {name: np.full(idx.size, np.nan) for name, _ in all_bands}
        chi2abs = {name: np.full(idx.size, np.nan) for name, _ in all_bands}
        chi2flat = {name: np.full(idx.size, np.nan) for name, _ in all_bands}
        # Fraction of the row measured below zero. Splits out the population
        # the `negative_flux` weighting choice can actually reach: it is
        # exclusively moon-down and blue, so an aggregate median hides it.
        neg_frac = np.full(idx.size, np.nan)
        for k, i in enumerate(idx):
            flux_row = np.asarray(flux_all[i], dtype=np.float64) * float(factor)
            model_row = np.asarray(model_all[i], dtype=np.float64)
            w = reference_ivar(
                flux_row, wave, sens=sens, flux_scale=float(factor),
                mask=science_line_mask,
            )
            finite = np.isfinite(flux_row)
            neg_frac[k] = (float(np.mean(flux_row[finite] < 0.0))
                           if finite.any() else np.nan)
            resid2 = (flux_row - model_row) ** 2
            good = np.isfinite(resid2) & np.isfinite(w) & (w > 0.0)
            if not good.any():
                continue
            w_abs = None
            if n_fibres is not None:
                _nf = float(np.asarray(n_fibres)[i])
                w_abs = reference_ivar(
                    flux_row, wave, sens=sens, flux_scale=float(factor),
                    mask=science_line_mask, normalise=False,
                    n_fibres=_nf if np.isfinite(_nf) and _nf > 0 else None,
                )
            for name, band in all_bands:
                sel = good & band
                den = float(np.sum(w[sel]))
                if den > 0.0:
                    chi2[name][k] = float(np.sum(w[sel] * resid2[sel])) / den
                if sel.any():
                    # Uniform weights on the same pixel set. Normalised by the
                    # row's own mean so it is comparable across rows the way
                    # the row-normalised photon score is.
                    chi2flat[name][k] = float(np.mean(resid2[sel]))
                if w_abs is not None and sel.any():
                    chi2abs[name][k] = float(np.mean(w_abs[sel] * resid2[sel]))

        bits = np.asarray(meta["reliability"], dtype=np.int64)[idx]
        moon_share = np.asarray(meta["moon_share"], dtype=np.float64)[idx]
        zodi_int = np.asarray(meta["zodi_int"], dtype=np.float64)[idx]

    # A band can legitimately be entirely masked out; nanmedian of nothing is
    # NaN, which is the right answer, so do not let numpy shout about it.
    warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)

    out = {
        "scorer_version": SCORER_VERSION,
        "n_rows": int(idx.size),
        "kind": kind,
        "run": decomp_fits.name,
    }
    for name in list(bands) + ["full"]:
        v = chi2[name]
        out[f"chi2_{name}_median"] = float(np.nanmedian(v))
        out[f"chi2_{name}_p90"] = float(np.nanpercentile(v, 90))
        out[f"chi2flat_{name}_median"] = float(np.nanmedian(chi2flat[name]))
        if n_fibres is not None:
            out[f"chi2abs_{name}_median"] = float(np.nanmedian(chi2abs[name]))
    for label, bit in _ERROR_BITS + _WARNING_BITS:
        out[f"frac_{label}"] = float(np.mean((bits & bit) != 0))
    out["frac_any_error"] = float(np.mean((bits & _rel.RELIABILITY_ERROR_MASK) != 0))
    out["frac_any_warning"] = float(np.mean((bits & _rel.RELIABILITY_WARNING_MASK) != 0))

    pinned = (bits & _rel.RELIABILITY_ZODI_ANCHOR_PINNED) != 0
    out["moon_share_median"] = float(np.nanmedian(moon_share))
    out["zodi_int_median"] = float(np.nanmedian(zodi_int))
    dark = np.isfinite(neg_frac) & (neg_frac > 0.01)
    splits = [("pinned", pinned), ("free", ~pinned)]
    if dark.any():
        splits += [("negflux", dark)]
    out["frac_rows_negflux"] = float(np.mean(np.isfinite(neg_frac) & (neg_frac > 0.01)))
    out["neg_frac_median"] = float(np.nanmedian(neg_frac))
    if moon_alt is not None:
        up = np.asarray(moon_alt, dtype=np.float64)[idx] > float(moon_up_deg)
        splits += [("moonup", up), ("moondown", ~up)]
        out["frac_moon_up"] = float(np.mean(up))
    for label, sel in splits:
        out[f"zodi_int_median_{label}"] = (
            float(np.nanmedian(zodi_int[sel])) if sel.any() else np.nan
        )
        out[f"moon_share_median_{label}"] = (
            float(np.nanmedian(moon_share[sel])) if sel.any() else np.nan
        )
        for name in ("b", "full"):
            out[f"chi2_{name}_median_{label}"] = (
                float(np.nanmedian(chi2[name][sel])) if sel.any() else np.nan
            )

    out["per_row"] = {"idx": idx, "bits": bits, "moon_share": moon_share,
                      "zodi_int": zodi_int, "neg_frac": neg_frac,
                      **{f"chi2_{k}": v for k, v in chi2.items()},
                      **{f"chi2flat_{k}": v for k, v in chi2flat.items()},
                      **({f"chi2abs_{k}": v for k, v in chi2abs.items()}
                         if n_fibres is not None else {})}
    return out


def compare_paired(baseline, arm, *, bands=("b", "r", "z", "full"), prefix="chi2"):
    """Row-matched chi2 comparison: median per-row ratio, arm / baseline.

    Paired because the row-to-row spread of chi2 is far larger than the
    effect sizes being chased, so a difference of two independently-taken
    medians is mostly noise. Rows scored in only one of the two arms are
    dropped from the pair, and the count of those is reported: if it is not
    tiny, the arms are no longer being compared on the same data.
    """
    a_rows, b_rows = arm["per_row"], baseline["per_row"]
    if not np.array_equal(a_rows["idx"], b_rows["idx"]):
        raise ValueError("arms were scored over different row sets")
    out = {"n_rows": int(b_rows["idx"].size)}
    for name in bands:
        b, a = b_rows[f"{prefix}_{name}"], a_rows[f"{prefix}_{name}"]
        ok = np.isfinite(b) & np.isfinite(a) & (b > 0) & (a > 0)
        out[f"n_dropped_{name}"] = int((~ok).sum())
        if not ok.any():
            out[f"ratio_{name}_median"] = np.nan
            continue
        lr = np.log(a[ok] / b[ok])
        out[f"ratio_{name}_median"] = float(np.exp(np.median(lr)))
        out[f"ratio_{name}_p16"] = float(np.exp(np.percentile(lr, 16)))
        out[f"ratio_{name}_p84"] = float(np.exp(np.percentile(lr, 84)))
        # Sign test: how often did the arm actually improve the row?
        out[f"frac_improved_{name}"] = float(np.mean(a[ok] < b[ok]))
    return out


def compare(baseline, arm, *, keys=None):
    """Tabulate one arm against a baseline score dict, as (value, delta, %)."""
    if keys is None:
        keys = [k for k in baseline
                if k.startswith(("chi2_", "frac_", "moon_share_", "zodi_int_"))
                and isinstance(baseline[k], float)]
    lines = []
    for k in keys:
        b, a = baseline[k], arm.get(k, np.nan)
        pct = 100.0 * (a - b) / b if np.isfinite(b) and b != 0 else np.nan
        lines.append((k, b, a, a - b, pct))
    return lines


def format_comparison(baseline, arm, *, keys=None):
    rows = compare(baseline, arm, keys=keys)
    width = max(len(r[0]) for r in rows)
    head = f"{'metric'.ljust(width)}  {'baseline':>12}  {'arm':>12}  {'delta':>12}  {'%':>8}"
    body = [
        f"{k.ljust(width)}  {b:12.6g}  {a:12.6g}  {d:12.6g}  {p:8.2f}"
        for k, b, a, d, p in rows
    ]
    return "\n".join([head, "-" * len(head), *body])


__all__ = [
    "BANDS", "SCORER_VERSION", "compare", "compare_paired",
    "format_comparison", "reference_ivar", "score_decomposition",
]
