"""High-level inference API: predict SCI sky spectrum from minimal per-row inputs.

Wraps

- context construction from raw pointings/time (using astropy),
- ensemble prediction via
  :func:`mlp_predictor.trainer.predict_sci_coefficients_default`,
- per-row confidence via the ensemble spread across seeds,

so callers can go from ``(mjd, sci_ra, sci_dec, sky_e_ra, sky_e_dec,
sky_w_ra, sky_w_dec, coef_e, coef_w)`` straight to a predicted science-arm
coefficient vector plus a per-row confidence estimate.  The near-vs-far arm
assignment (which the model actually consumes) is computed internally from
angular separation, so callers do not have to know which of the 39 context
features enter which specialised branch of the model, nor which arm is
"near" in the training corpus's convention.
"""

from __future__ import annotations

from typing import Mapping, Any

import numpy as np
from astropy.table import Table

from . import data, config as _cfg_module
from .trainer import predict_sci_coefficients_default


# Default context feature list.  Matches the deployed training pipeline:
# 28 base features that expand to 39 after the two augment stages (ecliptic
# geometry and physics-prior moon-scatter proxies).
_DEFAULT_CONTEXT_COLUMNS = tuple(_cfg_module.DataConfig().context_columns)


def _as_1d_f64(arr, name):
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError(f"{name} is empty.")
    return a


def _angular_separation_deg(ra1, dec1, ra2, dec2):
    """Vincenty spherical separation in degrees, per-row, no astropy roundtrip."""
    ra1_r, dec1_r = np.deg2rad(ra1), np.deg2rad(dec1)
    ra2_r, dec2_r = np.deg2rad(ra2), np.deg2rad(dec2)
    dra = ra2_r - ra1_r
    n = np.hypot(np.cos(dec2_r) * np.sin(dra),
                 np.cos(dec1_r) * np.sin(dec2_r)
                 - np.sin(dec1_r) * np.cos(dec2_r) * np.cos(dra))
    d = (np.sin(dec1_r) * np.sin(dec2_r)
         + np.cos(dec1_r) * np.cos(dec2_r) * np.cos(dra))
    return np.rad2deg(np.arctan2(n, d))


def _compute_moon_phase_deg(obstime_mjd: np.ndarray) -> np.ndarray:
    """Per-row moon phase in degrees, in the convention of META ``MOON_PHASE``.

    0 = new, 90 = first quarter, 180 = full, 270 = last quarter: the lunar
    elongation from the Sun while waxing (Moon east of the Sun), 360 minus it
    while waning.  ``moon_fli = (1 - cos(phase)) / 2`` then holds, and so do the
    ``moon_phase_sin`` / ``moon_phase_cos`` features the ensemble was trained on.

    Until 2026-09-24 this returned ``180 - elongation`` on [0, 180]: INVERTED
    (0 = full moon) and UNSIGNED (waxing and waning indistinguishable), so any
    caller that did not pass ``moon_phase_deg`` -- including the example
    notebook -- fed the model moon_fli and moon_phase_cos flipped relative to
    training.  Checked against the 1.3.2 corpus META on 2000 rows: median
    |diff| 0.003 deg, p99 0.005 deg (one row near new moon flips sign, where
    the waxing/waning call is a few degrees from ambiguous).
    """
    from astropy.coordinates import GeocentricTrueEcliptic, get_body, get_sun
    from astropy.time import Time
    import astropy.units as u

    lco = data._lco_earth_location()
    time = Time(np.asarray(obstime_mjd, dtype=np.float64), format="mjd", scale="utc")
    sun = get_sun(time)
    moon = get_body("moon", time, location=lco)
    elong_deg = moon.separation(sun).to_value(u.deg)
    ecl = GeocentricTrueEcliptic(equinox=time)
    east = (((moon.transform_to(ecl).lon.deg - sun.transform_to(ecl).lon.deg)
             + 180.0) % 360.0 - 180.0) > 0.0
    return np.where(east, elong_deg, 360.0 - elong_deg).astype(np.float64)


def build_triplet_from_pointings(
    *,
    obstime_mjd,
    sci_ra, sci_dec,
    sky_near_ra, sky_near_dec,
    sky_far_ra,  sky_far_dec,
    moon_phase_deg=None,
    sky_near_label=None,
    sky_far_label=None,
    context_columns=None,
    moon_model_inputs=None,
    verbose=True,
) -> dict:
    """Build a triplet dict with 39-dim per-arm ctx from minimal inputs.

    All ``ra`` / ``dec`` are in degrees (ICRS).  ``obstime_mjd`` is UT MJD.
    All arrays are broadcast to a common length (per-row); scalars are
    accepted for single-row predictions.

    Parameters
    ----------
    obstime_mjd:
        UT MJD of the observation.  Everything time-related derives from this
        (cyclic time features, solar-activity indices from the F10.7/Kp cache,
        moon/sun ephemerides).
    sci_ra, sci_dec:
        Pointing of the science fiber (degrees).
    sky_near_ra, sky_near_dec, sky_far_ra, sky_far_dec:
        Pointings of the two sky arms (degrees).  Which arm is "near" vs "far"
        is determined by the caller — the training pipeline defines "near" as
        the sky arm at smaller angular separation from the science pointing.
    moon_phase_deg:
        Optional per-row moon phase in degrees using the convention
        ``0 = new moon`` / ``180 = full moon``.  If omitted, computed from
        astropy sun/moon ephemerides.
    sky_near_label, sky_far_label:
        Optional per-row string labels 'SKYE' or 'SKYW' selecting the
        east/west assignment for the ``ew`` context feature.  Defaults to
        ('SKYE', 'SKYW') — this only affects the sign of a single ctx dim
        and its impact on predictions is small.
    context_columns:
        Override the default 28-feature base list.  Must be a subset of what
        ``mlp_predictor.data`` knows how to compute; see
        :attr:`DataConfig.context_columns` for the deployed default.

    Returns
    -------
    dict
        Keys: ``ctx_names``, ``ctx_sci`` / ``ctx_near`` / ``ctx_far`` (each
        of shape ``(n_rows, 39)``), and ``obstime_mjd``.  Ready to hand to
        :func:`predict_sky_from_minimal_inputs` or, together with the two
        sky-arm ``coef`` vectors, to
        :func:`~mlp_predictor.trainer.predict_sci_coefficients_default`
        directly.
    """
    obstime_mjd = _as_1d_f64(obstime_mjd, "obstime_mjd")
    n_rows = obstime_mjd.size
    _bcast = lambda a, name: np.broadcast_to(_as_1d_f64(a, name), (n_rows,)).astype(np.float64).copy()

    sci_ra_arr  = _bcast(sci_ra,  "sci_ra")
    sci_dec_arr = _bcast(sci_dec, "sci_dec")
    n1_ra  = _bcast(sky_near_ra,  "sky_near_ra")
    n1_dec = _bcast(sky_near_dec, "sky_near_dec")
    f1_ra  = _bcast(sky_far_ra,   "sky_far_ra")
    f1_dec = _bcast(sky_far_dec,  "sky_far_dec")

    if moon_phase_deg is None:
        moon_phase_arr = _compute_moon_phase_deg(obstime_mjd)
    else:
        moon_phase_arr = _bcast(moon_phase_deg, "moon_phase_deg")

    if sky_near_label is None:
        near_lbl = np.array(["SKYE"] * n_rows)
    else:
        near_lbl = np.asarray(sky_near_label).astype(str)
        if near_lbl.size == 1:
            near_lbl = np.repeat(near_lbl, n_rows)
    if sky_far_label is None:
        far_lbl = np.array(["SKYW"] * n_rows)
    else:
        far_lbl = np.asarray(sky_far_label).astype(str)
        if far_lbl.size == 1:
            far_lbl = np.repeat(far_lbl, n_rows)

    meta = Table({
        "OBSTIME":         obstime_mjd,
        "SCI_RA":          sci_ra_arr,
        "SCI_DEC":         sci_dec_arr,
        "SKY_NEAR_RA":     n1_ra,
        "SKY_NEAR_DEC":    n1_dec,
        "SKY_FAR_RA":      f1_ra,
        "SKY_FAR_DEC":     f1_dec,
        "MOON_PHASE":      moon_phase_arr,
        "SKY_NEAR_LABEL":  near_lbl,
        "SKY_FAR_LABEL":   far_lbl,
    })

    ctx_cols = list(context_columns) if context_columns is not None else list(_DEFAULT_CONTEXT_COLUMNS)
    ctx_sci,  ctx_names   = data._build_context_matrix(meta, ctx_cols, "sci")
    ctx_near, ctx_names_a = data._build_context_matrix(meta, ctx_cols, "sky1")
    ctx_far,  ctx_names_b = data._build_context_matrix(meta, ctx_cols, "sky2")
    if ctx_names != ctx_names_a or ctx_names != ctx_names_b:
        raise RuntimeError(
            "Context feature ordering diverged between arms — this is a bug in "
            "_build_context_matrix and should never happen with matching context_columns.")

    triplet: dict = {
        "ctx_names":   list(ctx_names),
        "ctx_sci":     np.asarray(ctx_sci,  dtype=np.float32),
        "ctx_near":    np.asarray(ctx_near, dtype=np.float32),
        "ctx_far":     np.asarray(ctx_far,  dtype=np.float32),
        "obstime_mjd": obstime_mjd,
        # Per-arm RA/Dec needed by the ecliptic augment.
        "sci_ra":  sci_ra_arr,  "sci_dec":  sci_dec_arr,
        "near_ra": n1_ra,       "near_dec": n1_dec,
        "far_ra":  f1_ra,       "far_dec":  f1_dec,
    }
    data._augment_triplet_with_ecliptic(triplet, force=True)
    data._augment_triplet_with_physics_priors(triplet, force=True)
    if moon_model_inputs:
        _moon_model_augment_direct(triplet, verbose=verbose, **moon_model_inputs)
    return triplet


# --- moon/zodi model features, computed directly (no corpus cache) ----------
# The deployed ensemble's context is 42 features: 28 base + 3 ecliptic + 8
# physics priors + `moon_model_log_ratio` + `zodi_po_log10` + `moon_frac_po`.
# The last three came in on 2026-09-09 and are normally read from a per-corpus
# cache (`mlp_predictor.moon_model_cache`), which is keyed to a corpus stack
# and so is unavailable for the arbitrary pointings this module serves.  They
# are pure geometry plus the frozen physical model, so they can be evaluated
# per row instead -- that is what this does, mirroring
# `data._augment_triplet_with_moon_model` column for column.
#
# The extra cost over the cache path is one `MoonZodiPhysicalModel.predict` and
# one `geometry_amplitude_prior` per arm per row.
def _moon_model_augment_direct(triplet, *, wave, lsf_near, lsf_far, lsf_sci,
                               date_obs, expnum=None, exposure_seconds=900.0,
                               verbose=True, zodi_correction="none"):
    """Append the three moon/zodi-model ctx features, computed per row.

    Mirrors `data._augment_triplet_with_moon_model` exactly: one shared
    `moon_model_log_ratio` (identical in all three arms, gated to 0.0 where the
    moon is down or the ratio is not finite) followed by per-arm
    `zodi_po_log10` and `moon_frac_po` (both 0.0 where the physics-only model
    has no usable prediction).
    """
    from sky_decomp.moon_zodi_model import (MoonZodiObservation,
                                            MoonZodiPhysicalModel,
                                            geometry_amplitude_prior)
    _FIT_FLUX_SCALE = 1.0e14          # moon_model_cache.FIT_FLUX_SCALE
    _EXPOSURE_SOURCE = "assumed_900s"  # validated against a closed set
    names = list(triplet["ctx_names"])
    n_rows = int(np.asarray(triplet["ctx_sci"]).shape[0])
    _date = np.atleast_1d(np.asarray(date_obs, dtype=object))
    if _date.size == 1 and n_rows > 1:
        _date = np.repeat(_date, n_rows)
    if _date.size != n_rows:
        raise ValueError(f"date_obs has {_date.size} entries for {n_rows} rows")
    _exp = (np.full(n_rows, -1, dtype=np.int64) if expnum is None
            else np.broadcast_to(np.asarray(expnum, dtype=np.int64),
                                 (n_rows,)))
    _wave = np.asarray(wave, dtype=np.float64)
    _arms = {
        "near": ("sky_near", triplet["near_ra"], triplet["near_dec"], lsf_near),
        "far":  ("sky_far",  triplet["far_ra"],  triplet["far_dec"],  lsf_far),
        "sci":  ("sci",      triplet["sci_ra"],  triplet["sci_dec"],  lsf_sci),
    }
    model = MoonZodiPhysicalModel()
    moon_total, zodi_po, frac_po = {}, {}, {}
    for arm, (role, ra, dec, lsf) in _arms.items():
        _lsf = np.asarray(lsf, dtype=np.float64)
        mt = np.full(n_rows, np.nan)
        zp = np.full(n_rows, np.nan)
        fp = np.full(n_rows, np.nan)
        for i in range(n_rows):
            _l = _lsf if _lsf.ndim == 1 else _lsf[i]
            obs = MoonZodiObservation(
                expnum=int(_exp[i]),
                date_obs=str(_date[i]).strip(),
                role=role,
                target_ra_deg=float(np.asarray(ra, dtype=np.float64).ravel()[i]),
                target_dec_deg=float(np.asarray(dec, dtype=np.float64).ravel()[i]),
                exposure_seconds=float(exposure_seconds),
                exposure_seconds_source=_EXPOSURE_SOURCE)
            try:
                pred = model.predict(_wave, _l, obs,
                                     physical_to_fit_flux_scale=_FIT_FLUX_SCALE)
                # moon_model_cache uses nansum(pred.moon), not an attribute.
                mt[i] = float(np.nansum(np.asarray(pred.moon, dtype=np.float64)))
            except Exception:
                pass
            try:
                _f, _z, _ = geometry_amplitude_prior(
                    _wave, _l, obs, physical_to_fit_flux_scale=_FIT_FLUX_SCALE,
                    zodi_correction=zodi_correction)
                zp[i] = float(_z); fp[i] = float(_f)
            except Exception:
                pass
        moon_total[arm], zodi_po[arm], frac_po[arm] = mt, zp, fp
    # `moon_alt` is already a base ctx column, so take the gate from there
    # rather than recomputing an ephemeris that could disagree with it.
    if "moon_alt" not in names:
        raise RuntimeError("moon_alt is not in the base context; the "
                           "moon-model gate cannot be reproduced")
    _alt = np.asarray(triplet["ctx_sci"], dtype=np.float64)[:, names.index("moon_alt")]
    with np.errstate(divide="ignore", invalid="ignore"):
        _num, _den = moon_total["sci"], moon_total["near"]
        _good = (np.isfinite(_num) & np.isfinite(_den)
                 & (_num > 1e-30) & (_den > 1e-30))
        ratio = np.where(_good, np.log10(np.where(_good, _num / _den, 1.0)), np.nan)
    col = np.where((_alt > 0.0) & np.isfinite(ratio), ratio, 0.0).astype(np.float32)
    _extra = {}
    for _key, _arm in (("ctx_near", "near"), ("ctx_far", "far"), ("ctx_sci", "sci")):
        _zp, _fp = zodi_po[_arm], frac_po[_arm]
        _ok = np.isfinite(_zp) & (_zp > 0.0) & np.isfinite(_fp) & (_fp > 0.0)
        _extra[_key] = np.column_stack([
            np.where(_ok, np.log10(np.where(_zp > 0.0, _zp, 1.0)), 0.0),
            np.where(_ok, _fp, 0.0)]).astype(np.float32)
    for key in ("ctx_near", "ctx_far", "ctx_sci"):
        triplet[key] = np.hstack([np.asarray(triplet[key], dtype=np.float32),
                                  col[:, None], _extra[key]])
    triplet["ctx_names"] = (names + list(data.MOON_MODEL_FEATURE_NAMES)
                            + list(data.ZODI_CEILING_FEATURE_NAMES))
    if verbose:
        _n_gate = int((col == 0.0).sum())
        _f = _extra["ctx_sci"][:, 1]
        print(f"  moon-model augment (direct, no cache): n_ctx now "
              f"{len(triplet['ctx_names'])}; {_n_gate}/{col.size} rows gated to "
              f"0; {int((_f <= 0.0).sum())}/{_f.size} rows have no physics-only "
              f"prediction")
    return triplet


# Fraction of a coefficient's training-set UPPER BOUND below which the fitted
# or predicted value counts as switched off.  The decomposition gate
# (`data.diffuse_zeroed_mask`) uses 1e-3 of the corpus MEDIAN instead; inference
# has no corpus, but the ensemble carries `coef_upper_bound`, and the population
# is bimodal (values ~1e-9 against ~1e-2), so any threshold in the empty middle
# picks the same rows.  Checked against the median-based gate on the telluric
# every10 corpus: identical row sets.
PREDICTION_OFF_FRAC_OF_UPPER = 1.0e-6


def _upper_bound_by_name(coef_names, coef_upper_bound, group_indices=None):
    """Per-coefficient upper bounds keyed by NAME.

    The ensemble stores them as ``{group: per-coefficient array}`` (moon 15,
    zodi 5, continuum 3, mesospheric 358, ionospheric 4, ...), so the group's
    own coefficient indices are needed to attach a bound to a name.  A plain
    array over all coefficients is accepted too.
    """
    names = [str(n) for n in coef_names]
    if coef_upper_bound is None:
        return {}
    if isinstance(coef_upper_bound, Mapping):
        if not group_indices:
            return {}
        out = {}
        for group, bounds in coef_upper_bound.items():
            index = np.asarray(group_indices.get(group, ()), dtype=int).ravel()
            values = np.asarray(bounds, dtype=np.float64).ravel()
            if index.size != values.size:
                continue
            for position, value in zip(index, values):
                if 0 <= int(position) < len(names):
                    out[names[int(position)]] = float(value)
        return out
    values = np.asarray(coef_upper_bound, dtype=np.float64).ravel()
    if values.size != len(names):
        return {}
    return {name: float(value) for name, value in zip(names, values)}


def prediction_reliability(coef, coef_names, coef_upper_bound=None,
                           group_indices=None,
                           off_frac=PREDICTION_OFF_FRAC_OF_UPPER):
    """Reliability bits computable from PREDICTED coefficients alone.

    Production sky subtraction cannot drop a row, so the pathologies that the
    corpus build gates away have to be reported instead -- see
    `sky_decomp.reliability` for the bit values and the error/warning split.

    Only the diffuse-collapse test is available without a basis: a reversal is a
    property of the reconstructed moon and zodi CONTINUA, so it lives in
    `reliability_from_components`, which the caller invokes with the components
    it has already reconstructed to build the sky spectrum.  The constraint
    warnings (anchor pinned, caps binding) have no meaning here at all: there is
    no QP at prediction time.  They describe how the row's TRAINING TARGETS were
    shaped, which is a property of the corpus, not of this prediction.

    Returns ``(n_rows,)`` int32 bits, or -1 per row when the test could not be
    evaluated at all (no usable upper-bound reference).  -1 rather than 0
    deliberately: a test that did not run must not read as a clean row.
    """
    from sky_decomp import reliability as rel

    coef_arr = np.atleast_2d(np.asarray(coef, dtype=np.float64))
    names = [str(n) for n in coef_names]
    if coef_arr.shape[1] != len(names):
        raise ValueError(
            f"coef has {coef_arr.shape[1]} columns for {len(names)} names")
    bits = np.zeros(coef_arr.shape[0], dtype=np.int32)
    upper = _upper_bound_by_name(names, coef_upper_bound, group_indices)
    try:
        index = [names.index(n) for n in data.DIFFUSE_COMPONENT_NAMES]
    except ValueError:
        return np.full(coef_arr.shape[0], -1, dtype=np.int32)
    scale = np.array(
        [upper.get(names[i], float("nan")) for i in index], dtype=np.float64)
    if not np.all(np.isfinite(scale) & (scale > 0.0)):
        return np.full(coef_arr.shape[0], -1, dtype=np.int32)
    collapsed = np.all(
        coef_arr[:, index] < float(off_frac) * scale[None, :], axis=1)
    bits[collapsed] |= rel.RELIABILITY_DIFFUSE_COLLAPSED
    return bits


def reliability_from_components(components, wave):
    """Reliability bits that need the RECONSTRUCTED components of one row.

    Call this with the components dict already built to make the sky spectrum
    (`data.reconstruct_with_lsf`) and OR the result into the ``reliability``
    column from `predict_sky_from_minimal_inputs`.  Kept out of the predict
    call so inference never has to build a basis it does not otherwise need.
    """
    from sky_decomp import reliability as rel

    log_wave = np.log(np.asarray(wave, dtype=np.float64))
    is_reversed, testable, info = rel.reversal_state(components, log_wave)
    bits = 0
    if is_reversed:
        bits |= rel.RELIABILITY_REVERSED
    if not testable:
        bits |= rel.RELIABILITY_REVERSAL_UNTESTABLE
    return np.int32(bits), info


def predict_sky_from_minimal_inputs(
    ensemble: Mapping[str, Any],
    *,
    obstime_mjd,
    sci_ra, sci_dec,
    sky_e_ra, sky_e_dec,
    sky_w_ra, sky_w_dec,
    coef_e, coef_w,
    moon_phase_deg=None,
    return_per_seed: bool = False,
    # Moon/zodi-model ctx inputs.  Named _e/_w like coef_e/coef_w because they
    # are swapped in lockstep with the near-vs-far reassignment below -- an LSF
    # that did not follow its own arm would pair each row's geometry with the
    # other arm's resolution.
    wave=None, lsf_e=None, lsf_w=None, lsf_sci=None,
    date_obs=None, expnum=None, exposure_seconds=900.0,
    verbose=True,
) -> dict:
    """Predict the science-arm decomposition coefficients + a confidence score.

    Parameters
    ----------
    ensemble:
        The dict returned by :func:`~mlp_predictor.serialization.load_ensemble`.
        Must have ``is_ensemble=True``.
    obstime_mjd, sci_ra, sci_dec:
        UT MJD and science-fiber pointing (degrees, ICRS).  Everything time-
        related derives from ``obstime_mjd`` (cyclic time features, solar-
        activity indices, moon/sun ephemerides).
    sky_e_ra, sky_e_dec, sky_w_ra, sky_w_dec:
        Pointings of the SkyE and SkyW fibre bundles (degrees, ICRS).  The
        function computes the angular separation of each arm to the science
        pointing per row and internally assigns the closer one as "near" and
        the farther one as "far" — the sign convention the training corpus
        uses (see training notebook \u00a72.4).
    coef_e, coef_w:
        The SkyE and SkyW decomposition coefficient vectors, shape ``(n_rows,
        n_coef)`` where ``n_coef`` matches ``len(ensemble['coef_names'])``.
        These come from running the QP decomposition on the two sky-arm
        spectra separately.  They are swapped internally in lockstep with the
        near-vs-far reassignment above.
    moon_phase_deg:
        Optional per-row moon phase in degrees (0 = new moon, 180 = full).
        Computed from astropy sun/moon ephemerides when omitted.
    return_per_seed:
        If True, additionally include the ``(n_seeds, n_rows, n_coef)`` array
        of per-seed predictions under key ``'per_seed'``.

    Returns
    -------
    dict
        Keys:

        - ``coef``:        ``(n_rows, n_coef)`` predicted SCI coefficient
          vector (ensemble mean, with the Jensen post-training lift applied).
        - ``coef_std``:    ``(n_rows, n_coef)`` per-coefficient std across
          the 10 ensemble members (epistemic uncertainty, physical units).
        - ``confidence``:  ``(n_rows,)`` scalar confidence in ``(0, 1]``,
          computed as ``1 / (1 + median_k(coef_std_k / |coef_k|))``.
          Higher = more agreement across seeds = more confident.
        - ``coef_names``:  the list of coefficient names matching the column
          order of ``coef`` and ``coef_std``.
        - ``triplet``:     the built context dict (useful for debugging).
        - ``near_is_east``: ``(n_rows,)`` bool, ``True`` where the SkyE arm
          was closer to the science pointing than SkyW.
        - ``per_seed`` (optional): raw per-seed predictions.

    Notes
    -----
    * The "confidence" summary is a simple monotone rescaling of the median
      relative std across coefficient dimensions.  For detailed downstream
      logic use ``coef_std`` directly.
    * All Jensen post-training bias corrections are applied inside
      :func:`predict_sci_coefficients_default`, so ``coef`` is already the
      final calibrated prediction.
    """
    if not ensemble.get("is_ensemble", False):
        raise ValueError("`ensemble` must be an ensemble artifact (is_ensemble=True).")

    obstime_mjd_arr = _as_1d_f64(obstime_mjd, "obstime_mjd")
    n_rows = obstime_mjd_arr.size
    _bcast = lambda a, name: np.broadcast_to(
        _as_1d_f64(a, name), (n_rows,)).astype(np.float64).copy()

    sci_ra_arr  = _bcast(sci_ra,  "sci_ra")
    sci_dec_arr = _bcast(sci_dec, "sci_dec")
    e_ra_arr    = _bcast(sky_e_ra,  "sky_e_ra")
    e_dec_arr   = _bcast(sky_e_dec, "sky_e_dec")
    w_ra_arr    = _bcast(sky_w_ra,  "sky_w_ra")
    w_dec_arr   = _bcast(sky_w_dec, "sky_w_dec")

    # Per-row near/far assignment by angular separation to the science pointing.
    sep_e = _angular_separation_deg(sci_ra_arr, sci_dec_arr, e_ra_arr, e_dec_arr)
    sep_w = _angular_separation_deg(sci_ra_arr, sci_dec_arr, w_ra_arr, w_dec_arr)
    near_is_east = sep_e <= sep_w

    sky_near_ra    = np.where(near_is_east, e_ra_arr,   w_ra_arr)
    sky_near_dec   = np.where(near_is_east, e_dec_arr,  w_dec_arr)
    sky_far_ra     = np.where(near_is_east, w_ra_arr,   e_ra_arr)
    sky_far_dec    = np.where(near_is_east, w_dec_arr,  e_dec_arr)
    sky_near_label = np.where(near_is_east, "SKYE", "SKYW")
    sky_far_label  = np.where(near_is_east, "SKYW", "SKYE")

    _mm_inputs = None
    if wave is not None:
        _missing = [k for k, v in (("lsf_e", lsf_e), ("lsf_w", lsf_w),
                                   ("lsf_sci", lsf_sci), ("date_obs", date_obs))
                    if v is None]
        if _missing:
            raise ValueError(f"wave= was given, so the moon/zodi-model context "
                             f"features can be computed, but {_missing} "
                             f"are missing.")
        def _swap(_e, _w):
            _e = np.asarray(_e, dtype=np.float64)
            _w = np.asarray(_w, dtype=np.float64)
            if _e.ndim == 1 and _w.ndim == 1:      # one LSF for every row
                return (_e, _w) if bool(np.all(near_is_east)) else (
                    (_w, _e) if not bool(np.any(near_is_east)) else
                    (np.where(near_is_east[:, None], _e[None, :], _w[None, :]),
                     np.where(near_is_east[:, None], _w[None, :], _e[None, :])))
            _e2 = np.broadcast_to(_e, (near_is_east.size, _e.shape[-1]))
            _w2 = np.broadcast_to(_w, (near_is_east.size, _w.shape[-1]))
            return (np.where(near_is_east[:, None], _e2, _w2),
                    np.where(near_is_east[:, None], _w2, _e2))
        _lsf_near, _lsf_far = _swap(lsf_e, lsf_w)
        _mm_inputs = dict(wave=wave, lsf_near=_lsf_near, lsf_far=_lsf_far,
                          lsf_sci=lsf_sci, date_obs=date_obs, expnum=expnum,
                          exposure_seconds=exposure_seconds,
                          # the correction the ensemble's targets were anchored with
                          zodi_correction=str(ensemble.get("zodi_correction", "none")))

    triplet = build_triplet_from_pointings(
        obstime_mjd=obstime_mjd_arr,
        sci_ra=sci_ra_arr, sci_dec=sci_dec_arr,
        sky_near_ra=sky_near_ra, sky_near_dec=sky_near_dec,
        sky_far_ra=sky_far_ra,   sky_far_dec=sky_far_dec,
        moon_phase_deg=moon_phase_deg,
        sky_near_label=sky_near_label, sky_far_label=sky_far_label,
        moon_model_inputs=_mm_inputs, verbose=verbose,
    )
    _need = [n for n in ensemble["ctx_names"] if n not in triplet["ctx_names"]]
    if _need:
        raise ValueError(
            f"this ensemble needs {len(ensemble['ctx_names'])} context features "
            f"but only {len(triplet['ctx_names'])} could be built; missing "
            f"{_need}. Those come from the moon/zodi physical model, so pass "
            f"wave=, lsf_e=, lsf_w=, lsf_sci= and date_obs= (and optionally "
            f"expnum=, exposure_seconds=) to compute them for these pointings.")

    n_coef = len(ensemble["coef_names"])
    coef_e_arr = np.asarray(coef_e, dtype=np.float32)
    coef_w_arr = np.asarray(coef_w, dtype=np.float32)
    if coef_e_arr.ndim == 1:
        coef_e_arr = coef_e_arr.reshape(1, -1)
    if coef_w_arr.ndim == 1:
        coef_w_arr = coef_w_arr.reshape(1, -1)
    for name, arr in (("coef_e", coef_e_arr), ("coef_w", coef_w_arr)):
        if arr.shape[1] != n_coef:
            raise ValueError(
                f"{name} has {arr.shape[1]} columns but the ensemble expects "
                f"{n_coef} (coef_names length).")
    # Swap E/W into near/far to match the pointings above.
    near_mask = near_is_east[:, None]
    coef_near_arr = np.where(near_mask, coef_e_arr, coef_w_arr).astype(np.float32)
    coef_far_arr  = np.where(near_mask, coef_w_arr, coef_e_arr).astype(np.float32)

    ctx_near = triplet["ctx_near"]
    ctx_far  = triplet["ctx_far"]
    ctx_sci  = triplet["ctx_sci"]

    per_seed = []
    for member in ensemble["members"]:
        pred = predict_sci_coefficients_default(
            member,
            coef_near_phys=coef_near_arr, coef_far_phys=coef_far_arr,
            ctx_near_phys=ctx_near, ctx_far_phys=ctx_far, ctx_sci_phys=ctx_sci,
        )
        per_seed.append(np.asarray(pred, dtype=np.float64))
    per_seed_arr = np.stack(per_seed, axis=0)  # (n_seeds, n_rows, n_coef)

    coef_mean = per_seed_arr.mean(axis=0).astype(np.float32)
    coef_std  = per_seed_arr.std(axis=0, ddof=1).astype(np.float32)

    # Row-level scalar confidence: 1 / (1 + median_k(std_k / |mean_k|)).
    _denom = np.maximum(np.abs(coef_mean), 1e-12)
    _rel_std = coef_std / _denom
    _row_med = np.median(_rel_std, axis=1)
    confidence = (1.0 / (1.0 + _row_med)).astype(np.float32)

    result = {
        "coef":         coef_mean,
        "coef_std":     coef_std,
        # Per-row reliability bits, same vocabulary as the decomposition
        # products (sky_decomp.reliability).  Coefficient-computable bits only;
        # add the reversal bits with `reliability_from_components` once the
        # caller has reconstructed the components.
        "reliability":  prediction_reliability(
            coef_mean, ensemble["coef_names"],
            ensemble.get("coef_upper_bound"),
            ensemble.get("group_indices")),
        "confidence":   confidence,
        "coef_names":   list(ensemble["coef_names"]),
        "triplet":      triplet,
        "near_is_east": near_is_east.astype(bool),
    }
    if return_per_seed:
        result["per_seed"] = per_seed_arr.astype(np.float32)
    return result


__all__ = [
    "build_triplet_from_pointings",
    "predict_sky_from_minimal_inputs",
]
