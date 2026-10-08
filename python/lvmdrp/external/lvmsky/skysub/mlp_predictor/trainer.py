"""Ensemble trainer and inference for the dual-encoder group-head MLP.

Every knob in ``default_dual_group_config`` is read by ``Trainer``; the class
asserts this on construction and warns about any config key it does not
consume.

Training
--------
* Targets are per-group compressed *scores*, not raw coefficients: each group
  is transformed (``sqrt`` for the continuum families, ``asinh`` for the 358
  mesospheric coefficients), centred and scaled.  ``compress_coefs_to_scores``
  and ``expand_scores_to_coefs`` are the two directions.
* The loss is a per-group mean, weighted by ``group_loss_weight`` = the config
  weight over ``sqrt(n_group)`` so a 358-coefficient group does not swamp a
  3-coefficient one.  Row weights are uniform.
* ``moon`` and ``zodi`` are scored in FLUX space rather than coefficient
  space: their predicted scores are inverted through the compressor and their
  own basis matrices to per-pixel flux, and the MSE is taken there.  This
  replaces (not augments) the coefficient-space ``smooth_l1`` for those two
  groups, and it is load-bearing -- removing it costs the moon 18 percentage
  points of its gain over copying the near arm, and also degrades the
  continuum and mesospheric groups, which have no flux term of their own,
  through the shared trunk.
* Remaining groups use ``smooth_l1`` weighted per element by
  ``1/sigma^2`` from the decomposition's own ``COEF_ERR``, with a per-group
  relative floor (``coef_err_sigma_floor_rel``) so near-zero uncertainties
  cannot dominate.
* The blend alphas train with their own learning-rate multiplier
  (``alpha_lr_mult``); at the shared rate they do not move measurably.

Ensembling and calibration
--------------------------
``run_ensemble`` trains one member per seed on a night-disjoint,
moon-phase-stratified split, then fits a per-group scalar mean-bias
correction (a Jensen-style lift, since the compressor transforms are convex)
on the training and validation rows.

Constraint-derived amplitudes
-----------------------------
Half the corpus has one of its two continuum amplitudes set by a
DECOMPOSITION CONSTRAINT rather than by the data, so on those rows the target
is a known function of geometry and is derived here rather than learned:

* ``apply_zodi_ceiling_rule`` -- bright moon, where the Leinert anchor binds
  on 87.5% of rows.  Clamps every valid row to the anchor ceiling (a hard
  upper bound) and snaps gated rows onto it.
* ``apply_moon_down_amplitude_rule`` -- dark time, where the moon-share
  bracket has collapsed onto ``amp_prior_floor``.  Restores the amplitude
  from the near arm's measured moon/zodi ratio, and ``moon_down_amp_free``
  makes the flux loss amplitude-blind on those rows so only the shape trains.
* ``degenerate_continuum_flag`` -- marks the ~1.7% of rows whose SKY-ARM
  decompositions collapsed the whole continuum into the zodi spline.  Not a
  fix; those rows cannot be predicted from degenerate inputs.

Both rules rescale a coefficient block only, so the predicted SHAPE survives;
both are no-ops on artifacts that lack them, so older ensembles stay
bit-identical; both are idempotent and linear, so an ensemble mean of
rule-satisfying members satisfies them too.  The zodi rule runs FIRST because
the moon rule reads the predicted zodi amplitude.  Each prints its own
calibration and a train-set comparison at fit time -- that self-reporting is
what caught the ratio transfer being a no-op on filtered data.

See notebook chapter 3.9 for the measurements behind every threshold.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from astropy.io import fits

from sky_decomp.lsf_surface_iterative import SkyDecompLSFSurfaceIterative

from . import data as _data_mod
from .compressor import (
    compress_coef_err_to_score_sigma,
    compress_coefs_to_scores,
    expand_scores_to_coefs,
    inverse_group_compressor,
)
from .data import (_infer_base_dir_for_reconstruction, airglow_geometry_scale,
                   make_corpus_basis_decomposer)
from . import noise
from .metrics import metric_row
from .ml_utils import (
    RobustScaler,
    moon_phase_deg_from_ctx,
    set_reproducibility,
    split_indices_by_moon_phase,
)
from .model import DualEncoderGroupHeadMLPCompressed

DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP = {
    # Per-group floors on the relative sigma (fraction of the per-column
    # median finite sigma).  Set on 2026-08-16 from the sigma / weight
    # diagnostic (§7): moon / continuum / ionospheric take the historical
    # 5% floor; mesospheric (403 lines) and atomic have p99 / p50 sigma
    # ratios of 10^3-10^6 in compressed-score space, so they need a 20%
    # floor to keep w_p99 / w_median below ~30 per column.
    'moon': 0.05,
    'zodi': 0.05,
    'continuum': 0.05,
    'mesospheric': 0.20,
    'ionospheric': 0.05,
    'atomic': 0.20,
}


# --- Moon-down moon amplitude: derived, not learned ------------------------
# When the moon is below the horizon the decomposition does not MEASURE a moon
# amplitude.  `_physics_only_model` predicts a moon fraction of ~3e-5 there, so
# the moon-share bracket collapses onto `SkyDecompBase.amp_prior_floor` (0.02)
# and the QP simply sits on that ceiling: measured on gaia-stars-mask with the
# exact per-row design rebuilt from the stored LSF surface, 87.5% of moon-down
# sci rows have the share pinned at the floor, i.e.
#     A_moon = R * A_zodi,   R = eps/(1-eps) in true basis-integral units.
# Fitting R as the median ratio over moon-down training rows reproduces the
# true moon amplitude to MAD 0.00051 dex with p10-p95 inside +/-0.0017 dex
# (7189 rows).  The target therefore carries NO information of its own, and
# the network was previously spending a head on it: `amplitude_error_vs_ctx`
# reports moon-down MAD 0.0455 against moon-up 0.0098, and the moon-down
# number is mostly the zodi amplitude error (0.0344) arriving through this
# same constraint.
#
# The 7.4% of moon-down rows the rule misses are ones where the QP wanted
# essentially no moon at all (share interior, far below the ceiling).  The
# rule over-predicts those, but the whole component is 1.04% of the fitted
# continuum on moon-down rows (p90 1.75%) against 54% when the moon is up, so
# the absolute cost is bounded by ~1% of the continuum.
#
# The SHAPE is not learnable either, though it is left in the loss because it
# is weakly better than chance: 93.9% of the 14 adjacent Moon_bs knot pairs
# sit exactly on a beta ratio bound (a CORNER of the feasible polytope, which
# is a discontinuous function of the data and unrepresentable by a smooth
# network), and the SCI and NEAR fits of the SAME exposure disagree at L1
# 0.311 against a shuffled-pairing null of 0.612 -- only 2x better than
# chance, where moon-up rows reach 0.050 against 0.441.
#
# So: the loss is made amplitude-BLIND on these rows (the prediction is
# rescaled to the true integral before the pixel term, and the log-amplitude
# term is dropped) so only the shape trains, and the amplitude is restored
# analytically at predict time by `apply_moon_down_amplitude_rule`.
MOON_DOWN_ALT_DEG = 0.0
MOON_DOWN_AMP_FREE_GROUPS = ('moon',)
_MOON_FRAC_PO_FEATURE = 'moon_frac_po'

# The gate is NOT `moon_alt <= 0`.  The rule is valid exactly where the
# moon-share bracket has collapsed onto `SkyDecompBase.amp_prior_floor`, and
# that happens iff `amp_prior_tol * moon_frac_po <= amp_prior_floor`, i.e.
#     moon_frac_po <= 0.02 / 3 = 0.006667.
# Testing that condition directly beats every altitude cut on BOTH coverage
# and exactness at once, which is the signature of using the real criterion
# rather than a proxy:
#
#   gate                     cover   exact<0.01   |d|>0.05
#   moon_alt <= 0            49.7%       92.6%       7.4%
#   moon_alt <= -2           48.4%       94.9%       5.1%
#   moon_alt <= -4           47.0%       96.1%       3.9%
#   moon_alt <= -6           45.7%       96.1%       3.9%
#   moon_alt <= -12          41.4%       95.7%       4.2%
#   moon_frac_po <= 0.00667  47.8%       96.2%       3.7%
#
# WHY a fixed altitude cut cannot win: scattered moonlight with the moon just
# below the horizon is real and the model carries it as
# `horizon_scale = exp(-8.2182 * tanh(depth / 5 deg))` -- only -0.70 dex at
# -1 deg, -2.72 at -5, and not saturated at -3.57 until about -15.  But that
# depth term multiplies the phase and separation terms, so a thin crescent 2
# deg down contributes nothing while a full moon 5 deg down still contributes.
# `moon_frac_po` already carries all three; altitude alone carries one.  The
# deployed gate therefore corresponds to no single altitude: the gated rows
# reach up to -0.69 deg, while the rows it EXCLUDES that `moon_alt <= 0` kept
# run from -0.0 down to -5.32 deg (median -1.43).  Those excluded rows are
# pinned only 0.4% of the time and were most of the old tail.
#
# The residual 3.7% that still miss inside the gate are a different
# population, not a horizon effect: median moon_alt -62 deg with
# A_moon/A_zodi ~ 1.4e-6 against R = 0.024, i.e. rows where the QP wanted no
# moon at all.  No altitude threshold can reach them.
MOON_DOWN_FRAC_MAX = 0.02 / 3.0


def _moon_down_mask_from_ctx(ctx_sci_phys, ctx_names,
                             alt_deg=MOON_DOWN_ALT_DEG,
                             frac_max=MOON_DOWN_FRAC_MAX, verbose=False):
    """Rows where the moon-share bracket has collapsed onto its floor.

    Prefers ``moon_frac_po <= frac_max``, which IS that condition; falls back
    to ``moon_alt <= alt_deg`` only when the physics-only feature is absent
    (a triplet built without the v2 model cache).  Both come from the CTX
    block rather than the cache, so the gate needs no extra input at predict
    time and is guaranteed to be the one training used.  Returns ``None`` when
    neither column exists, which makes every caller a no-op rather than a
    crash.
    """
    names = [str(x) for x in ctx_names]
    ctx = np.asarray(ctx_sci_phys, dtype=np.float64)
    if _MOON_FRAC_PO_FEATURE in names and frac_max is not None:
        col = ctx[:, names.index(_MOON_FRAC_PO_FEATURE)]
        # `moon_frac_po == 0` is the augment's invalid flag, so a row with no
        # physics-only prediction is excluded rather than swept in as "no
        # moon" -- which a naive `<= frac_max` test would do.
        return np.isfinite(col) & (col > 0.0) & (col <= float(frac_max))
    if 'moon_alt' not in names:
        return None
    if verbose:
        print(f'  [moon-down] {_MOON_FRAC_PO_FEATURE} absent; falling back to '
              f'moon_alt <= {float(alt_deg):g}, which is the weaker gate '
              f'(92.6% of its rows pinned against 96.2%).')
    col = ctx[:, names.index('moon_alt')]
    return np.isfinite(col) & (col <= float(alt_deg))


def _fit_moon_down_amp_rule(coef_sci, group_indices, flux_basis_matrices,
                            moon_down, train_idx, ctx_names,
                            alt_deg=MOON_DOWN_ALT_DEG,
                            frac_max=MOON_DOWN_FRAC_MAX,
                            coef_near=None, ratio_transfer=True, verbose=True):
    """Calibrate ``A_moon = R * A_zodi`` on moon-down TRAINING rows.

    ``R`` is a ratio of basis integrals, so it absorbs whatever normalisation
    the supplied ``flux_basis_matrices`` carry (the trainer's are on a strided
    wavelength grid, and the static build differs from the per-row refined
    design by a constant ~1.18 on the moon block).  Calibrating it here rather
    than hard-coding eps/(1-eps) is what keeps that offset from leaking in.
    Returns ``None`` when anything needed is missing.
    """
    if moon_down is None:
        return None
    A_m = flux_basis_matrices.get('moon') if flux_basis_matrices else None
    A_z = flux_basis_matrices.get('zodi') if flux_basis_matrices else None
    if A_m is None or A_z is None:
        if verbose:
            print('  [moon-down rule] no moon/zodi basis supplied; rule disabled.')
        return None
    m_idx = np.asarray(group_indices['moon'], dtype=int)
    z_idx = np.asarray(group_indices['zodi'], dtype=int)
    v_m = np.asarray(A_m, dtype=np.float64).sum(axis=1)
    v_z = np.asarray(A_z, dtype=np.float64).sum(axis=1)
    if v_m.size != m_idx.size or v_z.size != z_idx.size:
        if verbose:
            print(f'  [moon-down rule] basis rows {v_m.size}/{v_z.size} do not '
                  f'match coefficient counts {m_idx.size}/{z_idx.size}; disabled.')
        return None
    c = np.asarray(coef_sci, dtype=np.float64)
    a_m = c[:, m_idx] @ v_m
    a_z = c[:, z_idx] @ v_z
    fit_rows = np.zeros(c.shape[0], dtype=bool)
    fit_rows[np.asarray(train_idx, dtype=int)] = True
    fit_rows &= moon_down & (a_m > 0.0) & (a_z > 0.0)
    if int(fit_rows.sum()) < 50:
        if verbose:
            print(f'  [moon-down rule] only {int(fit_rows.sum())} usable '
                  f'moon-down training rows; rule disabled.')
        return None
    ratio = a_m[fit_rows] / a_z[fit_rows]
    R = float(np.median(ratio))
    resid = np.log10(R * a_z[fit_rows] / a_m[fit_rows])
    # Median moon-down SHAPE, used only when the network predicts an
    # identically zero moon block and there is nothing to rescale.
    _cm = c[fit_rows][:, m_idx]
    _shape = np.median(_cm / _cm.sum(axis=1, keepdims=True), axis=0)
    if verbose:
        print(f'  [moon-down rule] R = {R:.6g} from {int(fit_rows.sum())} '
              f'moon-down train rows; '
              f'residual log10(R*A_zodi/A_moon) MAD '
              f'{float(np.median(np.abs(resid - np.median(resid)))):.5f} dex, '
              f'|d|>0.01 on {100.0 * float(np.mean(np.abs(resid) > 0.01)):.1f}% '
              f'of them.')
    # The moon share on these rows is BIMODAL, not constant.  On 96.25% of
    # gated rows the QP sits on the 0.02 ceiling (t = A_moon/(R A_zodi) >=
    # 0.99); on 3.25% it wants no moon at all (t < 1e-4), with only 0.49% in
    # between.  A flat R therefore over-predicts the second mode by the whole
    # component -- 2.40% of the sci continuum, which is the entire error on
    # those rows.
    #
    # The NEAR ARM knows which mode a row is in: of the zero-moon sci rows
    # 96.0% have the near arm also at t < 0.5, against 0.72% of the pinned
    # ones.  So transfer the measured per-row ratio instead of assuming R:
    #     A_moon(sci) = clip(A_moon/A_zodi |near, 0, R) * A_zodi(sci)
    # which reproduces BOTH modes with no threshold and no mode assignment.
    # Measured on 6914 gated corpus rows, moon flux error as a fraction of the
    # sci continuum:
    #
    #   predictor                  median      p99   >0.5%   >0.2%
    #   flat R * A_zodi           0.00086%   2.4038%  3.47%   3.66%
    #   near ratio * A_zodi       0.00060%   0.0107%  0.56%   0.84%
    #   arm-mean ratio            0.00059%   0.3959%  0.81%   1.72%
    #   copy near verbatim        0.10913%   0.9695%  9.07%  31.07%
    #
    # so 240 badly-wrong rows become 39.  It also beats flat R on the PINNED
    # rows (0.00060% against 0.00086%), because the measured ratio absorbs
    # per-row variation a global median cannot.  Near alone beats the arm mean
    # -- the far arm's own mode can differ -- and copying the near arm's
    # amplitude verbatim is much worse in the bulk, so it is specifically the
    # RATIO that transfers.
    # The clamp at R is free: the share ceiling bounds the ratio, and
    # r_near/R exceeds 1.01 on 0.014% of rows (one outlier at 50x).
    _ratio_stats = None
    if ratio_transfer and coef_near is not None:
        cn = np.asarray(coef_near, dtype=np.float64)
        a_m_n = cn[:, m_idx] @ v_m
        a_z_n = cn[:, z_idx] @ v_z
        _use = fit_rows & np.isfinite(a_m_n) & np.isfinite(a_z_n) & (a_z_n > 0.0)
        if _use.sum() >= 50:
            _r = np.clip(a_m_n[_use] / a_z_n[_use], 0.0, R)
            _e_flat = np.abs(R * a_z[_use] - a_m[_use])
            _e_tran = np.abs(_r * a_z[_use] - a_m[_use])
            _scale = np.maximum(a_m[_use] + a_z[_use], 1e-30)
            _ratio_stats = {
                'n': int(_use.sum()),
                'frac_gt_flat': float(np.mean(_e_tran > _e_flat)),
                'bad_flat': float(np.mean(_e_flat / _scale > 0.005)),
                'bad_tran': float(np.mean(_e_tran / _scale > 0.005)),
            }
            if verbose:
                print(f'  [moon-down rule] near-arm ratio transfer ON: on '
                      f'{_ratio_stats["n"]} train rows it is worse than flat R '
                      f'on {100.0 * _ratio_stats["frac_gt_flat"]:.2f}% of them, '
                      f'and the fraction wrong by >0.5% of moon+zodi falls '
                      f'{100.0 * _ratio_stats["bad_flat"]:.2f}% -> '
                      f'{100.0 * _ratio_stats["bad_tran"]:.2f}%')
        elif verbose:
            print('  [moon-down rule] too few usable near-arm rows for the '
                  'ratio transfer; falling back to flat R.')
    return {
        'R': R,
        'ratio_transfer': bool(ratio_transfer),
        'ratio_stats': _ratio_stats,
        'alt_deg': float(alt_deg),
        'frac_max': (None if frac_max is None else float(frac_max)),
        'moon_cols': m_idx.astype(int),
        'zodi_cols': z_idx.astype(int),
        'v_moon': v_m.astype(np.float64),
        'v_zodi': v_z.astype(np.float64),
        'shape_fallback': np.asarray(_shape, dtype=np.float64),
        'n_fit_rows': int(fit_rows.sum()),
        'resid_mad_dex': float(np.median(np.abs(resid - np.median(resid)))),
    }


def apply_moon_down_amplitude_rule(coef, ctx_sci_phys, artifacts,
                                   coef_near_phys=None):
    """Set the moon amplitude from the near arm's moon/zodi ratio.

    Rescales the 15 moon coefficients, so the network's SHAPE is kept and only
    the amplitude is replaced.  A no-op when the artifacts carry no rule, which
    keeps ensembles trained before this existed bit-identical.

    Safe to apply more than once: after one application the constraint holds
    exactly, and because ``A`` is linear in the coefficients the ensemble mean
    of rule-satisfying members satisfies the rule too.
    """
    rule = artifacts.get('moon_down_amp_rule')
    if not rule:
        return coef
    mask = _moon_down_mask_from_ctx(ctx_sci_phys, artifacts['ctx_names'],
                                    rule['alt_deg'],
                                    rule.get('frac_max', MOON_DOWN_FRAC_MAX))
    if mask is None or not mask.any():
        return coef
    out = np.asarray(coef, dtype=np.float64).copy()
    m_idx = np.asarray(rule['moon_cols'], dtype=int)
    z_idx = np.asarray(rule['zodi_cols'], dtype=int)
    v_m = np.asarray(rule['v_moon'], dtype=np.float64)
    v_z = np.asarray(rule['v_zodi'], dtype=np.float64)
    a_moon = out[:, m_idx] @ v_m
    a_zodi = out[:, z_idx] @ v_z
    R = float(rule['R'])
    # Per-row ratio from the NEAR arm where available, else the flat R.  This
    # is what reproduces the zero-moon mode; see _fit_moon_down_amp_rule.
    ratio = np.full(out.shape[0], R, dtype=np.float64)
    if rule.get('ratio_transfer') and coef_near_phys is not None:
        cn = np.asarray(coef_near_phys, dtype=np.float64)
        if cn.shape[0] == out.shape[0]:
            a_m_n = cn[:, m_idx] @ v_m
            a_z_n = cn[:, z_idx] @ v_z
            _good = (np.isfinite(a_m_n) & np.isfinite(a_z_n) & (a_z_n > 0.0)
                     & (a_m_n >= 0.0))
            ratio = np.where(
                _good,
                np.clip(a_m_n / np.where(a_z_n > 0.0, a_z_n, 1.0), 0.0, R), R)
    want = ratio * np.clip(a_zodi, 0.0, None)
    # Rescale where there is a shape to rescale; substitute the median
    # moon-down shape where the predicted moon block is identically zero.
    live = mask & (a_moon > 0.0)
    dead = mask & ~(a_moon > 0.0)
    if live.any():
        scale = np.ones(out.shape[0], dtype=np.float64)
        scale[live] = want[live] / a_moon[live]
        out[np.ix_(np.flatnonzero(live), m_idx)] *= scale[live][:, None]
    if dead.any():
        shp = np.asarray(rule['shape_fallback'], dtype=np.float64)
        denom = float(shp @ np.asarray(rule['v_moon'], dtype=np.float64))
        if denom > 0.0:
            out[np.ix_(np.flatnonzero(dead), m_idx)] = (
                shp[None, :] * (want[dead] / denom)[:, None])
    return out.astype(np.asarray(coef).dtype)


DEGENERATE_CONTINUUM_FRAC = 0.5


def degenerate_continuum_flag(coef_near_phys, coef_far_phys, artifacts,
                              frac=DEGENERATE_CONTINUUM_FRAC):
    """Flag rows whose SKY-ARM decompositions collapsed into the zodi spline.

    On ~1.7% of corpus rows the QP puts the ENTIRE continuum into ``Zodi_bs``
    and zeroes both the moon and the diffuse block.  These are not a physical
    "no moon" state, they are failed fits: median reduced chi2 2.09 against
    0.139 on healthy dark rows, a factor of 15, and 67.2% of them also trip
    the diffuse-zeroed gate.  The science row's continuum split is then
    moon 0.0001% / zodi 100.0% / diffuse 0.0% against a healthy
    1.0 / 43.1 / 55.8.

    Nothing downstream of the decomposition can recover such a row -- the ML
    is predicting from degenerate inputs -- so the useful action is to MARK
    it.  This detector needs only the two sky arms, which exist at prediction
    time, so it works in production where the science decomposition does not
    exist and the training filters do not run.

    Measured on 14 457 corpus rows against the science-side truth:

    | detector | flagged | precision | recall |
    |---|---|---|---|
    | near arm alone | 288 | 83.3% | 96.0% |
    | **both arms** | **270** | **87.8%** | **94.8%** |
    | both arms diffuse ~ 0 | 65 | 84.6% | 22.0% |

    It cannot misfire with the moon up: on 7543 moon-up rows it never fires,
    and the 1st percentile of their moon/zodi ratio is 6x the threshold.

    Returns a boolean array, or ``None`` when the artifacts carry no
    moon-down rule to take the basis and ``R`` from.
    """
    rule = artifacts.get('moon_down_amp_rule')
    if not rule:
        return None
    m_idx = np.asarray(rule['moon_cols'], dtype=int)
    z_idx = np.asarray(rule['zodi_cols'], dtype=int)
    v_m = np.asarray(rule['v_moon'], dtype=np.float64)
    v_z = np.asarray(rule['v_zodi'], dtype=np.float64)
    thresh = float(frac) * float(rule['R'])
    out = None
    for arm in (coef_near_phys, coef_far_phys):
        if arm is None:
            continue
        c = np.asarray(arm, dtype=np.float64)
        a_m = c[:, m_idx] @ v_m
        a_z = c[:, z_idx] @ v_z
        # A non-positive zodi is itself pathological, but it is not THIS
        # pathology, so it is not flagged here.
        bad = np.isfinite(a_m) & np.isfinite(a_z) & (a_z > 0.0) & (
            a_m / np.where(a_z > 0.0, a_z, 1.0) < thresh)
        out = bad if out is None else (out & bad)
    return out



# --- Bright-moon zodi amplitude: derived, not learned ----------------------
# The mirror image of the moon-down rule above, on the other half of the
# corpus.  The decomposition brackets the fitted zodi total to
# [Z_pred/kappa_z, kappa_z * Z_pred] around a Leinert prediction, and with the
# moon up the QP presses that CEILING: measured on gaia-stars-mask with the
# exact per-row design and the physics-only Z_pred, 87.5% of moon-up sci rows
# sit exactly on it.  So the target there is `kappa_z * calibration * Z_pred`,
# a deterministic function of geometry.
#
# The ceiling is CORRECT and must not be widened.  Releasing kappa_z on 294
# lunation-stratified spectra moves the freed zodi up 0.193 dex while the moon
# falls 0.107 and the diffuse block falls 0.435, with the total continuum
# conserved to 0.5% and rms improving only 0.6% -- pure re-partitioning among
# degenerate families.  And the excess tracks the MOON, not the ecliptic:
# partial rho(excess, FLI | log B500) = +0.733 against partial
# rho(excess, log B500 | FLI) = -0.322, which has the wrong sign for
# zodiacal light.  It is moon-into-zodi leakage the bracket exists to stop.
#
# Two separate uses of the same prediction, both measured on every10 (1445
# rows) against the fitted zodi:
#
#  * CLAMP, everywhere.  `S * Z_pred` is a hard upper bound: over the 14 457
#    valid corpus rows the fitted zodi never exceeds it by more than 0.0024
#    dex and only 0.01% exceed it at all, moon-up or moon-down.  So clipping
#    any prediction back to it can only move it toward the truth.
#  * SNAP, on gated rows, to remove the network's residual error where the QP
#    pinned the target (its moon-up zodi MAD is 0.01051 dex).
#
# GATE, chosen on the full corpus.  Fraction of gated rows within 0.01 dex of
# the ceiling, and the p95/p99 of |log10(S Z/A_zodi)| over them:
#
#   gate                cover   exact      p95      p99
#   moon_alt > 0        50.3%   83.9%   0.3104   0.6022
#   moon_frac_po > 0.4  45.9%   90.8%   0.1117   0.3221
#   moon_frac_po > 0.5  43.8%   94.2%   0.0246   0.2153
#   moon_frac_po > 0.6  41.5%   97.2%   0.0005   0.1358
#   moon_frac_po > 0.7  38.1%   98.6%   0.0002   0.1059
#
# BEWARE the every10 subsample here: it put gate 0.5 at 95.3% exact with p95
# 0.0010, five percentiles better than the corpus, and picking the gate off it
# would have chosen 0.5.  Use the full corpus for this.
#
# The snap is GUARDED: it fires only where the network already predicts at
# least `snap_frac` of the ceiling, so the few percent of gated rows that are
# genuinely interior keep the network's own answer.  Simulating the network as
# truth x 10^(noise) with noise sd 0.0156 dex and scoring over EVERY valid row
# (not just gated ones, so a narrower gate pays for what it leaves behind):
#
#   variant                  MAD      p99      max   missed-pinned
#   clamp only           0.00661   0.0384   0.0750        --
#   gate 0.6, beta 0.80  0.00190   0.0387   0.1283      0.00%
#   gate 0.6, beta 0.85  0.00188   0.0379   0.1103      0.00%
#   gate 0.6, beta 0.90  0.00190   0.0375   0.0791      0.07%
#   no gate,  beta 0.80  0.00097   0.0813   0.1548      0.00%
#
# So the deployed pair buys a 3.5x better MAD and a slightly better p99 for
# 0.05 dex on the single worst row, and dropping the gate trades a better MAD
# for a 2x worse p99.  A missed pinned row is benign -- it keeps
# network-plus-clamp -- which is why beta errs high.  Re-checked against
# Student-t(3) noise at the same sd (beta 0.90 max 0.68 against 1.22 at 0.80)
# and against twice the noise (beta 0.90 misses 2.9% of pinned rows and still
# beats clamp-only 2.3x on MAD).
#
# That guard is also why, unlike the moon-down case, the loss is left ALONE
# here by default.  The moon-down moon target is noise in shape and a constant
# in amplitude, so there was nothing to learn; the bright-moon zodi amplitude
# is a deterministic function the network partly learns -- and it now receives
# `zodi_po_log10` directly -- and its learned value is exactly what lets the
# guard separate pinned rows from interior ones.  Making it amplitude-blind
# would destroy the signal the guard runs on.  `zodi_ceiling_amp_free` exists
# to A/B that, and defaults to False.
#
# Needs `data.ZODI_CEILING_FEATURE_NAMES` in the ctx block, which come from
# moon/zodi model cache v2.  v1 stored only the LEARNED-parameter prediction,
# which is ~1.5x larger and made this whole effect invisible.
ZODI_CEILING_GATE_FRAC = 0.6
ZODI_CEILING_SNAP_FRAC = 0.9
ZODI_CEILING_AMP_FREE_GROUPS = ('zodi',)
_ZODI_PO_FEATURE = 'zodi_po_log10'


def _zodi_ceiling_inputs_from_ctx(ctx_sci_phys, ctx_names):
    """``(log10 Z_pred, moon_frac_po, valid)`` from the ctx block, or None.

    ``moon_frac_po == 0`` is the invalid flag written by the augment: a real
    physics-only moon fraction never reaches zero, while the log has no
    impossible value to spare.
    """
    names = [str(x) for x in ctx_names]
    if _ZODI_PO_FEATURE not in names or _MOON_FRAC_PO_FEATURE not in names:
        return None
    ctx = np.asarray(ctx_sci_phys, dtype=np.float64)
    log_z = ctx[:, names.index(_ZODI_PO_FEATURE)]
    frac = ctx[:, names.index(_MOON_FRAC_PO_FEATURE)]
    valid = np.isfinite(log_z) & np.isfinite(frac) & (frac > 0.0)
    return log_z, frac, valid


def _fit_zodi_ceiling_rule(coef_sci, group_indices, flux_basis_matrices,
                           ctx_sci_phys, ctx_names, train_idx,
                           gate_frac=ZODI_CEILING_GATE_FRAC,
                           snap_frac=ZODI_CEILING_SNAP_FRAC, verbose=True):
    """Calibrate ``A_zodi = S * Z_pred`` on gated moon-up TRAINING rows.

    ``S`` comes out at ``kappa_z * calibration`` times whatever normalisation
    the supplied basis carries -- 3.22444 against the nominal 3.200 on this
    corpus, the 0.76% difference being the same basis-integral offset the
    moon-down rule absorbs.  Fitting it is what keeps that offset out.

    ``S`` IS BASIS-DEPENDENT, so do not sanity-check it against 3.2244.  The
    trainer's ``flux_basis_matrices`` live on the stride-5 wavelength grid, so
    the deployed value is ~1/5 of that: 0.6450 in the 2026-09-09 run against
    3.22444 measured on the full grid (ratio 5.0002, the 0.02% being the
    subsampling).  What is worth checking is that ``S`` is stable run to run
    and that the upper-bound line below stays at ~0%, since the clamp is only
    valid while it does.
    """
    got = _zodi_ceiling_inputs_from_ctx(ctx_sci_phys, ctx_names)
    if got is None:
        if verbose:
            print(f'  [zodi-ceiling] ctx has no {_ZODI_PO_FEATURE}/'
                  f'{_MOON_FRAC_PO_FEATURE}; rule disabled.  Enable the '
                  f'moon-model augment with add_zodi_ceiling=True and a v2 '
                  f'model cache.')
        return None
    log_z, frac, valid = got
    A_z = flux_basis_matrices.get('zodi') if flux_basis_matrices else None
    if A_z is None:
        if verbose:
            print('  [zodi-ceiling] no zodi basis supplied; rule disabled.')
        return None
    z_idx = np.asarray(group_indices['zodi'], dtype=int)
    v_z = np.asarray(A_z, dtype=np.float64).sum(axis=1)
    if v_z.size != z_idx.size:
        if verbose:
            print(f'  [zodi-ceiling] basis rows {v_z.size} != n_coef '
                  f'{z_idx.size}; rule disabled.')
        return None
    a_z = np.asarray(coef_sci, dtype=np.float64)[:, z_idx] @ v_z
    fit_rows = np.zeros(a_z.size, dtype=bool)
    fit_rows[np.asarray(train_idx, dtype=int)] = True
    fit_rows &= valid & (frac > float(gate_frac)) & (a_z > 0.0)
    if int(fit_rows.sum()) < 50:
        if verbose:
            print(f'  [zodi-ceiling] only {int(fit_rows.sum())} usable gated '
                  f'training rows; rule disabled.')
        return None
    z_pred = 10.0 ** log_z
    S = float(np.median(a_z[fit_rows] / z_pred[fit_rows]))
    resid = np.log10(S * z_pred[fit_rows] / a_z[fit_rows])
    # How hard an upper bound is it, over EVERY valid row, not just gated ones?
    _all = valid & (a_z > 0.0)
    _excess = np.log10(a_z[_all] / (S * z_pred[_all]))
    if verbose:
        print(f'  [zodi-ceiling] S = {S:.6g} from {int(fit_rows.sum())} gated '
              f'train rows (moon_frac_po > {float(gate_frac):g}); on them '
              f'|log10(S*Z/A_zodi)| < 0.01 for '
              f'{100.0 * float(np.mean(np.abs(resid) < 0.01)):.1f}%, p95 '
              f'{float(np.percentile(np.abs(resid), 95)):.4f} dex')
        print(f'  [zodi-ceiling] upper-bound check over all {int(_all.sum())} '
              f'valid rows: max excess {float(_excess.max()):+.4f} dex, '
              f'{100.0 * float(np.mean(_excess > 0.002)):.2f}% above it by '
              f'>0.002 dex (should be ~0; the clamp relies on this)')
    return {
        'S': S,
        'gate_frac': float(gate_frac),
        'snap_frac': float(snap_frac),
        'zodi_cols': z_idx.astype(int),
        'v_zodi': v_z.astype(np.float64),
        'n_fit_rows': int(fit_rows.sum()),
        'resid_p95_dex': float(np.percentile(np.abs(resid), 95)),
        'max_excess_dex': float(_excess.max()),
    }


def apply_zodi_ceiling_rule(coef, ctx_sci_phys, artifacts):
    """Clamp the zodi amplitude to the anchor ceiling, and snap it where pinned.

    Two effects, both scaling the 5 Zodi_bs coefficients so the spline SHAPE is
    untouched:

    * every valid row is clipped to ``S * Z_pred``, which is a hard upper bound
      on the fitted zodi by construction;
    * gated rows whose prediction is already within ``snap_frac`` of that
      ceiling are set to it exactly, because they are the ones the QP pinned.
      A gated row the network puts well below the ceiling keeps its own
      answer: that is how genuinely interior rows survive the rule.

    A no-op when the artifacts carry no rule, so ensembles trained before this
    existed stay bit-identical.  Idempotent, and linear in the coefficients, so
    an ensemble mean of rule-satisfying members satisfies it too.
    """
    rule = artifacts.get('zodi_ceiling_rule')
    if not rule:
        return coef
    got = _zodi_ceiling_inputs_from_ctx(ctx_sci_phys, artifacts['ctx_names'])
    if got is None:
        return coef
    log_z, frac, valid = got
    z_idx = np.asarray(rule['zodi_cols'], dtype=int)
    out = np.asarray(coef, dtype=np.float64).copy()
    a_z = out[:, z_idx] @ np.asarray(rule['v_zodi'], dtype=np.float64)
    ceiling = float(rule['S']) * 10.0 ** log_z
    live = valid & (a_z > 0.0) & np.isfinite(ceiling) & (ceiling > 0.0)
    over = live & (a_z > ceiling)
    snap = live & (frac > float(rule['gate_frac'])) & (
        a_z >= float(rule['snap_frac']) * ceiling)
    hit = over | snap
    if hit.any():
        scale = np.ones(out.shape[0], dtype=np.float64)
        scale[hit] = ceiling[hit] / a_z[hit]
        out[np.ix_(np.flatnonzero(hit), z_idx)] *= scale[hit][:, None]
    return out.astype(np.asarray(coef).dtype)


def train_compressed_group_mlp(
    filtered, compressors, group_indices, geom_kwargs,
    split_indices=None,
    n_epochs=50, batch_size=256, lr=7e-4,
    encoder_dims=(768, 384), ctx_dims=(96,),
    trunk_dims=(320, 160), head_dim=192,
    zodi_head_extra_dims=(32,),
    continuum_head_extra_dims=(64,),
    continuum_branch_dims=(128, 64),
    moon_zodi_coupling_dims=(64, 32),
    weight_decay=1e-4, grad_clip=1.0, patience=4,
    seed=42,
    moon_group_weight=1.0,
    zodi_group_weight=1.0,
    continuum_group_weight=1.0,
    mesospheric_group_weight=1.0,
    ionospheric_group_weight=1.0,
    blend_init_alpha=0.7,
    # 2026-09-01: learning-rate multiplier for the blend-alpha parameters
    # (``blend_alpha_direct`` + the ctx-alpha predictors).  At the default 1.0
    # alpha barely moves from its init within the early-stopping budget -- the
    # loss surface in alpha is nearly flat because the additive head absorbs any
    # systematic part of a mis-set blend, and the ctx path is damped further by
    # the sigmoid (sigma' = 0.25 at alpha=0.5, 0.128 at 0.85).  Measured travel
    # over a run: scalar groups +0.08..+0.24, ctx groups +0.01..+0.05, always
    # toward the near arm and still climbing at early stop.  Raising this lets
    # alpha reach its own optimum so the init stops being a hyperparameter.
    alpha_lr_mult=1.0,
    alpha_ctx_features=("moon_up_smooth", "ecl_beta_deg", "airmass"),
    zodi_ctx_restriction=(),
    continuum_ctx_restriction=(),
    moon_zodi_ctx_restriction=(),
    # 2026-08-25c per-pixel flux MSE for moon and/or zodi (deployed default).
    flux_mse_groups=(),
    flux_amp_lambda=0.0,
    flux_amp_floor_frac=0.05,
    flux_pixel_weight=None,
    flux_basis_matrices=None,
    flux_geom_sc_sci=None,
    # 2026-09-11: native-grid per-coefficient template integrals, used to make
    # the empirical mean-bias calibration correct the FLUX-weighted amplitude.
    calib_amplitude_weights=None,
    # 2026-08-21 tight-mask bright-moon row boost (Phase A'').
    # Heteroscedastic-Gaussian loss weighting from decomposition COEF_ERR.
    coef_err_sigma_floor_rel=None,
    # 2026-09-09 moon-down handling; see MOON_DOWN_ALT_DEG above.
    moon_down_amp_free=True,
    moon_down_amp_rule=True,
    moon_down_alt_deg=MOON_DOWN_ALT_DEG,
    moon_down_frac_max=MOON_DOWN_FRAC_MAX,
    moon_down_ratio_transfer=True,
    # 2026-09-09 bright-moon zodi ceiling; see ZODI_CEILING_GATE_FRAC above.
    zodi_ceiling_rule=True,
    zodi_ceiling_amp_free=False,
    zodi_ceiling_gate_frac=ZODI_CEILING_GATE_FRAC,
    zodi_ceiling_snap_frac=ZODI_CEILING_SNAP_FRAC,
    # Ablation hook: boolean over ALL rows; False removes a row from TRAIN and
    # VAL while leaving TEST untouched, so a data-ablation A/B keeps an
    # identical evaluation set.  Compressors are deliberately NOT refit -- they
    # come in already fitted, so the only thing that changes is which rows the
    # network's gradient sees.
    train_row_mask=None,
):
    """Train one seed of the compressed dual-encoder group-head MLP."""
    set_reproducibility(seed)

    ctx_near = np.asarray(filtered['ctx_near'], dtype=np.float32)
    ctx_far = np.asarray(filtered['ctx_far'], dtype=np.float32)
    ctx_sci = np.asarray(filtered['ctx_sci'], dtype=np.float32)

    if split_indices is not None:
        train_idx, val_idx, test_idx = split_indices
    else:
        _moon_phase_col = moon_phase_deg_from_ctx(filtered)
        train_idx, val_idx, test_idx = split_indices_by_moon_phase(
            filtered['obstime_mjd'], _moon_phase_col, seed=seed)
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if train_row_mask is not None:
        _m = np.asarray(train_row_mask, dtype=bool)
        _n_tr0, _n_va0 = train_idx.size, val_idx.size
        train_idx = train_idx[_m[train_idx]]
        val_idx = val_idx[_m[val_idx]]
        if train_idx.size < 100:
            raise ValueError(
                f"train_row_mask leaves only {train_idx.size} training rows")
        print(f'  [train-row-mask] train {_n_tr0} -> {train_idx.size} '
              f'(-{_n_tr0 - train_idx.size}), val {_n_va0} -> {val_idx.size}; '
              f'test untouched at {test_idx.size}')

    coef_near = np.asarray(filtered['coef_near'], dtype=np.float64)
    coef_far = np.asarray(filtered['coef_far'], dtype=np.float64)
    coef_sci = np.asarray(filtered['coef_sci'], dtype=np.float64)

    scores_near, slices_near = compress_coefs_to_scores(
        coef_near, ctx_near, compressors, geom_kwargs, group_indices)
    scores_far, slices_far = compress_coefs_to_scores(
        coef_far, ctx_far, compressors, geom_kwargs, group_indices)
    scores_sci, slices_sci = compress_coefs_to_scores(
        coef_sci, ctx_sci, compressors, geom_kwargs, group_indices)
    assert slices_near == slices_far == slices_sci, 'score slices disagree'
    score_slices = slices_near
    n_input_score = scores_near.shape[1]
    group_score_dims = {g: (hi - lo) for g, (lo, hi) in score_slices.items()}

    print(f'Compressed input dimension: {n_input_score} '
          f'(uncompressed was {coef_near.shape[1]}); per-group scores: '
          + ', '.join(f'{g}={n}' for g, n in group_score_dims.items()))

    # Per-group multiplier m_g on top of 1/sqrt(n_g_score).
    _group_multipliers = {'moon': float(moon_group_weight),
                          'zodi': float(zodi_group_weight),
                          'continuum': float(continuum_group_weight),
                          'mesospheric': float(mesospheric_group_weight),
                          'ionospheric': float(ionospheric_group_weight)}
    group_loss_weight = {
        g: (1.0 / max(float(np.sqrt(max(int(n), 1))), 1.0))
           * _group_multipliers.get(g, 1.0)
        for g, n in group_score_dims.items()
    }
    print(f'Loss weights (moon_group_weight={float(moon_group_weight):.2f}, '
          f'zodi_group_weight={float(zodi_group_weight):.2f}, '
          f'continuum_group_weight={float(continuum_group_weight):.2f}, '
          f'mesospheric_group_weight={float(mesospheric_group_weight):.2f}, '
          f'ionospheric_group_weight={float(ionospheric_group_weight):.2f}):')
    print(f"  {'group':<14s} {'n_score':>7s} {'m_g':>6s} {'w_g':>8s} {'w_g/n':>10s}")
    for _g, _n in group_score_dims.items():
        _m = _group_multipliers.get(_g, 1.0)
        _w = group_loss_weight[_g]
        print(f'  {_g:<14s} {int(_n):>7d} {_m:>6.2f} {_w:>8.4f} {(_w / max(int(_n), 1)):>10.5f}')

    score_scaler = RobustScaler().fit(np.vstack([
        scores_near[train_idx], scores_far[train_idx], scores_sci[train_idx],
    ]).astype(np.float32))
    ctx_scaler = RobustScaler().fit(np.vstack([
        ctx_near[train_idx], ctx_far[train_idx], ctx_sci[train_idx],
    ]))

    near_s = np.clip(score_scaler.transform(scores_near.astype(np.float32)),
                     -25.0, 25.0).astype(np.float32)
    far_s = np.clip(score_scaler.transform(scores_far.astype(np.float32)),
                    -25.0, 25.0).astype(np.float32)
    sci_s = np.clip(score_scaler.transform(scores_sci.astype(np.float32)),
                    -25.0, 25.0).astype(np.float32)

    # -----------------------------------------------------------------
    # Per-element inverse-variance weights from decomposition COEF_ERR.
    # Propagate through the compressor with a first-order Jacobian, then
    # divide out the score_scaler scale so weights live in the same
    # scaled-score space as sci_s; clip to per-column floor and normalise
    # so E_train[w_pe] = 1 per column.  Missing/boundary sigmas fall
    # through to the floor.  See §7 of the methods cell for derivation.
    coef_err_sci_local = np.asarray(
        filtered.get('coef_err_sci', np.full_like(coef_sci, np.nan)),
        dtype=np.float64,
    )
    _n_finite_err = int(np.sum(np.isfinite(coef_err_sci_local)))
    _n_total_err = int(coef_err_sci_local.size)
    if _n_finite_err == 0:
        raise RuntimeError(
            'coef_err_sci is entirely non-finite; the deployed pipeline requires '
            'the pipeline COEF_ERR HDU to be present.')
    _sigma_scores_sci, _sigma_slices_sci = compress_coef_err_to_score_sigma(
        coef_sci, coef_err_sci_local, ctx_sci,
        compressors, geom_kwargs, group_indices,
    )
    assert _sigma_slices_sci == score_slices, 'sigma slices disagree with score slices'
    _scale_ = np.asarray(getattr(score_scaler, 'scale_',
                                 np.ones(n_input_score)),
                         dtype=np.float64)
    _scale_ = np.where(_scale_ > 0.0, _scale_, 1.0)
    _sigma_scaled = _sigma_scores_sci / _scale_[None, :]
    _finite = np.where(np.isfinite(_sigma_scaled) & (_sigma_scaled > 0.0),
                       _sigma_scaled, np.nan)
    _floor_col = np.nanmedian(_finite, axis=0)
    _floor_col = np.where(np.isfinite(_floor_col) & (_floor_col > 0.0),
                          _floor_col, 1.0)
    _floor_arg = coef_err_sigma_floor_rel or DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP
    _floor_by_group = {str(k): float(v) for k, v in _floor_arg.items()}
    _floor_rel_col = np.ones(_floor_col.shape[0], dtype=np.float64)
    _missing_groups = []
    for _g, (_lo, _hi) in score_slices.items():
        _fr = _floor_by_group.get(_g)
        if _fr is None:
            _missing_groups.append(_g)
            _fr = 0.05
        _floor_rel_col[_lo:_hi] = _fr
    if _missing_groups:
        print(f'  warning: coef_err_sigma_floor_rel missing groups '
              f'{_missing_groups}; using fallback 0.05 for each.')
    _floor_col = _floor_rel_col * _floor_col
    _sigma_scaled = np.where(_sigma_scaled > 0.0, _sigma_scaled,
                             _floor_col[None, :])
    _sigma_scaled = np.maximum(_sigma_scaled, _floor_col[None, :])
    w_pe_np = 1.0 / (_sigma_scaled ** 2)
    _wcm = np.mean(w_pe_np[train_idx], axis=0)
    _wcm = np.where(_wcm > 0.0, _wcm, 1.0)
    w_pe_np = (w_pe_np / _wcm[None, :]).astype(np.float32)
    _floor_report = ', '.join(f'{g}={_floor_by_group.get(g, 0.05):.3g}'
                              for g in score_slices)
    print(
        f'Per-element loss weights from coef_err_sci: '
        f'finite fraction={_n_finite_err / max(_n_total_err, 1):.3f} in native space, '
        f'compressed w median={float(np.median(w_pe_np[train_idx])):.3f}, '
        f'1-99% = [{float(np.percentile(w_pe_np[train_idx], 1)):.2f}, '
        f'{float(np.percentile(w_pe_np[train_idx], 99)):.2f}]'
    )
    print(f'  per-group floor_rel: {_floor_report}')
    _resolved_floor_by_group = {g: float(_floor_by_group.get(g, 0.05))
                                for g in score_slices}

    ctx_near_n = np.clip(ctx_scaler.transform(ctx_near), -25.0, 25.0).astype(np.float32)
    ctx_far_n = np.clip(ctx_scaler.transform(ctx_far), -25.0, 25.0).astype(np.float32)
    ctx_sci_n = np.clip(ctx_scaler.transform(ctx_sci), -25.0, 25.0).astype(np.float32)

    # Row weights are uniform.  Every per-row reweighting scheme tried was
    # rejected on measurement: high_airmass and moon_down_ecliptic boosts never
    # helped, and the bright-moon-close boost (fli>=0.90, sep<=30 deg, alt>0,
    # x1.5) was carried at 1.0 (= off) because it leaked into the blue atlas
    # band (+23% RMS|frac|) for a mid-band gain the flux-space loss now
    # provides directly.  Kept as a named constant rather than a knob so the
    # loss has one less silent degree of freedom.
    _row_weights_np = np.ones(ctx_sci.shape[0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Per-pixel flux MSE precomputation for moon / zodi (deployed default).
    # For each group g, undo the compressor + geometry to recover native
    # coefs; A_g @ c gives flux.  Loss = mean_lambda((f_pred - f_true)^2)
    # per row, scale-matched to the diagonal Huber magnitude on train rows.
    # ------------------------------------------------------------------
    _flux_mse_state_by_group = {}
    if flux_mse_groups and flux_basis_matrices is not None and flux_geom_sc_sci is not None:
        _center_ = np.asarray(getattr(score_scaler, 'center_',
                                      getattr(score_scaler, 'mean_',
                                              np.zeros(n_input_score))),
                              dtype=np.float64)
        for _g_fm in flux_mse_groups:
            if _g_fm not in score_slices:
                print(f'  [flux-mse] {_g_fm}: not in score_slices; skipping.')
                continue
            _A_np = flux_basis_matrices.get(_g_fm)
            if _A_np is None:
                print(f'  [flux-mse] {_g_fm}: no basis matrix supplied; skipping.')
                continue
            _comp_fm = compressors[_g_fm]
            _gidx_fm = np.asarray(group_indices[_g_fm], dtype=int)
            _lo_fm, _hi_fm = score_slices[_g_fm]
            _n_kept_fm = int(_hi_fm - _lo_fm)
            _n_full_fm = int(_comp_fm['basis'].shape[0])
            if _n_kept_fm != _n_full_fm:
                print(f'  [flux-mse] {_g_fm}: kept={_n_kept_fm} != full={_n_full_fm}; '
                      f'PCA-subset flux MSE not implemented, skipping.')
                continue
            if str(_comp_fm.get('kind')) != 'sqrt':
                print(f'  [flux-mse] {_g_fm}: only sqrt compressor is supported '
                      f'(got {_comp_fm.get("kind")}); skipping.')
                continue
            _A_g = np.asarray(_A_np, dtype=np.float32)
            if _A_g.shape[0] != _gidx_fm.size:
                print(f'  [flux-mse] {_g_fm}: basis rows {_A_g.shape[0]} != '
                      f'n_coef {_gidx_fm.size}; skipping.')
                continue
            _sc_fm = _scale_[_lo_fm:_hi_fm]
            _mu_fm = _center_[_lo_fm:_hi_fm]
            _sc_sci_g_np = np.asarray(flux_geom_sc_sci[:, _gidx_fm], dtype=np.float32)
            _c_true_train = coef_sci[train_idx][:, _gidx_fm].astype(np.float64)
            # Geometry goes INSIDE the basis product, exactly as
            # `compressed_loss` applies it: `(_em_true * _g_row) @ A`.
            # Calibrating `scale_match` on a geometry-free flux measures a
            # quantity the loss never evaluates.  Harmless for groups whose
            # geometry scale is identically 1 (moon, zodi), but `continuum`
            # is an airglow group carrying a van Rhijn scale with median
            # 1.275 (p5-p95 1.016-1.689), so its flux term ran ~1.6x
            # over-weighted against the other groups.
            _flux_true_train = ((_c_true_train
                                 * _sc_sci_g_np[train_idx].astype(np.float64))
                                @ _A_g.astype(np.float64))
            # Weighted, so the scale-match below stays exact when the photon
            # weights are on.  They are row-normalised to mean 1, so this is a
            # small correction, not a change of units.
            if flux_pixel_weight is None:
                _row_flux_norm_train = np.mean(_flux_true_train ** 2, axis=1)
            else:
                # Index THEN widen: the other order builds a float64 copy of
                # every row (194 MB here) before throwing most of it away, once
                # per group per seed.
                _row_flux_norm_train = np.mean(
                    np.asarray(flux_pixel_weight)[train_idx].astype(np.float64)
                    * _flux_true_train ** 2, axis=1)
            _fin_row = np.isfinite(_row_flux_norm_train) & (_row_flux_norm_train > 0.0)
            _median_flux_norm = (float(np.median(_row_flux_norm_train[_fin_row]))
                                 if _fin_row.any() else 1.0)
            _tr_diag_fm = np.nanmean(
                w_pe_np[train_idx, _lo_fm:_hi_fm]
                * (sci_s[train_idx, _lo_fm:_hi_fm].astype(np.float64) ** 2),
                axis=1)
            _med_diag_fm = (float(np.nanmedian(_tr_diag_fm))
                            if _tr_diag_fm.size else 1.0)
            _scale_match_fm = (_med_diag_fm / max(_median_flux_norm, 1e-30)
                               if _med_diag_fm > 0 else 1.0)
            # Floor for the log-amplitude term (step 2).  The term is
            # log((A_pred + eps)/(A_true + eps))**2 on the WAVELENGTH-INTEGRATED
            # flux, so eps decides where it stops caring: rows whose true
            # amplitude is well below eps contribute ~0.  Set from the median
            # train amplitude of this group so it is scale-free.  It is the
            # integral of the same geometry-scaled flux the pixel term uses,
            # so integrate that rather than repeating the matmul.
            _amp_true_train = _flux_true_train.sum(axis=1)
            _amp_pos = _amp_true_train[np.isfinite(_amp_true_train)
                                       & (_amp_true_train > 0.0)]
            _amp_med = float(np.median(_amp_pos)) if _amp_pos.size else 1.0
            _amp_eps_fm = float(flux_amp_floor_frac) * _amp_med
            _flux_mse_state_by_group[_g_fm] = {
                'A': _A_g,
                'basis_T': _comp_fm['basis'].T.astype(np.float32),
                'sd_vec': _comp_fm['sd_vec'].astype(np.float32),
                'mean_vec': _comp_fm['mean_vec'].astype(np.float32),
                'score_scale': _sc_fm.astype(np.float32),
                'score_center': _mu_fm.astype(np.float32),
                'g_scale_sci': _sc_sci_g_np,
                'scale_match': float(_scale_match_fm),
                'median_flux_norm': float(_median_flux_norm),
                'amp_eps': _amp_eps_fm,
                'amp_median': _amp_med,
            }
            print(f'  [flux-mse] {_g_fm}: n_coef={_gidx_fm.size}, '
                  f'n_wave_ds={_A_g.shape[1]}, '
                  f'median mean(flux^2)={_median_flux_norm:.3g}, '
                  f'scale-match={_scale_match_fm:.4g} '
                  f'(median diag={_med_diag_fm:.4g}); '
                  f'amp median={_amp_med:.4g}, amp eps={_amp_eps_fm:.4g} '
                  f'({100*float(flux_amp_floor_frac):g}% of median), '
                  f'amp lambda={flux_amp_lambda!r}.')

    # --- Does the IN-LOSS reconstruction match the DEPLOYED one? -----------
    # The loss rebuilds each group's spectrum from the compressed scores with
    # its own torch implementation of the inverse compressor.  Everything
    # OUTSIDE the loss -- diagnostics, inference, the flux-space scoring -- goes
    # through `compressor.inverse_group_compressor` instead.  Two
    # implementations of one transform is exactly the kind of pair that drifts,
    # so compare them here on real rows, starting from the SAME raw scores.
    #
    # This checks the two IMPLEMENTATIONS against each other on the same raw
    # scores.  It deliberately does NOT compare against the raw corpus
    # coefficients: `filtered` inside this function is not the caller's triplet
    # row-for-row, so such a comparison needs the row mapping resolved first --
    # see the note in the changelog before adding one.
    if _flux_mse_state_by_group:
        _chk_rows = np.asarray(train_idx, dtype=int)[:64]
        for _g_ck, _st_ck in _flux_mse_state_by_group.items():
            _lo_ck, _hi_ck = score_slices[_g_ck]
            _comp_ck = compressors[_g_ck]
            # Undo the RobustScaler exactly as the loss does, to get raw scores.
            _raw_ck = (sci_s[_chk_rows][:, _lo_ck:_hi_ck].astype(np.float64)
                       * np.asarray(_st_ck['score_scale'], dtype=np.float64)
                       + np.asarray(_st_ck['score_center'], dtype=np.float64))
            # (a) the deployed inverse, used everywhere outside the loss
            _em_ref = inverse_group_compressor(_comp_ck, _raw_ck)
            # (b) the loss's own chain, in torch, on the same input
            _t = torch.from_numpy(_raw_ck.astype(np.float32))
            _z_t = (_t @ torch.from_numpy(_st_ck['basis_T'])) \
                * torch.from_numpy(_st_ck['sd_vec']) \
                + torch.from_numpy(_st_ck['mean_vec'])
            _em_t = torch.clamp(_z_t, min=0.0) ** 2
            _em_loss = _em_t.detach().cpu().numpy().astype(np.float64)
            # Compare in FLUX space, which is what the loss consumes.  A
            # per-COEFFICIENT relative difference is the wrong metric here: the
            # asinh groups carry hundreds of ~1e-6 coefficients whose float32
            # sinh differs from float64 by percent while contributing nothing
            # to the spectrum (measured: 3.2e-2 per coefficient, 6e-8 in flux).
            _A_ck2 = _st_ck['A'].astype(np.float64)
            _f_ref = _em_ref @ _A_ck2
            _f_loss = _em_loss @ _A_ck2
            _rel = float(np.nanmax(
                np.abs(_f_loss - _f_ref).sum(axis=1)
                / np.maximum(np.abs(_f_ref).sum(axis=1), 1e-30)))
            print(f'  [flux-recon check] {_g_ck}: in-loss torch inverse vs '
                  f'compressor.inverse_group_compressor -- max relative FLUX '
                  f'difference {_rel:.3e} over {_chk_rows.size} rows')
            if not (_rel < 1e-4):
                raise RuntimeError(
                    f"[flux-recon check] {_g_ck}: the loss's torch inverse "
                    f"disagrees with compressor.inverse_group_compressor by "
                    f"{_rel:.3e} (relative); the two implementations have drifted")

    # Device selection.
    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # 2026-08-24: skip DataLoader for the ~O(10MB) in-memory training set.
    # Move all tensors to device once and iterate with torch.randperm --
    # avoids ~180k per-batch CPU->device transfers per full 10-seed run.
    _flux_mse_torch_by_group = {}
    for _g_fm, _st_fm in _flux_mse_state_by_group.items():
        _flux_mse_torch_by_group[_g_fm] = {
            'A':           torch.from_numpy(_st_fm['A']).to(device),
            'basis_T':     torch.from_numpy(_st_fm['basis_T']).to(device),
            'sd_vec':      torch.from_numpy(_st_fm['sd_vec']).to(device),
            'mean_vec':    torch.from_numpy(_st_fm['mean_vec']).to(device),
            'score_scale':  torch.from_numpy(_st_fm['score_scale']).to(device),
            'score_center': torch.from_numpy(_st_fm['score_center']).to(device),
            'g_scale_sci':  torch.from_numpy(_st_fm['g_scale_sci']).to(device),
            'scale_match':  float(_st_fm['scale_match']),
            'amp_eps':      float(_st_fm['amp_eps']),
        }
    # Per-pixel photon weights are a property of the ROW and WAVELENGTH, not of
    # the coefficient group, so they are staged once and shared by every group
    # rather than duplicated into each group's state dict.
    _flux_w_pix_t = (None if flux_pixel_weight is None
                     else torch.from_numpy(
                         np.ascontiguousarray(flux_pixel_weight,
                                              dtype=np.float32)).to(device))
    _row_idx_all_np = np.arange(int(coef_sci.shape[0]), dtype=np.int64)

    def _stage_on_device(_idx):
        return (
            torch.from_numpy(near_s[_idx]).to(device),
            torch.from_numpy(far_s[_idx]).to(device),
            torch.from_numpy(ctx_near_n[_idx]).to(device),
            torch.from_numpy(ctx_far_n[_idx]).to(device),
            torch.from_numpy(ctx_sci_n[_idx]).to(device),
            torch.from_numpy(_row_weights_np[_idx]).to(device),
            torch.from_numpy(w_pe_np[_idx]).to(device),
            torch.from_numpy(sci_s[_idx]).to(device),
            torch.from_numpy(_row_idx_all_np[_idx]).to(device),
        )
    _tr_tensors = _stage_on_device(train_idx)
    _va_tensors = _stage_on_device(val_idx)
    _tr_bs = int(batch_size)
    _va_bs = 512
    _n_train = _tr_tensors[0].shape[0]
    _n_val = _va_tensors[0].shape[0]

    model = DualEncoderGroupHeadMLPCompressed(
        n_score=n_input_score, n_ctx=ctx_near_n.shape[1],
        group_score_dims=group_score_dims,
        ctx_names=[str(x) for x in filtered['ctx_names']],
        encoder_dims=tuple(int(v) for v in encoder_dims),
        ctx_dims=tuple(int(v) for v in ctx_dims),
        trunk_dims=tuple(int(v) for v in trunk_dims),
        head_dim=int(head_dim),
        zodi_head_extra_dims=tuple(int(v) for v in zodi_head_extra_dims),
        continuum_head_extra_dims=tuple(int(v) for v in continuum_head_extra_dims),
        continuum_branch_dims=tuple(int(v) for v in continuum_branch_dims),
        moon_zodi_coupling_dims=tuple(int(v) for v in moon_zodi_coupling_dims),
        blend_init_alpha=(dict(blend_init_alpha) if isinstance(blend_init_alpha, Mapping)
                          else float(blend_init_alpha)),
        alpha_ctx_features=alpha_ctx_features,
        zodi_ctx_restriction=zodi_ctx_restriction,
        continuum_ctx_restriction=continuum_ctx_restriction,
        moon_zodi_ctx_restriction=moon_zodi_ctx_restriction,
    ).to(device)

    # Blend params (per-group alpha) get weight_decay=0, an optional LR boost,
    # and -- for the direct-parametrised ones -- clamping to [eps, 1-eps] after
    # each optimizer step.  The ctx-alpha predictors join this group: they are
    # blend parameters too, and leaving them under weight decay was never the
    # intent.  (At alpha_lr_mult=1.0 that regrouping is numerically a no-op:
    # decoupled decay would have shrunk them by lr*wd per step, i.e. <0.03% over
    # a full run.)
    blend_pnames = {f'blend_alpha_direct.{_k}' for _k in model.blend_alpha_direct}
    blend_params, other_params = [], []
    for _n, _p in model.named_parameters():
        if _n in blend_pnames or _n.startswith('alpha_predictors.'):
            blend_params.append(_p)
        else:
            other_params.append(_p)
    _alpha_lr = float(lr) * float(alpha_lr_mult)
    opt = torch.optim.AdamW(
        [{'params': other_params, 'weight_decay': float(weight_decay), 'lr': float(lr)},
         {'params': blend_params, 'weight_decay': 0.0, 'lr': _alpha_lr}],
        lr=float(lr))
    print(f"Blend optim: direct (n_blend_params={sum(p.numel() for p in blend_params)}, "
          f"alpha_lr={_alpha_lr:.2e} = {float(alpha_lr_mult):g} x lr)")

    # lambda may be a scalar (every flux group) or a per-group dict.  Per-group
    # matters: measured
    # 2026-09-04: a global lambda helps the MOON (median relative integrated
    # amplitude error 0.0774 -> 0.0613 at lambda 1-5) and monotonically wrecks
    # the ZODI (gain over copy-near +38.3% -> +21.8%, tail +25%, amp share
    # 0.877 -> 0.765 at lambda 20).  The zodi amplitude is already pinned by
    # the decomposition's absolute Leinert anchor -- 93% of moon-up rows sit
    # exactly on the kappa_z ceiling, so it is a deterministic function of
    # geometry -- and penalising it harder only trades colour for amplitude
    # the network already had.  Hence per-group.
    if isinstance(flux_amp_lambda, dict):
        _amp_lambda_by_group = {str(k): float(v)
                                for k, v in flux_amp_lambda.items()}
    else:
        _amp_lambda_by_group = {str(g): float(flux_amp_lambda)
                                for g in _flux_mse_torch_by_group}
    if any(v > 0.0 for v in _amp_lambda_by_group.values()):
        print('  [flux-amp] lambda by group: '
              + ', '.join(f'{g}={_amp_lambda_by_group.get(g, 0.0):g}'
                          for g in _flux_mse_torch_by_group))

    # Moon-down rows: the moon amplitude is a constraint, not a measurement.
    _moon_down = _moon_down_mask_from_ctx(ctx_sci, filtered['ctx_names'],
                                          moon_down_alt_deg, moon_down_frac_max,
                                          verbose=True)
    if _moon_down is None:
        print('  [moon-down] ctx has neither moon_frac_po nor moon_alt; '
              'moon-down handling is disabled for this run.')
    else:
        _gate_desc = (
            f'moon_frac_po <= {float(moon_down_frac_max):g}'
            if (_MOON_FRAC_PO_FEATURE in [str(x) for x in filtered['ctx_names']]
                and moon_down_frac_max is not None)
            else f'moon_alt <= {float(moon_down_alt_deg):g}')
        print(f'  [moon-down] {int(_moon_down.sum())}/{_moon_down.size} rows '
              f'gated by {_gate_desc} '
              f'({100.0 * float(_moon_down.mean()):.1f}%); '
              f'amp_free={bool(moon_down_amp_free)}, '
              f'amp_rule={bool(moon_down_amp_rule)}.')
    _moon_down_rule = None
    if moon_down_amp_rule and _moon_down is not None:
        _moon_down_rule = _fit_moon_down_amp_rule(
            coef_sci, group_indices, flux_basis_matrices, _moon_down,
            train_idx, filtered['ctx_names'], alt_deg=moon_down_alt_deg,
            frac_max=moon_down_frac_max, coef_near=coef_near,
            ratio_transfer=bool(moon_down_ratio_transfer))
    # Bright-moon zodi: the anchor ceiling, calibrated on gated train rows.
    _zodi_ceiling = None
    if zodi_ceiling_rule:
        _zodi_ceiling = _fit_zodi_ceiling_rule(
            coef_sci, group_indices, flux_basis_matrices, ctx_sci,
            filtered['ctx_names'], train_idx,
            gate_frac=zodi_ceiling_gate_frac,
            snap_frac=zodi_ceiling_snap_frac)
    _zodi_gated = None
    if _zodi_ceiling is not None:
        _got_zc = _zodi_ceiling_inputs_from_ctx(ctx_sci, filtered['ctx_names'])
        _lz_zc, _fr_zc, _ok_zc = _got_zc
        _zodi_gated = _ok_zc & (_fr_zc > float(zodi_ceiling_gate_frac))
        print(f'  [zodi-ceiling] gate selects {int(_zodi_gated.sum())}/'
              f'{_zodi_gated.size} rows '
              f'({100.0 * float(_zodi_gated.mean()):.1f}%); '
              f'amp_free={bool(zodi_ceiling_amp_free)}, '
              f'snap_frac={float(zodi_ceiling_snap_frac):g}')

    # Per-group row masks marking rows whose AMPLITUDE must not enter the loss.
    _amp_free_t = {}
    if zodi_ceiling_amp_free and _zodi_gated is not None:
        for _g_zc in ZODI_CEILING_AMP_FREE_GROUPS:
            if _g_zc in _flux_mse_torch_by_group:
                _amp_free_t[_g_zc] = torch.from_numpy(
                    _zodi_gated.astype(np.bool_)).to(device)
            else:
                print(f'  [zodi-ceiling] {_g_zc} is not in flux_mse_groups, so '
                      f'amplitude masking is not available for it; skipped.')
    if moon_down_amp_free and _moon_down is not None:
        for _g_af in MOON_DOWN_AMP_FREE_GROUPS:
            if _g_af in _flux_mse_torch_by_group:
                _amp_free_t[_g_af] = torch.from_numpy(
                    _moon_down.astype(np.bool_)).to(device)
            else:
                print(f'  [moon-down] {_g_af} is not in flux_mse_groups, so its '
                      f'loss is coefficient-space only and cannot be made '
                      f'amplitude-blind; amplitude masking skipped for it.')

    def compressed_loss(pred_dict, yb, w_row, w_pe, row_idx_b):
        loss = torch.tensor(0.0, device=yb.device)
        for g, y_head in pred_dict.items():
            lo, hi = score_slices[g]
            target = yb[:, lo:hi].contiguous()
            if g in _flux_mse_torch_by_group:
                # Per-pixel flux MSE for the groups that have a basis.  Inverse
                # compressor in torch: undo RobustScaler, PCA, sqrt, geometry.
                _st_fm = _flux_mse_torch_by_group[g]
                _raw_target = target * _st_fm['score_scale'] + _st_fm['score_center']
                _raw_pred   = y_head * _st_fm['score_scale'] + _st_fm['score_center']
                _cent_true = _raw_target @ _st_fm['basis_T']
                _cent_pred = _raw_pred   @ _st_fm['basis_T']
                _z_true = _cent_true * _st_fm['sd_vec'] + _st_fm['mean_vec']
                _z_pred = _cent_pred * _st_fm['sd_vec'] + _st_fm['mean_vec']
                _em_true = torch.clamp(_z_true, min=0.0) ** 2
                _em_pred = torch.clamp(_z_pred, min=0.0) ** 2
                _g_row = _st_fm['g_scale_sci'][row_idx_b, :]
                _flux_true = (_em_true * _g_row) @ _st_fm['A']
                _flux_pred = (_em_pred * _g_row) @ _st_fm['A']
                # Amplitude-blind rows (moon-down for the moon group): rescale
                # the PREDICTION to the true integral so the pixel term below
                # measures shape only.  The factor is left attached to the
                # graph on purpose -- the rescaled prediction then has exactly
                # the true integral, so the gradient along the amplitude
                # direction is identically zero rather than merely small.
                _af = _amp_free_t.get(g)
                _af_b = None
                if _af is not None:
                    _af_b = _af[row_idx_b]
                    if bool(_af_b.any()):
                        _a_t_af = _flux_true.sum(dim=1)
                        _a_p_af = _flux_pred.sum(dim=1)
                        _ok_af = _af_b & (_a_p_af > 0) & (_a_t_af > 0)
                        _s_af = torch.where(
                            _ok_af, _a_t_af / _a_p_af.clamp(min=1e-30),
                            torch.ones_like(_a_p_af))
                        _flux_pred = _flux_pred * _s_af[:, None]
                _d_flux_sq = (_flux_pred - _flux_true) ** 2
                if _flux_w_pix_t is None:
                    _per_row = _d_flux_sq.mean(dim=1)
                else:
                    # Inverse-variance weighting from the photon-noise model
                    # (mlp_predictor.noise): sigma_flux = sqrt(flux * sens), so
                    # w = 1/(flux*sens).  Without this the term weighted 3600 A
                    # -- where the throughput is 4.3x worse than at 5000 A --
                    # exactly like the middle of the b channel.
                    _per_row = (_d_flux_sq
                                * _flux_w_pix_t[row_idx_b, :]).mean(dim=1)
                _per_row = _per_row * float(_st_fm['scale_match'])
                _amp_lambda = _amp_lambda_by_group.get(g, 0.0)
                if _amp_lambda > 0.0:
                    # Log-amplitude term.  The flux MSE above is ABSOLUTE, so
                    # it is dominated by the brightest rows and is nearly blind
                    # to a 20% brightness miss on a faint one -- yet 83% of the
                    # moon tail's MSE and 89% of the zodi tail's is removed by a
                    # single per-row rescale, i.e. the error is brightness, not
                    # colour, across a 16x amplitude range.  This term is scale
                    # free: it penalises the same fractional miss equally at
                    # every brightness.  Additive, not a replacement -- dropping
                    # the per-pixel term entirely (ablation A3) was catastrophic.
                    _a_true = _flux_true.sum(dim=1).clamp(min=0.0)
                    _a_pred = _flux_pred.sum(dim=1).clamp(min=0.0)
                    _eps_a = float(_st_fm['amp_eps'])
                    _d_log = torch.log((_a_pred + _eps_a) / (_a_true + _eps_a))
                    _amp_pen = _amp_lambda * _d_log ** 2
                    if _af_b is not None:
                        # These rows have already been rescaled to the true
                        # integral, so _d_log is ~0 for them anyway; zeroing it
                        # keeps that from depending on the rescale's numerics.
                        _amp_pen = _amp_pen * (~_af_b).to(_amp_pen.dtype)
                    _per_row = _per_row + _amp_pen
            else:
                w_pe_g = w_pe[:, lo:hi].contiguous()
                _per_elem = F.smooth_l1_loss(y_head, target, reduction='none') * w_pe_g
                _per_row = _per_elem.mean(dim=1)
            loss = loss + float(group_loss_weight[g]) * (w_row * _per_row).mean()
        return loss / max(len(pred_dict), 1)

    # Science context of the validation rows, sliced once: the alpha snapshot
    # below runs every epoch.
    _alpha_ctx_va = _va_tensors[4][:, model.alpha_ctx_idx]

    def _snapshot_alpha():
        """Per-group near-arm blend alpha AS THE FORWARD PASS USES IT.

        2026-09-23 fix.  `moon`, `zodi` and `continuum` never read
        `blend_alpha_direct` -- `model.forward` takes their alpha from
        `alpha_predictors[g]` instead -- so that parameter gets no gradient and
        sits at its init for the whole run.  Reporting it meant every run
        printed `delta alpha = +0.000` for exactly the three groups whose blend
        is most interesting, and the training-history plot drew them as three
        flat lines.

        They were learning the whole time.  Measured on the 1.3.2 ensemble over
        the 1429 validation rows, effective medians against a 0.85 init:
        moon 0.923, zodi 0.884, continuum 0.869, all with real context
        dependence (the moon predictor reaches |w| = 1.90 on `moon_up_smooth`
        from a zero init -- lean harder on the near arm when the moon is up).

        The ctx alpha is per row, so this records the median over the
        validation rows, with the p16/p84 spread under `<group>_p16`/`_p84`.
        Groups that really do use `blend_alpha_direct` (mesospheric,
        ionospheric, atomic) are reported unchanged and carry no spread keys.
        """
        out = {}
        with torch.no_grad():
            for g in group_score_dims:
                g = str(g)
                if g in model.alpha_predictors:
                    # numpy percentiles: torch.quantile is not available on
                    # every backend, and 1429 values x 3 groups is free.
                    _a = torch.sigmoid(
                        model.alpha_predictors[g](_alpha_ctx_va)
                    ).detach().cpu().numpy().ravel()
                    _p16, _p50, _p84 = np.percentile(_a, [16.0, 50.0, 84.0])
                    out[g] = float(_p50)
                    out[f'{g}_p16'] = float(_p16)
                    out[f'{g}_p84'] = float(_p84)
                else:
                    out[g] = float(model.blend_alpha_direct[g].item())
        return out

    history = []
    blend_history = []
    _init_blend = _snapshot_alpha()
    blend_history.append({'epoch': 0, **_init_blend})
    best_val = np.inf
    best_epoch = -1
    best_state = None
    stale = 0
    for ep in range(1, int(n_epochs) + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        _perm = torch.randperm(_n_train, device=device)
        for _b0 in range(0, _n_train, _tr_bs):
            _sel = _perm[_b0:_b0 + _tr_bs]
            near_b, far_b, near_ctx_b, far_ctx_b, ctx_b, w_row_b, w_pe_b, yb, row_idx_b = tuple(
                t[_sel] for t in _tr_tensors)
            pred = model(near_b, far_b, near_ctx_b, far_ctx_b, ctx_b)
            loss = compressed_loss(pred, yb, w_row_b, w_pe_b, row_idx_b)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            with torch.no_grad():
                _eps = float(model.blend_alpha_eps)
                for _g in group_score_dims:
                    model.blend_alpha_direct[str(_g)].clamp_(_eps, 1.0 - _eps)
            tr_loss += float(loss.item())
            tr_n += 1

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for _b0 in range(0, _n_val, _va_bs):
                _sel = slice(_b0, _b0 + _va_bs)
                near_b, far_b, near_ctx_b, far_ctx_b, ctx_b, w_row_b, w_pe_b, yb, row_idx_b = tuple(
                    t[_sel] for t in _va_tensors)
                pred = model(near_b, far_b, near_ctx_b, far_ctx_b, ctx_b)
                loss = compressed_loss(pred, yb, w_row_b, w_pe_b, row_idx_b)
                if torch.isfinite(loss):
                    va_loss += float(loss.item())
                    va_n += 1

        tr_mean = tr_loss / max(tr_n, 1)
        va_mean = va_loss / max(va_n, 1)
        history.append({'epoch': ep, 'train_loss': tr_mean, 'val_loss': va_mean})
        blend_history.append({'epoch': ep, **_snapshot_alpha()})
        if va_mean < best_val:
            best_val = va_mean
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if ep == 1 or ep % 5 == 0 or ep == int(n_epochs):
            print(f'[compressed] epoch={ep:03d} train={tr_mean:.6f} val={va_mean:.6f}')
        if stale >= int(patience):
            print(f'Early stopping at epoch {ep} (patience={patience})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _final_blend = _snapshot_alpha()
    print(f"Learned per-group near-arm blend alpha at best epoch "
          f"(ctx groups {tuple(model.alpha_predictors)}: median over val rows): "
          + '  '.join(f'{_g}={_final_blend[_g]:.3f}' for _g in group_score_dims))
    _deltas = [f'{_g}={_final_blend[_g] - _init_blend[_g]:+.3f}' for _g in group_score_dims]
    _a0_txt = ('per group' if isinstance(blend_init_alpha, Mapping)
               else f'{float(blend_init_alpha):.3f} (uniform)')
    print(f'  init alpha={_a0_txt}  |  delta alpha at best epoch: '
          + '  '.join(_deltas))

    # --- Fit per-group empirical mean-bias calibration on train + val rows ---
    # Since 2026-08-16: fit on train+val (test excluded), applied to ALL groups.
    # Previously fit on val only and gated to asinh/log groups -- that left
    # linear-compressor groups (continuum, ionospheric, atomic) with the raw
    # SmoothL1 median-bias, producing a -1.77% systematic mean bias on continuum
    # in the group-bias diagnostic.  The lift is a scalar mean_true/mean_pred
    # ratio: valid for any group with a coherent non-zero mean, so the asinh/log
    # gate was unnecessarily restrictive.  Train+val gives ~85% of rows for
    # groups as small as n=3 (continuum, atomic) where val-only sampling noise
    # dominated the estimator.  Stored as a uniform per-coefficient array so
    # `inverse_group_compressor` continues to accept the same shape.
    jensen_corrections = {}
    # Which functional each scalar lift was fitted on, and the two means behind
    # it, so the summary table reports the numbers the lift actually came from
    # rather than recomputing an unweighted mean that no longer matches.
    _calib_lift_basis = {}
    _calib_lift_means = {}
    _calib_amp_w = (np.asarray(calib_amplitude_weights, dtype=np.float64)
                    if calib_amplitude_weights is not None else None)
    if _calib_amp_w is not None and _calib_amp_w.size != int(coef_sci.shape[1]):
        raise ValueError(
            f'calib_amplitude_weights has {_calib_amp_w.size} entries but there '
            f'are {int(coef_sci.shape[1])} coefficients; it must be aligned to '
            f'coef_names or the lift would weight the wrong coefficients.')
    _calib_idx = np.concatenate([train_idx, val_idx]).astype(int)
    model.eval()
    with torch.no_grad():
        _calib_pred_dict = model(
            torch.from_numpy(near_s[_calib_idx]).to(device),
            torch.from_numpy(far_s[_calib_idx]).to(device),
            torch.from_numpy(ctx_near_n[_calib_idx]).to(device),
            torch.from_numpy(ctx_far_n[_calib_idx]).to(device),
            torch.from_numpy(ctx_sci_n[_calib_idx]).to(device),
        )
    _calib_scores_scaled = np.zeros((_calib_idx.size, n_input_score), dtype=np.float64)
    for _g, (_lo, _hi) in score_slices.items():
        _calib_scores_scaled[:, _lo:_hi] = _calib_pred_dict[_g].detach().cpu().numpy().astype(np.float64)
    _calib_pred_scores = score_scaler.inverse_transform(
        _calib_scores_scaled.astype(np.float32)).astype(np.float64)

    _calib_ctx_sci_phys = np.asarray(ctx_sci, dtype=np.float64)[_calib_idx]
    _calib_pred_phys_naive = expand_scores_to_coefs(
        _calib_pred_scores, _calib_ctx_sci_phys,
        compressors, group_indices, geom_kwargs,
        int(coef_sci.shape[1]), score_slices,
        jensen_corrections=None,
    )
    _calib_true_phys = np.asarray(coef_sci, dtype=np.float64)[_calib_idx]

    # --- Rows whose amplitude an inference-time RULE replaces ----------------
    # 2026-09-23.  `predict_sci_coefficients_default` overwrites the moon
    # amplitude on moon-down rows from the near arm's moon/zodi ratio, so the
    # network's raw moon there is discarded at inference.  It must not enter
    # the calibration mean either, and until now it did.
    #
    # Measured on the 1.3.2 corpus, blue-band moon flux over the calib rows:
    #
    #     pre-lift, all rows      +0.244%
    #     pre-lift, moon-UP only  +0.011%   <- carries ~all the blue flux
    #     pre-lift, moon-down    +129%      <- raw, and thrown away at inference
    #
    # The moon-down rows dragged the fitted lift to 0.99745, and because the
    # lift is one factor per coefficient applied to EVERY row, that -0.24%
    # landed on the moon-up rows that were already unbiased: +0.011% ->
    # -0.232%.  The moon is ~80% of the blue flux, so this showed up as a
    # coherent negative bias in the reconstructed sky -- wavelength_residual_
    # atlas mean_bias_frac -0.248 / -0.227 / -0.142% (blue/mid/NIR).  Fitting
    # on moon-up rows only gives lift 0.99964 and atlas bias +0.032 / -0.041 /
    # -0.040%, i.e. as good as dropping the lift entirely while keeping it for
    # the rows it legitimately applies to.
    #
    # The same defect was present on 1.2.1 and SILENT: there the moon-down
    # excess (+74%) happened to cancel the moon-up deficit (-0.155%), so the
    # fitted lift came out 0.99974, a no-op.  That was luck, not correctness.
    #
    # The ZODI ceiling is deliberately NOT excluded.  It CLIPS rather than
    # replaces, so the raw prediction on gated rows is already close to the
    # truth (+0.069% on 1.3.2, against the moon's +129%) and its lift moves
    # the zodi amplitude by only +0.030%; excluding its 54% of rows would
    # halve the calibration sample for no measured gain.  Measured on both
    # corpora 2026-09-23.  The plumbing below is per-group, so adding 'zodi'
    # to `_calib_rule_free` is all it would take if that ever changes.
    _CALIB_MIN_RULE_FREE_ROWS = 200
    _calib_rule_free: dict[str, np.ndarray] = {}
    if _moon_down_rule is not None and _moon_down is not None:
        _keep_moon = ~np.asarray(_moon_down, dtype=bool)[_calib_idx]
        _n_keep = int(_keep_moon.sum())
        if _n_keep >= _CALIB_MIN_RULE_FREE_ROWS:
            _calib_rule_free['moon'] = _keep_moon
            print(f'Calibration: moon lift fitted on the {_n_keep}/'
                  f'{_keep_moon.size} calib rows whose amplitude the moon-down '
                  f'rule does NOT replace.')
        else:
            print(f'Calibration: only {_n_keep} moon-up calib rows '
                  f'(< {_CALIB_MIN_RULE_FREE_ROWS}); moon lift falls back to '
                  f'all rows, including rule-determined ones.')

    def _calib_rows_for(_gname, _mask=None):
        """Row mask for group ``_gname``, intersected with any rule exclusion."""
        _rf = _calib_rule_free.get(_gname)
        if _rf is None:
            return _mask
        return _rf if _mask is None else (_mask & _rf)

    _CALIB_LIFT_CLIP = (0.5, 2.0)  # sanity bounds; outside indicates a broken group
    # 2026-08-19: moon uses a per-coefficient lift; every other group uses the historical
    # scalar lift.  Rationale: cell 27 shows the moon residual has a spectral tilt bias
    # (+0.054 blue / -0.018 NIR) that a single per-group scalar cannot correct.  Per-coef
    # lift on the 29 Moon_bs knots nulls the tilt by construction.  Tight [0.7, 1.4] clip
    # so a broken knot doesn't cascade.
    # 2026-08-24: zodi promoted to per-coef too. Residual coherent bias is
    # ~4% moon-down / ~5.7% Q4 phase after the ctx expansion; it is shape-
    # dependent (7 Zodi_bs knots have different spectral response) so a
    # single scalar cannot correct it.
    _MOON_LIFT_CLIP = (0.7, 1.4)
    _ZODI_LIFT_CLIP = (0.7, 1.4)
    _PER_COEF_LIFT_GROUPS = ('moon', 'zodi')  # 2026-08-24e: zodi re-enabled with regime-aware per-coef lift (moon_up / moon_horizon / moon_down buckets).
    _ZODI_REGIME_THR_UP = 10.0
    _ZODI_REGIME_THR_DN = -10.0
    _ZODI_REGIME_BOUNDARY_DEG = 5.0
    _ZODI_REGIME_MIN_ROWS = 30
    _ctx_names_calib = list(filtered['ctx_names'])
    _moon_alt_idx_calib = _ctx_names_calib.index('moon_alt') if 'moon_alt' in _ctx_names_calib else None
    if _moon_alt_idx_calib is not None:
        _calib_moon_alt = np.asarray(_calib_ctx_sci_phys[:, _moon_alt_idx_calib], dtype=np.float64)
    else:
        _calib_moon_alt = np.zeros(_calib_ctx_sci_phys.shape[0], dtype=np.float64)
    for _gname, _comp in compressors.items():
        _gidx = np.asarray(_comp['coef_indices'], dtype=int)
        if _gname in _PER_COEF_LIFT_GROUPS:
            _pc_clip = _MOON_LIFT_CLIP if _gname == 'moon' else _ZODI_LIFT_CLIP
            def _per_coef_lift(_mask=None, _gname=_gname):
                # Per-coef lift on the specified row mask; falls back to global
                # if mask is None.  Rows whose amplitude an inference rule
                # replaces are removed first -- see `_calib_rule_free` above.
                _mask = _calib_rows_for(_gname, _mask)
                if _mask is None:
                    _mt = np.mean(_calib_true_phys[:, _gidx], axis=0).astype(np.float64)
                    _mp = np.mean(_calib_pred_phys_naive[:, _gidx], axis=0).astype(np.float64)
                else:
                    _mt = np.mean(_calib_true_phys[_mask][:, _gidx], axis=0).astype(np.float64)
                    _mp = np.mean(_calib_pred_phys_naive[_mask][:, _gidx], axis=0).astype(np.float64)
                _rel = np.abs(_mp) / np.maximum(np.abs(_mt), 1e-30)
                _lift = np.where(
                    (_mt > 0) & (_mp > 0) & (_rel >= 0.05),
                    _mt / np.where(_mp != 0.0, _mp, 1.0),
                    1.0,
                )
                return np.clip(_lift, _pc_clip[0], _pc_clip[1]).astype(np.float64)
            if _gname == 'zodi':
                # 1D regime lift: 3 moon_alt buckets (moon_up/moon_horizon/moon_down).
                _regime_masks = {
                    'moon_up':      _calib_moon_alt > _ZODI_REGIME_THR_UP,
                    'moon_horizon': (_calib_moon_alt >= _ZODI_REGIME_THR_DN)
                                    & (_calib_moon_alt <= _ZODI_REGIME_THR_UP),
                    'moon_down':    _calib_moon_alt < _ZODI_REGIME_THR_DN,
                }
                _lift_global = _per_coef_lift(None)
                _regime_lifts = {}
                for _rname, _mask in _regime_masks.items():
                    _n_regime = int(_mask.sum())
                    if _n_regime < _ZODI_REGIME_MIN_ROWS:
                        _regime_lifts[_rname] = _lift_global.copy()
                        print(f'Calibration: zodi/{_rname} n={_n_regime} '
                              f'(< {_ZODI_REGIME_MIN_ROWS}); using global lift.')
                        continue
                    _regime_lifts[_rname] = _per_coef_lift(_mask)
                    _l = _regime_lifts[_rname]
                    print(f'Calibration: zodi/{_rname} n={_n_regime} per-coef '
                          f'lift range=[{_l.min():.3f}, {_l.max():.3f}] '
                          f'median={float(np.median(_l)):.3f}')
                jensen_corrections[_gname] = {
                    'moon_up': _regime_lifts['moon_up'],
                    'moon_horizon': _regime_lifts['moon_horizon'],
                    'moon_down': _regime_lifts['moon_down'],
                    'moon_up_threshold_deg': _ZODI_REGIME_THR_UP,
                    'moon_down_threshold_deg': _ZODI_REGIME_THR_DN,
                    'boundary_scale_deg': _ZODI_REGIME_BOUNDARY_DEG,
                }
                continue
            _lift_pc_clipped = _per_coef_lift(None)   # the helper clips internally
            jensen_corrections[_gname] = _lift_pc_clipped.astype(np.float64)
            print(f'Calibration: {_gname} per-coef lift range=[{_lift_pc_clipped.min():.3f}, '
                  f'{_lift_pc_clipped.max():.3f}] median={float(np.median(_lift_pc_clipped)):.3f} '
                  f'(clip range {_pc_clip})')
            continue
        # 2026-09-11: the scalar lift is fitted on the FLUX-WEIGHTED amplitude
        # A_g = c_g . w_g, not on the unweighted mean coefficient.
        #
        # Why this had to change.  What the delivered sky spectrum depends on is
        # the linear functional A_g, with w_g the per-coefficient template
        # integral; mean(c_g) is a different functional, and for a group whose
        # basis functions have very different integrals the two disagree in
        # SIGN.  Measured on gaia-stars-mask-cont (956 filtered every10 rows,
        # 10-seed ensemble): across the 357 OH sticks w spans 1.4e9x and
        # rho(w, mean true coef) = -0.488 -- the template integral is
        # ANTI-correlated with coefficient size, so the unweighted mean is
        # dominated by exactly the sticks that carry the least flux.  The lift
        # that came out was x0.98566 (scaling OH down) where the flux needed
        # x1.0209 (up), which took the OH band-integrated bias from -0.38%
        # before calibration to -1.81% after: the correction had the wrong sign
        # for the functional that matters, and OH carries 72-78% of the
        # band-integrated flux error tail.
        #
        # The same failure was already diagnosed for `continuum` on 2026-09-04
        # (see `_precompute_flux_basis_and_geometry`) and patched there by
        # wiring up its flux-space loss term; this fixes the calibration itself,
        # which is the step that actually sets the mean bias.
        #
        # Corroboration that the functional is the whole story: the two groups
        # that already used PER-COEFFICIENT lifts came out unbiased in amplitude
        # (moon -0.00225 dex, zodi -0.00000) while both scalar-lift groups did
        # not (continuum +0.00169, mesospheric -0.00793), ordered exactly by how
        # far their template integrals spread (109x, 3.4x, 11.7x, 1.4e9x).
        #
        # Coefficients with no static basis get w = 0 and so drop out of the
        # FIT (O2_b01, whose template is the per-row VECTOR_O2).  The resulting
        # scalar still multiplies them, exactly as before -- a uniform per-group
        # lift is all `inverse_group_compressor` accepts, and O2_b01's flux is
        # not separable here to do better.
        _w_g = None
        if _calib_amp_w is not None:
            _w_try = np.asarray(_calib_amp_w, dtype=np.float64)[_gidx]
            if np.any(np.isfinite(_w_try) & (_w_try > 0.0)):
                _w_g = np.where(np.isfinite(_w_try), np.clip(_w_try, 0.0, None), 0.0)
        _rows = _calib_rows_for(_gname)
        _true_g = (_calib_true_phys if _rows is None else _calib_true_phys[_rows])
        _pred_g = (_calib_pred_phys_naive if _rows is None
                   else _calib_pred_phys_naive[_rows])
        if _w_g is None:
            _mean_true = float(np.mean(_true_g[:, _gidx]))
            _mean_pred = float(np.mean(_pred_g[:, _gidx]))
            _lift_basis = 'mean coefficient (no basis weights)'
        else:
            _mean_true = float(np.mean(_true_g[:, _gidx] @ _w_g))
            _mean_pred = float(np.mean(_pred_g[:, _gidx] @ _w_g))
            _lift_basis = 'flux amplitude'
        _calib_lift_basis[_gname] = _lift_basis
        _calib_lift_means[_gname] = (_mean_true, _mean_pred)
        _rel_mag = abs(_mean_pred) / max(abs(_mean_true), 1e-30)
        if (not np.isfinite(_mean_true) or not np.isfinite(_mean_pred)
                or _mean_true * _mean_pred <= 0.0
                or _rel_mag < 0.05):
            print(f'Calibration: {_gname} skipped (degenerate or near-zero '
                  f'{_lift_basis}; true={_mean_true:.4g}, '
                  f'pred_naive={_mean_pred:.4g}).')
            continue
        _raw_lift = _mean_true / _mean_pred
        _lift = float(np.clip(_raw_lift, _CALIB_LIFT_CLIP[0], _CALIB_LIFT_CLIP[1]))
        if _raw_lift != _lift:
            print(f'Calibration: {_gname} lift {_raw_lift:.4f} clipped to {_lift:.4f}.')
        jensen_corrections[_gname] = np.full(_gidx.size, _lift, dtype=np.float64)

    if jensen_corrections:
        print('Empirical per-group mean-bias calibration (train+val rows, uniform per-group scalar):')
        print(f"  {'group':<14s} {'n_g':>4s} {'mean_true':>10s} {'mean_pred_naive':>16s} "
              f"{'lift':>7s} {'delta_%':>8s}  fitted on")
        for _gname, _corr in jensen_corrections.items():
            _gidx = np.asarray(compressors[_gname]['coef_indices'], dtype=int)
            # Prefer the means the lift was actually computed from; the per-coef
            # groups never recorded any, so fall back to the unweighted mean for
            # display only.  That fallback uses the SAME rows the lift was fitted
            # on, or the printed mean would not match the printed lift for a
            # group with a rule exclusion (moon).
            _rows_show = _calib_rows_for(_gname)
            _mean_true, _mean_pred = _calib_lift_means.get(
                _gname,
                (float(np.mean((_calib_true_phys if _rows_show is None
                                else _calib_true_phys[_rows_show])[:, _gidx])),
                 float(np.mean((_calib_pred_phys_naive if _rows_show is None
                                else _calib_pred_phys_naive[_rows_show])[:, _gidx]))))
            _basis_txt = _calib_lift_basis.get(_gname, 'per-coefficient')
            if _rows_show is not None:
                _basis_txt = f'{_basis_txt}, rule-free rows'
            # 2026-08-24e: regime lift dict summary uses the moon_horizon per-coef vector.
            if isinstance(_corr, dict) and ('moon_horizon' in _corr or 'phase_q2_moon_horizon' in _corr):
                _lift_vec = np.asarray((_corr['moon_horizon'] if 'moon_horizon' in _corr else _corr['phase_q2_moon_horizon']), dtype=np.float64)
                _lift = float(np.median(_lift_vec))
                _n_show = int(_lift_vec.size)
                _delta_pct = 100.0 * (_lift - 1.0)
                print(f'  {_gname:<14s} {_n_show:>4d} {_mean_true:>10.4g} '
                      f'{_mean_pred:>16.4g} {_lift:>7.4f} {_delta_pct:>+7.2f}% '
                      f' {_basis_txt} (regime, median)')
                continue
            _lift = float(_corr[0])
            _delta_pct = 100.0 * (_lift - 1.0)
            print(f'  {_gname:<14s} {len(_corr):>4d} {_mean_true:>10.4g} '
                  f'{_mean_pred:>16.4g} {_lift:>7.4f} {_delta_pct:>+7.2f}%'
                  f'  {_basis_txt}')

    # Per-group upper cap = 3.0 x max(coef_sci_train, axis=0) per coefficient.
    # Defensive guard applied at inference in expand_scores_to_coefs (§11 item 11).
    coef_upper_bound = {}
    _coef_sci_train_arr = np.asarray(coef_sci, dtype=np.float64)[train_idx]
    for _gname, _gidx_local in group_indices.items():
        _gidx_local = np.asarray(_gidx_local, dtype=int)
        if _gidx_local.size == 0:
            continue
        _max_train = np.max(_coef_sci_train_arr[:, _gidx_local], axis=0)
        coef_upper_bound[_gname] = (
            3.0 * np.maximum(_max_train, 0.0)).astype(np.float32)

    return {
        'model': model,
        'device': device,
        'score_scaler': score_scaler,
        'ctx_scaler': ctx_scaler,
        'compressors': compressors,
        'jensen_corrections': jensen_corrections,
        'coef_upper_bound': coef_upper_bound,
        'geom_kwargs': geom_kwargs,
        'group_indices': group_indices,
        'score_slices': score_slices,
        'group_score_dims': group_score_dims,
        'n_input_score': n_input_score,
        'history': history,
        'blend_history': blend_history,
        'blend_init_alpha': (dict(blend_init_alpha) if isinstance(blend_init_alpha, Mapping)
                             else float(blend_init_alpha)),
        'alpha_lr_mult': float(alpha_lr_mult),
        'best_val_loss': float(best_val),
        'best_epoch': int(best_epoch),
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
        'coef_names': [str(x) for x in filtered['coef_names']],
        'ctx_names': [str(x) for x in filtered['ctx_names']],
        'zodi_correction': str(filtered.get('zodi_correction', 'none')),
        'ctx_geometry_version': int(_data_mod.CTX_GEOMETRY_VERSION),
        'moon_down_amp_rule': _moon_down_rule,
        'zodi_ceiling_rule': _zodi_ceiling,
        'config': {
            'n_epochs': int(n_epochs), 'batch_size': int(batch_size),
            'lr': float(lr), 'encoder_dims': tuple(int(v) for v in encoder_dims),
            'ctx_dims': tuple(int(v) for v in ctx_dims),
            'trunk_dims': tuple(int(v) for v in trunk_dims),
            'head_dim': int(head_dim), 'weight_decay': float(weight_decay),
            'zodi_head_extra_dims': tuple(int(v) for v in zodi_head_extra_dims),
            'continuum_head_extra_dims': tuple(int(v) for v in continuum_head_extra_dims),
            'continuum_branch_dims': tuple(int(v) for v in continuum_branch_dims),
            'zodi_ctx_restriction': tuple(str(x) for x in zodi_ctx_restriction),
            'continuum_ctx_restriction': tuple(str(x) for x in continuum_ctx_restriction),
            'moon_zodi_ctx_restriction': tuple(str(x) for x in moon_zodi_ctx_restriction),
            'moon_zodi_coupling_dims': tuple(int(v) for v in moon_zodi_coupling_dims),
            'alpha_ctx_features': tuple(str(x) for x in alpha_ctx_features),
            'patience': int(patience), 'seed': int(seed),
            'moon_group_weight': float(moon_group_weight),
            'zodi_group_weight': float(zodi_group_weight),
            'continuum_group_weight': float(continuum_group_weight),
            'mesospheric_group_weight': float(mesospheric_group_weight),
            'ionospheric_group_weight': float(ionospheric_group_weight),
            'blend_init_alpha': (dict(blend_init_alpha) if isinstance(blend_init_alpha, Mapping)
                                 else float(blend_init_alpha)),
            'alpha_lr_mult': float(alpha_lr_mult),
            'flux_mse_groups': tuple(flux_mse_groups),
            'flux_amp_lambda': (dict(flux_amp_lambda)
                                if isinstance(flux_amp_lambda, dict)
                                else float(flux_amp_lambda)),
            'flux_amp_floor_frac': float(flux_amp_floor_frac),
            'flux_pixel_weighted': bool(flux_pixel_weight is not None),
            'coef_err_sigma_floor_rel': _resolved_floor_by_group,
            'moon_down_amp_free': bool(moon_down_amp_free),
            'moon_down_amp_rule': bool(moon_down_amp_rule),
            'moon_down_alt_deg': float(moon_down_alt_deg),
            'moon_down_ratio_transfer': bool(moon_down_ratio_transfer),
            'train_row_mask_n_dropped': (None if train_row_mask is None
                                        else int((~np.asarray(train_row_mask, bool)).sum())),
            'moon_down_frac_max': (None if moon_down_frac_max is None
                                   else float(moon_down_frac_max)),
            'zodi_ceiling_rule': bool(zodi_ceiling_rule),
            'zodi_ceiling_amp_free': bool(zodi_ceiling_amp_free),
            'zodi_ceiling_gate_frac': float(zodi_ceiling_gate_frac),
            'zodi_ceiling_snap_frac': float(zodi_ceiling_snap_frac),
        },
    }


def predict_sci_coefficients_default(artifacts, coef_near_phys, coef_far_phys,
                                     ctx_near_phys, ctx_far_phys, ctx_sci_phys):
    """Default sky-to-science coefficient predictor (compressed model, §5.5).

    Pipeline: physical coefficients -> divide by geometry factor (§4) ->
    per-group forward compressor (asinh / linear / sqrt + PCA rotation +
    xarm-selected retained subspace) -> encoder + fusion + trunk + heads ->
    inverse compressor -> multiply by science-pointing geometry factor ->
    non-negativity clip.

    When ``artifacts`` is an N-seed ensemble (``is_ensemble=True``), predicts
    with each member and returns the arithmetic mean in physical space
    (the deployed default since 2026-08-11; see §12).
    """
    if artifacts.get('is_ensemble', False):
        _preds = [predict_sci_coefficients_default(
                      m, coef_near_phys, coef_far_phys,
                      ctx_near_phys, ctx_far_phys, ctx_sci_phys)
                  for m in artifacts['members']]
        return np.mean(np.stack(_preds, axis=0), axis=0).astype(np.float32)
    model = artifacts['model']
    device = artifacts['device']
    score_scaler = artifacts['score_scaler']
    ctx_scaler = artifacts['ctx_scaler']
    compressors = artifacts['compressors']
    geom_kwargs = artifacts['geom_kwargs']
    group_indices = artifacts['group_indices']
    score_slices = artifacts['score_slices']
    n_coef = len(artifacts['coef_names'])

    scores_near, _ = compress_coefs_to_scores(
        np.asarray(coef_near_phys, dtype=np.float64),
        np.asarray(ctx_near_phys, dtype=np.float64),
        compressors, geom_kwargs, group_indices)
    scores_far, _ = compress_coefs_to_scores(
        np.asarray(coef_far_phys, dtype=np.float64),
        np.asarray(ctx_far_phys, dtype=np.float64),
        compressors, geom_kwargs, group_indices)

    near_s = np.clip(score_scaler.transform(scores_near.astype(np.float32)),
                     -25.0, 25.0).astype(np.float32)
    far_s = np.clip(score_scaler.transform(scores_far.astype(np.float32)),
                    -25.0, 25.0).astype(np.float32)
    ctx_near_n = np.clip(ctx_scaler.transform(np.asarray(ctx_near_phys, dtype=np.float32)),
                         -25.0, 25.0).astype(np.float32)
    ctx_far_n = np.clip(ctx_scaler.transform(np.asarray(ctx_far_phys, dtype=np.float32)),
                        -25.0, 25.0).astype(np.float32)
    ctx_sci_n = np.clip(ctx_scaler.transform(np.asarray(ctx_sci_phys, dtype=np.float32)),
                        -25.0, 25.0).astype(np.float32)

    with torch.no_grad():
        pred_dict = model(
            torch.from_numpy(near_s).to(device),
            torch.from_numpy(far_s).to(device),
            torch.from_numpy(ctx_near_n).to(device),
            torch.from_numpy(ctx_far_n).to(device),
            torch.from_numpy(ctx_sci_n).to(device),
        )

    n_rows = near_s.shape[0]
    n_score_total = artifacts['n_input_score']
    pred_scaled = np.zeros((n_rows, n_score_total), dtype=np.float64)
    for g, (lo, hi) in score_slices.items():
        pred_scaled[:, lo:hi] = pred_dict[g].detach().cpu().numpy().astype(np.float64)

    pred_scores = score_scaler.inverse_transform(
        pred_scaled.astype(np.float32)).astype(np.float64)

    coef_predicted = expand_scores_to_coefs(
        pred_scores, np.asarray(ctx_sci_phys, dtype=np.float64),
        compressors, group_indices, geom_kwargs, n_coef, score_slices,
        jensen_corrections=artifacts.get('jensen_corrections'),
        coef_upper_bound=artifacts.get('coef_upper_bound'))
    # Constraint-determined amplitudes, DERIVED rather than learned.  Both are
    # no-ops for artifacts with no rule attached, so ensembles trained before
    # they existed stay bit-identical, and both are applied per member rather
    # than after the ensemble mean because A is linear in the coefficients, so
    # the mean of rule-satisfying members satisfies the rules exactly too.
    #
    # ORDER MATTERS: the zodi ceiling runs FIRST because the moon-down rule
    # reads the predicted zodi amplitude, and should read the corrected one.
    coef_predicted = apply_zodi_ceiling_rule(
        coef_predicted, ctx_sci_phys, artifacts)
    coef_predicted = apply_moon_down_amplitude_rule(
        coef_predicted, ctx_sci_phys, artifacts,
        coef_near_phys=coef_near_phys)
    return np.asarray(coef_predicted).astype(np.float32)


# --- Deployed ensemble config (matches the shipped mlp_ensemble_split_zodi_current.pt) ---
default_dual_group_config: dict[str, Any] = {
    "name": "dual_group_mlp_compressed",
    # 2026-09-18: 50 -> 300 epochs, patience 12 -> 100.  Early stopping here
    # does NOT protect against overfitting -- the loop keeps `best_state` and
    # restores it -- so patience only ever saved compute, and at 12 it was
    # firing inside the plateau noise.  Simulated on four recorded 300-epoch
    # curves (telluric corpus), stop/best epoch per seed:
    #
    #   patience   seed42    seed43    seed44    seed45   mean best_val
    #         12    17/  5    51/ 39    58/ 46    24/ 12      0.018813
    #         20    91/ 71    59/ 39    66/ 46    51/ 31      0.018537
    #         60   194/134    99/ 39   217/157   120/ 60      0.018423
    #        120   300/299   159/ 39   277/157   180/ 60      0.018358
    #        none  300/299   300/242   300/157   300/ 60      0.018292
    #
    # At 12 two of four seeds stopped at epoch 17 and 24, keeping a model from
    # epoch 5 and 12.  The stall before the eventual best epoch is 28-131
    # epochs, so anything under ~100 truncates a real run.
    #
    # The budget is worth paying because it is measured in FLUX space, not in
    # the coefficient metrics: 50 -> 300 epochs improves the reconstructed sky
    # by -8.9% fractional RMS (-13.8% in the blue) and -12.0% integrated error,
    # better on 4/4 paired seeds.  mean_eRMSE/median_corr say the OPPOSITE
    # because 92.3% of those averages is the mesospheric block -- see the
    # §3 of the notebook doc.
    "n_epochs": 300,
    "batch_size": 512,
    "lr": 1.0e-3,
    "encoder_dims": (768, 384),
    "ctx_dims": (96,),
    "trunk_dims": (320, 160),
    "head_dim": 192,
    "zodi_head_extra_dims": (32,),
    "continuum_head_extra_dims": (64,),
    "continuum_branch_dims": (128, 64),
    "moon_zodi_coupling_dims": (64, 32),
    "blend_init_alpha": 0.7,
    "alpha_lr_mult": 1.0,
    "weight_decay": 1.0e-4,
    "patience": 100,
    "moon_group_weight": 2.0,
    "zodi_group_weight": 2.0,
    "continuum_group_weight": 1.0,
    "mesospheric_group_weight": 1.0,
    "ionospheric_group_weight": 1.0,
    "flux_mse_groups": ("moon", "zodi"),
    "flux_amp_lambda": 0.0,
    "moon_down_amp_free": True,
    "moon_down_amp_rule": True,
    "moon_down_alt_deg": MOON_DOWN_ALT_DEG,
    "moon_down_frac_max": MOON_DOWN_FRAC_MAX,
    "moon_down_ratio_transfer": True,
    "zodi_ceiling_rule": True,
    "zodi_ceiling_amp_free": False,
    "zodi_ceiling_gate_frac": ZODI_CEILING_GATE_FRAC,
    "zodi_ceiling_snap_frac": ZODI_CEILING_SNAP_FRAC,
    "flux_amp_floor_frac": 0.05,
    # Per-pixel inverse-variance weighting of the flux-space term, from the
    # photon-noise model in mlp_predictor.noise (sigma = sqrt(flux * sens)).
    # Requires input_fits_flux; without it the term stays unweighted.
    "flux_pixel_weighting": True,
    "flux_pixel_weight_floor_frac": 0.05,
    "coef_err_sigma_floor_rel": dict(DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP),
    "ensemble_seeds": (42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    "zodi_ctx_restriction": (
        "airmass", "vanrhijn_285km",
        "ecl_beta_deg", "ecl_lon_sin", "ecl_lon_cos",
        "zodi_log10_v", "sun_sep",
        "moon_alt", "moon_sep",
        "moon_phase_sin", "moon_phase_cos",
        "moon_fli", "moon_up_smooth",
        "moon_airmass_up", "moon_signal_proxy",
    ),
    "continuum_ctx_restriction": (
        "moon_alt", "moon_sep",
        "moon_phase_sin", "moon_phase_cos",
        "moon_fli", "airmass",
        "moon_fli_x_phase_cos",
        "moon_sig_x_lon_cos", "moon_sig_x_lon_sin",
    ),
    "moon_zodi_ctx_restriction": (
        "airmass", "vanrhijn_285km",
        "ecl_beta_deg", "ecl_lon_sin", "ecl_lon_cos",
        "zodi_log10_v", "sun_sep",
        "moon_alt", "moon_sep",
        "moon_phase_sin", "moon_phase_cos",
        "moon_fli", "moon_up_smooth",
        "moon_airmass_up", "moon_signal_proxy",
        "moon_fli_x_phase_cos",
        "moon_sig_x_lon_cos", "moon_sig_x_lon_sin",
    ),
    "alpha_ctx_features": ("moon_up_smooth", "ecl_beta_deg", "airmass"),
}

_WAVE_STRIDE_FLUX_LOSS = 1


_ENSEMBLE_SHARED: dict = {}


def _ensemble_worker_init(payload):
    """Stage the shared training inputs once per worker process."""
    global _ENSEMBLE_SHARED
    _ENSEMBLE_SHARED = payload


def _member_to_cpu(member):
    """Move a trained member off the accelerator so it can be pickled home.

    A member trained on MPS/CUDA holds device tensors, and those cannot cross
    a process boundary.  Only the top level is walked: everything the ensemble
    consumes (`model`, `state_dict`) lives there.
    """
    for key, value in list(member.items()):
        if isinstance(value, torch.nn.Module):
            member[key] = value.to('cpu')
        elif torch.is_tensor(value):
            member[key] = value.detach().to('cpu')
        elif isinstance(value, dict) and value and all(
                torch.is_tensor(v) for v in value.values()):
            member[key] = {k: v.detach().to('cpu') for k, v in value.items()}
    return member


def _select_torch_device():
    """The device `train_compressed_group_mlp` would pick, same order."""
    if torch.cuda.is_available():
        return 'cuda'
    _mps = getattr(torch.backends, 'mps', None)
    if _mps is not None and _mps.is_available():
        return 'mps'
    return 'cpu'


def _member_to_device(member, device):
    """Undo `_member_to_cpu` so a worker-trained member matches a local one.

    The sequential path leaves each member's model on the accelerator and
    everything downstream (per-seed test metrics, the ensemble prediction)
    stages its inputs there, so a member that came home on CPU has to go back
    or the first forward pass raises a device mismatch.
    """
    for key, value in list(member.items()):
        if isinstance(value, torch.nn.Module):
            member[key] = value.to(device)
        elif torch.is_tensor(value):
            member[key] = value.to(device)
        elif isinstance(value, dict) and value and all(
                torch.is_tensor(v) for v in value.values()):
            member[key] = {k: v.to(device) for k, v in value.items()}
    return member


def _ensemble_worker_run(seed):
    """Train ONE ensemble member in this worker; returns (seed, member)."""
    payload = _ENSEMBLE_SHARED
    if not payload:
        raise RuntimeError("ensemble worker was not initialised")
    member = train_compressed_group_mlp(
        payload['filtered_triplet'], payload['compressors'],
        payload['group_indices'], payload['geom_kwargs'],
        split_indices=payload['split_indices'],
        seed=int(seed),
        train_row_mask=payload['train_row_mask'],
        **payload['shared'],
    )
    return int(seed), _member_to_cpu(member)


def _precompute_flux_basis_and_geometry(
    *,
    filtered_triplet,
    compress_geom_kwargs,
    input_fits_for_basis,
    n_moon_knots,
    split_zodi,
    n_zodi_knots,
    palace_oh_suffix=None,
    palace_diffuse_suffix=None,
    decomp_suffix=None,
    group_indices=None,
    input_fits_flux=None,
    pixel_weight_floor_frac=0.05,
    flux_exptime_s=900.0,
    verbose=True,
):
    """Native-grid flux bases, geometry, pixel weights, and calibration weights.

    The fourth return value, ``calib_amplitude_weights``, is a length-n_coef
    vector of NATIVE-GRID template integrals, one per coefficient, aligned to
    ``filtered_triplet['coef_names']`` BY NAME.  The empirical mean-bias
    calibration below uses it to correct the flux-weighted amplitude rather
    than the unweighted mean coefficient -- see the comment on
    ``_CALIB_AMPLITUDE_WEIGHTED`` in the training function.

    The loss and calibration both use every approved native pixel. No
    resampling, binning, stride, or wavelength removal is applied.
    """

    with fits.open(str(input_fits_for_basis)) as hdul:
        wave_ref = np.asarray(hdul["WAVE"].data, dtype=np.float64)
    if decomp_suffix:
        # Build the basis the corpus was actually fitted with. Without this
        # the loss maps coefficients to flux through a DIFFERENT basis than
        # the one that produced them: for the telluric spline2d variant the
        # OH template integrals come out a median 11.2% high (p5-p95
        # 0.713-1.475, worst 2.93x), and lsf_sigma=1.0 A below is 50% wider
        # than the corpus's real 0.67 A.
        model, _basis_note = make_corpus_basis_decomposer(
            wave_ref,
            input_fits_for_basis=input_fits_for_basis,
            decomp_suffix=decomp_suffix,
            n_spline_knots=n_moon_knots,
            split_zodi=split_zodi,
            n_zodi_spline_knots=n_zodi_knots,
            palace_oh_suffix=palace_oh_suffix,
            palace_diffuse_suffix=palace_diffuse_suffix,
        )
        if verbose:
            print(f"  [flux-basis] {_basis_note}")
    else:
        # Historical path, kept so a caller that passes no suffix gets exactly
        # what it got before. It is WRONG for every current corpus -- see the
        # branch above -- so pass cfg.data.decomp_suffix.
        if verbose:
            print("  [flux-basis] no decomp_suffix given: split-zodi basis at "
                  "lsf_sigma=1.0 A, which matches no deployed corpus.")
        model = SkyDecompLSFSurfaceIterative(
            wave_ref, lsf_sigma=1.0, n_spline_knots=n_moon_knots,
            base_dir=_infer_base_dir_for_reconstruction(),
            palace_oh_suffix=palace_oh_suffix,
            palace_diffuse_suffix=palace_diffuse_suffix,
            split_zodi=split_zodi, n_zodi_spline_knots=n_zodi_knots,
        )
    # Per-coefficient NATIVE-grid template integral, keyed BY NAME.
    # `design_names` and `_assemble_design_matrix()` are built from the same
    # block order inside `_build_static_basis`, so zipping them is the class's
    # own invariant rather than an assumption made here; the length check makes
    # a future reordering a loud failure instead of a silent mis-pairing.
    _design_names = [str(_n) for _n in getattr(model, "design_names", ())]
    _design_matrix = np.asarray(getattr(model, "design_matrix", np.empty((0, 0))),
                                dtype=np.float64)
    calib_amplitude_weights = None
    if _design_names and _design_matrix.shape[0] == len(_design_names):
        _w_by_name = dict(zip(_design_names,
                              _design_matrix.sum(axis=1).astype(np.float64)))
        _coef_names_calib = [str(_n) for _n in filtered_triplet["coef_names"]]
        _absent = [_n for _n in _coef_names_calib if _n not in _w_by_name]
        if _absent:
            print(f"  [calib-weights] {len(_absent)} coefficient(s) have no basis "
                  f"row in design_names ({_absent[:4]}); amplitude-weighted "
                  f"calibration disabled, falling back to the mean coefficient.")
        else:
            calib_amplitude_weights = np.array(
                [_w_by_name[_n] for _n in _coef_names_calib], dtype=np.float64)
            if verbose:
                _nz = int(np.sum(calib_amplitude_weights > 0))
                print(f"  [calib-weights] native-grid template integrals for "
                      f"{calib_amplitude_weights.size} coefficients "
                      f"({_nz} non-zero; a zero means no static basis, e.g. "
                      f"O2_b01 whose template is the per-row VECTOR_O2).")
    elif verbose:
        print(f"  [calib-weights] model exposes no aligned design_names / "
              f"design_matrix pair (names={len(_design_names)}, "
              f"rows={_design_matrix.shape[0]}); amplitude-weighted "
              f"calibration disabled.")

    stride = slice(None, None, int(_WAVE_STRIDE_FLUX_LOSS))
    # WHICH basis do the corpus coefficients actually multiply?  Measured
    # 2026-09-18 as integral(c_g @ B) / integral(COMP_g) over 40 every10 sci
    # rows of the telluric corpus -- 1.0 means "this is the convention":
    #
    #     family    refined (_assemble_refined_matrices)   static matrix_*
    #     moon                    0.99987                      1.17357
    #     zodi                    0.99918                      0.99954
    #     diffuse                 1.00000                      1.00000
    #     oh                      0.48751                      2.44421
    #
    # So the REFINED bundle is the corpus convention and `model.matrix_moon`,
    # used here until now, was 17% off in integral.  A pure scale cancels in
    # `scale_match` below, but the shape difference does not, so the refined
    # bundle is used for every family from now on.
    _refined = model._assemble_refined_matrices()
    _names_all = [str(_n) for _n in filtered_triplet["coef_names"]]
    _sel = lambda _f: [i for i, n in enumerate(_names_all) if _f(n)]
    _BLOCKS = {
        "oh": _sel(lambda n: n.startswith("OH_")),
        "moon": _sel(lambda n: n.startswith("Moon_bs")),
        "diffuse": _sel(lambda n: n in ("HO2", "FeO", "O2Ac")),
        "orc": _sel(lambda n: n.startswith("ATOM_Orc")),
        "atom": _sel(lambda n: n.startswith("ATOM_") and not n.startswith("ATOM_Orc")),
        "o2": _sel(lambda n: n == "O2_b01"),
        "zodi": _sel(lambda n: n.startswith("Zodi_bs")),
    }
    _B_all = np.zeros((len(_names_all), wave_ref.size), dtype=np.float64)
    for _k, _idx in _BLOCKS.items():
        _M = np.asarray(_refined.get(_k, np.zeros((0, wave_ref.size))), dtype=np.float64)
        if _M.shape[0] != len(_idx):
            raise RuntimeError(
                f"refined basis {_k!r} has {_M.shape[0]} rows for {len(_idx)} "
                f"coefficient names; the flux basis would pair coefficients "
                f"with the wrong rows")
        if _idx:
            _B_all[np.asarray(_idx, int)] = _M
    if not np.isfinite(_B_all).all():
        raise RuntimeError("refined flux basis contains non-finite rows")

    def _group_basis(_g):
        """Basis rows for a group, IN THE ORDER the loss will index its coefs."""
        if group_indices is None or _g not in group_indices:
            return None
        _gi = np.asarray(group_indices[_g], dtype=int)
        return np.asarray(_B_all[_gi][:, stride], dtype=np.float32)

    flux_basis_matrices = {}
    _moon_basis = _group_basis("moon")
    flux_basis_matrices["moon"] = (
        np.asarray(model.matrix_moon[:, stride], dtype=np.float32)
        if _moon_basis is None else _moon_basis)
    if split_zodi and model.matrix_zodi.shape[0] > 0:
        _zb = _group_basis("zodi")
        flux_basis_matrices["zodi"] = (
            np.asarray(model.matrix_zodi[:, stride], dtype=np.float32)
            if _zb is None else _zb)
    # The diffuse continuum (HO2 + FeO + O2Ac) has its own basis in exactly the
    # same layout, and until 2026-09-04 it simply was not wired up here -- so
    # `continuum` in flux_mse_groups silently did nothing.  It matters: with no
    # flux-space term the group is fit purely in compressed coefficient space,
    # and the trainer's empirical calibration corrects its MEAN COEFFICIENT,
    # which is not its flux-weighted bias because the three basis functions have
    # very different flux integrals.  Measured on new-oh-2: continuum flux bias
    # -3.8%, against moon and zodi inside +/-0.5%.
    _diffuse = getattr(model, "matrix_diffuse", None)
    if _diffuse is not None and np.asarray(_diffuse).shape[0] > 0:
        # Row order is ["HO2", "FeO", "O2Ac"] from _build_diffuse(); the loss
        # indexes the group by position, so a reordering upstream would silently
        # pair each coefficient with the wrong basis row.  The count check in
        # the loss cannot see that, so assert the order here.
        _dnames = [str(n) for n in getattr(model, "diffuse_names", ())]
        _cnames = [str(n) for n in filtered_triplet["coef_names"]]
        _seen = [n for n in _cnames if n in set(_dnames)]
        if _dnames and _seen != _dnames:
            raise RuntimeError(
                f"diffuse basis row order {_dnames} does not match the order the "
                f"same names appear in coef_names ({_seen}); the continuum flux "
                f"term would pair coefficients with the wrong basis rows.")
        _cb = _group_basis("continuum")
        flux_basis_matrices["continuum"] = (
            np.asarray(_diffuse[:, stride], dtype=np.float32) if _cb is None else _cb)
    if group_indices is not None:
        _covered = sum(int(np.asarray(group_indices[_g]).size)
                       for _g in flux_basis_matrices if _g in group_indices)
        print(f"[flux-mse prep] flux basis covers {_covered} of "
              f"{len(_names_all)} coefficients "
              f"({', '.join(sorted(flux_basis_matrices))}); the other groups "
              f"are fitted in compressed coefficient space only")
    flux_geom_sc_sci = airglow_geometry_scale(
        filtered_triplet["ctx_sci"], **compress_geom_kwargs
    ).astype(np.float32)

    # Per-pixel photon weights, on the same native grid as the basis above.  The
    # noise is set by the TOTAL observed science flux, so this reads FLUX_SCI
    # from the full corpus stack -- not `input_fits_for_basis`, which is only
    # the every10 subsample and does not contain the training rows at all.
    flux_pixel_weight = None
    if input_fits_flux:
        row_index = np.asarray(filtered_triplet["row_index"], dtype=np.int64)
        with fits.open(str(input_fits_flux), memmap=True) as hdul:
            wave_flux = np.asarray(hdul["WAVE"].data, dtype=np.float64)
            wave_flux = wave_flux if wave_flux.ndim == 1 else wave_flux[0]
            if (wave_flux.shape != wave_ref.shape
                    or not np.allclose(wave_flux, wave_ref, rtol=0.0, atol=1e-6)):
                raise RuntimeError(
                    f"{input_fits_flux} is on a different wavelength grid from "
                    f"{input_fits_for_basis}; the pixel weights would not line "
                    f"up with the flux basis")
            # Read in row BLOCKS off the memmap, not via `.section[:, stride]`.
            # A strided slice on the SECOND axis makes `.section` fall back to
            # per-element reads: measured 220.5 s against 0.1 s for the blocked
            # form on this 14469 x 12401 float32 array, a 3000x difference, and
            # it was the whole of the delay before training started.  (The two
            # give bitwise-identical values; an `array_equal` check that says
            # otherwise is only seeing NaN != NaN.)
            _hd = hdul["FLUX_SCI"]
            _n_row_all = int(_hd.shape[0])
            _n_ds = len(range(0, int(_hd.shape[1]), int(_WAVE_STRIDE_FLUX_LOSS)))
            _obs_all = np.empty((_n_row_all, _n_ds), dtype=np.float32)
            for _i0 in range(0, _n_row_all, 512):
                _obs_all[_i0:_i0 + 512] = _hd.data[_i0:_i0 + 512, stride]
            obs = _obs_all[row_index].astype(np.float64)
            # Fibre count per row: the science arm is a MEDIAN STACK of a
            # median 536 fibres against ~50 in the sky arms, ranging 4 to 1615,
            # so it sets the effective exposure and cannot be folded into a
            # constant.  Absent, the variance is per-fibre and only the
            # WAVELENGTH weighting survives -- which is all the row-normalised
            # weights use anyway, so it degrades gracefully.
            _nfib = None
            if "META" in [h.name for h in hdul]:
                _mt = hdul["META"].data
                for _c in ("fibers_sci_used", "fibers_sci"):
                    if _c in (_mt.columns.names or []):
                        _nfib = np.asarray(_mt[_c], dtype=np.float64)[row_index]
                        break
        # ABSOLUTE Poisson variance (2026-09-09).  The old relative curve had
        # its per-arm scale divided out upstream and the replacement was
        # eyeballed off a throughput plot; measured against the absolute
        # percentile table it was wrong by 1.60x in r and 2.25x in z, which is
        # a systematic mis-weighting ACROSS the band, exactly what this term
        # exists to set.
        sens = noise.load_absolute_sensitivity(wave_ref, verbose=verbose)[stride]
        # Native dispersion: every 0.5 A pixel is retained, with no resampling,
        # binning, smoothing, or wavelength removal.
        _dwave = float(np.median(np.diff(wave_ref)))
        _var = noise.photon_variance_absolute(
            obs, sens, exptime=float(flux_exptime_s), dwave=_dwave,
            n_fibres=_nfib)
        flux_pixel_weight = noise.weights_from_variance(
            _var, floor_frac=float(pixel_weight_floor_frac))
        if verbose:
            _sg = np.sqrt(np.maximum(_var, 0.0))
            _fr = np.where(np.isfinite(obs) & (obs > 0) & (_sg > 0),
                           _sg / np.abs(obs), np.nan)
            print(f"  [flux-mse] absolute photon sigma/flux: p10/p50/p90 = "
                  f"{100*np.nanpercentile(_fr,10):.3f}% / "
                  f"{100*np.nanpercentile(_fr,50):.3f}% / "
                  f"{100*np.nanpercentile(_fr,90):.3f}%"
                  + ("" if _nfib is None else
                     f"; fibres/row median {np.nanmedian(_nfib):.0f}"))
        if verbose:
            _fin = flux_pixel_weight[np.isfinite(flux_pixel_weight)]
            print(f"  [flux-mse] photon pixel weights {flux_pixel_weight.shape}: "
                  f"1-99% = [{np.percentile(_fin, 1):.3g}, "
                  f"{np.percentile(_fin, 99):.3g}], variance floored at "
                  f"{pixel_weight_floor_frac:g} x the row median")
    if verbose:
        print(
            f"[flux-mse prep] wave grid: n_full={wave_ref.size}, "
            f"stride={_WAVE_STRIDE_FLUX_LOSS}, "
            f"n_ds={flux_basis_matrices['moon'].shape[1]}"
        )
        for k, m in flux_basis_matrices.items():
            print(
                f"[flux-mse prep] {k}: basis shape={m.shape}, "
                f"||A[k]|| range=[{float(np.linalg.norm(m, axis=1).min()):.3g}, "
                f"{float(np.linalg.norm(m, axis=1).max()):.3g}]"
            )
        print(
            f"[flux-mse prep] geom sc_sci: shape={flux_geom_sc_sci.shape}, "
            f"median={float(np.median(flux_geom_sc_sci)):.3g}"
        )
    return (flux_basis_matrices, flux_geom_sc_sci, flux_pixel_weight,
            calib_amplitude_weights)


@dataclass
class EnsembleArtifacts:
    """Combined artifacts of a full ensemble train run."""

    seeds: list
    members: list
    mlp_artifacts: dict
    per_seed_test_metrics: pd.DataFrame
    seed_std_rmse: float
    ensemble_stderr: float


class Trainer:
    """Orchestrates the ensemble training loop.

    Notebook workflow condenses to::

        trainer = Trainer(cfg=default_dual_group_config)
        artifacts = trainer.run_ensemble(
            filtered_triplet, compressors, group_indices, geom_kwargs,
            input_fits_for_basis=..., n_moon_knots=..., split_zodi=..., n_zodi_knots=...,
        )
    """

    #: Every config key the Trainer actually reads.  ``_shared_train_kwargs``
    #: reads all of these except ``ensemble_seeds`` / ``name``, which
    #: ``run_ensemble`` handles.  Keep in sync when adding a knob -- the
    #: ``test_consumed_cfg_keys_match_source`` check greps the class body for
    #: ``c["..."]`` / ``c.get("...")`` and compares against this set.
    _CONSUMED_CFG_KEYS = frozenset({
        "name", "ensemble_seeds",
        "n_epochs", "batch_size", "lr", "weight_decay", "patience",
        "encoder_dims", "ctx_dims", "trunk_dims", "head_dim",
        "zodi_head_extra_dims", "continuum_head_extra_dims",
        "continuum_branch_dims", "moon_zodi_coupling_dims",
        "moon_zodi_ctx_restriction", "zodi_ctx_restriction",
        "continuum_ctx_restriction", "alpha_ctx_features",
        "blend_init_alpha", "alpha_lr_mult",
        "moon_group_weight", "zodi_group_weight", "continuum_group_weight",
        "mesospheric_group_weight", "ionospheric_group_weight",
        "coef_err_sigma_floor_rel", "flux_mse_groups",
        "flux_amp_lambda", "flux_amp_floor_frac",
        "flux_pixel_weighting", "flux_pixel_weight_floor_frac",
    })

    # Keys the trainer reads but does not REQUIRE.  Unlike _CONSUMED_CFG_KEYS a
    # missing entry here is not an error: the trainer's own default IS the
    # intended behaviour, so an older cfg inherits it rather than being
    # rejected.  Present-but-unread keys are still reported below, which is
    # the failure mode _CONSUMED_CFG_KEYS exists to catch.
    _OPTIONAL_CFG_KEYS = frozenset({
        # read via cfg.get(); absent means the trainer default applies
        "ensemble_workers", "ensemble_start_method",
        "moon_down_amp_free", "moon_down_amp_rule", "moon_down_alt_deg",
        "moon_down_frac_max", "moon_down_ratio_transfer",
        "zodi_ceiling_rule", "zodi_ceiling_amp_free",
        "zodi_ceiling_gate_frac", "zodi_ceiling_snap_frac",
    })

    def __init__(self, cfg=None):
        self.cfg = dict(cfg) if cfg is not None else dict(default_dual_group_config)
        # Warn loudly about knobs the Trainer will not read.  The 2026-08-27d
        # dead-code sweep removed a batch of them from the trainer signature
        # but callers (notebook cell 9) still carry the pre-sweep list, so an
        # edit to one of those entries silently does nothing -- exactly how a
        # blend_init_alpha=0.5 A/B ran at the 0.7 default on 2026-09-01.
        _ignored = sorted(set(self.cfg) - self._CONSUMED_CFG_KEYS
                          - self._OPTIONAL_CFG_KEYS)
        if _ignored:
            print(
                f"[Trainer] WARNING: {len(_ignored)} config key(s) are NOT read by "
                f"the trainer and will have no effect on this run:\n"
                f"           {', '.join(_ignored)}\n"
                f"           Remove them, or check the spelling if you meant to "
                f"change behaviour."
            )
        _missing = sorted(self._CONSUMED_CFG_KEYS
                          - {"name", "ensemble_seeds"} - set(self.cfg))
        if _missing:
            raise KeyError(
                f"Trainer cfg is missing required key(s): {', '.join(_missing)}. "
                f"Start from mlp_predictor.trainer.default_dual_group_config.")

    def _shared_train_kwargs(self, *, flux_basis_matrices, flux_geom_sc_sci,
                             flux_pixel_weight=None,
                             calib_amplitude_weights=None,
                             ):
        c = self.cfg
        return dict(
            n_epochs=int(c["n_epochs"]),
            batch_size=int(c["batch_size"]),
            lr=float(c["lr"]),
            encoder_dims=tuple(int(v) for v in c["encoder_dims"]),
            ctx_dims=tuple(int(v) for v in c["ctx_dims"]),
            trunk_dims=tuple(int(v) for v in c["trunk_dims"]),
            head_dim=int(c["head_dim"]),
            zodi_head_extra_dims=tuple(int(v) for v in c["zodi_head_extra_dims"]),
            continuum_head_extra_dims=tuple(int(v) for v in c["continuum_head_extra_dims"]),
            continuum_branch_dims=tuple(int(v) for v in c["continuum_branch_dims"]),
            moon_zodi_ctx_restriction=c["moon_zodi_ctx_restriction"],
            moon_zodi_coupling_dims=tuple(int(v) for v in c["moon_zodi_coupling_dims"]),
            blend_init_alpha=(dict(c["blend_init_alpha"])
                              if isinstance(c["blend_init_alpha"], Mapping)
                              else float(c["blend_init_alpha"])),
            alpha_lr_mult=float(c["alpha_lr_mult"]),
            weight_decay=float(c["weight_decay"]),
            patience=int(c["patience"]),
            moon_group_weight=float(c["moon_group_weight"]),
            zodi_group_weight=float(c["zodi_group_weight"]),
            continuum_group_weight=float(c["continuum_group_weight"]),
            mesospheric_group_weight=float(c["mesospheric_group_weight"]),
            ionospheric_group_weight=float(c["ionospheric_group_weight"]),
            coef_err_sigma_floor_rel=c["coef_err_sigma_floor_rel"],
            zodi_ctx_restriction=c["zodi_ctx_restriction"],
            continuum_ctx_restriction=c["continuum_ctx_restriction"],
            alpha_ctx_features=c["alpha_ctx_features"],
            flux_mse_groups=tuple(c.get("flux_mse_groups", ())),

            flux_amp_lambda=(dict(c["flux_amp_lambda"])
                             if isinstance(c["flux_amp_lambda"], dict)
                             else float(c["flux_amp_lambda"])),
            flux_amp_floor_frac=float(c["flux_amp_floor_frac"]),
            flux_basis_matrices=flux_basis_matrices,
            flux_geom_sc_sci=flux_geom_sc_sci,
            flux_pixel_weight=flux_pixel_weight,
            calib_amplitude_weights=calib_amplitude_weights,
            moon_down_amp_free=bool(c.get("moon_down_amp_free", True)),
            moon_down_amp_rule=bool(c.get("moon_down_amp_rule", True)),
            moon_down_alt_deg=float(c.get("moon_down_alt_deg",
                                          MOON_DOWN_ALT_DEG)),
            moon_down_frac_max=float(c.get("moon_down_frac_max",
                                           MOON_DOWN_FRAC_MAX)),
            moon_down_ratio_transfer=bool(
                c.get("moon_down_ratio_transfer", True)),
            zodi_ceiling_rule=bool(c.get("zodi_ceiling_rule", True)),
            zodi_ceiling_amp_free=bool(c.get("zodi_ceiling_amp_free", False)),
            zodi_ceiling_gate_frac=float(c.get("zodi_ceiling_gate_frac",
                                               ZODI_CEILING_GATE_FRAC)),
            zodi_ceiling_snap_frac=float(c.get("zodi_ceiling_snap_frac",
                                               ZODI_CEILING_SNAP_FRAC)),
        )

    def run_ensemble(
        self,
        filtered_triplet,
        compressors,
        group_indices,
        geom_kwargs,
        *,
        input_fits_for_basis,
        input_fits_flux=None,
        n_moon_knots,
        split_zodi,
        n_zodi_knots,
        palace_oh_suffix=None,
        palace_diffuse_suffix=None,
        decomp_suffix=None,
        train_row_mask=None,
        verbose=True,
    ):
        """Fit the full seed ensemble, assemble artifacts, report per-seed metrics."""

        seeds = tuple(int(s) for s in self.cfg["ensemble_seeds"])
        if verbose:
            print(
                f"=== Training compressed dual-encoder group-head MLP "
                f"({len(seeds)}-seed ensemble default) ==="
            )
            print(self.cfg)

        flux_basis_matrices = None
        flux_geom_sc_sci = None
        flux_pixel_weight = None
        calib_amplitude_weights = None
        if self.cfg.get("flux_mse_groups"):
            # Pixel weighting needs the full-corpus stack.  When the caller does
            # not supply it the loss falls back to the unweighted mean, so an
            # older caller keeps working instead of silently mis-weighting.
            _want_w = bool(self.cfg.get("flux_pixel_weighting", True))
            if _want_w and not input_fits_flux:
                print("  [flux-mse] flux_pixel_weighting is on but "
                      "input_fits_flux was not given; pixels stay UNWEIGHTED "
                      "(pass cfg.data.input_fits_flux to enable it).")
            (flux_basis_matrices, flux_geom_sc_sci, flux_pixel_weight,
             calib_amplitude_weights) = _precompute_flux_basis_and_geometry(
                filtered_triplet=filtered_triplet,
                compress_geom_kwargs=geom_kwargs,
                input_fits_for_basis=input_fits_for_basis,
                n_moon_knots=n_moon_knots,
                split_zodi=split_zodi,
                n_zodi_knots=n_zodi_knots,
                palace_oh_suffix=palace_oh_suffix,
                palace_diffuse_suffix=palace_diffuse_suffix,
                decomp_suffix=decomp_suffix,
                group_indices=group_indices,
                input_fits_flux=(input_fits_flux if _want_w else None),
                pixel_weight_floor_frac=float(
                    self.cfg.get("flux_pixel_weight_floor_frac", 0.05)),
                verbose=verbose,
            )

        shared = self._shared_train_kwargs(
            flux_basis_matrices=flux_basis_matrices,
            flux_geom_sc_sci=flux_geom_sc_sci,
            flux_pixel_weight=flux_pixel_weight,
            calib_amplitude_weights=calib_amplitude_weights,
        )
        split_for_members = (
            filtered_triplet["compress_train_idx"],
            filtered_triplet["compress_val_idx"],
            filtered_triplet["compress_test_idx"],
        )
        # Members are independent given their seed, so they can train
        # concurrently.  MEASURED on this laptop (MPS): one member already uses
        # about half the GPU, so 2 workers buy ~1.5x and 4 buy ~1.9x -- real,
        # but far from linear, and CPU threads are irrelevant (34 s vs 35 s for
        # 30 epochs at 8 vs 2 OMP threads).  Default stays 1 so nothing changes
        # unless asked.
        _n_workers = max(1, int(self.cfg.get("ensemble_workers", 1)))
        members = []
        if _n_workers > 1 and len(seeds) > 1:
            # START METHOD -- 'spawn', and it has to be.  MEASURED: 'fork'
            # kills the worker with BrokenProcessPool as soon as the parent has
            # touched MPS (which it has, if any member trained in-process
            # first), because forking a live Metal context is undefined.  Spawn
            # costs one pickle of the shared arrays per worker, done ONCE by
            # the initializer rather than per member.
            #
            # Spawn re-imports the caller's __main__, so a SCRIPT caller needs
            # the usual `if __name__ == "__main__":` guard or it will re-run
            # its own top level in every worker.  A notebook has no such top
            # level and is fine.  `ensemble_start_method` overrides this.
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            _method = str(self.cfg.get("ensemble_start_method", "spawn"))
            _payload = {
                'filtered_triplet': filtered_triplet,
                'compressors': compressors,
                'group_indices': group_indices,
                'geom_kwargs': geom_kwargs,
                'split_indices': split_for_members,
                'train_row_mask': train_row_mask,
                'shared': shared,
            }
            _n_workers = min(_n_workers, len(seeds))
            if verbose:
                print(f"\n--- Training {len(seeds)} members on {_n_workers} "
                      f"worker process(es) via {_method}; per-member logs "
                      f"interleave ---")
            _by_seed = {}
            with ProcessPoolExecutor(
                    max_workers=_n_workers,
                    mp_context=_mp.get_context(_method),
                    initializer=_ensemble_worker_init,
                    initargs=(_payload,)) as _pool:
                _futs = {_pool.submit(_ensemble_worker_run, int(s)): int(s)
                         for s in seeds}
                for _fut in as_completed(_futs):
                    try:
                        _s, _m = _fut.result()
                    except Exception as _exc:
                        raise RuntimeError(
                            f"ensemble worker failed ({type(_exc).__name__}: "
                            f"{_exc}). With start method {_method!r}: 'spawn' "
                            f"needs a script caller to guard its top level "
                            f"with `if __name__ == \"__main__\":`, and 'fork' "
                            f"breaks once the parent has touched MPS/CUDA. "
                            f"Set ensemble_workers=1 to train sequentially."
                        ) from _exc
                    _by_seed[_s] = _m
                    if verbose:
                        print(f"  [ensemble] seed {_s} done "
                              f"({len(_by_seed)}/{len(seeds)})")
            # Restore the REQUESTED order: as_completed yields by finish time,
            # and the ensemble's member order is part of its identity (the
            # per-member Jensen corrections are indexed by it).
            _dev = _select_torch_device()
            members = [_member_to_device(_by_seed[int(s)], _dev) for s in seeds]
        else:
            for seed in seeds:
                if verbose:
                    print(f"\n--- Ensemble member seed={seed} ---")
                member = train_compressed_group_mlp(
                    filtered_triplet, compressors, group_indices, geom_kwargs,
                    split_indices=split_for_members,
                    seed=int(seed),
                    train_row_mask=train_row_mask,
                    **shared,
                )
                members.append(member)

        first = members[0]
        mlp_artifacts = {
            "is_ensemble": True,
            "seeds": list(seeds),
            "members": members,
            "compressors": first["compressors"],
            "coef_upper_bound": first["coef_upper_bound"],
            "geom_kwargs": first["geom_kwargs"],
            "group_indices": first["group_indices"],
            "score_slices": first["score_slices"],
            "group_score_dims": first["group_score_dims"],
            "n_input_score": first["n_input_score"],
            "coef_names": first["coef_names"],
            "ctx_names": first["ctx_names"],
            "zodi_correction": first.get("zodi_correction", "none"),
            "ctx_geometry_version": int(first.get("ctx_geometry_version", 1)),
            "train_idx": first["train_idx"],
            "val_idx": first["val_idx"],
            "test_idx": first["test_idx"],
            "config": first["config"],
            "best_epochs": [int(m["best_epoch"]) for m in members],
            "best_val_losses": [float(m["best_val_loss"]) for m in members],
        }
        if verbose:
            print(
                f"\nEnsemble assembled: {len(members)} members, "
                f"best_epochs={mlp_artifacts['best_epochs']}, "
                f"best_val_losses={[f'{v:.5f}' for v in mlp_artifacts['best_val_losses']]}"
            )

        per_seed_df, seed_std, stderr = self._report_test_metrics(
            filtered_triplet, mlp_artifacts, group_indices, verbose=verbose,
        )
        return EnsembleArtifacts(
            seeds=list(seeds),
            members=members,
            mlp_artifacts=mlp_artifacts,
            per_seed_test_metrics=per_seed_df,
            seed_std_rmse=seed_std,
            ensemble_stderr=stderr,
        )

    def _report_test_metrics(self, filtered_triplet, mlp_artifacts, group_indices, verbose=True):
        test_idx = np.asarray(mlp_artifacts["test_idx"], dtype=int)
        coef_near = np.asarray(filtered_triplet["coef_near"], dtype=np.float32)
        coef_far = np.asarray(filtered_triplet["coef_far"], dtype=np.float32)
        coef_sci = np.asarray(filtered_triplet["coef_sci"], dtype=np.float32)
        ctx_near = np.asarray(filtered_triplet["ctx_near"], dtype=np.float32)
        ctx_far = np.asarray(filtered_triplet["ctx_far"], dtype=np.float32)
        ctx_sci = np.asarray(filtered_triplet["ctx_sci"], dtype=np.float32)
        coef_err_sci = np.asarray(
            filtered_triplet.get("coef_err_sci", np.full_like(coef_sci, np.nan)),
            dtype=np.float32,
        )
        y_te = coef_sci[test_idx]
        sig_te = coef_err_sci[test_idx]
        floor_te = dict(DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP)

        coef_pred_det = predict_sci_coefficients_default(
            mlp_artifacts,
            coef_near_phys=coef_near[test_idx],
            coef_far_phys=coef_far[test_idx],
            ctx_near_phys=ctx_near[test_idx],
            ctx_far_phys=ctx_far[test_idx],
            ctx_sci_phys=ctx_sci[test_idx],
        ).astype(np.float32)

        rows = []
        for seed, member in zip(mlp_artifacts["seeds"], mlp_artifacts["members"]):
            pred_seed = predict_sci_coefficients_default(
                member,
                coef_near_phys=coef_near[test_idx],
                coef_far_phys=coef_far[test_idx],
                ctx_near_phys=ctx_near[test_idx],
                ctx_far_phys=ctx_far[test_idx],
                ctx_sci_phys=ctx_sci[test_idx],
            ).astype(np.float32)
            rows.append({
                "variant": f"seed={seed}",
                **metric_row(y_te, pred_seed, f"seed={seed}",
                             sigma=sig_te, group_indices=group_indices,
                             floor_by_group=floor_te),
            })
        rows.append({
            "variant": f"{len(mlp_artifacts['members'])}-seed ensemble (default)",
            **metric_row(y_te, coef_pred_det, "ensemble",
                         sigma=sig_te, group_indices=group_indices,
                         floor_by_group=floor_te),
        })
        df = pd.DataFrame(rows)
        if verbose:
            print(
                f"\nPer-seed and ensemble test metrics on the night-held-out split "
                f"(n_test = {test_idx.size} rows):"
            )
            print(df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
        per_seed_rmse = np.array([r["mean_eRMSE"] for r in rows[:len(mlp_artifacts["members"])]])
        seed_std = float(np.std(per_seed_rmse, ddof=1))
        ensemble_rmse = float(df.iloc[-1]["mean_eRMSE"])
        stderr = seed_std / (len(mlp_artifacts["members"]) ** 0.5)
        if verbose:
            print(
                f"\nSeed-to-seed test mean_eRMSE std: {seed_std:.3f}  "
                f"(single-seed noise scale on this dataset)"
            )
            print(
                f"Ensemble-mean stderr = std/sqrt(N={len(mlp_artifacts['members'])}) = "
                f"{stderr:.3f}  (~{100.0 * stderr / max(ensemble_rmse, 1e-30):.1f}% of "
                f"ensemble mean_eRMSE {ensemble_rmse:.3f})."
            )
        return df, seed_std, stderr


__all__ = [
    "DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP",
    "EnsembleArtifacts",
    "Trainer",
    "default_dual_group_config",
    "predict_sci_coefficients_default",
    "train_compressed_group_mlp",
]
