# Sky-arm zodi bias diagnostic per slice.
# Answers: is the residual Q1/Q4 zodi bias an ML tuning issue or a decomp-truth
# artifact? Compares geometry-corrected sky1 (B1) and sky2 (B_far) predictions
# vs sci truth on the same slices used by the ML per-slice metrics cell.
#
# The question this cell answers is ONLY about the ML bias.  A large arm bias
# with a near-zero ML bias is the ML *working* -- the head is correcting a
# geometry-corrected-copy error the arms cannot avoid -- and must never be
# reported as a tuning opportunity.  The triage is therefore:
#   1. |bias_ML| below ML_BIAS_TOL_PCT (relative to the slice mean) AND below
#      ML_BIAS_TOL_GLOBAL_PCT (relative to the all-test mean)
#        -> nothing to fix in this slice, whatever the arms are doing.  If the
#           arms are badly biased here, that is a win, and it is labelled so.
#   2. bias_ML material and tracking bias_B_mean in sign and magnitude
#        -> ML is faithfully reproducing what the sky arms tell it; the residual
#           comes from the decomp truth itself.  No ML tuning can fix that.
#   3. bias_ML material and NOT explained by the arms
#        -> ML is genuinely over- or under-predicting; row-weight boost or
#           regime lift extension can help.
#
# Both tolerances matter.  A percentage bias on a slice whose zodi amplitude is
# a fraction of the corpus mean (moon-down rows sit ~5x below the all-test mean)
# is a small absolute flux error, so a slice-relative percentage on its own
# over-states how much of the deliverable it costs.

# Flag thresholds: a slice is only worth tuning if the ML bias clears BOTH.
ML_BIAS_TOL_PCT = 1.5          # % of the slice's own zodi mean
ML_BIAS_TOL_GLOBAL_PCT = 1.0   # % of the all-test zodi mean (absolute-flux scale)
# bias_ML counts as inherited from the arms when it has the same sign as
# bias_B_mean and |bias_ML - bias_B_mean| <= INHERIT_FRAC * |bias_B_mean|.
INHERIT_FRAC = 0.5

import numpy as np
import pandas as pd

required = ['filtered_triplet', 'test_idx',
            'coef_near_all', 'coef_far_all', 'coef_sci_all',
            'ctx_near_all', 'ctx_far_all', 'ctx_sci_all',
            'coef_pred_det', '_group_indices_compress',
            'compress_geom_kwargs', 'airglow_geometry_scale',
            '_moon_phase_deg_from_ctx']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run prerequisite cells first. Missing: ' + ', '.join(_missing))

_te = np.asarray(test_idx, dtype=int)
_ctx_names = list(filtered_triplet['ctx_names'])

_ctx_near_te = np.asarray(ctx_near_all[_te], dtype=np.float64)
_ctx_far_te  = np.asarray(ctx_far_all[_te],  dtype=np.float64)
_ctx_sci_te  = np.asarray(ctx_sci_all[_te],  dtype=np.float64)

_coef_near_te = np.asarray(coef_near_all[_te], dtype=np.float64)
_coef_far_te  = np.asarray(coef_far_all[_te],  dtype=np.float64)
_coef_sci_te  = np.asarray(coef_sci_all[_te],  dtype=np.float64)
_coef_ml_te   = np.asarray(coef_pred_det,       dtype=np.float64)

# Geometry-corrected sky arms projected onto sci geometry (matches B1/B2 in cell 34).
_G_near = airglow_geometry_scale(_ctx_near_te, **compress_geom_kwargs)
_G_far  = airglow_geometry_scale(_ctx_far_te,  **compress_geom_kwargs)
_G_sci  = airglow_geometry_scale(_ctx_sci_te,  **compress_geom_kwargs)
_em_near_sci = (_coef_near_te / _G_near) * _G_sci
_em_far_sci  = (_coef_far_te  / _G_far)  * _G_sci
_em_mean_sci = 0.5 * (_em_near_sci + _em_far_sci)

_zodi_idx = np.asarray(_group_indices_compress['zodi'], dtype=int)

_moon_phase_te = _moon_phase_deg_from_ctx(
    {'ctx_sci': _ctx_sci_te, 'ctx_names': _ctx_names})
_moon_alt_te = _ctx_sci_te[:, _ctx_names.index('moon_alt')]
_airmass_te = _ctx_sci_te[:, _ctx_names.index('airmass')]

_slices = {
    'all_test':                     np.ones(len(_te), dtype=bool),
    'moon_phase Q1 [2-110 deg]':    (_moon_phase_te >= 2)   & (_moon_phase_te < 110),
    'moon_phase Q2 [110-172 deg]':  (_moon_phase_te >= 110) & (_moon_phase_te < 172),
    'moon_phase Q3 [172-265 deg]':  (_moon_phase_te >= 172) & (_moon_phase_te < 265),
    'moon_phase Q4 [265-358 deg]':  (_moon_phase_te >= 265) & (_moon_phase_te < 358),
    'moon_alt > 0 (moon up)':       _moon_alt_te > 0,
    'moon_alt <= 0 (moon down)':    _moon_alt_te <= 0,
    'airmass <= 1.5 (low)':         _airmass_te <= 1.5,
    'airmass > 1.5 (high)':         _airmass_te > 1.5,
}


def _zodi_bias_pct(pred, truth, mask, idx):
    _t = float(np.mean(truth[mask][:, idx]))
    _p = float(np.mean(pred[mask][:, idx]))
    return 100.0 * (_p - _t) / max(abs(_t), 1e-30)


# All-test zodi mean: the common denominator that puts every slice's ML bias on
# one absolute-flux scale, so a big percentage on a faint slice is visible as the
# small flux error it is.
_zodi_mean_all = float(np.mean(_coef_sci_te[:, _zodi_idx]))

_rows = []
for _name, _mask in _slices.items():
    _n = int(_mask.sum())
    if _n < 20:
        continue
    _ml_abs = float(np.mean(_coef_ml_te[_mask][:, _zodi_idx])
                    - np.mean(_coef_sci_te[_mask][:, _zodi_idx]))
    _rows.append({
        'slice': _name,
        'n_rows': _n,
        'zodi_sci_mean': float(np.mean(_coef_sci_te[_mask][:, _zodi_idx])),
        'bias_B1_sky1_geo_%': _zodi_bias_pct(_em_near_sci, _coef_sci_te, _mask, _zodi_idx),
        'bias_B_sky2_geo_%':  _zodi_bias_pct(_em_far_sci,  _coef_sci_te, _mask, _zodi_idx),
        'bias_B_mean_geo_%':  _zodi_bias_pct(_em_mean_sci, _coef_sci_te, _mask, _zodi_idx),
        'bias_ML_default_%':  _zodi_bias_pct(_coef_ml_te,  _coef_sci_te, _mask, _zodi_idx),
        'bias_ML_vs_alltest_%': 100.0 * _ml_abs / max(abs(_zodi_mean_all), 1e-30),
    })
_sky_zodi_bias_df = pd.DataFrame(_rows)

print('=' * 118)
print(f'Sky-arm zodi bias diagnostic per slice (test rows, n={len(_te)}).')
print('Compares geometry-corrected sky1/sky2/mean and ML against sci truth on the zodi group.')
print(f'bias_ML_vs_alltest_% re-expresses the ML bias against the all-test zodi mean '
      f'({_zodi_mean_all:.4g}).')
print('=' * 118)
print(_sky_zodi_bias_df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print()

print('-' * 118)
print(f'Per-slice verdict.  A slice is ML-tuning-fixable only when the ML bias itself is '
      f'material:\n'
      f'  |bias_ML| > {ML_BIAS_TOL_PCT:.1f}% of the slice mean AND '
      f'> {ML_BIAS_TOL_GLOBAL_PCT:.1f}% of the all-test mean,\n'
      f'  and it is not simply inherited from the sky arms '
      f'(same sign, within {INHERIT_FRAC:.0%} of bias_B_mean).')
print('-' * 118)
_n_tunable = 0
_n_arm_corrected = 0
for _, _r in _sky_zodi_bias_df.iterrows():
    _ml = float(_r['bias_ML_default_%'])
    _bm = float(_r['bias_B_mean_geo_%'])
    _ml_glob = float(_r['bias_ML_vs_alltest_%'])
    _delta = _ml - _bm
    _material = (abs(_ml) > ML_BIAS_TOL_PCT) and (abs(_ml_glob) > ML_BIAS_TOL_GLOBAL_PCT)
    _inherited = (_ml * _bm > 0.0) and (abs(_delta) <= INHERIT_FRAC * abs(_bm))
    if not _material:
        if abs(_bm) > ML_BIAS_TOL_PCT:
            _n_arm_corrected += 1
            _tag = (f' <-- OK: ML costs {_ml_glob:+.2f}% of the all-test mean while removing '
                    f'the {_bm:+.1f}% arm bias (arm-copy error, not an ML error)')
        elif abs(_ml) > ML_BIAS_TOL_PCT:
            _tag = (f' <-- OK: {_ml:+.1f}% is on a faint slice; only {_ml_glob:+.2f}% of the '
                    f'all-test mean, so it costs little flux')
        else:
            _tag = ' <-- OK: no material bias in the arms or the ML'
    elif _inherited:
        _tag = ' <-- ML bias tracks the arms / decomp truth; ML tuning cannot fix it'
    else:
        _n_tunable += 1
        _tag = ' <-- ML-specific bias; row-weight boost or regime-lift extension can help'
    print(f"  {_r['slice']:<32} B_mean={_bm:+7.2f}%  ML={_ml:+6.2f}%  "
          f"(={_ml_glob:+5.2f}% of all-test mean)  delta={_delta:+7.2f}%{_tag}")

print()
print(f'Summary: {_n_tunable} slice(s) carry an ML-specific zodi bias worth tuning; '
      f'{_n_arm_corrected} slice(s) show the ML correcting a material arm bias.')
if _n_tunable == 0:
    print('  No ML-specific zodi bias above threshold -- the zodi head is calibrated on '
          'these slices, and a large bias_B_mean here reflects the arms, not the model.')
