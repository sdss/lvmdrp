# Per-context-slice test metrics (item 15 of the review). A single aggregate
# mean_eRMSE hides the cases the sky-subtraction stage cares about most --
# bright time, moon-up, high-airmass. This cell splits the test set by
# moon_phase quartile, moon_alt sign, and airmass threshold and reports
# mean_eRMSE / median_corr / max per-group |bias| for each slice. Uses the
# ensemble prediction (coef_pred_det) already computed in the trainer cell.

required = ['filtered_triplet', 'mlp_artifacts', 'coef_pred_det',
            '_group_indices_compress', 'coef_sci_all', 'ctx_sci_all',
            'test_idx', '_metric_row', '_moon_phase_deg_from_ctx']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run prerequisite cells first. Missing: ' + ', '.join(_missing))

_te_slice = np.asarray(test_idx, dtype=int)
_ctx_names_ll = list(filtered_triplet['ctx_names'])
_ctx_sci_te = np.asarray(ctx_sci_all[_te_slice], dtype=np.float64)
_y_te_ll = coef_sci_all[_te_slice].astype(np.float32)
_pred_te_ll = np.asarray(coef_pred_det, dtype=np.float32)

# Decoded moon_phase in [0, 360). Reuses the helper the split code uses so the
# slice quartiles match the training split's phase-quantile convention.
_moon_phase_te = _moon_phase_deg_from_ctx(
    {'ctx_sci': _ctx_sci_te, 'ctx_names': _ctx_names_ll})
_moon_alt_te = _ctx_sci_te[:, _ctx_names_ll.index('moon_alt')]
_airmass_te = _ctx_sci_te[:, _ctx_names_ll.index('airmass')]


def _slice_row(name, mask):
    n = int(mask.sum())
    if n < 20:
        return {'slice': name, 'n_rows': n, 'mean_eRMSE': np.nan,
                'median_eRMSE': np.nan, 'median_corr': np.nan,
                'max_abs_bias_%': np.nan, 'worst_bias_group': ''}
    _y = _y_te_ll[mask]
    _p = _pred_te_ll[mask]
    _s = (coef_err_sci_all[test_idx][mask]
          if 'coef_err_sci_all' in globals() else None)
    _floor_g = (dict(DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP)
                 if 'DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP' in globals() else None)
    _m = _metric_row(_y, _p, name,
                     sigma=_s,
                     group_indices=_group_indices_compress,
                     floor_by_group=_floor_g)
    _max_abs = -np.inf
    _worst = ''
    for _gname, _idx in _group_indices_compress.items():
        _idx = np.asarray(_idx, dtype=int)
        _mt = float(np.mean(_y[:, _idx]))
        _mp = float(np.mean(_p[:, _idx]))
        _b = 100.0 * (_mp - _mt) / max(abs(_mt), 1e-30)
        if abs(_b) > _max_abs:
            _max_abs = abs(_b)
            _worst = _gname
    return {'slice': name, 'n_rows': n,
            'mean_eRMSE': float(_m['mean_eRMSE']),
            'median_eRMSE': float(_m['median_eRMSE']),
            'median_corr': float(_m['median_corr']),
            'max_abs_bias_%': float(_max_abs),
            'worst_bias_group': _worst}


_slices = [_slice_row('all_test', np.ones(_te_slice.size, dtype=bool))]

# Moon-phase quartiles across the actual test-set phase distribution.
_q = np.quantile(_moon_phase_te, [0.0, 0.25, 0.5, 0.75, 1.0])
for _k in range(4):
    _lo, _hi = float(_q[_k]), float(_q[_k + 1])
    _mask = ((_moon_phase_te >= _lo)
             & (_moon_phase_te <= _hi if _k == 3 else _moon_phase_te < _hi))
    _slices.append(_slice_row(
        f'moon_phase Q{_k + 1} [{_lo:.0f}-{_hi:.0f} deg]', _mask))

# Moon-alt sign: moon above/below horizon at the science pointing.
_slices.append(_slice_row('moon_alt > 0 (moon up)',    _moon_alt_te > 0.0))
_slices.append(_slice_row('moon_alt <= 0 (moon down)', _moon_alt_te <= 0.0))

# Airmass threshold: 1.5 corresponds to z ~ 48 deg, a common science limit.
_slices.append(_slice_row('airmass <= 1.5 (low)', _airmass_te <= 1.5))
_slices.append(_slice_row('airmass > 1.5 (high)', _airmass_te > 1.5))

slice_metrics_df = pd.DataFrame(_slices)
print(f'Per-context-slice test metrics on the ensemble prediction '
      f'({_te_slice.size} test rows total):')
print(slice_metrics_df.to_string(index=False, float_format=lambda v: f'{v:.4g}'))

# Compact deployment view: relative degradation vs the aggregate, with a flag
# on any slice that carries >20% higher mean_eRMSE or >2% |bias|.
_all = slice_metrics_df.iloc[0]
print(f"\nAggregate reference: n={int(_all['n_rows'])}, "
      f"mean_eRMSE={float(_all['mean_eRMSE']):.3f}, "
      f"median_corr={float(_all['median_corr']):.4f}, "
      f"max_abs_bias={float(_all['max_abs_bias_%']):.2f}% "
      f"({_all['worst_bias_group']})")
print('Slice deviations (>20% mean_eRMSE gap or >2% bias flagged):')
for _, r in slice_metrics_df.iloc[1:].iterrows():
    if r['n_rows'] < 20 or not np.isfinite(r['mean_eRMSE']):
        print(f"  {r['slice']:<40s} n={int(r['n_rows']):>4d}  "
              f'(too few rows, skipped)')
        continue
    _rel = float(r['mean_eRMSE']) / max(float(_all['mean_eRMSE']), 1e-9) - 1.0
    _flag = ''
    if abs(_rel) > 0.2 or float(r['max_abs_bias_%']) > 2.0:
        _flag = '  <-- flagged'
    print(f"  {r['slice']:<40s} n={int(r['n_rows']):>4d}  "
          f"rmse={float(r['mean_eRMSE']):.3f} ({100.0 * _rel:+.1f}%)  "
          f"median_corr={float(r['median_corr']):.4f}  "
          f"max_abs_bias={float(r['max_abs_bias_%']):.2f}% "
          f"({r['worst_bias_group']}){_flag}")
