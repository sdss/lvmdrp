# Predictive uncertainty from ensemble spread. For each row + coefficient
# sigma_row = std_across_seeds(pred). z = (y - mean_pred) / sigma is the standardized
# residual: mean 0 / std 1 under a well-calibrated ensemble. §11.7 wants this before
# any downstream code trusts a sigma. Reports var(z), coverage, PIT chi2; a var(z) >> 1
# means the 4-seed spread only captures init noise, not epistemic + aleatoric error.

from scipy.special import erf as _erf_vec

required = ['filtered_triplet', 'mlp_artifacts', 'coef_sci_all',
            'coef_near_all', 'coef_far_all', 'ctx_sci_all', 'ctx_near_all',
            'ctx_far_all', 'val_idx', 'test_idx',
            '_group_indices_compress', 'predict_sci_coefficients_default']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run prerequisite cells first. Missing: ' + ', '.join(_missing))
if not mlp_artifacts.get('is_ensemble', False):
    raise RuntimeError('mlp_artifacts is not an ensemble; item 6 requires N>=2 members.')

_members = mlp_artifacts['members']
_seeds = list(mlp_artifacts['seeds'])
print(f'Item 6 -- predictive uncertainty from ensemble spread '
      f'(N={len(_members)} seeds: {_seeds}).')


def _predict_all_seeds(idx):
    _stack = np.stack([predict_sci_coefficients_default(
        _m,
        coef_near_phys=coef_near_all[idx], coef_far_phys=coef_far_all[idx],
        ctx_near_phys=ctx_near_all[idx], ctx_far_phys=ctx_far_all[idx],
        ctx_sci_phys=ctx_sci_all[idx],
    ).astype(np.float32) for _m in _members], axis=0)
    return _stack   # (N_seed, N_row, N_coef)


_val_idx_arr = np.asarray(val_idx, dtype=int)
_test_idx_arr = np.asarray(test_idx, dtype=int)

# Coefficients that are ~0 in the training data collapse sigma to ~0; floor per
# group at 1e-3 x MAD(coef_train) so a handful of dead coefficients don't push
# a few |z| values into 1e9 and blow up mean/std of z.
_train_idx_arr = np.asarray(train_idx, dtype=int)
_sigma_floor_by_group = {}
for _gname, _idx in _group_indices_compress.items():
    _idx = np.asarray(_idx, dtype=int)
    _train_coef = coef_sci_all[_train_idx_arr][:, _idx].astype(np.float32)
    _mad = np.median(np.abs(_train_coef - np.median(_train_coef, axis=0)), axis=0)
    _sigma_floor_by_group[_gname] = np.maximum(1e-3 * _mad, 1e-12).astype(np.float32)

_val_stack = _predict_all_seeds(_val_idx_arr)
_val_mean = _val_stack.mean(axis=0)
_val_sigma = _val_stack.std(axis=0, ddof=1)
for _gname, _idx in _group_indices_compress.items():
    _idx = np.asarray(_idx, dtype=int)
    _val_sigma[:, _idx] = np.maximum(_val_sigma[:, _idx], _sigma_floor_by_group[_gname])
_y_val = coef_sci_all[_val_idx_arr].astype(np.float32)
_z_val = (_y_val - _val_mean) / _val_sigma

_rows = []
for _gname, _idx in _group_indices_compress.items():
    _idx = np.asarray(_idx, dtype=int)
    _zg = _z_val[:, _idx].ravel()
    _sg = _val_sigma[:, _idx].ravel()
    _rg = (_y_val - _val_mean)[:, _idx].ravel()
    _true_rms = float(np.sqrt(np.mean(_rg ** 2)))
    _pred_rms = float(np.sqrt(np.mean(_sg ** 2)))
    _rows.append({
        'group': _gname,
        'n_pts': int(_zg.size),
        'median|z|': float(np.median(np.abs(_zg))),
        'MAD_z/0.6745': float(np.median(np.abs(_zg - np.median(_zg))) / 0.6745),
        'sigma_true/pred': _true_rms / max(_pred_rms, 1e-30),
        'cov_|z|<1.96': float(np.mean(np.abs(_zg) < 1.96)),
        'cov_|z|<1.00': float(np.mean(np.abs(_zg) < 1.00)),
    })
_z_all = _z_val.ravel()
_r_all = (_y_val - _val_mean).ravel()
_s_all = _val_sigma.ravel()
_rows.append({
    'group': 'all_groups',
    'n_pts': int(_z_all.size),
    'median|z|': float(np.median(np.abs(_z_all))),
    'MAD_z/0.6745':
        float(np.median(np.abs(_z_all - np.median(_z_all))) / 0.6745),
    'sigma_true/pred':
        float(np.sqrt(np.mean(_r_all ** 2))
              / max(np.sqrt(np.mean(_s_all ** 2)), 1e-30)),
    'cov_|z|<1.96': float(np.mean(np.abs(_z_all) < 1.96)),
    'cov_|z|<1.00': float(np.mean(np.abs(_z_all) < 1.00)),
})
ensemble_uncertainty_val_df = pd.DataFrame(_rows)
print(f'\nPer-group predictive-uncertainty calibration on val '
      f'(n_val = {_val_idx_arr.size} rows):')
print(ensemble_uncertainty_val_df.to_string(
    index=False, float_format=lambda v: f'{v:.4g}'))

# PIT histogram: u = Phi(z) should be U[0,1] under a well-calibrated Gaussian ensemble.
_u = 0.5 * (1.0 + _erf_vec(_z_all / np.sqrt(2.0)))
_hist_edges = np.linspace(0.0, 1.0, 21)
_hist, _ = np.histogram(_u, bins=_hist_edges)
_expected = _u.size / (len(_hist_edges) - 1)
_pit_chi2 = float(np.sum((_hist - _expected) ** 2 / max(_expected, 1e-30)))
print(f'\nPIT histogram (u = Phi(z), 20 bins on [0,1]; expected {_expected:.0f}/bin):')
_bar_max = max(int(_hist.max()), 1)
for _lo, _hi, _c in zip(_hist_edges[:-1], _hist_edges[1:], _hist):
    _bar = '#' * int(round(50.0 * _c / _bar_max))
    print(f'  [{_lo:.2f}, {_hi:.2f})  n={int(_c):>6d}  {_bar}')
print(f'PIT chi2 (20 bins): {_pit_chi2:.1f}  '
      f'(uniform target chi2_19 = 19 +/- 6; large values -> uncalibrated).')

# Headline transfer check on test (same per-group floor computed on train).
_test_stack = _predict_all_seeds(_test_idx_arr)
_test_mean = _test_stack.mean(axis=0)
_test_sigma = _test_stack.std(axis=0, ddof=1)
for _gname, _idx in _group_indices_compress.items():
    _idx = np.asarray(_idx, dtype=int)
    _test_sigma[:, _idx] = np.maximum(_test_sigma[:, _idx], _sigma_floor_by_group[_gname])
_y_test = coef_sci_all[_test_idx_arr].astype(np.float32)
_z_test = (_y_test - _test_mean) / _test_sigma
_r_test = _y_test - _test_mean
_true_rmse_test = float(np.sqrt(np.mean(_r_test ** 2)))
_pred_rmse_test = float(np.sqrt(np.mean(_test_sigma ** 2)))
_sigma_ratio = _true_rmse_test / max(_pred_rmse_test, 1e-30)

print(f'\nHeadline calibration on test (n_test = {_test_idx_arr.size} rows):')
_robust_std_val = float(np.median(np.abs(_z_val - np.median(_z_val))) / 0.6745)
_robust_std_test = float(np.median(np.abs(_z_test - np.median(_z_test))) / 0.6745)
print(f'  robust std(z) on val  (MAD/0.6745) = {_robust_std_val:.3f}')
print(f'  robust std(z) on test (MAD/0.6745) = {_robust_std_test:.3f}')
print(f'  true pooled RMSE (y - mean_pred)         = {_true_rmse_test:.4f}')
print(f'  ensemble-predicted sigma RMS           = {_pred_rmse_test:.4f}')
print(f'  sigma under-estimation factor (true/pred) = {_sigma_ratio:.2f}x')
print(f'  coverage |z|<1.96 (nominal 0.95) on test = '
      f'{float(np.mean(np.abs(_z_test) < 1.96)):.3f}')

if _sigma_ratio > 1.2:
    print('  Verdict: ensemble spread UNDER-estimates true error (factor > 1.2x). '
          'A per-group sigma_scale calibration or a deeper ensemble (bootstrap + '
          'longer training-init spread) is required before shipping sigma downstream. '
          'See item 7 (spectrum-space validation) and §11.7.')
elif _sigma_ratio < 0.8:
    print('  Verdict: ensemble spread OVER-estimates true error (factor < 0.8x). '
          'Sigma is safe as an upper bound but overly conservative.')
else:
    print('  Verdict: ensemble spread is within 20% of the true error scale. '
          'A single scalar sigma_scale per group should suffice for §11.7.')

# Save per-group sigma_scale factors (val -> test transfer) so downstream code
# can bolt them onto sigma_row with one multiplication.
_sigma_scale_by_group = {}
for _r in _rows:
    if _r['group'] == 'all_groups':
        continue
    _sigma_scale_by_group[_r['group']] = float(_r['sigma_true/pred'])
mlp_artifacts.setdefault('predictive_uncertainty', {}).update({
    'sigma_scale_by_group_val': _sigma_scale_by_group,
    'sigma_floor_by_group_train': {
        _g: _v.tolist() for _g, _v in _sigma_floor_by_group.items()},
    'val_robust_std_z': _robust_std_val,
    'test_robust_std_z': _robust_std_test,
    'test_sigma_ratio': _sigma_ratio,
    'pit_chi2_val': _pit_chi2,
})
print(f'\nStored per-group sigma_scale in '
      f"mlp_artifacts['predictive_uncertainty']['sigma_scale_by_group_val'].")
