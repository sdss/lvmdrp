# Per-lunation bias drift on the test split (item 14 of the review).
# Aerosol optical depth varies on ~lunation timescales; the fitted k_eff (§4.5)
# uses one value averaged over the training window, so a systematic per-lunation
# drift in the per-group mean coefficient bias would flag that k_eff is stale
# on the deployment nights and motivate a rolling-window refit (§11.3 open).

required = ['filtered_triplet', 'mlp_artifacts', 'coef_pred_det',
            '_group_indices_compress', 'coef_sci_all', 'test_idx']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run prerequisite cells first. Missing: ' + ', '.join(_missing))

_te_ll = np.asarray(test_idx, dtype=int)
_mjd_te = np.asarray(filtered_triplet['obstime_mjd'], dtype=np.float64)[_te_ll]
_LUNATION_DAYS = 29.53058867
_lun_bin = np.floor(_mjd_te / _LUNATION_DAYS).astype(int)

_y_te_ll = coef_sci_all[_te_ll].astype(np.float64)
_pred_te_ll = np.asarray(coef_pred_det, dtype=np.float64)

# Small bins are dominated by shot noise on the bias estimate; require enough rows
# per bin that a 3% group-mean bias is resolvable above sampling variance.
_MIN_ROWS_PER_LUN = 30
_lun_rows = []
for _lid in np.unique(_lun_bin):
    _mask = _lun_bin == _lid
    if int(_mask.sum()) < _MIN_ROWS_PER_LUN:
        continue
    _row = {
        'lunation': int(_lid),
        'mjd_mid': float((float(_lid) + 0.5) * _LUNATION_DAYS),
        'n_rows': int(_mask.sum()),
    }
    _y_lun = _y_te_ll[_mask]
    _p_lun = _pred_te_ll[_mask]
    for _gname, _idx in _group_indices_compress.items():
        _idx = np.asarray(_idx, dtype=int)
        _mt = float(np.mean(_y_lun[:, _idx]))
        _mp = float(np.mean(_p_lun[:, _idx]))
        _row[f'bias_{_gname}_%'] = (100.0 * (_mp - _mt)
                                    / max(abs(_mt), 1e-30))
    _lun_rows.append(_row)

lunation_bias_df = (pd.DataFrame(_lun_rows).sort_values('lunation')
                     .reset_index(drop=True))
print(f'Per-lunation bias drift on test split ({_te_ll.size} rows total, '
      f'{len(lunation_bias_df)} lunation bins with >= {_MIN_ROWS_PER_LUN} rows):')
print(lunation_bias_df.to_string(index=False, float_format=lambda v: f'{v:.3g}'))

_bias_cols = [c for c in lunation_bias_df.columns if c.startswith('bias_')]
_group_ord = [c[len('bias_'):-len('_%')] for c in _bias_cols]

_fig_lun = go.Figure()
for _g in _group_ord:
    _fig_lun.add_trace(go.Scatter(
        x=lunation_bias_df['mjd_mid'],
        y=lunation_bias_df[f'bias_{_g}_%'],
        mode='lines+markers', name=_g))
_fig_lun.add_hline(y=0.0, line=dict(color='rgba(0,0,0,0.4)', width=0.8, dash='dash'))
_fig_lun.update_layout(
    template='plotly_white',
    title='Per-group mean coefficient bias vs lunation on the test split '
          '(item 14: k_eff staleness diagnostic)',
    xaxis_title='mid-lunation MJD',
    yaxis_title='per-group mean coefficient bias (%)',
    height=460,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
)
_fig_lun.show()

# Linear-trend summary: significant slope means k_eff is drifting through the
# test window; a large range without a slope means episodic aerosol events.
print()
print('Per-group linear trend of bias(%) vs mid-lunation MJD:')
print(f"  {'group':<14s} {'slope [%/day]':>14s} {'r':>7s} "
      f"{'range [%]':>10s} {'max |bias| [%]':>15s}")
_xt = np.asarray(lunation_bias_df['mjd_mid'], dtype=np.float64)
_absmax_over_all = 0.0
for _g in _group_ord:
    _yt = np.asarray(lunation_bias_df[f'bias_{_g}_%'], dtype=np.float64)
    if _yt.size < 3 or float(np.std(_xt)) < 1e-6:
        continue
    _slope = float(np.polyfit(_xt, _yt, 1)[0])
    _r = (float(np.corrcoef(_xt, _yt)[0, 1])
          if float(np.std(_yt)) > 1e-12 else 0.0)
    _rng = float(_yt.max() - _yt.min())
    _absmax = float(np.max(np.abs(_yt)))
    _absmax_over_all = max(_absmax_over_all, _absmax)
    print(f'  {_g:<14s} {_slope:>+14.3g} {_r:>+7.3f} '
          f'{_rng:>10.2f} {_absmax:>15.2f}')

print()
if _absmax_over_all > 3.0:
    print(f'Verdict: at least one group hits per-lunation |bias| > 3% '
          f'(max = {_absmax_over_all:.2f}%). Refit k_eff on a rolling '
          f'~lunation window (§4.5, §11.3 open) before deploying on nights '
          f'outside the training epoch.')
else:
    print(f'Verdict: per-lunation |bias| stays below 3% across all groups '
          f'(max = {_absmax_over_all:.2f}%). k_eff appears stable across the '
          f'test window; no rolling refit needed at this level.')
