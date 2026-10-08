# Ensemble-spread vs residual calibration (uncertainty trust check).
#
# For each test row r and each group g we form:
#   sigma_ens_g(r) = per-row per-group std across seeds of the ML prediction
#                    (across coefficients then averaged, so it is a scalar
#                    per row per group -- comparable to the sRMSE style)
#   err_g(r)       = per-row per-group RMS of |c_pred_ensemble - c_true|
# If the ensemble std were a well-calibrated one-sigma of the actual error,
# then:
#   (a) mean(err**2) ~ mean(sigma_ens**2), a per-group calibration ratio near 1
#   (b) rows with high sigma_ens should have high err (rank correlation > 0).
# Deployment-time gating on sigma_ens only makes sense when (b) holds.
#
# What to look for:
#   ratio ~ 1, tau_kendall > 0.3 -> well-calibrated; use sigma_ens to gate risky rows.
#   ratio << 1                   -> the ensemble under-estimates its own error;
#                                    ensembling reduces bias but not variance,
#                                    train with different data seeds.
#   ratio >> 1                   -> over-estimates; ensemble is over-diversified;
#                                    use fewer seeds or reduce architecture variance.
#   tau near 0 with ratio ~ 1    -> mean scale is fine but per-row uncertainty is
#                                    not informative; sigma_ens cannot gate.

from scipy.stats import kendalltau, pearsonr

required = ['filtered_triplet', 'mlp_artifacts', 'predict_sci_coefficients_default',
            '_group_indices_compress',
            'coef_near_all', 'coef_far_all', 'coef_sci_all',
            'ctx_near_all', 'ctx_far_all', 'ctx_sci_all',
            'test_idx']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Run prerequisite cells first. Missing: ' + ', '.join(_missing))
if not mlp_artifacts.get('is_ensemble', False):
    raise RuntimeError('mlp_artifacts is not an ensemble; ensemble spread cannot be measured.')

_members = mlp_artifacts['members']
_seeds   = mlp_artifacts['seeds']
_te      = np.asarray(test_idx, dtype=int)

_ctx_near_te = np.asarray(ctx_near_all[_te], dtype=np.float32)
_ctx_far_te  = np.asarray(ctx_far_all[_te],  dtype=np.float32)
_ctx_sci_te  = np.asarray(ctx_sci_all[_te],  dtype=np.float32)
_coef_near_te = np.asarray(coef_near_all[_te], dtype=np.float32)
_coef_far_te  = np.asarray(coef_far_all[_te],  dtype=np.float32)
_coef_sci_te  = np.asarray(coef_sci_all[_te],  dtype=np.float64)

_per_seed_pred = np.stack([
    predict_sci_coefficients_default(
        _m,
        coef_near_phys=_coef_near_te, coef_far_phys=_coef_far_te,
        ctx_near_phys=_ctx_near_te, ctx_far_phys=_ctx_far_te,
        ctx_sci_phys=_ctx_sci_te,
    ).astype(np.float32)
    for _m in _members
], axis=0)  # (n_seeds, n_test, n_coef)
_pred_mean = _per_seed_pred.mean(axis=0).astype(np.float64)
_pred_std  = _per_seed_pred.std(axis=0,  ddof=0).astype(np.float64)

_group_names = list(_group_indices_compress.keys())

# Scalar per-row per-group sigma_ens and err.
_sigma_per_group = {}
_err_per_group   = {}
for gname, idx in _group_indices_compress.items():
    idx = np.asarray(idx, dtype=int)
    _sigma_per_group[gname] = np.sqrt(np.mean(_pred_std[:, idx] ** 2, axis=1))
    _resid = _pred_mean[:, idx] - _coef_sci_te[:, idx]
    _err_per_group[gname] = np.sqrt(np.mean(_resid * _resid, axis=1))

# Row-subset masks on ctx_sci.
_ctx_names_ll = list(filtered_triplet['ctx_names'])
_ma_col = _ctx_names_ll.index('moon_alt') if 'moon_alt' in _ctx_names_ll else None
_eb_col = _ctx_names_ll.index('ecl_beta_deg') if 'ecl_beta_deg' in _ctx_names_ll else None
_masks = {'all': np.ones(_te.size, dtype=bool)}
if _ma_col is not None:
    _moon_alt_te = _ctx_sci_te[:, _ma_col]
    _masks['moon_up']   = _moon_alt_te > 0.0
    _masks['moon_down'] = _moon_alt_te <= 0.0
if _eb_col is not None:
    _abs_eb_te = np.abs(_ctx_sci_te[:, _eb_col])
    _masks['close_zodi'] = _abs_eb_te < 20.0


print('=' * 90)
print(f'Ensemble-spread vs residual calibration '
      f'(n_seeds = {len(_seeds)}, n_test = {_te.size})')
print('=' * 90)

for regime, mask in _masks.items():
    _n = int(mask.sum())
    if _n < 30:
        continue
    print()
    print(f'--- regime: {regime}  (n_rows = {_n}) ---')
    _rows_out = []
    for gname in _group_names:
        _s = _sigma_per_group[gname][mask]
        _e = _err_per_group[gname][mask]
        _finite = np.isfinite(_s) & np.isfinite(_e) & (_s > 1e-30) & (_e > 1e-30)
        if _finite.sum() < 20:
            continue
        _ss = _s[_finite]
        _ee = _e[_finite]
        _ratio = float(np.sqrt(np.mean(_ee ** 2) / np.mean(_ss ** 2)))  # err/sigma
        _p50_ratio = float(np.median(_ee / _ss))
        _tau, _tau_p = kendalltau(_ss, _ee)
        _rp, _rp_p = pearsonr(np.log(_ss), np.log(_ee))
        _rows_out.append({
            'group':          gname,
            'p50_sigma_ens':  float(np.median(_ss)),
            'p50_err':        float(np.median(_ee)),
            'RMS ratio':      _ratio,          # RMS(err) / RMS(sigma)
            'p50 ratio':      _p50_ratio,      # median(err/sigma)
            'kendall_tau':    float(_tau),
            'pearson_log':    float(_rp),
        })
    if _rows_out:
        print(pd.DataFrame(_rows_out).to_string(index=False,
              float_format=lambda v: f'{v:.4g}'))

# Reliability curve on the "all" regime: bin by predicted sigma decile,
# report empirical RMS(err) per decile.  Each group in its own panel.
print()
print('Reliability curve: binning by sigma_ens deciles, comparing empirical '
      'RMS(err) to the mean sigma_ens.  If the model is calibrated the '
      'points should sit on the diagonal.')

_n_g = len(_group_names)
_ncols = min(3, _n_g)
_nrows = int(np.ceil(_n_g / _ncols))
_fig = make_subplots(rows=_nrows, cols=_ncols,
    subplot_titles=[f'{g}: empirical RMS(err) vs sigma_ens (deciles)'
                    for g in _group_names])
for _i, gname in enumerate(_group_names):
    _row = _i // _ncols + 1
    _col = _i %  _ncols + 1
    _s = _sigma_per_group[gname]
    _e = _err_per_group[gname]
    _finite = np.isfinite(_s) & np.isfinite(_e) & (_s > 1e-30) & (_e > 1e-30)
    _ss = _s[_finite]
    _ee = _e[_finite]
    if _ss.size < 30:
        continue
    _bins = np.quantile(_ss, np.linspace(0, 1, 11))
    _bins[-1] += 1e-30
    _idx = np.clip(np.digitize(_ss, _bins) - 1, 0, 9)
    _x = np.array([float(np.mean(_ss[_idx == k])) for k in range(10)])
    _y = np.array([float(np.sqrt(np.mean(_ee[_idx == k] ** 2))) for k in range(10)])
    _fig.add_trace(go.Scatter(x=_x, y=_y, mode='lines+markers',
        marker=dict(size=6, color='steelblue'),
        line=dict(color='steelblue', width=1),
        showlegend=False, hoverinfo='skip'), row=_row, col=_col)
    _lim_lo = float(min(_x.min(), _y.min()))
    _lim_hi = float(max(_x.max(), _y.max()))
    _grid = np.array([_lim_lo, _lim_hi])
    _fig.add_trace(go.Scatter(x=_grid, y=_grid, mode='lines',
        line=dict(color='black', dash='dot', width=1),
        showlegend=False, hoverinfo='skip'), row=_row, col=_col)
    _fig.update_xaxes(title_text='mean sigma_ens in decile',
                      type='log', row=_row, col=_col)
    _fig.update_yaxes(title_text='empirical RMS(err) in decile',
                      type='log', row=_row, col=_col)
_fig.update_layout(height=280 * _nrows, width=340 * _ncols,
    title='Ensemble-spread reliability curve per group '
          '(diagonal = calibrated one-sigma)')
_fig.show()
