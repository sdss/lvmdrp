# Truth-conditioned |z| = |residual|/sigma calibration.
# Cell 43 shows median|z| far below 0.6745 across all groups. That could
# be a real miscalibration OR a diagnostic artifact from the boundary-
# zero coefficients (both truth and prediction ~= 0, so |z| ~= 0 for
# those regardless of whether sigma is calibrated). This cell restricts
# |z| to coefficients that actually carry signal, using two criteria:
#   * |coef_true| > population p50 of the nonzero coefs in that group
#   * |coef_true| > k * sigma  (SNR > k)  for k = 1, 3, 10
# If median|z| approaches 0.6745 on the SNR-restricted sets, sigma is
# calibrated where it matters and cell 43's low tail is a diagnostic
# artifact of the near-zero population.
import numpy as np
import pandas as pd

required = ['coef_pred_det', 'coef_sci_all', 'test_idx',
            '_group_indices_compress', 'coef_err_sci_all']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError('Missing kernel state; run trainer + cell 43 first: '
                       + ', '.join(_missing))

_TE = np.asarray(test_idx, dtype=int)
_ytrue = np.asarray(coef_sci_all[_TE], dtype=np.float64)
_ypred = np.asarray(coef_pred_det, dtype=np.float64)
_sig   = np.asarray(coef_err_sci_all[_TE], dtype=np.float64)
_resid = _ypred - _ytrue

_GAUSS_MEDIAN = 0.6744897501960817  # scipy.stats.halfnorm.median()
_groups = list(_group_indices_compress.keys())

# Population magnitude threshold per group: median of |coef_true| across the
# WHOLE corpus (train + val + test) restricted to strictly nonzero coefs.
_pop_p50 = {}
for _g in _groups:
    _gidx = np.asarray(_group_indices_compress[_g], dtype=int)
    _vals = np.abs(np.asarray(coef_sci_all[:, _gidx], dtype=np.float64)).ravel()
    _mask = np.isfinite(_vals) & (_vals > 0.0)
    _pop_p50[_g] = (float(np.median(_vals[_mask])) if _mask.sum() > 100
                    else 1e-30)

_rows = []
for _g in _groups:
    _gidx = np.asarray(_group_indices_compress[_g], dtype=int)
    _r_flat = np.abs(_resid[:, _gidx]).ravel()
    _s_flat = _sig[:, _gidx].ravel()
    _y_flat = np.abs(_ytrue[:, _gidx]).ravel()
    _base = np.isfinite(_r_flat) & np.isfinite(_s_flat) & (_s_flat > 0.0)
    if _base.sum() == 0:
        _rows.append(dict(group=_g, n_all=0))
        continue
    _z = _r_flat[_base] / _s_flat[_base]
    _y = _y_flat[_base]
    _s = _s_flat[_base]

    _n_all = int(_z.size)
    _md_all = float(np.median(_z))
    _pop_frac_nonzero = float((_y > 0.0).mean())

    def _md_and_n(_mask):
        if not _mask.any():
            return np.nan, 0
        return float(np.median(_z[_mask])), int(_mask.sum())

    _md_mag, _n_mag = _md_and_n(_y > _pop_p50[_g])
    _md_s1,  _n_s1  = _md_and_n(_y > 1.0 * _s)
    _md_s3,  _n_s3  = _md_and_n(_y > 3.0 * _s)
    _md_s10, _n_s10 = _md_and_n(_y > 10.0 * _s)

    _rows.append(dict(
        group=_g,
        n_all=_n_all,
        frac_nonzero=_pop_frac_nonzero,
        median_z_all=_md_all,
        n_mag=_n_mag, median_z_mag=_md_mag,
        n_snr_1=_n_s1,   median_z_snr_1=_md_s1,
        n_snr_3=_n_s3,   median_z_snr_3=_md_s3,
        n_snr_10=_n_s10, median_z_snr_10=_md_s10,
    ))

_df = pd.DataFrame(_rows)

print('Truth-conditioned |z| = |residual|/sigma  (target median = 0.6745)')
print()
_cols_show = ['group', 'n_all', 'frac_nonzero', 'median_z_all',
              'median_z_mag',   'median_z_snr_1',
              'median_z_snr_3', 'median_z_snr_10']
_fmt = {}
for _c in _cols_show:
    if _c == 'group':
        continue
    if _c == 'n_all':
        _fmt[_c] = (lambda v: f'{int(v):,}')
    elif _c == 'frac_nonzero':
        _fmt[_c] = (lambda v: f'{v:.1%}')
    else:
        _fmt[_c] = (lambda v: f'{v:.4f}' if np.isfinite(v) else '   n/a')
with pd.option_context('display.width', 220, 'display.max_columns', None):
    print(_df[_cols_show].to_string(index=False, formatters=_fmt))

# Sample-size context.
print()
print('Sample sizes per restriction (matches denominators above):')
_size_cols = ['group', 'n_all', 'n_mag', 'n_snr_1', 'n_snr_3', 'n_snr_10']
with pd.option_context('display.width', 200):
    print(_df[_size_cols].to_string(index=False,
        formatters={c: (lambda v: f'{int(v):,}') for c in _size_cols
                    if c != 'group'}))

# Sigma-scale hints.
print()
print('sigma_scale hint = median|z| / 0.6745   (target = 1.00)')
print(f"  {'group':<12}{'all':>10}{'|c|>p50':>10}{'SNR>1':>10}{'SNR>3':>10}{'SNR>10':>10}")
for _r in _rows:
    if 'median_z_all' not in _r:
        continue
    def _fmt_hint(_v):
        if not np.isfinite(_v):
            return '  n/a'
        return f'{_v / _GAUSS_MEDIAN:.2f}'
    print(f"  {_r['group']:<12}"
          f"{_fmt_hint(_r['median_z_all']):>10}"
          f"{_fmt_hint(_r['median_z_mag']):>10}"
          f"{_fmt_hint(_r['median_z_snr_1']):>10}"
          f"{_fmt_hint(_r['median_z_snr_3']):>10}"
          f"{_fmt_hint(_r['median_z_snr_10']):>10}")

print()
print('Interpretation:')
print('  * frac_nonzero == 1.0 for all groups => truth is never boundary-zero.')
print('  * If SNR>3 median|z| ~ 0.67 while ALL median|z| << 0.67, the ALL')
print('    diagnostic is dominated by near-zero coefs and sigma is fine on the')
print('    signal-carrying part.  A per-group scalar sigma rescale is a no-op.')
print('  * If SNR>3 median|z| is still << 0.67, sigma is genuinely over-')
print('    estimated on active coefs and a row-dependent correction is warranted.')


# Joint (Mahalanobis) |z| on the moon and zodi blocks -- uses the full active-
# set covariance persisted in the decomp FITS. Under a well-calibrated Gaussian
# posterior, sqrt(r^T Sigma^-1 r / n_active) has median ~ sqrt(chi2_n median /
# n) which is close to sqrt(0.67) for large n; we report both metrics.
_cov_moon_sci = filtered_triplet.get('coef_cov_moon_sci')
_cov_zodi_sci = filtered_triplet.get('coef_cov_zodi_sci')
if _cov_moon_sci is not None or _cov_zodi_sci is not None:
    print()
    print('Joint Mahalanobis |z|_joint = sqrt(r^T Sigma^-1 r / n_active)  per row')
    print('(uses the persisted COEF_COV_* blocks; target median ~ 0.7)')

    def _row_mahalanobis(resid_block, cov_block):
        n_row = resid_block.shape[0]
        out = np.full(n_row, np.nan, dtype=np.float64)
        n_use = np.zeros(n_row, dtype=np.int64)
        for i in range(n_row):
            r = np.asarray(resid_block[i], dtype=np.float64)
            C = np.asarray(cov_block[i], dtype=np.float64)
            good = np.isfinite(r) & np.isfinite(np.diagonal(C)) & (np.diagonal(C) > 0)
            if good.sum() < 2:
                continue
            r_g = r[good]
            C_g = C[np.ix_(good, good)]
            # Symmetric regularization for tiny eigenvalues.
            C_g = 0.5 * (C_g + C_g.T) + 1e-24 * np.eye(C_g.shape[0])
            try:
                z2 = float(r_g @ np.linalg.solve(C_g, r_g))
            except np.linalg.LinAlgError:
                continue
            out[i] = np.sqrt(max(z2, 0.0) / good.sum())
            n_use[i] = int(good.sum())
        return out, n_use

    for _block_name, _cov in (('moon', _cov_moon_sci), ('zodi', _cov_zodi_sci)):
        if _cov is None:
            continue
        _gidx = np.asarray(_group_indices_compress[_block_name], dtype=int)
        _resid_block = _resid[:, _gidx]
        _cov_te = _cov[_TE]
        _z_joint, _n_active = _row_mahalanobis(_resid_block, _cov_te)
        _finite = np.isfinite(_z_joint)
        if _finite.any():
            _med = float(np.median(_z_joint[_finite]))
            _n_ok = int(_finite.sum())
            _n_med_active = int(np.median(_n_active[_finite]))
            print(f'  {_block_name:>6s}: median |z|_joint = {_med:.4f}   '
                  f'(n_row_ok={_n_ok}, median n_active={_n_med_active})')
        else:
            print(f'  {_block_name:>6s}: all rows had insufficient active coefs; '
                  f'no median available.')